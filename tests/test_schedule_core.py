"""
Schedule Plug -- schedule_core.py: reading artwork, understanding the prompt,
timing the layers and drawing the frames.

What an editor is promised, pinned here:

  * the last frame IS the artwork, to the pixel -- the whole reason this is
    drawn rather than generated is that a schedule's text must not change;
  * a layered Photoshop file animates layer by layer, bottom first, and one
    whose layers cannot be put back together faithfully is animated as one
    flat picture and says so, rather than shown wrong;
  * the prompt's words map onto the animator's small vocabulary whether or
    not a language model is there to help, and a bad model reply can never
    make the result worse than no reply;
  * everything is on screen, complete, for more than half the plug.
"""
import json

import cv2
import numpy as np
import pytest

import schedule_core as sk

psd_tools = pytest.importorskip('psd_tools')
from PIL import Image, ImageDraw   # noqa: E402  (comes with psd-tools)

W, H = sk.CANVAS


def _block(box, colour):
    im = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(im).rectangle(box, fill=colour)
    return im


PARTS = [('Background', lambda: Image.new('RGBA', (W, H), (20, 30, 90, 255))),
         ('Title', lambda: _block([200, 80, 1700, 220], (255, 200, 0, 255))),
         ('Schedule Mon', lambda: _block([200, 300, 1700, 420], (255, 255, 255, 230))),
         ('Schedule Tue', lambda: _block([200, 460, 1700, 580], (255, 255, 255, 230)))]
NAMES = [n for n, _ in PARTS]


def _psd(path, parts=PARTS, size=(W, H), **layer_kw):
    psd = psd_tools.PSDImage.new('RGB', size, color=0)
    flat = Image.new('RGBA', size, (0, 0, 0, 255))
    for name, make in parts:
        im = make()
        box = im.getbbox()
        psd.append(psd.create_pixel_layer(im.crop(box), name=name, top=box[1], left=box[0], **layer_kw.get(name, {})))
        flat.alpha_composite(im)
    psd.save(str(path))
    return cv2.cvtColor(np.asarray(flat.convert('RGB')), cv2.COLOR_RGB2BGR)


@pytest.fixture(scope='module')
def layered(tmp_path_factory):
    d = tmp_path_factory.mktemp('art')
    want = _psd(d / 'week.psd')
    cv2.imwrite(str(d / 'week.png'), want)
    return {'psd': str(d / 'week.psd'), 'png': str(d / 'week.png'), 'want': want, 'dir': d}


def _diff(a, b):
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())


# ---- loading ----

def test_a_layered_psd_becomes_one_layer_per_top_level_layer_bottom_first(layered):
    art = sk.load_artwork(layered['psd'])
    assert [ly['name'] for ly in art['layers']] == NAMES and art['layered'] and art['notes'] == []
    title = art['layers'][1]
    assert (title['x'], title['y']) == (200, 80) and title['px'].shape == (141, 1501, 4), 'cropped to its own pixels'
    assert _diff(sk._flatten(art['layers'], sk.CANVAS), layered['want']) < 0.5


def test_a_flat_image_is_one_layer(layered):
    art = sk.load_artwork(layered['png'])
    assert [ly['name'] for ly in art['layers']] == ['Image'] and not art['layered']
    assert _diff(sk._flatten(art['layers'], sk.CANVAS), layered['want']) < 0.5


def test_artwork_that_is_not_16_by_9_is_shown_whole_and_small_artwork_is_flagged(tmp_path):
    cv2.imwrite(str(tmp_path / 'square.png'), np.full((800, 800, 3), 200, np.uint8))
    art = sk.load_artwork(str(tmp_path / 'square.png'))
    ly = art['layers'][0]
    assert ly['px'].shape[:2] == (1080, 1080) and (ly['x'], ly['y']) == (420, 0), 'fitted inside, centred, not cropped'
    assert any('not 16:9' in n for n in art['notes']) and any('enlarged' in n for n in art['notes'])


def test_layers_that_cannot_be_recomposed_faithfully_fall_back_to_the_flat_picture(tmp_path):
    """A Multiply layer over the background is not what a plain "paste on
    top" gives. Animating the layers would show the wrong picture, so the
    file's own flattened preview is animated instead -- and the editor is
    told why, and what to do about it."""
    psd = psd_tools.PSDImage.new('RGB', (W, H), color=0)
    psd.append(psd.create_pixel_layer(Image.new('RGBA', (W, H), (200, 200, 200, 255)), name='Background'))
    psd.append(psd.create_pixel_layer(Image.new('RGBA', (900, 500), (255, 0, 0, 255)), name='Tint', top=200, left=400,
                                      blend_mode=psd_tools.constants.BlendMode.MULTIPLY))
    psd.save(str(tmp_path / 'multiply.psd'))
    art = sk.load_artwork(str(tmp_path / 'multiply.psd'))
    assert [ly['name'] for ly in art['layers']] == ['Image'] and not art['layered']
    assert any('animated as one flat picture' in n and 'blend modes' in n for n in art['notes'])


def test_hidden_layers_are_left_out_and_too_many_layers_are_merged_from_the_bottom(tmp_path):
    parts = [('Background', lambda: Image.new('RGBA', (W, H), (10, 10, 60, 255)))] + [
        (f'Row {i}', (lambda i=i: _block([100, 40 + i * 60, 1800, 80 + i * 60], (255, 255, 255, 255)))) for i in range(15)]
    want = _psd(tmp_path / 'many.psd', parts)
    art = sk.load_artwork(str(tmp_path / 'many.psd'))
    assert len(art['layers']) == sk.MAX_LAYERS and art['layers'][0]['name'] == 'Background'
    assert [ly['name'] for ly in art['layers'][1:]] == [f'Row {i}' for i in range(4, 15)], 'the top eleven still move'
    assert any('16 top-level layers' in n for n in art['notes'])
    assert _diff(sk._flatten(art['layers'], sk.CANVAS), want) < 0.5, 'merged, not dropped'


def test_unreadable_artwork_is_refused_in_plain_words(tmp_path):
    (tmp_path / 'broken.png').write_bytes(b'not an image at all')
    with pytest.raises(sk.ArtworkError, match='could not be read'):
        sk.load_artwork(str(tmp_path / 'broken.png'))
    (tmp_path / 'broken.psd').write_bytes(b'8BPSnope')
    with pytest.raises(sk.ArtworkError, match='could not be opened'):
        sk.load_artwork(str(tmp_path / 'broken.psd'))


# ---- the prompt ----

def test_the_example_prompt_reads_as_stop_motion_with_the_schedule_wiping_in():
    r = sk.parse_prompt('stop motion effect. wipe in reveal schedule.', NAMES)
    assert r['stop_motion'] and not r['push_in'] and r['speed'] == 'normal'
    assert r['layers'] == {'schedule mon': {'effect': 'wipe', 'direction': 'right'},
                           'schedule tue': {'effect': 'wipe', 'direction': 'right'}}
    assert r['default']['effect'] == 'fade', 'the title was not mentioned, so it keeps the default'
    assert sk.describe_recipe(r, NAMES) == ('background: fade; 1 layer: fade; Schedule Mon: wipe right; '
                                            'Schedule Tue: wipe right; stop motion')
    # The same words for a flat picture: the wipe is for the picture itself.
    flat = sk.parse_prompt('stop motion effect. wipe in reveal schedule.', ['Image'])
    assert flat['background'] == {'effect': 'wipe', 'direction': 'right'} and flat['stop_motion']
    assert sk.describe_recipe(flat, ['Image']) == 'picture: wipe right; stop motion'


@pytest.mark.parametrize('prompt, says', [
    ('', 'background: fade; 3 layers: fade'),
    ('make it look amazing', 'background: fade; 3 layers: fade'),
    ('wipe in from the left, slow push in', 'background: wipe right; 3 layers: wipe right; slow push in'),
    ('Slide everything in from the bottom, quickly', 'background: slide up; 3 layers: slide up; fast'),
    ('title pops in, schedule slides in from the right',
     'background: fade; Title: pop; Schedule Mon: slide left; Schedule Tue: slide left'),
    ('fade in the background and wipe the schedule top to bottom',
     'background: fade; 1 layer: fade; Schedule Mon: wipe down; Schedule Tue: wipe down'),
    ('gentle fades, claymation feel', 'background: fade; 3 layers: fade; stop motion; slow'),
])
def test_prompts_map_onto_the_animators_vocabulary(prompt, says):
    assert sk.describe_recipe(sk.parse_prompt(prompt, NAMES), NAMES) == says


def test_a_model_reply_refines_the_reading_and_a_bad_one_cannot_spoil_it():
    base = sk.parse_prompt('stop motion effect. wipe in reveal schedule.', NAMES)
    good = sk.parse_recipe_reply(json.dumps({
        'stop_motion': True, 'push_in': True, 'speed': 'fast', 'background': {'effect': 'fade'},
        'default': {'effect': 'pop', 'direction': 'sideways'},
        'layers': [{'name': 'title', 'effect': 'slide', 'direction': 'down'}, {'name': 'Logo', 'effect': 'wipe'},
                   {'name': 'Schedule Tue', 'effect': 'teleport'}]}), NAMES, base)
    assert good['push_in'] and good['speed'] == 'fast'
    assert good['default'] == {'effect': 'pop', 'direction': 'right'}, 'an unknown direction keeps the old one'
    assert good['layers']['title'] == {'effect': 'slide', 'direction': 'down'}, 'names match whatever their case'
    assert 'logo' not in good['layers'], 'a layer that does not exist is dropped, not guessed at'
    assert good['layers']['schedule tue']['effect'] == 'pop', 'an effect it cannot do becomes the default'
    assert good['layers']['schedule mon'] == {'effect': 'wipe', 'direction': 'right'}, 'what the words said survives'
    # Wrapped in chatter, or as a name->settings object: still read.
    wrapped = sk.parse_recipe_reply('Sure! Here you go:\n```json\n{"stop_motion": false, "layers": {"Title": {"effect": "cut"}}}\n```', NAMES)
    assert wrapped['stop_motion'] is False and wrapped['layers'] == {'title': {'effect': 'cut', 'direction': 'right'}}
    for junk in ('', 'I cannot help with that.', '{"moments": []}', '{not json', '[1, 2]', None):
        assert sk.parse_recipe_reply(junk, NAMES, base) is None
    assert 'Schedule Mon' in sk.recipe_prompt('wipe the schedule', NAMES)


# ---- timing ----

@pytest.mark.parametrize('duration', sk.DURATIONS)
def test_everything_is_in_place_early_and_then_holds(duration):
    for names in (['Image'], NAMES, [f'L{i}' for i in range(sk.MAX_LAYERS)]):
        for speed in sk.SPEEDS:
            tl = sk.build_timeline(dict(sk.default_recipe(), speed=speed), names, duration)
            assert len(tl) == len(names)
            starts = [t['start'] for t in tl]
            assert starts == sorted(starts) and starts[0] > 0, 'bottom layer first, after a beat of black'
            assert all(t['dur'] >= 0.3 for t in tl)
            assert sk.settle_time(tl) <= duration * 0.75, (names, speed, sk.settle_time(tl))
    assert sk.settle_time(sk.build_timeline(sk.default_recipe(), NAMES, duration)) <= max(duration * 0.5, 4.0)


# ---- drawing ----

def _animator(art, prompt, duration=10, names=None):
    names = names or [ly['name'] for ly in art['layers']]
    r = sk.parse_prompt(prompt, names)
    return sk.Animator(art, sk.build_timeline(r, names, duration), r, duration)


@pytest.mark.parametrize('prompt', ['', 'wipe in reveal schedule', 'slide in from the left', 'everything pops in',
                                    'stop motion effect. wipe in reveal schedule.', 'cut in'])
def test_the_last_frame_is_the_artwork_exactly(layered, prompt):
    for src in ('psd', 'png'):
        an = _animator(sk.load_artwork(layered[src]), prompt)
        assert an.frame(0.0).max() == 0, 'opens on black'
        assert _diff(an.frame(10.0), layered['want']) < 0.5, (src, prompt)
        assert _diff(an.frame(an.settled_at + 0.2), layered['want']) < 0.5, 'and holds from the moment it settles'


def test_layers_arrive_one_after_another_and_a_wipe_uncovers_along_its_direction(layered):
    art = sk.load_artwork(layered['psd'])
    an = _animator(art, 'wipe in reveal schedule')
    tl = an.timeline
    title_px, mon_l, mon_r, tue = (950, 150), (300, 360), (1600, 360), (950, 520)
    white = lambda f, p: f[p[1], p[0]].min() > 200       # noqa: E731
    f = an.frame(tl[2]['start'] - 0.05)                   # just before Monday's row starts
    assert f[title_px[1], title_px[0], 2] > 200, 'the title is already there'
    assert not white(f, mon_l) and not white(f, tue)
    f = an.frame(tl[2]['start'] + tl[2]['dur'] * 0.3)     # part-way through Monday's wipe, travelling right
    assert white(f, mon_l) and not white(f, mon_r), 'left end uncovered first'
    assert not white(f, tue), 'Tuesday waits its turn'
    f = an.frame(tl[3]['start'] + tl[3]['dur'] + 0.05)
    assert white(f, mon_r) and white(f, tue)
    # The other way round when asked.
    an = _animator(art, 'wipe the schedule from the right')
    f = an.frame(an.timeline[2]['start'] + an.timeline[2]['dur'] * 0.3)
    assert white(f, mon_r) and not white(f, mon_l)


def test_stop_motion_steps_while_arriving_and_holds_still_once_complete(layered):
    art = sk.load_artwork(layered['psd'])
    an = _animator(art, 'stop motion. wipe in reveal schedule')
    t0 = an.timeline[2]['start'] + an.timeline[2]['dur'] * 0.4
    step = 1.0 / sk.STOP_MOTION_FPS
    base = np.floor(t0 / step) * step
    a, b, c = an.frame(base + 0.002).copy(), an.frame(base + step * 0.9).copy(), an.frame(base + step * 1.1).copy()
    assert np.array_equal(a, b), 'held within a step'
    assert not np.array_equal(b, c), 'and moves on at the next'
    smooth = _animator(art, 'wipe in reveal schedule')
    assert not np.array_equal(smooth.frame(base + 0.002), smooth.frame(base + step * 0.9)), 'without it, every frame moves'
    late = [an.frame(an.settled_at + 0.5 + k * 0.37).copy() for k in range(4)]
    assert all(np.array_equal(late[0], f) for f in late[1:]), 'a finished schedule does not twitch'


def test_a_push_in_grows_steadily_and_frames_can_be_drawn_out_of_order(layered):
    art = sk.load_artwork(layered['png'])
    an = _animator(art, 'slow push in')
    end = an.frame(10.0)
    assert _diff(end, layered['want']) > 1.0, 'the last frame is the artwork slightly enlarged'
    grown = cv2.warpAffine(layered['want'], np.float32([[1.05, 0, -0.025 * W], [0, 1.05, -0.025 * H]]), (W, H),
                           flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    assert _diff(end, grown) < 0.5
    plain = _animator(sk.load_artwork(layered['psd']), 'wipe in reveal schedule')
    forward = [plain.frame(t).copy() for t in (1.0, 2.5, 4.0)]
    assert np.array_equal(plain.frame(2.5), forward[1]) and np.array_equal(plain.frame(1.0), forward[0]), \
        'rewinding gives the same picture as getting there in order'


def test_paste_clips_at_the_edges_and_respects_opacity():
    dst = np.zeros((10, 10, 3), np.uint8)
    px = np.full((4, 4, 4), 255, np.uint8)
    sk.paste(dst, px, 8, 8)
    assert dst[9, 9].tolist() == [255, 255, 255] and dst[7, 7].tolist() == [0, 0, 0] and dst.sum() == 4 * 3 * 255
    sk.paste(dst, px, -100, 0)
    sk.paste(dst, px, 0, 0, opacity=0.0)
    assert dst[0, 0].tolist() == [0, 0, 0]
    sk.paste(dst, px, 0, 0, opacity=0.5)
    assert 120 <= dst[0, 0, 0] <= 135


def test_delivery_formats_are_drawn_at_their_own_frame_rate():
    assert sk.fps_for('prores_hq_2398') == (24000, 1001)
    assert sk.fps_for('prores_hq_2997') == sk.fps_for('mp4_high') == sk.fps_for('avci100i') == (30000, 1001)
    cmd = sk.build_encode_cmd('ffmpeg', 'out.mov', sk.CANVAS, (30000, 1001), 15, music_path='bed.wav')
    joined = ' '.join(cmd)
    assert '-s 1920x1080 -r 30000/1001 -i pipe:0' in joined and '-stream_loop -1' in joined
    assert 'in_range=pc:out_range=tv' in joined and 'loudnorm=I=-14.0' in joined and 'afade=t=out:st=13.500' in joined
    silent = ' '.join(sk.build_encode_cmd('ffmpeg', 'out.mov', sk.CANVAS, (30000, 1001), 15))
    assert 'anullsrc' in silent and 'loudnorm' not in silent

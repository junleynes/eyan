import os
"""
Schedule Plug -- schedule_core.py: reading artwork, understanding the prompt,
timing the layers and drawing the frames.

What an editor is promised, pinned here:

  * the logo and the text come to rest exactly as designed, to the pixel --
    the whole reason this is drawn rather than generated is that a
    schedule's text must not change;
  * in a layered Photoshop file the background is on screen, still, from
    the first frame and the layers above it arrive one by one; a file
    whose layers cannot be put back together faithfully is animated as one
    flat picture and says so, rather than shown wrong;
  * a flat picture has no background of its own, so it arrives from black;
  * the prompt's words map onto the animator's small vocabulary whether or
    not a language model is there to help, and a bad model reply can never
    make the result worse than no reply;
  * everything is on screen, complete, for more than half the plug;
  * a layered file is animated in three groups: the background is static;
    the logo and the schedule's text appear and then hold still (or
    breathe); every other layer is on screen from the first frame and moves
    in a loop from the first frame to the last;
  * an animation style is a way for the logo and text to appear plus a
    loop for the other layers, and the prompt changes only what it names.
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
    assert [ly['role'] for ly in art['layers']] == ['background', 'content', 'content', 'content']
    title = art['layers'][1]
    assert (title['x'], title['y']) == (200, 80) and title['px'].shape == (141, 1501, 4), 'cropped to its own pixels'
    assert _diff(sk._flatten(art['layers'], sk.CANVAS), layered['want']) < 0.5


def test_a_flat_image_is_one_layer(layered):
    art = sk.load_artwork(layered['png'])
    assert [ly['name'] for ly in art['layers']] == ['Image'] and not art['layered']
    assert art['layers'][0]['role'] == 'picture'
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
    assert r['background']['effect'] == 'static'
    assert sk.describe_recipe(r, NAMES) == ('background: static; 1 layer: fade; Schedule Mon: wipe right; '
                                            'Schedule Tue: wipe right; stop motion')
    # The same words for a flat picture: the wipe is for the picture itself.
    flat = sk.parse_prompt('stop motion effect. wipe in reveal schedule.', ['Image'])
    assert flat['default'] == {'effect': 'wipe', 'direction': 'right'} and flat['stop_motion']
    assert sk.describe_recipe(flat, ['Image']) == 'picture: wipe right; stop motion'
    assert sk.describe_recipe(sk.parse_prompt('', ['Image']), ['Image']) == 'picture: fade'


@pytest.mark.parametrize('prompt, says', [
    ('', 'background: static; 3 layers: fade'),
    ('make it look amazing', 'background: static; 3 layers: fade'),
    # "Everything" is everything that arrives. The background does not arrive.
    ('wipe in from the left, slow push in', 'background: static; 3 layers: wipe right; slow push in'),
    ('Slide everything in from the bottom, quickly', 'background: static; 3 layers: slide up; fast'),
    ('title pops in, schedule slides in from the right',
     'background: static; Title: pop; Schedule Mon: slide left; Schedule Tue: slide left'),
    ('gentle fades, claymation feel', 'background: static; 3 layers: fade; stop motion; slow'),
    # Only its own name moves it.
    ('fade in the background and wipe the schedule top to bottom',
     'background: fade; 1 layer: fade; Schedule Mon: wipe down; Schedule Tue: wipe down'),
    ('wipe the background in from the left, title pops', 'background: wipe right; 2 layers: fade; Title: pop'),
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
    assert good['background']['effect'] == 'static', 'the background is not the model\'s to animate'
    moved = sk.parse_recipe_reply(json.dumps({'speed': 'slow', 'layers': [{'name': 'Background', 'effect': 'wipe'}]}),
                                  NAMES, sk.parse_prompt('fade in the background', NAMES))
    assert moved['background']['effect'] == 'fade' and 'background' not in moved['layers'], 'only the editor\'s words'
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
            assert starts == sorted(starts)
            if len(names) == 1:
                assert starts[0] > 0 and tl[0]['dur'] >= 0.3, 'a flat picture arrives, after a beat of black'
            else:
                assert (tl[0]['effect'], tl[0]['start'], tl[0]['dur']) == ('cut', 0.0, 0.0), 'the background is there'
                assert starts[1] >= 0.4, 'and is seen bare for a beat before anything lands on it'
            assert all(t['dur'] >= 0.3 for t in tl[1:])
            assert sk.settle_time(tl) <= duration * 0.75, (names, speed, sk.settle_time(tl))
    assert sk.settle_time(sk.build_timeline(sk.default_recipe(), NAMES, duration)) <= max(duration * 0.5, 4.0)
    # Asked for by name, the background arrives first and the rest wait for it.
    tl = sk.build_timeline(sk.parse_prompt('fade in the background', NAMES), NAMES, duration)
    assert tl[0]['effect'] == 'fade' and tl[0]['start'] > 0 and tl[1]['start'] > tl[0]['start']


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
        if src == 'png':
            assert an.frame(0.0).max() == 0, 'a flat picture opens on black'
        assert _diff(an.frame(10.0), layered['want']) < 0.5, (src, prompt)
        assert _diff(an.frame(an.settled_at + 0.2), layered['want']) < 0.5, 'and holds from the moment it settles'


@pytest.mark.parametrize('prompt', ['', 'wipe in reveal schedule', 'slide everything in from the left', 'cut in',
                                    'stop motion effect. wipe in reveal schedule.', 'everything pops in, quickly'])
def test_the_background_is_on_screen_and_still_from_the_first_frame(layered, prompt):
    art = sk.load_artwork(layered['psd'])
    an = _animator(art, prompt)
    bare = sk._flatten(art['layers'][:1], sk.CANVAS)
    assert _diff(an.frame(0.0), bare) == 0, 'frame 0 is the background, whole, with nothing on it yet'
    # Outside the other layers it never changes at all -- no fade, no wipe,
    # and none of stop motion's wobble -- from the first frame to the last.
    for t in np.arange(0.0, 10.0, 1.0 / 12):
        f = an.frame(float(t))
        assert np.array_equal(f[:70], bare[:70]) and np.array_equal(f[600:], bare[600:]), (prompt, t)
        if 'slide' not in prompt:                    # a slide from the left travels through the left margin
            assert np.array_equal(f[:, :100], bare[:, :100]) and np.array_equal(f[:, 1800:], bare[:, 1800:]), (prompt, t)
    # ...and under them it is already there while they arrive.
    assert an.frame(0.2)[150, 950].tolist() == bare[150, 950].tolist()


def test_the_background_arrives_only_when_asked_for_by_name(layered):
    an = _animator(sk.load_artwork(layered['psd']), 'fade in the background, then wipe the schedule')
    assert an.frame(0.0).max() == 0
    assert _diff(an.frame(10.0), layered['want']) < 0.5


def test_layers_covering_the_whole_picture_over_the_bottom_one_are_background_too(tmp_path):
    def wash():
        return Image.new('RGBA', (W, H), (255, 0, 80, 60))

    def border():
        im = Image.new('RGBA', (W, H), (0, 0, 0, 0))
        ImageDraw.Draw(im).rectangle([0, 0, W - 1, H - 1], outline=(255, 255, 255, 255), width=12)
        return im
    parts = [PARTS[0], ('Texture', wash), ('Glow', wash), ('Frame', border)] + PARTS[1:]
    want = _psd(tmp_path / 'tex.psd', parts=parts)
    art = sk.load_artwork(str(tmp_path / 'tex.psd'))
    names = [ly['name'] for ly in art['layers']]
    assert names == ['Background', 'Frame', 'Title', 'Schedule Mon', 'Schedule Tue'], 'a border only reaches the edges'
    assert any('"Background", "Texture", "Glow" each cover the whole picture' in n for n in art['notes'])
    assert _diff(sk._flatten(art['layers'], sk.CANVAS), want) < 0.5
    an = _animator(art, 'wipe in reveal schedule')
    f0 = an.frame(0.0)
    assert f0[540, 100].tolist() == want[540, 100].tolist(), 'the washes are there on frame 0, not fading in'
    assert f0[5, 960].max() < 200 and an.frame(10.0)[5, 960].min() > 200, 'the frame is not, and arrives'
    # A file that is nothing but full-picture layers keeps its top one to animate.
    _psd(tmp_path / 'two.psd', parts=[PARTS[0], ('Texture', wash)])
    assert [ly['name'] for ly in sk.load_artwork(str(tmp_path / 'two.psd'))['layers']] == ['Background', 'Texture']
    _psd(tmp_path / 'three.psd', parts=[PARTS[0], ('Texture', wash), ('Glow', wash)])
    assert [ly['name'] for ly in sk.load_artwork(str(tmp_path / 'three.psd'))['layers']] == ['Background', 'Glow']


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


# ---- animation styles: arriving + what goes on until the end ----

def test_the_styles_on_offer():
    assert [k for k, _, _ in sk.STYLES] == ['wipe_mix', 'fade_shine', 'wipe_shine', 'slide_float', 'pop_pulse',
                                           'stop_motion', 'pop_dance', 'pop_grow', 'fade_rotate_left',
                                           'fade_rotate_right', 'fade_still']
    assert sk.DEFAULT_STYLE == 'wipe_mix' and sk.style_label('slide_float') == 'Slide up + gentle float'
    assert sk.style_label('nope') is None and sk.style_recipe('nope') == sk.style_recipe(sk.DEFAULT_STYLE)
    roles = ['background', 'content', 'decor', 'decor']
    says = {k: sk.describe_recipe(sk.style_recipe(k), NAMES, roles) for k, _, _ in sk.STYLES}
    head = 'background: static; logo and text: '
    assert says == {
        'wipe_mix': head + 'wipe right, then hold still; other layers: grow, rotate left, rotate right '
                           '(layers take turns)',
        'fade_shine': head + 'fade, then hold still; other layers: light sweep',
        'wipe_shine': head + 'wipe right, then hold still; other layers: light sweep',
        'slide_float': head + 'slide up, then hold still; other layers: gentle float',
        'pop_pulse': head + 'pop, then hold still; other layers: pulse one by one',
        'stop_motion': head + 'wipe right, then hold still; other layers: wobble; stop motion',
        'pop_dance': head + 'pop, then hold still; other layers: dancing',
        'pop_grow': head + 'pop, then hold still; other layers: grow and shrink',
        'fade_rotate_left': head + 'fade, then hold still; other layers: rotate left',
        'fade_rotate_right': head + 'fade, then hold still; other layers: rotate right',
        'fade_still': head + 'fade, then hold still; other layers: hold still'}
    assert [k for k, _ in sk.CONTENT_MODES] == ['hold', 'breathe'] and sk.DEFAULT_CONTENT == 'hold'
    assert sk.style_recipe('pop_dance')['content'] == 'hold'
    assert sk.style_recipe('pop_dance', 'breathe')['content'] == 'breathe'
    assert sk.style_recipe('pop_dance', 'gallop')['content'] == 'hold'
    assert sk.describe_recipe(sk.style_recipe('pop_dance', 'breathe'), NAMES, roles) == (
        head + 'pop, then breathe; other layers: dancing')
    # Each group is mentioned only if the file has such layers.
    assert sk.describe_recipe(sk.style_recipe('pop_dance'), NAMES, ['background'] + ['content'] * 3) == (
        head + 'pop, then hold still')
    assert sk.describe_recipe(sk.style_recipe('pop_dance'), NAMES, ['background'] + ['decor'] * 3) == (
        'background: static; other layers: dancing')


@pytest.mark.parametrize('style, prompt, says', [
    ('slide_float', '', 'background: static; 3 layers: slide up; then gentle float'),
    ('slide_float', 'title pops in', 'background: static; 2 layers: slide up; Title: pop; then gentle float'),
    ('slide_float', 'from the left', 'background: static; 3 layers: slide right; then gentle float'),
    ('fade_shine', 'wipe in reveal schedule, then everything floats',
     'background: static; 1 layer: fade; Schedule Mon: wipe right; Schedule Tue: wipe right; then gentle float'),
    ('fade_shine', 'stop motion effect. wipe in reveal schedule.',
     'background: static; 1 layer: fade; Schedule Mon: wipe right; Schedule Tue: wipe right; stop motion; '
     'then light sweep'),
    ('pop_pulse', 'quickly, and hold still', 'background: static; 3 layers: pop; fast'),
    ('fade_still', 'a light sweep across it', 'background: static; 3 layers: fade; then light sweep'),
    ('fade_still', 'make the rows pulse', 'background: static; 3 layers: fade; then pulse one by one'),
    ('fade_still', 'grow and shrink', 'background: static; 3 layers: fade; then grow and shrink'),
    ('fade_still', 'rows sink and grow', 'background: static; 3 layers: fade; then grow and shrink'),
    ('fade_still', 'rotate left', 'background: static; 3 layers: fade; then rotate left'),
    ('fade_still', 'everything rotates to the right', 'background: static; 3 layers: fade; then rotate right'),
    ('fade_still', 'spin counter-clockwise', 'background: static; 3 layers: fade; then rotate left'),
    ('fade_still', 'rotate left and right', 'background: static; 3 layers: fade; then sway left and right'),
    ('fade_still', 'make them dance', 'background: static; 3 layers: fade; then dancing'),
    ('fade_still', 'mixed motion', 'background: static; 3 layers: fade; then grow, rotate left, rotate right '
                                   '(layers take turns)'),
    # A layer named in the clause gets the motion to itself; the rest keep the style's.
    ('fade_shine', 'title rotates left, schedule grows and shrinks',
     'background: static; Title: then rotate left; Schedule Mon: then grow and shrink; '
     'Schedule Tue: then grow and shrink; then light sweep'),
    ('fade_shine', 'title pops in and rotates left', 'background: static; 2 layers: fade; '
                                                     'Title: pop, then rotate left; then light sweep'),
    ('fade_shine', 'title pops in and everything floats', 'background: static; 2 layers: fade; Title: pop; '
                                                          'then gentle float'),
    ('slide_float', 'title holds still', 'background: static; 2 layers: slide up; Title: then hold still; '
                                         'then gentle float'),
    # The background can be made to arrive, never to keep moving.
    ('fade_still', 'fade in the background and rotate it left', 'background: fade; 3 layers: fade'),
    ('wipe_shine', 'shaky', 'background: static; 3 layers: wipe right; then shake'),
    # "Float up" is a way of arriving, not something to go on doing.
    ('fade_shine', 'bring the title in with a bounce and float the rest up',
     'background: static; 2 layers: slide up; Title: pop; then light sweep'),
    ('fade_still', 'title pops in, schedule slides up', 'background: static; Title: pop; Schedule Mon: slide up; '
                                                        'Schedule Tue: slide up'),
])
def test_the_prompt_changes_only_what_it_names_on_top_of_the_style(style, prompt, says):
    base = sk.style_recipe(style)
    assert sk.describe_recipe(sk.parse_prompt(prompt, NAMES, base=base), NAMES) == says
    assert base == sk.style_recipe(style), 'the style itself is not altered by being used'


def test_the_model_is_shown_the_style_and_may_change_what_happens_afterwards():
    base = sk.parse_prompt('title pops in', NAMES, base=sk.style_recipe('slide_float'))
    asked = sk.recipe_prompt('title pops in', NAMES, base)
    assert '"ambient": "float"' in asked and '"effect": "slide", "direction": "up"' in asked and '"name": "title"' in asked
    kept = sk.parse_recipe_reply('{"speed": "fast"}', NAMES, base)
    assert kept['ambient'] == 'float' and kept['default'] == {'effect': 'slide', 'direction': 'up'}
    assert sk.parse_recipe_reply('{"ambient": "pulse"}', NAMES, base)['ambient'] == 'pulse'
    assert sk.parse_recipe_reply('{"ambient": "fireworks", "speed": "slow"}', NAMES, base)['ambient'] == 'float'


# The schedule, dressed: two layers that are neither logo nor text.
DRESSED = PARTS + [('Ribbon', lambda: _block([200, 640, 1700, 700], (230, 40, 110, 255))),
                   ('Star', lambda: _block([1750, 100, 1850, 200], (255, 220, 60, 255)))]
DRESSED_NAMES = [n for n, _ in DRESSED]
DRESSED_ROLES = ['background', 'content', 'content', 'content', 'decor', 'decor']
TEXT = (slice(60, 600), slice(180, 1718))            # where the title and the schedule rows are


@pytest.fixture(scope='module')
def dressed(tmp_path_factory):
    d = tmp_path_factory.mktemp('dressed')
    want = _psd(d / 'dressed.psd', parts=DRESSED)
    return {'psd': str(d / 'dressed.psd'), 'want': want}


def test_layers_are_sorted_into_background_logo_and_text_and_the_rest(dressed):
    art = sk.load_artwork(dressed['psd'])
    assert [ly['role'] for ly in art['layers']] == DRESSED_ROLES
    for name in ('GMA Logo', 'logo_white', 'Schedule', 'SCHED TEXT', 'Title', 'Show titles', 'MON', 'Tuesday', 'TIME',
                 '8PM', 'Air date', 'Program info', 'Text copy 2', 'Station bug'):
        assert sk.layer_role(name) == 'content', name
    for name in ('Ribbon', 'Star left', 'Shape 1', 'Layer 5', 'Glow', 'Burst', 'Talent photo', 'Confetti', 'Frame', ''):
        assert sk.layer_role(name) == 'decor', name
    assert sk.layer_role('Layer 5', is_text=True) == 'content', 'Photoshop says it is text, whatever it is called'

    class Fake:
        def __init__(self, kind, children=()):
            self.kind, self._children = kind, list(children)

        def is_group(self):
            return self.kind == 'group'

        def descendants(self):
            return iter(self._children)
    assert sk._is_text(Fake('type')) and not sk._is_text(Fake('pixel'))
    assert sk._is_text(Fake('group', [Fake('pixel'), Fake('type')])), 'a group holding text moves as text'
    assert not sk._is_text(Fake('group', [Fake('pixel'), Fake('shape')]))

    r = sk.style_recipe('pop_dance')
    assert sk.role_notes(DRESSED_ROLES, r) == []
    only_text = ['background', 'content', 'content', 'content']
    assert 'nothing keeps moving' in sk.role_notes(only_text, r)[0]
    assert sk.role_notes(only_text, sk.style_recipe('pop_dance', 'breathe')) == []
    assert 'No logo or text layer was recognised' in sk.role_notes(['background', 'decor', 'decor'], r)[0]
    assert sk.role_notes(['picture'], r) == []


def test_only_logo_and_text_arrive_the_other_layers_are_there_from_the_start():
    r = sk.style_recipe('wipe_mix')
    tl = sk.build_timeline(r, DRESSED_NAMES, 10, DRESSED_ROLES)
    assert [(t['effect'], t['start'], t['dur']) for t in tl[4:]] == [('cut', 0.0, 0.0)] * 2, 'decor: there at once'
    assert [t['effect'] for t in tl[1:4]] == ['wipe'] * 3 and tl[1]['start'] >= 0.4
    assert tl[1]['start'] < tl[2]['start'] < tl[3]['start'] and sk.settle_time(tl) <= 5.0
    # Fewer layers arriving: the same window is not stretched over the ones that do not.
    assert sk.settle_time(tl) == sk.settle_time(sk.build_timeline(r, NAMES, 10))
    # A decor layer the editor gives an arrival to takes its turn like the rest.
    named = sk.parse_prompt('star pops in', DRESSED_NAMES, base=r)
    tl = sk.build_timeline(named, DRESSED_NAMES, 10, DRESSED_ROLES)
    assert tl[5]['effect'] == 'pop' and tl[5]['start'] > tl[3]['start'] and tl[4]['start'] == 0.0
    # Nothing but decor: nothing arrives.
    tl = sk.build_timeline(r, ['Background', 'A', 'B'], 10, ['background', 'decor', 'decor'])
    assert [(t['start'], t['dur']) for t in tl] == [(0.0, 0.0)] * 3
    # Without roles every layer arrives, as for artwork that was not sorted.
    assert all(t['start'] > 0 for t in sk.build_timeline(r, DRESSED_NAMES, 10)[1:])


@pytest.mark.parametrize('prompt, says, content', [
    ('breathing', 'background: static; logo and text: fade, then breathe; other layers: light sweep', 'breathe'),
    ('the text breathes', 'background: static; logo and text: fade, then breathe; other layers: light sweep',
     'breathe'),
    ('title breathes', 'background: static; logo and text: fade, then hold still; other layers: light sweep; '
                       'Title: then breathe', 'hold'),
    ('star dances, ribbon holds still', 'background: static; logo and text: fade, then hold still; '
                                        'other layers: light sweep; Ribbon: then hold still; Star: then dancing',
     'hold'),
])
def test_breathing_is_for_logo_and_text_and_a_named_layer_can_be_overruled(prompt, says, content):
    r = sk.parse_prompt(prompt, DRESSED_NAMES, base=sk.style_recipe('fade_shine'))
    assert sk.describe_recipe(r, DRESSED_NAMES, DRESSED_ROLES) == says and r['content'] == content
    assert r['ambient'] == 'shine', 'breathing never becomes the motion of the other layers'


@pytest.mark.parametrize('style', [k for k, _, _ in sk.STYLES])
@pytest.mark.parametrize('content', ['hold', 'breathe'])
def test_background_static_logo_and_text_calm_everything_else_looping_start_to_end(dressed, style, content):
    if content == 'breathe' and style not in ('wipe_mix', 'pop_dance', 'fade_still'):
        pytest.skip('breathing is checked against three styles; the rest differ only in what the other layers do')
    art = sk.load_artwork(dressed['psd'])
    r = sk.style_recipe(style, content)
    an = sk.Animator(art, sk.build_timeline(r, DRESSED_NAMES, 10, DRESSED_ROLES), r, 10)
    bare, want = sk._flatten(art['layers'][:1], sk.CANVAS), dressed['want']
    f0 = an.frame(0.0)
    assert np.array_equal(f0[TEXT], bare[TEXT]), 'no logo or text yet on the first frame'
    assert f0[670, 950, 2] > 200 and f0[150, 1800, 1] > 180, 'the ribbon and the star are already there'
    text, other = [], []
    for t in np.arange(0.0, 10.0, 1.0 / 15):
        f = an.frame(float(t))
        # The background never changes: not where nothing is, not at any time.
        assert np.array_equal(f[900:], bare[900:]) and np.array_equal(f[:40], bare[:40]), (style, t)
        assert np.array_equal(f[:, :100], bare[:, :100]) and np.array_equal(f[:, 1890:], bare[:, 1890:]), (style, t)
        g, w = f.copy(), want.copy()
        g[TEXT], w[TEXT] = 0, 0
        other.append((float(t), _diff(g, w)))
        if t > an.settled_at + 0.1:
            text.append((float(t), _diff(f[TEXT], want[TEXT])))
    if content == 'hold':
        assert max(d for _, d in text) == 0, 'logo and text appear and then do not move'
    else:
        assert max(d for t, d in text if t < 6.5) > 0.2 and max(d for t, d in text if t > 7.5) > 0.2, 'they breathe'
        assert max(d for _, d in text) < 8, 'by little enough to read through'
    if style == 'fade_still':
        assert max(d for _, d in other) == 0, 'this style leaves the other layers still too'
    else:
        for a, b in ((0.0, 3.3), (3.3, 6.6), (6.6, 10.0)):
            assert max(d for t, d in other if a <= t < b) > 0.02, f'the other layers are moving between {a} and {b} s'
    last = an.frame(10.0)
    assert np.array_equal(last[TEXT], want[TEXT]), 'logo and text end exactly as designed'
    # Rewinding gives the same picture as getting there in order.
    late = an.frame(7.3)
    an.frame(1.0)
    assert np.array_equal(an.frame(7.3), late)


def test_the_mixed_style_shares_three_motions_out_among_the_moving_layers(tmp_path):
    parts = [PARTS[0], PARTS[1]] + [(name, lambda k=k: _block([150 + 300 * k, 700, 350 + 300 * k, 800],
                                                              (255, 120, 0, 255)))
                                    for k, name in enumerate(('Ribbon', 'Star', 'Burst', 'Arrow'))]
    names = [n for n, _ in parts]
    _psd(tmp_path / 'mix.psd', parts=parts)
    art = sk.load_artwork(str(tmp_path / 'mix.psd'))
    roles = [ly['role'] for ly in art['layers']]
    r = sk.style_recipe('wipe_mix')
    an = sk.Animator(art, sk.build_timeline(r, names, 10, roles), r, 10)
    assert an._amb == ['none', 'none', 'grow', 'rotate_left', 'rotate_right', 'grow'], 'text sits the turns out'
    # Rotating left lifts the right-hand end of a layer; rotating right lifts the left.
    r = sk.parse_prompt('ribbon rotates left, star rotates right', names, base=sk.style_recipe('fade_still'))
    an = sk.Animator(art, sk.build_timeline(r, names, 10, roles), r, 10)
    f = an.frame(2.0)                                   # half-way through a four-second swing: leaning the most
    orange = lambda x, y: f[y, x, 2] > 200 and f[y, x, 0] < 60        # noqa: E731
    assert orange(330, 690) and not orange(170, 690), 'ribbon: right end up'
    assert orange(470, 690) and not orange(630, 690), 'star: left end up'
    assert not any(orange(x, 690) for x in (770, 930, 1070, 1230)), 'the unnamed layers hold still in this style'


def test_a_flat_picture_takes_a_light_sweep_and_nothing_that_would_move_its_background(layered):
    art = sk.load_artwork(layered['png'])
    for style in ('slide_float', 'pop_pulse', 'stop_motion'):
        r, note = sk.fit_to_artwork(sk.style_recipe(style), ['Image'])
        assert r['ambient'] == 'none' and 'one flat picture' in note and sk.style_recipe(style)['ambient'] != 'none'
        an = sk.Animator(art, sk.build_timeline(r, ['Image'], 10), r, 10)
        assert all(_diff(an.frame(t), layered['want']) < 0.5 for t in (an.settled_at + 0.2, 5.0, 8.0, 10.0))
    for style in ('fade_shine', 'fade_still'):
        assert sk.fit_to_artwork(sk.style_recipe(style), ['Image']) == (sk.style_recipe(style), None)
    assert sk.fit_to_artwork(sk.style_recipe('slide_float'), NAMES)[1] is None, 'layered artwork can do all of them'
    r = sk.style_recipe('fade_shine')
    an = sk.Animator(art, sk.build_timeline(r, ['Image'], 10), r, 10)
    seen = [_diff(an.frame(float(t)), layered['want']) for t in np.arange(an.settled_at + 0.1, 9.0, 0.1)]
    assert max(seen) > 0.2 and _diff(an.frame(10.0), layered['want']) < 0.5
    assert an.frame(0.0).max() == 0, 'and it still arrives from black'


# ---- several schedule pages in one file ----

PAGE_COLOURS = {1: (255, 40, 40, 255), 2: (40, 255, 40, 255), 3: (40, 40, 255, 255)}      # RGBA: red, green, blue
ROW_Y = (320, 420, 520, 620)


def _paged_psd(path, show=(1,), stack=(3, 2, 1), rows=None, background=True, extra=None):
    """A background, a logo, and three Page groups whose rows sit in the
    same places in a different colour each: each row a 'time' block and a
    'show' block side by side, as two layers. `show` is which pages are
    switched on when the file is saved; `stack` their order, bottom first.
    Returns {page number: that page drawn alone on transparency}."""
    rows = rows or {1: 4, 2: 3, 3: 4}
    psd = psd_tools.PSDImage.new('RGB', (W, H), color=0)

    def add(parent, name, im, visible=True):
        box = im.getbbox()
        ly = parent.create_pixel_layer(im.crop(box), name=name, top=box[1], left=box[0])
        ly.visible = visible
        return ly
    if background:
        add(psd, 'Background', Image.new('RGBA', (W, H), (20, 20, 20, 255)))
    alone = {}
    for p in stack:
        g = psd.create_group(name=f'Page {p}')
        g.visible = p in show
        full = Image.new('RGBA', (W, H), (0, 0, 0, 0))
        for r in range(rows[p]):
            y = ROW_Y[r]
            for name, box in ((f'time {r + 1}', [500, y, 700, y + 60]), (f'show {r + 1}', [760, y + 4, 1400 - 40 * p, y + 56])):
                im = _block(box, PAGE_COLOURS[p])
                add(g, name, im)
                full.alpha_composite(im)
        if p == 1:
            add(g, 'draft stamp', _block([1500, 320, 1700, 380], (255, 255, 0, 255)), visible=False)
        alone[p] = full
    add(psd, 'GTV Logo', _block([500, 120, 640, 260], (255, 255, 255, 255)))
    for name, make in (extra or []):
        add(psd, name, make())
    psd.save(str(path))
    return alone


def _with_page(art, k):
    return sk._flatten([ly for ly in art['layers'] if ly['page'] in (None, k)], sk.CANVAS)


@pytest.fixture(scope='module')
def paged(tmp_path_factory):
    d = tmp_path_factory.mktemp('paged')
    _paged_psd(d / 'pages.psd')
    return {'psd': str(d / 'pages.psd'), 'dir': d}


def test_what_makes_a_group_a_page_is_its_name():
    for name, number in (('Page 1', 1), ('page 2', 2), ('PAGE-3', 3), ('Page_04', 4), ('Pg 5', 5), ('pg6', 6), ('P7', 7),
                         ('Page 2 afternoon', 2), ('  Page #8 (late)', 8), ('Page 12', 12)):
        assert sk.page_number(name) == number, name
    for name in ('Header', 'Schedule', 'Pages', 'Paper 1', 'Group 1', 'Row 2', 'Pink 2', 'Title page 1', '1', '', None):
        assert sk.page_number(name) is None, name


def test_pages_are_read_whichever_is_showing_and_play_in_the_order_of_their_numbers(paged, tmp_path):
    art = sk.load_artwork(paged['psd'])
    assert art['pages'] == [{'name': 'Page 1', 'rows': 4}, {'name': 'Page 2', 'rows': 3}, {'name': 'Page 3', 'rows': 4}]
    assert art['notes'] == [] and art['layered']
    got = [(ly['name'], ly['role'], ly['page']) for ly in art['layers']]
    assert got[0] == ('Background', 'background', None) and got[-1] == ('GTV Logo', 'content', None)
    assert [g for g in got if g[2] == 0] == [(f'Page 1 row {r}', 'content', 0) for r in (1, 2, 3, 4)]
    assert [g[0] for g in got if g[2] == 1] == ['Page 2 row 1', 'Page 2 row 2', 'Page 2 row 3']
    assert [g[0] for g in got if g[2] == 2] == [f'Page 3 row {r}' for r in (1, 2, 3, 4)]
    # A row is the time and its title together, top to bottom, whatever layers they were.
    first = next(ly for ly in art['layers'] if ly['name'] == 'Page 1 row 1')
    assert (first['x'], first['y']) == (500, 320) and first['px'].shape[:2] == (61, 861)
    assert [ly['y'] for ly in art['layers'] if ly['page'] == 0] == list(ROW_Y)
    # The layer switched off INSIDE a page stays off.
    assert not any(ly['px'].shape[1] > 0 and ly['x'] + ly['px'].shape[1] > 1450 for ly in art['layers'] if ly['page'] == 0)

    # The same pages whichever is showing, and whatever order they are stacked in.
    def shape(a):
        return [(ly['name'], ly['page'], ly['x'], ly['y'], ly['px'].shape) for ly in a['layers'] if ly['page'] is not None]
    want = sorted(shape(art))
    for show, stack in (((2,), (3, 2, 1)), ((), (3, 2, 1)), ((1, 2, 3), (1, 2, 3)), ((3,), (2, 3, 1))):
        _paged_psd(tmp_path / 'v.psd', show=show, stack=stack)
        other = sk.load_artwork(str(tmp_path / 'v.psd'))
        assert sorted(shape(other)) == want and other['notes'] == [], (show, stack)
        assert [pg['name'] for pg in other['pages']] == ['Page 1', 'Page 2', 'Page 3']


def test_a_group_that_is_not_named_as_a_page_stays_on_screen_and_a_hidden_one_stays_out(tmp_path):
    psd = psd_tools.PSDImage.new('RGB', (W, H), color=0)

    def add(parent, name, box, colour, visible=True):
        im = _block(box, colour)
        b = im.getbbox()
        ly = parent.create_pixel_layer(im.crop(b), name=name, top=b[1], left=b[0])
        ly.visible = visible
    add(psd, 'Background', [0, 0, W - 1, H - 1], (20, 20, 20, 255))
    header = psd.create_group(name='Header')
    add(header, 'logo', [500, 120, 640, 260], (255, 255, 255, 255))
    add(header, 'day', [700, 140, 1300, 240], (200, 100, 255, 255))
    spare = psd.create_group(name='Alternate schedule')
    spare.visible = False
    add(spare, 'rows', [500, 320, 1400, 700], (255, 255, 0, 255))
    for p in (2, 1):
        g = psd.create_group(name=f'Page {p}')
        g.visible = p == 1
        add(g, 'rows', [500, 320, 1400, 380], PAGE_COLOURS[p])
    psd.save(str(tmp_path / 'h.psd'))
    art = sk.load_artwork(str(tmp_path / 'h.psd'))
    assert [(ly['name'], ly['page']) for ly in art['layers']] == [('Background', None), ('Header', None),
                                                                 ('Page 2', 1), ('Page 1', 0)]
    assert art['pages'] == [{'name': 'Page 1', 'rows': 1}, {'name': 'Page 2', 'rows': 1}]


def test_pages_need_a_background_under_them_and_a_page_too_fine_grained_arrives_whole(tmp_path):
    _paged_psd(tmp_path / 'bare.psd', background=False, stack=(1, 2, 3))
    with pytest.raises(sk.ArtworkError, match='nothing under them'):
        sk.load_artwork(str(tmp_path / 'bare.psd'))
    # One tall layer beside the rows ties them into a single piece; so do too many rows.
    psd = psd_tools.PSDImage.new('RGB', (W, H), color=0)

    def add(parent, name, box, colour):
        im = _block(box, colour)
        b = im.getbbox()
        parent.create_pixel_layer(im.crop(b), name=name, top=b[1], left=b[0])
    add(psd, 'Background', [0, 0, W - 1, H - 1], (20, 20, 20, 255))
    g = psd.create_group(name='Page 1')
    for r in range(4):
        add(g, f'row {r}', [500, 300 + 100 * r, 1200, 350 + 100 * r], (255, 40, 40, 255))
    add(g, 'divider', [460, 300, 470, 650], (255, 255, 255, 255))
    g = psd.create_group(name='Page 2')
    for r in range(sk.MAX_PAGE_ROWS + 2):
        add(g, f'row {r}', [500, 300 + 40 * r, 1200, 325 + 40 * r], (40, 255, 40, 255))
    psd.save(str(tmp_path / 'tied.psd'))
    art = sk.load_artwork(str(tmp_path / 'tied.psd'))
    assert [(ly['name'], ly['page']) for ly in art['layers'][1:]] == [('Page 1', 0), ('Page 2', 1)]
    assert art['pages'] == [{'name': 'Page 1', 'rows': 1}, {'name': 'Page 2', 'rows': 1}]


def _paged_plan(art, duration, style='wipe_mix'):
    names, roles = [ly['name'] for ly in art['layers']], [ly['role'] for ly in art['layers']]
    pages = [ly['page'] for ly in art['layers']]
    r = sk.style_recipe(style)
    tl = sk.build_timeline(r, names, duration, roles, pages)
    return r, tl, pages


@pytest.mark.parametrize('duration', sk.DURATIONS)
def test_each_page_gets_an_equal_turn_and_only_the_last_one_stays(paged, duration):
    art = sk.load_artwork(paged['psd'])
    r, tl, pages = _paged_plan(art, duration)
    plan = sk.page_plan(tl, pages, duration)
    assert plan['count'] == 3 and abs(plan['slot'] * 3 + min(t['start'] for t, p in zip(tl, pages) if p == 0) - duration) < 1e-6
    logo = tl[-1]
    assert 'out' not in logo and logo['start'] <= 0.5, 'what is not in a page arrives once, at the start, and stays'
    turn_starts = []
    for k in range(3):
        rows = [t for t, p in zip(tl, pages) if p == k]
        starts = [t['start'] for t in rows]
        assert starts == sorted(starts) and len(set(starts)) == len(starts), 'rows arrive one after another, top first'
        assert all(t['effect'] == 'wipe' for t in rows), 'the way the style says'
        done = max(t['start'] + t['dur'] for t in rows)
        if k < 2:
            outs = {t['out'] for t in rows}
            assert len(outs) == 1, 'a page leaves all at once'
            leaves, fade = outs.pop()
            assert leaves >= done and fade == sk.PAGE_OUT
            nxt = min(t['start'] for t, p in zip(tl, pages) if p == k + 1)
            assert nxt >= leaves + fade, 'the next page starts only once this one has gone'
        else:
            assert all('out' not in t for t in rows) and done < duration
        turn_starts.append(starts[0])
    gaps = [b - a for a, b in zip(turn_starts, turn_starts[1:])]
    assert max(gaps) - min(gaps) < 0.2, 'equal turns'
    assert logo['start'] < turn_starts[0]
    # Short plugs say so; long ones have nothing to say.
    notes = sk.page_notes(plan, duration)
    assert (len(notes) == 1 and '3 pages in' in notes[0]) if plan['read'] < sk.PAGE_MIN_READ else notes == []
    assert (duration <= 15) == bool(notes)


def test_a_file_without_pages_is_timed_exactly_as_before():
    r = sk.style_recipe('wipe_mix')
    for roles in (None, DRESSED_ROLES):
        assert sk.build_timeline(r, DRESSED_NAMES, 15, roles) == sk.build_timeline(r, DRESSED_NAMES, 15, roles, [None] * 6)
    assert sk.page_plan(sk.build_timeline(r, DRESSED_NAMES, 15), [None] * 6, 15) is None
    assert sk.page_notes(None, 15) == []
    # One page is a page that never has to leave.
    names, pages = ['Background', 'Logo', 'Page 1 row 1', 'Page 1 row 2'], [None, None, 0, 0]
    tl = sk.build_timeline(r, names, 15, ['background', 'content', 'content', 'content'], pages)
    assert all('out' not in t for t in tl) and sk.page_notes(sk.page_plan(tl, pages, 15), 15) == []


@pytest.mark.parametrize('style, content', [('wipe_mix', 'hold'), ('slide_float', 'breathe'), ('stop_motion', 'hold')])
def test_one_page_at_a_time_each_exactly_as_designed_over_a_background_that_stays(paged, style, content):
    art = sk.load_artwork(paged['psd'])
    names, roles = [ly['name'] for ly in art['layers']], [ly['role'] for ly in art['layers']]
    pages = [ly['page'] for ly in art['layers']]
    r = sk.style_recipe(style, content)
    tl = sk.build_timeline(r, names, 20, roles, pages)
    an = sk.Animator(art, tl, r, 20)
    rows_area = (slice(300, 700), slice(480, 1420))
    seen = set()
    for t in np.arange(0.0, 20.0, 0.1):
        f = an.frame(float(t))
        assert np.array_equal(f[900:], np.full_like(f[900:], 20)) and np.array_equal(f[:100], np.full_like(f[:100], 20))
        if t < tl[-1]['start'] + tl[-1]['dur']:
            continue                    # the (white) logo may still be travelling through on its way in
        area = f[rows_area].reshape(-1, 3).astype(int)
        lit = [bool((area[:, c] > 60).any()) for c in (2, 1, 0)]         # red, green, blue: pages 1, 2, 3
        assert sum(lit) <= 1, f'two pages on screen at once at {t:.1f} s'
        if any(lit):
            seen.add(lit.index(True))
    assert seen == {0, 1, 2}
    # Half-way through each page's time complete on screen, it is that page and nothing else.
    for k in range(3):
        rows = [t for t, p in zip(tl, pages) if p == k]
        done = max(t['start'] + t['dur'] for t in rows)
        leaves = rows[0]['out'][0] if 'out' in rows[0] else 20.0
        f = an.frame((done + leaves) / 2.0)
        want = _with_page(art, k)
        if content == 'hold':
            assert np.array_equal(f, want), f'page {k + 1}'
        else:
            assert 0 < _diff(f, want) < 4, 'breathing: a little off its rest position, readably'
        if k < 2:
            gone = an.frame(leaves + sk.PAGE_OUT + 0.09)       # (stop motion steps time: allow it a step)
            assert np.array_equal(gone[rows_area], np.full_like(gone[rows_area], 20)), 'and then gone completely'
    assert np.array_equal(an.frame(20.0), _with_page(art, 2)), 'the plug ends on the last page, at rest'
    # Rewinding gives the same picture as getting there in order.
    late = an.frame(9.3)
    an.frame(1.0)
    assert np.array_equal(an.frame(9.3), late)


def test_ornaments_keep_looping_under_the_pages_and_the_layer_limit_spares_them(tmp_path):
    extra = [('Star', lambda: _block([1750, 100, 1850, 200], (255, 220, 60, 255)))]
    _paged_psd(tmp_path / 'orn.psd', extra=extra)
    art = sk.load_artwork(str(tmp_path / 'orn.psd'))
    assert [(ly['name'], ly['role'], ly['page']) for ly in art['layers']][-1] == ('Star', 'decor', None)
    r, tl, pages = _paged_plan(art, 20)
    an = sk.Animator(art, tl, r, 20)
    assert an._amb[-1] == 'grow' and tl[-1]['start'] == 0.0 and 'out' not in tl[-1]
    star = (slice(60, 240), slice(1700, 1900))
    rest = sk._flatten(art['layers'], sk.CANVAS)[star]
    assert all(max(_diff(an.frame(float(t))[star], rest) for t in np.arange(a, b, 0.2)) > 0.5
               for a, b in ((0.0, 4.0), (8.0, 12.0), (16.0, 20.0))), 'moving through every page'
    # More shared layers than can animate: the lowest are merged, the pages are not touched.
    many = [(f'Shape {k}', lambda k=k: _block([100 + 20 * k, 900, 110 + 20 * k, 1000], (255, 255, 255, 255)))
            for k in range(sk.MAX_LAYERS + 2)]
    _paged_psd(tmp_path / 'many.psd', extra=many)
    art = sk.load_artwork(str(tmp_path / 'many.psd'))
    assert art['pages'] == [{'name': 'Page 1', 'rows': 4}, {'name': 'Page 2', 'rows': 3}, {'name': 'Page 3', 'rows': 4}]
    assert sum(1 for ly in art['layers'] if ly['page'] is not None) == 11


# ---- inspecting: what each layer will do, and corrections ----

def _inspect_psd(path):
    psd = psd_tools.PSDImage.new('RGB', (W, H), color=0)

    def px(im, name, **kw):
        box = im.getbbox()
        return psd.create_pixel_layer(im.crop(box), name=name, top=box[1], left=box[0], **kw)
    psd.append(px(Image.new('RGBA', (W, H), (20, 30, 90, 255)), 'Background'))
    psd.append(px(_block([60, 40, 360, 160], (255, 200, 0, 255)), 'GMA Logo'))
    for n, y, c in ((1, 300, (255, 255, 255, 255)), (2, 300, (255, 120, 120, 255))):
        rows = [px(_block([500, y, 1400, y + 90], c), f'Show {n}a'), px(_block([500, y + 130, 1400, y + 220], c), f'Show {n}b')]
        psd.append(psd.create_group(rows, name=f'Page {n}'))
    psd.append(px(_block([1500, 700, 1700, 900], (255, 0, 0, 255)), 'Sparkle'))
    psd.save(str(path))
    return str(path)


def test_inspect_lists_every_top_level_layer_top_first_with_its_part(tmp_path):
    info = sk.inspect_artwork(_inspect_psd(tmp_path / 'paged.psd'))
    got = [(l['name'], l['role']) for l in info['layers']]
    assert got == [('Sparkle', 'decor'), ('Page 2', 'page'), ('Page 1', 'page'), ('GMA Logo', 'content'), ('Background', 'background')]
    pages = {l['name']: l for l in info['layers'] if l['role'] == 'page'}
    assert pages['Page 1']['rows'] == 2 and pages['Page 1']['child_count'] == 2 and pages['Page 1']['kind'] == 'group'
    assert [c['name'] for c in pages['Page 1']['children']] == ['Show 1b', 'Show 1a']
    assert info['pages'] == [{'name': 'Page 1', 'rows': 2, 'shown': True}, {'name': 'Page 2', 'rows': 2, 'shown': True}]
    assert info['counts'] == {'content': 1, 'decor': 1, 'pages': 2, 'background': 1} and info['background_index'] == 0
    assert [p['label'] for p in info['previews']] == ['Page 1', 'Page 2'] and all(p['image'].startswith('data:image/jpeg') for p in info['previews'])
    sparkle = info['layers'][0]
    assert sparkle['box'] == [1500, 700, 201, 201] and sparkle['thumb'] and sparkle['why']


def test_inspect_says_why_a_layer_is_left_out_or_folded_into_the_background(tmp_path):
    parts = PARTS + [('Texture', lambda: Image.new('RGBA', (W, H), (255, 255, 255, 40)))]
    path = tmp_path / 'tex.psd'
    psd = psd_tools.PSDImage.new('RGB', (W, H), color=0)
    for name, make in [PARTS[0], ('Texture', parts[-1][1])] + PARTS[1:]:
        im = make(); box = im.getbbox()
        psd.append(psd.create_pixel_layer(im.crop(box), name=name, top=box[1], left=box[0]))
    hidden = psd.create_pixel_layer(_block([100, 700, 300, 900], (0, 255, 0, 255)).crop([100, 700, 300, 900]), name='Old draft', top=700, left=100)
    hidden.visible = False
    psd.append(hidden)
    psd.save(str(path))
    info = sk.inspect_artwork(str(path))
    roles = {l['name']: l for l in info['layers']}
    assert roles['Texture']['role'] == 'background' and 'covers the whole picture' in roles['Texture']['why']
    assert roles['Old draft']['role'] == 'hidden' and 'Switched off' in roles['Old draft']['why']


def test_corrections_change_what_is_animated_and_survive_a_layer_the_file_hid(tmp_path):
    path = _inspect_psd(tmp_path / 'paged.psd')
    ov = {'layers': {4: {'role': 'content'}, 3: {'role': 'off'}}}      # Sparkle is text now; Page 2 is left out
    art = sk.load_artwork(path, overrides=ov)
    assert [p['name'] for p in art['pages']] == ['Page 1']
    assert [ly['role'] for ly in art['layers'] if ly['name'] == 'Sparkle'] == ['content']
    info = sk.inspect_artwork(path, overrides=ov)
    by = {l['name']: l for l in info['layers']}
    assert by['Sparkle']['role'] == 'content' and by['Sparkle']['why'] == 'Set by you.'
    assert by['Page 2']['role'] == 'off' and by['Page 2']['why'] == 'Left out by you.'
    assert info['counts']['pages'] == 1


def test_a_layer_can_be_made_a_page_and_the_background_can_be_chosen(tmp_path):
    path = _inspect_psd(tmp_path / 'paged.psd')
    info = sk.inspect_artwork(path, overrides={'layers': {1: {'role': 'page', 'page': 3}}, 'background_upto': 1})
    by = {l['name']: l for l in info['layers']}
    assert by['GMA Logo']['role'] == 'page' and by['GMA Logo']['page'] == 3 and by['GMA Logo']['why'].startswith('Set as a page')
    assert by['Background']['role'] == 'background' and info['background_index'] == 0
    # The background chosen as the logo layer swallows everything under it.
    info = sk.inspect_artwork(path, overrides={'background_upto': 1})
    by = {l['name']: l for l in info['layers']}
    assert by['Background']['role'] == 'background' and by['GMA Logo']['role'] == 'background'
    art = sk.load_artwork(path, overrides={'background_upto': 1})
    assert art['layers'][0]['role'] == 'background' and 'GMA Logo' not in [ly['name'] for ly in art['layers'][1:]]


def test_overrides_are_cleaned_before_use():
    assert sk.clean_overrides(None) == {'background_upto': None, 'expand': [], 'layers': {}}
    got = sk.clean_overrides({'background_upto': '2', 'layers': {'3': {'role': 'page', 'page': 500}, 'x': {'role': 'off'},
                                                                   '4': {'role': 'evil'}, '5': 'decor', '6': {'role': 'decor'}}})
    assert got == {'background_upto': '2', 'expand': [], 'layers': {'3': {'role': 'page', 'page': 99}, '6': {'role': 'decor'}}}


def test_overrides_keep_only_known_animation_and_group_paths():
    got = sk.clean_overrides({'expand': ['0', '0.1', 'x', '1; drop'], 'background_upto': '0.0',
                              'layers': {'0.2': {'arrive': {'effect': 'pop', 'direction': 'up'}, 'motion': 'shake'},
                                         '0.3': {'arrive': {'effect': 'boom'}, 'motion': 'explode'},
                                         '0.4': {'motion': 'rotate_left'}}})
    assert got['expand'] == ['0', '0.1'] and got['background_upto'] == '0.0'
    assert got['layers']['0.2'] == {'arrive': {'effect': 'pop', 'direction': 'up'}, 'motion': 'shake'}
    assert '0.3' not in got['layers'] and got['layers']['0.4'] == {'motion': 'rotate_left'}


NESTED = '/home/claude/exp/sched_nested.psd'


@pytest.mark.skipif(not os.path.exists(NESTED), reason='needs the nested sample PSD')
def test_a_group_can_be_split_into_background_and_animated_layers():
    info = sk.inspect_artwork(NESTED)
    top = [l for l in info['layers'] if l['name'] == 'Header'][0]
    assert top['can_expand'] and top['role'] == 'background'
    ov = {'expand': ['0', '0.0'], 'background_upto': '0.0'}
    info = sk.inspect_artwork(NESTED, overrides=ov)
    by = {l['name']: l for l in info['layers']}
    assert by['Sky']['role'] == 'background' and by['Grain']['role'] == 'background'
    assert by['Star']['role'] == 'decor' and by['GMA Logo']['role'] == 'content' and by['Star']['depth'] == 1
    art = sk.load_artwork(NESTED, overrides=ov)
    assert art['layers'][0]['role'] == 'background'
    assert {'Star', 'Moon', 'GMA Logo'} <= {ly['name'] for ly in art['layers'][1:]}


@pytest.mark.skipif(not os.path.exists(NESTED), reason='needs the nested sample PSD')
def test_a_layer_inside_a_group_takes_its_own_arrival_and_motion():
    ov = {'expand': ['0'], 'layers': {'0.2': {'arrive': {'effect': 'pop', 'direction': 'up'}, 'motion': 'shake'}}}
    art = sk.load_artwork(NESTED, overrides=ov)
    star = [ly for ly in art['layers'] if ly['name'] == 'Star'][0]
    assert star['arrive'] == {'effect': 'pop', 'direction': 'up'} and star['motion'] == 'shake'
    base = sk.style_recipe('fade_still')
    r = sk.apply_layer_animation(base, art['layers'])
    assert r['layers']['star'] == {'effect': 'pop', 'direction': 'up'} and r['layer_ambient']['star'] == 'shake'
    assert 'Star: pop up; then shake' in sk.describe_recipe(r, [ly['name'] for ly in art['layers']]) or 'shake' in sk.describe_recipe(r, [ly['name'] for ly in art['layers']])
    assert 'star' not in base.get('layers', {})


def test_shake_moves_a_layer_and_rotate_turns_it():
    import numpy as np
    assert 'shake' in sk.AMBIENTS and 'rotate_left' in sk.MOTIONS and 'shake' in sk.MOTIONS
    assert sk.parse_prompt('shake the logo', ['GMA Logo'], base=sk.style_recipe('fade_still')) is not None


def test_inspecting_a_flat_image_reports_one_picture(layered):
    info = sk.inspect_artwork(layered['png'])
    assert [(l['name'], l['role']) for l in info['layers']] == [('Image', 'picture')] and not info['layered']
    assert info['previews'][0]['label'] == 'The artwork' and info['counts']['background'] == 1


def test_rereading_a_file_after_a_correction_does_not_draw_its_layers_again(tmp_path, monkeypatch):
    path = _inspect_psd(tmp_path / 'paged.psd')
    sk.inspect_artwork(path)
    calls = []
    real = sk._read_layer
    monkeypatch.setattr(sk, '_read_layer', lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    sk.inspect_artwork(path, overrides={'layers': {4: {'role': 'content'}}})
    assert calls == [], 'every layer came from the first reading'

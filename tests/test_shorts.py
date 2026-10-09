"""
Vertical Shorts tab (shorts.py + shorts_core.py), exercised for real where it
can be: PySceneDetect, OpenCV and ffmpeg all run against small generated
videos, and the shorts that come out are decoded and inspected frame by
frame. Only the two network services this sandbox doesn't have are stood in
for -- Ollama (vision + story model) and faster-whisper -- by patching the
HTTP call / the transcription function.

What a stand-in can and can't show: these tests prove the plumbing (what is
sent, how replies are used, what happens when a service misbehaves) and the
ffmpeg output. They say nothing about how well a real model picks moments.

Requested by JUN: a separate section for cutting long-form video into
vertical drama shorts, rather than changing the promo generator.
"""
import json
import os
import re
import shutil
import subprocess
import time
import unittest.mock as mock
import zipfile
import io
import hashlib

import cv2
import numpy as np
import pytest

with mock.patch('requests.post'), mock.patch('requests.get'):
    import main
    import pipeline
    import shorts
    import core
    import auth
import shorts_core as sc


pytestmark = pytest.mark.skipif(shutil.which('ffmpeg') is None, reason='ffmpeg not available')

FF = ['ffmpeg', '-y', '-loglevel', 'error']


def _client(user_id=1, role='admin', username='admin'):
    client = main.app.test_client()
    tok = f'csrf-shorts-{user_id}'
    with client.session_transaction() as sess:
        sess.update(authed=True, user_id=user_id, username=username, role=role, csrf_token=tok)
    return client, {'X-CSRF-Token': tok}


def _frames(path):
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    cap.release()
    return out


def _rgb(frame, x, y):
    b, g, r = [int(v) for v in frame[y, x]]
    return r, g, b


def _colour(frame, x, y):
    r, g, b = _rgb(frame, x, y)
    if r > 150 and g < 100 and b < 100:
        return 'red'
    if r > 150 and g > 150 and b < 100:
        return 'yellow'
    if b > 150 and r < 100:
        return 'blue'
    if g > 90 and r < 90 and b < 90:
        return 'green'
    return f'other{(r, g, b)}'


def _probe(path):
    r = subprocess.run(['ffprobe', '-v', 'error', '-show_entries',
                        'stream=codec_type,codec_name,width,height,nb_frames,channels,sample_rate,color_space,pix_fmt,'
                        'r_frame_rate,duration',
                        '-of', 'json', str(path)], capture_output=True, text=True, timeout=30)
    streams = json.loads(r.stdout)['streams']
    return ({s['codec_type']: s for s in streams})


# --------------------------------------------------------------------------
# Generated sources
# --------------------------------------------------------------------------

def _halves(path, left, right, frames, rate='25', size='320x360'):
    subprocess.run(FF + ['-f', 'lavfi', '-i', f'color=c={left}:s={size}:r={rate}:d=10',
                         '-f', 'lavfi', '-i', f'color=c={right}:s={size}:r={rate}:d=10',
                         '-filter_complex', '[0][1]hstack', '-frames:v', str(frames),
                         '-c:v', 'libx264', '-g', '250', '-pix_fmt', 'yuv420p', str(path)], check=True, timeout=60)


@pytest.fixture(scope='module')
def split_source(tmp_path_factory):
    """640x360, two 50-frame shots, each a different colour on each half:
    shot A red|blue, shot B green|yellow. Which colour a 9:16 crop shows
    says exactly which side of which shot it was taken from. The video
    stream starts 60 ms after the audio -- more than a frame -- which is the
    condition that used to start every render a frame early."""
    d = tmp_path_factory.mktemp('split')
    _halves(d / 'a.mp4', 'red', 'blue', 50)
    _halves(d / 'b.mp4', 'green', 'yellow', 50)
    (d / 'l.txt').write_text(f"file '{d / 'a.mp4'}'\nfile '{d / 'b.mp4'}'\n")
    subprocess.run(FF + ['-f', 'concat', '-safe', '0', '-i', str(d / 'l.txt'), '-c', 'copy', str(d / 'ab.mp4')],
                   check=True, timeout=60)
    src = d / 'split.mp4'
    subprocess.run(FF + ['-itsoffset', '0.06', '-i', str(d / 'ab.mp4'),
                         '-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=48000:duration=6',
                         '-map', '0:v', '-map', '1:a', '-c:v', 'copy', '-c:a', 'aac', '-shortest', str(src)],
                   check=True, timeout=60)
    return src


@pytest.fixture(scope='module')
def episode(tmp_path_factory):
    """A 24-second 'episode': eight 3-second shots of different colours with
    a tone under them. Long enough to cut several 5-12 s shorts from, with
    hard cuts PySceneDetect reliably finds at 3, 6, 9 ... seconds."""
    d = tmp_path_factory.mktemp('episode')
    parts = []
    for i, c in enumerate(['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'navy', 'white']):
        p = d / f'p{i}.mp4'
        subprocess.run(FF + ['-f', 'lavfi', '-i', f'color=c={c}:s=320x180:d=3:r=25',
                             '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(p)], check=True, timeout=60)
        parts.append(p)
    (d / 'l.txt').write_text('\n'.join(f"file '{p}'" for p in parts))
    subprocess.run(FF + ['-f', 'concat', '-safe', '0', '-i', str(d / 'l.txt'), '-c', 'copy', str(d / 'v.mp4')],
                   check=True, timeout=60)
    src = d / 'episode.mp4'
    subprocess.run(FF + ['-i', str(d / 'v.mp4'), '-f', 'lavfi', '-i', 'sine=frequency=330:sample_rate=48000:duration=24',
                         '-map', '0:v', '-map', '1:a', '-c:v', 'copy', '-c:a', 'aac', '-shortest', str(src)],
                   check=True, timeout=60)
    return src


def _transcript():
    """Eleven 1.6-second lines, 0.4 s apart, starting at 1.0 s."""
    words, segs, t = [], [], 1.0
    for i in range(11):
        toks = ['Linya', str(i), 'ng', 'usapan.']
        for k, w in enumerate(toks):
            words.append({'start': round(t + k * 0.4, 2), 'end': round(t + k * 0.4 + 0.3, 2), 'word': w})
        segs.append({'start': t, 'end': round(t + 1.5, 2), 'text': ' '.join(toks)})
        t = round(t + 2.0, 2)
    return words, segs


# --------------------------------------------------------------------------
# Rendering (shorts_core against real ffmpeg)
# --------------------------------------------------------------------------

def test_probe_reports_the_video_start_offset(split_source):
    info = sc.probe_source('ffprobe', str(split_source))
    assert (info['width'], info['height'], info['frames']) == (640, 360, 100)
    assert abs(info['fps'] - 25.0) < 1e-6
    assert abs(info['v_offset'] - 0.06) < 0.002, 'fixture must reproduce the audio-starts-first condition'
    assert info['audio_index'] == 0 and info['sar'] == 1.0


def test_render_is_frame_exact_across_a_cut_even_when_video_starts_late(split_source, tmp_path):
    """The crop for each shot must switch on exactly the frame the picture
    does. Starting one frame early or late shows as a single frame of the
    next shot through the previous shot's crop window -- a visible flash at
    every cut -- and that is what happened before the source's video start
    offset was taken into account."""
    info = sc.probe_source('ffprobe', str(split_source))
    crop_w, _ = sc.crop_geometry(info['disp_w'], info['disp_h'])
    max_x = info['disp_w'] - crop_w
    start_f, n = 20, 60                       # source cut at frame 50 -> frame 30 of the short
    segs = [{'a': 0, 'b': 29, 'layout': 'crop', 'x': 0.0, 'keys': None},            # left of shot A: red
            {'a': 30, 'b': 59, 'layout': 'crop', 'x': float(max_x), 'keys': None}]  # right of shot B: yellow
    out = tmp_path / 'exact.mp4'
    ok, err = sc.render_short('ffmpeg', str(split_source), str(out), start_f, n, info, segs, preset='ultrafast')
    assert ok, err
    fr = _frames(out)
    assert len(fr) == n
    assert fr[0].shape[:2] == (1920, 1080)
    cols = [_colour(f, 540, 960) for f in fr]
    assert cols[:30] == ['red'] * 30, cols[:32]
    assert cols[30:] == ['yellow'] * 30, cols[28:34]

    st = _probe(out)
    assert st['video']['codec_name'] == 'h264' and st['video']['pix_fmt'] == 'yuv420p'
    assert st['video'].get('color_space') == 'bt709'
    assert st['audio']['codec_name'] == 'aac' and st['audio']['channels'] == 2 and st['audio']['sample_rate'] == '48000'


def test_render_survives_a_picture_that_decodes_smaller_than_it_was_measured(split_source, tmp_path):
    """The planner's size comes from OpenCV and the render's from ffmpeg,
    and they are not always the same number: a newer ffmpeg applies a .mov's
    clean-aperture crop, a Windows capture backend counts the 1088 coded
    lines of a 1080 picture. The crop window used to be cut straight out of
    whatever arrived, so a window one line taller than the picture stopped
    the render with "Invalid argument" and no short at all (seen on a
    Windows server). Here the planner believes the 640x360 source is
    648x368."""
    # sd_matrix off: an SD source is always rescaled (for its colour), which hid this; HD is where it bit.
    info = dict(sc.probe_source('ffprobe', str(split_source)), width=648, height=368, disp_w=648, disp_h=368,
                sd_matrix=False)
    crop_w, crop_h = sc.crop_geometry(info['disp_w'], info['disp_h'])
    assert crop_h == 368 > 360, 'a window taller than the real picture'
    for name, segs in (('crop', [{'a': 0, 'b': 39, 'layout': 'crop', 'x': 0.0, 'keys': None}]),
                       ('mixed', [{'a': 0, 'b': 19, 'layout': 'crop', 'x': 0.0, 'keys': None},
                                  {'a': 20, 'b': 39, 'layout': 'fit', 'x': None, 'keys': None}])):
        out = tmp_path / f'{name}.mp4'
        ok, err = sc.render_short('ffmpeg', str(split_source), str(out), 0, 40, info, segs, preset='ultrafast')
        assert ok, err
        fr = _frames(out)
        assert len(fr) == 40 and fr[0].shape[:2] == (1920, 1080)
        assert _colour(fr[0], 540, 960) == 'red', 'still the left of shot A'


def test_mixed_layouts_switch_on_the_cut_and_fit_shows_the_whole_frame(split_source, tmp_path):
    info = sc.probe_source('ffprobe', str(split_source))
    segs = [{'a': 0, 'b': 29, 'layout': 'crop', 'x': 0.0, 'keys': None},
            {'a': 30, 'b': 59, 'layout': 'fit', 'x': None, 'keys': None}]
    out = tmp_path / 'mixed.mp4'
    ok, err = sc.render_short('ffmpeg', str(split_source), str(out), 20, 60, info, segs, preset='ultrafast')
    assert ok, err
    fr = _frames(out)
    assert len(fr) == 60
    assert _colour(fr[29], 540, 960) == 'red' and _colour(fr[29], 540, 100) == 'red', 'crop fills the frame'
    # Fit: the full 16:9 picture as a band across the middle (608 px tall on a
    # 1920 canvas), both halves of shot B visible, blurred picture above and below.
    assert _colour(fr[30], 200, 960) == 'green' and _colour(fr[30], 880, 960) == 'yellow'
    band_top = (1920 - 608) // 2
    assert _colour(fr[30], 200, band_top + 20) == 'green'
    top = _rgb(fr[30], 200, 100)
    assert top != _rgb(fr[30], 200, 960) and sum(top) > 30, 'bars are a darkened blur of the picture, not black'


def test_split_screen_stacks_the_two_sides_and_switches_on_the_cut(split_source, tmp_path):
    """Shot A is red|blue and shot B green|yellow, so a pane's colour says
    which side of which shot it was cut from. Two "people", one deep in each
    half of the picture: the left one must come out on top, the right one
    below, each in a pane the other's side does not reach the middle of."""
    info = sc.probe_source('ffprobe', str(split_source))
    crop_w, _ = sc.crop_geometry(info['disp_w'], info['disp_h'])
    faces = [(60.0, 150.0, 60.0, 60.0), (580.0, 150.0, 60.0, 60.0)]
    samples = [(i, [(100.0, 150.0, 60.0, 60.0)] if i < 20 else faces) for i in range(0, 60, 2)]
    # The short runs from source frame 20; the source cuts at 50, i.e. frame 30 here.
    segs = sc.plan_reframe(samples, [20, 30], 60, info['disp_w'], info['disp_h'], crop_w, mode='split')
    assert [(s['a'], s['b'], s['layout']) for s in segs] == [(0, 19, 'crop'), (20, 59, 'split')]
    out = tmp_path / 'splitscreen.mp4'
    ok, err = sc.render_short('ffmpeg', str(split_source), str(out), 20, 60, info, segs, preset='ultrafast')
    assert ok, err
    fr = _frames(out)
    assert len(fr) == 60 and fr[0].shape[:2] == (1920, 1080)
    assert _colour(fr[19], 540, 480) == 'red' and _colour(fr[19], 540, 1440) == 'red', 'still the single crop'
    # Frame 20 on: top pane from the left of the picture, bottom pane from the right.
    assert (_colour(fr[20], 200, 480), _colour(fr[20], 880, 1440)) == ('red', 'blue')
    assert (_colour(fr[29], 200, 480), _colour(fr[29], 880, 1440)) == ('red', 'blue')
    # And the source's own cut lands on its frame inside the split.
    assert (_colour(fr[30], 200, 480), _colour(fr[30], 880, 1440)) == ('green', 'yellow')
    assert sum(_rgb(fr[40], 540, 960)) < 120, 'a dark line where the two panes meet'

    # All three layouts in one short.
    three = [{'a': 0, 'b': 19, 'layout': 'fit', 'x': None, 'keys': None}, dict(segs[1], b=39),
             {'a': 40, 'b': 59, 'layout': 'crop', 'x': float(info['disp_w'] - crop_w), 'keys': None}]
    out3 = tmp_path / 'three.mp4'
    ok, err = sc.render_short('ffmpeg', str(split_source), str(out3), 20, 60, info, three, preset='ultrafast')
    assert ok, err
    fr = _frames(out3)
    assert (_colour(fr[10], 200, 960), _colour(fr[10], 880, 960)) == ('red', 'blue'), 'fit: both halves side by side'
    assert (_colour(fr[25], 200, 480), _colour(fr[25], 880, 1440)) == ('red', 'blue'), 'split: stacked'
    assert _colour(fr[45], 540, 480) == 'yellow' and _colour(fr[45], 540, 1440) == 'yellow', 'crop: right of shot B'


def test_a_panning_crop_moves_smoothly_between_keyframes(split_source, tmp_path):
    info = sc.probe_source('ffprobe', str(split_source))
    crop_w, _ = sc.crop_geometry(info['disp_w'], info['disp_h'])
    max_x = float(info['disp_w'] - crop_w)
    # Shot A only (frames 0-49): pan from the red half to the blue half.
    segs = [{'a': 0, 'b': 39, 'layout': 'crop', 'x': None, 'keys': [(0, 0.0), (20, max_x / 2), (39, max_x)]}]
    out = tmp_path / 'pan.mp4'
    ok, err = sc.render_short('ffmpeg', str(split_source), str(out), 0, 40, info, segs, preset='ultrafast')
    assert ok, err
    fr = _frames(out)
    assert len(fr) == 40
    # The red|blue boundary sits at x=320 in the source; as the window slides
    # right it must travel steadily left across the output, never jump.
    def boundary(f):
        row = f[960, :, :].astype(int)
        blue = np.where(row[:, 0] > 150)[0]
        return int(blue[0]) if len(blue) else 1080
    xs = [boundary(f) for f in fr]
    assert xs[0] == 1080 and xs[-1] == 0, (xs[0], xs[-1])
    assert all(b <= a for a, b in zip(xs, xs[1:])), 'monotonic'
    assert max(a - b for a, b in zip(xs, xs[1:])) < 120, 'no jump between keyframes'


def test_captions_are_burned_in(split_source, tmp_path):
    if not sc.has_filter('ffmpeg', 'ass'):
        pytest.skip('this ffmpeg build has no libass')
    info = sc.probe_source('ffprobe', str(split_source))
    segs = [{'a': 0, 'b': 24, 'layout': 'crop', 'x': 0.0, 'keys': None}]
    work = tmp_path / 'work dir with spaces'       # the caption file is found by bare name, wherever this is
    work.mkdir()
    sc.write_ass([{'start': 0.0, 'end': 1.0, 'text': 'WALA KANG KARAPATAN'}], str(work / 'cap_1.ass'))
    plain, capped = tmp_path / 'plain.mp4', tmp_path / 'capped.mp4'
    for out, ass in ((plain, None), (capped, 'cap_1.ass')):
        ok, err = sc.render_short('ffmpeg', str(split_source), str(out), 0, 25, info, segs, ass_name=ass,
                                  work_dir=str(work), preset='ultrafast')
        assert ok, err
    a, b = _frames(plain)[10], _frames(capped)[10]
    region = (slice(1300, 1560), slice(80, 1000))          # where a bottom-centre cue with a 23% lift lands
    assert int((a[region].min(axis=2) > 200).sum()) == 0, 'no white on the plain red frame'
    assert int((b[region].min(axis=2) > 200).sum()) > 2000, 'white caption text is present'
    untouched = (slice(100, 900), slice(0, 1080))
    assert float(np.abs(a[untouched].astype(int) - b[untouched].astype(int)).mean()) < 1.0


def test_sd_anamorphic_source_is_squared_up_and_converted_to_bt709(tmp_path):
    r_, g_, b_ = 30, 200, 60
    src = tmp_path / 'sd.mp4'
    subprocess.run(FF + ['-f', 'lavfi', '-i', f'color=c=0x{r_:02x}{g_:02x}{b_:02x}:s=720x480:r=30000/1001:d=1',
                         '-vf', 'scale=out_color_matrix=bt601:out_range=tv,format=yuv420p,setsar=32/27',
                         '-colorspace', 'smpte170m', '-color_primaries', 'smpte170m', '-color_trc', 'smpte170m',
                         '-c:v', 'libx264', str(src)], check=True, timeout=60)
    info = sc.probe_source('ffprobe', str(src))
    assert info['sd_matrix'] and abs(info['sar'] - 32 / 27) < 1e-3
    assert (info['disp_w'], info['disp_h']) == (854, 480) and info['audio_index'] is None
    out = tmp_path / 'sd_out.mp4'
    segs = [{'a': 0, 'b': 9, 'layout': 'crop', 'x': 292.0, 'keys': None}]
    ok, err = sc.render_short('ffmpeg', str(src), str(out), 0, 10, info, segs, preset='ultrafast')
    assert ok, err
    st = _probe(out)
    assert (st['video']['width'], st['video']['height']) == (1080, 1920)
    assert 'audio' not in st, 'a silent source renders without an audio track rather than failing'
    raw = subprocess.run(['ffmpeg', '-loglevel', 'error', '-i', str(out), '-frames:v', '1',
                          '-f', 'rawvideo', '-pix_fmt', 'yuv420p', '-'], capture_output=True, timeout=60).stdout
    w, h = 1080, 1920
    y = raw[(h // 2) * w + w // 2]
    u = raw[w * h + (h // 4) * (w // 2) + w // 4]
    v = raw[w * h + (w * h) // 4 + (h // 4) * (w // 2) + w // 4]
    # The same colour coded as BT.709 is (148, 84, 59); left as BT.601 it would be (130, 92, 63).
    assert abs(y - 148) <= 4 and abs(u - 84) <= 4 and abs(v - 59) <= 4, (y, u, v)


def test_probe_picks_the_audio_stream_with_the_most_channels(split_source, tmp_path):
    src = tmp_path / 'two_audio.mkv'
    subprocess.run(FF + ['-i', str(split_source), '-f', 'lavfi', '-i', 'sine=frequency=200:duration=4',
                         '-map', '0:v', '-map', '1:a', '-map', '0:a', '-c:v', 'copy',
                         '-c:a:0', 'aac', '-ac:a:0', '1', '-c:a:1', 'aac', '-ac:a:1', '2', '-shortest', str(src)],
                   check=True, timeout=60)
    assert sc.probe_source('ffprobe', str(src))['audio_index'] == 1


def test_render_failure_reports_ffmpeg_error_and_leaves_no_file(split_source, tmp_path):
    info = sc.probe_source('ffprobe', str(split_source))
    segs = [{'a': 0, 'b': 9, 'layout': 'crop', 'x': 0.0, 'keys': None}]
    out = tmp_path / 'broken.mp4'
    ok, err = sc.render_short('ffmpeg', str(split_source), str(out), 0, 10, info, segs,
                              ass_name='does_not_exist.ass', work_dir=str(tmp_path), preset='ultrafast')
    assert ok is False and err and not out.exists()


def test_face_detector_falls_back_to_bundled_cascades_and_scales_boxes_back():
    det = sc.FaceDetector(yunet_model='/nonexistent/model.onnx')
    assert det.kind == 'haar' and det.available()
    assert det.detect(np.zeros((1080, 1920, 3), np.uint8)) == []


# --------------------------------------------------------------------------
# The tab: fixtures
# --------------------------------------------------------------------------

@pytest.fixture
def env(tmp_path, monkeypatch, episode):
    """Everything a tab test needs: an isolated batches folder, no leftover
    analyses, jobs that run inline, a private rate limiter (so these tests
    neither trip nor use up the shared one), and the episode staged the way
    Browse library stages a network file."""
    monkeypatch.setattr(shorts, 'SHORTS_DIR', str(tmp_path / 'shorts'))
    os.makedirs(shorts.SHORTS_DIR)
    # ...and an isolated projects folder with one project in it, since shorts
    # cannot be generated without one to file them under.
    monkeypatch.setattr(shorts, 'SHORTS_PROJECTS_DIR', str(tmp_path / 'projects'))
    os.makedirs(shorts.SHORTS_PROJECTS_DIR)
    project = shorts.sp.create(shorts.SHORTS_PROJECTS_DIR, {'title': 'Tadhana', 'episode': 'Ep. 101',
                                                            'air_date': None, 'description': ''},
                               user_id=7, username='ana')
    monkeypatch.setattr(shorts, 'ANALYSES', {})
    monkeypatch.setattr(shorts, 'SHORTS_ANALYSES_DIR', str(tmp_path / 'analyses'))
    monkeypatch.setattr(shorts, '_spawn', lambda fn, *a, **k: fn(*a, **k))
    monkeypatch.setattr(shorts, '_job_submit_limiter', core._RateLimiter(1000, 300))
    monkeypatch.setattr(shorts, 'SHORTS_PRESET', 'ultrafast')
    monkeypatch.setattr(pipeline, 'ALLOW_LOCAL_MEDIA_UPLOAD', False)
    staged = f'net_{int(time.time())}_episode.mp4'
    shutil.copy(str(episode), os.path.join(main.app.config['UPLOAD_FOLDER'], staged))
    yield {'staged': staged, 'path': os.path.join(main.app.config['UPLOAD_FOLDER'], staged),
           'project': project['project_id']}
    try:
        os.remove(os.path.join(main.app.config['UPLOAD_FOLDER'], staged))
    except OSError:
        pass


class Services:
    """Stand-in Ollama + whisper. Records what it was asked."""

    def __init__(self, monkeypatch, words=None, segs=None):
        w, s = _transcript()
        self.words = w if words is None else words
        self.segs = s if segs is None else segs
        self.vision_calls, self.story_prompts, self.unloaded = [], [], []
        self.vision_reply = lambda n: {'response': json.dumps({'score': 4 if n % 2 else 2, 'desc': 'two people arguing'})}
        self.story_reply = self._default_story
        monkeypatch.setattr(sc.requests, 'post', self._post)
        self.heard = {'ok': True, 'reason': None if (self.words or self.segs)
                      else 'the speech-to-text service found no speech in the audio'}
        monkeypatch.setattr(pipeline, 'transcribe_video_detailed',
                            lambda path: (self.words, self.segs, self.heard))
        monkeypatch.setattr(pipeline, 'unload_ollama_model', lambda m: self.unloaded.append(m))

    @staticmethod
    def _default_story(ids, payload):
        return {'response': json.dumps({'moments': [
            {'start_id': ids[1], 'end_id': ids[4], 'title': 'Ang lihim ni Ramon', 'hook': 'Linya 1', 'why': 'A reveal.', 'score': 9},
            {'start_id': ids[7], 'end_id': ids[10], 'title': 'Huling babala', 'hook': 'Linya 7', 'why': 'A threat.', 'score': 6},
        ]})}

    def _post(self, url, json=None, timeout=None, **kw):
        import re
        resp = mock.Mock()
        payload = json or {}
        if payload.get('images'):
            self.vision_calls.append(payload)
            out = self.vision_reply(len(self.vision_calls))
        else:
            self.story_prompts.append(payload)
            ids = [int(x) for x in re.findall(r'^\[(\d+)\]', payload.get('prompt', ''), re.M)]
            out = self.story_reply(ids, payload)
        if isinstance(out, Exception):
            raise out
        resp.json.return_value = out
        return resp


def _analyze(client, headers, env, **form):
    data = {'shorts_file_network': env['staged'], 'min_dur': 5, 'max_dur': 12, 'count': 5,
            'project_id': env['project']}
    data.update(form)
    r = client.post('/api/shorts/analyze', data=data, headers=headers)
    assert r.status_code == 200, r.get_json()
    return client.get(f"/api/shorts/progress/{r.get_json()['job_id']}").get_json()


def _analysis(client, job):
    assert job.get('error') is None, job.get('error')
    return client.get(f"/api/shorts/analysis/{job['result']['analysis_id']}").get_json()


def _render(client, headers, aid, items, **opts):
    body = {'analysis_id': aid, 'items': items}
    body.update(opts)
    r = client.post('/api/shorts/render', json=body, headers=headers)
    assert r.status_code == 200, r.get_json()
    return client.get(f"/api/shorts/progress/{r.get_json()['job_id']}").get_json()


# --------------------------------------------------------------------------
# The tab: registration and permissions
# --------------------------------------------------------------------------

def test_tab_is_registered_and_gated_by_its_own_permission(users_db):
    assert 'vertical_shorts' in auth._PERMISSION_KEYS
    client, _ = _client()
    html = client.get('/').get_data(as_text=True)
    assert "switchTab('p-shorts',this)" in html and 'id="p-shorts"' in html
    assert html.index("switchTab('p-trailer',this)") < html.index("switchTab('p-shorts',this)") \
        < html.index("switchTab('p-music',this)"), 'sits directly under Episodic Plug'

    # An account whose group grants promos but not shorts: no tab, and the API says no.
    ok, _err, gid = auth.group_create('Promo editors')
    auth.group_set_permissions(gid, ['promo_generation'])
    auth.user_create('editor', 'a-genuinely-long-password-1')
    uid = auth.user_get_by_username('editor')['id']
    auth.user_set_group(uid, gid)
    client, headers = _client(user_id=uid, role='user', username='editor')
    html = client.get('/').get_data(as_text=True)
    assert "switchTab('p-shorts',this)" not in html and "switchTab('p-trailer',this)" in html
    for method, url in (('get', '/api/shorts/options'), ('get', '/api/shorts/batches'),
                        ('post', '/api/shorts/analyze'), ('post', '/api/shorts/render')):
        assert getattr(client, method)(url, headers=headers).status_code == 403, url

    auth.group_set_permissions(gid, ['promo_generation', 'vertical_shorts'])
    assert "switchTab('p-shorts',this)" in client.get('/').get_data(as_text=True)
    assert client.get('/api/shorts/batches').status_code == 200


def test_endpoints_require_a_login():
    client = main.app.test_client()
    assert client.get('/api/shorts/batches').status_code == 401
    assert client.get('/api/shorts/file/1700000000_abcdef/x.mp4').status_code == 401


# --------------------------------------------------------------------------
# The tab: analyse
# --------------------------------------------------------------------------

def test_analyze_rejects_bad_requests_before_starting_a_job(env, monkeypatch):
    client, headers = _client()
    started = []
    monkeypatch.setattr(shorts, '_spawn', lambda fn, *a, **k: started.append(1))

    pj = {'project_id': env['project']}
    # No project, or one that does not exist: nowhere to file the shorts, so nothing starts.
    for missing in ({}, {'project_id': ''}, {'project_id': 'p0000000000'}, {'project_id': '../../etc'}):
        r = client.post('/api/shorts/analyze', data=dict(missing, shorts_file_network=env['staged']), headers=headers)
        assert r.status_code == 400 and 'Choose the project' in r.get_json()['error'], missing
    r = client.post('/api/shorts/analyze', data=pj, headers=headers)
    assert r.status_code == 400 and 'No video' in r.get_json()['error']
    r = client.post('/api/shorts/analyze', data=dict(pj, shorts_file_network='net_1_gone.mp4'), headers=headers)
    assert r.status_code == 400 and 're-select' in r.get_json()['error']
    # Only files this app staged itself are trusted -- never an arbitrary name or path.
    for bad in ('../../etc/passwd', os.path.basename(env['path']).replace('net_', 'src_')):
        r = client.post('/api/shorts/analyze', data=dict(pj, shorts_file_network=bad), headers=headers)
        assert r.status_code == 400
    r = client.post('/api/shorts/analyze', headers=headers,
                    data=dict(pj, shorts_file_network=env['staged'], min_dur=60, max_dur=62))
    assert r.status_code == 400 and 'at least 5 seconds' in r.get_json()['error']
    # Direct upload is refused server-side when the deployment has it off.
    r = client.post('/api/shorts/analyze', headers=headers, content_type='multipart/form-data',
                    data=dict(pj, shorts_file=(io.BytesIO(b'not really a video'), 'ep.mp4')))
    assert r.status_code == 400 and 'Direct file upload is disabled' in r.get_json()['error']
    assert started == []

    monkeypatch.setattr(shorts, '_job_submit_limiter', core._RateLimiter(1, 300))
    ok = dict(pj, shorts_file_network=env['staged'])
    assert client.post('/api/shorts/analyze', data=ok, headers=headers).status_code == 200
    assert client.post('/api/shorts/analyze', data=ok, headers=headers).status_code == 429
    assert len(started) == 1


def test_analyze_refuses_to_start_when_a_required_service_is_down(env, monkeypatch):
    Services(monkeypatch)
    monkeypatch.setattr(pipeline, '_check_service',
                        lambda name, url, path='/', timeout=3: {'status': 'down', 'error': 'connection refused'}
                        if name == 'whisper' else {'status': 'up'})
    detect = mock.Mock(side_effect=AssertionError('must not get as far as scene detection'))
    monkeypatch.setattr(pipeline, 'detect_scenes', detect)
    client, headers = _client()
    job = _analyze(client, headers, env)
    assert job['done'] and 'faster-whisper' in job['error'] and 'connection refused' in job['error']
    assert 'Ollama' not in job['error'], 'only the service that is actually down is named'
    assert shorts.ANALYSES == {}


def test_analyze_produces_ranked_snapped_candidates(env, monkeypatch):
    svc = Services(monkeypatch)
    client, headers = _client()
    job = _analyze(client, headers, env, focus='  ang lihim   ni Ramon ', vision_model='test-vl', vision_frames=10)
    a = _analysis(client, job)
    assert a['ok'] and a['orig_name'] == 'episode.mp4' and abs(a['duration'] - 24.0) < 0.1
    assert a['stats']['shots'] == 8 and a['stats']['transcript_lines'] == 11 and a['stats']['story_parts'] == 1
    assert a['warnings'] == []

    # Layer 1: a bounded number of vision calls, to the chosen model, with an image and a constrained reply.
    assert 1 <= len(svc.vision_calls) <= 10 + 4 * 4 and a['stats']['frames_rated'] == len(svc.vision_calls)
    assert a['stats']['frames_inside'] >= 1, 'frames inside the moments are rated too'
    assert all(c['model'] == 'test-vl' and c['images'] and c['format']['required'] == ['score', 'desc', 'kind']
               for c in svc.vision_calls)
    # Layer 2: one story call (24 s fits one chunk), same model by default, big enough context, with
    # the visual notes and the editor's focus in the prompt.
    assert len(svc.story_prompts) == 1
    p = svc.story_prompts[0]
    assert p['model'] == 'test-vl' and p['options']['num_ctx'] >= 8192 and 'moments' in p['format']['properties']
    assert '[SCREEN ' in p['prompt'] and 'two people arguing' in p['prompt'] and 'ang lihim ni Ramon' in p['prompt']
    assert '5 to 12 seconds' in p['prompt']
    # The GPU is handed back before whisper, after the story, and after the frames inside the moments.
    assert svc.unloaded == ['test-vl', 'test-vl', 'test-vl']

    cands = a['candidates']
    assert [c['title'] for c in cands] == ['Ang lihim ni Ramon', 'Huling babala']
    assert cands[0]['score'] > cands[1]['score'] and cands[0]['story_score'] == 9
    words, segs = _transcript()
    for c, (i, j) in zip(cands, ((1, 4), (7, 10))):
        assert 5 <= c['duration'] <= 12.5
        assert c['start'] <= segs[i]['start'] and c['end'] >= segs[j]['end'], 'the chosen lines are fully inside'
        cut_words = [w for w in words if w['start'] < c['start'] < w['end'] or w['start'] < c['end'] < w['end']]
        assert cut_words == [], 'no word straddles an in or out point'
        assert c['thumb'] and os.path.exists(os.path.join(shorts.SHORTS_ANALYSES_DIR, a['analysis_id'],
                                                          os.path.basename(c['thumb'])))
        assert client.get(c['thumb']).status_code == 200
        assert c['visual_score'] is not None and c['text'].startswith(f'Linya {i} ')
    # Line 4 ends at 10.5 s and line 5 starts at 11.0: no cut fits between them, so it ends on a short tail.
    assert 10.5 <= cands[0]['end'] < 11.0
    # Line 7 starts at 15.0, exactly on the cut at 15.0 s -- the in point lands just before the word, after line 6.
    assert 14.5 < cands[1]['start'] <= 15.0


def test_analyze_uses_a_separate_story_model_when_one_is_chosen(env, monkeypatch):
    svc = Services(monkeypatch)
    client, headers = _client()
    _analysis(client, _analyze(client, headers, env, vision_model='eyes', story_model='brain'))
    assert {c['model'] for c in svc.vision_calls} == {'eyes'}
    assert [p['model'] for p in svc.story_prompts] == ['brain']
    assert svc.unloaded == ['eyes', 'brain', 'eyes']


def test_vision_model_failing_on_every_frame_fails_the_job_with_the_reason(env, monkeypatch):
    svc = Services(monkeypatch)
    svc.vision_reply = lambda n: {'error': "model 'nope' not found"}
    client, headers = _client()
    job = _analyze(client, headers, env, vision_model='nope')
    assert job['done'] and '"nope"' in job['error'] and "not found" in job['error']
    assert svc.story_prompts == [], 'no point transcribing or asking for a story on half the evidence'


def test_some_unrated_frames_are_a_warning_not_a_failure(env, monkeypatch):
    svc = Services(monkeypatch)
    svc.vision_reply = lambda n: ({'error': 'out of memory'} if n == 1 else
                                  {'response': json.dumps({'score': 3, 'desc': 'talking'})})
    monkeypatch.setattr(pipeline, 'AI_SCORE_WORKERS', 1)
    client, headers = _client()
    a = _analysis(client, _analyze(client, headers, env))
    assert len(a['candidates']) == 2
    assert any('could not be rated' in w and 'out of memory' in w for w in a['warnings'])


def test_story_model_failing_everywhere_fails_the_job(env, monkeypatch):
    svc = Services(monkeypatch)
    svc.story_reply = lambda ids, payload: {'error': 'model requires more system memory'}
    client, headers = _client()
    job = _analyze(client, headers, env, story_model='too-big')
    assert job['done'] and '"too-big"' in job['error'] and 'more system memory' in job['error']


def test_no_story_found_falls_back_to_dialogue_stretches_and_says_so(env, monkeypatch):
    svc = Services(monkeypatch)
    svc.story_reply = lambda ids, payload: {'response': '{"moments": []}'}
    client, headers = _client()
    a = _analysis(client, _analyze(client, headers, env))
    assert a['candidates'] and all(c['source'] == 'heuristic' and c['story_score'] is None for c in a['candidates'])
    assert any('did not find any moment that stands on its own' in w for w in a['warnings'])


def test_no_dialogue_falls_back_to_visual_peaks_and_says_so(env, monkeypatch):
    svc = Services(monkeypatch, words=[], segs=[])
    svc.vision_reply = lambda n: {'response': json.dumps({'score': 5, 'desc': 'a fight'})}
    client, headers = _client()
    a = _analysis(client, _analyze(client, headers, env))
    assert svc.story_prompts == []
    assert a['candidates'] and all(c['source'] == 'visual' for c in a['candidates'])
    assert any('No dialogue was transcribed' in w for w in a['warnings'])
    # With nothing spoken to protect, in and out points sit on shot changes.
    for c in a['candidates']:
        assert abs(c['start'] / 3.0 - round(c['start'] / 3.0)) < 0.02
        assert abs(c['end'] / 3.0 - round(c['end'] / 3.0)) < 0.02


def test_what_to_avoid_reaches_the_story_model_and_is_kept_with_the_analysis(env, monkeypatch):
    svc = Services(monkeypatch)
    client, headers = _client()
    a = _analysis(client, _analyze(client, headers, env, focus='the will', avoid='  the  hospital scenes  '))
    assert svc.story_prompts and all('The editor does NOT want: the hospital scenes' in p['prompt']
                                     for p in svc.story_prompts)
    assert all('especially wants moments about: the will' in p['prompt'] for p in svc.story_prompts)
    assert not any('neither was applied' in w for w in a['warnings'])
    # Left empty, the prompt is exactly what it was before the field existed.
    svc.story_prompts.clear()
    _analyze(client, headers, env, avoid='   ')
    assert svc.story_prompts and not any('does NOT want' in p['prompt'] for p in svc.story_prompts)


def test_what_to_avoid_cannot_steer_a_fallback_and_the_editor_is_told(env, monkeypatch):
    """With no dialogue the moments are picked on picture alone. Nothing
    there reads meaning, so an "avoid" note silently did nothing -- which
    reads as the note having been obeyed. It has to be said."""
    svc = Services(monkeypatch, words=[], segs=[])
    svc.vision_reply = lambda n: {'response': json.dumps({'score': 5, 'desc': 'a fight'})}
    client, headers = _client()
    a = _analysis(client, _analyze(client, headers, env, avoid='fights'))
    assert all(c['source'] == 'visual' for c in a['candidates'])
    assert any('neither was applied' in w for w in a['warnings'])
    plain = _analysis(client, _analyze(client, headers, env))
    assert not any('neither was applied' in w for w in plain['warnings']), 'only when a note was actually given'


def test_generate_without_preview_analyses_then_renders_everything_as_one_job(env, monkeypatch):
    Services(monkeypatch)
    client, headers = _client(user_id=7, role='user', username='ana')
    seen = []
    real = pipeline.job_set

    def spy(jid, **kw):
        seen.append(dict(kw))
        return real(jid, **kw)
    monkeypatch.setattr(pipeline, 'job_set', spy)
    job = _analyze(client, headers, env, auto_render='1', reframe='fit', subtitles='false', speaker='true',
                   subtitle_size='l')
    assert job['error'] is None and job['done'] and job['percent'] == 100
    assert [s['label'] for s in job['stages']] == [
        'Reading video', 'Detecting cuts', 'Rating frames', 'Transcribing dialogue', 'Reading the sound',
        'Finding story beats', 'Building candidates', 'Rating frames inside the moments', 'Rendering shorts', 'Done']
    assert [s['percent'] for s in job['stages']] == sorted(s['percent'] for s in job['stages'])

    # One bar for both halves: it never runs backwards, and the job is only
    # ever marked finished once -- at the very end, with the batch.
    pcts = [k['percent'] for k in seen if k.get('percent') is not None]
    assert pcts == sorted(pcts) and pcts[-1] == 100 and shorts.AUTO_SPLIT in pcts
    assert [bool(k.get('result')) for k in seen if k.get('done')] == [True], 'no "done" from the analysis half'

    res = job['result']
    batch = res['batch']
    a = client.get(f"/api/shorts/analysis/{res['analysis_id']}").get_json()
    assert a['ok'], 'the moments are still there to review, re-time and render again'
    assert len(batch['shorts']) == len(a['candidates']) >= 2 and batch['status'] == 'complete'
    assert [s['title'] for s in batch['shorts']] == [c['title'] for c in a['candidates']]
    assert [(s['start'], s['end']) for s in batch['shorts']] == \
        [(round(c['start'] * 25) / 25, round(c['end'] * 25) / 25) for c in a['candidates']]
    assert batch['options'] == {'reframe': 'fit', 'subtitles': False, 'subtitle_size': 'l', 'face_detector': None,
                                'speaker': False, 'ending': 'none', 'format': 'mp4_high',
                                'format_label': 'MP4 (H.264 High Profile)', 'loudness': -14.0}, \
        'the Output settings sent with the request'
    assert all(s['layouts']['fit'] >= 1 and s['layouts']['crop'] == 0 for s in batch['shorts'])
    assert batch['username'] == 'ana'
    for s in batch['shorts']:
        assert os.path.getsize(os.path.join(shorts.SHORTS_DIR, batch['batch_id'], s['file'])) > 0
    # And it is saved like any other batch.
    assert [b['batch_id'] for b in client.get('/api/shorts/batches').get_json()['items']] == [batch['batch_id']]
    # A second render from the same analysis, the way the review list does it.
    again = _render(client, headers, res['analysis_id'], [{'start': 1.0, 'end': 6.0, 'title': 'Again'}],
                    reframe='fit', subtitles=False)
    assert again['error'] is None and len(again['result']['batch']['shorts']) == 1


def test_generate_without_preview_stops_at_a_failed_analysis(env, monkeypatch):
    Services(monkeypatch)
    monkeypatch.setattr(pipeline, '_check_service',
                        lambda name, url, path='/', timeout=3: {'status': 'down', 'error': 'connection refused'}
                        if name == 'whisper' else {'status': 'up'})
    render = mock.Mock(side_effect=AssertionError('must not render after a failed analysis'))
    monkeypatch.setattr(sc, 'render_short', render)
    client, headers = _client()
    job = _analyze(client, headers, env, auto_render='1')
    assert job['done'] and 'faster-whisper' in job['error'] and os.listdir(shorts.SHORTS_DIR) == []


def test_generate_without_preview_carries_the_analysis_warnings_onto_the_batch(env, monkeypatch):
    """Nobody reviews this render, so what the analysis had to say about its
    own shortcomings has to arrive with the shorts."""
    svc = Services(monkeypatch, words=[], segs=[])
    svc.vision_reply = lambda n: {'response': json.dumps({'score': 5, 'desc': 'a fight'})}
    client, headers = _client()
    job = _analyze(client, headers, env, auto_render='1', reframe='fit')
    assert job['error'] is None
    assert any('No dialogue was transcribed' in w for w in job['result']['batch']['warnings'])


def test_a_phase_maps_its_progress_and_keeps_done_to_itself(monkeypatch):
    calls = []
    monkeypatch.setattr(pipeline, 'job_set', lambda jid, **kw: calls.append((jid, kw)))
    first = shorts._Phase('j1', 0, 55, last=False)
    first(percent=0, step='Reading video')
    first(percent=46, step='Transcribing dialogue')
    first(percent=100, step='Done', done=True, result={'analysis_id': 'abc'})
    assert calls == [('j1', {'percent': 0, 'step': 'Reading video'}),
                     ('j1', {'percent': 25, 'step': 'Transcribing dialogue'}),
                     ('j1', {'percent': 55})]
    assert first.result == {'analysis_id': 'abc'}
    first(error='Whisper is down')
    assert calls[-1] == ('j1', {'error': 'Whisper is down'}), 'an error ends the job whichever half it is in'

    calls.clear()
    last = shorts._Phase('j1', 55, 100, last=True, extra_result={'analysis_id': 'abc'})
    last(percent=2, step='Preparing')
    last(percent=100, step='Done', done=True, result={'batch': {'batch_id': 'b'}})
    assert calls == [('j1', {'percent': 56, 'step': 'Preparing'}),
                     ('j1', {'percent': 100, 'step': 'Done', 'done': True,
                             'result': {'batch': {'batch_id': 'b'}, 'analysis_id': 'abc'}})]


def test_a_failed_transcription_stops_the_job_with_the_reason_instead_of_guessing(env, monkeypatch):
    """The whisper service was up (it passed the check at the start) and then
    answered the actual request with an error. That used to look exactly
    like an episode with no dialogue: moments were picked on picture alone
    and offered as if they had been checked for story."""
    svc = Services(monkeypatch, words=[], segs=[])
    svc.heard = {'ok': False, 'reason': 'the speech-to-text service at http://localhost:8000 answered with an '
                                        'error (HTTP 500 for model "large-v2"): CUDA out of memory'}
    client, headers = _client()
    job = _analyze(client, headers, env)
    assert job['done'] and job['error'].startswith('Could not transcribe the dialogue: ')
    assert 'HTTP 500' in job['error'] and 'CUDA out of memory' in job['error']
    assert shorts.ANALYSES == {} and svc.story_prompts == []
    # The one-button path stops there too: nothing is rendered from a guess.
    job = _analyze(client, headers, env, auto_render='1')
    assert 'Could not transcribe' in job['error'] and os.listdir(shorts.SHORTS_DIR) == []


def test_a_source_with_nothing_to_hear_still_falls_back_and_says_what_was_found(env, monkeypatch):
    svc = Services(monkeypatch, words=[], segs=[])
    svc.heard = {'ok': True, 'reason': 'the file has no audio track'}
    svc.vision_reply = lambda n: {'response': json.dumps({'score': 5, 'desc': 'a fight'})}
    client, headers = _client()
    a = _analysis(client, _analyze(client, headers, env))
    assert a['candidates'] and all(c['source'] == 'visual' for c in a['candidates'])
    assert any(w.startswith('No dialogue was transcribed (the file has no audio track)') for w in a['warnings'])


def test_dialogue_found_on_another_track_is_reported_and_the_shorts_use_that_audio(env, monkeypatch):
    svc = Services(monkeypatch)
    svc.heard = {'ok': True, 'reason': None, 'audio': 'tracks 3+4', 'take': [[2, 0], [3, 0]]}
    client, headers = _client()
    job = _analyze(client, headers, env)
    a = _analysis(client, job)
    assert "The dialogue was read from tracks 3+4 of this file's audio; the shorts use the same audio." in a['warnings']
    assert shorts.ANALYSES[job['result']['analysis_id']]['info']['audio_take'] == [[2, 0], [3, 0]]
    # A surround mix: said, but the shorts keep the whole mix.
    svc.heard = {'ok': True, 'reason': None, 'audio': 'channel 3'}
    job = _analyze(client, headers, env)
    a = _analysis(client, job)
    assert "The dialogue was read from channel 3 of this file's audio." in a['warnings']
    assert 'audio_take' not in shorts.ANALYSES[job['result']['analysis_id']]['info']
    # An ordinary file: nothing to say.
    svc.heard = {'ok': True, 'reason': None}
    a = _analysis(client, _analyze(client, headers, env))
    assert not any('dialogue was read from' in w for w in a['warnings'])


def test_a_source_too_short_for_the_requested_length_is_refused(env, monkeypatch):
    Services(monkeypatch)
    client, headers = _client()
    job = _analyze(client, headers, env, min_dur=30, max_dur=60)
    assert job['done'] and 'only 24s long' in job['error']


def test_an_analysis_is_private_to_its_owner_and_expires(env, monkeypatch):
    Services(monkeypatch)
    client, headers = _client(user_id=7, role='user', username='ana')
    job = _analyze(client, headers, env)
    aid = job['result']['analysis_id']
    other, oh = _client(user_id=8, role='user', username='ben')
    assert other.get(f'/api/shorts/analysis/{aid}').status_code == 403
    assert other.get(f"/api/shorts/progress/{job['id']}").status_code == 403
    assert other.post('/api/shorts/render', json={'analysis_id': aid, 'items': [{'start': 1, 'end': 9}]},
                      headers=oh).status_code == 403
    assert other.post('/api/shorts/clip', json={'analysis_id': aid, 'start': 1, 'end': 9}, headers=oh).status_code == 403
    admin, _ = _client(user_id=1, role='admin')
    assert admin.get(f'/api/shorts/analysis/{aid}').status_code == 200
    assert client.get('/api/shorts/analysis/ffffffffffffffff').status_code == 404
    monkeypatch.setattr(shorts, 'SHORTS_ANALYSIS_DAYS', -1)
    r = client.get(f'/api/shorts/analysis/{aid}')
    assert r.status_code == 404 and 'expired' in r.get_json()['error']


def test_progress_reports_the_right_stage_list_and_cancel_of_a_finished_job(env, monkeypatch):
    Services(monkeypatch)
    client, headers = _client()
    job = _analyze(client, headers, env)
    assert [s['label'] for s in job['stages']][:2] == ['Reading video', 'Detecting cuts']
    assert job['percent'] == 100 and 'elapsed' in job
    assert client.get('/api/shorts/progress/nope').status_code == 404
    assert client.post(f"/api/shorts/cancel/{job['id']}", headers=headers).status_code == 409
    assert client.post('/api/shorts/cancel/nope', headers=headers).status_code == 404


def test_preview_clip_is_a_small_proxy_of_the_requested_range(env, monkeypatch):
    Services(monkeypatch)
    client, headers = _client()
    aid = _analyze(client, headers, env)['result']['analysis_id']
    r = client.post('/api/shorts/clip', json={'analysis_id': aid, 'start': 3.0, 'end': 9.0}, headers=headers)
    d = r.get_json()
    assert r.status_code == 200 and d['ok'] and d['url'].startswith('/uploads/shclip_')
    path = os.path.join(main.app.config['UPLOAD_FOLDER'], os.path.basename(d['url']))
    fr = _frames(path)
    assert abs(len(fr) - 150) <= 2 and fr[0].shape[:2] == (180, 320), 'source framing, not reframed'
    assert _colour(fr[5], 160, 90) == 'blue' and _colour(fr[-5], 160, 90) == 'green'
    assert client.post('/api/shorts/clip', json={'analysis_id': aid, 'start': 9, 'end': 9.1},
                       headers=headers).status_code == 400
    os.remove(env['path'])
    r = client.post('/api/shorts/clip', json={'analysis_id': aid, 'start': 12.0, 'end': 15.0}, headers=headers)
    assert r.status_code == 410 and 're-analyse' in r.get_json()['error']


# --------------------------------------------------------------------------
# The tab: render, batches, delivery
# --------------------------------------------------------------------------

def test_render_validates_every_item(env, monkeypatch):
    Services(monkeypatch)
    client, headers = _client()
    aid = _analyze(client, headers, env)['result']['analysis_id']
    started = []
    monkeypatch.setattr(shorts, '_spawn', lambda fn, *a, **k: started.append(a[1]))

    def post(items, **extra):
        return client.post('/api/shorts/render', json=dict({'analysis_id': aid, 'items': items}, **extra), headers=headers)

    assert post([]).status_code == 400
    assert post('nope').status_code == 400
    assert post([{'start': 1, 'end': 9}] * 101).status_code == 400, 'a hundred is the most one render takes'
    assert 'numbers' in post([{'start': 'a', 'end': 9, 'title': 'T'}]).get_json()['error']
    r = post([{'start': 5, 'end': 6, 'title': 'Blip'}])
    assert r.status_code == 400 and '"Blip"' in r.get_json()['error'] and 'at least 3 seconds' in r.get_json()['error']
    assert post([{'start': 30, 'end': 40, 'title': 'Past the end'}]).status_code == 400
    monkeypatch.setattr(shorts, 'SHORTS_MAX_CLIP', 10.0)
    assert 'limit is 10s' in post([{'start': 0, 'end': 20, 'title': 'Long'}]).get_json()['error']
    assert client.post('/api/shorts/render', json={'analysis_id': 'gone', 'items': [{'start': 1, 'end': 9}]},
                       headers=headers).status_code == 404
    assert started == []

    # A valid request: range clamped to the video, blank title given a name, unknown options defaulted.
    r = post([{'start': -5, 'end': 8, 'title': '  '}, {'start': 16, 'end': 99, 'title': 'x' * 200}],
             reframe='sideways', subtitle_size='huge', subtitles=False)
    post([{'start': 0, 'end': 8, 'title': 'T'}], reframe='split')
    assert started.pop()['reframe'] == 'split'
    assert r.status_code == 200 and len(started) == 1
    p = started[0]
    assert [(i['start'], i['title'][:8]) for i in p['items']] == [(0.0, 'Short 1'), (16.0, 'xxxxxxxx')]
    assert abs(p['items'][1]['end'] - 24.0) < 0.1 and len(p['items'][1]['title']) == 80
    assert (p['reframe'], p['subtitle_size'], p['subtitles']) == ('auto', 'm', False)
    assert p['speaker'] is False, '"Follow the speaker" is off unless asked for'

    # "Follow the speaker": a per-render choice; the server setting is only what an unspecified request gets.
    one = [{'start': 0, 'end': 8, 'title': 'T'}]
    post(one, speaker=True)
    post(one, speaker=True, reframe='fit')
    post(one, speaker='yes please')
    monkeypatch.setattr(shorts, 'SHORTS_SPEAKER_CROP', True)
    post(one)
    post(one, speaker=False)
    assert [q['speaker'] for q in started[1:]] == [True, False, False, True, False], \
        'on when asked; never with Fit (nothing is cropped); junk is not a yes; default from the server; off wins'


def _planes(path, k, w=1080, h=1920):
    """Frame `k` as the planes the file holds -- brightness, and how far the
    two colour planes are from grey -- not through an RGB conversion, which
    turns a change of colour into an apparent change of brightness."""
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path), '-vf', f"select='eq(n,{k})'", '-frames:v', '1',
                          '-f', 'rawvideo', '-pix_fmt', 'yuv420p', '-'], capture_output=True, timeout=120).stdout
    size = w * h
    y = np.frombuffer(raw[:size], np.uint8).astype(int).reshape(h, w)
    colour = float(np.abs(np.frombuffer(raw[size:size + size // 2], np.uint8).astype(int) - 128).mean())
    return y, colour


def _sound(path):
    import wave
    wav = str(path) + '.wav'
    subprocess.run(FF + ['-i', str(path), '-vn', '-ac', '1', '-ar', '8000', '-c:a', 'pcm_s16le', wav],
                   check=True, timeout=60)
    w = wave.open(wav)
    x = np.frombuffer(w.readframes(w.getnframes()), dtype='<i2').astype(float) / 32768.0
    w.close()
    os.remove(wav)
    return x


def test_the_cliffhanger_ending_stops_dead_holds_and_cuts_to_black(env, monkeypatch):
    Services(monkeypatch)
    client, headers = _client()
    a = _analysis(client, _analyze(client, headers, env))
    # 2.0-6.0 s: a second of the first shot, then all three of the second. A
    # line of dialogue (5.0-6.6 s) is still being spoken when it ends.
    item = [{'start': 2.0, 'end': 6.0, 'title': 'Moment'}]
    plain = _render(client, headers, a['analysis_id'], item, reframe='fit', subtitles=True)['result']['batch']
    ended = _render(client, headers, a['analysis_id'], item, reframe='fit', subtitles=True,
                    ending='cliffhanger')['result']['batch']
    old = _render(client, headers, a['analysis_id'], item, reframe='fit', subtitles=False,
                  ending='freeze')['result']['batch']
    assert plain['options']['ending'] == 'none' and ended['options']['ending'] == 'cliffhanger'
    assert old['options']['ending'] == 'none', 'an ending this version does not have is no ending'
    assert client.get('/api/shorts/options').get_json()['ending_seconds'] == 2.4
    p, e = plain['shorts'][0], ended['shorts'][0]
    assert abs(p['duration'] - 4.0) < 0.05 and abs(e['duration'] - 6.24) < 0.05
    plain_path = os.path.join(shorts.SHORTS_DIR, plain['batch_id'], p['file'])
    end_path = os.path.join(shorts.SHORTS_DIR, ended['batch_id'], e['file'])
    fr_p, fr = _frames(plain_path), _frames(end_path)
    # The out point is on a cut, so there is nothing after it to make the
    # hold from: it is made from the moment's own last 4 frames (the shot is
    # motionless). 96 of the moment, 50 of hold, 10 of black.
    assert len(fr_p) == 100 and len(fr) == 156

    def caption(frame):                                   # white lettering on a blue picture
        return int((frame.min(axis=2) > 200).sum())

    # Up to the frame it stops on, the short is the short: same picture, same caption.
    assert float(np.abs(fr[95].astype(int) - fr_p[95].astype(int)).mean()) < 2.0
    assert caption(fr_p[95]) > 500 and caption(fr[95]) > 500, 'a line is on screen as it stops'
    # Then, on the very next frame, the look -- and the caption gone with the action.
    live_y, live_c = _planes(end_path, 95)
    hold_y, hold_c = _planes(end_path, 96)
    assert caption(fr[96]) == 0 and all(caption(fr[k]) == 0 for k in range(96, 156, 7))
    assert hold_c < 0.92 * live_c, 'colour pulled back at once, not eased in'
    late_y, late_c = _planes(end_path, 145)
    assert abs(late_c - hold_c) < 1.0, 'and the same look to the end of the hold'
    # A cut to black, not a fade: the last frame of the hold is as bright as
    # the one a second before it, and the next is black.
    assert abs(float(late_y.mean()) - float(_planes(end_path, 120)[0].mean())) < 1.5
    assert float(late_y[860:1060].mean()) > 25 and all(float(fr[k].mean()) < 4 for k in range(146, 156))

    x = _sound(end_path)
    assert abs(len(x) / 8000.0 - 6.24) < 0.15, 'the sound track runs the whole length'

    def level(t0, t1):
        seg = x[int(t0 * 8000):int(t1 * 8000)]
        return 20 * np.log10(max(1e-9, float(np.sqrt(np.mean(seg ** 2)))))
    # The sound is not tied to where the picture stops (3.84 s): it runs to
    # the out point (4.0 s), under the first frames of the hold. This is a
    # steady tone -- no word to find the end of -- so that is where it stops.
    assert level(1.0, 3.7) > -40 and level(3.84, 3.98) > level(1.0, 3.7) - 6, 'heard right up to the out point'
    assert level(4.14, 4.3) < level(1.0, 3.7) - 30, 'and gone just after it'
    assert float(np.abs(x[int(4.4 * 8000):]).max()) == 0.0, 'then nothing at all, through the hold and the black'


def _talking_source(path, words, seconds=8, fps=25, cut_at=None, bed=None):
    """A picture that barely moves, with 'words' under it: bursts of a tone
    at the given (start, end) times, silence (or a quiet bed) between. With
    `cut_at`, the picture changes colour at that frame."""
    n = seconds * fps
    frames = []
    for i in range(n):
        f = np.full((180, 320, 3), 70 if cut_at is None or i < cut_at else 150, np.uint8)
        cv2.rectangle(f, (140, 70), (180, 110), (100 + (i % 3), 160, 200), -1)       # a face-coloured patch, breathing
        frames.append(f)
    rate = 48000
    t = np.arange(seconds * rate) / rate
    x = np.zeros(len(t)) if bed is None else 10 ** (bed / 20.0) * np.sin(2 * np.pi * 90 * t)
    for a, b in words:
        m = (t >= a) & (t < b)
        env = np.minimum(1.0, np.minimum((t[m] - a) / 0.02, (b - t[m]) / 0.05))
        x[m] += env * 0.25 * np.sin(2 * np.pi * 220 * t[m]) * (0.7 + 0.3 * np.sin(2 * np.pi * 6 * t[m]))
    wav = str(path) + '.wav'
    import wave
    w = wave.open(wav, 'wb')
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
    w.writeframes((np.clip(x, -1, 1) * 32767).astype('<i2').tobytes())
    w.close()
    proc = subprocess.Popen(FF + ['-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '320x180', '-r', str(fps), '-i', '-',
                                  '-i', wav, '-c:v', 'libx264', '-crf', '10', '-pix_fmt', 'yuv420p', '-g', '1',
                                  '-c:a', 'aac', '-b:a', '192k', '-shortest', str(path)], stdin=subprocess.PIPE)
    proc.communicate(b''.join(f.tobytes() for f in frames))
    assert proc.returncode == 0
    os.remove(wav)
    return str(path)


def _heard(path, rate=8000):
    """(samples, level(t0, t1) in dB) of a file's sound."""
    x = _sound(path)

    def level(t0, t1):
        seg = x[int(t0 * rate):int(t1 * rate)]
        return 20 * np.log10(max(1e-9, float(np.sqrt(np.mean(seg ** 2)))))
    return x, level


def test_the_cliffhanger_lets_the_last_word_finish_and_gives_up_none_of_the_moment(tmp_path):
    """Reported: the sound was being cut mid-word. Two causes. The hold was
    made from the moment's own last frames, and the sound stopped where the
    picture did -- 0.16 s short of the out point. And the out point itself
    comes from a transcript, whose idea of where a word ends is early."""
    # The last word really runs 3.60-4.45 s. The transcript had it ending at
    # 4.0, so that is where the out point is: 0.45 s short of the truth.
    src = _talking_source(tmp_path / 'talk.mp4', [(0.5, 1.2), (1.5, 2.4), (2.7, 3.3), (3.60, 4.45), (6.5, 7.2)])
    info = sc.probe_source('ffprobe', src)
    fps, start_f, n = info['fps'], 25, 75                 # the short is 1.0-4.0 s of the source
    segs = [{'a': 0, 'b': n - 1, 'layout': 'fit', 'x': None, 'keys': None}]

    # 1. The hold comes from the frames after the out point: the shot carries on and is still.
    after = sc.still_frames_after(src, start_f + n, fps, 50)
    assert after == 4 and sc.cliffhanger_stop(n, fps, None, after) == n and sc.cliffhanger_extra(n, fps, None, after) == 60
    # 2. The sound is measured: the word under way at the out point ends 0.45 s later.
    out, fade, how = sc.measure_audio_out('ffmpeg', src, info, start_f, n, next_speech=5.5)
    assert how == 'pause' and abs(out - 3.45) < 0.06 and fade == sc.AUDIO_FADE['pause'], (out, fade, how)

    new = str(tmp_path / 'new.mp4')
    ok, err = sc.render_short('ffmpeg', src, new, start_f, n, info, segs, work_dir=str(tmp_path), preset='ultrafast',
                              ending=True, ending_after=after, audio_out=out, audio_fade=fade)
    assert ok, err
    old = str(tmp_path / 'old.mp4')                       # as it was: hold from the clip's own end, sound stops with the picture
    ok, err = sc.render_short('ffmpeg', src, old, start_f, n, info, segs, work_dir=str(tmp_path), preset='ultrafast',
                              ending=True, ending_room=4)
    assert ok, err
    fr_new, fr_old = _frames(new), _frames(old)
    assert len(fr_new) == 75 + 50 + 10 and len(fr_old) == 71 + 50 + 10

    def tone(frames, k):                                  # the hold is graded: its balance of blue and red is not the action's
        b, g, r = frames[k][900:1000, 100:400].reshape(-1, 3).mean(axis=0)
        return float(b - r)
    live = tone(fr_new, 10)
    assert abs(tone(fr_new, 74) - live) < 2 and abs(tone(fr_new, 75) - live) > 4, 'the action runs to its last frame'
    assert abs(tone(fr_old, 70) - live) < 2 and abs(tone(fr_old, 71) - live) > 4, '(it used to stop four frames short)'

    # 3. The sound outlasts the picture: the word finishes under the hold, then silence.
    x, level = _heard(new)
    word = level(2.7, 2.95)
    assert level(3.0, 3.35) > word - 4, 'the rest of the word, after the picture has stopped at 3.0 s'
    assert level(3.75, 4.0) < word - 40 and float(np.abs(x[int(4.0 * 8000):]).max()) == 0.0
    assert abs(len(x) / 8000.0 - 135 / 25.0) < 0.15
    # Before: gone by the out point, a third of a second into a word that had 0.45 s to run.
    _, level_old = _heard(old)
    assert level_old(3.1, 3.35) < word - 35

    # The same short, when the next shot starts right at the out point: the
    # hold has to come from the moment's own frames, but the sound still finishes.
    cut = _talking_source(tmp_path / 'cut.mp4', [(0.5, 1.2), (1.5, 2.4), (2.7, 3.3), (3.60, 4.45)], cut_at=100)
    assert sc.still_frames_after(cut, start_f + n, fps, 0) == 0
    room = sc.still_frames(cut, start_f + n - 1, fps, n)
    out, fade, how = sc.measure_audio_out('ffmpeg', cut, info, start_f, n)
    assert room == 4 and how == 'pause' and abs(out - 3.45) < 0.06
    onb = str(tmp_path / 'oncut.mp4')
    ok, err = sc.render_short('ffmpeg', cut, onb, start_f, n, info, segs, work_dir=str(tmp_path), preset='ultrafast',
                              ending=True, ending_room=room, ending_after=0, audio_out=out, audio_fade=fade)
    assert ok, err
    fr = _frames(onb)
    assert len(fr) == 71 + 50 + 10 and max(float(f[900:1000, 100:400].mean()) for f in fr[60:121]) < 110, \
        'none of the next shot (the lighter one) in the hold'
    _, level = _heard(onb)
    assert level(3.0, 3.35) > level(2.7, 2.95) - 4 and level(3.75, 4.0) < level(2.7, 2.95) - 40


def test_where_the_sound_of_a_cliffhanger_stops_is_measured_case_by_case(tmp_path):
    words = [(0.5, 1.2), (1.5, 2.4), (2.7, 3.3), (3.60, 4.45), (4.50, 5.4), (6.5, 7.2)]
    src = _talking_source(tmp_path / 'talk.mp4', words)
    info = sc.probe_source('ffprobe', src)
    start_f = 25

    def out_at(source_t, next_speech=None, s=src):
        n = int(round(source_t * 25)) - start_f
        o, fade, how = sc.measure_audio_out('ffmpeg', s, info, start_f, n,
                                            None if next_speech is None else next_speech - 1.0)
        return round(o + 1.0, 2), fade, how               # back on the source's clock
    # Already in a pause: it stops there.
    assert out_at(3.44) == (3.44, sc.AUDIO_FADE['quiet'], 'quiet')
    # Speech runs straight on (the next word 50 ms after this one): the gap between the two, fast.
    o, fade, how = out_at(4.40, next_speech=4.50)
    assert how in ('dip', 'pause', 'quiet') and 4.43 <= o <= 4.52 and fade <= sc.AUDIO_FADE['quiet'], (o, fade, how)
    # ...also when the out point has strayed into the start of that next word.
    o, fade, how = out_at(4.56, next_speech=4.50)
    assert how == 'dip' and 4.44 <= o <= 4.52 and fade == sc.AUDIO_FADE['dip'], (o, fade, how)
    # A word that would run on past the reach is not chased: the out point stands.
    long_word = _talking_source(tmp_path / 'long.mp4', [(0.5, 1.2), (1.5, 2.4), (3.0, 6.0)])
    assert out_at(4.0, s=long_word) == (4.0, sc.AUDIO_FADE[None], None)
    # Dialogue over a music bed never goes silent; it drops to the music, and that is the pause.
    bed = _talking_source(tmp_path / 'bed.mp4', words[:4] + words[5:], bed=-34.0)       # without the run-on word
    o, fade, how = out_at(4.0, s=bed)
    assert how == 'pause' and abs(o - 4.45) < 0.08, (o, how)
    # Nothing to tell a word from a gap by: a file with no sound at all, and one that is all one level.
    assert sc.measure_audio_out('ffmpeg', src, dict(info, audio_index=None), start_f, 75) == (3.0, 0.12, None)
    flat = _talking_source(tmp_path / 'flat.mp4', [(0.0, 8.0)])
    assert out_at(4.0, s=flat)[2] is None


def test_shorts_can_be_levelled_for_where_they_are_going(env, monkeypatch):
    Services(monkeypatch)
    client, headers = _client()
    opts = client.get('/api/shorts/options').get_json()
    assert opts['loudness'] == -14.0
    assert [(l['value'], l['default']) for l in opts['levels']] == [
        (-12.0, False), (-14.0, True), (-16.0, False), (-18.0, False), (-23.0, False), (-24.0, False)]
    assert 'EBU R128' in opts['levels'][4]['label'] and opts['levels'][1]['label'].startswith('\u221214 LUFS')
    a = _analysis(client, _analyze(client, headers, env))
    item = [{'start': 2.0, 'end': 8.0, 'title': 'Moment'}]

    def loudness(batch):
        path = os.path.join(shorts.SHORTS_DIR, batch['batch_id'], batch['shorts'][0]['file'])
        r = subprocess.run(['ffmpeg', '-hide_banner', '-nostats', '-i', path, '-af', 'ebur128', '-f', 'null', '-'],
                           capture_output=True, text=True, timeout=120)
        return float(re.findall(r'I:\s*(-?[\d.]+) LUFS', r.stderr)[-1])
    usual = _render(client, headers, a['analysis_id'], item, reframe='fit', subtitles=False)['result']['batch']
    air = _render(client, headers, a['analysis_id'], item, reframe='fit', subtitles=False, loudness='-23')['result']['batch']
    assert usual['options']['loudness'] == -14.0 and air['options']['loudness'] == -23.0
    lu, la = loudness(usual), loudness(air)
    assert abs(lu - -14.0) < 1.5 and abs(la - -23.0) < 1.5 and 7.5 < lu - la < 10.5, (lu, la)
    # Anything that is not a level is the usual one, not an error and not a silent file.
    started = []
    monkeypatch.setattr(shorts, '_spawn', lambda fn, *a, **k: started.append(a[1]))
    for asked in ('loud', '', None, 3, -80, 'nan'):
        client.post('/api/shorts/render', headers=headers,
                    json={'analysis_id': a['analysis_id'], 'items': item, 'loudness': asked})
    assert [p['loudness'] for p in started] == [-14.0] * 6
    client.post('/api/shorts/render', headers=headers, json={'analysis_id': a['analysis_id'], 'items': item, 'loudness': -16.04})
    assert started[-1]['loudness'] == -16.0
    # A short rendered again for new captions keeps the level it was made at.
    seen = []
    real = sc.render_short
    monkeypatch.setattr(sc, 'render_short', lambda *x, **k: (seen.append(k['loudness']), real(*x, **k))[1])
    monkeypatch.setattr(shorts, '_spawn', lambda fn, *a, **k: fn(*a, **k))
    air2 = _render(client, headers, a['analysis_id'], item, reframe='fit', subtitles=True, loudness=-23)['result']['batch']
    url = f"/api/shorts/batches/{air2['batch_id']}/captions/{air2['shorts'][0]['index']}"
    job = client.get(f"/api/shorts/progress/{client.post(url, headers=headers, json={'cues': [{'start': 1, 'end': 2, 'text': 'Bago'}]}).get_json()['job_id']}").get_json()
    assert job.get('error') is None and seen == [-23.0, -23.0], (job, seen)


def test_the_cliffhanger_plan_a_short_was_rendered_with_is_kept_for_rendering_it_again(env, monkeypatch):
    """A short between cuts: the hold comes from after its out point and its
    sound stops where it was measured to. Rendered again for new captions,
    it must come out the same length, from the same plan."""
    Services(monkeypatch)
    client, headers = _client()
    a = _analysis(client, _analyze(client, headers, env))
    batch = _render(client, headers, a['analysis_id'], [{'start': 3.4, 'end': 7.4, 'title': 'Mid shot'}],
                    reframe='fit', ending='cliffhanger')['result']['batch']
    s = batch['shorts'][0]
    bdir = os.path.join(shorts.SHORTS_DIR, batch['batch_id'])
    plan = shorts._read_manifest(bdir)['shorts'][0]['plan']
    assert plan['after'] == 4 and plan['audio_out'] == 4.0 and plan['audio_fade'] == 0.12, plan
    assert abs(s['duration'] - 6.4) < 0.01 and len(_frames(os.path.join(bdir, s['file']))) == 100 + 50 + 10
    seen = []
    real = sc.render_short
    monkeypatch.setattr(sc, 'render_short', lambda *x, **k: (seen.append(k), real(*x, **k))[1])
    url = f"/api/shorts/batches/{batch['batch_id']}/captions/{s['index']}"
    job = client.get(f"/api/shorts/progress/{client.post(url, headers=headers, json={'cues': [{'start': 1, 'end': 2, 'text': 'Bago'}]}).get_json()['job_id']}").get_json()
    assert job.get('error') is None, job
    assert (seen[0]['ending_after'], seen[0]['audio_out'], seen[0]['audio_fade']) == (4, 4.0, 0.12)
    assert len(_frames(os.path.join(bdir, s['file']))) == 160 and job['result']['batch']['shorts'][0]['duration'] == s['duration']

    # One made before any of this (its plan says nothing about it) is rendered again the way it was made.
    m = shorts._read_manifest(bdir)
    for k in ('after', 'audio_out', 'audio_fade'):
        m['shorts'][0]['plan'].pop(k)
    m['options'].pop('loudness')
    shorts._write_manifest(bdir, m)
    seen.clear()
    job = client.get(f"/api/shorts/progress/{client.post(url, headers=headers, json={'cues': []}).get_json()['job_id']}").get_json()
    assert job.get('error') is None, job
    assert (seen[0]['ending_after'], seen[0]['audio_out'], seen[0]['loudness']) == (0, None, -14.0)
    assert len(_frames(os.path.join(bdir, s['file']))) == 96 + 50 + 10


def _source(path, frames, fps='25'):
    """Frames given as arrays, encoded nearly losslessly, with a tone."""
    h, w = frames[0].shape[:2]
    proc = subprocess.Popen(FF + ['-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{w}x{h}', '-r', fps, '-i', '-',
                                  '-f', 'lavfi', '-i', 'sine=frequency=300:sample_rate=48000', '-shortest',
                                  '-c:v', 'libx264', '-crf', '6', '-pix_fmt', 'yuv420p', '-c:a', 'aac', str(path)],
                            stdin=subprocess.PIPE)
    proc.communicate(b''.join(f.tobytes() for f in frames))
    assert proc.returncode == 0
    return str(path)


def test_the_cliffhanger_hold_pushes_in_and_keeps_what_was_barely_moving_alive(tmp_path):
    """The pose is locked; what was only just moving at the end -- here, a
    patch brightening by one grey level a frame, standing in for a breath --
    goes on moving, out and back, very slowly. Something really moving in
    those frames gets a single held frame instead."""
    def frame(i, pan=0):
        f = np.full((360, 640, 3), 40, np.uint8)
        for cx in (160 + pan, 480 + pan):                 # two marks 320 apart: a ruler
            cv2.rectangle(f, (cx - 12, 60), (cx + 12, 84), (235, 235, 235), -1)
        cv2.rectangle(f, (290, 150), (350, 210), (100 + max(0, i - 96),) * 3, -1)     # still, then 101, 102, 103
        return f

    def spread(frame_bgr):                                # distance between the two marks, in output pixels
        cols = np.where((frame_bgr[..., 1] > 0.7 * frame_bgr[..., 1].max()).any(axis=0))[0]
        left, right = cols[cols < 540], cols[cols >= 540]
        return float(right.mean() - left.mean())

    def patch(path, k):
        return float(_planes(path, k)[0][930:990, 510:570].mean())

    calm = _source(tmp_path / 'calm.mp4', [frame(i) for i in range(100)])
    info = sc.probe_source('ffprobe', calm)
    segs = [{'a': 0, 'b': 99, 'layout': 'fit', 'x': None, 'keys': None}]
    room = sc.still_frames(calm, 99, info['fps'], 100)
    assert room == 4
    out = str(tmp_path / 'calm_out.mp4')
    ok, err = sc.render_short('ffmpeg', calm, out, 0, 100, info, segs, work_dir=str(tmp_path), crf=14,
                              preset='veryfast', ending=True, ending_room=room)
    assert ok, err
    fr = _frames(out)
    assert len(fr) == 156
    # A slow push in: 3% over the hold, a little at a time, and none before it.
    assert abs(spread(fr[90]) - 540.0) < 1.0 and abs(spread(fr[95]) - 540.0) < 1.0
    assert abs(spread(fr[96]) - 540.0) < 1.5, 'starting from where the action stopped'
    ratio = [spread(fr[k]) / 540.0 for k in (96, 108, 120, 132, 145)]
    assert ratio == sorted(ratio) and abs(ratio[-1] - 1.03) < 0.004, ratio
    assert 0.02 <= ratio[-1] - 1 <= 0.04 and max(b - a for a, b in zip(ratio, ratio[1:])) < 0.012, 'no jump'
    # The breath: forwards to the last frame of the action by the middle of the hold, and back again.
    start, middle, end = patch(out, 96), patch(out, 119), patch(out, 145)
    assert middle > start + 1.5 and middle > end + 1.5 and abs(start - end) < 0.8, (start, middle, end)
    steps = [patch(out, k) for k in range(96, 121, 4)]
    assert all(b >= a - 0.3 for a, b in zip(steps, steps[1:])), ('blended across, not stepped back and forth', steps)

    # The same clip with the camera panning through its last frames: one frame, held.
    busy = _source(tmp_path / 'busy.mp4', [frame(i, pan=3 * max(0, i - 90)) for i in range(100)])
    room = sc.still_frames(busy, 99, info['fps'], 100)
    assert room == 1
    out = str(tmp_path / 'busy_out.mp4')
    ok, err = sc.render_short('ffmpeg', busy, out, 0, 100, info, segs, work_dir=str(tmp_path), crf=14,
                              preset='veryfast', ending=True, ending_room=room)
    assert ok, err
    fr = _frames(out)
    assert len(fr) == 159, '99 of the action, 50 of hold, 10 of black'
    assert abs(patch(out, 99) - patch(out, 123)) < 0.5 and abs(patch(out, 99) - patch(out, 148)) < 0.5, 'locked'
    left = [float(np.where((fr[k][..., 1] > 0.7 * fr[k][..., 1].max()).any(axis=0))[0].min()) for k in (97, 98)]
    assert left[1] - left[0] > 3.0, 'still panning on the last frame of the action'
    assert abs(spread(fr[148]) / spread(fr[99]) - 1.03) < 0.004 and all(float(fr[k].mean()) < 4 for k in range(149, 159))


def test_the_cliffhanger_look_is_a_grade_and_a_light_vignette_not_a_blackout(tmp_path):
    """On a plain mid-grey picture with a patch of colour, where each part
    of the look can be read off on its own."""
    def frame():
        f = np.full((360, 640, 3), 150, np.uint8)
        cv2.rectangle(f, (290, 150), (350, 210), (60, 90, 200), -1)       # warm, saturated, in the centre crop
        cv2.rectangle(f, (270, 110), (370, 140), (225, 225, 225), -1)     # a neutral highlight above it
        cv2.rectangle(f, (270, 220), (370, 250), (45, 45, 45), -1)        # and a neutral shadow below
        return f
    src = _source(tmp_path / 'flat.mp4', [frame() for _ in range(30)])
    info = sc.probe_source('ffprobe', src)
    crop_w, _ = sc.crop_geometry(info['disp_w'], info['disp_h'])
    segs = [{'a': 0, 'b': 29, 'layout': 'crop', 'x': (640 - crop_w) / 2.0, 'keys': None}]
    out = str(tmp_path / 'flat_out.mp4')
    ok, err = sc.render_short('ffmpeg', src, out, 0, 30, info, segs, work_dir=str(tmp_path), crf=14,
                              preset='veryfast', ending=True, ending_room=sc.still_frames(src, 29, info['fps'], 30))
    assert ok, err
    assert len(_frames(out)) == 26 + 50 + 10
    (live_y, _), (hold_y, _) = _planes(out, 25), _planes(out, 30)

    def mean(y, rows, cols):
        return float(y[rows[0]:rows[1], cols[0]:cols[1]].mean())
    # Grey right beside the patch: the middle of the picture.
    grey_live, grey_hold = mean(live_y, (900, 1020), (180, 330)), mean(hold_y, (900, 1020), (180, 330))
    assert abs(grey_live - 150 * 219 / 255.0 - 16) < 4, 'the action is untouched'
    assert abs(grey_hold - grey_live) < 14, 'the middle of the picture keeps its brightness, near enough'
    corners_live = np.mean([mean(live_y, r, c) for r in ((0, 120), (1800, 1920)) for c in ((0, 120), (960, 1080))])
    corners_hold = np.mean([mean(hold_y, r, c) for r in ((0, 120), (1800, 1920)) for c in ((0, 120), (960, 1080))])
    assert abs(corners_live - grey_live) < 2, 'no vignette on the action'
    assert 0.72 < corners_hold / grey_hold < 0.95, ('darker at the corners, and only a little', corners_hold, grey_hold)
    halfway = mean(hold_y, (300, 500), (440, 640))
    assert corners_hold + 3 < halfway < grey_hold + 1, ('falling away gradually from the middle', halfway)

    # The grade's split tone, read as blue against red on the two neutral
    # bands, before and after (the difference of the two, so the decoder's
    # own cast cancels). That colour is pulled back is measured on the
    # rendered short above, where the picture is one colour.
    fr = _frames(out)

    def rgb(k, rows, cols):
        b, g, r = fr[k][rows[0]:rows[1], cols[0]:cols[1]].reshape(-1, 3).mean(axis=0)
        return float(r), float(g), float(b)
    cool = lambda px: px[2] - px[0]
    dark = cool(rgb(30, (1180, 1300), (300, 780))) - cool(rgb(20, (1180, 1300), (300, 780)))
    light = cool(rgb(30, (620, 740), (300, 780))) - cool(rgb(20, (620, 740), (300, 780)))
    assert dark > 4 and light < -4, ('shadows toward blue, highlights toward amber', dark, light)


def test_the_place_badge_on_a_short_does_not_share_a_class_with_the_number_boxes(users_db):
    """It did: the badge that numbers each short was styled as `.sh-num`,
    the class the length and frame-count boxes already had. They became
    26 px pills pinned to the corner of the panel that ignored the mouse:
    the lengths could not be seen or changed."""
    client, _ = _client()
    html = client.get('/').get_data(as_text=True)
    boxes = re.findall(r'<input type=number id=(sh-min|sh-max|sh-frames) class=([\w-]+)', html)
    assert [b[0] for b in boxes] == ['sh-min', 'sh-max', 'sh-frames']
    css = html[html.index('<style'):html.index('</style>')]
    for _, cls in boxes:
        for selector, body in re.findall(r'([^{}]+)\{([^{}]*)\}', css):
            if any(sel.strip() in ('.' + cls, 'input.' + cls) for sel in selector.split(',')):
                assert 'position:absolute' not in body and 'pointer-events:none' not in body, selector
    assert '<span class="sh-place"' in html and '.sh-place{position:absolute' in css


def test_a_shorts_job_is_filed_as_a_shorts_job_and_kept_out_of_the_episodic_list(env, monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, 'JOBS_DB_PATH', str(tmp_path / 'jobs.db'))
    pipeline.jobs_db_init()
    Services(monkeypatch)
    client, headers = _client()
    r = client.post('/api/shorts/analyze', headers=headers, data={
        'shorts_file_network': env['staged'], 'min_dur': 5, 'max_dur': 12, 'count': 5, 'project_id': env['project']})
    assert r.status_code == 200, r.get_json()
    jid = r.get_json()['job_id']
    job = client.get(f'/api/shorts/progress/{jid}').get_json()
    assert job.get('error') is None and pipeline.job_get(jid)['kind'] == 'shorts'
    rid = client.post('/api/shorts/render', headers=headers, json={
        'analysis_id': job['result']['analysis_id'], 'items': [{'start': 2.0, 'end': 6.0, 'title': 'Moment'}],
        'reframe': 'fit', 'subtitles': False}).get_json()['job_id']
    assert pipeline.job_get(rid)['kind'] == 'shorts'

    def listed(url):
        d = client.get(url).get_json()
        return {j['job_id'] for k in ('active', 'queued', 'finished') for j in d[k]}
    assert listed('/api/monitor') == set(), 'not among the episodic plug tab\'s jobs'
    assert listed('/api/monitor?kind=shorts') == {jid, rid} == listed('/api/monitor?kind=all')


# --------------------------------------------------------------------------
# How many moments, and how long
# --------------------------------------------------------------------------

def test_lengths_default_to_60_to_120_seconds_and_a_hundred_moments_can_be_asked_for(env, monkeypatch):
    Services(monkeypatch)
    client, headers = _client()
    started = []
    monkeypatch.setattr(shorts, '_spawn', lambda fn, *a, **k: started.append(a[1]))

    def post(**form):
        data = {'shorts_file_network': env['staged'], 'project_id': env['project']}
        data.update(form)
        assert client.post('/api/shorts/analyze', data=data, headers=headers).status_code == 200
        return started[-1]
    p = post()
    assert (p['min_dur'], p['max_dur'], p['count']) == (60, 120, 8)
    assert [post(count=c)['count'] for c in ('50', '100', '250', 'auto', 'AUTO', 'lots')] == \
        [50, 100, 100, 'auto', 'auto', 8]

    html = client.get('/').get_data(as_text=True)
    assert 'id=sh-min class=sh-num value=60 ' in html and 'id=sh-max class=sh-num value=120 ' in html
    menu = html[html.index('<select id=sh-count'):]
    menu = menu[:menu.index('</select>')]
    assert re.findall(r'<option value=(\w+)( selected)?>', menu) == [
        ('5', ''), ('8', ' selected'), ('12', ''), ('20', ''), ('50', ''), ('100', ''), ('auto', '')]


def test_a_moment_that_cannot_reach_the_minimum_length_is_left_out_and_the_editor_is_told(env, monkeypatch):
    """Reported: a 16.5-second moment in the list at a 60-second minimum."""
    # Six lines back to back (0.5-12.5 s), then one on its own after more than ten seconds of nothing.
    segs = [{'start': 0.5 + 2.0 * k, 'end': 2.4 + 2.0 * k, 'text': f'Linya {k} ng usapan.'} for k in range(6)]
    segs.append({'start': 23.0, 'end': 23.8, 'text': 'Salamat.'})
    svc = Services(monkeypatch, words=[], segs=segs)
    svc.story_reply = lambda ids, payload: {'response': json.dumps({'moments': [
        {'start_id': ids[1], 'end_id': ids[3], 'title': 'The talk', 'score': 6},
        {'start_id': ids[-1], 'end_id': ids[-1], 'title': 'On its own', 'score': 9}]})}
    client, headers = _client()
    a = _analysis(client, _analyze(client, headers, env, min_dur=12, max_dur=18))
    assert [c['title'] for c in a['candidates']] == ['The talk'], 'the stronger one is the one that cannot be made long enough'
    c = a['candidates'][0]
    assert c['duration'] >= 12 and 'short' not in c['flags'] and 'extended' in c['flags']
    assert any('1 moment the story model picked could not be brought up to the 12-second minimum' in w
               and 'was left out' in w for w in a['warnings']), a['warnings']
    # When that is all there is, it is listed as it is, marked, with the reason.
    svc.story_reply = lambda ids, payload: {'response': json.dumps({'moments': [
        {'start_id': ids[-1], 'end_id': ids[-1], 'title': 'On its own', 'score': 9}]})}
    b = _analysis(client, _analyze(client, headers, env, min_dur=12, max_dur=18))
    assert [(c['title'], 'short' in c['flags']) for c in b['candidates']] == [('On its own', True)]
    assert any('None of the moments found could be brought up to the 12-second minimum' in w for w in b['warnings'])


def test_auto_keeps_every_moment_the_story_model_rates_well_and_no_others(env, monkeypatch):
    svc = Services(monkeypatch)
    svc.story_reply = lambda ids, payload: {'response': json.dumps({'moments': [
        {'start_id': ids[0], 'end_id': ids[2], 'title': 'Strong', 'score': 9},
        {'start_id': ids[4], 'end_id': ids[6], 'title': 'Good enough', 'score': 6},
        {'start_id': ids[8], 'end_id': ids[10], 'title': 'Weak', 'score': 4}]})}
    client, headers = _client()
    a = _analysis(client, _analyze(client, headers, env, count='auto'))
    assert sorted(c['title'] for c in a['candidates']) == ['Good enough', 'Strong']
    assert a['options']['count'] == 'auto' and not any('Auto keeps' in w for w in a['warnings'])
    assert 'Find up to 8 moments' in svc.story_prompts[0]['prompt'], 'as many as a stretch can hold, eight at most'
    assert svc.story_prompts[0]['options']['num_predict'] == 450 + 170 * 8, 'and room in the reply to list them'
    # A number is a ceiling and nothing else: the weak one is listed too.
    b = _analysis(client, _analyze(client, headers, env, count='12'))
    assert sorted(c['title'] for c in b['candidates']) == ['Good enough', 'Strong', 'Weak']

    # Nothing reaches the mark: all of them, with the reason, rather than an empty list.
    svc.story_reply = lambda ids, payload: {'response': json.dumps({'moments': [
        {'start_id': ids[0], 'end_id': ids[2], 'title': 'Meh', 'score': 5},
        {'start_id': ids[6], 'end_id': ids[8], 'title': 'Weak', 'score': 3}]})}
    c = _analysis(client, _analyze(client, headers, env, count='auto'))
    assert sorted(x['title'] for x in c['candidates']) == ['Meh', 'Weak']
    assert any('None reached that here, so all 2' in w for w in c['warnings'])


# --------------------------------------------------------------------------
# Delivery format
# --------------------------------------------------------------------------

def test_a_short_can_be_delivered_as_prores_and_that_is_the_file_handed_over(env, monkeypatch):
    Services(monkeypatch)
    client, headers = _client()
    a = _analysis(client, _analyze(client, headers, env))
    item = [{'start': 2.0, 'end': 6.0, 'title': 'Moment'}]
    crfs = []
    real = sc.render_short
    monkeypatch.setattr(sc, 'render_short', lambda *x, **k: (crfs.append(k['crf']), real(*x, **k))[1])
    batch = _render(client, headers, a['analysis_id'], item, reframe='fit', format='prores_hq_2997')['result']['batch']
    assert batch['options']['format'] == 'prores_hq_2997'
    assert batch['options']['format_label'] == 'Apple ProRes 422 HQ — 29.97fps'
    s = batch['shorts'][0]
    assert s['file'].endswith('.mp4') and s['delivery'] == s['file'][:-4] + '.mov' and s['delivery_size'] > s['size'] > 0
    bdir = os.path.join(shorts.SHORTS_DIR, batch['batch_id'])
    mov, mp4 = _probe(os.path.join(bdir, s['delivery'])), _probe(os.path.join(bdir, s['file']))
    assert (mov['video']['codec_name'], mov['video']['width'], mov['video']['height']) == ('prores', 1080, 1920)
    assert mov['video']['r_frame_rate'] == '30000/1001' and mov['video']['pix_fmt'] == 'yuv422p10le'
    assert mov['audio']['codec_name'] == 'pcm_s16le' and abs(float(mov['video']['duration']) - 4.0) < 0.1
    assert mp4['video']['codec_name'] == 'h264', 'the copy that plays in the browser'
    assert crfs == [min(shorts.SHORTS_CRF, shorts.SHORTS_MASTER_CRF)] and crfs[0] < shorts.SHORTS_CRF, \
        'made from a better render than an ordinary MP4 short gets'

    # Both are served; the download-all and Send hand over the .mov.
    assert client.get(s['delivery_url']).status_code == 200 and client.get(s['url']).status_code == 200
    r = client.get(f"/api/shorts/batches/{batch['batch_id']}/zip")
    assert sorted(zipfile.ZipFile(io.BytesIO(r.data)).namelist()) == sorted([s['delivery'], s['srt']])
    r.close()
    sent = []
    monkeypatch.setattr(shorts, 'network_destination_get',
                        lambda i: {'id': 1, 'name': 'Playout', 'delivery_kind': 'video', 'path': r'\\x\y'})
    monkeypatch.setattr(pipeline, 'send_file_to_network_destination',
                        lambda local, name, dest: sent.append((os.path.basename(local), name)))
    r = client.post(f"/api/shorts/batches/{batch['batch_id']}/send", headers=headers,
                    json={'destination_id': 1, 'files': [s['file']], 'filename': 'TADHANA EP101 teaser.mp4',
                          'include_srt': True})
    assert r.status_code == 200, r.get_json()
    assert sent == [(s['delivery'], 'TADHANA_EP101_teaser.mov'), (s['srt'], 'TADHANA_EP101_teaser.srt')], \
        'renamed, but still a .mov: the name cannot change what the file is'

    # MP4 is one file, and anything unknown -- or AVC-Intra, which is a landscape format -- is MP4.
    crfs.clear()
    for asked in ('mp4_high', 'avci100i', 'betamax', None):
        opts = {} if asked is None else {'format': asked}
        b = _render(client, headers, a['analysis_id'], item, reframe='fit', subtitles=False, **opts)['result']['batch']
        assert b['options']['format'] == 'mp4_high', asked
        assert b['shorts'][0]['delivery'] == b['shorts'][0]['file'] and b['shorts'][0]['delivery_url'] == b['shorts'][0]['url']
    assert crfs == [shorts.SHORTS_CRF] * 4


def test_a_delivery_file_that_cannot_be_made_fails_that_short_and_leaves_nothing_behind(env, monkeypatch):
    Services(monkeypatch)
    client, headers = _client()
    a = _analysis(client, _analyze(client, headers, env))
    monkeypatch.setattr(pipeline, 'build_export_cmd',
                        lambda src, dst, fmt: ['ffmpeg', '-v', 'error', '-i', src, '-c:v', 'no_such_codec', dst])
    job = _render(client, headers, a['analysis_id'], [{'start': 2.0, 'end': 6.0, 'title': 'Moment'}],
                  reframe='fit', format='prores_hq_2398')
    assert 'could not be made' in job['error'] and '23.976' in job['error'], job
    assert os.listdir(shorts.SHORTS_DIR) == [], 'no batch with an MP4 passed off as the delivery'


# --------------------------------------------------------------------------
# Captions: corrected before rendering
# --------------------------------------------------------------------------

def _caption_px(frame):
    return int((frame.min(axis=2) > 200).sum())


def test_captions_can_be_read_and_corrected_before_rendering(env, monkeypatch):
    Services(monkeypatch)
    client, headers = _client()
    a = _analysis(client, _analyze(client, headers, env))
    aid = a['analysis_id']

    def captions(**body):
        return client.post('/api/shorts/captions', headers=headers, json=dict({'analysis_id': aid}, **body))
    # Clip 2.0-6.0 s of the episode. Lines are spoken at 1.0-2.5, 3.0-4.5 and 5.0-6.5.
    r = captions(start=2.0, end=6.0).get_json()
    assert r['ok'] and not r['edited'] and r['spoken'] and (r['start'], r['end']) == (2.0, 6.0)
    auto = r['cues']
    assert [c['text'] for c in auto] == ['ng usapan.', 'Linya 1 ng usapan.', 'Linya 2 ng']
    assert all(0 <= c['start'] < c['end'] <= 4.0 for c in auto)
    # A smaller caption size holds more words to a line; the answer is what that size would burn in.
    assert captions(start=2.0, end=6.0, subtitle_size='huge').get_json()['cues'] == auto
    assert captions(start=2.0, end=3.0).status_code == 400 and captions(start='x', end=6).status_code == 400
    assert captions(analysis_id='gone', start=2, end=6).status_code == 404

    # Corrected: a spelling fixed, a line dropped (emptied), one added where nothing is said.
    mine = [dict(auto[1], text='Linya UNO ng usapan!'), dict(auto[0], text='   '),
            {'start': 0.62, 'end': 0.98, 'text': '(katahimikan)'}, auto[2]]
    r = captions(start=2.0, end=6.0, edited={'start': 2.0, 'end': 6.0, 'cues': mine}).get_json()
    assert r['ok'] and r['edited']
    assert [c['text'] for c in r['cues']] == ['(katahimikan)', 'Linya UNO ng usapan!', 'Linya 2 ng'], 'in time order'
    edited = {'start': 2.0, 'end': 6.0, 'cues': r['cues']}
    # What cannot be used is said, in the editor's terms.
    bad = captions(start=2.0, end=6.0, edited={'start': 2.0, 'end': 6.0, 'cues': [{'start': 3, 'end': 1, 'text': 'Backwards'}]})
    assert bad.status_code == 400 and 'must end after it starts' in bad.get_json()['error']
    assert captions(start=2.0, end=6.0, edited={'start': 2.0, 'end': 6.0, 'cues': 'nope'}).status_code == 400
    assert captions(start=2.0, end=6.0, edited={'start': 2.0, 'end': 6.0,
                                               'cues': [{'start': 0, 'end': 1, 'text': 'x' * 201}]}).status_code == 400

    # The out point is then pulled in and the in point moved earlier: the
    # corrections that are still inside stay, moved; the new second at the
    # front gets the automatic captions.
    r = captions(start=1.0, end=5.0, edited=edited).get_json()
    assert [c['text'] for c in r['cues']] == ['Linya 0 ng', '(katahimikan)', 'Linya UNO ng usapan!']
    assert abs(r['cues'][1]['start'] - 1.62) < 1e-6 and r['cues'][2]['end'] <= 4.0

    # Rendered with them: burned in, in the .srt, and marked as edited.
    plain = _render(client, headers, aid, [{'start': 2.0, 'end': 6.0, 'title': 'Plain'}], reframe='fit')['result']['batch']
    batch = _render(client, headers, aid, [{'start': 2.0, 'end': 6.0, 'title': 'Fixed', 'captions': edited}],
                    reframe='fit')['result']['batch']
    s, p = batch['shorts'][0], plain['shorts'][0]
    assert s['captions'] and s['captions_edited'] and s['caption_lines'] == 3
    assert p['captions'] and not p['captions_edited'] and p['caption_lines'] == 3
    srt = client.get(s['srt_url']).get_data(as_text=True)
    assert 'Linya UNO ng usapan!' in srt and '(katahimikan)' in srt and 'Linya 1 ng' not in srt
    assert '00:00:00,620 --> 00:00:00,980' in srt
    fr = _frames(os.path.join(shorts.SHORTS_DIR, batch['batch_id'], s['file']))
    fr_p = _frames(os.path.join(shorts.SHORTS_DIR, plain['batch_id'], p['file']))
    assert _caption_px(fr[20]) > 300 and _caption_px(fr_p[20]) == 0, 'the added line is on screen at 0.8 s'
    assert _caption_px(fr[5]) == 0 and _caption_px(fr_p[5]) > 100, 'and the dropped one is not, at 0.2 s'

    # A render request with captions that cannot be used is refused before anything starts.
    r = client.post('/api/shorts/render', headers=headers, json={'analysis_id': aid, 'items': [
        {'start': 2.0, 'end': 6.0, 'title': 'Bad', 'captions': {'start': 2.0, 'end': 6.0, 'cues': [{'start': 'a', 'end': 1, 'text': 'x'}]}}]})
    assert r.status_code == 400 and '"Bad"' in r.get_json()['error']


# --------------------------------------------------------------------------
# Captions: corrected on a saved short
# --------------------------------------------------------------------------

def _batch(client, headers, env, monkeypatch, **opts):
    Services(monkeypatch)
    a = _analysis(client, _analyze(client, headers, env))
    return _render(client, headers, a['analysis_id'], [{'start': 2.0, 'end': 6.0, 'title': 'Moment'}],
                   reframe='fit', **opts)['result']['batch']


def test_burned_in_captions_of_a_saved_short_are_changed_by_rendering_it_again(env, monkeypatch):
    client, headers = _client(user_id=7, role='user', username='ana')
    batch = _batch(client, headers, env, monkeypatch, ending='cliffhanger')
    bid, s = batch['batch_id'], batch['shorts'][0]
    path = os.path.join(shorts.SHORTS_DIR, bid, s['file'])
    url = f'/api/shorts/batches/{bid}/captions/{s["index"]}'
    shorts.ANALYSES.clear()                     # long after the analysis has gone
    shutil.rmtree(shorts.SHORTS_ANALYSES_DIR)

    r = client.get(url).get_json()
    assert r['ok'] and r['how'] == 'render' and r['burned'] and r['can_change'] and r['why'] is None
    assert [c['text'] for c in r['cues']] == ['ng usapan.', 'Linya 1 ng usapan.', 'Linya 2 ng']
    assert abs(r['duration'] - 4.0) < 1e-6, 'the moment itself: no captions over the cliffhanger hold'
    before = _frames(path)
    assert _caption_px(before[20]) == 0 and _caption_px(before[40]) > 100 and len(before) == 156

    cues = [{'start': 0.62, 'end': 0.98, 'text': 'Sandali lang!'}] + [dict(c, text=c['text'].upper()) for c in r['cues'][1:]]
    p = client.post(url, headers=headers, json={'cues': cues})
    assert p.status_code == 200 and p.get_json()['job_id'], p.get_json()
    jid = p.get_json()['job_id']
    job = client.get(f'/api/shorts/progress/{jid}').get_json()
    assert job.get('error') is None and job['done'], job
    assert [st['label'] for st in job['stages']][1] == 'Rendering with the new captions'
    assert pipeline.job_get(jid)['kind'] == 'shorts'
    now = job['result']['batch']['shorts'][0]
    assert now['file'] == s['file'] and now['srt'] == s['srt'], 'the same names'
    assert now['captions_edited'] and now['caption_lines'] == 3 and '?v=' in now['url'] and '?v=' in now['srt_url']
    assert now['url'] != s['url'], 'a new address, so the browser does not play the copy it has'
    assert client.get(now['url']).status_code == 200 and client.get(shorts_dl(now['url'])).status_code == 200
    after = _frames(path)
    assert len(after) == 156, 'same length, cliffhanger and all'
    assert _caption_px(after[20]) > 300 and _caption_px(after[5]) == 0
    assert float(np.abs(after[90].astype(int) - before[90].astype(int)).mean()) < 30, 'the same picture underneath'
    srt = client.get(now['srt_url']).get_data(as_text=True)
    assert 'Sandali lang!' in srt and 'LINYA 1 NG USAPAN.' in srt and 'usapan.\n\n2' not in srt
    assert [c['text'] for c in client.get(url).get_json()['cues']] == ['Sandali lang!', 'LINYA 1 NG USAPAN.', 'LINYA 2 NG']
    assert sorted(n for n in os.listdir(os.path.join(shorts.SHORTS_DIR, bid)) if n.startswith('.new_')) == []

    # Every line taken out: rendered clean, the .srt removed -- and lines can still be put back.
    job = client.get(f"/api/shorts/progress/{client.post(url, headers=headers, json={'cues': []}).get_json()['job_id']}").get_json()
    bare = job['result']['batch']['shorts'][0]
    assert bare['srt'] is None and bare['captions'] is False and bare['caption_lines'] == 0
    assert max(_caption_px(f) for f in _frames(path)[:96:8]) == 0
    assert client.get(url).get_json()['how'] == 'render'

    # Someone else on the team can read them, not change them; an admin can.
    ben, bh = _client(user_id=8, role='user', username='ben')
    assert ben.get(url).get_json()['can_change'] is False
    assert ben.post(url, headers=bh, json={'cues': cues}).status_code == 404
    admin, ah = _client()
    assert admin.post(url, headers=ah, json={'cues': cues}).status_code == 200
    # What cannot be used is refused before any job starts.
    assert client.post(url, headers=headers, json={'cues': [{'start': 2, 'end': 1, 'text': 'x'}]}).status_code == 400
    assert client.get(f'/api/shorts/batches/{bid}/captions/99').status_code == 404


def shorts_dl(url):
    return url + ('&' if '?' in url else '?') + 'download=1'


def test_a_failed_or_blocked_re_render_leaves_the_short_as_it_was(env, monkeypatch):
    client, headers = _client()
    batch = _batch(client, headers, env, monkeypatch, format='prores_hq_2997')
    bid, s = batch['batch_id'], batch['shorts'][0]
    bdir = os.path.join(shorts.SHORTS_DIR, bid)
    url = f'/api/shorts/batches/{bid}/captions/{s["index"]}'
    cues = [{'start': 0.5, 'end': 1.5, 'text': 'Bago'}]

    def sizes():
        return {n: os.path.getsize(os.path.join(bdir, n)) for n in sorted(os.listdir(bdir))}
    was = sizes()

    def run():
        return client.get(f"/api/shorts/progress/{client.post(url, headers=headers, json={'cues': cues}).get_json()['job_id']}").get_json()
    # The encode fails.
    monkeypatch.setattr(sc, 'render_short', lambda *a, **k: (False, 'boom'))
    job = run()
    assert 'could not be rendered with the new captions: boom' in job['error'] and sizes() == was
    # The file is open in a player (Windows refuses to replace it).
    monkeypatch.setattr(sc, 'render_short', lambda ff, src, out, *a, **k: (open(out, 'wb').write(b'x' * 10), (True, None))[1])
    monkeypatch.setattr(pipeline, 'build_export_cmd', lambda src, dst, fmt: ['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
                                                                             'color=d=0.2:s=64x64', '-y', dst])
    real_replace = os.replace
    monkeypatch.setattr(shorts.os, 'replace', lambda a, b: (_ for _ in ()).throw(PermissionError('in use'))
                        if '.new_' in os.path.basename(a) else real_replace(a, b))
    job = run()
    assert 'in use' in job['error'] and sizes() == was, 'nothing swapped, nothing left over'
    assert client.get(url).get_json()['cues'][0]['text'] == 'ng usapan.', 'and its captions are still the old ones'
    # Two saves at once for the same short: the second is told, not queued.
    shorts._RECAPTIONING.add((bid, s['index']))
    assert client.post(url, headers=headers, json={'cues': cues}).status_code == 409
    shorts._RECAPTIONING.clear()
    # Caption changes have their own allowance: working through a batch is
    # one small job per short, and must not use up -- or be stopped by -- the
    # few analyses and renders a user may start in five minutes.
    monkeypatch.setattr(shorts, '_job_submit_limiter', core._RateLimiter(0, 300))
    monkeypatch.setattr(shorts, '_start_job', lambda *a, **k: (shorts._RECAPTIONING.clear(), 'j')[1])
    assert shorts._recaption_limiter.limit >= 60
    assert client.post(url, headers=headers, json={'cues': cues}).status_code == 200
    monkeypatch.setattr(shorts, '_recaption_limiter', core._RateLimiter(0, 300))
    assert client.post(url, headers=headers, json={'cues': cues}).status_code == 429


def test_a_prores_short_re_rendered_for_captions_replaces_both_of_its_files(env, monkeypatch):
    client, headers = _client()
    batch = _batch(client, headers, env, monkeypatch, format='prores_hq_2997')
    bid, s = batch['batch_id'], batch['shorts'][0]
    bdir = os.path.join(shorts.SHORTS_DIR, bid)
    url = f'/api/shorts/batches/{bid}/captions/{s["index"]}'
    stamp = {n: os.path.getmtime(os.path.join(bdir, n)) for n in (s['file'], s['delivery'])}
    time.sleep(1.1)
    job = client.get(f"/api/shorts/progress/{client.post(url, headers=headers, json={'cues': [{'start': 0.62, 'end': 0.98, 'text': 'Bago'}]}).get_json()['job_id']}").get_json()
    assert job.get('error') is None, job
    now = job['result']['batch']['shorts'][0]
    assert now['delivery'] == s['delivery'] and now['delivery_size'] > 0
    assert all(os.path.getmtime(os.path.join(bdir, n)) > stamp[n] for n in stamp), 'the .mov as well as the .mp4'
    assert _probe(os.path.join(bdir, s['delivery']))['video']['codec_name'] == 'prores'
    cap = cv2.VideoCapture(os.path.join(bdir, s['delivery']))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 24)                 # 0.8 s at 29.97
    ok, frame = cap.read()
    cap.release()
    assert ok and _caption_px(frame) > 300, 'the new caption is in the delivery file'


def test_captions_that_are_not_burned_in_are_saved_straight_to_the_srt(env, monkeypatch):
    client, headers = _client()
    batch = _batch(client, headers, env, monkeypatch, subtitles=False)
    bid, s = batch['batch_id'], batch['shorts'][0]
    url = f'/api/shorts/batches/{bid}/captions/{s["index"]}'
    path = os.path.join(shorts.SHORTS_DIR, bid, s['file'])
    stamp = os.path.getmtime(path)
    r = client.get(url).get_json()
    assert r['how'] == 'srt' and not r['burned'] and len(r['cues']) == 3
    started = []
    monkeypatch.setattr(shorts, '_start_job', lambda *a, **k: started.append(a) or 'j')
    p = client.post(url, headers=headers, json={'cues': [dict(r['cues'][1], text='Itinama na.')]}).get_json()
    assert p['ok'] and 'job_id' not in p and p['srt_only'] is False and started == []
    now = p['batch']['shorts'][0]
    assert now['captions_edited'] and now['caption_lines'] == 1 and os.path.getmtime(path) == stamp
    srt = client.get(now['srt_url']).get_data(as_text=True)
    assert srt.count('-->') == 1 and 'Itinama na.' in srt
    # All of them removed: no .srt at all, and it is not offered.
    p = client.post(url, headers=headers, json={'cues': []}).get_json()
    assert p['batch']['shorts'][0]['srt'] is None and p['batch']['shorts'][0]['srt_url'] is None
    assert not [n for n in os.listdir(os.path.join(shorts.SHORTS_DIR, bid)) if n.endswith('.srt')]
    # And put back.
    p = client.post(url, headers=headers, json={'cues': [{'start': 1, 'end': 2, 'text': 'Balik'}]}).get_json()
    assert 'Balik' in client.get(p['batch']['shorts'][0]['srt_url']).get_data(as_text=True)


def test_burned_in_captions_cannot_be_redone_once_the_episode_is_gone_and_it_says_so(env, monkeypatch):
    client, headers = _client()
    batch = _batch(client, headers, env, monkeypatch)
    bid, s = batch['batch_id'], batch['shorts'][0]
    bdir = os.path.join(shorts.SHORTS_DIR, bid)
    url = f'/api/shorts/batches/{bid}/captions/{s["index"]}'
    path = os.path.join(bdir, s['file'])
    stamp = os.path.getmtime(path)
    os.remove(env['path'])                                # the staged episode has been swept
    r = client.get(url).get_json()
    assert r['how'] == 'frozen' and r['burned'] and 'no longer on the server' in r['why'] and len(r['cues']) == 3
    p = client.post(url, headers=headers, json={'cues': [dict(r['cues'][0], text='Sa srt lang')]}).get_json()
    assert p['ok'] and p['srt_only'] is True and 'job_id' not in p and os.path.getmtime(path) == stamp
    assert 'Sa srt lang' in client.get(p['batch']['shorts'][0]['srt_url']).get_data(as_text=True)

    # A batch saved before any of this was kept: its captions are read from
    # the .srt, and what is burned in cannot be redone.
    m = shorts._read_manifest(bdir)
    m.pop('source')
    for short in m['shorts']:
        for k in ('cues', 'plan', 'captions_edited', 'rev'):
            short.pop(k, None)
    shorts._write_manifest(bdir, m)
    r = client.get(url).get_json()
    assert r['how'] == 'frozen' and 'before captions could be changed' in r['why']
    assert [c['text'] for c in r['cues']] == ['Sa srt lang'] and r['cues'][0]['end'] > r['cues'][0]['start']
    old = client.get(f'/api/shorts/batches?project_id={env["project"]}').get_json()['items'][0]['shorts'][0]
    assert old['caption_lines'] is None and old['delivery'] == old['file'] and '?v=' not in old['url']


def test_what_a_saved_batch_keeps_for_later_stays_on_the_server(env, monkeypatch):
    client, headers = _client()
    batch = _batch(client, headers, env, monkeypatch)
    m = shorts._read_manifest(os.path.join(shorts.SHORTS_DIR, batch['batch_id']))
    assert m['source']['path'] == env['path'] and m['source']['burn'] is True and m['shorts'][0]['plan']['segs']
    text = json.dumps(batch) + client.get(f'/api/shorts/batches?project_id={env["project"]}').get_data(as_text=True)
    assert env['staged'] not in text and '"plan"' not in text and '"source"' not in text and '"cues"' not in text


# --------------------------------------------------------------------------
# Framing: checked in 9:16 and corrected shot by shot before rendering
# --------------------------------------------------------------------------

def test_a_moments_shots_are_shown_with_their_framing_and_can_be_corrected(env, monkeypatch):
    Services(monkeypatch)
    client, headers = _client()
    a = _analysis(client, _analyze(client, headers, env))
    aid = a['analysis_id']

    def post(url, **body):
        return client.post(url, headers=headers, json=dict({'analysis_id': aid}, **body))
    # 2.0-8.0 s of the episode: the end of one coloured shot and two whole ones (cuts at 3 and 6).
    d = post('/api/shorts/framing', start=2.0, end=8.0).get_json()
    assert d['ok'] and (d['disp_w'], d['disp_h'], d['crop_w'], d['max_x']) == (320, 180, 102, 218)
    assert [(sh['start'], sh['end']) for sh in d['shots']] == [(2.0, 3.0), (3.0, 6.0), (6.0, 8.0)]
    assert all(sh['auto'] == 'crop' and sh['x'] == 109.0 for sh in d['shots']), 'no faces: the middle'
    assert all(2.0 <= sh['at'] < 8.0 and sh['start'] <= sh['at'] < sh['end'] for sh in d['shots'])
    thumb = client.get(d['shots'][1]['thumb'])
    assert thumb.status_code == 200 and cv2.imdecode(np.frombuffer(thumb.data, np.uint8), 1).shape[:2] == (216, 384)
    assert post('/api/shorts/framing', start=2.0, end=3.0).status_code == 400
    assert client.post('/api/shorts/framing', headers=headers, json={'analysis_id': 'gone', 'start': 2, 'end': 8}).status_code == 404
    # The plan is worked out once for an in/out and setting, not every time it is asked for.
    calls = []
    real = shorts._plan_moment
    monkeypatch.setattr(shorts, '_plan_moment', lambda *x, **k: (calls.append(1), real(*x, **k))[1])
    post('/api/shorts/framing', start=2.0, end=8.0)
    post('/api/shorts/framing', start=2.0, end=8.0, reframe='fit')
    assert calls == [1], 'the second ask is answered from the first; a different setting is worked out'

    # The vertical preview, as the editor has set it: the middle shot shown whole, the last cropped at the left edge.
    framing = {'shots': [{'at': d['shots'][1]['at'], 'layout': 'fit'}, {'at': d['shots'][2]['at'], 'layout': 'crop', 'x': 0}]}
    pv = post('/api/shorts/vpreview', start=2.0, end=8.0, framing=framing, subtitles=False).get_json()
    assert pv['ok'] and pv['url'].startswith('/uploads/shv_')
    path = os.path.join(main.app.config['UPLOAD_FOLDER'], pv['url'].split('/')[-1])
    st = _probe(path)
    assert (st['video']['width'], st['video']['height']) == (360, 640) and abs(float(st['video']['duration']) - 6.0) < 0.1
    again = post('/api/shorts/vpreview', start=2.0, end=8.0, framing=framing, subtitles=False).get_json()
    assert again['url'] == pv['url'], 'the same preview asked for twice is made once'
    for bad in ({'shots': [{'at': 4.0, 'layout': 'zoom'}]}, {'shots': 'all'}, {'shots': [{'at': 99, 'layout': 'fit'}]},
                {'shots': [{'at': 4.0, 'layout': 'crop', 'x': 'left'}]}):
        r = post('/api/shorts/vpreview', start=2.0, end=8.0, framing=bad)
        assert r.status_code == 400 and 'framing is not valid' in r.get_json()['error'], bad

    # Rendered with it: the plan the short was made from says so, and a re-render for captions keeps it.
    batch = _render(client, headers, aid, [{'start': 2.0, 'end': 8.0, 'title': 'Set by hand', 'framing': framing}],
                    reframe='auto')['result']['batch']
    m = shorts._read_manifest(os.path.join(shorts.SHORTS_DIR, batch['batch_id']))
    sh = m['shorts'][0]
    assert sh['framing_edited'] is True
    assert [(g['a'], g['b'], g['layout'], g['x']) for g in sh['plan']['segs']] == [
        (0, 24, 'crop', 109.0), (25, 99, 'fit', None), (100, 149, 'crop', 0.0)]
    # The in point moved a second earlier afterwards: the corrections stay on their shots.
    batch = _render(client, headers, aid, [{'start': 1.0, 'end': 8.0, 'title': 'Moved', 'framing': framing}],
                    reframe='auto')['result']['batch']
    segs = shorts._read_manifest(os.path.join(shorts.SHORTS_DIR, batch['batch_id']))['shorts'][0]['plan']['segs']
    assert [(g['a'], g['b'], g['layout']) for g in segs] == [(0, 49, 'crop'), (50, 124, 'fit'), (125, 174, 'crop')]
    r = client.post('/api/shorts/render', headers=headers, json={'analysis_id': aid, 'items': [
        {'start': 2.0, 'end': 8.0, 'title': 'Bad', 'framing': {'shots': [{'at': 4, 'layout': 'tilt'}]}}]})
    assert r.status_code == 400 and '"Bad"' in r.get_json()['error']


def test_line_times_are_moved_onto_the_sound_when_the_service_gives_lines_only(env, monkeypatch):
    # Line times rounded to the whole second, as the reported service gives them.
    segs = [{'start': 1.29, 'end': 3.29, 'text': 'Linya isa.'}, {'start': 3.29, 'end': 5.29, 'text': 'Linya dalawa.'},
            {'start': 6.29, 'end': 9.29, 'text': 'Linya tatlo.'}, {'start': 10.29, 'end': 12.29, 'text': 'Linya apat.'},
            {'start': 13.29, 'end': 15.29, 'text': 'Linya lima.'}, {'start': 16.29, 'end': 19.29, 'text': 'Linya anim.'}]
    svc = Services(monkeypatch, words=[], segs=segs)
    svc.story_reply = lambda ids, payload: {'response': json.dumps({'moments': [
        {'start_id': ids[0], 'end_id': ids[-1], 'title': 'All of it', 'score': 8}]})}
    true = [(1.42, 2.66), (3.48, 4.71), (6.67, 8.48), (10.6, 12.1), (13.5, 15.0), (16.9, 18.7)]
    hop = sc.ENVELOPE_HOP
    db = np.full(int(24 / hop), -60.0, np.float32)
    for a, b in true:
        db[int(a / hop):int(b / hop)] = -18.0
    asked = []
    monkeypatch.setattr(sc, 'read_envelope', lambda ffmpeg, src, info, **k: (asked.append(src), db)[1])
    client, headers = _client()
    a = _analysis(client, _analyze(client, headers, env))
    assert asked == [env['path']] and a['stats']['lines_aligned'] == 6
    assert any('gave times for whole lines only, rounded to the second' in w and '6 of 6 lines were moved' in w
               for w in a['warnings'])
    an = shorts.analysis_get(a['analysis_id'])
    for (s0, e0), sg in zip(true, an['segments']):
        assert abs(sg['start'] - s0) < 0.06 and abs(sg['end'] - e0) < 0.06, sg
    # Line times given finely, not rounded: left as they are (the sound is read once all the same,
    # for the in and out points and the score).
    asked.clear()
    fine = Services(monkeypatch, words=[], segs=[dict(sg, start=sg['start'] + 0.013 * k, end=sg['end'] - 0.037 * k)
                                                 for k, sg in enumerate(segs)])
    fine.story_reply = svc.story_reply
    c = _analysis(client, _analyze(client, headers, env))
    assert asked == [env['path']] and c['stats']['lines_aligned'] == 0
    assert not sc.coarse_times(segs[:5]) and sc.coarse_times(segs), 'and it takes a few lines to tell'
    # With word timings there is nothing to move.
    asked.clear()
    Services(monkeypatch)
    b = _analysis(client, _analyze(client, headers, env))
    assert asked == [env['path']] and b['stats']['lines_aligned'] == 0 and not any('whole lines only' in w for w in b['warnings'])


def test_the_whole_files_sound_is_read_as_a_level_without_holding_it_in_memory(tmp_path):
    src = _talking_source(tmp_path / 't.mp4', [(1.0, 2.0), (3.0, 3.5)], seconds=5)
    info = sc.probe_source('ffprobe', src)
    db = sc.read_envelope('ffmpeg', src, info)
    assert abs(len(db) - 500) <= 3 and db.dtype == np.float32
    assert db[150] > db[250] + 30 and db[320] > db[250] + 30, 'the words, and the quiet between them'
    assert sc.read_envelope('ffmpeg', src, dict(info, audio_index=None)) is None
    assert sc.read_envelope('ffmpeg', str(tmp_path / 'missing.mp4'), info) is None


def test_the_better_face_model_can_be_installed_by_an_admin_and_only_as_published(env, monkeypatch, tmp_path):
    target = tmp_path / 'models' / 'face_detection_yunet_2023mar.onnx'
    monkeypatch.setattr(shorts, 'YUNET_FILE', str(target))
    user, uh = _client(user_id=7, role='user', username='ana')
    assert user.post('/api/shorts/face-model', headers=uh).status_code == 403
    admin, ah = _client()

    class Reply:
        def __init__(self, data):
            self.content, self.raw = data, None

        def raise_for_status(self):
            pass
    monkeypatch.setattr(shorts.requests, 'get', lambda url, **k: Reply(b'not the model'))
    r = admin.post('/api/shorts/face-model', headers=ah)
    assert r.status_code == 502 and 'checksum does not match' in r.get_json()['error'] and not target.exists()

    def offline(url, **k):
        raise shorts.requests.ConnectionError('no route to host')
    monkeypatch.setattr(shorts.requests, 'get', offline)
    r = admin.post('/api/shorts/face-model', headers=ah)
    assert r.status_code == 502 and 'no internet access' in r.get_json()['error'] and str(target) in r.get_json()['error']
    # The published file: saved where Vertical Shorts looks, and used.
    data = b'a model' * 100
    monkeypatch.setattr(shorts, 'YUNET_SHA256', hashlib.sha256(data).hexdigest())
    monkeypatch.setattr(shorts.requests, 'get', lambda url, **k: Reply(data))
    monkeypatch.setattr(shorts, '_face_model_path', lambda: str(target) if target.exists() else None)
    monkeypatch.setattr(sc, 'FaceDetector', lambda model=None: type('D', (), {'kind': 'yunet' if model else 'haar'})())
    r = admin.post('/api/shorts/face-model', headers=ah)
    assert r.status_code == 200 and r.get_json()['face_detector'] == 'yunet' and target.read_bytes() == data
    assert shorts.YUNET_URL.endswith('/face_detection_yunet/face_detection_yunet_2023mar.onnx')


# --------------------------------------------------------------------------
# Projects
# --------------------------------------------------------------------------

def _png(w=1280, h=720, colour=(40, 90, 200)):
    ok, buf = cv2.imencode('.png', np.full((h, w, 3), colour, np.uint8))
    assert ok
    return buf.tobytes()


def test_a_project_holds_the_episode_details_and_a_picture(env):
    ana, ah = _client(user_id=7, role='user', username='ana')
    ben, bh = _client(user_id=8, role='user', username='ben')
    admin, adh = _client(user_id=1, role='admin')
    url = '/api/shorts/projects'

    r = ana.post(url, headers=ah, content_type='multipart/form-data', data={
        'title': '  Maria   Clara at Ibarra ', 'episode': 'Ep. 12', 'air_date': '2026-10-12',
        'description': 'Finale week.\n  Two   teasers for social. ', 'thumbnail': (io.BytesIO(_png()), 'key art.PNG')})
    assert r.status_code == 200, r.get_json()
    p = r.get_json()['project']
    assert (p['title'], p['episode'], p['air_date']) == ('Maria Clara at Ibarra', 'Ep. 12', '2026-10-12')
    assert p['description'] == 'Finale week.\nTwo teasers for social.' and p['name'] == 'Maria Clara at Ibarra \u2014 Ep. 12'
    assert p['username'] == 'ana' and p['has_thumb'] and (p['batches'], p['shorts']) == (0, 0)
    # The picture is stored as PRISM's own, smaller, JPEG of it -- never the file as sent.
    t = ben.get(p['thumb_url'])
    assert t.status_code == 200 and t.content_type == 'image/jpeg'
    img = cv2.imdecode(np.frombuffer(t.data, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[:2] == (360, 640) and abs(int(img[100, 100, 2]) - 200) < 6
    t.close()
    assert sorted(os.listdir(os.path.join(shorts.SHORTS_PROJECTS_DIR, p['project_id']))) == ['project.json', 'thumb.jpg']

    # The whole team sees it, newest change first; only the title is required.
    r = ben.post(url, headers=bh, data={'title': 'Black Rider'})
    q = r.get_json()['project']
    assert (q['episode'], q['air_date'], q['description'], q['thumb_url'], q['has_thumb']) == ('', None, '', None, False)
    for c in (ana, ben, admin):
        assert [x['title'] for x in c.get(url).get_json()['items']] == ['Black Rider', 'Maria Clara at Ibarra', 'Tadhana']
    mine = {x['project_id']: x['can_delete'] for x in ana.get(url).get_json()['items']}
    assert mine[p['project_id']] is True and mine[q['project_id']] is False

    # What cannot be accepted says why, and creates nothing.
    for bad, says in (({'title': '   '}, 'programme title'), ({'title': 'X', 'air_date': '12/10/2026'}, 'air date'),
                      ({'title': 'X', 'thumbnail': (io.BytesIO(b'not an image'), 'a.jpg')}, 'could not be read'),
                      ({'title': 'X', 'thumbnail': (io.BytesIO(_png()), 'a.gif')}, 'JPEG, PNG or WebP'),
                      ({'title': 'X', 'thumbnail': (io.BytesIO(b'x' * (shorts.sp.THUMB_MAX_BYTES + 5)), 'a.png')}, 'larger than')):
        r = ana.post(url, headers=ah, content_type='multipart/form-data', data=bad)
        assert r.status_code == 400 and says in r.get_json()['error'], bad.get('title')
    assert len(ana.get(url).get_json()['items']) == 3

    # Anyone on the team can correct the details; the picture can be replaced or removed.
    one = f"{url}/{p['project_id']}"
    r = ben.post(one, headers=bh, data={'title': 'Maria Clara at Ibarra', 'episode': 'Ep. 13', 'air_date': ''})
    e = r.get_json()['project']
    assert (e['episode'], e['air_date'], e['has_thumb'], e['username']) == ('Ep. 13', None, True, 'ana')
    assert ana.get(url).get_json()['items'][0]['project_id'] == p['project_id'], 'the one just changed comes first'
    r = ana.post(one, headers=ah, data={'title': 'Maria Clara at Ibarra', 'remove_thumbnail': '1'})
    assert r.get_json()['project']['has_thumb'] is False and ana.get(f'{one}/thumb').status_code == 404

    # Deleting: its creator or an admin, not a teammate. Ids are never paths.
    assert ben.delete(one, headers=bh).status_code == 403
    assert admin.delete(f"{url}/{q['project_id']}", headers=adh).get_json() == {'ok': True}
    assert ana.delete(one, headers=ah).get_json() == {'ok': True}
    assert not os.path.exists(os.path.join(shorts.SHORTS_PROJECTS_DIR, p['project_id']))
    for odd in ('p0000000000', '..', 'project.json', env['project'] + 'x'):
        assert ana.get(f'{url}/{odd}').status_code == 404
    anon = main.app.test_client()
    assert anon.get(url).status_code in (302, 401, 403)


def test_shorts_are_filed_under_their_project_and_a_project_with_shorts_cannot_be_deleted(env, monkeypatch):
    Services(monkeypatch)
    ana, ah = _client(user_id=7, role='user', username='ana')
    ben, bh = _client(user_id=8, role='user', username='ben')
    a = _analysis(ana, _analyze(ana, ah, env))
    assert a['project_id'] == env['project']
    batch = _render(ana, ah, a['analysis_id'], [{'start': 2.0, 'end': 6.0, 'title': 'One'}], reframe='fit',
                    subtitles=False)['result']['batch']
    assert batch['project_id'] == env['project'] and batch['project_name'] == 'Tadhana \u2014 Ep. 101'
    # "Generate without preview" files its shorts the same way.
    auto = _analyze(ana, ah, env, auto_render='1', reframe='fit')['result']['batch']
    assert auto['project_id'] == env['project']

    p = ben.get('/api/shorts/projects').get_json()['items'][0]
    assert (p['batches'], p['shorts']) == (2, 1 + len(auto['shorts'])) and p['has_thumb'] is False
    assert p['thumb_url'].startswith(f"/api/shorts/file/{auto['batch_id']}/"), 'no picture of its own: its newest short'
    assert ben.get(p['thumb_url']).status_code == 200
    assert {b['batch_id'] for b in ben.get(f"/api/shorts/batches?project_id={env['project']}").get_json()['items']} == {
        batch['batch_id'], auto['batch_id']}

    r = ana.delete(f"/api/shorts/projects/{env['project']}", headers=ah)
    assert r.status_code == 409 and 'still holds' in r.get_json()['error']
    assert shorts.sp.load(shorts.SHORTS_PROJECTS_DIR, env['project']) is not None

    # Moved to another project by its maker; a teammate cannot move it.
    other = ana.post('/api/shorts/projects', headers=ah, data={'title': 'Specials'}).get_json()['project']
    move = f"/api/shorts/batches/{batch['batch_id']}/project"
    assert ben.post(move, json={'project_id': other['project_id']}, headers=bh).status_code == 404
    assert ana.post(move, json={'project_id': 'p0000000000'}, headers=ah).status_code == 400
    moved = ana.post(move, json={'project_id': other['project_id']}, headers=ah).get_json()['batch']
    assert moved['project_id'] == other['project_id'] and moved['project_name'] == 'Specials'
    counts = {x['title']: x['batches'] for x in ana.get('/api/shorts/projects').get_json()['items']}
    assert counts == {'Specials': 1, 'Tadhana': 1}


def test_shorts_made_before_projects_stay_their_makers_until_filed(env, monkeypatch):
    Services(monkeypatch)
    ana, ah = _client(user_id=7, role='user', username='ana')
    ben, bh = _client(user_id=8, role='user', username='ben')
    admin, _ = _client(user_id=1, role='admin')
    a = _analysis(ana, _analyze(ana, ah, env))
    batch = _render(ana, ah, a['analysis_id'], [{'start': 2.0, 'end': 6.0, 'title': 'One'}], reframe='fit',
                    subtitles=False)['result']['batch']
    # As it would have been written before: no project in the manifest.
    path = os.path.join(shorts.SHORTS_DIR, batch['batch_id'], 'batch.json')
    with open(path) as f:
        m = json.load(f)
    m.pop('project_id')
    with open(path, 'w') as f:
        json.dump(m, f)
    url = batch['shorts'][0]['url']
    assert [b['project_id'] for b in ana.get('/api/shorts/batches?unfiled=1').get_json()['items']] == [None]
    assert len(admin.get('/api/shorts/batches?unfiled=1').get_json()['items']) == 1
    assert ben.get('/api/shorts/batches').get_json()['items'] == [] and ben.get(url).status_code == 404
    assert ana.get(f"/api/shorts/batches?project_id={env['project']}").get_json()['items'] == []
    # Filed, it becomes the team's.
    r = ana.post(f"/api/shorts/batches/{batch['batch_id']}/project", json={'project_id': env['project']}, headers=ah)
    assert r.get_json()['batch']['project_id'] == env['project']
    assert ana.get('/api/shorts/batches?unfiled=1').get_json()['items'] == []
    r = ben.get(url)
    assert r.status_code == 200
    r.close()
    # A batch whose project has since gone falls back to being its maker's.
    m['project_id'] = 'p0123456789'
    with open(path, 'w') as f:
        json.dump(m, f)
    assert ben.get(url).status_code == 404 and ana.get('/api/shorts/batches?unfiled=1').get_json()['items'][0]['project_id'] is None


def test_shorts_are_numbered_and_listed_in_the_order_they_happen_in_the_episode(env, monkeypatch):
    """Ticked in any order, ranked in any order: short 01 is the earliest
    moment and the last number is the latest, in the filenames and on screen."""
    Services(monkeypatch)
    client, headers = _client()
    a = _analysis(client, _analyze(client, headers, env))
    items = [{'start': 17.0, 'end': 21.0, 'title': 'Late'}, {'start': 2.0, 'end': 6.0, 'title': ''},
             {'start': 9.0, 'end': 13.0, 'title': 'Middle'}]
    batch = _render(client, headers, a['analysis_id'], items, reframe='fit', subtitles=False)['result']['batch']
    assert [(s['number'], s['index'], s['title'], s['start']) for s in batch['shorts']] == [
        (1, 1, 'Short 1', 2.0), (2, 2, 'Middle', 9.0), (3, 3, 'Late', 17.0)]
    assert [s['file'] for s in batch['shorts']] == ['episode_short_01_Short_1.mp4', 'episode_short_02_Middle.mp4',
                                                    'episode_short_03_Late.mp4']
    bdir = os.path.join(shorts.SHORTS_DIR, batch['batch_id'])
    assert sorted(f for f in os.listdir(bdir) if f.endswith('.mp4')) == [s['file'] for s in batch['shorts']], \
        'sorted by name, the folder is in episode order too'
    # A batch saved before this, numbered by rank: still listed in episode
    # order, each one called by its place in that order.
    path = os.path.join(bdir, 'batch.json')
    with open(path) as f:
        m = json.load(f)
    m.pop('numbering')
    m['shorts'] = [dict(m['shorts'][2], index=1), dict(m['shorts'][0], index=2), dict(m['shorts'][1], index=3)]
    with open(path, 'w') as f:
        json.dump(m, f)
    old = client.get('/api/shorts/batches').get_json()['items'][0]
    assert [(s['number'], s['index'], s['start']) for s in old['shorts']] == [(1, 2, 2.0), (2, 3, 9.0), (3, 1, 17.0)]
    # One that could not be rendered leaves its number unused rather than renumbering the rest.
    m['numbering'] = 'episode'
    m['shorts'] = [dict(m['shorts'][1], index=1), dict(m['shorts'][0], index=3)]
    m['errors'] = [{'index': 2, 'title': 'Middle', 'error': 'x'}]
    with open(path, 'w') as f:
        json.dump(m, f)
    gap = client.get('/api/shorts/batches').get_json()['items'][0]
    assert [s['number'] for s in gap['shorts']] == [1, 3] and gap['errors'][0]['index'] == 2


def test_full_flow_render_save_download_send_delete(env, monkeypatch):
    Services(monkeypatch)
    client, headers = _client(user_id=7, role='user', username='ana')
    a = _analysis(client, _analyze(client, headers, env))
    c = a['candidates'][0]
    # The editor retitles the first candidate, and adds a range of their own that spans a cut.
    items = [{'start': c['start'], 'end': c['end'], 'title': 'Ang lihim: "ni Ramon"!'},
             {'start': 17.0, 'end': 21.0, 'title': ''}]
    job = _render(client, headers, a['analysis_id'], items, reframe='auto', subtitles=True, subtitle_size='l')
    assert job['error'] is None and job['percent'] == 100
    assert [s['label'] for s in job['stages']] == ['Preparing', 'Rendering shorts', 'Done']
    batch = job['result']['batch']
    assert batch['status'] == 'complete' and batch['errors'] == [] and batch['orig_name'] == 'episode.mp4'
    assert batch['options']['reframe'] == 'auto' and batch['options']['face_detector'] in ('haar', 'yunet')
    s1, s2 = batch['shorts']
    assert s1['file'] == 'episode_short_01_Ang_lihim_ni_Ramon.mp4', 'a safe filename from the title'
    assert s2['file'] == 'episode_short_02_Short_2.mp4' and s2['title'] == 'Short 2'
    assert batch['options']['speaker'] is False
    assert abs(s2['duration'] - 4.0) < 0.05 and s2['layouts'] == {'crop': 1, 'fit': 0, 'tracked': 0, 'speaker': 0, 'split': 0}, \
        'two faceless shots, both centre-cropped at the same x, merge into one instruction'
    bdir = os.path.join(shorts.SHORTS_DIR, batch['batch_id'])
    assert sorted(os.listdir(bdir)) == sorted(['batch.json'] + [s[k] for s in (s1, s2) for k in ('file', 'srt')] +
                                              [os.path.basename(s['thumb_url']) for s in (s1, s2)])

    # The files themselves.
    fr = _frames(os.path.join(bdir, s2['file']))
    assert len(fr) == 100 and fr[0].shape[:2] == (1920, 1080)

    def shade(f):
        r, g, b = _rgb(f, 540, 300)
        return 'orange' if r > 200 and 120 < g < 210 and b < 80 else 'navy' if b > 90 and r < 60 and g < 60 else (r, g, b)
    # 17-21 s of the episode: the orange shot (15-18 s) then the navy one, cut on frame 25 exactly.
    assert [shade(f) for f in fr[:25]] == ['orange'] * 25
    assert [shade(f) for f in fr[25:]] == ['navy'] * 75
    st = _probe(os.path.join(bdir, s1['file']))
    assert (st['video']['width'], st['video']['height']) == (1080, 1920) and st['audio']['channels'] == 2
    srt = open(os.path.join(bdir, s1['srt']), encoding='utf-8').read()
    assert srt.startswith('1\n00:00:0') and 'Linya 1 ng usapan.' in srt and 'Linya 5' not in srt
    if sc.has_filter('ffmpeg', 'ass'):
        assert s1['captions'] is True and batch['warnings'] == []
    assert not [n for n in os.listdir(main.app.config['UPLOAD_FOLDER']) if n.startswith('shsub_')], \
        'caption work files are cleaned up'

    # Filed under the project it was made for, which the whole team sees:
    # another account can play, download and send it, but only its maker
    # (or an admin) can delete it.
    assert batch['project_id'] == env['project'] and batch['project_name'] == 'Tadhana \u2014 Ep. 101'
    listed = client.get('/api/shorts/batches').get_json()['items']
    assert [b['batch_id'] for b in listed] == [batch['batch_id']] and listed[0]['can_delete'] is True
    other, oh = _client(user_id=8, role='user', username='ben')
    admin, ah = _client(user_id=1, role='admin')
    theirs = other.get(f"/api/shorts/batches?project_id={env['project']}").get_json()['items']
    assert [b['batch_id'] for b in theirs] == [batch['batch_id']] and theirs[0]['can_delete'] is False
    assert other.get('/api/shorts/batches?unfiled=1').get_json()['items'] == []
    assert admin.get('/api/shorts/batches').get_json()['items'][0]['can_delete'] is True

    # Download: one file (with Range, for seeking), as an attachment, and the whole batch.
    r = client.get(s1['url'])
    assert r.status_code == 200 and r.content_type == 'video/mp4' and len(r.data) == s1['size']
    r = client.get(s1['url'], headers={'Range': 'bytes=0-99'})
    assert r.status_code == 206 and len(r.data) == 100
    r = client.get(s1['srt_url'] + '?download=1')
    assert 'attachment' in r.headers['Content-Disposition'] and s1['srt'] in r.headers['Content-Disposition']
    assert client.get(s1['thumb_url']).content_type == 'image/jpeg'
    r = client.get(f"/api/shorts/batches/{batch['batch_id']}/zip")
    assert r.status_code == 200 and 'episode_shorts.zip' in r.headers['Content-Disposition']
    names = zipfile.ZipFile(io.BytesIO(r.data)).namelist()
    assert sorted(names) == sorted([s[k] for s in (s1, s2) for k in ('file', 'srt')])
    r.close()
    assert not [n for n in os.listdir(main.app.config['UPLOAD_FOLDER']) if n.endswith('.zip')], \
        'the zip built for the download is removed once it has been sent'

    # Nothing in the folder is reachable except what the manifest lists. A
    # teammate can open what is in the project but cannot delete or move it.
    assert client.get(f"/api/shorts/file/{batch['batch_id']}/batch.json").status_code == 404
    assert client.get(f"/api/shorts/file/{batch['batch_id']}/..%2F..%2Fbatch.json").status_code == 404
    for url in (s1['url'], f"/api/shorts/batches/{batch['batch_id']}/zip"):
        r = other.get(url)
        assert r.status_code == 200
        r.close()
    assert other.delete(f"/api/shorts/batches/{batch['batch_id']}", headers=oh).status_code == 404
    assert other.post(f"/api/shorts/batches/{batch['batch_id']}/project", json={'project_id': env['project']},
                      headers=oh).status_code == 404

    # Send to a network destination: video destinations only; MP4s, plus captions when asked.
    dests = {1: {'id': 1, 'name': 'Social team', 'delivery_kind': 'video', 'path': r'\\nas\social'},
             2: {'id': 2, 'name': 'EDL drop', 'delivery_kind': 'csv', 'path': r'\\nas\edl'}}
    monkeypatch.setattr(shorts, 'network_destination_get', lambda i: dests.get(i))
    sent = []
    monkeypatch.setattr(pipeline, 'send_file_to_network_destination',
                        lambda local, remote, dest: sent.append((os.path.basename(local), remote, dest['name'])))
    send_url = f"/api/shorts/batches/{batch['batch_id']}/send"
    assert other.post(send_url, json={'destination_id': 1, 'files': [s1['file']]}, headers=oh).get_json()['ok'], \
        'a teammate can send what is in the project'
    del sent[:]
    assert client.post(send_url, json={'destination_id': 99}, headers=headers).status_code == 400
    r = client.post(send_url, json={'destination_id': 2}, headers=headers)
    assert r.status_code == 400 and 'not finished video' in r.get_json()['error'] and sent == []
    r = client.post(send_url, json={'destination_id': 1}, headers=headers)
    assert r.get_json()['sent'] == [s1['file'], s2['file']] and r.get_json()['destination'] == 'Social team'
    assert sent == [(s1['file'], s1['file'], 'Social team'), (s2['file'], s2['file'], 'Social team')]
    del sent[:]
    r = client.post(send_url, json={'destination_id': 1, 'include_srt': True, 'files': [s2['file']]}, headers=headers)
    assert r.get_json()['sent'] == [s2['file'], s2['srt']]

    # One short, renamed on the way out: its captions go with it under the same
    # name, the extensions stay the files' own, and the saved short is untouched.
    for typed, stem in (('Tadhan EP101 teaser', 'Tadhan_EP101_teaser'), (' teaser_v1.2.mp4 ', 'teaser_v1.2'),
                        ('..\\..\\share\\x', 'share_x')):
        del sent[:]
        r = client.post(send_url, json={'destination_id': 1, 'include_srt': True, 'files': [s2['file']],
                                        'filename': typed}, headers=headers).get_json()
        assert r == {'ok': True, 'sent': [stem + '.mp4', stem + '.srt'], 'destination': 'Social team'}, typed
        assert sent == [(s2['file'], stem + '.mp4', 'Social team'), (s2['srt'], stem + '.srt', 'Social team')]
    del sent[:]
    r = client.post(send_url, json={'destination_id': 1, 'files': [s1['file']], 'filename': 'solo'}, headers=headers)
    assert r.get_json()['sent'] == ['solo.mp4'] and sent == [(s1['file'], 'solo.mp4', 'Social team')]
    # Left blank, the short's own name. A name for several at once, or one that is unusable, sends nothing.
    del sent[:]
    r = client.post(send_url, json={'destination_id': 1, 'files': [s1['file']], 'filename': '  '}, headers=headers)
    assert r.get_json()['sent'] == [s1['file']]
    del sent[:]
    r = client.post(send_url, json={'destination_id': 1, 'filename': 'everything'}, headers=headers)
    assert r.status_code == 400 and 'one short at a time' in r.get_json()['error']
    r = client.post(send_url, json={'destination_id': 1, 'files': [s1['file']], 'filename': '///'}, headers=headers)
    assert r.status_code == 400 and 'can be used in a filename' in r.get_json()['error'] and sent == []
    assert sorted(f for f in os.listdir(os.path.join(shorts.SHORTS_DIR, batch['batch_id'])) if f.endswith('.mp4')) == \
        sorted([s1['file'], s2['file']])

    def boom(local, remote, dest):
        raise ValueError('Could not write to "Social team": access denied')
    monkeypatch.setattr(pipeline, 'send_file_to_network_destination', boom)
    r = client.post(send_url, json={'destination_id': 1}, headers=headers)
    assert r.status_code == 502 and 'access denied' in r.get_json()['error']

    # Delete.
    assert client.delete(f"/api/shorts/batches/{batch['batch_id']}", headers=headers).get_json() == {'ok': True}
    assert not os.path.exists(bdir)
    assert client.get('/api/shorts/batches').get_json()['items'] == []
    assert client.get(s1['url']).status_code == 404


def test_batch_ids_cannot_escape_the_shorts_folder(env, tmp_path):
    secret = tmp_path / 'outside'
    secret.mkdir()
    (secret / 'batch.json').write_text(json.dumps({'batch_id': 'x', 'user_id': 1, 'shorts': [
        {'index': 1, 'file': 'a.mp4', 'title': 't'}]}))
    (secret / 'a.mp4').write_bytes(b'secret')
    client, headers = _client()
    for bid in ('..', '../outside', 'outside', '1700000000_ABCDEF', '1700000000_abcdef/..', '%2e%2e'):
        assert shorts._batch_dir(bid) is None
        assert client.get(f'/api/shorts/file/{bid}/a.mp4').status_code == 404
        assert client.delete(f'/api/shorts/batches/{bid}', headers=headers).status_code in (404, 405)
    assert (secret / 'a.mp4').exists()
    assert shorts._batch_dir('1700000000_abcdef') == os.path.join(shorts.SHORTS_DIR, '1700000000_abcdef')


def test_follow_the_speaker_is_applied_only_when_asked_and_recorded_on_the_batch(env, monkeypatch):
    """The render job's side of "Follow the speaker": mouths are only
    measured when the option is on, the transcript is handed over in the
    clip's own time, and what was done is written where the editor sees it.
    (How a speaker is chosen is pinned in test_shorts_core.py.)"""
    Services(monkeypatch)
    client, headers = _client()
    aid = _analyze(client, headers, env)['result']['analysis_id']
    asked = []

    def two_people(path, start_f, n_frames, fps, detector, step_sec=0.2, sar=1.0, mouth=False, bodies=None, **_):
        # A wide two-shot throughout. The one on the left has the first two
        # lines of the transcript (1.0-4.5 s), the one on the right the next two.
        asked.append(mouth)
        out = []
        for i in range(0, n_frames, 5):
            t = (start_f + i) / fps
            left, right = (40.0, 60.0, 30.0, 30.0), (280.0, 60.0, 30.0, 30.0)
            if mouth:
                left, right = left + (0.1 if 1.0 <= t < 4.6 else 0.0,), right + (0.1 if 5.0 <= t < 8.6 else 0.0,)
            out.append((i, [left, right]))
        return out
    monkeypatch.setattr(sc, 'sample_faces', two_people)
    items = [{'start': 1.0, 'end': 9.0, 'title': 'Two people'}]

    off = _render(client, headers, aid, items, subtitles=False)['result']['batch']
    assert off['options']['speaker'] is False
    assert off['shorts'][0]['layouts'] == {'crop': 0, 'fit': 1, 'tracked': 0, 'speaker': 0, 'split': 0}, 'shown whole, as before'

    on = _render(client, headers, aid, items, subtitles=False, speaker=True)['result']['batch']
    assert on['options']['speaker'] is True and on['warnings'] == []
    assert on['shorts'][0]['layouts'] == {'crop': 2, 'fit': 0, 'tracked': 0, 'speaker': 2, 'split': 0}, \
        'one framing per person: the cut between them falls inside the 3-6 s shot, not on a shot change'
    assert asked == [False, True]
    fr = _frames(os.path.join(shorts.SHORTS_DIR, on['batch_id'], on['shorts'][0]['file']))
    assert len(fr) == 200 and fr[0].shape[:2] == (1920, 1080)

    # "Fit" crops nothing, so there is nothing for the option to do.
    fit = _render(client, headers, aid, items, subtitles=False, speaker=True, reframe='fit')['result']['batch']
    assert fit['options']['speaker'] is False and fit['shorts'][0]['layouts']['speaker'] == 0

    # "Split screen": the same two people stacked instead of shown whole...
    split = _render(client, headers, aid, items, subtitles=False, reframe='split')['result']['batch']
    assert split['options']['reframe'] == 'split'
    assert split['shorts'][0]['layouts'] == {'crop': 0, 'fit': 0, 'tracked': 0, 'speaker': 0, 'split': 1}
    fr = _frames(os.path.join(shorts.SHORTS_DIR, split['batch_id'], split['shorts'][0]['file']))
    assert len(fr) == 200 and fr[0].shape[:2] == (1920, 1080)
    # ...and with "Follow the speaker" as well, only where both of them talk.
    # Here no shot holds a real exchange (the second voice gets a second of
    # the middle one), so each is framed on its speaker and nobody is stacked.
    both = _render(client, headers, aid, items, subtitles=False, reframe='split', speaker=True)['result']['batch']
    assert both['options'] == dict(split['options'], speaker=True)
    assert both['shorts'][0]['layouts'] == {'crop': 2, 'fit': 0, 'tracked': 0, 'speaker': 2, 'split': 0}


def test_follow_the_speaker_without_a_transcript_is_skipped_with_a_warning(env, monkeypatch):
    Services(monkeypatch, words=[], segs=[])
    client, headers = _client()
    aid = _analyze(client, headers, env)['result']['analysis_id']
    batch = _render(client, headers, aid, [{'start': 1.0, 'end': 9.0, 'title': 'Silent'}],
                    subtitles=False, speaker=True)['result']['batch']
    assert batch['status'] == 'complete' and batch['options']['speaker'] is False
    assert any('Follow the speaker' in w for w in batch['warnings'])


def test_sampling_faces_measures_mouths_only_when_asked(split_source):
    class Fixed:
        def detect(self, frame):
            return [(40.0, 60.0, 120.0, 160.0, 1.0)]
    plain = sc.sample_faces(str(split_source), 0, 30, 25.0, Fixed())
    assert [i for i, _ in plain] == [0, 5, 10, 15, 20, 25] and all(len(f) == 4 for _, fs in plain for f in fs)
    mouths = sc.sample_faces(str(split_source), 0, 30, 25.0, Fixed(), mouth=True)
    assert [i for i, _ in mouths] == [i for i, _ in plain], 'the same samples, on the same frames'
    assert [fs[0][:4] for _, fs in mouths] == [fs[0] for _, fs in plain]
    assert all(len(f) == 5 and f[4] is None for _, fs in mouths for f in fs), \
        'a flat colour has no face to lock on to, so the measurement is declined, not invented'


def test_one_failed_short_does_not_lose_the_others(env, monkeypatch):
    Services(monkeypatch)
    client, headers = _client()
    aid = _analyze(client, headers, env)['result']['analysis_id']
    real = sc.render_short

    def flaky(ffmpeg, src, out_path, *a, **k):
        if '_02_' in os.path.basename(out_path):
            return False, 'Conversion failed!'
        return real(ffmpeg, src, out_path, *a, **k)
    monkeypatch.setattr(sc, 'render_short', flaky)
    items = [{'start': 0, 'end': 4, 'title': 'One'}, {'start': 5, 'end': 9, 'title': 'Two'},
             {'start': 10, 'end': 14, 'title': 'Three'}]
    job = _render(client, headers, aid, items, reframe='fit', subtitles=False)
    batch = job['result']['batch']
    assert batch['status'] == 'partial'
    assert [s['title'] for s in batch['shorts']] == ['One', 'Three']
    assert batch['errors'] == [{'index': 2, 'title': 'Two', 'error': 'Conversion failed!'}]
    assert all(s['layouts'] == {'crop': 0, 'fit': 1, 'tracked': 0, 'speaker': 0, 'split': 0} and s['captions'] is False for s in batch['shorts'])
    assert all(s['srt'] for s in batch['shorts']), 'the .srt is written even when captions are not burned in'


def test_every_short_failing_fails_the_job_and_leaves_no_batch(env, monkeypatch):
    Services(monkeypatch)
    client, headers = _client()
    aid = _analyze(client, headers, env)['result']['analysis_id']
    monkeypatch.setattr(sc, 'render_short', lambda *a, **k: (False, 'Unknown encoder libx264'))
    job = _render(client, headers, aid, [{'start': 0, 'end': 4, 'title': 'One'}])
    assert job['done'] and 'None of the 1 shorts' in job['error'] and 'Unknown encoder libx264' in job['error']
    assert os.listdir(shorts.SHORTS_DIR) == []


def test_cancelling_mid_batch_keeps_what_already_finished(env, monkeypatch):
    Services(monkeypatch)
    client, headers = _client()
    aid = _analyze(client, headers, env)['result']['analysis_id']
    jobs = []
    real_new = pipeline.job_new
    monkeypatch.setattr(pipeline, 'job_new', lambda **k: jobs.append(real_new(**k)) or jobs[-1])
    real = sc.render_short

    def render_then_cancel(*a, **k):
        out = real(*a, **k)
        pipeline.job_cancel(jobs[-1])          # the user hits Cancel while short 1 is encoding
        return out
    monkeypatch.setattr(sc, 'render_short', render_then_cancel)
    job = _render(client, headers, aid, [{'start': 0, 'end': 4, 'title': 'One'}, {'start': 5, 'end': 9, 'title': 'Two'}],
                  reframe='fit', subtitles=False)
    assert job['done'] and job['error'] == 'Cancelled'
    listed = client.get('/api/shorts/batches').get_json()['items']
    assert len(listed) == 1 and listed[0]['status'] == 'partial'
    assert [s['title'] for s in listed[0]['shorts']] == ['One']
    assert os.path.getsize(os.path.join(shorts.SHORTS_DIR, listed[0]['batch_id'], listed[0]['shorts'][0]['file'])) > 0


def test_render_when_the_staged_source_has_been_cleaned_up(env, monkeypatch):
    Services(monkeypatch)
    client, headers = _client()
    aid = _analyze(client, headers, env)['result']['analysis_id']
    os.remove(env['path'])
    job = _render(client, headers, aid, [{'start': 0, 'end': 4, 'title': 'One'}])
    assert job['done'] and 'no longer on the server' in job['error'] and os.listdir(shorts.SHORTS_DIR) == []


def test_analysis_keeps_the_staged_source_alive_against_the_upload_sweeper(env, monkeypatch):
    Services(monkeypatch)
    old = time.time() - 5 * 3600
    os.utime(env['path'], (old, old))          # staged five hours ago; the sweeper reclaims at six
    ages = []
    real = pipeline.detect_scenes
    monkeypatch.setattr(pipeline, 'detect_scenes',
                        lambda p, **k: ages.append(time.time() - os.path.getmtime(p)) or real(p, **k))
    client, headers = _client()
    _analyze(client, headers, env)
    assert ages and ages[0] < 60, 'refreshed before the long steps start, not only once they finish'
    assert time.time() - os.path.getmtime(env['path']) < 60


def test_options_reports_models_and_what_this_server_can_do(env, monkeypatch):
    def fake_get(url, timeout=None, **kw):
        r = mock.Mock()
        r.json.return_value = {'models': [{'name': 'qwen3-vl:8b'}, {'name': 'llama3.1:8b'}]}
        return r
    monkeypatch.setattr(shorts.requests, 'get', fake_get)
    monkeypatch.setattr(pipeline, '_model_supports_vision', lambda name: 'vl' in name)
    client, _ = _client()
    d = client.get('/api/shorts/options').get_json()
    assert d['ok'] and d['vision_models'] == ['qwen3-vl:8b'] and d['text_models'] == ['qwen3-vl:8b', 'llama3.1:8b']
    assert d['face_detector'] in ('haar', 'yunet') and isinstance(d['captions_available'], bool)
    assert d['max_items'] == 100 and d['default_vision_model'] and d['auto_min_story'] == 6
    # The delivery formats: what the rest of PRISM exports, without AVC-Intra (a 1920x1080 format).
    assert [f['key'] for f in d['formats']] == ['mp4_high', 'prores_hq_2997', 'prores_hq_2398']
    assert d['formats'][0] == {'key': 'mp4_high', 'label': 'MP4 (H.264 High Profile)', 'ext': 'mp4'}
    assert d['speaker_default'] is False, '"Follow the speaker" starts unticked unless SHORTS_SPEAKER_CROP says otherwise'
    monkeypatch.setattr(shorts, 'SHORTS_SPEAKER_CROP', True)
    assert client.get('/api/shorts/options').get_json()['speaker_default'] is True

    def down(url, timeout=None, **kw):
        raise ConnectionError('refused')
    monkeypatch.setattr(shorts.requests, 'get', down)
    d = client.get('/api/shorts/options').get_json()
    assert d['ok'] and d['vision_models'] == [] and 'Could not reach Ollama' in d['error']


# --------------------------------------------------------------------------
# Shared job gate
# --------------------------------------------------------------------------

def test_shorts_jobs_wait_for_a_slot_at_the_shared_gate(env, monkeypatch):
    """With the server's one slot already taken (by a promo render, say), a
    shorts job must wait rather than run alongside it."""
    import threading
    Services(monkeypatch)
    monkeypatch.setattr(shorts, '_spawn', lambda fn, *a, **k: threading.Thread(
        target=fn, args=a, kwargs=k, daemon=True).start())
    monkeypatch.setattr(pipeline.GATE, 'limit', 1)
    with pipeline.GATE.cond:
        pipeline.GATE.running += 1              # someone else's job holds the only slot
    client, headers = _client()
    try:
        r = client.post('/api/shorts/analyze', headers=headers,
                        data={'shorts_file_network': env['staged'], 'min_dur': 5, 'max_dur': 12,
                              'project_id': env['project']})
        jid = r.get_json()['job_id']
        time.sleep(1.5)
        j = client.get(f'/api/shorts/progress/{jid}').get_json()
        assert j['done'] is False and j['status'] == 'queued' and 'Queued' in j['step']
    finally:
        with pipeline.GATE.cond:
            pipeline.GATE.running -= 1          # that job finishes
            pipeline.GATE.cond.notify_all()
    deadline = time.time() + 60
    while time.time() < deadline:
        j = client.get(f'/api/shorts/progress/{jid}').get_json()
        if j['done']:
            break
        time.sleep(0.3)
    assert j['done'] and j['error'] is None and j['result']['candidates'] == 2
    assert pipeline.GATE.status()['running'] == 0


def test_shorts_jobs_are_submitted_through_the_promo_gate_function(env, monkeypatch):
    Services(monkeypatch)
    seen = []
    real = pipeline.run_trailer_job_gated

    def spy(jid, params, runner=None):
        seen.append((runner is not None, pipeline.GATE.status()['running']))
        return real(jid, params, runner=runner)
    monkeypatch.setattr(pipeline, 'run_trailer_job_gated', spy)
    client, headers = _client()
    job = _analyze(client, headers, env)
    assert job['error'] is None
    assert seen == [(True, 0)], 'goes through the gate function, with its own job body'
    assert pipeline.GATE.status()['running'] == 0, 'slot released'


def test_gate_runs_the_promo_job_by_default_and_any_runner_when_given(monkeypatch):
    calls = []
    monkeypatch.setattr(pipeline, 'run_trailer_job', lambda jid, params: calls.append(('promo', params)))
    jid = pipeline.job_new(user_id=1, username='admin')
    pipeline.run_trailer_job_gated(jid, {'p': 1})
    assert calls == [('promo', {'p': 1})], 'existing callers are unaffected'

    def runner(j, params):
        calls.append(('custom', pipeline.GATE.status()['running']))
        raise RuntimeError('job body blew up')
    jid = pipeline.job_new(user_id=1, username='admin')
    with pytest.raises(RuntimeError):
        pipeline.run_trailer_job_gated(jid, {}, runner=runner)
    assert calls[-1] == ('custom', 1), 'the runner holds a slot while it runs'
    assert pipeline.GATE.status()['running'] == 0, 'and gives it back even when it raises'


def test_a_crashing_job_body_ends_the_job_with_a_message(env, monkeypatch):
    Services(monkeypatch)
    monkeypatch.setattr(pipeline, 'detect_scenes', mock.Mock(side_effect=RuntimeError('decoder exploded')))
    client, headers = _client()
    job = _analyze(client, headers, env)
    assert job['done'] and job['error'] == 'Unexpected error: decoder exploded'


# --------------------------------------------------------------------------
# Found in review
# --------------------------------------------------------------------------

def test_a_batch_interrupted_by_a_restart_is_settled_at_startup(env):
    """A crash or restart mid-render leaves batch.json saying 'rendering'
    with no job left to ever change that. Delete refuses a rendering batch,
    so without this it could never be removed."""
    def make(bid, status, shorts_):
        d = os.path.join(shorts.SHORTS_DIR, bid)
        os.makedirs(d)
        with open(os.path.join(d, 'batch.json'), 'w') as f:
            json.dump({'batch_id': bid, 'user_id': 1, 'orig_name': 'ep.mp4', 'status': status, 'shorts': shorts_}, f)
        for s_ in shorts_:
            open(os.path.join(d, s_['file']), 'wb').write(b'x')
        return d
    one = {'index': 1, 'title': 'One', 'file': 'ep_short_01_One.mp4'}
    kept = make('1700000001_aaaaaa', 'rendering', [one])
    empty = make('1700000002_bbbbbb', 'rendering', [])
    done = make('1700000003_cccccc', 'complete', [one])
    tomb = os.path.join(shorts.SHORTS_DIR, '.deleting_1700000004_dddddd_ab12')
    os.makedirs(tomb)
    stray = os.path.join(shorts.SHORTS_DIR, 'not_a_batch')
    os.makedirs(stray)

    client, headers = _client()
    r = client.delete('/api/shorts/batches/1700000001_aaaaaa', headers=headers)
    assert r.status_code == 409, 'while marked rendering it cannot be deleted'

    shorts.settle_interrupted_batches()
    assert json.load(open(os.path.join(kept, 'batch.json')))['status'] == 'partial'
    assert not os.path.exists(empty) and not os.path.exists(tomb)
    assert json.load(open(os.path.join(done, 'batch.json')))['status'] == 'complete'
    assert os.path.isdir(stray), 'only folders this feature created are touched'
    assert client.delete('/api/shorts/batches/1700000001_aaaaaa', headers=headers).get_json() == {'ok': True}
    assert not os.path.exists(kept)


def test_delete_is_all_or_nothing_when_a_file_is_in_use(env, monkeypatch):
    bid = '1700000005_eeeeee'
    d = os.path.join(shorts.SHORTS_DIR, bid)
    os.makedirs(d)
    json.dump({'batch_id': bid, 'user_id': 1, 'orig_name': 'ep.mp4', 'status': 'complete',
               'shorts': [{'index': 1, 'title': 'One', 'file': 'a.mp4'}]}, open(os.path.join(d, 'batch.json'), 'w'))
    open(os.path.join(d, 'a.mp4'), 'wb').write(b'x')
    real = os.rename

    def locked(src, dst):          # what Windows does while a file inside the folder is open
        raise PermissionError(13, 'The process cannot access the file because it is being used by another process')
    monkeypatch.setattr(os, 'rename', locked)
    client, headers = _client()
    r = client.delete(f'/api/shorts/batches/{bid}', headers=headers)
    assert r.status_code == 409 and 'in use' in r.get_json()['error']
    assert sorted(os.listdir(d)) == ['a.mp4', 'batch.json'], 'nothing was removed'
    assert len(client.get('/api/shorts/batches').get_json()['items']) == 1, 'and it is still listed'
    monkeypatch.setattr(os, 'rename', real)
    assert client.delete(f'/api/shorts/batches/{bid}', headers=headers).status_code == 200
    assert os.listdir(shorts.SHORTS_DIR) == []


def test_manifest_write_retries_when_the_file_is_briefly_held(env, monkeypatch):
    d = os.path.join(shorts.SHORTS_DIR, '1700000006_ffffff')
    os.makedirs(d)
    real, calls = os.replace, []

    def flaky(src, dst):
        calls.append(1)
        if len(calls) < 3:
            raise PermissionError(13, 'Access is denied')
        return real(src, dst)
    monkeypatch.setattr(os, 'replace', flaky)
    monkeypatch.setattr(shorts.time, 'sleep', lambda s: None)
    shorts._write_manifest(d, {'batch_id': 'x', 'shorts': []})
    assert len(calls) == 3 and shorts._read_manifest(d) == {'batch_id': 'x', 'shorts': []}


def test_a_preview_being_encoded_is_never_served_half_written(env, monkeypatch):
    Services(monkeypatch)
    client, headers = _client()
    aid = _analyze(client, headers, env)['result']['analysis_id']
    seen = []
    real = pipeline.run_ffmpeg

    def spy(cmd, **k):
        out = cmd[-1]
        seen.append(os.path.basename(out))
        # While this encode runs, a second request for the same range must not find a file to serve.
        final = [n for n in os.listdir(main.app.config['UPLOAD_FOLDER']) if n.startswith('shclip_')]
        seen.append(list(final))
        return real(cmd, **k)
    for n in os.listdir(main.app.config['UPLOAD_FOLDER']):
        if n.startswith('shclip_'):
            os.remove(os.path.join(main.app.config['UPLOAD_FOLDER'], n))
    monkeypatch.setattr(pipeline, 'run_ffmpeg', spy)
    d = client.post('/api/shorts/clip', json={'analysis_id': aid, 'start': 6.0, 'end': 9.5}, headers=headers).get_json()
    assert d['ok'] and seen[0].startswith('shpart_') and seen[1] == []
    assert os.path.basename(d['url']).startswith('shclip_')
    assert not [n for n in os.listdir(main.app.config['UPLOAD_FOLDER']) if n.startswith('shpart_')]
    del seen[:]
    assert client.post('/api/shorts/clip', json={'analysis_id': aid, 'start': 6.0, 'end': 9.5},
                       headers=headers).get_json()['url'] == d['url']
    assert seen == [], 'second request is served from the finished file, no re-encode'


def test_a_cancel_or_timeout_does_not_strand_work_files(env, monkeypatch):
    Services(monkeypatch)
    client, headers = _client()
    aid = _analyze(client, headers, env)['result']['analysis_id']
    up = main.app.config['UPLOAD_FOLDER']

    def timeout(*a, **k):
        raise sc.ToolTimeout('shorts render exceeded 900s')
    monkeypatch.setattr(sc, 'run_tool', timeout)
    job = _render(client, headers, aid, [{'start': 1, 'end': 9, 'title': 'One'}])
    assert job['done'] and 'took too long' in job['error']
    assert not [n for n in os.listdir(up) if n.startswith('shsub_')], 'caption file removed'
    assert os.listdir(shorts.SHORTS_DIR) == [], 'no batch folder and no partial MP4 left'


# --------------------------------------------------------------------------
# Analyses kept on disk: they survive a restart, keep the editor's review,
# are listed with their project, and fetch their episode again
# --------------------------------------------------------------------------

def test_an_analysis_and_its_review_survive_a_restart(env, monkeypatch):
    Services(monkeypatch)
    client, headers = _client(user_id=7, role='user', username='ana')
    a = _analysis(client, _analyze(client, headers, env))
    aid = a['analysis_id']
    assert a['review'] is None and a['created']
    c0, c1 = a['candidates'][0], a['candidates'][1]
    review = [dict(id=c0['id'], keep=True, title='Ang lihim', start=c0['start'] + 0.5, end=c0['end'],
                   captions={'start': c0['start'] + 0.5, 'end': c0['end'],
                             'cues': [{'start': 0.0, 'end': 1.5, 'text': 'Tama na.'}]}),
              dict(id=c1['id'], keep=False, title=c1['title'], start=c1['start'], end=c1['end']),
              dict(id='m123', keep=True, title='My moment', start=1.0, end=7.0, source='manual')]
    r = client.post(f'/api/shorts/analysis/{aid}/review', json={'items': review}, headers=headers)
    assert r.status_code == 200 and r.get_json()['saved'] == 3
    # A bad edit is refused, and the saved review stands.
    bad = client.post(f'/api/shorts/analysis/{aid}/review', json={'items': [dict(review[0], start='x')]},
                      headers=headers)
    assert bad.status_code == 400

    shorts.ANALYSES.clear()                     # the server restarts
    b = client.get(f'/api/shorts/analysis/{aid}').get_json()
    assert b['ok'] and [c['id'] for c in b['candidates']] == [c['id'] for c in a['candidates']]
    rv = b['review']
    assert [x['id'] for x in rv] == [c0['id'], c1['id'], 'm123'] and [x['keep'] for x in rv] == [True, False, True]
    assert rv[0]['title'] == 'Ang lihim' and abs(rv[0]['start'] - (c0['start'] + 0.5)) < 1e-6
    assert rv[0]['captions']['cues'][0]['text'] == 'Tama na.' and rv[0]['story_score'] == c0['story_score']
    assert rv[2]['source'] == 'manual' and rv[2]['score'] is None
    assert client.get(rv[0]['thumb']).status_code == 200, 'the stills are kept with it'
    # ...and it renders, from the analysis read back from disk.
    job = _render(client, headers, aid, [{'start': rv[2]['start'], 'end': rv[2]['end'], 'title': 'Mine'}],
                  reframe='fit')
    assert job.get('error') is None and len(job['result']['batch']['shorts']) == 1

    # Someone else's: not theirs to read or change. Path tricks go nowhere.
    other, oh = _client(user_id=8, role='user', username='ben')
    assert other.post(f'/api/shorts/analysis/{aid}/review', json={'items': []}, headers=oh).status_code == 403
    assert other.get(rv[0]['thumb']).status_code == 403
    assert client.get(f'/api/shorts/analysis/{aid}/thumb/..%2fanalysis.json').status_code == 404
    assert client.get('/api/shorts/analysis/..%2f..%2fetc').status_code == 404


def test_earlier_cuts_are_listed_with_their_project_and_can_be_deleted(env, monkeypatch):
    Services(monkeypatch)
    client, headers = _client(user_id=7, role='user', username='ana')
    first = _analyze(client, headers, env)['result']['analysis_id']
    second = _analyze(client, headers, env)['result']['analysis_id']
    a = client.get(f'/api/shorts/analysis/{first}').get_json()
    c = a['candidates'][0]
    client.post(f'/api/shorts/analysis/{first}/review', headers=headers,
                json={'items': [dict(id=c['id'], keep=True, title='x', start=c['start'], end=c['end'])]})
    shorts.ANALYSES.clear()
    items = client.get(f"/api/shorts/analyses?project_id={env['project']}").get_json()['items']
    assert {x['analysis_id'] for x in items} == {first, second}
    one = next(x for x in items if x['analysis_id'] == first)
    assert one['reviewed'] and one['kept'] == 1 and one['source_available'] and not one['refetchable']
    assert one['orig_name'] and one['username'] == 'ana'
    assert client.get('/api/shorts/analyses?project_id=pnothere').get_json()['items'] == []
    other, _ = _client(user_id=8, role='user', username='ben')
    assert other.get(f"/api/shorts/analyses?project_id={env['project']}").get_json()['items'] == []

    assert client.delete(f'/api/shorts/analysis/{second}', headers=headers).status_code == 200
    assert client.get(f'/api/shorts/analysis/{second}').status_code == 404
    assert not os.path.exists(os.path.join(shorts.SHORTS_ANALYSES_DIR, second))


def test_an_analysis_unopened_for_its_days_is_swept(env, monkeypatch):
    Services(monkeypatch)
    client, headers = _client()
    aid = _analyze(client, headers, env)['result']['analysis_id']
    a = shorts._load_analysis(aid)
    a['opened'] = a['created'] = time.time() - shorts.SHORTS_ANALYSIS_DAYS * 86400 - 60
    shorts._save_analysis(aid, a)
    old = time.time() - shorts.SHORTS_ANALYSIS_DAYS * 86400 - 60
    os.utime(os.path.join(shorts.SHORTS_ANALYSES_DIR, aid, 'analysis.json'), (old, old))
    shorts.ANALYSES.clear()
    _analyze(client, headers, env)              # a new one sweeps the old
    assert not os.path.exists(os.path.join(shorts.SHORTS_ANALYSES_DIR, aid))


def test_an_episode_cleared_from_the_server_is_fetched_again_from_its_network_folder(env, monkeypatch):
    Services(monkeypatch)
    keep = os.path.join(shorts.SHORTS_DIR, '..', 'nas_copy.mp4')
    shutil.copy(env['path'], keep)
    fetched = []

    def fetch(name, category, subpath):
        fetched.append((name, category, subpath))
        local = f'net_{int(time.time())}_again_{name}'
        shutil.copy(keep, os.path.join(main.app.config['UPLOAD_FOLDER'], local))
        return local

    monkeypatch.setattr(pipeline, 'fetch_network_file', fetch)
    pipeline.STAGED_ORIGINS[env['staged']] = {'category': 'shorts', 'subpath': 'Tadhana', 'name': 'episode.mp4'}
    client, headers = _client()
    aid = _analyze(client, headers, env)['result']['analysis_id']
    os.remove(env['path'])
    shorts.ANALYSES.clear()                     # and the server restarted, for good measure
    a = client.get(f'/api/shorts/analysis/{aid}').get_json()
    assert not a['source_available'] and a['refetchable']
    r = client.post('/api/shorts/clip', json={'analysis_id': aid, 'start': 1, 'end': 6}, headers=headers)
    assert r.status_code == 410 and r.get_json()['refetchable'] and 'Fetch the episode again' in r.get_json()['error']
    r = client.post(f'/api/shorts/analysis/{aid}/refetch', headers=headers)
    assert r.status_code == 200 and fetched == [('episode.mp4', 'shorts', 'Tadhana')]
    assert client.get(f'/api/shorts/analysis/{aid}').get_json()['source_available']
    # Gone again: the render fetches it itself.
    os.remove(shorts.analysis_get(aid)['path'])
    job = _render(client, headers, aid, [{'start': 1.0, 'end': 6.0, 'title': 'One'}], reframe='fit')
    assert job.get('error') is None and len(fetched) == 2
    pipeline.STAGED_ORIGINS.pop(env['staged'], None)


def test_fetching_a_network_file_remembers_where_it_came_from():
    assert pipeline.staged_origin('net_0_nothing.mp4') is None


# --------------------------------------------------------------------------
# Billboards, credits, narration and the editor's own ranges stay out
# --------------------------------------------------------------------------

def test_lines_the_story_model_marks_as_not_story_and_ranges_left_out_are_never_in_a_moment(env, monkeypatch):
    svc = Services(monkeypatch)

    def reply(ids, payload):
        out = json.loads(Services._default_story(ids, payload)['response'])
        out['not_story'] = [{'first_id': ids[3], 'last_id': ids[4], 'kind': 'narration'}]
        return {'response': json.dumps(out)}

    svc.story_reply = reply
    client, headers = _client()
    a = _analysis(client, _analyze(client, headers, env, leave_out='0:14-end'))
    p = svc.story_prompts[0]
    assert 'not_story' in p['format']['required'] and 'OBB/CBB' in p['prompt'] and 'sa nakaraang' in p['prompt']
    words, segs = _transcript()
    assert [c['title'] for c in a['candidates']] == ['Ang lihim ni Ramon'], 'the other one is all in the part left out'
    c = a['candidates'][0]
    assert c['end'] <= segs[3]['start'] and 'cleaned' in c['flags'], 'cut short of the narration'
    assert c['duration'] >= 5 and c['start'] <= segs[1]['start'], 'and made up from the line before, not the narration'
    w = next(w for w in a['warnings'] if w.startswith('Left out of every moment'))
    assert 'narration 00:07' in w and 'left out by you 00:14' in w
    an = shorts.analysis_get(a['analysis_id'])
    assert [e['kinds'] for e in an['excluded']] == [['narration'], ['editor']]
    assert an['options']['leave_out'][0][0] == 14.0
    # A range that cannot be read is refused before anything runs.
    r = client.post('/api/shorts/analyze', headers=headers, data={
        'shorts_file_network': env['staged'], 'project_id': env['project'], 'min_dur': 5, 'max_dur': 12,
        'leave_out': '0:00 until 1:30'})
    assert r.status_code == 400 and 'Leave out' in r.get_json()['error']


def test_frames_the_vision_model_sees_as_credits_keep_moments_clear_of_their_shot(env, monkeypatch):
    svc = Services(monkeypatch)
    story_post = svc._post
    # Every frame from 14 s on is credits.
    monkeypatch.setattr(shorts, '_grab_frames', lambda path, times, fps, **k: [
        (t, 'CREDITS' if t >= 14 else 'SCENE') for t in times])

    def post(url, json=None, timeout=None, **kw):
        payload = json or {}
        if not payload.get('images'):
            return story_post(url, json=json, timeout=timeout, **kw)
        svc.vision_calls.append(payload)
        credits = payload['images'][0] == 'CREDITS'
        resp = mock.Mock()
        resp.json.return_value = {'response': __import__('json').dumps(
            {'score': 1 if credits else 4, 'desc': 'x', 'kind': 'credits' if credits else 'story'})}
        return resp

    monkeypatch.setattr(sc.requests, 'post', post)
    client, headers = _client()
    a = _analysis(client, _analyze(client, headers, env))
    assert a['candidates'] and all(c['end'] <= 14.0 + 1e-6 for c in a['candidates'])
    assert any('credits' in w for w in a['warnings'] if w.startswith('Left out of every moment'))


def test_a_batch_records_the_running_time_of_its_source(env, monkeypatch):
    Services(monkeypatch)
    client, headers = _client()
    aid = _analyze(client, headers, env)['result']['analysis_id']
    batch = _render(client, headers, aid, [{'start': 1.0, 'end': 6.0, 'title': 'One'}], reframe='fit')['result']['batch']
    assert abs(batch['source_duration'] - float(_probe(env['path'])['video']['duration'])) < 0.2
    listed = client.get(f"/api/shorts/batches?project_id={env['project']}").get_json()['items']
    assert listed[0]['source_duration'] == batch['source_duration']


# --------------------------------------------------------------------------
# Multi-part sources
# --------------------------------------------------------------------------

@pytest.fixture
def two_parts(env):
    """The staged episode plus a second copy of it as 'part 2' (24 s each)."""
    second = f'net_{int(time.time())}_episode_pt2.mp4'
    dst = os.path.join(main.app.config['UPLOAD_FOLDER'], second)
    shutil.copy(env['path'], dst)
    yield dict(env, second=second)
    try:
        os.remove(dst)
    except OSError:
        pass


def test_walls_keep_candidates_on_one_side_of_a_part_break():
    segs = [{'start': 1.0 + 2.0 * i, 'end': 2.5 + 2.0 * i, 'text': f'Linya {i} ng usapan.'} for i in range(24)]
    words = []
    for s in segs:
        words += [{'start': s['start'] + k * 0.3, 'end': s['start'] + k * 0.3 + 0.25, 'word': w}
                  for k, w in enumerate(s['text'].split())]
    beats = [{'start_id': 0, 'end_id': 23, 'title': 'T', 'score': 8}]
    cands = sc.build_candidates(beats, segs, words, [0.0, 24.0], [], 48.0, 10, 30, walls=[24.0])
    assert cands
    for c in cands:
        assert not (c['start'] < 24.0 - 0.01 and c['end'] > 24.0 + 0.01), c


def test_a_two_part_source_is_analysed_on_one_timeline(two_parts, monkeypatch):
    Services(monkeypatch)
    client, headers = _client()
    job = _analyze(client, headers, two_parts, shorts_file2_network=two_parts['second'], count='auto')
    a = _analysis(client, job)
    assert len(a['parts']) == 2
    assert abs(a['duration'] - 48.0) < 0.5
    assert [p['offset'] for p in a['parts']][0] == 0
    assert 23 < a['parts'][1]['offset'] < 25
    for c in a['candidates']:
        assert int((c['start'] + 0.01) // a['parts'][1]['offset']) == int((c['end'] - 0.01) // a['parts'][1]['offset']), \
            'a moment never runs across the break between parts'


def test_a_short_from_part_two_is_cut_from_part_twos_file(two_parts, monkeypatch):
    Services(monkeypatch)
    client, headers = _client()
    a = _analysis(client, _analyze(client, headers, two_parts, shorts_file2_network=two_parts['second']))
    off = a['parts'][1]['offset']
    batch = _render(client, headers, a['analysis_id'],
                    [{'start': off + 3.0, 'end': off + 9.0, 'title': 'Second part'}], reframe='fit')['result']['batch']
    s = batch['shorts'][0]
    assert s['part'] == 2 and abs(s['start'] - (off + 3.0)) < 0.1
    assert len(batch['parts']) == 2
    out = os.path.join(shorts.SHORTS_DIR, batch['batch_id'], s['file'])
    assert abs(float(_probe(out)['video']['duration']) - 6.0) < 0.3


def test_leave_out_can_name_a_part(two_parts, monkeypatch):
    Services(monkeypatch)
    client, headers = _client()
    a = _analysis(client, _analyze(client, headers, two_parts, shorts_file2_network=two_parts['second'],
                                   leave_out='part 2 0:00-end', count='auto'))
    off = a['parts'][1]['offset']
    assert all(c['end'] <= off + 0.05 for c in a['candidates'])
    r = client.post('/api/shorts/analyze', headers=headers, data={
        'shorts_file_network': two_parts['staged'], 'shorts_file2_network': two_parts['second'],
        'project_id': two_parts['project'], 'leave_out': 'part 3 0:00-1:00'})
    assert r.status_code == 400 and 'no part 3' in r.get_json()['error']


def test_a_missing_second_part_is_named(two_parts):
    client, headers = _client()
    r = client.post('/api/shorts/analyze', headers=headers, data={
        'shorts_file_network': two_parts['staged'], 'shorts_file2_network': 'net_1_gone.mp4',
        'project_id': two_parts['project']})
    assert r.status_code == 400 and r.get_json()['error'].startswith('Part 2')


def test_an_episode_read_in_place_is_analysed_and_rendered_where_it_is_and_never_touched(env, monkeypatch, tmp_path):
    """The file stays on 'the share' (a folder outside the upload folder): nothing is copied, its
    modified time is not refreshed, and it is not deleted by the job."""
    Services(monkeypatch)
    share = tmp_path / 'share'
    share.mkdir()
    src = str(share / 'episode.mp4')
    shutil.copy(env['path'], src)
    os.utime(src, (1_600_000_000, 1_600_000_000))
    staged = f'net_{int(time.time())}_inplace_episode.mp4'
    pipeline.INPLACE[staged] = {'path': src, 'category': 'shorts', 'subpath': '', 'name': 'episode.mp4'}
    pipeline.INPLACE_ORIGINS[src] = {'category': 'shorts', 'subpath': '', 'name': 'episode.mp4'}
    client, headers = _client()
    a = _analysis(client, _analyze(client, headers, dict(env, staged=staged)))
    stored = shorts.analysis_get(a['analysis_id'])
    assert stored['path'] == src and a['source_available'] and a['refetchable']
    assert not [f for f in os.listdir(main.app.config['UPLOAD_FOLDER']) if f.endswith('inplace_episode.mp4')]
    batch = _render(client, headers, a['analysis_id'], [{'start': 1.0, 'end': 6.0, 'title': 'One'}],
                    reframe='fit')['result']['batch']
    assert batch['shorts'][0]['file']
    assert os.path.exists(src) and int(os.path.getmtime(src)) == 1_600_000_000, 'the share file is never touched'
    pipeline.INPLACE.pop(staged, None)
    pipeline.INPLACE_ORIGINS.pop(src, None)

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
import shutil
import subprocess
import time
import unittest.mock as mock
import zipfile
import io

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
                        'stream=codec_type,codec_name,width,height,nb_frames,channels,sample_rate,color_space,pix_fmt',
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
    monkeypatch.setattr(shorts, 'ANALYSES', {})
    monkeypatch.setattr(shorts, '_spawn', lambda fn, *a, **k: fn(*a, **k))
    monkeypatch.setattr(shorts, '_job_submit_limiter', core._RateLimiter(1000, 300))
    monkeypatch.setattr(shorts, 'SHORTS_PRESET', 'ultrafast')
    monkeypatch.setattr(pipeline, 'ALLOW_LOCAL_MEDIA_UPLOAD', False)
    staged = f'net_{int(time.time())}_episode.mp4'
    shutil.copy(str(episode), os.path.join(main.app.config['UPLOAD_FOLDER'], staged))
    yield {'staged': staged, 'path': os.path.join(main.app.config['UPLOAD_FOLDER'], staged)}
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
        monkeypatch.setattr(pipeline, 'transcribe_video', lambda path: (self.words, self.segs))
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
    data = {'shorts_file_network': env['staged'], 'min_dur': 5, 'max_dur': 12, 'count': 5}
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

    r = client.post('/api/shorts/analyze', data={}, headers=headers)
    assert r.status_code == 400 and 'No video' in r.get_json()['error']
    r = client.post('/api/shorts/analyze', data={'shorts_file_network': 'net_1_gone.mp4'}, headers=headers)
    assert r.status_code == 400 and 're-select' in r.get_json()['error']
    # Only files this app staged itself are trusted -- never an arbitrary name or path.
    for bad in ('../../etc/passwd', os.path.basename(env['path']).replace('net_', 'src_')):
        r = client.post('/api/shorts/analyze', data={'shorts_file_network': bad}, headers=headers)
        assert r.status_code == 400
    r = client.post('/api/shorts/analyze', headers=headers,
                    data={'shorts_file_network': env['staged'], 'min_dur': 60, 'max_dur': 62})
    assert r.status_code == 400 and 'at least 5 seconds' in r.get_json()['error']
    # Direct upload is refused server-side when the deployment has it off.
    r = client.post('/api/shorts/analyze', headers=headers, content_type='multipart/form-data',
                    data={'shorts_file': (io.BytesIO(b'not really a video'), 'ep.mp4')})
    assert r.status_code == 400 and 'Direct file upload is disabled' in r.get_json()['error']
    assert started == []

    monkeypatch.setattr(shorts, '_job_submit_limiter', core._RateLimiter(1, 300))
    assert client.post('/api/shorts/analyze', data={'shorts_file_network': env['staged']}, headers=headers).status_code == 200
    assert client.post('/api/shorts/analyze', data={'shorts_file_network': env['staged']}, headers=headers).status_code == 429
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
    assert 1 <= len(svc.vision_calls) <= 10 and a['stats']['frames_rated'] == len(svc.vision_calls)
    assert all(c['model'] == 'test-vl' and c['images'] and c['format']['required'] == ['score', 'desc']
               for c in svc.vision_calls)
    # Layer 2: one story call (24 s fits one chunk), same model by default, big enough context, with
    # the visual notes and the editor's focus in the prompt.
    assert len(svc.story_prompts) == 1
    p = svc.story_prompts[0]
    assert p['model'] == 'test-vl' and p['options']['num_ctx'] >= 8192 and 'moments' in p['format']['properties']
    assert '[SCREEN ' in p['prompt'] and 'two people arguing' in p['prompt'] and 'ang lihim ni Ramon' in p['prompt']
    assert '5 to 12 seconds' in p['prompt']
    # The GPU is handed back before whisper and again at the end.
    assert svc.unloaded == ['test-vl', 'test-vl']

    cands = a['candidates']
    assert [c['title'] for c in cands] == ['Ang lihim ni Ramon', 'Huling babala']
    assert cands[0]['score'] > cands[1]['score'] and cands[0]['story_score'] == 9
    words, segs = _transcript()
    for c, (i, j) in zip(cands, ((1, 4), (7, 10))):
        assert 5 <= c['duration'] <= 12.5
        assert c['start'] <= segs[i]['start'] and c['end'] >= segs[j]['end'], 'the chosen lines are fully inside'
        cut_words = [w for w in words if w['start'] < c['start'] < w['end'] or w['start'] < c['end'] < w['end']]
        assert cut_words == [], 'no word straddles an in or out point'
        assert c['thumb'] and os.path.exists(os.path.join(main.app.config['UPLOAD_FOLDER'], os.path.basename(c['thumb'])))
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
    assert svc.unloaded == ['eyes', 'brain']


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
    monkeypatch.setattr(pipeline, 'PREVIEW_TTL', -1)
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
    assert post([{'start': 1, 'end': 9}] * 21).status_code == 400
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
    assert r.status_code == 200 and len(started) == 1
    p = started[0]
    assert [(i['start'], i['title'][:8]) for i in p['items']] == [(0.0, 'Short 1'), (16.0, 'xxxxxxxx')]
    assert abs(p['items'][1]['end'] - 24.0) < 0.1 and len(p['items'][1]['title']) == 80
    assert (p['reframe'], p['subtitle_size'], p['subtitles']) == ('auto', 'm', False)


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
    assert abs(s2['duration'] - 4.0) < 0.05 and s2['layouts'] == {'crop': 1, 'fit': 0, 'tracked': 0}, \
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

    # Saved shorts: the owner and an admin see it, another account does not.
    listed = client.get('/api/shorts/batches').get_json()['items']
    assert [b['batch_id'] for b in listed] == [batch['batch_id']]
    other, oh = _client(user_id=8, role='user', username='ben')
    admin, ah = _client(user_id=1, role='admin')
    assert other.get('/api/shorts/batches').get_json()['items'] == []
    assert len(admin.get('/api/shorts/batches').get_json()['items']) == 1

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

    # Nothing in the folder is reachable except what the manifest lists, and nothing at all by another account.
    assert client.get(f"/api/shorts/file/{batch['batch_id']}/batch.json").status_code == 404
    assert client.get(f"/api/shorts/file/{batch['batch_id']}/..%2F..%2Fbatch.json").status_code == 404
    for url in (s1['url'], f"/api/shorts/batches/{batch['batch_id']}/zip"):
        assert other.get(url).status_code == 404
    assert other.delete(f"/api/shorts/batches/{batch['batch_id']}", headers=oh).status_code == 404
    assert other.post(f"/api/shorts/batches/{batch['batch_id']}/send", json={'destination_id': 1},
                      headers=oh).status_code == 404

    # Send to a network destination: video destinations only; MP4s, plus captions when asked.
    dests = {1: {'id': 1, 'name': 'Social team', 'delivery_kind': 'video', 'path': r'\\nas\social'},
             2: {'id': 2, 'name': 'EDL drop', 'delivery_kind': 'csv', 'path': r'\\nas\edl'}}
    monkeypatch.setattr(shorts, 'network_destination_get', lambda i: dests.get(i))
    sent = []
    monkeypatch.setattr(pipeline, 'send_file_to_network_destination',
                        lambda local, remote, dest: sent.append((os.path.basename(local), remote, dest['name'])))
    send_url = f"/api/shorts/batches/{batch['batch_id']}/send"
    assert client.post(send_url, json={'destination_id': 99}, headers=headers).status_code == 400
    r = client.post(send_url, json={'destination_id': 2}, headers=headers)
    assert r.status_code == 400 and 'not finished video' in r.get_json()['error'] and sent == []
    r = client.post(send_url, json={'destination_id': 1}, headers=headers)
    assert r.get_json()['sent'] == [s1['file'], s2['file']] and r.get_json()['destination'] == 'Social team'
    assert sent == [(s1['file'], s1['file'], 'Social team'), (s2['file'], s2['file'], 'Social team')]
    del sent[:]
    r = client.post(send_url, json={'destination_id': 1, 'include_srt': True, 'files': [s2['file']]}, headers=headers)
    assert r.get_json()['sent'] == [s2['file'], s2['srt']]

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
    assert all(s['layouts'] == {'crop': 0, 'fit': 1, 'tracked': 0} and s['captions'] is False for s in batch['shorts'])
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
    assert job['done'] and 'no longer staged' in job['error'] and os.listdir(shorts.SHORTS_DIR) == []


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
    assert d['max_items'] == 20 and d['default_vision_model']

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
                        data={'shorts_file_network': env['staged'], 'min_dur': 5, 'max_dur': 12})
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

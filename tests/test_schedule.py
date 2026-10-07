"""
Schedule Plug tab (schedule.py + schedule_core.py), exercised for real where
it matters: real artwork (a layered Photoshop file built here), real ffmpeg,
the real job path -- with only the language model and the network share
stood in for.

What has to hold from an editor's side:

  * artwork + length + (optionally) music + a few words in, one finished
    plug out, in the delivery format asked for, at that format's frame rate;
  * what comes out ends on the artwork itself, not an approximation of it;
  * the prompt is understood with or without the AI model, and the result
    says how it was understood;
  * a plug is saved, playable in the browser whatever its delivery codec,
    private to its owner, and can be downloaded, sent and deleted.
"""
import io
import json
import os
import shutil
import subprocess
import time
import unittest.mock as mock

import cv2
import numpy as np
import pytest

with mock.patch('requests.post'), mock.patch('requests.get'):
    import main
    import pipeline
    import schedule
    import core
    import auth
import schedule_core as sk

psd_tools = pytest.importorskip('psd_tools')
from PIL import Image, ImageDraw   # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which('ffmpeg') is None, reason='ffmpeg not available')

W, H = sk.CANVAS
LAYERS = ['Background', 'Title', 'Schedule Mon', 'Schedule Tue']


@pytest.fixture(scope='module')
def artwork(tmp_path_factory):
    """A four-layer schedule as a .psd, the same picture as a .png, and a
    short music bed."""
    d = tmp_path_factory.mktemp('schedule_art')
    parts = [(LAYERS[0], Image.new('RGBA', (W, H), (20, 30, 90, 255)))]
    for name, box, colour in ((LAYERS[1], [200, 80, 1700, 220], (255, 200, 0, 255)),
                              (LAYERS[2], [200, 300, 1700, 420], (255, 255, 255, 255)),
                              (LAYERS[3], [200, 460, 1700, 580], (255, 255, 255, 255))):
        im = Image.new('RGBA', (W, H), (0, 0, 0, 0))
        ImageDraw.Draw(im).rectangle(box, fill=colour)
        parts.append((name, im))
    psd = psd_tools.PSDImage.new('RGB', (W, H), color=0)
    flat = Image.new('RGBA', (W, H), (0, 0, 0, 255))
    for name, im in parts:
        box = im.getbbox()
        psd.append(psd.create_pixel_layer(im.crop(box), name=name, top=box[1], left=box[0]))
        flat.alpha_composite(im)
    psd.save(str(d / 'week.psd'))
    want = cv2.cvtColor(np.asarray(flat.convert('RGB')), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(d / 'week.png'), want)
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'lavfi', '-i', 'sine=frequency=220:duration=4',
                    str(d / 'bed.wav')], check=True, timeout=60)
    return {'psd': d / 'week.psd', 'png': d / 'week.png', 'music': d / 'bed.wav', 'want': want}


@pytest.fixture
def env(tmp_path, monkeypatch, artwork):
    """An isolated plugs folder, jobs that run inline, a private rate
    limiter, no AI model unless a test brings one, and the artwork staged
    the way Browse library stages a network file."""
    monkeypatch.setattr(schedule, 'SCHEDULE_DIR', str(tmp_path / 'plugs'))
    os.makedirs(schedule.SCHEDULE_DIR)
    monkeypatch.setattr(schedule, '_spawn', lambda fn, *a, **k: fn(*a, **k))
    monkeypatch.setattr(schedule, '_job_submit_limiter', core._RateLimiter(1000, 300))
    monkeypatch.setattr(schedule, 'SCHEDULE_USE_LLM', False)
    monkeypatch.setattr(pipeline, 'ALLOW_LOCAL_MEDIA_UPLOAD', False)
    up = main.app.config['UPLOAD_FOLDER']
    staged = {}
    for key, src, name in (('psd', artwork['psd'], 'Primetime Week 42.psd'), ('png', artwork['png'], 'week.png'),
                           ('music', artwork['music'], 'bed.wav')):
        staged[key] = f'net_{int(time.time())}_{name.replace(" ", "_")}'
        shutil.copy(str(src), os.path.join(up, staged[key]))
    yield staged
    for name in staged.values():
        try:
            os.remove(os.path.join(up, name))
        except OSError:
            pass


def _client(user_id=1, role='admin', username='admin'):
    client = main.app.test_client()
    tok = f'csrf-schedule-{user_id}'
    with client.session_transaction() as sess:
        sess.update(authed=True, user_id=user_id, username=username, role=role, csrf_token=tok)
    return client, {'X-CSRF-Token': tok}


def _render(client, headers, **form):
    r = client.post('/api/schedule/render', data=form, headers=headers)
    assert r.status_code == 200, r.get_json()
    return client.get(f"/api/schedule/progress/{r.get_json()['job_id']}").get_json()


def _probe(path):
    out = subprocess.run(['ffprobe', '-v', 'error', '-show_streams', '-of', 'json', str(path)],
                         capture_output=True, text=True, timeout=60).stdout
    streams = json.loads(out)['streams']
    return {s['codec_type']: s for s in streams}


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


def _diff(a, b):
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())


# ---- registration ----

def test_tab_is_registered_under_create_and_gated_by_its_own_permission(users_db):
    assert 'schedule_plug' in auth._PERMISSION_KEYS
    client, _ = _client()
    html = client.get('/').get_data(as_text=True)
    assert 'id="p-schedule"' in html
    assert html.index("switchTab('p-shorts',this)") < html.index("switchTab('p-schedule',this)") \
        < html.index("switchTab('p-music',this)"), 'in the Create group, after Vertical Shorts'

    ok, _err, gid = auth.group_create('Promo editors only')
    auth.group_set_permissions(gid, ['promo_generation'])
    auth.user_create('sched_editor', 'a-genuinely-long-password-1')
    uid = auth.user_get_by_username('sched_editor')['id']
    auth.user_set_group(uid, gid)
    client, headers = _client(user_id=uid, role='user', username='sched_editor')
    assert "switchTab('p-schedule',this)" not in client.get('/').get_data(as_text=True)
    for method, url in (('get', '/api/schedule/options'), ('get', '/api/schedule/items'),
                        ('post', '/api/schedule/render')):
        assert getattr(client, method)(url, headers=headers).status_code == 403, url
    auth.group_set_permissions(gid, ['promo_generation', 'schedule_plug'])
    assert "switchTab('p-schedule',this)" in client.get('/').get_data(as_text=True)
    assert client.get('/api/schedule/items').status_code == 200
    assert main.app.test_client().get('/api/schedule/items').status_code == 401


def test_options_and_the_artwork_network_folder():
    client, _ = _client()
    d = client.get('/api/schedule/options').get_json()
    assert d['ok'] and d['durations'] == [10, 15, 20, 30] and d['psd_supported'] is True
    assert [f['key'] for f in d['formats']] == list(pipeline.EXPORT_FORMATS)
    cat = pipeline._network_categories()['schedule']
    assert {'psd', 'png', 'jpg', 'tif'} <= cat['exts'] and 'mp4' not in cat['exts'] and 'fallback' not in cat
    assert 'schedule' in pipeline.NETWORK_CATEGORY_KEYS
    with pytest.raises(ValueError, match='Invalid filename'):
        pipeline.fetch_network_file('episode.mp4', 'schedule')


def test_render_refuses_bad_requests_before_starting_a_job(env, monkeypatch):
    client, headers = _client()
    started = []
    monkeypatch.setattr(schedule, '_spawn', lambda fn, *a, **k: started.append(a))

    def post(**form):
        return client.post('/api/schedule/render', data=form, headers=headers)
    for bad in ('12', '0', 'abc', '', '300'):
        r = post(schedule_image_network=env['psd'], duration=bad)
        assert r.status_code == 400 and '10, 15, 20, 30' in r.get_json()['error'], bad
    assert 'Pick the schedule artwork' in post(duration='15').get_json()['error']
    assert post(duration='15', schedule_image_network='net_1_gone.psd').status_code == 400
    assert post(duration='15', schedule_image_network='../../etc/passwd').status_code == 400
    r = post(duration='15', schedule_image_network=env['psd'], schedule_music_network='net_1_gone.wav')
    assert r.status_code == 400 and 'music is no longer available' in r.get_json()['error']
    # A direct upload is refused while the deployment only takes media from its network shares.
    r = client.post('/api/schedule/render', headers=headers, content_type='multipart/form-data',
                    data={'duration': '15', 'schedule_image': (io.BytesIO(b'x'), 'week.png')})
    assert r.status_code == 400
    assert started == []
    # A valid one: unknown format falls back to MP4, the prompt is tidied and capped.
    r = post(duration='20', schedule_image_network=env['psd'], delivery_format='betamax', prompt='  wipe \n in ' + 'x' * 900)
    assert r.status_code == 200 and len(started) == 1
    p = started[0][1]
    assert (p['duration'], p['format'], p['orig_name'], p['music']) == (20, 'mp4_high', 'Primetime_Week_42.psd', None)
    assert p['prompt'].startswith('wipe in x') and len(p['prompt']) == 600


# ---- the whole thing ----

def test_layered_psd_with_music_and_the_example_prompt_end_to_end(env, artwork):
    client, headers = _client(user_id=7, role='user', username='ana')
    job = _render(client, headers, schedule_image_network=env['psd'], schedule_music_network=env['music'],
                  duration='10', delivery_format='mp4_high', prompt='stop motion effect. wipe in reveal schedule.')
    assert job['error'] is None and job['done'] and job['percent'] == 100
    assert [s['label'] for s in job['stages']] == ['Reading artwork', 'Planning the animation', 'Drawing frames',
                                                   'Making the delivery file', 'Saving', 'Done']
    plug = job['result']['plug']
    assert plug['title'] == 'Primetime_Week_42.psd' and plug['file'] == 'Primetime_Week_42_schedule_10s.mp4'
    assert plug['duration'] == 10 and plug['format'] == 'mp4_high' and plug['layers'] == LAYERS and plug['layered']
    assert plug['animation'] == ('background: fade; 1 layer: fade; Schedule Mon: wipe right; '
                                 'Schedule Tue: wipe right; stop motion')
    assert plug['read_by'] == 'built-in' and plug['music'] == 'bed.wav' and plug['notes'] == []
    assert plug['preview_url'] is None, 'an MP4 plays in the browser as it is'

    pdir = os.path.join(schedule.SCHEDULE_DIR, plug['plug_id'])
    assert sorted(os.listdir(pdir)) == ['Primetime_Week_42_schedule_10s.mp4', 'plug.json', 'poster.jpg']
    st = _probe(os.path.join(pdir, plug['file']))
    assert (st['video']['codec_name'], st['video']['width'], st['video']['height']) == ('h264', 1920, 1080)
    assert st['video']['r_frame_rate'] == '30000/1001' and st['video']['pix_fmt'] == 'yuv420p'
    assert abs(float(st['video']['duration']) - 10.0) < 0.1 and abs(float(st['audio']['duration']) - 10.0) < 0.1
    assert st['audio']['channels'] == 2, 'a 4-second bed looped to cover 10'

    fr = _frames(os.path.join(pdir, plug['file']))
    assert len(fr) == 300
    assert fr[0].mean() < 20, 'opens on black'
    assert _diff(fr[-1], artwork['want']) < 3.0, 'ends on the artwork itself'
    assert _diff(fr[150], artwork['want']) < 3.0, 'and has been holding on it since well before half way'
    monday = lambda f: f[360, 950].min() > 180        # noqa: E731
    first = next(i for i, f in enumerate(fr) if monday(f))
    assert 20 < first < 120 and not monday(fr[first - 12]), 'the schedule rows arrive part-way in, not with the title'
    poster = cv2.imread(os.path.join(pdir, 'poster.jpg'))
    assert poster.shape[:2] == (360, 640) and _diff(poster, cv2.resize(artwork['want'], (640, 360))) < 6.0

    # Saved, served, private, deletable.
    assert [p['plug_id'] for p in client.get('/api/schedule/items').get_json()['items']] == [plug['plug_id']]
    r = client.get(plug['url'] + '?download=1')
    assert r.status_code == 200 and 'attachment' in r.headers['Content-Disposition'] and len(r.data) == plug['size']
    assert client.get(plug['poster_url']).status_code == 200
    assert client.get(f"/api/schedule/file/{plug['plug_id']}/plug.json").status_code == 404, 'only listed files'
    other, oh = _client(user_id=8, role='user', username='ben')
    assert other.get('/api/schedule/items').get_json()['items'] == []
    assert other.get(plug['url']).status_code == 404
    assert other.delete(f"/api/schedule/items/{plug['plug_id']}", headers=oh).status_code == 404
    admin, _ = _client()
    assert len(admin.get('/api/schedule/items').get_json()['items']) == 1, 'an admin sees everyone\'s'
    assert client.delete(f"/api/schedule/items/{plug['plug_id']}", headers=headers).get_json() == {'ok': True}
    assert os.listdir(schedule.SCHEDULE_DIR) == [] and client.get(plug['url']).status_code == 404


def test_flat_image_no_music_prores_at_its_own_frame_rate_with_a_browser_preview(env, artwork):
    client, headers = _client()
    job = _render(client, headers, schedule_image_network=env['png'], duration='10',
                  delivery_format='prores_hq_2398', prompt='wipe in from the left')
    assert job['error'] is None
    plug = job['result']['plug']
    assert not plug['layered'] and plug['layers'] == ['Image'] and plug['music'] is None
    assert plug['animation'] == 'picture: wipe right' and plug['file'].endswith('_schedule_10s.mov')
    pdir = os.path.join(schedule.SCHEDULE_DIR, plug['plug_id'])
    st = _probe(os.path.join(pdir, plug['file']))
    assert st['video']['codec_name'] == 'prores' and st['video']['r_frame_rate'] == '24000/1001'
    assert st['audio']['codec_name'] == 'pcm_s16le', 'silent, but there: broadcast files need an audio track'
    assert plug['preview_url'].endswith('/preview.mp4')
    pv = _probe(os.path.join(pdir, 'preview.mp4'))
    assert pv['video']['codec_name'] == 'h264' and pv['video']['width'] == 1280
    fr = _frames(os.path.join(pdir, 'preview.mp4'))
    assert len(fr) == 240 and _diff(fr[-1], cv2.resize(artwork['want'], (1280, 720))) < 4.0
    # Part-way through the wipe the left of the picture is there and the right is not yet.
    mid = next(f for f in fr if f[100, 200].sum() > 60)
    assert mid[100, 1100].sum() < 60
    assert not os.path.exists(os.path.join(main.app.config['UPLOAD_FOLDER'], f"schedmaster_{job['id']}.mov"))


def test_the_ai_model_refines_the_prompt_when_it_is_there_and_is_not_needed_when_it_is_not(env, monkeypatch):
    monkeypatch.setattr(schedule, 'SCHEDULE_USE_LLM', True)
    monkeypatch.setattr(pipeline, '_check_service', lambda name, url, path='/', timeout=3: {'status': 'up'})
    unloaded, asked = [], []
    monkeypatch.setattr(pipeline, 'unload_ollama_model', unloaded.append)

    def model(base_url, payload, timeout=180):
        asked.append(payload)
        return json.dumps({'stop_motion': False, 'default': {'effect': 'slide', 'direction': 'up'},
                           'layers': [{'name': 'Title', 'effect': 'pop'}, {'name': 'Sponsor', 'effect': 'wipe'}]}), {}
    monkeypatch.setattr(schedule.shorts_core, 'ollama_generate', model)
    client, headers = _client()
    plug = _render(client, headers, schedule_image_network=env['psd'], duration='10',
                   prompt='bring the title in with a bounce and float the rest up')['result']['plug']
    assert plug['read_by'].startswith('AI model (') and plug['notes'] == []
    assert plug['animation'] == 'background: fade; 2 layers: slide up; Title: pop'
    assert all(n in asked[0]['prompt'] for n in LAYERS) and 'float the rest up' in asked[0]['prompt']
    assert len(unloaded) == 1, 'the GPU is handed back, as after any other use of the model'

    # The model errors, or answers with something unusable: the words are still read, and the plug says how.
    for failure in (RuntimeError('model not found'), 'I would be happy to help!'):
        def broken(base_url, payload, timeout=180, failure=failure):
            if isinstance(failure, Exception):
                raise failure
            return failure, {}
        monkeypatch.setattr(schedule.shorts_core, 'ollama_generate', broken)
        plug = _render(client, headers, schedule_image_network=env['psd'], duration='10',
                       prompt='wipe in reveal schedule')['result']['plug']
        assert plug['read_by'] == 'built-in' and 'Schedule Mon: wipe right' in plug['animation']
        assert any('read by keyword matching' in n for n in plug['notes'])
    # No prompt: nothing to ask the model about.
    asked.clear()
    monkeypatch.setattr(schedule.shorts_core, 'ollama_generate', model)
    plug = _render(client, headers, schedule_image_network=env['psd'], duration='10')['result']['plug']
    assert asked == [] and plug['animation'] == 'background: fade; 3 layers: fade'


def test_artwork_that_cannot_be_used_ends_the_job_with_the_reason_and_leaves_nothing_behind(env):
    up = main.app.config['UPLOAD_FOLDER']
    bad = f'net_{int(time.time())}_broken.png'
    with open(os.path.join(up, bad), 'wb') as f:
        f.write(b'this is not a picture')
    try:
        client, headers = _client()
        job = _render(client, headers, schedule_image_network=bad, duration='15')
        assert job['done'] and 'could not be read' in job['error']
        assert os.listdir(schedule.SCHEDULE_DIR) == []
    finally:
        os.remove(os.path.join(up, bad))


def test_a_failed_encode_reports_ffmpegs_reason_and_removes_the_half_made_plug(env, monkeypatch):
    def boom(*a, **k):
        raise sk.EncodeError('Unknown encoder libx264')
    monkeypatch.setattr(sk, 'encode', boom)
    client, headers = _client()
    job = _render(client, headers, schedule_image_network=env['png'], duration='10')
    assert 'could not be encoded: Unknown encoder libx264' in job['error']
    assert os.listdir(schedule.SCHEDULE_DIR) == []


def test_send_to_a_video_destination(env, monkeypatch):
    client, headers = _client()
    plug = _render(client, headers, schedule_image_network=env['png'], duration='10')['result']['plug']
    dests = {1: {'id': 1, 'name': 'Playout inbox', 'delivery_kind': 'video'},
             2: {'id': 2, 'name': 'Edit bay', 'delivery_kind': 'fcpxml'}}
    monkeypatch.setattr(schedule, 'network_destination_get', lambda i: dests.get(i))
    sent = []
    monkeypatch.setattr(pipeline, 'send_file_to_network_destination', lambda local, name, dest: sent.append((os.path.basename(local), name, dest['name'])))
    url = f"/api/schedule/items/{plug['plug_id']}/send"
    assert client.post(url, json={'destination_id': 9}, headers=headers).status_code == 400
    r = client.post(url, json={'destination_id': 2}, headers=headers)
    assert r.status_code == 400 and 'not finished video' in r.get_json()['error'].replace('not\n', 'not ')
    r = client.post(url, json={'destination_id': 1}, headers=headers).get_json()
    assert r == {'ok': True, 'sent': [plug['file']], 'destination': 'Playout inbox'}
    assert sent == [(plug['file'], plug['file'], 'Playout inbox')]

    def unreachable(local, name, dest):
        raise ValueError('Could not reach \\\\playout\\inbox')
    monkeypatch.setattr(pipeline, 'send_file_to_network_destination', unreachable)
    r = client.post(url, json={'destination_id': 1}, headers=headers)
    assert r.status_code == 502 and 'Could not reach' in r.get_json()['error']


def test_leftovers_from_an_interrupted_process_are_cleared_on_start(tmp_path, monkeypatch):
    root = tmp_path / 'plugs'
    monkeypatch.setattr(schedule, 'SCHEDULE_DIR', str(root))
    for name in ('1700000000_aaaaaa', '1700000001_bbbbbb', '.deleting_1700000002_cccccc_ab', 'not-a-plug'):
        (root / name).mkdir(parents=True)
    (root / '1700000001_bbbbbb' / 'plug.json').write_text('{"plug_id": "1700000001_bbbbbb", "file": "x.mp4"}')
    schedule.settle_interrupted()
    assert sorted(os.listdir(root)) == ['1700000001_bbbbbb', 'not-a-plug'], \
        'a finished plug and anything that is not ours are left alone'

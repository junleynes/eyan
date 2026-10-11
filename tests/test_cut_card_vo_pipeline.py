"""Clean cuts, card fit and script-aware narration inside the pipeline.

cutcraft.py has the rules (tests/test_cutcraft.py); these tests cover the part
that needs ffmpeg: measuring a card, cutting the narration, and the job itself
reporting what it did.
"""
import shutil
import subprocess
import time
import unittest.mock as mock
from io import BytesIO

import pytest

with mock.patch('requests.post'), mock.patch('requests.get'):
    import main
    import pipeline

import cutcraft

pytestmark = pytest.mark.skipif(shutil.which('ffmpeg') is None, reason='ffmpeg not available')


def _run(*args):
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', *args], check=True, timeout=120)


@pytest.fixture(scope='module')
def settled_card(tmp_path_factory):
    """4 s card: 1.5 s of motion, then a still hold; a 1 s voice then silence."""
    d = tmp_path_factory.mktemp('card_settled')
    move, still = d / 'move.mp4', d / 'still.mp4'
    _run('-f', 'lavfi', '-i', 'testsrc2=s=320x240:d=1.5:r=25', '-pix_fmt', 'yuv420p', str(move))
    _run('-f', 'lavfi', '-i', 'color=c=black:s=320x240:d=2.5:r=25', '-pix_fmt', 'yuv420p', str(still))
    video = d / 'v.mp4'
    lst = d / 'l.txt'
    lst.write_text(f"file '{move}'\nfile '{still}'\n")
    _run('-f', 'concat', '-safe', '0', '-i', str(lst), '-c', 'copy', str(video))
    out = d / 'card.mp4'
    _run('-i', str(video), '-f', 'lavfi', '-i', 'aevalsrc=if(lt(t\\,1)\\,0.5*sin(2*PI*440*t)\\,0):d=4:s=44100',
         '-c:v', 'copy', '-c:a', 'aac', '-shortest', str(out))
    return out


@pytest.fixture(scope='module')
def moving_card(tmp_path_factory):
    d = tmp_path_factory.mktemp('card_moving')
    out = d / 'card.mp4'
    _run('-f', 'lavfi', '-i', 'testsrc2=s=320x240:d=3:r=25', '-pix_fmt', 'yuv420p', '-an', str(out))
    return out


def test_card_measure_finds_where_a_card_settles(settled_card):
    dur = pipeline.probe_duration(str(settled_card))
    floor, det = pipeline._card_measure(str(settled_card), dur)
    # picture settles at ~1.5 s (+0.4 s beat) and the voice ends at 1.0 s (+0.25 s)
    assert 1.7 <= floor <= 2.4, (floor, det)
    assert det['still_from'] is not None and 1.2 <= det['still_from'] <= 1.8
    assert det['voice_end'] is not None and 0.8 <= det['voice_end'] <= 1.3


def test_card_that_never_settles_is_not_shortened(moving_card):
    dur = pipeline.probe_duration(str(moving_card))
    floor, _ = pipeline._card_measure(str(moving_card), dur)
    assert floor == dur


def test_shrinking_a_card_with_audio_fades_the_new_end(tmp_path, settled_card):
    out = tmp_path / 'shrunk.mp4'
    assert pipeline._adjust_card_duration(str(settled_card), -1.5, str(out))
    assert abs(pipeline.probe_duration(str(out)) - 2.5) < 0.25


# ---------------------------------------------------------------- narration

VO_WORDS = [{'word': 'Abangan', 'start': 0.5, 'end': 1.0}, {'word': 'ngayong', 'start': 1.05, 'end': 1.6},
            {'word': 'gabi.', 'start': 1.65, 'end': 2.1}, {'word': 'Only', 'start': 3.0, 'end': 3.3},
            {'word': 'on', 'start': 3.35, 'end': 3.5}, {'word': 'GMA.', 'start': 3.55, 'end': 4.1}]
VO_SEGS = [{'start': 0.5, 'end': 2.1, 'text': 'Abangan ngayong gabi.'}, {'start': 3.0, 'end': 4.1, 'text': 'Only on GMA.'}]


@pytest.fixture
def vo_file(tmp_path):
    p = tmp_path / 'vo.wav'
    _run('-f', 'lavfi', '-i', 'sine=frequency=300:duration=6', '-ac', '1', '-ar', '44100', str(p))
    return str(p)


@pytest.fixture
def fake_whisper(monkeypatch):
    monkeypatch.setattr(pipeline, 'transcribe_audio_file',
                        lambda path, trim_start=0.0, trim_end=None: (list(VO_WORDS), list(VO_SEGS)))


def test_vo_plan_timed_script_is_edited(vo_file, fake_whisper):
    plan = pipeline._build_vo_plan(vo_file, '00:02 Abangan ngayong gabi\n00:06 Only on GMA', 0.0, None, 0.0, 14.85)
    assert plan['edited'] and plan['timed'] and plan['fits']
    assert [round(p['at']) for p in plan['pieces']] == [2, 6]
    summary = pipeline._vo_plan_summary(plan)
    assert summary['lines'][0]['status'] == 'ok' and summary['lines'][1]['at'] == pytest.approx(6.0, abs=0.01)


def test_vo_plan_without_times_plays_the_recording_untouched_when_it_fits(vo_file, fake_whisper):
    plan = pipeline._build_vo_plan(vo_file, 'Abangan ngayong gabi. Only on GMA.', 0.0, None, 0.0, 14.85)
    assert not plan['edited'] and plan['scripted']


def test_vo_plan_edits_a_recording_that_overruns_the_plug(vo_file, fake_whisper):
    # the file is 6 s; start at 2 s and the plug allows only up to 5 s: it overruns
    plan = pipeline._build_vo_plan(vo_file, '', 0.0, None, 2.0, 5.0)
    assert plan['overruns'] and plan['edited']
    ends = [(w['start'], w['end']) for w in VO_WORDS]
    for p in plan['pieces']:
        assert cutcraft.classify_cut('out', p['b'], ends) != 'clipped'
    assert plan['end'] <= 5.0 + 0.02


def test_render_vo_from_plan_builds_the_edited_audio(tmp_path, vo_file, fake_whisper):
    plan = pipeline._build_vo_plan(vo_file, '00:02 Abangan ngayong gabi\n00:06 Only on GMA', 0.0, None, 0.0, 14.85)
    out = pipeline._render_vo_from_plan(plan, vo_file, str(tmp_path / 'edit.wav'))
    assert out
    assert pipeline.probe_duration(out) == pytest.approx(plan['end'], abs=0.15)


def test_vo_plan_is_none_without_speech(vo_file, monkeypatch):
    monkeypatch.setattr(pipeline, 'transcribe_audio_file', lambda *a, **k: ([], []))
    assert pipeline._build_vo_plan(vo_file, 'x', 0.0, None, 0.0, 10) is None


# ---------------------------------------------------------------- the job

_COLORS = ['red', 'blue', 'green', 'yellow', 'purple', 'cyan', 'orange', 'white',
           'pink', 'gray', 'brown', 'gold']


@pytest.fixture(scope='module')
def scenes_video(tmp_path_factory):
    d = tmp_path_factory.mktemp('cut_scenes')
    parts = []
    for i, c in enumerate(_COLORS):
        part = d / f'p{i}.mp4'
        _run('-f', 'lavfi', '-i', f'color=c={c}:s=320x240:d=2.5:r=25', '-pix_fmt', 'yuv420p', str(part))
        parts.append(part)
    lst = d / 'l.txt'
    lst.write_text('\n'.join(f"file '{p}'" for p in parts))
    out = d / 'combined.mp4'
    _run('-f', 'concat', '-safe', '0', '-i', str(lst), '-c', 'copy', str(out))
    return out


@pytest.fixture
def authed_client(tmp_path, monkeypatch):
    app = main.app
    up = tmp_path / 'uploads'
    up.mkdir()
    monkeypatch.setitem(app.config, 'UPLOAD_FOLDER', str(up))
    monkeypatch.setattr(main.pipeline, 'ALLOW_LOCAL_MEDIA_UPLOAD', True)
    client = app.test_client()
    with client.session_transaction() as sess:
        sess.update(authed=True, user_id=1, username='admin', role='admin', csrf_token='tok')
    return client, {'X-CSRF-Token': 'tok'}


def _poll(client, job_id, headers, timeout_s=120):
    end = time.time() + timeout_s
    while time.time() < end:
        d = client.get(f'/api/trailer/progress/{job_id}', headers=headers).get_json()
        if d.get('done'):
            return d
        time.sleep(1)
    pytest.fail('job did not finish')


def _preview(client, headers, video, **extra):
    data = {'file': (BytesIO(video.read_bytes()), 'combined.mp4'), 'genre': '', 'trailer_length': '15',
            'scoring_mode': 'none', 'sfx_mode': 'none', 'vo_mode': 'none', 'transition': 'cut',
            'preview_only': '1', 'sync_beats': '', 'min_scene_len': '0.4'}
    data.update(extra)
    r = client.post('/api/trailer/generate', data=data, headers=headers, content_type='multipart/form-data')
    assert r.status_code == 200, r.get_data(as_text=True)
    d = _poll(client, r.get_json()['job_id'], headers)
    assert d.get('error') is None, d.get('error')
    return d['result']


def test_review_reports_how_clean_each_cut_is(authed_client, scenes_video, monkeypatch):
    client, headers = authed_client
    # one short word near the middle of every 2.5 s scene
    words = [{'word': 'hey', 'start': 0.6 + 2.5 * i, 'end': 1.1 + 2.5 * i} for i in range(12)]
    segs = [{'start': w['start'], 'end': w['end'], 'text': 'hey'} for w in words]
    monkeypatch.setattr(pipeline, 'transcribe_video', lambda path: (words, segs))
    result = _preview(client, headers, scenes_video)
    info = result['cut_info']
    assert info['checked'] and info['cleaned']
    assert info['edges'] > 0 and info['clipped'] == 0
    assert all(s['cut'] and s['cut']['in'] in ('clean', 'silent', 'word', 'tight')
               for s in result['scenes'])


def test_review_without_speech_data_does_not_claim_a_check(authed_client, scenes_video, monkeypatch):
    client, headers = authed_client
    monkeypatch.setattr(pipeline, 'transcribe_video', lambda path: ([], []))
    result = _preview(client, headers, scenes_video)
    assert result['cut_info']['checked'] is False
    assert all(s['cut'] is None for s in result['scenes'])


@pytest.fixture(scope='module')
def long_static_card(tmp_path_factory):
    """6 s card: 1 s of motion then a 5 s hold."""
    d = tmp_path_factory.mktemp('long_card')
    move, still, lst = d / 'm.mp4', d / 's.mp4', d / 'l.txt'
    _run('-f', 'lavfi', '-i', 'testsrc2=s=320x240:d=1:r=25', '-pix_fmt', 'yuv420p', str(move))
    _run('-f', 'lavfi', '-i', 'color=c=black:s=320x240:d=5:r=25', '-pix_fmt', 'yuv420p', str(still))
    lst.write_text(f"file '{move}'\nfile '{still}'\n")
    out = d / 'card.mp4'
    _run('-f', 'concat', '-safe', '0', '-i', str(lst), '-c', 'copy', '-an', str(out))
    return out


def test_long_cards_are_shortened_so_scenes_have_room(authed_client, scenes_video, long_static_card, monkeypatch):
    client, headers = authed_client
    monkeypatch.setattr(pipeline, 'transcribe_video', lambda path: ([], []))
    data = dict(end_card_video=(BytesIO(long_static_card.read_bytes()), 't.mp4'),
                schedule_video=(BytesIO(long_static_card.read_bytes()), 'e.mp4'))
    result = _preview(client, headers, scenes_video, **data)
    fit = result['card_fit']
    assert fit['policy'] == 'fit'
    secs = [c['secs'] for c in fit['cards']]
    assert len(secs) == 2 and all(s < 6.0 for s in secs)          # both were 6 s
    assert fit['scene_budget'] >= cutcraft.min_scene_time(15) - 0.3
    assert any('shortened' in n for n in fit['notes'])


def test_card_fit_off_keeps_cards_as_they_were(authed_client, scenes_video, long_static_card, monkeypatch):
    client, headers = authed_client
    monkeypatch.setattr(pipeline, 'transcribe_video', lambda path: ([], []))
    result = _preview(client, headers, scenes_video, card_fit='off',
                      end_card_video=(BytesIO(long_static_card.read_bytes()), 't.mp4'),
                      schedule_video=(BytesIO(long_static_card.read_bytes()), 'e.mp4'))
    assert result['card_fit']['policy'] == 'off'
    assert all(c['secs'] >= 5.0 for c in result['card_fit']['cards'])


def test_review_shows_narration_cuts_and_the_render_uses_them(authed_client, scenes_video, vo_file,
                                                              fake_whisper, monkeypatch):
    client, headers = authed_client
    monkeypatch.setattr(pipeline, 'transcribe_video', lambda path: ([], []))
    seen = {}
    real = pipeline._render_vo_from_plan

    def spy(plan, path, out):
        seen['plan'] = plan
        return real(plan, path, out)

    monkeypatch.setattr(pipeline, '_render_vo_from_plan', spy)
    result = _preview(client, headers, scenes_video, vo_mode='upload',
                      vo_upload=(BytesIO(open(vo_file, 'rb').read()), 'vo.wav'),
                      vo_text='00:02 Abangan ngayong gabi\n00:06 Only on GMA')
    vp = result['vo_plan']
    assert vp and vp['edited'] and vp['timed'] and vp['fits']
    assert [round(l['at']) for l in vp['lines']] == [2, 6]

    r = client.post('/api/trailer/render', data={'preview_id': result['preview_id']}, headers=headers)
    assert r.status_code == 200, r.get_data(as_text=True)
    done = _poll(client, r.get_json()['job_id'], headers, timeout_s=240)
    assert done.get('error') is None, done.get('error')
    assert seen['plan']['edited']                      # the render built the narration from the reviewed plan

"""
User-requested: if AI Vision (Ollama) or faster-whisper is unreachable,
generating a promo plug should not be available at all -- not silently
degrade to generic OpenCV-only scoring with no mid-word cut protection,
which is what used to happen (every per-scene AI/whisper call failing the
same way, falling back to a neutral score, with nothing surfacing to the
user that scoring/cut-safety never actually ran).

_run_trailer_job now checks both services' reachability up front (before
any scene detection work starts) and fails the job immediately with a
clear reason if either is down -- unless the job is a pure timecode-cue
job, which uses neither service at all (build_cue_clips reads only the
script's own timecodes).

The rest of the suite assumes services are unreachable in this sandbox
(there's no real Ollama/whisper here) and depends on the old graceful-
degradation behavior to reach the logic it's actually testing -- see
conftest.py's autouse _assume_ai_services_reachable fixture, which stubs
the new preflight check itself (not the underlying per-call requests) so
every other test is unaffected. This file overrides that stub per-test to
exercise the preflight check directly.
"""
import shutil
import subprocess
import unittest.mock as mock
from io import BytesIO

import pytest

with mock.patch('requests.post'), mock.patch('requests.get'):
    import main
    import pipeline


def _ffmpeg_available():
    return shutil.which('ffmpeg') is not None


@pytest.fixture(scope='module')
def sample_video(tmp_path_factory):
    if not _ffmpeg_available():
        pytest.skip('ffmpeg not available in this environment')
    tmp = tmp_path_factory.mktemp('preflight_fixture')
    path = tmp / 'sample.mp4'
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'lavfi',
                    '-i', 'color=c=red:s=320x240:d=25:r=25',
                    '-f', 'lavfi', '-i', 'sine=frequency=300:duration=25',
                    '-c:v', 'libx264', '-c:a', 'aac', '-shortest', str(path)],
                   check=True, timeout=60)
    return path


@pytest.fixture
def authed_client(tmp_path, monkeypatch):
    app = main.app
    upload_dir = tmp_path / 'uploads'
    upload_dir.mkdir()
    monkeypatch.setitem(app.config, 'UPLOAD_FOLDER', str(upload_dir))
    monkeypatch.setattr(main.pipeline, 'ALLOW_LOCAL_MEDIA_UPLOAD', True)
    client = app.test_client()
    csrf_token = 'test-csrf-service-preflight'
    with client.session_transaction() as sess:
        sess['authed'] = True
        sess['user_id'] = 1
        sess['username'] = 'admin'
        sess['role'] = 'admin'
        sess['csrf_token'] = csrf_token
    return client, {'X-CSRF-Token': csrf_token}


def _down(name, url, path='/', timeout=3):
    return {'name': name, 'url': url, 'status': 'down', 'error': 'connection refused'}


def _up(name, url, path='/', timeout=3):
    return {'name': name, 'url': url, 'status': 'up'}


def _poll(client, job_id, headers, timeout_s=30):
    import time
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        d = client.get(f'/api/trailer/progress/{job_id}', headers=headers).get_json()
        if d.get('done'):
            return d
        time.sleep(0.5)
    pytest.fail(f'job {job_id} did not finish within {timeout_s}s')


def _generate(client, headers, video_path, extra=None):
    data = {
        'file': (BytesIO(video_path.read_bytes()), 'sample.mp4'),
        'genre': '', 'trailer_length': '15', 'scoring_mode': 'none',
        'sfx_mode': 'none', 'vo_mode': 'none', 'transition': 'cut',
        'preview_only': '1', 'sync_beats': '',
    }
    data.update(extra or {})
    return client.post('/api/trailer/generate', data=data, headers=headers, content_type='multipart/form-data')


def test_generation_is_blocked_when_ollama_is_unreachable(authed_client, sample_video, monkeypatch):
    monkeypatch.setattr(pipeline, '_check_service', _down)
    client, headers = authed_client
    r = _generate(client, headers, sample_video)
    assert r.status_code == 200, r.get_data(as_text=True)
    d = _poll(client, r.get_json()['job_id'], headers)
    assert d.get('error') is not None
    assert 'AI Vision' in d['error']
    assert 'Ollama' in d['error']


def test_generation_is_blocked_when_whisper_is_unreachable(authed_client, sample_video, monkeypatch):
    # Ollama reachable, whisper is not -- either one alone should be enough
    # to block, since cut-safety depends on whisper regardless of scoring.
    def _selective(name, url, path='/', timeout=3):
        return _up(name, url) if name == 'ollama' else _down(name, url)
    monkeypatch.setattr(pipeline, '_check_service', _selective)
    client, headers = authed_client
    r = _generate(client, headers, sample_video)
    assert r.status_code == 200, r.get_data(as_text=True)
    d = _poll(client, r.get_json()['job_id'], headers)
    assert d.get('error') is not None
    assert 'Speech-to-text' in d['error'] or 'whisper' in d['error'].lower()


def test_generation_proceeds_when_both_services_are_reachable(authed_client, sample_video, monkeypatch):
    monkeypatch.setattr(pipeline, '_check_service', _up)
    client, headers = authed_client
    r = _generate(client, headers, sample_video)
    assert r.status_code == 200, r.get_data(as_text=True)
    d = _poll(client, r.get_json()['job_id'], headers)
    # No real Ollama/whisper server is actually running, so the individual
    # AI/transcription calls made further into the job still fail -- but
    # that's the existing, unrelated per-call degradation this test isn't
    # about. The point here is narrower: the preflight check itself must
    # not block the job when it reports both services reachable.
    assert d.get('error') is None or 'unreachable' not in d.get('error', '')


def test_timecode_cue_jobs_are_never_blocked_by_the_preflight_check(authed_client, sample_video, monkeypatch):
    # A pure cue-driven job never calls either service (build_cue_clips
    # only reads the script's own timecodes), so requiring them here would
    # block a job that was never going to use them.
    monkeypatch.setattr(pipeline, '_check_service', _down)
    client, headers = authed_client
    r = _generate(client, headers, sample_video, extra={
        'manual_cues': '[{"material": 1, "start": "0:01", "end": "0:03"}]',
        'materials_count': '1',
    })
    assert r.status_code == 200, r.get_data(as_text=True)
    d = _poll(client, r.get_json()['job_id'], headers)
    assert d.get('error') is None or 'unreachable' not in (d.get('error') or ''), d.get('error')

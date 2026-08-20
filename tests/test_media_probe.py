"""
Tests for /api/media/probe -- the fix for a real reported bug: the HIRES
file picker's preview relied on the BROWSER decoding a staged video to read
its duration/resolution and show a frame. Real broadcast masters are very
often ProRes/DNxHD/MXF, which no mainstream browser decodes natively, so
that preview would hang on "reading video info..." forever with a
permanently black frame -- not occasionally slow, never resolving at all.

This endpoint reads duration/resolution server-side via get_video_info()
(OpenCV's own ffmpeg backend, not the browser) and extracts a real thumbnail
frame via ffmpeg, so the preview no longer depends on what the requesting
browser happens to be able to decode.

Uses pipeline.app (the same shared import every other test file in this
suite uses via conftest.py) rather than importing main fresh -- doing that
here collided with conftest.py's own earlier `import pipeline`, since
main.py re-imports pipeline internally and picks up its already-initialized
module rather than a clean one, which broke the login flow specifically in
this nested-import context (confirmed by the same login working correctly
standalone, outside the tests/ directory, and failing only here). Session
is set directly via session_transaction() rather than a real POST to
/login, sidestepping that whole admin-bootstrap/login chain for something
this test doesn't actually need to exercise.
"""
import os
import shutil
import subprocess
import unittest.mock as mock

import pytest

import pipeline


def _ffmpeg_available():
    return shutil.which('ffmpeg') is not None


@pytest.fixture
def probe_client(tmp_path, monkeypatch):
    """An authenticated test client against the real pipeline.app, with
    UPLOAD_FOLDER redirected to a fresh temp dir so this test's staged
    files/thumbnails never collide with anything else. Sets the CSRF token
    directly in the session (matching ensure_csrf_token()'s own storage key)
    alongside the auth session keys, and returns the matching header every
    POST needs to send -- _csrf_protect() rejects any non-GET request from
    an authenticated session that doesn't carry it, which a directly-set
    session (skipping the normal page load that would embed it) doesn't
    have without this."""
    upload_dir = tmp_path / 'uploads'
    upload_dir.mkdir()
    monkeypatch.setitem(pipeline.app.config, 'UPLOAD_FOLDER', str(upload_dir))
    client = pipeline.app.test_client()
    csrf_token = 'test-csrf-token-for-probe-tests'
    with client.session_transaction() as sess:
        sess['authed'] = True
        sess['user_id'] = 1
        sess['username'] = 'admin'
        sess['role'] = 'admin'
        sess['csrf_token'] = csrf_token
    return client, str(upload_dir), {'X-CSRF-Token': csrf_token}


@pytest.fixture
def staged_normal_mp4(tmp_path):
    """A small, ordinary H.264 file -- browsers CAN decode this one fine;
    confirms the probe endpoint works for the easy case too, not just the
    broken-in-browsers one."""
    if not _ffmpeg_available():
        pytest.skip('ffmpeg not available in this environment')
    path = tmp_path / 'normal.mp4'
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'lavfi',
                    '-i', 'color=c=green:s=320x240:d=3:r=25', '-c:v', 'libx264', str(path)],
                   check=True, timeout=30)
    return path


@pytest.fixture
def staged_prores_mov(tmp_path):
    """A genuine ProRes .mov -- the exact codec family real broadcast HIRES
    masters commonly use, and the one browsers do not decode natively. This
    is the actual regression test for the reported bug: get_video_info()
    goes through OpenCV's own ffmpeg backend, not the browser, so it must
    read this correctly where a <video> element's onloadedmetadata never
    would have fired at all."""
    if not _ffmpeg_available():
        pytest.skip('ffmpeg not available in this environment')
    path = tmp_path / 'prores.mov'
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'lavfi',
                    '-i', 'color=c=blue:s=320x240:d=3:r=25',
                    '-c:v', 'prores_ks', '-profile:v', '3', str(path)],
                   check=True, timeout=30)
    return path


def test_probe_reads_a_normal_browser_friendly_file(probe_client, staged_normal_mp4):
    client, upload_dir, headers = probe_client
    staged_name = 'net_1000_normal.mp4'
    shutil.copy(staged_normal_mp4, os.path.join(upload_dir, staged_name))

    r = client.post('/api/media/probe', data={'filename': staged_name}, headers=headers)
    d = r.get_json()
    assert d['ok'] is True
    assert 2.5 < d['duration'] < 3.5
    assert d['width'] == 320 and d['height'] == 240
    assert d['thumbnail_url'].startswith('/uploads/thumb_')


def test_probe_reads_a_prores_file_browsers_cannot_decode(probe_client, staged_prores_mov):
    # The actual bug this endpoint exists to fix: confirm probing succeeds
    # on a file that is, by design, impossible for a browser's own <video>
    # element to read metadata from at all.
    client, upload_dir, headers = probe_client
    staged_name = 'net_1001_prores.mov'
    shutil.copy(staged_prores_mov, os.path.join(upload_dir, staged_name))

    r = client.post('/api/media/probe', data={'filename': staged_name}, headers=headers)
    d = r.get_json()
    assert d['ok'] is True
    assert 2.5 < d['duration'] < 3.5
    assert d['width'] == 320 and d['height'] == 240

    thumb_name = d['thumbnail_url'].split('/')[-1]
    thumb_path = os.path.join(upload_dir, thumb_name)
    assert os.path.exists(thumb_path)
    assert os.path.getsize(thumb_path) > 0


def test_probe_thumbnail_is_cached_not_regenerated(probe_client, staged_prores_mov):
    client, upload_dir, headers = probe_client
    staged_name = 'net_1002_prores.mov'
    shutil.copy(staged_prores_mov, os.path.join(upload_dir, staged_name))

    r1 = client.post('/api/media/probe', data={'filename': staged_name}, headers=headers)
    thumb1 = r1.get_json()['thumbnail_url']
    mtime1 = os.path.getmtime(os.path.join(upload_dir, thumb1.split('/')[-1]))

    r2 = client.post('/api/media/probe', data={'filename': staged_name}, headers=headers)
    thumb2 = r2.get_json()['thumbnail_url']
    mtime2 = os.path.getmtime(os.path.join(upload_dir, thumb2.split('/')[-1]))

    assert thumb1 == thumb2
    assert mtime1 == mtime2  # not rewritten on the second call


def test_probe_missing_file_returns_clean_404(probe_client):
    client, upload_dir, headers = probe_client
    r = client.post('/api/media/probe', data={'filename': 'net_9999_never_staged.mp4'}, headers=headers)
    assert r.status_code == 404
    assert r.get_json()['ok'] is False


def test_probe_rejects_path_traversal(probe_client):
    client, upload_dir, headers = probe_client
    r = client.post('/api/media/probe', data={'filename': '../../etc/passwd'}, headers=headers)
    d = r.get_json()
    assert d['ok'] is False

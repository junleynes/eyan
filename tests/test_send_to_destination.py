"""
Tests for "Send to destination" -- copies a finished render straight to a
configured network share (e.g. a Vantage watch folder) instead of only
being downloadable through the browser. Covers the new 'destination'
network category, send_file_to_network_destination()'s SMB write, and the
/library/<id>/send-to-destination route end to end.

Also covers a real bug found and fixed while building this: the job's
persisted result never included library_id, since job_set(...,result=result)
snapshots the dict (via JSON serialization) at call time, before
library_add() had a chance to set library_id on it -- meaning any feature
reading library_id from a job's stored result (this one included) would
never see it. The existing Download button was never visibly affected only
because it already had a filename-based fallback for a missing library_id;
Send to destination has no such fallback, which is what surfaced the bug.
"""
import os
import re
import shutil
import subprocess
import time
import unittest.mock as mock
from io import BytesIO

import pytest

import library_db
import pipeline

with mock.patch('requests.post'), mock.patch('requests.get'):
    import main


def test_destination_is_a_valid_network_category():
    assert 'destination' in library_db.NETWORK_CATEGORY_KEYS


def test_destination_folder_can_be_saved_and_loaded(tmp_path, monkeypatch):
    monkeypatch.setattr(library_db, 'NETWORK_FOLDERS_FILE', str(tmp_path / 'nf.json'))
    ok, err = library_db.save_network_folder('destination', {'path': '\\\\vantage\\ingest', 'username': 'svc'})
    assert ok is True
    folders = library_db.load_network_folders()
    assert folders['destination']['path'] == '\\\\vantage\\ingest'


def test_send_without_a_configured_destination_raises_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.setattr(library_db, 'NETWORK_FOLDERS_FILE', str(tmp_path / 'nf.json'))
    with pytest.raises(ValueError, match='Config > Network'):
        pipeline.send_file_to_network_destination(str(tmp_path / 'x.mp4'), 'x.mp4')


def test_send_writes_to_the_configured_share_with_the_right_credentials(tmp_path, monkeypatch):
    monkeypatch.setattr(library_db, 'NETWORK_FOLDERS_FILE', str(tmp_path / 'nf.json'))
    library_db.save_network_folder('destination', {'path': '\\\\vantage\\ingest', 'username': 'svc_prism', 'password': 'secret'})
    local_file = tmp_path / 'render.mp4'
    local_file.write_bytes(b'fake video bytes')

    captured = {}
    def fake_open_file(path, mode=None, **kw):
        captured['path'] = path
        captured['mode'] = mode
        return BytesIO()
    def fake_register_session(host, username=None, password=None, **kw):
        captured['host'] = host
        captured['username'] = username
        captured['password'] = password

    with mock.patch('pipeline.smbclient.open_file', side_effect=fake_open_file), \
         mock.patch('pipeline.smbclient.register_session', side_effect=fake_register_session):
        pipeline.send_file_to_network_destination(str(local_file), 'render.mp4')

    assert captured['host'] == 'vantage'
    assert captured['username'] == 'svc_prism'
    assert captured['password'] == 'secret'
    assert captured['path'] == '\\\\vantage\\ingest\\render.mp4'
    assert captured['mode'] == 'wb'


def test_send_wraps_smb_failures_as_a_clean_value_error(tmp_path, monkeypatch):
    monkeypatch.setattr(library_db, 'NETWORK_FOLDERS_FILE', str(tmp_path / 'nf.json'))
    library_db.save_network_folder('destination', {'path': '\\\\vantage\\ingest'})
    local_file = tmp_path / 'render.mp4'
    local_file.write_bytes(b'fake')
    with mock.patch('pipeline.smbclient.register_session'), \
         mock.patch('pipeline.smbclient.open_file', side_effect=Exception('connection refused')):
        with pytest.raises(ValueError, match='Could not write to the destination share'):
            pipeline.send_file_to_network_destination(str(local_file), 'render.mp4')


# ---- End-to-end: real render -> real library_id -> real route ----

def _ffmpeg_available():
    return shutil.which('ffmpeg') is not None


@pytest.fixture
def rendered_trailer(tmp_path, monkeypatch):
    """A real completed render, saved to the library, via the actual route
    -- not constructed by hand -- so this exercises the real
    job_set/library_add sequence the library_id bug lived in."""
    if not _ffmpeg_available():
        pytest.skip('ffmpeg not available in this environment')
    app = main.app
    upload_dir = tmp_path / 'uploads'
    upload_dir.mkdir()
    monkeypatch.setitem(app.config, 'UPLOAD_FOLDER', str(upload_dir))
    monkeypatch.setattr(pipeline, 'ALLOW_LOCAL_MEDIA_UPLOAD', True)

    client = app.test_client()
    csrf_token = 'test-csrf-send-dest'
    with client.session_transaction() as sess:
        sess['authed'] = True
        sess['user_id'] = 1
        sess['username'] = 'admin'
        sess['role'] = 'admin'
        sess['csrf_token'] = csrf_token
    headers = {'X-CSRF-Token': csrf_token}

    src = tmp_path / 'src.mp4'
    parts = []
    for i, color in enumerate(['red', 'blue', 'green', 'yellow', 'purple', 'cyan']):
        part = tmp_path / f'part{i}.mp4'
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'lavfi',
                        '-i', f'color=c={color}:s=320x240:d=6:r=25',
                        '-f', 'lavfi', '-i', f'sine=frequency={200 + i * 100}:duration=6',
                        '-c:v', 'libx264', '-c:a', 'aac', '-shortest', str(part)], check=True, timeout=30)
        parts.append(part)
    list_file = tmp_path / 'list.txt'
    list_file.write_text('\n'.join(f"file '{p}'" for p in parts))
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0',
                    '-i', str(list_file), '-c', 'copy', str(src)], check=True, timeout=30)

    r = client.post('/api/trailer/generate', data={
        'file': (BytesIO(src.read_bytes()), 'src.mp4'),
        'genre': '', 'trailer_length': '15', 'scoring_mode': 'none',
        'sfx_mode': 'none', 'vo_mode': 'none', 'transition': 'cut',
    }, headers=headers, content_type='multipart/form-data')
    assert r.status_code == 200, r.get_data(as_text=True)
    job_id = r.get_json()['job_id']

    deadline = time.time() + 60
    d = None
    while time.time() < deadline:
        d = client.get(f'/api/trailer/progress/{job_id}', headers=headers).get_json()
        if d.get('done'):
            break
        time.sleep(1)
    assert d and d.get('error') is None, (d or {}).get('error')
    return client, headers, (d.get('result') or {})


def test_real_render_has_a_library_id_in_its_stored_result(rendered_trailer):
    # The actual regression check for the job_set/library_add ordering bug:
    # a real render's PERSISTED result (what the frontend reads back via
    # /api/trailer/progress) must include library_id, not just the
    # in-memory dict at the moment library_add() happened to run.
    client, headers, result = rendered_trailer
    assert result.get('library_id') is not None


def test_send_to_destination_route_end_to_end(rendered_trailer, tmp_path, monkeypatch):
    client, headers, result = rendered_trailer
    library_id = result['library_id']
    monkeypatch.setattr(library_db, 'NETWORK_FOLDERS_FILE', str(tmp_path / 'nf.json'))
    library_db.save_network_folder('destination', {'path': '\\\\vantage\\ingest'})

    captured = {}
    with mock.patch('pipeline.smbclient.open_file', side_effect=lambda p, mode=None, **k: (captured.update(path=p, mode=mode), BytesIO())[1]), \
         mock.patch('pipeline.smbclient.register_session'):
        r = client.post(f'/library/{library_id}/send-to-destination', json={'format': 'mp4_high'}, headers=headers)

    assert r.status_code == 200, r.get_data(as_text=True)
    d = r.get_json()
    assert d['ok'] is True
    assert captured['path'].startswith('\\\\vantage\\ingest\\')
    assert captured['mode'] == 'wb'


def test_send_to_destination_with_a_custom_filename(rendered_trailer, tmp_path, monkeypatch):
    client, headers, result = rendered_trailer
    library_id = result['library_id']
    monkeypatch.setattr(library_db, 'NETWORK_FOLDERS_FILE', str(tmp_path / 'nf.json'))
    library_db.save_network_folder('destination', {'path': '\\\\vantage\\ingest'})

    captured = {}
    with mock.patch('pipeline.smbclient.open_file', side_effect=lambda p, mode=None, **k: (captured.update(path=p), BytesIO())[1]), \
         mock.patch('pipeline.smbclient.register_session'):
        r = client.post(f'/library/{library_id}/send-to-destination',
                        json={'format': 'mp4_high', 'filename': 'Week10_HILITES_Custom'}, headers=headers)

    assert r.status_code == 200, r.get_data(as_text=True)
    d = r.get_json()
    assert d['filename'] == 'Week10_HILITES_Custom.mp4'
    assert captured['path'] == '\\\\vantage\\ingest\\Week10_HILITES_Custom.mp4'


def test_download_with_a_custom_name(rendered_trailer):
    client, headers, result = rendered_trailer
    library_id = result['library_id']
    r = client.get(f'/library/{library_id}/download?format=mp4_high&name=My_Custom_Name', headers=headers)
    assert r.status_code == 200
    assert r.headers['Content-Disposition'] == 'attachment; filename="My_Custom_Name.mp4"'


def test_download_custom_name_rejects_path_traversal(rendered_trailer):
    client, headers, result = rendered_trailer
    library_id = result['library_id']
    r = client.get(f'/library/{library_id}/download?format=mp4_high&name=../../etc/passwd', headers=headers)
    assert r.status_code == 200
    disposition = r.headers['Content-Disposition']
    assert '..' not in disposition and '/' not in disposition


def test_download_custom_name_strips_a_fake_pasted_extension(rendered_trailer):
    # If someone pastes "promo.mov" as the name while downloading the MP4
    # format, the file must still come back as .mp4 -- the real export
    # format's extension always wins, never whatever the person typed.
    client, headers, result = rendered_trailer
    library_id = result['library_id']
    r = client.get(f'/library/{library_id}/download?format=mp4_high&name=my_promo.mov', headers=headers)
    assert r.status_code == 200
    assert r.headers['Content-Disposition'] == 'attachment; filename="my_promo.mp4"'


def test_renaming_does_not_create_duplicate_cached_exports(rendered_trailer):
    # The cache key (on disk) is always trailer id + format, independent of
    # whatever display name is requested -- confirms two different names for
    # the same format reuse the same cached file rather than re-exporting
    # (or worse, colliding) per name.
    client, headers, result = rendered_trailer
    library_id = result['library_id']
    r1 = client.get(f'/library/{library_id}/download?format=mp4_high&name=First_Name', headers=headers)
    r2 = client.get(f'/library/{library_id}/download?format=mp4_high&name=Second_Different_Name', headers=headers)
    assert r1.status_code == 200 and r2.status_code == 200
    assert r1.headers['Content-Disposition'] == 'attachment; filename="First_Name.mp4"'
    assert r2.headers['Content-Disposition'] == 'attachment; filename="Second_Different_Name.mp4"'
    # Both requests must return the exact same underlying bytes -- proving
    # the second call reused the cache rather than re-exporting.
    assert r1.data == r2.data


def test_send_to_destination_requires_ownership(rendered_trailer, tmp_path, monkeypatch):
    client, headers, result = rendered_trailer
    library_id = result['library_id']
    monkeypatch.setattr(library_db, 'NETWORK_FOLDERS_FILE', str(tmp_path / 'nf.json'))
    library_db.save_network_folder('destination', {'path': '\\\\vantage\\ingest'})

    # A different, non-admin user's session should not be able to send
    # someone else's render.
    with client.session_transaction() as sess:
        sess['user_id'] = 999
        sess['role'] = 'user'
    r = client.post(f'/library/{library_id}/send-to-destination', json={'format': 'mp4_high'}, headers=headers)
    assert r.status_code == 404

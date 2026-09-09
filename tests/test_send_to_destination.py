"""
Tests for "Send to destination" -- copies a finished render (and/or its
scene-list CSV) to one of potentially several named, admin-managed network
destinations instead of only being downloadable through the browser.
Covers the network_destinations table CRUD, send_file_to_network_destination
/ send_bytes_to_network_destination's SMB writes, build_scene_list_csv, and
the /library/<id>/send-to-destination route end to end.

This replaces an earlier single-destination version of this file: the
original design had exactly one, unnamed "destination" network category
(like HIRES/music/VO). Rebuilt as a proper table so a facility can
configure several named destinations, each choosing whether it receives
the rendered video, the scene-list CSV, or both.

Also covers a real bug found and fixed while building the original
version: the job's persisted result never included library_id, since
job_set(...,result=result) snapshots the dict (via JSON serialization) at
call time, before library_add() had a chance to set library_id on it --
meaning any feature reading library_id from a job's stored result (this
one included) would never see it. The existing Download button was never
visibly affected only because it already had a filename-based fallback
for a missing library_id; Send to destination has no such fallback, which
is what surfaced the bug.
"""
import shutil
import subprocess
import time
import unittest.mock as mock
from io import BytesIO

import pytest

import library_db
import pipeline
import core

with mock.patch('requests.post'), mock.patch('requests.get'):
    import main


def test_destination_no_longer_a_single_network_category():
    assert 'destination' not in library_db.NETWORK_CATEGORY_KEYS


def test_destination_add_get_list(tmp_path, monkeypatch):
    monkeypatch.setattr(library_db, 'LIBRARY_DB_PATH', str(tmp_path / 'lib.db'))
    library_db.library_db_init()
    dest_id = library_db.network_destination_add('PMC_MAMS_PUBLISHING', '\\\\vantage\\ingest',
                                                   username='svc', password='secret', delivery_kind='video')
    got = library_db.network_destination_get(dest_id)
    assert got['name'] == 'PMC_MAMS_PUBLISHING'
    assert got['path'] == '\\\\vantage\\ingest'
    assert got['delivery_kind'] == 'video'
    items = library_db.network_destinations_list()
    assert len(items) == 1 and items[0]['id'] == dest_id


def test_destination_invalid_delivery_kind_falls_back_to_video(tmp_path, monkeypatch):
    monkeypatch.setattr(library_db, 'LIBRARY_DB_PATH', str(tmp_path / 'lib.db'))
    library_db.library_db_init()
    dest_id = library_db.network_destination_add('X', '\\\\a\\b', delivery_kind='not_a_real_kind')
    assert library_db.network_destination_get(dest_id)['delivery_kind'] == 'video'


def test_destination_update_only_touches_passed_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(library_db, 'LIBRARY_DB_PATH', str(tmp_path / 'lib.db'))
    library_db.library_db_init()
    dest_id = library_db.network_destination_add('Name1', '\\\\a\\b', username='u1', password='p1', delivery_kind='video')
    library_db.network_destination_update(dest_id, name='Name2')
    got = library_db.network_destination_get(dest_id)
    assert got['name'] == 'Name2'
    assert got['path'] == '\\\\a\\b'
    assert got['username'] == 'u1'
    assert got['password'] == 'p1'
    assert got['delivery_kind'] == 'video'


def test_destination_remove(tmp_path, monkeypatch):
    monkeypatch.setattr(library_db, 'LIBRARY_DB_PATH', str(tmp_path / 'lib.db'))
    library_db.library_db_init()
    dest_id = library_db.network_destination_add('X', '\\\\a\\b')
    assert library_db.network_destination_remove(dest_id) is True
    assert library_db.network_destination_get(dest_id) is None
    assert library_db.network_destination_remove(dest_id) is False


def test_send_file_writes_to_the_destinations_own_credentials(tmp_path):
    local_file = tmp_path / 'render.mp4'
    local_file.write_bytes(b'fake video bytes')
    destination = {'name': 'PMC', 'path': '\\\\vantage\\ingest', 'username': 'svc_prism', 'password': 'secret'}

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
        pipeline.send_file_to_network_destination(str(local_file), 'render.mp4', destination)

    assert captured['host'] == 'vantage'
    assert captured['username'] == 'svc_prism'
    assert captured['password'] == 'secret'
    assert captured['path'] == '\\\\vantage\\ingest\\render.mp4'
    assert captured['mode'] == 'wb'


def test_send_file_wraps_smb_failures_with_the_destination_name(tmp_path):
    local_file = tmp_path / 'render.mp4'
    local_file.write_bytes(b'fake')
    destination = {'name': 'PMC_MAMS_PUBLISHING', 'path': '\\\\vantage\\ingest'}
    with mock.patch('pipeline.smbclient.register_session'), \
         mock.patch('pipeline.smbclient.open_file', side_effect=Exception('connection refused')):
        with pytest.raises(ValueError, match='PMC_MAMS_PUBLISHING'):
            pipeline.send_file_to_network_destination(str(local_file), 'render.mp4', destination)


def test_send_bytes_writes_content_directly_no_temp_file(tmp_path):
    destination = {'name': 'CSV_Dest', 'path': '\\\\srv\\csv'}
    captured = {}
    def fake_open_file(path, mode=None, **kw):
        buf = BytesIO()
        orig_write = buf.write
        def tracking_write(data):
            captured['content'] = data
            return orig_write(data)
        buf.write = tracking_write
        buf.__enter__ = lambda: buf
        buf.__exit__ = lambda *a: None
        return buf
    with mock.patch('pipeline.smbclient.open_file', side_effect=fake_open_file), \
         mock.patch('pipeline.smbclient.register_session'):
        pipeline.send_bytes_to_network_destination(b'hello,csv\n1,2\n', 'scenes.csv', destination)
    assert captured['content'] == b'hello,csv\n1,2\n'


def test_build_scene_list_csv_shape():
    row = {'result_json': '{"scenes": [{"scene": 1, "start": 0.0, "end": 6.0, "duration": 6.0, "quality": 4, "description": "a \\"quoted\\" desc"}]}'}
    csv_text = pipeline.build_scene_list_csv(row)
    lines = csv_text.strip().split('\n')
    assert lines[0] == '#,Start_s,End_s,Used_s,Score,Description'
    assert lines[1] == '1,0.0,6.0,6.0,4,"a ""quoted"" desc"'


def test_build_scene_list_csv_empty_scenes():
    row = {'result_json': '{"scenes": []}'}
    csv_text = pipeline.build_scene_list_csv(row)
    assert csv_text.strip() == '#,Start_s,End_s,Used_s,Score,Description'


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

    # This fixture triggers a real render per test that uses it, and this
    # file alone now has many such tests -- combined with every other test
    # file's own real renders, all sharing the same per-client-IP rate
    # limiter (test_client() submits everything from the same address),
    # the cumulative count across the WHOLE suite can exceed the app's own
    # render rate limit well before this file even gets a chance to run.
    # That's not a bug in the limiter or in the app -- it's a real safety
    # feature working exactly as intended -- so clearing its bucket here,
    # right before this fixture's own submission, is the correct fix: it
    # only removes the test-suite-induced cross-file interference, not the
    # actual rate-limiting behavior itself, which stays fully in effect for
    # everything else including any real usage.
    core._job_submit_limiter.buckets.clear()
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


def _make_destination(delivery_kind='video'):
    return library_db.network_destination_add('TestDest', '\\\\vantage\\ingest', delivery_kind=delivery_kind)


def test_real_render_has_a_library_id_in_its_stored_result(rendered_trailer):
    client, headers, result = rendered_trailer
    assert result.get('library_id') is not None


def test_send_requires_a_real_destination_id(rendered_trailer):
    client, headers, result = rendered_trailer
    library_id = result['library_id']
    r = client.post(f'/library/{library_id}/send-to-destination', json={'destination_id': 999999}, headers=headers)
    assert r.status_code == 400
    assert 'no longer exists' in r.get_json()['error']


def test_send_video_only_destination(rendered_trailer):
    client, headers, result = rendered_trailer
    library_id = result['library_id']
    dest_id = _make_destination('video')
    captured = []
    with mock.patch('pipeline.smbclient.open_file', side_effect=lambda p, mode=None, **k: (captured.append(p), BytesIO())[-1]), \
         mock.patch('pipeline.smbclient.register_session'):
        r = client.post(f'/library/{library_id}/send-to-destination',
                        json={'destination_id': dest_id, 'format': 'mp4_high'}, headers=headers)
    assert r.status_code == 200, r.get_data(as_text=True)
    d = r.get_json()
    assert d['ok'] is True
    assert len(captured) == 1
    assert captured[0].endswith('.mp4')


def test_send_csv_only_destination_writes_no_video(rendered_trailer):
    client, headers, result = rendered_trailer
    library_id = result['library_id']
    dest_id = _make_destination('csv')
    captured = []
    with mock.patch('pipeline.smbclient.open_file', side_effect=lambda p, mode=None, **k: (captured.append(p), BytesIO())[-1]), \
         mock.patch('pipeline.smbclient.register_session'):
        r = client.post(f'/library/{library_id}/send-to-destination',
                        json={'destination_id': dest_id, 'format': 'mp4_high'}, headers=headers)
    assert r.status_code == 200, r.get_data(as_text=True)
    assert len(captured) == 1
    assert captured[0].endswith('_scenes.csv')
    assert not any(p.endswith('.mp4') for p in captured)


def test_send_csv_video_destination_writes_both(rendered_trailer):
    client, headers, result = rendered_trailer
    library_id = result['library_id']
    dest_id = _make_destination('csv_video')
    captured = []
    with mock.patch('pipeline.smbclient.open_file', side_effect=lambda p, mode=None, **k: (captured.append(p), BytesIO())[-1]), \
         mock.patch('pipeline.smbclient.register_session'):
        r = client.post(f'/library/{library_id}/send-to-destination',
                        json={'destination_id': dest_id, 'format': 'mp4_high'}, headers=headers)
    assert r.status_code == 200, r.get_data(as_text=True)
    d = r.get_json()
    assert len(d['sent']) == 2
    assert any(p.endswith('.mp4') for p in captured)
    assert any(p.endswith('_scenes.csv') for p in captured)


def test_send_to_destination_with_a_custom_filename(rendered_trailer):
    client, headers, result = rendered_trailer
    library_id = result['library_id']
    dest_id = _make_destination('video')
    captured = {}
    with mock.patch('pipeline.smbclient.open_file', side_effect=lambda p, mode=None, **k: (captured.update(path=p), BytesIO())[1]), \
         mock.patch('pipeline.smbclient.register_session'):
        r = client.post(f'/library/{library_id}/send-to-destination',
                        json={'destination_id': dest_id, 'format': 'mp4_high', 'filename': 'Week10_HILITES_Custom'}, headers=headers)
    assert r.status_code == 200, r.get_data(as_text=True)
    assert captured['path'].endswith('Week10_HILITES_Custom.mp4')


def test_send_to_destination_requires_ownership(rendered_trailer):
    client, headers, result = rendered_trailer
    library_id = result['library_id']
    dest_id = _make_destination('video')

    with client.session_transaction() as sess:
        sess['user_id'] = 999
        sess['role'] = 'user'
    r = client.post(f'/library/{library_id}/send-to-destination', json={'destination_id': dest_id, 'format': 'mp4_high'}, headers=headers)
    assert r.status_code == 404


def test_add_destination_requires_admin(rendered_trailer):
    client, headers, result = rendered_trailer
    with client.session_transaction() as sess:
        sess['role'] = 'user'
    r = client.post('/api/network/destinations', json={'name': 'X', 'path': '\\\\a\\b'}, headers=headers)
    assert r.status_code == 403


def test_list_destinations_never_includes_passwords(rendered_trailer):
    client, headers, result = rendered_trailer
    library_db.network_destination_add('WithPassword', '\\\\a\\b', password='supersecret')
    r = client.get('/api/network/destinations', headers=headers)
    items = r.get_json()['items']
    assert all('password' not in item for item in items)


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
    client, headers, result = rendered_trailer
    library_id = result['library_id']
    r = client.get(f'/library/{library_id}/download?format=mp4_high&name=my_promo.mov', headers=headers)
    assert r.status_code == 200
    assert r.headers['Content-Disposition'] == 'attachment; filename="my_promo.mp4"'


def test_renaming_does_not_create_duplicate_cached_exports(rendered_trailer):
    client, headers, result = rendered_trailer
    library_id = result['library_id']
    r1 = client.get(f'/library/{library_id}/download?format=mp4_high&name=First_Name', headers=headers)
    r2 = client.get(f'/library/{library_id}/download?format=mp4_high&name=Second_Different_Name', headers=headers)
    assert r1.status_code == 200 and r2.status_code == 200
    assert r1.headers['Content-Disposition'] == 'attachment; filename="First_Name.mp4"'
    assert r2.headers['Content-Disposition'] == 'attachment; filename="Second_Different_Name.mp4"'
    assert r1.data == r2.data

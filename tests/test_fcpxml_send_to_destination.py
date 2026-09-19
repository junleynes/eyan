"""
Tests for features added on top of Send to destination:

  1. The 'fcpxml' delivery_kind ("XML + media only") -- sends the original
     HIRES source, any surviving music/VO/card assets, and the FCP XML
     rough-cut package (see build_fcpxml_package in pipeline.py), with no
     video export or scene-list CSV. fcpxml_audio_tracks is a per-
     destination toggle for whether that XML embeds real audio clipitems.
     (An earlier version had a separate include_fcpxml checkbox add-on
     that bolted the XML onto a video/csv/csv_video destination -- removed
     since it was redundant once this dedicated kind existed; the DB
     column is kept, inert, for backward compatibility -- see a couple of
     tests below that confirm setting it directly no longer does anything.)
  2. A preview-stage Send to destination route
     (/api/trailer/preview/<id>/send-to-destination) so a csv/csv_video/
     fcpxml destination -- whose whole point is the ORIGINAL source video
     plus a CSV or XML of the reviewed cut -- doesn't require sitting
     through Lock cut & render first, since both are already known right
     after Preview.
"""
import json
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

from tests.test_send_to_destination import rendered_trailer_from_network, _make_destination  # noqa: F401


def _ffmpeg_available():
    return shutil.which('ffmpeg') is not None


# ---- include_fcpxml / fcpxml_audio_tracks CRUD ----

def test_destination_add_defaults_fcpxml_flags_off(tmp_path, monkeypatch):
    monkeypatch.setattr(library_db, 'LIBRARY_DB_PATH', str(tmp_path / 'lib.db'))
    library_db.library_db_init()
    dest_id = library_db.network_destination_add('D', '\\\\x\\y')
    dest = library_db.network_destination_get(dest_id)
    assert dest['include_fcpxml'] == 0
    assert dest['fcpxml_audio_tracks'] == 0


def test_destination_add_with_fcpxml_flags_on(tmp_path, monkeypatch):
    monkeypatch.setattr(library_db, 'LIBRARY_DB_PATH', str(tmp_path / 'lib.db'))
    library_db.library_db_init()
    dest_id = library_db.network_destination_add('D', '\\\\x\\y', include_fcpxml=True, fcpxml_audio_tracks=True)
    dest = library_db.network_destination_get(dest_id)
    assert dest['include_fcpxml'] == 1
    assert dest['fcpxml_audio_tracks'] == 1


def test_destination_update_fcpxml_flags_only_touches_passed_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(library_db, 'LIBRARY_DB_PATH', str(tmp_path / 'lib.db'))
    library_db.library_db_init()
    dest_id = library_db.network_destination_add('D', '\\\\x\\y', include_fcpxml=True)
    library_db.network_destination_update(dest_id, name='D2')  # fcpxml flags not passed
    dest = library_db.network_destination_get(dest_id)
    assert dest['name'] == 'D2'
    assert dest['include_fcpxml'] == 1  # unchanged


def test_api_add_destination_accepts_fcpxml_flags(tmp_path, monkeypatch):
    monkeypatch.setattr(library_db, 'LIBRARY_DB_PATH', str(tmp_path / 'lib.db'))
    library_db.library_db_init()
    app = main.app
    client = app.test_client()
    csrf_token = 'test-csrf-fcpxml-add'
    with client.session_transaction() as sess:
        sess['authed'] = True
        sess['user_id'] = 1
        sess['username'] = 'admin'
        sess['role'] = 'admin'
        sess['csrf_token'] = csrf_token
    headers = {'X-CSRF-Token': csrf_token}
    r = client.post('/api/network/destinations', json={
        'name': 'D', 'path': '\\\\x\\y', 'delivery_kind': 'video',
        'include_fcpxml': True, 'fcpxml_audio_tracks': False,
    }, headers=headers)
    assert r.status_code == 200, r.get_data(as_text=True)
    items = client.get('/api/network/destinations', headers=headers).get_json()['items']
    dest = next(d for d in items if d['name'] == 'D')
    assert dest['include_fcpxml'] == 1
    assert dest['fcpxml_audio_tracks'] == 0


# ---- library_send_to_destination sending the FCP XML package ----

def test_send_to_destination_fcpxml_kind_sends_xml(rendered_trailer_from_network):
    # The XML package is now only ever sent via the dedicated 'fcpxml'
    # delivery_kind -- the old include_fcpxml checkbox add-on (bolt the XML
    # onto a video/csv/csv_video destination) has been removed; setting the
    # column directly no longer does anything, since gating is on kind alone.
    client, headers, result = rendered_trailer_from_network
    library_id = result['library_id']
    dest_id = _make_destination('fcpxml')

    sent_bytes = {}
    def fake_bytes(data, filename, destination):
        sent_bytes[filename] = data
    with mock.patch('pipeline.send_file_to_network_destination'), \
         mock.patch('pipeline.send_bytes_to_network_destination', side_effect=fake_bytes):
        r = client.post(f'/library/{library_id}/send-to-destination',
                        json={'destination_id': dest_id, 'format': 'mp4_high'}, headers=headers)
    assert r.status_code == 200, r.get_data(as_text=True)
    xml_files = [f for f in sent_bytes if f.endswith('_cut.xml')]
    assert len(xml_files) == 1
    xml_text = sent_bytes[xml_files[0]].decode('utf-8')
    assert '<xmeml' in xml_text
    assert '<clipitem' in xml_text


def test_send_to_destination_non_fcpxml_kind_sends_no_xml(rendered_trailer_from_network):
    client, headers, result = rendered_trailer_from_network
    library_id = result['library_id']
    dest_id = _make_destination('video')
    # Setting the legacy column directly (no UI exposes this anymore) has
    # no effect now -- only delivery_kind == 'fcpxml' sends the XML.
    library_db.network_destination_update(dest_id, include_fcpxml=True)

    sent = []
    with mock.patch('pipeline.send_file_to_network_destination'), \
         mock.patch('pipeline.send_bytes_to_network_destination', side_effect=lambda d, n, dest: sent.append(n)):
        r = client.post(f'/library/{library_id}/send-to-destination',
                        json={'destination_id': dest_id, 'format': 'mp4_high'}, headers=headers)
    assert r.status_code == 200, r.get_data(as_text=True)
    assert not any(f.endswith('.xml') for f in sent)


def test_send_to_destination_fcpxml_audio_tracks_toggle(rendered_trailer_from_network):
    client, headers, result = rendered_trailer_from_network
    library_id = result['library_id']
    dest_id = _make_destination('fcpxml')
    library_db.network_destination_update(dest_id, fcpxml_audio_tracks=True)

    with mock.patch('pipeline.build_fcpxml_package', wraps=pipeline.build_fcpxml_package) as spy, \
         mock.patch('pipeline.send_file_to_network_destination'), \
         mock.patch('pipeline.send_bytes_to_network_destination'):
        r = client.post(f'/library/{library_id}/send-to-destination',
                        json={'destination_id': dest_id, 'format': 'mp4_high'}, headers=headers)
    assert r.status_code == 200, r.get_data(as_text=True)
    assert spy.call_args.kwargs.get('include_audio_tracks') is True or spy.call_args.args[1:] == (True,)


# ---- preview-stage send-to-destination ----

@pytest.fixture
def preview_from_network(tmp_path, monkeypatch):
    """A real /api/trailer/preview run (no render) from a net_*-staged
    source, so its source survives just like rendered_trailer_from_network's
    does -- returns (client, headers, preview_id, preview_json)."""
    if not _ffmpeg_available():
        pytest.skip('ffmpeg not available in this environment')
    app = main.app
    upload_dir = tmp_path / 'uploads'
    upload_dir.mkdir()
    monkeypatch.setitem(app.config, 'UPLOAD_FOLDER', str(upload_dir))
    monkeypatch.setattr(pipeline, 'ALLOW_LOCAL_MEDIA_UPLOAD', True)

    client = app.test_client()
    csrf_token = 'test-csrf-preview-send-dest'
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

    staged_name = 'net_2000_src.mp4'
    shutil.copy(str(src), str(upload_dir / staged_name))

    core._job_submit_limiter.buckets.clear()
    r = client.post('/api/trailer/generate', data={
        'network_file': staged_name, 'preview_only': '1',
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
    result = d.get('result') or {}
    preview_id = result.get('preview_id')
    assert preview_id, 'preview run did not return a preview_id -- check the route name/shape'
    return client, headers, preview_id, result


def test_preview_send_to_destination_rejects_video_only(preview_from_network):
    client, headers, preview_id, _ = preview_from_network
    dest_id = _make_destination('video')
    r = client.post(f'/api/trailer/preview/{preview_id}/send-to-destination',
                    json={'destination_id': dest_id}, headers=headers)
    assert r.status_code == 400
    assert not r.get_json()['ok']


def test_preview_send_to_destination_sends_csv_and_source(preview_from_network):
    client, headers, preview_id, _ = preview_from_network
    dest_id = _make_destination('csv_video')

    sent_files = {}
    def fake_bytes(data, filename, destination):
        sent_files[filename] = data
    captured_video = []
    with mock.patch('pipeline.send_file_to_network_destination',
                    side_effect=lambda p, n, d: captured_video.append((p, n))), \
         mock.patch('pipeline.send_bytes_to_network_destination', side_effect=fake_bytes):
        r = client.post(f'/api/trailer/preview/{preview_id}/send-to-destination',
                        json={'destination_id': dest_id}, headers=headers)
    assert r.status_code == 200, r.get_data(as_text=True)
    assert len(captured_video) == 1
    assert any(f.endswith('_scenes.csv') for f in sent_files)


def test_preview_send_to_destination_honors_drop(preview_from_network):
    client, headers, preview_id, _ = preview_from_network
    dest_id = _make_destination('csv')

    sent_csv = {}
    with mock.patch('pipeline.send_bytes_to_network_destination',
                    side_effect=lambda d, n, dest: sent_csv.__setitem__(n, d)):
        r_all = client.post(f'/api/trailer/preview/{preview_id}/send-to-destination',
                            json={'destination_id': dest_id}, headers=headers)
    assert r_all.status_code == 200, r_all.get_data(as_text=True)
    csv_all = next(iter(sent_csv.values())).decode('utf-8')
    rows_all = [l for l in csv_all.splitlines() if l.strip()]

    sent_csv.clear()
    with mock.patch('pipeline.send_bytes_to_network_destination',
                    side_effect=lambda d, n, dest: sent_csv.__setitem__(n, d)):
        r_drop = client.post(f'/api/trailer/preview/{preview_id}/send-to-destination',
                             json={'destination_id': dest_id, 'drop': [1]}, headers=headers)
    assert r_drop.status_code == 200, r_drop.get_data(as_text=True)
    csv_drop = next(iter(sent_csv.values())).decode('utf-8')
    rows_drop = [l for l in csv_drop.splitlines() if l.strip()]
    assert len(rows_drop) == len(rows_all) - 1


def test_preview_send_to_destination_csv_kind_ignores_legacy_include_fcpxml_column(preview_from_network):
    # The old include_fcpxml checkbox add-on is gone -- setting the legacy
    # column directly on a csv destination no longer makes it send XML;
    # only the dedicated 'fcpxml' delivery_kind does (see the 'fcpxml as
    # its own delivery_kind' section below).
    client, headers, preview_id, _ = preview_from_network
    dest_id = _make_destination('csv')
    library_db.network_destination_update(dest_id, include_fcpxml=True)

    sent_files = {}
    with mock.patch('pipeline.send_bytes_to_network_destination',
                    side_effect=lambda d, n, dest: sent_files.__setitem__(n, d)):
        r = client.post(f'/api/trailer/preview/{preview_id}/send-to-destination',
                        json={'destination_id': dest_id}, headers=headers)
    assert r.status_code == 200, r.get_data(as_text=True)
    assert not any(f.endswith('_cut.xml') for f in sent_files)


# ---- 'fcpxml' as its own delivery_kind ("XML + media only") ----
# Distinct from include_fcpxml above: a destination configured with
# delivery_kind == 'fcpxml' always sends the XML package plus physical
# copies of the source/surviving materials, with no video/csv sent, and
# needs no separate checkbox to opt in.

def test_fcpxml_is_a_valid_delivery_kind():
    assert 'fcpxml' in library_db.DELIVERY_KINDS


def test_send_to_destination_fcpxml_kind_sends_source_and_xml_no_csv(rendered_trailer_from_network):
    client, headers, result = rendered_trailer_from_network
    library_id = result['library_id']
    dest_id = _make_destination('fcpxml')

    sent_bytes = {}
    sent_files = []
    with mock.patch('pipeline.send_file_to_network_destination',
                    side_effect=lambda p, n, d: sent_files.append(n)), \
         mock.patch('pipeline.send_bytes_to_network_destination',
                    side_effect=lambda d, n, dest: sent_bytes.__setitem__(n, d)):
        r = client.post(f'/library/{library_id}/send-to-destination',
                        json={'destination_id': dest_id, 'format': 'mp4_high'}, headers=headers)
    assert r.status_code == 200, r.get_data(as_text=True)
    xml_files = [f for f in sent_bytes if f.endswith('_cut.xml')]
    assert len(xml_files) == 1
    assert b'<xmeml' in sent_bytes[xml_files[0]]
    # The HIRES source itself was sent as a physical file (like csv_video)...
    assert len(sent_files) >= 1
    # ...but no scene-list CSV, since this kind's whole point is XML+media.
    assert not any(f.endswith('_scenes.csv') for f in sent_bytes)


def test_preview_send_to_destination_accepts_fcpxml_kind(preview_from_network):
    client, headers, preview_id, _ = preview_from_network
    dest_id = _make_destination('fcpxml')

    sent_bytes = {}
    sent_files = []
    with mock.patch('pipeline.send_file_to_network_destination',
                    side_effect=lambda p, n, d: sent_files.append(n)), \
         mock.patch('pipeline.send_bytes_to_network_destination',
                    side_effect=lambda d, n, dest: sent_bytes.__setitem__(n, d)):
        r = client.post(f'/api/trailer/preview/{preview_id}/send-to-destination',
                        json={'destination_id': dest_id}, headers=headers)
    assert r.status_code == 200, r.get_data(as_text=True)
    xml_files = [f for f in sent_bytes if f.endswith('_cut.xml')]
    assert len(xml_files) == 1
    assert not any(f.endswith('_scenes.csv') for f in sent_bytes)
    assert len(sent_files) >= 1  # the source video, sent as a physical file


def test_preview_send_to_destination_unknown_preview_id(rendered_trailer_from_network):
    # Reuses the render fixture purely to get an authenticated client/headers
    # cheaply -- the preview_id itself is deliberately bogus.
    client, headers, _ = rendered_trailer_from_network
    dest_id = _make_destination('csv')
    r = client.post('/api/trailer/preview/does-not-exist/send-to-destination',
                    json={'destination_id': dest_id}, headers=headers)
    assert r.status_code == 404

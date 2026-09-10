"""
Tests for went_through_preview -- a flag on the render result distinguishing
a render reached via "Preview the cut" -> "Render this cut" (the scene
selection was actually reviewed, and possibly edited: scenes dropped or
swapped, by a person before this render ran) from "Generate without
preview" (a direct, one-shot render with no review step at all).

User-requested behavior, confirmed explicitly: a csv/csv_video Send to
destination target -- whose whole point is sending the scene-list CSV
alongside the source video -- should only be offered for a REVIEWED
render, since that CSV describes exactly the selection a person looked
at. A plain video destination is the mirror case, meant specifically for
the unreviewed, direct path. The frontend does the actual filtering
(see renderTrailerResult's destinations fetch in templates/index.html);
these tests cover the backend flag these decisions are based on.
"""
import os
import re
import shutil
import subprocess
import time
import unittest.mock as mock
from io import BytesIO

import pytest

import core
import pipeline


with mock.patch('requests.post'), mock.patch('requests.get'):
    import main


def _ffmpeg_available():
    return shutil.which('ffmpeg') is not None


@pytest.fixture
def client_and_headers(tmp_path, monkeypatch):
    if not _ffmpeg_available():
        pytest.skip('ffmpeg not available in this environment')
    app = main.app
    upload_dir = tmp_path / 'uploads'
    upload_dir.mkdir()
    monkeypatch.setitem(app.config, 'UPLOAD_FOLDER', str(upload_dir))
    monkeypatch.setattr(pipeline, 'ALLOW_LOCAL_MEDIA_UPLOAD', True)

    client = app.test_client()
    csrf_token = 'test-csrf-preview-flag'
    with client.session_transaction() as sess:
        sess['authed'] = True
        sess['user_id'] = 1
        sess['username'] = 'admin'
        sess['role'] = 'admin'
        sess['csrf_token'] = csrf_token
    return client, {'X-CSRF-Token': csrf_token}


def _build_test_source(tmp_path):
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
    src = tmp_path / 'src.mp4'
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0',
                    '-i', str(list_file), '-c', 'copy', str(src)], check=True, timeout=30)
    return src


def _poll_until_done(client, headers, job_id, timeout=60):
    deadline = time.time() + timeout
    d = None
    while time.time() < deadline:
        d = client.get(f'/api/trailer/progress/{job_id}', headers=headers).get_json()
        if d.get('done'):
            return d
        time.sleep(1)
    return d


def test_direct_generate_has_went_through_preview_false(client_and_headers, tmp_path):
    client, headers = client_and_headers
    src = _build_test_source(tmp_path)
    core._job_submit_limiter.buckets.clear()
    r = client.post('/api/trailer/generate', data={
        'file': (BytesIO(src.read_bytes()), 'src.mp4'),
        'genre': '', 'trailer_length': '15', 'scoring_mode': 'none',
        'sfx_mode': 'none', 'vo_mode': 'none', 'transition': 'cut',
    }, headers=headers, content_type='multipart/form-data')
    assert r.status_code == 200, r.get_data(as_text=True)
    d = _poll_until_done(client, headers, r.get_json()['job_id'])
    assert d and d.get('error') is None
    result = d.get('result') or {}
    assert result.get('went_through_preview') is False


def test_preview_then_render_has_went_through_preview_true(client_and_headers, tmp_path):
    client, headers = client_and_headers
    src = _build_test_source(tmp_path)
    core._job_submit_limiter.buckets.clear()
    r = client.post('/api/trailer/generate', data={
        'file': (BytesIO(src.read_bytes()), 'src.mp4'),
        'genre': '', 'trailer_length': '15', 'scoring_mode': 'none',
        'sfx_mode': 'none', 'vo_mode': 'none', 'transition': 'cut',
        'preview_only': '1',
    }, headers=headers, content_type='multipart/form-data')
    assert r.status_code == 200, r.get_data(as_text=True)
    preview_done = _poll_until_done(client, headers, r.get_json()['job_id'])
    assert preview_done and preview_done.get('error') is None
    preview_id = (preview_done.get('result') or {}).get('preview_id')
    assert preview_id

    core._job_submit_limiter.buckets.clear()
    r2 = client.post('/api/trailer/render', data={'preview_id': preview_id}, headers=headers)
    assert r2.status_code == 200, r2.get_data(as_text=True)
    render_done = _poll_until_done(client, headers, r2.get_json()['job_id'])
    assert render_done and render_done.get('error') is None
    result = render_done.get('result') or {}
    assert result.get('went_through_preview') is True


def test_went_through_preview_is_persisted_to_the_library(client_and_headers, tmp_path):
    # library_add() copies the whole result dict as-is into result_json
    # (see library_db.library_add) -- confirms went_through_preview
    # specifically survives that round trip, since Send to destination
    # reads it back from the SAVED library row, not the in-memory job
    # result, by the time a user actually clicks Send.
    client, headers = client_and_headers
    src = _build_test_source(tmp_path)
    core._job_submit_limiter.buckets.clear()
    r = client.post('/api/trailer/generate', data={
        'file': (BytesIO(src.read_bytes()), 'src.mp4'),
        'genre': '', 'trailer_length': '15', 'scoring_mode': 'none',
        'sfx_mode': 'none', 'vo_mode': 'none', 'transition': 'cut',
        'preview_only': '1',
    }, headers=headers, content_type='multipart/form-data')
    preview_done = _poll_until_done(client, headers, r.get_json()['job_id'])
    preview_id = (preview_done.get('result') or {}).get('preview_id')

    core._job_submit_limiter.buckets.clear()
    r2 = client.post('/api/trailer/render', data={'preview_id': preview_id}, headers=headers)
    render_done = _poll_until_done(client, headers, r2.get_json()['job_id'])
    library_id = (render_done.get('result') or {}).get('library_id')
    assert library_id

    row = pipeline.library_get_row(library_id)
    import json
    saved_result = json.loads(row['result_json'])
    assert saved_result.get('went_through_preview') is True

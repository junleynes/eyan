"""
Tests for production_summary -- a single, complete record of what actually
went into a specific render (transition, title/schedule cards, VO/music
source, the prompts and script used), collected onto the render result so
a person can see it after the fact without having to remember it or
re-open the original job.

Covers both the two small helpers this is built from and the real bug
found while building this: a direct upload's local temp path throws the
browser's own original filename away entirely (see _resolve_upload's own
docstring), so a naive "just use the path's basename" approach showed an
ugly, meaningless internal name (e.g. scoring_audio_1789026122762_....mp3)
instead of what the person actually uploaded.
"""
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


def test_asset_display_name_strips_the_net_prefix():
    # A network-staged file's local name is net_<timestamp>_<original> --
    # confirms the real original name is recovered, not the local name
    # verbatim. Uses a forward-slash path deliberately: os.path.basename
    # only recognizes this sandbox's own OS-native separator, and this
    # sandbox is Linux -- the real deployment's own Windows paths use
    # backslashes, which os.path there handles the same way natively.
    assert pipeline._asset_display_name('/uploads/net_1788927396_MyMusic.mp3') == 'MyMusic.mp3'


def test_asset_display_name_passes_through_a_name_with_no_prefix():
    # A direct upload's own local name has no net_ prefix at all -- this
    # helper alone can't recover the real original name for that case (see
    # _upload_orig_name, which is what actually handles it); it should at
    # least not mangle a name that already has none of that prefix shape.
    assert pipeline._asset_display_name('/tmp/some_plain_name.wav') == 'some_plain_name.wav'


def test_asset_display_name_none_for_no_path():
    assert pipeline._asset_display_name(None) is None
    assert pipeline._asset_display_name('') is None


def test_upload_orig_name_prefers_the_real_direct_upload_filename():
    # The actual bug this exists to fix: a direct upload's local temp path
    # (field_name_timestamp_threadid.ext) never contains the real
    # filename at all -- _upload_orig_name must go back to request.files
    # for it instead of trying to derive it from the local path.
    app = main.app
    with app.test_request_context(
        '/api/trailer/generate', method='POST',
        data={'scoring_audio': (BytesIO(b'fake audio bytes'), 'My Real Filename.mp3')},
        content_type='multipart/form-data'
    ):
        assert pipeline._upload_orig_name('scoring_audio') == 'My_Real_Filename.mp3'


def test_upload_orig_name_falls_back_to_the_network_staged_name():
    app = main.app
    with app.test_request_context(
        '/api/trailer/generate', method='POST',
        data={'vo_upload_network': 'net_1788927396_original_vo.wav'}
    ):
        assert pipeline._upload_orig_name('vo_upload') == 'original_vo.wav'


def test_upload_orig_name_none_when_neither_is_present():
    app = main.app
    with app.test_request_context('/api/trailer/generate', method='POST', data={}):
        assert pipeline._upload_orig_name('scoring_audio') is None


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
    csrf_token = 'test-csrf-production-summary'
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


def test_real_render_shows_the_actual_uploaded_filenames_not_internal_names(client_and_headers, tmp_path):
    client, headers = client_and_headers
    src = _build_test_source(tmp_path)
    music = tmp_path / 'my_theme_song.mp3'
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'lavfi',
                    '-i', 'sine=frequency=440:duration=6', str(music)], check=True, timeout=30)
    vo = tmp_path / 'my_narration.wav'
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'lavfi',
                    '-i', 'sine=frequency=880:duration=4', str(vo)], check=True, timeout=30)

    core._job_submit_limiter.buckets.clear()
    r = client.post('/api/trailer/generate', data={
        'file': (BytesIO(src.read_bytes()), 'src.mp4'),
        'genre': '', 'trailer_length': '15', 'scoring_mode': 'upload',
        'scoring_audio': (BytesIO(music.read_bytes()), 'my_theme_song.mp3'),
        'sfx_mode': 'none', 'vo_mode': 'upload',
        'vo_upload': (BytesIO(vo.read_bytes()), 'my_narration.wav'),
        'transition': 'fade',
        'priority_prompt': 'the big reveal',
        'negative_prompt': 'black screen',
    }, headers=headers, content_type='multipart/form-data')
    assert r.status_code == 200, r.get_data(as_text=True)

    deadline = time.time() + 60
    d = None
    while time.time() < deadline:
        d = client.get(f'/api/trailer/progress/{r.get_json()["job_id"]}', headers=headers).get_json()
        if d.get('done'):
            break
        time.sleep(1)
    assert d and d.get('error') is None

    ps = (d.get('result') or {}).get('production_summary') or {}
    assert ps.get('music_file') == 'my_theme_song.mp3'
    assert ps.get('vo_file') == 'my_narration.wav'
    assert ps.get('transition') == 'fade'
    assert ps.get('priority_prompt') == 'the big reveal'
    assert ps.get('negative_prompt') == 'black screen'
    # Neither an internal timestamp-based name leaked through anywhere.
    assert not re.search(r'_\d{10,}', ps.get('music_file', ''))
    assert not re.search(r'_\d{10,}', ps.get('vo_file', ''))


def test_unused_fields_are_omitted_not_shown_as_empty(client_and_headers, tmp_path):
    client, headers = client_and_headers
    src = _build_test_source(tmp_path)
    core._job_submit_limiter.buckets.clear()
    r = client.post('/api/trailer/generate', data={
        'file': (BytesIO(src.read_bytes()), 'src.mp4'),
        'genre': '', 'trailer_length': '15', 'scoring_mode': 'none',
        'sfx_mode': 'none', 'vo_mode': 'none', 'transition': 'cut',
    }, headers=headers, content_type='multipart/form-data')
    deadline = time.time() + 60
    d = None
    while time.time() < deadline:
        d = client.get(f'/api/trailer/progress/{r.get_json()["job_id"]}', headers=headers).get_json()
        if d.get('done'):
            break
        time.sleep(1)
    ps = (d.get('result') or {}).get('production_summary') or {}
    assert ps.get('music_file') is None
    assert ps.get('vo_file') is None
    assert ps.get('title_card') is None
    assert ps.get('priority_prompt') is None
    assert ps.get('script_cue_count') is None

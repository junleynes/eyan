"""
User-requested: wire LoRA support into ACE-Step music generation, plus a
rename-before-download option and WAV/stem export options for generated
tracks.

Covers:
  - /api/music/loras listing files from ACE_STEP_LORA_DIR
  - /api/music/generate resolving a picked LoRA filename to a real path and
    passing lora_path/lora_weight through to acestep_generate() (mocked --
    no real ACE-Step server needed), and rejecting an unknown/removed LoRA
    before ever making a network call
  - _demucs_available()/separate_stems() (mocked subprocess, so this suite
    doesn't require demucs actually installed)
  - /api/music/export: custom filename, WAV re-encode, missing-file 404,
    and the stem path (mocked separate_stems) producing a zip; plus the
    clear 503 when demucs isn't installed
"""
import os
import shutil
import subprocess
import unittest.mock as mock
import zipfile

import pytest

with mock.patch('requests.post'), mock.patch('requests.get'):
    import main
    import pipeline


def _ffmpeg_available():
    return shutil.which('ffmpeg') is not None


def _client_with_session():
    app = main.app
    client = app.test_client()
    csrf_token = 'test-csrf-music-lora-export'
    with client.session_transaction() as sess:
        sess['authed'] = True
        sess['user_id'] = 1
        sess['username'] = 'admin'
        sess['role'] = 'admin'
        sess['csrf_token'] = csrf_token
    return client, {'X-CSRF-Token': csrf_token}


@pytest.fixture
def lora_dir(tmp_path, monkeypatch):
    d = tmp_path / 'loras'
    d.mkdir()
    monkeypatch.setattr(pipeline, 'ACE_STEP_LORA_DIR', str(d))
    return d


@pytest.fixture
def upload_dir(tmp_path, monkeypatch):
    d = tmp_path / 'uploads'
    d.mkdir()
    monkeypatch.setitem(main.app.config, 'UPLOAD_FOLDER', str(d))
    return d


# ---------------------------------------------------------------------------
# /api/music/loras
# ---------------------------------------------------------------------------

def test_loras_lists_only_recognized_extensions_and_ignores_junk(lora_dir):
    (lora_dir / 'my_style.safetensors').write_bytes(b'x')
    (lora_dir / 'another.pt').write_bytes(b'x')
    (lora_dir / 'readme.txt').write_bytes(b'not a lora')
    (lora_dir / 'sub').mkdir()  # directories must be skipped, not just wrong extensions
    client, headers = _client_with_session()
    r = client.get('/api/music/loras', headers=headers)
    assert r.status_code == 200
    body = r.get_json()
    assert body['ok'] is True
    assert sorted(body['loras']) == ['another.pt', 'my_style.safetensors']


def test_loras_empty_dir_is_not_an_error(lora_dir):
    client, headers = _client_with_session()
    r = client.get('/api/music/loras', headers=headers)
    assert r.status_code == 200
    body = r.get_json()
    assert body['ok'] is True
    assert body['loras'] == []


# ---------------------------------------------------------------------------
# /api/music/generate -- LoRA resolution
# ---------------------------------------------------------------------------

def test_generate_rejects_a_lora_name_not_present_on_disk(lora_dir):
    client, headers = _client_with_session()
    r = client.post('/api/music/generate', data={
        'prompt': 'cinematic instrumental',
        'lora': 'does_not_exist.safetensors',
    }, headers=headers, content_type='multipart/form-data')
    assert r.status_code == 400
    body = r.get_json()
    assert body['ok'] is False
    assert 'does_not_exist.safetensors' in body['error']
    assert 'not found' in body['error'].lower()


def test_generate_passes_resolved_lora_path_and_weight_through(lora_dir):
    lora_file = lora_dir / 'my_style.safetensors'
    lora_file.write_bytes(b'fake lora weights')
    client, headers = _client_with_session()

    captured = {}

    def _fake_acestep_generate(prompt, duration=None, **kwargs):
        captured.update(kwargs)
        return ['/tmp/does-not-matter.wav'], None

    with mock.patch('pipeline.acestep_generate', side_effect=_fake_acestep_generate), \
         mock.patch('pipeline.probe_duration', return_value=12.0):
        r = client.post('/api/music/generate', data={
            'prompt': 'cinematic instrumental',
            'lora': 'my_style.safetensors',
            'lora_weight': '0.75',
        }, headers=headers, content_type='multipart/form-data')

    assert r.status_code == 200, r.get_data(as_text=True)
    body = r.get_json()
    assert body['ok'] is True
    assert body['lora'] == 'my_style.safetensors'
    assert body['lora_weight'] == 0.75
    assert captured.get('lora_path') == str(lora_file)
    assert captured.get('lora_weight') == 0.75


def test_generate_with_no_lora_selected_omits_it_entirely(lora_dir):
    client, headers = _client_with_session()
    captured = {}

    def _fake_acestep_generate(prompt, duration=None, **kwargs):
        captured.update(kwargs)
        return ['/tmp/does-not-matter.wav'], None

    with mock.patch('pipeline.acestep_generate', side_effect=_fake_acestep_generate), \
         mock.patch('pipeline.probe_duration', return_value=12.0):
        r = client.post('/api/music/generate', data={'prompt': 'cinematic instrumental'},
                         headers=headers, content_type='multipart/form-data')
    assert r.status_code == 200, r.get_data(as_text=True)
    body = r.get_json()
    assert body.get('lora') is None
    assert captured.get('lora_path') is None


def test_acestep_generate_omits_lora_fields_from_payload_when_none_given():
    # Direct unit check of the payload-building logic itself (network call
    # mocked) -- proves an ordinary generation with no LoRA selected sends
    # exactly the same payload shape as before this feature existed.
    with mock.patch('pipeline.requests.post') as mock_post:
        mock_post.side_effect = Exception('stop before actually polling')
        try:
            pipeline.acestep_generate('cinematic', lora_path=None)
        except Exception:
            pass
        payload = mock_post.call_args.kwargs.get('json') or mock_post.call_args[1].get('json')
        assert 'lora_path' not in payload
        assert 'lora_weight' not in payload


def test_acestep_generate_includes_lora_fields_when_path_exists(tmp_path):
    lora_file = tmp_path / 'style.safetensors'
    lora_file.write_bytes(b'x')
    with mock.patch('pipeline.requests.post') as mock_post:
        mock_post.side_effect = Exception('stop before actually polling')
        try:
            pipeline.acestep_generate('cinematic', lora_path=str(lora_file), lora_weight=0.6)
        except Exception:
            pass
        payload = mock_post.call_args.kwargs.get('json') or mock_post.call_args[1].get('json')
        assert payload['lora_path'] == str(lora_file.resolve()) or payload['lora_path'] == os.path.abspath(str(lora_file))
        assert payload['lora_weight'] == 0.6


# ---------------------------------------------------------------------------
# _demucs_available() / separate_stems()
# ---------------------------------------------------------------------------

def test_demucs_available_reflects_shutil_which(monkeypatch):
    monkeypatch.setattr(pipeline.shutil, 'which', lambda name: None)
    assert pipeline._demucs_available() is False
    monkeypatch.setattr(pipeline.shutil, 'which', lambda name: '/usr/bin/demucs')
    assert pipeline._demucs_available() is True


def test_separate_stems_discovers_output_files(tmp_path, monkeypatch):
    src = tmp_path / 'track.wav'
    src.write_bytes(b'fake audio')

    def _fake_run(cmd, capture_output, text, timeout):
        # Simulate demucs' own output layout: <out_dir>/<model>/<track>/*.wav
        out_dir = cmd[cmd.index('-o') + 1]
        stem_dir = os.path.join(out_dir, pipeline._DEMUCS_MODEL, 'track')
        os.makedirs(stem_dir, exist_ok=True)
        for name in ('vocals.wav', 'no_vocals.wav'):
            with open(os.path.join(stem_dir, name), 'wb') as f:
                f.write(b'stem data')
        return mock.Mock(returncode=0, stdout='', stderr='')

    monkeypatch.setattr(pipeline.subprocess, 'run', _fake_run)
    out_dir = tmp_path / 'out'
    stems, err = pipeline.separate_stems(str(src), 'vocals', str(out_dir))
    assert err is None
    labels = sorted(label for label, _path in stems)
    assert labels == ['no_vocals', 'vocals']


def test_separate_stems_reports_nonzero_exit_as_error(tmp_path, monkeypatch):
    src = tmp_path / 'track.wav'
    src.write_bytes(b'fake audio')

    def _fake_run(cmd, capture_output, text, timeout):
        return mock.Mock(returncode=1, stdout='', stderr='CUDA out of memory')

    monkeypatch.setattr(pipeline.subprocess, 'run', _fake_run)
    stems, err = pipeline.separate_stems(str(src), '4stem', str(tmp_path / 'out'))
    assert stems is None
    assert 'CUDA out of memory' in err


# ---------------------------------------------------------------------------
# /api/music/export
# ---------------------------------------------------------------------------

def test_export_missing_file_is_a_clear_404(upload_dir):
    client, headers = _client_with_session()
    r = client.get('/api/music/export?filename=nope.wav', headers=headers)
    assert r.status_code == 404


def test_export_original_format_uses_the_custom_name(upload_dir):
    (upload_dir / 'music_abc123_0.wav').write_bytes(b'fake wav data')
    client, headers = _client_with_session()
    r = client.get('/api/music/export?filename=music_abc123_0.wav&name=My Cool Track',
                    headers=headers)
    assert r.status_code == 200
    cd = r.headers.get('Content-Disposition', '')
    assert 'My_Cool_Track.wav' in cd or 'My Cool Track.wav' in cd


def test_export_wav_conversion_from_non_wav_source(upload_dir):
    if not _ffmpeg_available():
        pytest.skip('ffmpeg not available in this environment')
    src = upload_dir / 'music_abc123_0.m4a'
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'lavfi',
                    '-i', 'sine=frequency=440:duration=1', '-c:a', 'aac', str(src)],
                   check=True, timeout=30)
    client, headers = _client_with_session()
    r = client.get('/api/music/export?filename=music_abc123_0.m4a&format=wav&name=converted',
                    headers=headers)
    assert r.status_code == 200, r.get_data(as_text=True)[:300]
    assert 'converted.wav' in r.headers.get('Content-Disposition', '')
    assert r.data[:4] == b'RIFF'  # WAV container magic bytes


def test_export_stem_without_demucs_returns_503(upload_dir, monkeypatch):
    (upload_dir / 'music_abc123_0.wav').write_bytes(b'fake wav data')
    monkeypatch.setattr(pipeline, '_demucs_available', lambda: False)
    client, headers = _client_with_session()
    r = client.get('/api/music/export?filename=music_abc123_0.wav&stem=vocals', headers=headers)
    assert r.status_code == 503
    assert 'demucs' in r.get_json()['error'].lower()


def test_export_stem_zips_the_separated_files(upload_dir, tmp_path, monkeypatch):
    (upload_dir / 'music_abc123_0.wav').write_bytes(b'fake wav data')
    stem_a = tmp_path / 'vocals.wav'
    stem_b = tmp_path / 'no_vocals.wav'
    stem_a.write_bytes(b'vocals data')
    stem_b.write_bytes(b'instrumental data')

    monkeypatch.setattr(pipeline, '_demucs_available', lambda: True)
    monkeypatch.setattr(pipeline, 'separate_stems',
                         lambda src, mode, out_dir: ([('vocals', str(stem_a)), ('no_vocals', str(stem_b))], None))

    client, headers = _client_with_session()
    r = client.get('/api/music/export?filename=music_abc123_0.wav&stem=vocals&name=mytrack', headers=headers)
    assert r.status_code == 200, r.get_data(as_text=True)[:300]
    assert r.mimetype == 'application/zip'
    assert 'mytrack_stems.zip' in r.headers.get('Content-Disposition', '')

    import io
    zf = zipfile.ZipFile(io.BytesIO(r.data))
    names = sorted(zf.namelist())
    assert names == ['mytrack_no_vocals.wav', 'mytrack_vocals.wav']


def test_export_stem_error_from_separation_is_surfaced(upload_dir, monkeypatch):
    (upload_dir / 'music_abc123_0.wav').write_bytes(b'fake wav data')
    monkeypatch.setattr(pipeline, '_demucs_available', lambda: True)
    monkeypatch.setattr(pipeline, 'separate_stems', lambda src, mode, out_dir: (None, 'boom'))
    client, headers = _client_with_session()
    r = client.get('/api/music/export?filename=music_abc123_0.wav&stem=4stem', headers=headers)
    assert r.status_code == 500
    assert r.get_json()['error'] == 'boom'


def test_export_rejects_path_traversal_in_filename(upload_dir):
    client, headers = _client_with_session()
    r = client.get('/api/music/export?filename=' + '..%2F..%2Fetc%2Fpasswd', headers=headers)
    # secure_filename() strips traversal components down to a plain
    # basename that won't exist in the upload folder, so this must come
    # back as an ordinary "not found", never a file from outside uploads.
    assert r.status_code in (400, 404)

"""
User-requested: extend the LoRA feature (test_music_lora_and_export.py) so
PRISM can also organize training datasets and kick off/track a training run,
not just apply an already-trained LoRA at generation time.

Confirmed scope (via AskUserQuestion): full dataset management, with
progress tracked by watching the training-output directory for checkpoint
files rather than a live progress feed (ACE-Step's REST API documents no
training-status endpoint).

Dataset file convention under test is verbatim from ACE-Step-1.5's own
LoRA training tutorial (docs/en/LoRA_Training_Tutorial.md):
    {name}.<ext>          audio (mp3/wav/flac/ogg/opus)
    {name}.lyrics.txt      lyrics
    {name}.caption.txt     caption/style description
    {name}.json            optional {"caption","bpm","keyscale","timesignature","language"}

/v1/training/start and /v1/training/start_lokr field names are taken from
ACE-Step-1.5's own acestep/api/train_api_models.py request models.
"""
import io
import json
import os
import time
import unittest.mock as mock

import pytest

with mock.patch('requests.post'), mock.patch('requests.get'):
    import main
    import pipeline


def _client_with_session():
    app = main.app
    client = app.test_client()
    csrf_token = 'test-csrf-lora-training'
    with client.session_transaction() as sess:
        sess['authed'] = True
        sess['user_id'] = 1
        sess['username'] = 'admin'
        sess['role'] = 'admin'
        sess['csrf_token'] = csrf_token
    return client, {'X-CSRF-Token': csrf_token}


@pytest.fixture
def training_dirs(tmp_path, monkeypatch):
    data_dir = tmp_path / 'training_data'
    tensor_dir = tmp_path / 'tensors'
    output_dir = tmp_path / 'lora_output'
    lora_dir = tmp_path / 'loras'
    for d in (data_dir, tensor_dir, output_dir, lora_dir):
        d.mkdir()
    monkeypatch.setattr(pipeline, 'ACE_STEP_TRAINING_DATA_DIR', str(data_dir))
    monkeypatch.setattr(pipeline, 'ACE_STEP_TENSOR_DIR', str(tensor_dir))
    monkeypatch.setattr(pipeline, 'ACE_STEP_LORA_TRAINING_OUTPUT_DIR', str(output_dir))
    monkeypatch.setattr(pipeline, 'ACE_STEP_LORA_DIR', str(lora_dir))
    monkeypatch.setattr(pipeline, 'ALLOW_LOCAL_MEDIA_UPLOAD', True)
    return {'data': data_dir, 'tensor': tensor_dir, 'output': output_dir, 'lora': lora_dir}


# ---------------------------------------------------------------------------
# Dataset CRUD
# ---------------------------------------------------------------------------

def test_create_list_and_delete_dataset(training_dirs):
    client, headers = _client_with_session()

    r = client.post('/api/music/lora_training/datasets', data={'name': 'my show vo style'}, headers=headers)
    assert r.status_code == 200, r.get_data(as_text=True)
    name = r.get_json()['name']
    assert os.path.isdir(os.path.join(str(training_dirs['data']), name))

    r2 = client.get('/api/music/lora_training/datasets', headers=headers)
    body = r2.get_json()
    assert body['ok'] is True
    assert any(d['name'] == name and d['tracks'] == 0 for d in body['datasets'])

    r3 = client.delete(f'/api/music/lora_training/datasets/{name}', headers=headers)
    assert r3.status_code == 200
    assert not os.path.isdir(os.path.join(str(training_dirs['data']), name))


def test_creating_a_duplicate_dataset_name_is_rejected(training_dirs):
    client, headers = _client_with_session()
    client.post('/api/music/lora_training/datasets', data={'name': 'dupe'}, headers=headers)
    r = client.post('/api/music/lora_training/datasets', data={'name': 'dupe'}, headers=headers)
    assert r.status_code == 400
    assert 'already exists' in r.get_json()['error']


def test_deleting_an_unknown_dataset_is_404(training_dirs):
    client, headers = _client_with_session()
    r = client.delete('/api/music/lora_training/datasets/nope', headers=headers)
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Track upload -- exact ACE-Step scanner file convention
# ---------------------------------------------------------------------------

def test_adding_a_track_writes_the_exact_scanner_file_convention(training_dirs):
    client, headers = _client_with_session()
    client.post('/api/music/lora_training/datasets', data={'name': 'ds1'}, headers=headers)

    r = client.post('/api/music/lora_training/datasets/ds1/tracks', data={
        'audio': (io.BytesIO(b'fake mp3 bytes'), 'my_song.mp3'),
        'name': 'track_one',
        'caption': 'energetic rock anthem',
        'lyrics': '[verse]\nhello world',
        'bpm': '128',
        'keyscale': 'D major',
    }, headers=headers, content_type='multipart/form-data')
    assert r.status_code == 200, r.get_data(as_text=True)
    assert r.get_json()['name'] == 'track_one'

    ds_dir = os.path.join(str(training_dirs['data']), 'ds1')
    assert os.path.exists(os.path.join(ds_dir, 'track_one.mp3'))
    assert open(os.path.join(ds_dir, 'track_one.lyrics.txt')).read() == '[verse]\nhello world'
    assert open(os.path.join(ds_dir, 'track_one.caption.txt')).read() == 'energetic rock anthem'
    meta = json.load(open(os.path.join(ds_dir, 'track_one.json')))
    assert meta['bpm'] == 128
    assert meta['keyscale'] == 'D major'
    assert meta['caption'] == 'energetic rock anthem'


def test_instrumental_track_with_no_lyrics_omits_the_lyrics_file(training_dirs):
    client, headers = _client_with_session()
    client.post('/api/music/lora_training/datasets', data={'name': 'ds1'}, headers=headers)
    client.post('/api/music/lora_training/datasets/ds1/tracks', data={
        'audio': (io.BytesIO(b'fake wav bytes'), 'inst.wav'),
        'caption': 'ambient pad',
    }, headers=headers, content_type='multipart/form-data')
    ds_dir = os.path.join(str(training_dirs['data']), 'ds1')
    assert os.path.exists(os.path.join(ds_dir, 'inst.wav'))
    assert not os.path.exists(os.path.join(ds_dir, 'inst.lyrics.txt'))
    assert os.path.exists(os.path.join(ds_dir, 'inst.caption.txt'))


def test_duplicate_track_name_in_same_dataset_is_rejected(training_dirs):
    client, headers = _client_with_session()
    client.post('/api/music/lora_training/datasets', data={'name': 'ds1'}, headers=headers)
    data = {'audio': (io.BytesIO(b'x'), 'a.mp3'), 'name': 'a'}
    client.post('/api/music/lora_training/datasets/ds1/tracks', data=dict(data),
                headers=headers, content_type='multipart/form-data')
    r = client.post('/api/music/lora_training/datasets/ds1/tracks', data={
        'audio': (io.BytesIO(b'y'), 'a.wav'), 'name': 'a',
    }, headers=headers, content_type='multipart/form-data')
    assert r.status_code == 400
    assert 'already exists' in r.get_json()['error']


def test_rejects_a_disallowed_audio_extension(training_dirs):
    client, headers = _client_with_session()
    client.post('/api/music/lora_training/datasets', data={'name': 'ds1'}, headers=headers)
    r = client.post('/api/music/lora_training/datasets/ds1/tracks', data={
        'audio': (io.BytesIO(b'not audio'), 'evil.exe'),
    }, headers=headers, content_type='multipart/form-data')
    assert r.status_code == 400


def test_upload_disabled_returns_403(training_dirs, monkeypatch):
    monkeypatch.setattr(pipeline, 'ALLOW_LOCAL_MEDIA_UPLOAD', False)
    client, headers = _client_with_session()
    client.post('/api/music/lora_training/datasets', data={'name': 'ds1'}, headers=headers)
    r = client.post('/api/music/lora_training/datasets/ds1/tracks', data={
        'audio': (io.BytesIO(b'x'), 'a.mp3'),
    }, headers=headers, content_type='multipart/form-data')
    assert r.status_code == 403


def test_list_tracks_reports_tensor_dir_suggestion(training_dirs):
    client, headers = _client_with_session()
    client.post('/api/music/lora_training/datasets', data={'name': 'ds1'}, headers=headers)
    client.post('/api/music/lora_training/datasets/ds1/tracks', data={
        'audio': (io.BytesIO(b'x'), 'a.mp3'), 'caption': 'test',
    }, headers=headers, content_type='multipart/form-data')
    r = client.get('/api/music/lora_training/datasets/ds1/tracks', headers=headers)
    body = r.get_json()
    assert body['ok'] is True
    assert len(body['tracks']) == 1
    assert body['tracks'][0]['has_caption'] is True
    assert body['tracks'][0]['has_lyrics'] is False
    assert body['tensor_dir_suggestion'] == os.path.join(str(training_dirs['tensor']), 'ds1')


def test_delete_track_removes_all_its_files(training_dirs):
    client, headers = _client_with_session()
    client.post('/api/music/lora_training/datasets', data={'name': 'ds1'}, headers=headers)
    client.post('/api/music/lora_training/datasets/ds1/tracks', data={
        'audio': (io.BytesIO(b'x'), 'a.mp3'), 'caption': 'c', 'lyrics': 'l',
    }, headers=headers, content_type='multipart/form-data')
    ds_dir = os.path.join(str(training_dirs['data']), 'ds1')
    assert len(os.listdir(ds_dir)) == 4  # audio + lyrics + caption + json
    r = client.delete('/api/music/lora_training/datasets/ds1/tracks/a', headers=headers)
    assert r.status_code == 200
    assert os.listdir(ds_dir) == []


# ---------------------------------------------------------------------------
# Training kickoff
# ---------------------------------------------------------------------------

def test_start_training_requires_tensor_dir(training_dirs):
    client, headers = _client_with_session()
    r = client.post('/api/music/lora_training/start', data={}, headers=headers)
    assert r.status_code == 400


def test_start_training_posts_lora_fields_and_keeps_output_dir_scoped(training_dirs):
    client, headers = _client_with_session()
    with mock.patch('pipeline.requests.post') as mock_post:
        mock_post.return_value = mock.Mock(ok=True, status_code=200, json=lambda: {'accepted': True})
        r = client.post('/api/music/lora_training/start', data={
            'tensor_dir': '/data/tensors/ds1',
            'lora_rank': '32', 'lora_alpha': '64', 'learning_rate': '0.0002',
            'train_epochs': '20', 'train_batch_size': '2',
        }, headers=headers)
    assert r.status_code == 200, r.get_data(as_text=True)
    body = r.get_json()
    assert body['ok'] is True
    assert body['lokr'] is False
    # output_dir must be a subfolder of ACE_STEP_LORA_TRAINING_OUTPUT_DIR --
    # never a raw caller-supplied path -- so use_checkpoint's later
    # containment check can trust it.
    assert body['output_dir'].startswith(str(training_dirs['output']))

    call_url = mock_post.call_args[0][0]
    payload = mock_post.call_args.kwargs.get('json')
    assert call_url == f'{pipeline.ACE_STEP_URL}/v1/training/start'
    assert payload['tensor_dir'] == '/data/tensors/ds1'
    assert payload['lora_rank'] == 32
    assert payload['lora_alpha'] == 64
    assert payload['learning_rate'] == 0.0002
    assert payload['train_epochs'] == 20
    assert payload['train_batch_size'] == 2
    assert payload['lora_output_dir'] == body['output_dir']


def test_start_training_lokr_variant_hits_the_lokr_endpoint(training_dirs):
    client, headers = _client_with_session()
    with mock.patch('pipeline.requests.post') as mock_post:
        mock_post.return_value = mock.Mock(ok=True, status_code=200, json=lambda: {'accepted': True})
        r = client.post('/api/music/lora_training/start', data={
            'tensor_dir': '/data/tensors/ds1', 'lokr': '1',
        }, headers=headers)
    assert r.status_code == 200, r.get_data(as_text=True)
    assert r.get_json()['lokr'] is True
    call_url = mock_post.call_args[0][0]
    payload = mock_post.call_args.kwargs.get('json')
    assert call_url == f'{pipeline.ACE_STEP_URL}/v1/training/start_lokr'
    assert 'output_dir' in payload
    assert 'lokr_linear_dim' in payload


def test_start_training_surfaces_ace_step_rejection(training_dirs):
    client, headers = _client_with_session()
    with mock.patch('pipeline.requests.post') as mock_post:
        mock_post.return_value = mock.Mock(ok=False, status_code=400, json=lambda: {'detail': 'bad tensor_dir'})
        r = client.post('/api/music/lora_training/start', data={'tensor_dir': '/nope'}, headers=headers)
    assert r.status_code == 502
    assert 'bad tensor_dir' in r.get_json()['error']


def test_start_training_connection_failure_is_reported(training_dirs):
    import requests as _requests
    client, headers = _client_with_session()
    with mock.patch('pipeline.requests.post', side_effect=_requests.exceptions.ConnectionError('refused')):
        r = client.post('/api/music/lora_training/start', data={'tensor_dir': '/data/tensors/ds1'}, headers=headers)
    assert r.status_code == 502
    assert 'Could not reach ACE-Step' in r.get_json()['error']


# ---------------------------------------------------------------------------
# Status watching
# ---------------------------------------------------------------------------

def test_status_reports_no_checkpoint_yet_for_nonexistent_dir(training_dirs):
    client, headers = _client_with_session()
    target = os.path.join(str(training_dirs['output']), 'run123')
    r = client.get(f'/api/music/lora_training/status?dir={target}', headers=headers)
    assert r.status_code == 200
    body = r.get_json()
    assert body['exists'] is False
    assert body['likely_done'] is False


def test_status_finds_newest_checkpoint_and_flags_stale_as_likely_done(training_dirs, monkeypatch):
    monkeypatch.setattr(pipeline, 'TRAINING_STALE_AFTER', 60)
    run_dir = training_dirs['output'] / 'run1'
    run_dir.mkdir()
    old_ckpt = run_dir / 'epoch_5.safetensors'
    old_ckpt.write_bytes(b'x' * 1000)
    old_time = time.time() - 3600
    os.utime(old_ckpt, (old_time, old_time))

    client, headers = _client_with_session()
    r = client.get(f'/api/music/lora_training/status?dir={run_dir}', headers=headers)
    body = r.get_json()
    assert body['ok'] is True
    assert body['exists'] is True
    assert len(body['checkpoints']) == 1
    assert body['likely_done'] is True
    assert 'likely' in body['note'].lower()


def test_status_recent_checkpoint_is_not_flagged_done(training_dirs, monkeypatch):
    monkeypatch.setattr(pipeline, 'TRAINING_STALE_AFTER', 3600)
    run_dir = training_dirs['output'] / 'run1'
    run_dir.mkdir()
    (run_dir / 'epoch_1.safetensors').write_bytes(b'x' * 1000)

    client, headers = _client_with_session()
    r = client.get(f'/api/music/lora_training/status?dir={run_dir}', headers=headers)
    body = r.get_json()
    assert body['likely_done'] is False


def test_status_rejects_a_dir_outside_the_managed_output_root(training_dirs):
    client, headers = _client_with_session()
    r = client.get('/api/music/lora_training/status?dir=/etc', headers=headers)
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# Promoting a checkpoint into the LoRA picker
# ---------------------------------------------------------------------------

def test_use_checkpoint_copies_into_ace_step_lora_dir(training_dirs):
    run_dir = training_dirs['output'] / 'run1'
    run_dir.mkdir()
    ckpt = run_dir / 'epoch_10.safetensors'
    ckpt.write_bytes(b'trained weights')

    client, headers = _client_with_session()
    r = client.post('/api/music/lora_training/use_checkpoint',
                     data={'path': str(ckpt), 'name': 'my_style'}, headers=headers)
    assert r.status_code == 200, r.get_data(as_text=True)
    body = r.get_json()
    assert body['name'] == 'my_style.safetensors'
    dest = os.path.join(str(training_dirs['lora']), 'my_style.safetensors')
    assert os.path.exists(dest)
    assert open(dest, 'rb').read() == b'trained weights'

    # And it now shows up in the ordinary LoRA picker used at generation time.
    r2 = client.get('/api/music/loras', headers=headers)
    assert 'my_style.safetensors' in r2.get_json()['loras']


def test_use_checkpoint_rejects_a_path_outside_the_managed_training_output_dir(training_dirs, tmp_path):
    outside = tmp_path / 'outside.safetensors'
    outside.write_bytes(b'not from a real training run')
    client, headers = _client_with_session()
    r = client.post('/api/music/lora_training/use_checkpoint', data={'path': str(outside)}, headers=headers)
    assert r.status_code == 400
    assert 'outside' in r.get_json()['error'].lower()


def test_use_checkpoint_missing_file_is_404(training_dirs):
    client, headers = _client_with_session()
    r = client.post('/api/music/lora_training/use_checkpoint',
                     data={'path': str(training_dirs['output'] / 'nope.safetensors')}, headers=headers)
    assert r.status_code == 404

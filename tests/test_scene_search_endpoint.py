"""
Regression coverage for /api/vision/analyze, reworked from a plain
"describe N sampled frames" tool into Scene Search: detect every cut in a
hires video, describe each with AI Vision, align it with Whisper dialogue,
then rank/filter against a typed prompt and an optional negative prompt --
deliberately stopping short of the full OpenCV+AI+dialogue *rating*
pipeline the real promo generator runs (no score, no selection).

Uses a real multi-scene video (three distinct color segments so
PySceneDetect reliably finds three hard cuts) with the AI Vision call
(Ollama) and dialogue transcription (Whisper) both mocked, so the test
exercises the endpoint's own detect -> describe -> align -> rank/filter
logic without depending on either external service being reachable.
"""
import shutil
import subprocess
import unittest.mock as mock

import pytest

with mock.patch('requests.post'), mock.patch('requests.get'):
    import main
    import pipeline


def _ffmpeg_available():
    return shutil.which('ffmpeg') is not None


def _client_with_session():
    app = main.app
    client = app.test_client()
    csrf_token = 'test-csrf-scene-search-endpoint'
    with client.session_transaction() as sess:
        sess['authed'] = True
        sess['user_id'] = 1
        sess['username'] = 'admin'
        sess['role'] = 'admin'
        sess['csrf_token'] = csrf_token
    return client, {'X-CSRF-Token': csrf_token}


@pytest.fixture
def three_scene_video(tmp_path, monkeypatch):
    if not _ffmpeg_available():
        pytest.skip('ffmpeg not available in this environment')
    monkeypatch.setattr(pipeline, 'ALLOW_LOCAL_MEDIA_UPLOAD', True)
    src = tmp_path / 'src.mp4'
    parts = []
    for i, color in enumerate(['red', 'blue', 'green']):
        part = tmp_path / f'part{i}.mp4'
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'lavfi',
                        '-i', f'color=c={color}:s=320x240:d=3:r=25',
                        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(part)],
                       check=True, timeout=30)
        parts.append(part)
    list_file = tmp_path / 'list.txt'
    list_file.write_text('\n'.join(f"file '{p}'" for p in parts))
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0',
                    '-i', str(list_file), '-c', 'copy', str(src)], check=True, timeout=30)
    return src


# Three scenes at roughly [0-3), [3-6), [6-9)s. Each fake vision description
# names a distinct subject so prompt matching has something real to key off.
_FAKE_DESCRIPTIONS = [
    'A red car chasing down a highway at night.',
    'A calm blue ocean with a sailboat drifting by.',
    'A person running through a green forest in daylight.',
]


def _run_search(client, headers, video_path, prompt='', negative_prompt=''):
    with mock.patch('pipeline.transcribe_video', return_value=([], [])), \
         mock.patch('pipeline.requests.post') as mock_post, \
         mock.patch('pipeline.unload_ollama_model'):
        call_count = {'n': 0}

        def _fake_post(url, **kwargs):
            resp = mock.Mock()
            i = call_count['n']
            call_count['n'] += 1
            resp.json.return_value = {'response': _FAKE_DESCRIPTIONS[i % len(_FAKE_DESCRIPTIONS)]}
            return resp
        mock_post.side_effect = _fake_post

        with open(video_path, 'rb') as f:
            data = {'file': (f, 'src.mp4'), 'scene_threshold': '10', 'min_scene_len': '0.5'}
            if prompt:
                data['prompt'] = prompt
            if negative_prompt:
                data['negative_prompt'] = negative_prompt
            r = client.post('/api/vision/analyze', data=data, headers=headers,
                             content_type='multipart/form-data')
        return r


def test_no_prompt_returns_every_detected_scene_unfiltered(three_scene_video):
    client, headers = _client_with_session()
    r = _run_search(client, headers, three_scene_video)
    assert r.status_code == 200, r.get_data(as_text=True)
    body = r.get_json()
    assert body['total_scenes'] == len(body['results']) == 3
    assert body['searched'] is False
    for res in body['results']:
        assert res['description']
        assert res['thumb']


def test_positive_prompt_ranks_matching_scene_first(three_scene_video):
    client, headers = _client_with_session()
    r = _run_search(client, headers, three_scene_video, prompt='car highway')
    assert r.status_code == 200, r.get_data(as_text=True)
    body = r.get_json()
    assert body['searched'] is True
    assert body['results'], 'expected at least the car scene to match'
    assert 'car' in body['results'][0]['description'].lower()
    assert body['results'][0]['match_hits'] >= 1


def test_negative_prompt_excludes_matching_scenes(three_scene_video):
    client, headers = _client_with_session()
    r = _run_search(client, headers, three_scene_video, negative_prompt='ocean sailboat')
    assert r.status_code == 200, r.get_data(as_text=True)
    body = r.get_json()
    assert body['excluded_count'] == 1
    assert all('sailboat' not in res['description'].lower() for res in body['results'])
    assert len(body['results']) == body['total_scenes'] - 1


def test_no_scenes_detected_returns_a_clear_error(tmp_path, monkeypatch):
    if not _ffmpeg_available():
        pytest.skip('ffmpeg not available in this environment')
    monkeypatch.setattr(pipeline, 'ALLOW_LOCAL_MEDIA_UPLOAD', True)
    # A single flat-color clip with an extremely high sensitivity threshold
    # produces zero detected cuts.
    src = tmp_path / 'flat.mp4'
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'lavfi',
                    '-i', 'color=c=gray:s=320x240:d=2:r=25',
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(src)],
                   check=True, timeout=30)
    client, headers = _client_with_session()
    with open(src, 'rb') as f:
        r = client.post('/api/vision/analyze', data={'file': (f, 'flat.mp4'), 'scene_threshold': '100'},
                         headers=headers, content_type='multipart/form-data')
    assert r.status_code == 400
    assert 'no scene cuts' in (r.get_json().get('error') or '').lower()

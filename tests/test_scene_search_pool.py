"""
Regression coverage for the Proposed cut screen's "Scene search" feature:
the preview response's `alternates` list is now the FULL detected-scene
pool minus whatever got selected (searched by typed description on the
frontend), not a short, score-ranked runner-up shortlist capped at
PREVIEW_ALTERNATES (12) -- see run_trailer_job's preview_only branch in
pipeline.py.

Uses a real multi-scene video (six distinct color segments, so
PySceneDetect reliably finds six hard cuts) and a short target length so
only a couple of scenes get selected, leaving a real, non-trivial pool of
alternates to assert against.
"""
import shutil
import subprocess
import time
import unittest.mock as mock

import pytest

import core
import pipeline

with mock.patch('requests.post'), mock.patch('requests.get'):
    import main


def _ffmpeg_available():
    return shutil.which('ffmpeg') is not None


@pytest.fixture
def multi_scene_preview(tmp_path, monkeypatch):
    if not _ffmpeg_available():
        pytest.skip('ffmpeg not available in this environment')
    app = main.app
    upload_dir = tmp_path / 'uploads'
    upload_dir.mkdir()
    monkeypatch.setitem(app.config, 'UPLOAD_FOLDER', str(upload_dir))
    monkeypatch.setattr(pipeline, 'ALLOW_LOCAL_MEDIA_UPLOAD', True)

    client = app.test_client()
    csrf_token = 'test-csrf-scene-search'
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

    staged_name = 'net_3000_src.mp4'
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
    assert result.get('total_scenes'), 'expected PySceneDetect to find multiple scenes in the fixture video'
    return result


def test_alternates_cover_the_full_non_selected_pool(multi_scene_preview):
    result = multi_scene_preview
    total = result['total_scenes']
    selected = result['selected_scenes']
    alternates = result.get('alternates') or []
    # The old behavior capped this at PREVIEW_ALTERNATES (12) regardless of
    # how many scenes existed; the new scene-search pool must cover every
    # detected scene that isn't already in the cut.
    assert len(alternates) == total - selected


def test_alternates_have_no_duplicate_scene_numbers_or_overlap_with_selected(multi_scene_preview):
    result = multi_scene_preview
    selected_starts = {(s.get('material'), round(s['start'], 3)) for s in result['scenes']}
    alt_starts = [(a.get('material'), round(a['start'], 3)) for a in (result.get('alternates') or [])]
    assert len(alt_starts) == len(set(alt_starts)), 'alternates must not contain duplicate scenes'
    assert not (set(alt_starts) & selected_starts), 'alternates must exclude scenes already selected'


def test_alternates_carry_description_and_thumb_for_search(multi_scene_preview):
    result = multi_scene_preview
    alternates = result.get('alternates') or []
    assert alternates, 'fixture should leave at least one scene out of the cut to search for'
    for alt in alternates:
        assert 'description' in alt
        assert 'thumb' in alt
        assert 'alt' in alt and isinstance(alt['alt'], int)

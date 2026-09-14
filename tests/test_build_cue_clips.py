"""
Tests for build_cue_clips(): when script/manual timecode cues exist, the
final promo uses ONLY those cues (user-requested and confirmed) -- this
builds each cue's own in/out span directly into a clip, bypassing
PySceneDetect's own scene boundaries entirely (a cue may legitimately span
several of its own shot changes combined into one cut), tagged with the
correct material and source file so a script-matched clip is never pulled
from the wrong source.

Uses real, generated video files (via ffmpeg) so probe_duration -- which
build_cue_clips calls to cap how far a cue can be stretched -- has a real
file to read, not a mocked one.
"""
import os
import subprocess
import time
import json
import unittest.mock as mock

import pytest

with mock.patch('requests.post'), mock.patch('requests.get'):
    import main
import pipeline

FFMPEG = os.environ.get('FFMPEG_BIN', 'ffmpeg')


def _make_video(path, duration_sec):
    subprocess.run([
        FFMPEG, '-y', '-loglevel', 'error',
        '-f', 'lavfi', '-i', f'color=c=red:s=320x240:d={duration_sec}:r=25',
        '-f', 'lavfi', '-i', f'anullsrc=r=44100:cl=stereo:d={duration_sec}',
        '-c:v', 'libx264', '-preset', 'ultrafast', '-c:a', 'aac', '-shortest', path,
    ], check=True)


@pytest.fixture(scope='module')
def two_materials(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp('build_cue_clips_fixtures')
    mat1 = str(tmp_dir / 'mat1.mp4')
    mat2 = str(tmp_dir / 'mat2.mp4')
    _make_video(mat1, 300)
    _make_video(mat2, 300)
    return [mat1, mat2]


def test_single_cue_with_out_uses_its_own_in_out_span(two_materials):
    cues = [{'time': 213, 'out': 217, 'desc': 'M1 HABOL TACKLE', 'material': 1}]
    clips = pipeline.build_cue_clips(cues, two_materials)
    assert len(clips) == 1
    c = clips[0]
    assert c['start'] == 213.0
    assert c['trim_start'] == 213.0
    assert c['selected_dur'] == 4.0  # 217 - 213
    assert c['material'] == 1
    assert c['source_path'] == two_materials[0]
    assert c['script_desc'] == 'M1 HABOL TACKLE'


def test_cue_with_no_out_uses_the_default_duration(two_materials):
    cues = [{'time': 50, 'out': None, 'desc': 'M2', 'material': 2}]
    clips = pipeline.build_cue_clips(cues, two_materials, default_cue_dur=4.0)
    assert len(clips) == 1
    assert clips[0]['selected_dur'] == 4.0
    assert clips[0]['material'] == 2
    assert clips[0]['source_path'] == two_materials[1]


def test_two_cues_from_different_materials_both_produced_correctly(two_materials):
    # The exact reported scenario: M1 3:33-3:37, M2 0:35-0:42.
    cues = [
        {'time': 213, 'out': 217, 'desc': 'M1 HABOL TACKLE', 'material': 1},
        {'time': 35, 'out': 42, 'desc': 'M2 BUNOT', 'material': 2},
    ]
    clips = pipeline.build_cue_clips(cues, two_materials)
    assert len(clips) == 2
    # Ordered by (material, start) -- material 1 first, matching entry
    # order, not by raw numeric time (which would put material 2's smaller
    # timestamp first despite being entered/listed second).
    assert clips[0]['material'] == 1 and clips[0]['start'] == 213.0
    assert clips[1]['material'] == 2 and clips[1]['start'] == 35.0
    assert clips[0]['selected_dur'] == 4.0
    assert clips[1]['selected_dur'] == 7.0


def test_stretch_ceiling_is_capped_by_the_next_cue_in_the_same_material(two_materials):
    # Two cues in the SAME material: the first's stretch ceiling must stop
    # at the second's own start, never overlapping into footage the second
    # cue already claims.
    cues = [
        {'time': 10, 'out': 12, 'desc': 'first', 'material': 1},
        {'time': 20, 'out': 22, 'desc': 'second', 'material': 1},
    ]
    clips = pipeline.build_cue_clips(cues, two_materials)
    first = next(c for c in clips if c['script_desc'] == 'first')
    assert first['duration'] == 10.0  # ceiling capped at the next cue's start (20), not the 300s file


def test_stretch_ceiling_for_the_last_cue_in_a_material_is_the_files_own_duration(two_materials):
    cues = [{'time': 10, 'out': 12, 'desc': 'only one', 'material': 1}]
    clips = pipeline.build_cue_clips(cues, two_materials)
    assert clips[0]['duration'] == 290.0  # 300s file - 10s start


def test_rebalancing_after_build_cue_clips_can_stretch_both_cues_proportionally(two_materials):
    cues = [
        {'time': 213, 'out': 217, 'desc': 'M1', 'material': 1},
        {'time': 35, 'out': 42, 'desc': 'M2', 'material': 2},
    ]
    clips = pipeline.build_cue_clips(cues, two_materials)
    rebalanced = pipeline._rebalance_selected_durations(clips, target_duration=15.0, min_seg_dur=0.8)
    total = sum(c['selected_dur'] for c in rebalanced)
    assert abs(total - 15.0) < 0.01
    # Both grew (proportionally) from their original 4s/7s spans, not just one.
    assert rebalanced[0]['selected_dur'] > 4.0
    assert rebalanced[1]['selected_dur'] > 7.0


def test_cue_naming_a_material_outside_the_loaded_range_is_skipped(two_materials):
    cues = [
        {'time': 10, 'out': 12, 'desc': 'valid', 'material': 1},
        {'time': 20, 'out': 22, 'desc': 'invalid', 'material': 3},  # only 2 materials loaded
    ]
    clips = pipeline.build_cue_clips(cues, two_materials)
    assert len(clips) == 1
    assert clips[0]['script_desc'] == 'valid'


def test_cue_with_no_material_at_all_is_skipped(two_materials):
    cues = [{'time': 10, 'out': 12, 'desc': 'unlabeled'}]
    clips = pipeline.build_cue_clips(cues, two_materials)
    assert clips == []


def test_empty_cues_returns_empty_list(two_materials):
    assert pipeline.build_cue_clips([], two_materials) == []


def test_clips_include_an_end_field_matching_start_plus_duration(two_materials):
    # Regression guard: a real crash (KeyError: 'end') happened because an
    # earlier version of this function didn't set 'end' at all, and
    # downstream result-building code reads it directly.
    cues = [{'time': 10, 'out': 15, 'desc': 'x', 'material': 1}]
    clips = pipeline.build_cue_clips(cues, two_materials)
    assert clips[0]['end'] == clips[0]['start'] + clips[0]['duration']


def test_preview_path_does_not_crash_with_cue_exclusive_selection(two_materials, tmp_path, monkeypatch):
    # Regression guard for a real, user-reported crash: "Writing preview
    # thumbnails" -> "cannot access free variable 'min_gap'". Cause: the
    # cue-exclusive selection branch in _run_trailer_job never assigned
    # min_gap at all (only the narration/best-scenes branch did), but the
    # preview path's alternates-building code further down references it
    # unconditionally regardless of which branch ran. Exercises the real,
    # full preview pipeline end to end -- a plain unit test of
    # build_cue_clips alone can't catch this, since the bug is about a
    # variable's scope across a much larger function.
    import io as _io
    import unittest.mock as mock
    import core

    monkeypatch.setattr(pipeline, 'ALLOW_LOCAL_MEDIA_UPLOAD', True)
    app = main.app
    client = app.test_client()
    with client.session_transaction() as sess:
        sess['authed'] = True
        sess['user_id'] = 1
        sess['username'] = 'admin'
        sess['role'] = 'admin'
        sess['csrf_token'] = 'test-csrf-cue-preview'
    headers = {'X-CSRF-Token': 'test-csrf-cue-preview'}

    upload_folder = app.config['UPLOAD_FOLDER']
    os.makedirs(upload_folder, exist_ok=True)
    net1, net2 = 'net_cue_preview_1.mp4', 'net_cue_preview_2.mp4'
    import shutil
    shutil.copy(two_materials[0], os.path.join(upload_folder, net1))
    shutil.copy(two_materials[1], os.path.join(upload_folder, net2))

    with open(two_materials[0], 'rb') as f:
        video_bytes = f.read()

    entries = [
        {'material': 1, 'start': '3:33', 'end': '3:37'},
        {'material': 2, 'start': '0:35', 'end': '0:42'},
    ]
    core._job_submit_limiter.buckets.clear()
    with mock.patch('requests.post'), mock.patch('requests.get'):
        r = client.post('/api/trailer/generate', data={
            'file': (_io.BytesIO(video_bytes), 'mat1.mp4'),
            'genre': '', 'trailer_length': '15',
            'scoring_mode': 'none', 'sfx_mode': 'none', 'vo_mode': 'none',
            'manual_cues': json.dumps(entries),
            'materials_network': json.dumps([net1, net2]),
            'preview_only': '1',
        }, headers=headers, content_type='multipart/form-data')
        assert r.status_code == 200, r.get_data(as_text=True)
        job_id = r.get_json()['job_id']
        d = None
        for _ in range(60):
            d = client.get(f'/api/trailer/progress/{job_id}').get_json()
            if d.get('done'):
                break
            time.sleep(1)
    assert d.get('error') is None, d.get('error')
    result = d.get('result') or {}
    scenes = result.get('scenes') or []
    assert len(scenes) == 2
    assert {s['material'] for s in scenes} == {1, 2}

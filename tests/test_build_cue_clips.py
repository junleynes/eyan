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


def test_cue_exclusive_render_skips_scene_detection_and_ai_scoring_entirely(two_materials, monkeypatch):
    # User-requested and confirmed: when timecode cues are present, the app
    # should skip scene selection, scene rating, and AI vision rating
    # entirely -- there's already a selected scene (the cue's own in/out),
    # so scoring alternatives nobody will use is pure wasted work (real,
    # measurable time, and previously produced real Ollama/Whisper
    # connection-refused errors in the logs for a service the render
    # didn't actually need). Confirms this at the HTTP layer, not just by
    # reading code: mocks requests.post/get (the transport AI vision and
    # Whisper both go through) and asserts NEITHER is ever called for a
    # cue-driven render, while the render itself still succeeds correctly.
    import io as _io
    import shutil
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
        sess['csrf_token'] = 'test-csrf-cue-skip'
    headers = {'X-CSRF-Token': 'test-csrf-cue-skip'}

    upload_folder = app.config['UPLOAD_FOLDER']
    os.makedirs(upload_folder, exist_ok=True)
    net1, net2 = 'net_cue_skip_1.mp4', 'net_cue_skip_2.mp4'
    shutil.copy(two_materials[0], os.path.join(upload_folder, net1))
    shutil.copy(two_materials[1], os.path.join(upload_folder, net2))

    with open(two_materials[0], 'rb') as f:
        video_bytes = f.read()

    entries = [
        {'material': 1, 'start': '3:33', 'end': '3:37'},
        {'material': 2, 'start': '0:35', 'end': '0:42'},
    ]
    core._job_submit_limiter.buckets.clear()
    with mock.patch('requests.post') as mock_post, mock.patch('requests.get') as mock_get:
        r = client.post('/api/trailer/generate', data={
            'file': (_io.BytesIO(video_bytes), 'mat1.mp4'),
            'genre': '', 'trailer_length': '15',
            'scoring_mode': 'none', 'sfx_mode': 'none', 'vo_mode': 'none',
            'manual_cues': json.dumps(entries),
            'materials_network': json.dumps([net1, net2]),
            'mode': 'ai',            # AI vision rating requested...
            'whisper_enhance': '1',  # ...and dialogue transcription too
        }, headers=headers, content_type='multipart/form-data')
        assert r.status_code == 200, r.get_data(as_text=True)
        job_id = r.get_json()['job_id']
        d = None
        for _ in range(60):
            d = client.get(f'/api/trailer/progress/{job_id}').get_json()
            if d.get('done'):
                break
            time.sleep(0.5)
        assert d.get('error') is None, d.get('error')
        # ...but neither Ollama (AI vision) nor Whisper (transcription) --
        # both real HTTP calls via requests.post -- was ever actually
        # reached, because cues drove selection exclusively and both
        # scoring passes are skipped entirely when cues are present.
        assert mock_post.call_count == 0

    result = d.get('result') or {}
    scenes = result.get('scenes') or []
    assert len(scenes) == 2
    assert {s['material'] for s in scenes} == {1, 2}


def test_preview_generates_real_thumbnails_and_per_scene_video_filename_for_cue_clips(two_materials, monkeypatch):
    # Regression guard for a real, user-reported bug pair from an actual
    # preview: "no thumbnail" for cue-selected scenes, and a material-2
    # scene's Play button actually playing material 1's own footage.
    #
    # Root cause of "no thumbnail": _thumb() only ever read a pre-captured
    # 'frame' key, set by _score_one_scene() during ordinary scene scoring
    # -- but build_cue_clips' own clips never go through that scoring at
    # all, so they never had a 'frame' to read, and _thumb() correctly (but
    # unhelpfully) returned None every time.
    #
    # Root cause of "wrong material's video": the preview response's
    # top-level video_filename is ONE shared file (the primary source),
    # and the frontend's Play button used that same one value for every
    # single scene card regardless of which material a given scene
    # actually belonged to.
    import io as _io
    import shutil
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
        sess['csrf_token'] = 'test-csrf-thumb'
    headers = {'X-CSRF-Token': 'test-csrf-thumb'}

    upload_folder = app.config['UPLOAD_FOLDER']
    os.makedirs(upload_folder, exist_ok=True)
    net1, net2 = 'net_thumb_1.mp4', 'net_thumb_2.mp4'
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
            time.sleep(0.5)
    assert d.get('error') is None, d.get('error')
    result = d.get('result') or {}
    scenes = result.get('scenes') or []
    assert len(scenes) == 2

    by_material = {s['material']: s for s in scenes}
    # Each scene has a real thumbnail (not None), and the underlying file
    # genuinely exists with real content.
    for mat, s in by_material.items():
        assert s.get('thumb'), f"material {mat} has no thumbnail"
        thumb_path = os.path.join(upload_folder, os.path.basename(s['thumb']))
        assert os.path.exists(thumb_path)
        assert os.path.getsize(thumb_path) > 0

    # Each scene's own video_filename matches ITS OWN material's staged
    # file, not a single shared one across both.
    assert by_material[1]['video_filename'] == net1
    assert by_material[2]['video_filename'] == net2
    assert by_material[1]['video_filename'] != by_material[2]['video_filename']


def test_lock_and_render_preserves_each_scenes_own_material_and_source(tmp_path, monkeypatch):
    # Regression guard for a real, user-reported bug: the PREVIEW correctly
    # showed each scene's own material and footage (fixed in the previous
    # commit), but approving that preview and rendering it for real (the
    # "lock and render" flow -- POST /api/trailer/render with a preview_id,
    # which reuses the preview's own STORED scene list rather than
    # re-selecting) produced a final video where material 2's own clip was
    # actually extracted from material 1's file instead.
    #
    # Root cause: _slim() -- which builds exactly what gets stored server-
    # side for a preview and later reused as-is for the real render --
    # dropped 'material' and 'source_path' entirely. The live preview
    # itself never goes through _slim() (it renders straight from the full
    # `selected` list, which is why the preview looked correct), but the
    # STORED version handed to a later, real render lost this information
    # completely -- so the actual final-render extraction step's own
    # `seg.get('source_path') or path` fallback silently used the single
    # primary source for every clip once a preview was actually approved
    # and rendered, not while merely previewing it.
    #
    # Exercises the real, full two-step flow end to end: a real preview,
    # then a real POST to /api/trailer/render with that preview's own ID
    # (not a direct /api/trailer/generate call, which wouldn't exercise
    # _slim() or the stored-preview code path at all) -- then verifies the
    # ACTUAL rendered video's own pixel content, not just its metadata.
    # Uses its own, dedicated red/blue materials (not the shared
    # two_materials fixture, which is red for both -- fine for the other
    # tests in this file, but useless for telling two clips apart by their
    # actual pixel content, which is the entire point here).
    import io as _io
    import shutil
    import unittest.mock as mock
    import core
    import cv2

    mat1 = str(tmp_path / 'lock_render_red.mp4')
    mat2 = str(tmp_path / 'lock_render_blue.mp4')
    for path, color in ((mat1, 'red'), (mat2, 'blue')):
        subprocess.run([
            FFMPEG, '-y', '-loglevel', 'error',
            '-f', 'lavfi', '-i', f'color=c={color}:s=320x240:d=300:r=25',
            '-f', 'lavfi', '-i', 'anullsrc=r=44100:cl=stereo:d=300',
            '-c:v', 'libx264', '-preset', 'ultrafast', '-c:a', 'aac', '-shortest', path,
        ], check=True)

    monkeypatch.setattr(pipeline, 'ALLOW_LOCAL_MEDIA_UPLOAD', True)
    app = main.app
    client = app.test_client()
    with client.session_transaction() as sess:
        sess['authed'] = True
        sess['user_id'] = 1
        sess['username'] = 'admin'
        sess['role'] = 'admin'
        sess['csrf_token'] = 'test-csrf-lock-render'
    headers = {'X-CSRF-Token': 'test-csrf-lock-render'}

    upload_folder = app.config['UPLOAD_FOLDER']
    os.makedirs(upload_folder, exist_ok=True)
    net1, net2 = 'net_lock_render_1.mp4', 'net_lock_render_2.mp4'
    shutil.copy(mat1, os.path.join(upload_folder, net1))  # red
    shutil.copy(mat2, os.path.join(upload_folder, net2))  # blue

    with open(mat1, 'rb') as f:
        video_bytes = f.read()

    entries = [
        {'material': 1, 'start': '3:33', 'end': '3:37'},
        {'material': 2, 'start': '0:35', 'end': '0:42'},
    ]

    with mock.patch('requests.post'), mock.patch('requests.get'):
        # Step 1: real preview.
        core._job_submit_limiter.buckets.clear()
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
            time.sleep(0.5)
        assert d.get('error') is None, d.get('error')
        preview_id = (d.get('result') or {}).get('preview_id')
        assert preview_id

        # Step 2: lock and render -- approve the preview exactly as-is.
        core._job_submit_limiter.buckets.clear()
        r2 = client.post('/api/trailer/render', data={'preview_id': preview_id}, headers=headers)
        assert r2.status_code == 200, r2.get_data(as_text=True)
        job_id2 = r2.get_json()['job_id']
        d2 = None
        for _ in range(60):
            d2 = client.get(f'/api/trailer/progress/{job_id2}').get_json()
            if d2.get('done'):
                break
            time.sleep(0.5)
    assert d2.get('error') is None, d2.get('error')

    result2 = d2.get('result') or {}
    final_scenes = result2.get('scenes') or []
    assert len(final_scenes) == 2
    assert {s['material'] for s in final_scenes} == {1, 2}

    trailer_url = result2.get('trailer_url')
    assert trailer_url
    trailer_path = os.path.join(upload_folder, os.path.basename(trailer_url))
    assert os.path.exists(trailer_path)

    cap = cv2.VideoCapture(trailer_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 5)
    ok1, frame1 = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total_frames - 10))
    ok2, frame2 = cap.read()
    cap.release()
    assert ok1 and ok2

    def _dominant(frame):
        b, g, r_ = frame[:, :, 0].mean(), frame[:, :, 1].mean(), frame[:, :, 2].mean()
        if r_ > b and r_ > g:
            return 'red'
        if b > r_ and b > g:
            return 'blue'
        return 'other'

    # The ACTUAL rendered pixels, not metadata: material 1's own clip
    # (red) first, material 2's own clip (blue) second -- never swapped.
    assert _dominant(frame1) == 'red'
    assert _dominant(frame2) == 'blue'

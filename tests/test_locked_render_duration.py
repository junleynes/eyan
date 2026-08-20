"""
Regression test for a real reported bug: rendering an approved preview cut
("Lock cut & render" -- drop/add scenes after reviewing a preview, then
render) applied NO duration correction at all. Dropping a scene without
adding a replacement shipped a trailer however short that made it, with no
attempt to close the gap -- confirmed by tracing the code (the entire
"final exact-length correction" block lived inside the else: branch of
`if preselected: ... else: ...`, which is skipped entirely when rendering
an approved cut) and then by reproducing it directly: an identical-settings
fresh render hit the target exactly while the locked-render path did not.

This fixture is built specifically to avoid a different pitfall discovered
while diagnosing the bug: hard, colour-cut synthetic clips get detected as
scenes with essentially zero headroom (selected_dur == duration already),
so a "shortfall" has nothing to grow into regardless of whether the
correction code runs at all, which produced several false negatives during
manual testing. Two long (25s), single-colour scenes are used here so a
modest target selects only a fraction of each, leaving deliberate, large
headroom -- meaning a real correction should be able to reach the target
duration exactly, not just get closer than before.
"""
import os
import re
import subprocess
import time
import unittest.mock as mock

import pytest

with mock.patch('requests.post'), mock.patch('requests.get'):
    import main


def _ffmpeg_available():
    import shutil
    return shutil.which('ffmpeg') is not None


@pytest.fixture(scope='module')
def two_scene_video(tmp_path_factory):
    if not _ffmpeg_available():
        pytest.skip('ffmpeg not available in this environment')
    tmp = tmp_path_factory.mktemp('locked_render_fixture')
    a = tmp / 'a.mp4'
    b = tmp / 'b.mp4'
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'lavfi',
                    '-i', 'color=c=red:s=320x240:d=25:r=25',
                    '-f', 'lavfi', '-i', 'sine=frequency=300:duration=25',
                    '-c:v', 'libx264', '-c:a', 'aac', '-shortest', str(a)], check=True, timeout=60)
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'lavfi',
                    '-i', 'color=c=blue:s=320x240:d=25:r=25',
                    '-f', 'lavfi', '-i', 'sine=frequency=500:duration=25',
                    '-c:v', 'libx264', '-c:a', 'aac', '-shortest', str(b)], check=True, timeout=60)
    combined = tmp / 'two_scenes.mp4'
    list_file = tmp / 'list.txt'
    list_file.write_text(f"file '{a}'\nfile '{b}'\n")
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0',
                    '-i', str(list_file), '-c', 'copy', str(combined)], check=True, timeout=60)
    return combined


@pytest.fixture
def authed_client(tmp_path, monkeypatch):
    app = main.app
    upload_dir = tmp_path / 'uploads'
    upload_dir.mkdir()
    monkeypatch.setitem(app.config, 'UPLOAD_FOLDER', str(upload_dir))
    # ALLOW_LOCAL_MEDIA_UPLOAD is read from os.environ once at import time
    # into a module-level constant (pipeline.py), not re-checked per
    # request -- monkeypatch.setenv() alone has no effect on an
    # already-imported module, so the constant itself needs patching.
    monkeypatch.setattr(main.pipeline, 'ALLOW_LOCAL_MEDIA_UPLOAD', True)
    client = app.test_client()
    csrf_token = 'test-csrf-token-locked-render'
    with client.session_transaction() as sess:
        sess['authed'] = True
        sess['user_id'] = 1
        sess['username'] = 'admin'
        sess['role'] = 'admin'
        sess['csrf_token'] = csrf_token
    return client, {'X-CSRF-Token': csrf_token}


def _poll(client, job_id, headers, timeout_s=90):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        d = client.get(f'/api/trailer/progress/{job_id}', headers=headers).get_json()
        if d.get('done'):
            return d
        time.sleep(1)
    pytest.fail(f'job {job_id} did not finish within {timeout_s}s')


def test_locked_render_corrects_duration_after_dropping_a_scene(authed_client, two_scene_video):
    client, headers = authed_client
    from io import BytesIO
    video_bytes = two_scene_video.read_bytes()

    # Preview at 15s: two 25s scenes are each much longer than needed, so
    # each gets only a fraction selected -- deliberate, large headroom.
    r = client.post('/api/trailer/generate', data={
        'file': (BytesIO(video_bytes), 'two_scenes.mp4'),
        'genre': '', 'trailer_length': '15', 'scoring_mode': 'none',
        'sfx_mode': 'none', 'vo_mode': 'none', 'transition': 'cut',
        'preview_only': '1', 'sync_beats': '',
        # Caps how much of any single scene can be used, forcing the
        # selector to draw from BOTH 25s scenes rather than satisfying the
        # whole 15s target from just one of them (which it otherwise does,
        # since either alone is already long enough) -- needed so dropping
        # one leaves a second real scene to test correction against.
        'max_scene_dur': '8',
    }, headers=headers, content_type='multipart/form-data')
    assert r.status_code == 200, r.get_data(as_text=True)
    job_id = r.get_json()['job_id']
    d = _poll(client, job_id, headers)
    assert d.get('error') is None, d.get('error')
    result = d.get('result') or {}
    preview_id = result.get('preview_id')
    scenes = result.get('scenes', [])
    assert preview_id, 'preview did not produce a preview_id'
    if len(scenes) < 2:
        pytest.skip(f'scene detection only found {len(scenes)} scene(s) in this environment; '
                    'need 2 to meaningfully test dropping one')

    # Drop the first scene without adding a replacement -- exactly the
    # reported workflow. The second scene (25s long, likely selected well
    # under that) has ample headroom to be grown to close the gap.
    r2 = client.post('/api/trailer/render', data={'preview_id': preview_id, 'drop': '[1]'}, headers=headers)
    assert r2.status_code == 200, r2.get_data(as_text=True)
    job_id2 = r2.get_json()['job_id']
    d2 = _poll(client, job_id2, headers)
    assert d2.get('error') is None, d2.get('error')
    result2 = d2.get('result') or {}

    duration = result2.get('trailer_duration')
    assert duration is not None
    # The actual regression check: before the fix, dropping a scene without
    # replacement left the render at whatever the remaining raw scene
    # duration happened to be, with no attempt to correct it -- typically
    # several seconds under target even with plenty of headroom available.
    # After the fix, duration correction runs for this path too and should
    # land at or extremely close to the 15s target given the headroom here.
    assert abs(duration - 15.0) < 0.6, (
        f'locked render duration {duration}s is not close to the 15s target -- '
        'duration correction does not appear to be running for the locked-render path'
    )

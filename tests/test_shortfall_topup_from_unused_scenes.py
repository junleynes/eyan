"""
Regression coverage for a real gap in the fully-automatic "Generate promo
plug" selection: after the best-scenes pass_attempt loop and the
grow-the-last-clip step, a persisting shortfall was simply shipped as-is --
there was no equivalent of the manual "Preview the cut" edit flow's
_autofill_short_selection_from_alternates, which pulls in additional,
not-yet-used scenes when rebalancing alone can't close the gap.

This matters specifically when many short, closely-spaced scenes are
detected: the pass_attempt loop's own spacing rule (min_gap) floors at
1.0s and never relaxes further, so scenes sitting closer together than
that in the source timeline are permanently excluded from every pass_attempt
-- even though using a couple of them would easily close a small remaining
gap. The fix adds a genuine last-resort top-up step after the last-clip-grow
step that relaxes spacing further (down to 0.3s) specifically because every
looser option already failed, on the reasoning that clustering two clips
closer together is a smaller problem than shipping a trailer noticeably
under the requested length.

Fixture: eight distinct-color scenes, each only 0.6s long and butted
directly against each other (starts 0.6s apart) -- well under the 1.0s
min_gap floor, so a naive single-pass selection could only ever pick roughly
every other one before running out of budget-worthy candidates, leaving a
real, unclosable-by-rebalancing-alone gap for an 4s target.
"""
import shutil
import subprocess
import time
import unittest.mock as mock

import pytest

with mock.patch('requests.post'), mock.patch('requests.get'):
    import main


def _ffmpeg_available():
    return shutil.which('ffmpeg') is not None


_COLORS = ['red', 'blue', 'green', 'yellow', 'purple', 'cyan', 'orange', 'white',
           'pink', 'gray', 'brown', 'gold', 'navy', 'teal', 'maroon', 'olive',
           'crimson', 'coral', 'indigo', 'khaki', 'salmon', 'plum', 'orchid',
           'tan', 'beige', 'lavender', 'turquoise', 'chocolate', 'chartreuse', 'azure']


@pytest.fixture(scope='module')
def many_short_scenes_video(tmp_path_factory):
    if not _ffmpeg_available():
        pytest.skip('ffmpeg not available in this environment')
    tmp = tmp_path_factory.mktemp('shortfall_topup_fixture')
    # 30 scenes x 0.8s = 24s of raw video (clears the 22.5s minimum a 15s
    # target requires), each scene butted directly against the next -- 0.8s
    # apart, well under the pass_attempt loop's 1.0s min_gap floor.
    parts = []
    for i, color in enumerate(_COLORS):
        part = tmp / f'part{i}.mp4'
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'lavfi',
                        '-i', f'color=c={color}:s=320x240:d=0.8:r=25',
                        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(part)],
                       check=True, timeout=30)
        parts.append(part)
    list_file = tmp / 'list.txt'
    list_file.write_text('\n'.join(f"file '{p}'" for p in parts))
    combined = tmp / 'combined.mp4'
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0',
                    '-i', str(list_file), '-c', 'copy', str(combined)], check=True, timeout=60)
    return combined


@pytest.fixture
def authed_client(tmp_path, monkeypatch):
    app = main.app
    upload_dir = tmp_path / 'uploads'
    upload_dir.mkdir()
    monkeypatch.setitem(app.config, 'UPLOAD_FOLDER', str(upload_dir))
    monkeypatch.setattr(main.pipeline, 'ALLOW_LOCAL_MEDIA_UPLOAD', True)
    client = app.test_client()
    csrf_token = 'test-csrf-shortfall-topup'
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


def test_closely_spaced_short_scenes_still_fill_the_target_duration(authed_client, many_short_scenes_video):
    client, headers = authed_client
    from io import BytesIO
    video_bytes = many_short_scenes_video.read_bytes()

    r = client.post('/api/trailer/generate', data={
        'file': (BytesIO(video_bytes), 'combined.mp4'),
        'genre': '', 'trailer_length': '15', 'scoring_mode': 'none',
        'sfx_mode': 'none', 'vo_mode': 'none', 'transition': 'cut',
        'preview_only': '1', 'sync_beats': '',
        'min_scene_len': '0.4',
    }, headers=headers, content_type='multipart/form-data')
    assert r.status_code == 200, r.get_data(as_text=True)
    job_id = r.get_json()['job_id']
    d = _poll(client, job_id, headers)
    assert d.get('error') is None, d.get('error')
    result = d.get('result') or {}
    scenes = result.get('scenes', [])
    if len(scenes) + len(result.get('alternates') or []) < 5:
        pytest.skip(f'scene detection only found a total of '
                    f'{len(scenes) + len(result.get("alternates") or [])} scene(s) in this '
                    'environment; need several closely-spaced ones to meaningfully test the '
                    'shortfall top-up')

    estimated = result.get('estimated_duration')
    assert estimated is not None
    # Before the fix: the pass_attempt loop's min_gap never relaxes past
    # 1.0s, so with scenes only 0.8s apart it could pick at most every
    # other one -- well short of a 15s target. The top-up step (relaxing
    # spacing further, as a genuine last resort) should close that gap.
    assert abs(estimated - 15.0) < 1.5, (
        f'estimated duration {estimated}s is not close to the 15s target -- the shortfall '
        'top-up from closely-spaced, previously-excluded scenes does not appear to be working'
    )

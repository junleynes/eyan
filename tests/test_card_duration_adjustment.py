"""
User-requested: "we can also do to fill gap, instead of pulling extra
scenes, is to also adjust tcard and endcard but there should be limit" --
a further gap-filling mechanism for the automatic promo generation,
alongside the scene top-up (test_shortfall_topup_from_unused_scenes.py) and
the proportional exact-length correction (test_locked_render_duration.py).

Confirmed design (via AskUserQuestion): grow a card by freeze-framing its
last frame (never looping or stretching); allow both growing AND shrinking;
cap the combined adjustment across both cards at 1.0s
(pipeline.CARD_DURATION_ADJUST_LIMIT) since a card is branding/
informational and should only absorb the last sliver of a gap, never carry
it.

This file has two parts:
  1. Direct, deterministic unit tests of the new _adjust_card_duration()
     helper -- growing (with and without an audio track), shrinking, and
     the no-op-for-negligible-delta short-circuit.
  2. An end-to-end integration test proving the last-resort step in
     _run_trailer_job actually fires and is bounded to
     CARD_DURATION_ADJUST_LIMIT when scene-clip-based correction alone
     cannot close a residual (every selected scene already has zero
     headroom -- selected_dur == duration -- the same zero-headroom
     scenario test_locked_render_duration.py's docstring documents as a
     real pitfall when using hard, colour-cut synthetic clips).
"""
import shutil
import subprocess
import time
import unittest.mock as mock
from io import BytesIO

import pytest

with mock.patch('requests.post'), mock.patch('requests.get'):
    import main
    import pipeline


def _ffmpeg_available():
    return shutil.which('ffmpeg') is not None


# ---------------------------------------------------------------------------
# Part 1: _adjust_card_duration unit tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def card_with_audio(tmp_path_factory):
    if not _ffmpeg_available():
        pytest.skip('ffmpeg not available in this environment')
    tmp = tmp_path_factory.mktemp('card_adjust_fixture')
    path = tmp / 'card_audio.mp4'
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'lavfi',
                    '-i', 'color=c=red:s=320x240:d=3:r=25',
                    '-f', 'lavfi', '-i', 'sine=frequency=300:duration=3',
                    '-c:v', 'libx264', '-c:a', 'aac', '-shortest', str(path)],
                   check=True, timeout=60)
    return path


@pytest.fixture(scope='module')
def card_no_audio(tmp_path_factory):
    if not _ffmpeg_available():
        pytest.skip('ffmpeg not available in this environment')
    tmp = tmp_path_factory.mktemp('card_adjust_fixture_silent')
    path = tmp / 'card_silent.mp4'
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'lavfi',
                    '-i', 'color=c=blue:s=320x240:d=3:r=25',
                    '-an', '-c:v', 'libx264', str(path)],
                   check=True, timeout=60)
    return path


def test_growing_a_card_with_audio_extends_video_and_audio(tmp_path, card_with_audio):
    out = tmp_path / 'grown.mp4'
    result = pipeline._adjust_card_duration(str(card_with_audio), 0.8, str(out))
    assert result == str(out)
    info = pipeline.probe_media_info(str(out))
    assert info['has_audio']
    # Original was 3s; grown by 0.8s -> ~3.8s. ffmpeg encode timing isn't
    # frame-exact, so allow a small tolerance.
    assert abs(info['duration'] - 3.8) < 0.25


def test_growing_a_card_without_audio_stays_video_only(tmp_path, card_no_audio):
    out = tmp_path / 'grown_silent.mp4'
    result = pipeline._adjust_card_duration(str(card_no_audio), 0.5, str(out))
    assert result == str(out)
    info = pipeline.probe_media_info(str(out))
    assert not info['has_audio']
    assert abs(info['duration'] - 3.5) < 0.25


def test_shrinking_a_card_trims_the_tail(tmp_path, card_with_audio):
    out = tmp_path / 'shrunk.mp4'
    result = pipeline._adjust_card_duration(str(card_with_audio), -1.0, str(out))
    assert result == str(out)
    info = pipeline.probe_media_info(str(out))
    assert abs(info['duration'] - 2.0) < 0.25


def test_negligible_delta_is_a_no_op(tmp_path, card_with_audio):
    out = tmp_path / 'noop.mp4'
    assert pipeline._adjust_card_duration(str(card_with_audio), 0.005, str(out)) is None
    assert not out.exists()


def test_card_duration_adjust_limit_is_small_and_positive():
    # Guards against an accidental future change silently loosening what was
    # an explicit, user-confirmed design constraint (a card should only
    # absorb the last sliver of a gap, never carry it).
    assert 0 < pipeline.CARD_DURATION_ADJUST_LIMIT <= 2.0


# ---------------------------------------------------------------------------
# Part 2: end-to-end -- the last-resort step inside _run_trailer_job
# ---------------------------------------------------------------------------

_MANY_COLORS = ['red', 'blue', 'green', 'yellow', 'purple', 'cyan', 'orange', 'white',
                'pink', 'gray', 'brown', 'gold', 'navy', 'teal', 'maroon', 'olive',
                'crimson', 'coral', 'indigo', 'khaki', 'salmon', 'plum', 'orchid']


@pytest.fixture(scope='module')
def zero_headroom_scenes_video(tmp_path_factory):
    if not _ffmpeg_available():
        pytest.skip('ffmpeg not available in this environment')
    tmp = tmp_path_factory.mktemp('card_adjust_scenes')
    # 23 hard, colour-cut 1.0s scenes (23s of raw video, clearing the 22.5s
    # minimum a 15s target requires) -- short enough that the selector
    # always takes each picked one in full (selected_dur == duration, i.e.
    # zero headroom to grow into), the same zero-headroom pitfall documented
    # in test_locked_render_duration.py, deliberately reproduced here so
    # that the *only* way left to close a residual gap is the card itself.
    parts = []
    for i, color in enumerate(_MANY_COLORS):
        part = tmp / f'part{i}.mp4'
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'lavfi',
                        '-i', f'color=c={color}:s=320x240:d=1.0:r=25',
                        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(part)],
                       check=True, timeout=30)
        parts.append(part)
    list_file = tmp / 'list.txt'
    list_file.write_text('\n'.join(f"file '{p}'" for p in parts))
    combined = tmp / 'combined.mp4'
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0',
                    '-i', str(list_file), '-c', 'copy', str(combined)], check=True, timeout=60)
    return combined


@pytest.fixture(scope='module')
def title_card_video(tmp_path_factory):
    if not _ffmpeg_available():
        pytest.skip('ffmpeg not available in this environment')
    tmp = tmp_path_factory.mktemp('card_adjust_titlecard')
    path = tmp / 'titlecard.mp4'
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'lavfi',
                    '-i', 'color=c=white:s=320x240:d=1.0:r=25',
                    '-an', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(path)],
                   check=True, timeout=30)
    return path


@pytest.fixture
def authed_client(tmp_path, monkeypatch):
    app = main.app
    upload_dir = tmp_path / 'uploads'
    upload_dir.mkdir()
    monkeypatch.setitem(app.config, 'UPLOAD_FOLDER', str(upload_dir))
    monkeypatch.setattr(main.pipeline, 'ALLOW_LOCAL_MEDIA_UPLOAD', True)
    client = app.test_client()
    csrf_token = 'test-csrf-card-duration-adjust'
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


def test_card_grows_to_close_a_residual_scene_correction_cannot_reach(
        authed_client, zero_headroom_scenes_video, title_card_video):
    client, headers = authed_client
    video_bytes = zero_headroom_scenes_video.read_bytes()
    card_bytes = title_card_video.read_bytes()

    # 23 zero-headroom 1.0s scenes (each fully used whenever picked, so the
    # proportional scene-based correction has nothing to grow into) plus a
    # 1.0s title card, targeting 15s -- trailer_length is restricted to
    # (15, 30, 45, 60) so 15 is the smallest usable target. Every added
    # transition costs xfade_dur (0.3s default) off the assembled total, so
    # picking N scenes + 1 card gives N*0.3 of unavoidable loss that
    # zero-headroom scenes can't absorb -- only the card-duration last
    # resort can close any of it, and it's capped at
    # CARD_DURATION_ADJUST_LIMIT (1.0s combined).
    r = client.post('/api/trailer/generate', data={
        'file': (BytesIO(video_bytes), 'combined.mp4'),
        'end_card_video': (BytesIO(card_bytes), 'titlecard.mp4'),
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
    if len(scenes) < 5:
        pytest.skip(f'scene detection/selection only produced {len(scenes)} scene(s) in this '
                    'environment; need several zero-headroom scenes to meaningfully test the '
                    'card-duration last resort')

    estimated = result.get('estimated_duration')
    assert estimated is not None

    # Reconstruct the baseline the job would have shipped with NO card
    # adjustment at all, from the actual selection this run made -- so the
    # assertion holds regardless of exactly how many/which scenes the
    # (scoring_mode=none) selector happened to pick. Each returned scene's
    # 'duration' is its final selected_dur (post scene-based correction,
    # pre card adjustment) -- see pipeline.py's preview payload (~line 5544).
    total_sel = sum(s['duration'] for s in scenes)
    original_card_duration = 1.0
    n_seg = len(scenes) + 1  # + the one title card
    xfade_dur = 0.3
    xfade_loss = max(0, n_seg - 1) * xfade_dur
    baseline = total_sel + original_card_duration - xfade_loss
    residual = 15.0 - baseline

    if residual <= 0.15:
        pytest.skip('this run\'s selection left ~zero residual for the scene-based correction '
                    'to already close on its own -- nothing left to distinguish the card '
                    'adjustment step from ordinary rounding')

    expected_growth = min(residual, pipeline.CARD_DURATION_ADJUST_LIMIT)
    expected = baseline + expected_growth
    # ffmpeg re-encode timing plus the proportional correction pass sharing
    # some of the gap first aren't frame-exact -- allow a generous tolerance
    # while still requiring growth to have clearly happened, and for it to
    # respect the cap rather than fully closing an over-the-cap residual.
    assert abs(estimated - expected) < 0.6, (
        f'estimated duration {estimated}s does not match the predicted post-card-adjustment '
        f'value ({expected}s = baseline {baseline:.2f}s + up to {pipeline.CARD_DURATION_ADJUST_LIMIT}s '
        'of card growth) -- the card-duration last-resort step does not appear to be running, '
        'or is not respecting CARD_DURATION_ADJUST_LIMIT'
    )
    if residual > pipeline.CARD_DURATION_ADJUST_LIMIT + 0.2:
        assert estimated < 15.0 - 0.3, (
            f'estimated duration {estimated}s is suspiciously close to the 15s target given a '
            f'{residual:.2f}s residual and zero scene headroom -- CARD_DURATION_ADJUST_LIMIT '
            'does not appear to be capping the card adjustment as designed'
        )

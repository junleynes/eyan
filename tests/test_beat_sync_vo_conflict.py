"""
Tests for a real, user-reported problem: with both "Sync cuts to beat" and
VO-driven scene selection on, cuts appeared to track the MUSIC's beat grid
rather than the VO's own word/phrase timing. Traced to
select_scenes_vo_led() (and the mirrored logic in the general best-scenes
loop) searching for a beat to snap to across the ENTIRE remaining scene,
with no regard for where the VO-word-boundary-aware target already was --
a beat seconds away could silently override carefully-placed VO timing,
and the speech-safety pass that runs afterward only searches a narrow
window of its own (~0.45s word / ~1.2s phrase), so it often couldn't
reach back and fix a cut beat-sync had already dragged far away.

Fixed by constraining the beat search to a tight window
(BEAT_SYNC_MAX_NUDGE, matching the same scale nearest_speech_out already
searches) around the already-computed target, not the whole scene.
"""
import pipeline


def _scene(start, duration, total_score=5.0, has_face=False):
    return {
        'start': start, 'duration': duration, 'total_score': total_score,
        'has_face': has_face, 'description': '', 'quality_score': 5.0,
    }


def _vo_beat(text, duration):
    words = text.split()
    return {'text': text, 'words': words, 'word_count': len(words), 'duration': duration}


def test_distant_beat_is_not_used_vo_led():
    # A beat 0.7s from the VO-driven target -- well outside
    # BEAT_SYNC_MAX_NUDGE (0.4s) -- must NOT be snapped to; the VO-derived
    # duration should survive essentially unchanged.
    scenes = [_scene(0, 20)]
    beats = [_vo_beat('hello there world', 5.0)]
    selected, total = pipeline.select_scenes_vo_led(
        scenes, None, trailer_duration=5.0, max_scene_dur=None, min_seg_dur=1.0,
        min_gap=1.0, sync_beats=True, beat_times=[5.7], vo_beats=beats,
    )
    assert selected
    assert abs(selected[0]['selected_dur'] - 5.0) < 0.05


def test_close_beat_is_still_used_vo_led():
    # A beat only 0.2s from the VO-driven target -- well within
    # BEAT_SYNC_MAX_NUDGE -- should still be snapped to; this isn't a
    # "beat-sync now does nothing" fix, just a bounded one.
    scenes = [_scene(0, 20)]
    beats = [_vo_beat('hello there world', 5.0)]
    selected, total = pipeline.select_scenes_vo_led(
        scenes, None, trailer_duration=5.0, max_scene_dur=None, min_seg_dur=1.0,
        min_gap=1.0, sync_beats=True, beat_times=[5.2], vo_beats=beats,
    )
    assert selected
    assert abs(selected[0]['selected_dur'] - 5.2) < 0.05


def test_no_beats_at_all_falls_back_to_vo_target():
    scenes = [_scene(0, 20)]
    beats = [_vo_beat('hello there world', 5.0)]
    selected, total = pipeline.select_scenes_vo_led(
        scenes, None, trailer_duration=5.0, max_scene_dur=None, min_seg_dur=1.0,
        min_gap=1.0, sync_beats=True, beat_times=[], vo_beats=beats,
    )
    assert selected
    assert abs(selected[0]['selected_dur'] - 5.0) < 0.05


def test_beat_sync_max_nudge_constant_is_reasonably_tight():
    # Not a behavioral test, just a sanity guard against the constant
    # drifting back toward "search the whole scene" territory, which is
    # the exact shape of bug this fix addresses. Comparable to (not larger
    # than) nearest_speech_out's own phrase-snap window, since a nudge the
    # safety net can't reach defeats the point of bounding it at all.
    assert 0 < pipeline.BEAT_SYNC_MAX_NUDGE <= 0.6

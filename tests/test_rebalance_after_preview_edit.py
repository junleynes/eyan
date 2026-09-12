"""
Tests for a real, user-reported bug: dropping or adding a clip in
"Preview the cut", then rendering, left every remaining scene's own
duration exactly as the ORIGINAL preview analysis had sized it -- correct
for that selection, but wrong the instant it changed. Removing a clip
left the total short of the target with nothing making up the
difference; adding an alternate (sized for a different remaining-budget
context) could just as easily push the total over. Neither drop nor add
triggered any recompute at all.

Fixed with _rebalance_selected_durations(): after edits, scales every
remaining scene's selected_dur proportionally back toward the original
target, clamped to what each scene can actually supply (at least a
sane minimum, at most its own real footage from trim_start to its own
end) -- not "dump the whole gap onto the last clip" or "stretch a scene
past footage it doesn't have".
"""
import pipeline


def _scene(start, duration, selected_dur, trim_start=None):
    return {
        'start': start, 'end': start + duration, 'duration': duration,
        'selected_dur': selected_dur, 'trim_start': trim_start if trim_start is not None else start,
    }


def test_no_change_when_already_close_to_target():
    scenes = [_scene(0, 10, 5.0), _scene(20, 10, 5.0), _scene(40, 10, 5.0)]
    result = pipeline._rebalance_selected_durations(scenes, target_duration=15.0)
    assert [round(s['selected_dur'], 2) for s in result] == [5.0, 5.0, 5.0]


def test_dropping_a_scene_stretches_the_remaining_ones_to_refill_the_target():
    # Original preview: 3 scenes totaling 15s for a 15s target. Drop the
    # middle one (5s) before rendering -- the remaining two must grow to
    # cover the 5s gap, not leave the render 5s short.
    remaining = [_scene(0, 10, 5.0), _scene(40, 10, 5.0)]
    result = pipeline._rebalance_selected_durations(remaining, target_duration=15.0)
    total = sum(s['selected_dur'] for s in result)
    assert abs(total - 15.0) < 0.2
    # Grew proportionally (both started equal, both still equal after) --
    # not "the whole gap dumped onto one clip".
    assert abs(result[0]['selected_dur'] - result[1]['selected_dur']) < 0.1


def test_adding_an_alternate_shrinks_scenes_back_down_to_the_target():
    # Original: 2 scenes totaling 10s. An alternate sized for its own,
    # different context (6s) gets added on top -- total is now 16s against
    # a 15s target, 1s over. Everything should shrink back down together.
    scenes = [_scene(0, 10, 5.0), _scene(20, 10, 5.0), _scene(40, 10, 6.0)]
    result = pipeline._rebalance_selected_durations(scenes, target_duration=15.0)
    total = sum(s['selected_dur'] for s in result)
    assert abs(total - 15.0) < 0.2


def test_a_scene_at_its_footage_ceiling_cant_be_stretched_further_so_others_absorb_the_rest():
    # First scene has only 5s of real footage available (trim_start=5,
    # end=10) -- it's already using all of it and cannot supply more no
    # matter how much the gap wants it to. The second scene, with plenty
    # of spare footage, must absorb the rest of the gap on its own.
    scenes = [
        _scene(0, 10, 5.0, trim_start=5.0),   # only 5s of footage available: [5,10]
        _scene(20, 20, 5.0),                   # 20s of footage available: [20,40]
    ]
    result = pipeline._rebalance_selected_durations(scenes, target_duration=20.0)
    total = sum(s['selected_dur'] for s in result)
    assert abs(total - 20.0) < 0.3
    assert result[0]['selected_dur'] <= 5.0 + 1e-6  # never exceeded its real ceiling
    assert result[1]['selected_dur'] > 5.0  # picked up the slack the first scene couldn't supply


def test_never_shrinks_a_scene_below_the_minimum_segment_duration():
    scenes = [_scene(0, 30, 10.0), _scene(50, 30, 10.0)]
    # Target far below what a real min_seg_dur would allow if evenly split
    result = pipeline._rebalance_selected_durations(scenes, target_duration=1.0, min_seg_dur=0.8)
    for s in result:
        assert s['selected_dur'] >= 0.8 - 1e-6


def test_empty_selection_returns_empty():
    assert pipeline._rebalance_selected_durations([], target_duration=15.0) == []


def test_original_list_and_dicts_are_not_mutated():
    scenes = [_scene(0, 10, 5.0), _scene(40, 10, 5.0)]
    original_dur = [s['selected_dur'] for s in scenes]
    pipeline._rebalance_selected_durations(scenes, target_duration=15.0)
    assert [s['selected_dur'] for s in scenes] == original_dur


def _alt(start, duration, selected_dur, total_score):
    s = _scene(start, duration, selected_dur)
    s['total_score'] = total_score
    return s


def test_autofill_pulls_in_alternates_when_rebalancing_alone_cant_close_the_gap():
    # Both remaining scenes are already fully used (selected_dur == their
    # entire real footage) -- the exact case rebalancing alone can't fix,
    # since neither has any slack left to stretch into. A 5s shortfall
    # against a 15s target with two 5s scenes (10s total) needs an actual
    # additional clip, not more squeezing out of what's already maxed.
    selected = [_scene(0, 5, 5.0), _scene(20, 5, 5.0)]
    alternates = [_alt(40, 6, 6.0, total_score=8.0), _alt(60, 6, 6.0, total_score=3.0)]
    result = pipeline._autofill_short_selection_from_alternates(
        selected, alternates, already_used_alt_numbers=set(), target_duration=15.0)
    assert len(result) == 3
    # The higher-scoring alternate (8.0) was preferred over the lower one (3.0)
    assert any(s['start'] == 40 for s in result)
    assert not any(s['start'] == 60 for s in result)


def test_autofill_skips_alternates_too_close_to_an_already_selected_scene():
    selected = [_scene(0, 5, 5.0), _scene(20, 5, 5.0)]
    # This alternate sits right next to the already-selected scene at 20 --
    # too close (default min_gap=1.0) to use without risking a near-duplicate.
    alternates = [_alt(20.5, 6, 6.0, total_score=9.0), _alt(50, 6, 6.0, total_score=2.0)]
    result = pipeline._autofill_short_selection_from_alternates(
        selected, alternates, already_used_alt_numbers=set(), target_duration=15.0)
    assert not any(s['start'] == 20.5 for s in result)
    assert any(s['start'] == 50 for s in result)


def test_autofill_does_nothing_when_shortfall_is_small():
    # A shortfall under the 1.0s threshold isn't worth pulling in a whole
    # extra clip for -- the rebalance pass alone (or simply being close
    # enough) already covers this.
    selected = [_scene(0, 10, 7.0), _scene(20, 10, 7.5)]  # 14.5s vs 15.0s target: 0.5s short
    alternates = [_alt(40, 6, 6.0, total_score=9.0)]
    result = pipeline._autofill_short_selection_from_alternates(
        selected, alternates, already_used_alt_numbers=set(), target_duration=15.0)
    assert len(result) == 2


def test_autofill_does_not_reuse_an_alternate_already_used_by_add():
    selected = [_scene(0, 5, 5.0), _scene(20, 5, 5.0)]
    alternates = [_alt(40, 6, 6.0, total_score=8.0)]
    # Alternate #1 was already pulled in via the user's own explicit "add" --
    # autofill must not double-count or re-add it.
    result = pipeline._autofill_short_selection_from_alternates(
        selected, alternates, already_used_alt_numbers={1}, target_duration=15.0)
    assert len(result) == 2

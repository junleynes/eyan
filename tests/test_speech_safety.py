"""
Tests for the speech-safe cutting helpers: nearest_word_boundary,
nearest_speech_out, speech_free_slack. These exist specifically to keep the
renderer from cutting a voice mid-word/mid-sentence, and from having the
exact-duration corrector silently undo that snapping again -- both were real
bugs earlier in this project's history, caught by hand each time. Codifying
the cases that caught them means the next regression fails a test instead of
needing another live-render investigation to notice at all.
"""
import pipeline


class TestNearestWordBoundary:
    def test_snaps_to_nearest_within_range(self):
        assert pipeline.nearest_word_boundary(5.0, [4.9, 6.0], max_snap=0.35) == 4.9

    def test_falls_back_to_target_when_nothing_in_range(self):
        # No boundary within max_snap -- must return target unchanged, not
        # the nearest boundary regardless of distance (that would let a cut
        # drift arbitrarily far from where the visual scene actually is).
        assert pipeline.nearest_word_boundary(5.0, [1.0, 20.0], max_snap=0.35) == 5.0

    def test_empty_boundaries_returns_target(self):
        assert pipeline.nearest_word_boundary(5.0, [], max_snap=0.35) == 5.0

    def test_picks_the_closer_of_two_in_range(self):
        assert pipeline.nearest_word_boundary(5.0, [4.8, 5.1], max_snap=0.35) == 5.1


class TestNearestSpeechOut:
    def test_prefers_phrase_end_over_closer_word_end(self):
        # A phrase end within its wider window wins even when a word end is
        # numerically closer -- landing on a finished sentence is worth
        # traveling further for than landing mid-sentence at a bare word gap.
        result = pipeline.nearest_speech_out(5.0, phrase_ends=[5.5], word_ends=[4.9])
        assert result == 5.5

    def test_falls_back_to_word_end_when_no_phrase_in_range(self):
        result = pipeline.nearest_speech_out(5.0, phrase_ends=[20.0], word_ends=[5.1])
        assert result == 5.1

    def test_falls_back_to_target_when_nothing_nearby(self):
        # No speech anywhere near this point at all -- e.g. silent B-roll --
        # so the original visual cut point is left alone.
        result = pipeline.nearest_speech_out(5.0, phrase_ends=[], word_ends=[])
        assert result == 5.0

    def test_phrase_window_wider_than_word_window(self):
        # A phrase end 1.0s away should be reachable (within the wider
        # phrase-snap window) even though that's well outside the tighter
        # word-snap window.
        result = pipeline.nearest_speech_out(5.0, phrase_ends=[6.0], word_ends=[])
        assert result == 6.0


class TestSpeechFreeSlack:
    def test_clip_with_speech_to_its_end_has_no_slack(self):
        # Dialogue runs right up to the clip's own end -- nothing here is
        # safe to trim without cutting into speech.
        spans = [(0.0, 5.5)]
        assert pipeline.speech_free_slack(0.0, 5.5, spans) == 0.0

    def test_fully_silent_clip_has_full_slack(self):
        spans = [(20.0, 25.0)]  # speech elsewhere entirely, not in this clip
        assert pipeline.speech_free_slack(0.0, 4.0, spans) == 4.0

    def test_partial_tail_after_guard(self):
        spans = [(0.0, 3.0)]
        # Speech ends at 3.0, guard is 0.12 by default -> safe zone starts at
        # 3.12, clip runs to 8.0 -> 4.88s of genuinely speech-free tail.
        result = pipeline.speech_free_slack(0.0, 8.0, spans)
        assert abs(result - 4.88) < 1e-9

    def test_no_speech_spans_at_all_is_fully_available(self):
        assert pipeline.speech_free_slack(0.0, 10.0, []) == 10.0

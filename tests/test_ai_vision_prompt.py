"""
Tests for build_ai_vision_prompt() -- assembles the actual scoring prompt
sent to the vision model, layering genre, an optional "feature this"
priority description, and an optional "avoid this" negative description.
Pulled out of the scoring loop specifically so each layer is independently
testable without needing a full render pipeline.
"""
import pipeline


def test_base_prompt_passes_through_unchanged_with_no_extras():
    result = pipeline.build_ai_vision_prompt('Rate this frame.', 'action')
    assert result == 'Rate this frame.'


def test_genre_substitution_requires_both_a_known_genre_and_the_desc_marker():
    base = 'DESC: describe the frame for a movie trailer.'
    result = pipeline.build_ai_vision_prompt(base, 'action')
    assert 'for a action promo trailer' in result
    assert 'for a movie trailer' not in result


def test_genre_substitution_skipped_without_the_desc_marker():
    # No 'DESC:' in the base prompt -- substitution must not fire even with
    # a valid, known genre, matching the original (pre-refactor) condition.
    base = 'Rate this frame for a movie trailer.'
    result = pipeline.build_ai_vision_prompt(base, 'action')
    assert result == base


def test_genre_substitution_skipped_for_an_unknown_genre():
    base = 'DESC: describe the frame for a movie trailer.'
    result = pipeline.build_ai_vision_prompt(base, 'not_a_real_genre')
    assert result == base


def test_priority_prompt_adds_a_clearly_labelled_boost_instruction():
    result = pipeline.build_ai_vision_prompt('Rate this frame.', 'action',
                                              priority_prompt='the confrontation scene')
    assert 'PRIORITY:' in result
    assert 'the confrontation scene' in result
    assert 'noticeably higher' in result


def test_negative_prompt_adds_a_clearly_labelled_penalty_instruction():
    result = pipeline.build_ai_vision_prompt('Rate this frame.', 'action',
                                              negative_prompt='no jump scares')
    assert 'AVOID:' in result
    assert 'no jump scares' in result
    assert 'noticeably lower' in result


def test_priority_and_negative_are_two_separate_instructions_not_merged():
    # Both present at once should appear as two distinct, separately
    # labelled blocks -- not folded into one sentence, which would be more
    # likely to confuse a vision model than two clear instructions.
    result = pipeline.build_ai_vision_prompt(
        'Rate this frame.', 'action',
        priority_prompt='the confrontation scene',
        negative_prompt='no jump scares',
    )
    assert 'PRIORITY:' in result and 'AVOID:' in result
    priority_idx = result.index('PRIORITY:')
    avoid_idx = result.index('AVOID:')
    assert priority_idx != avoid_idx
    # PRIORITY's own sentence shouldn't bleed into AVOID's, and vice versa.
    priority_block = result[priority_idx:avoid_idx]
    avoid_block = result[avoid_idx:]
    assert 'no jump scares' not in priority_block
    assert 'the confrontation scene' not in avoid_block


def test_empty_or_none_priority_and_negative_are_both_omitted():
    result = pipeline.build_ai_vision_prompt('Rate this frame.', 'action',
                                              priority_prompt='', negative_prompt=None)
    assert result == 'Rate this frame.'
    assert 'PRIORITY:' not in result and 'AVOID:' not in result

"""
Regression guard for a real, user-reported bug shown by screenshots of the
actual progress UI: the stepper visibly jumped backward from "Selecting
scenes" to "AI vision rating" partway through a real job, even though the
job was moving strictly forward the whole time.

Root cause: PIPELINE_STAGES (the percent->label table driving the stepper)
correctly climbs 18 -> 30 across the AI vision scoring loop, but several
job_set() calls for steps that genuinely run right AFTER that loop
finishes (transcribing dialogue, preparing music for beat-synced cuts, the
actual scene-selection step itself, transcribing uploaded narration) were
still using percent values from BEFORE that 30% ceiling -- so the reported
percent, and the stepper along with it, dropped back into an earlier
bucket every time one of those ran.

This can't be exercised end-to-end without Ollama/faster-whisper actually
running, so instead it checks the real, live source of truth directly:
pipeline.py's own text, and the actual job_set() calls in the stretch of
code between the AI vision scoring loop and "Extracting clips". A future
edit that reintroduces an out-of-order value here should fail this test
immediately, rather than waiting to be caught by someone reading a
screenshot of a real job again.
"""
import re

import pipeline


def _read_pipeline_source():
    with open(pipeline.__file__) as f:
        return f.read()


def test_pipeline_stages_table_is_strictly_increasing():
    percents = [p for p, _ in pipeline.PIPELINE_STAGES]
    assert percents == sorted(percents)
    assert len(percents) == len(set(percents)), "PIPELINE_STAGES has a duplicate percent"


def test_selecting_scenes_starts_at_or_after_ai_vision_ratings_own_ceiling():
    # The AI vision scoring loop's own formula is 18 + int(12 * done/total),
    # so its maximum possible value is 18 + 12 = 30. "Selecting scenes"
    # itself, and everything that runs between the AI scoring loop and it,
    # must never use a percent below that ceiling -- that's the exact shape
    # of the reported bug.
    stages = {label: percent for percent, label in pipeline.PIPELINE_STAGES}
    ai_ceiling = 18 + 12
    assert stages['Selecting scenes'] >= ai_ceiling


def test_steps_between_ai_scoring_and_selecting_scenes_never_regress():
    # Scans the real source for the specific job_set() calls that run in
    # this stretch of the pipeline (by name, not by an assumed line range,
    # so this survives unrelated edits elsewhere in the file) and confirms
    # each one's own percent is at or above "Selecting scenes"'s own
    # threshold and at or below "Extracting clips"'s -- i.e. genuinely
    # placed in the same monotonic band as the surrounding stages, not
    # reusing an earlier, already-passed value.
    src = _read_pipeline_source()
    stages = {label: percent for percent, label in pipeline.PIPELINE_STAGES}
    lower_bound = stages['Selecting scenes']
    upper_bound = stages['Extracting clips']

    steps_to_check = [
        'Transcribing dialogue (faster-whisper)',
        'Preparing music for beat-synced cuts',
        'Selecting scenes',
        'Transcribing uploaded narration for selection',
        'Selecting scenes from narration',
    ]
    for step_name in steps_to_check:
        pattern = re.compile(
            r"job_set\(jid,\s*percent=(\d+),\s*step='" + re.escape(step_name) + r"'"
        )
        match = pattern.search(src)
        assert match, f"Could not find a job_set() call for step {step_name!r} -- did its wording change?"
        percent = int(match.group(1))
        assert lower_bound <= percent <= upper_bound, (
            f"{step_name!r} uses percent={percent}, which falls outside "
            f"[{lower_bound}, {upper_bound}] -- this is exactly the shape of "
            f"bug that made the stepper jump backward to an earlier stage."
        )


def test_ai_vision_scoring_loop_ceiling_matches_the_hardcoded_assumption():
    # If the AI scoring loop's own formula (18 + int(12 * done/total)) is
    # ever changed, the other tests here assume its ceiling is still 30 --
    # this pins that assumption directly against the real source so a
    # silent mismatch can't creep in.
    src = _read_pipeline_source()
    assert "percent=18 + int(12 * _ai_progress['done']" in src

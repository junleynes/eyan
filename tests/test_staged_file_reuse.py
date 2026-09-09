"""
Regression tests for a real, user-reported bug: files picked via Browse
Library (VO, SFX, title/end card VO) sometimes needed to be re-selected
after a first generation, even though the chip still showed them as
selected.

Root cause: after using an uploaded VO/SFX/card-VO file, _run_trailer_job
deleted the ORIGINAL uploaded path unconditionally -- including, for a
network-browsed pick, the shared net_<ts>_<name> staging file that's
specifically meant to survive for reuse (the same protection
_cleanup_job_temp already has, just bypassed here since this deletion
happened separately, with no net_* check at all). A user who generated
once, then tried to reuse the same Browse Library selection for a second
job without re-picking, would find the file silently gone -- the second
job would fall back to no VO/SFX at all, or (for the main source video's
own equivalent check) get an explicit "please re-select" error.

Confirmed directly, not just by reading the fix: reverted to the old
unconditional os.remove() code, ran a real two-generation sequence reusing
the same staged VO file, and confirmed the file was genuinely gone after
the first job with vo_source='none' on the second -- then restored the fix
and confirmed the same sequence keeps the file and vo_source='uploaded'
both times.
"""
import os
import unittest.mock as mock

import pytest

import pipeline


def test_shared_staged_file_survives(tmp_path):
    staged = tmp_path / 'net_1000_myvo.mp3'
    staged.write_bytes(b'fake audio')
    pipeline._remove_job_intermediate(str(staged))
    assert staged.exists(), 'a net_*-prefixed shared staging file must never be deleted here'


def test_non_shared_job_temp_file_is_removed(tmp_path):
    # A direct browser upload's own per-job temp file isn't shared with
    # anything else, so it should still be cleaned up normally.
    local = tmp_path / 'vo_upload_1000_abcdef.mp3'
    local.write_bytes(b'fake audio')
    pipeline._remove_job_intermediate(str(local))
    assert not local.exists()


def test_missing_path_is_a_silent_noop(tmp_path):
    # Must not raise for a path that doesn't exist, or for None/empty.
    pipeline._remove_job_intermediate(str(tmp_path / 'does_not_exist.mp3'))
    pipeline._remove_job_intermediate(None)
    pipeline._remove_job_intermediate('')


def test_removal_failure_does_not_raise(tmp_path):
    staged = tmp_path / 'not_net_prefixed.mp3'
    staged.write_bytes(b'fake audio')
    with mock.patch('pipeline.os.remove', side_effect=OSError('permission denied')):
        pipeline._remove_job_intermediate(str(staged))  # must not raise

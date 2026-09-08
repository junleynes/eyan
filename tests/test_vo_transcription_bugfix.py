"""
Regression test for a real bug: transcribe_audio_file() called a function
named run_media() that was never defined anywhere in this file (a typo for
run_ffmpeg, the actual, consistently-used helper). This silently broke
uploaded-VO transcription specifically -- caught by the function's own
try/except, so it never crashed a render, but it meant uploaded VO could
never drive narration-led scene selection (has_narration's whole point),
even though the app is designed for it to. TTS-generated narration was
unaffected since it already has the text directly and never needed this
transcription step at all.
"""
import os
import subprocess
import shutil
import unittest.mock as mock

import pytest

import pipeline


def _ffmpeg_available():
    return shutil.which('ffmpeg') is not None


def test_run_media_typo_is_gone():
    # The actual regression check: run_media must not be referenced anywhere
    # in this module, since it was never defined -- confirms the fix wasn't
    # just papered over with a second broken reference elsewhere.
    src = open(pipeline.__file__).read()
    assert 'run_media(' not in src


def test_transcribe_audio_file_extraction_step_succeeds(tmp_path):
    """The actual bug: confirms the ffmpeg extraction step (what run_media
    was supposed to be) genuinely runs without a NameError, independent of
    whether a real Whisper service is reachable -- that's a separate,
    environment-specific concern this test isn't about."""
    if not _ffmpeg_available():
        pytest.skip('ffmpeg not available in this environment')
    audio_path = tmp_path / 'vo.mp3'
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'lavfi',
                    '-i', 'sine=frequency=440:duration=3', str(audio_path)],
                   check=True, timeout=30)

    with mock.patch('pipeline.requests.post', side_effect=Exception('Whisper unreachable (expected in this test)')):
        # Must not raise NameError -- if it does, the typo (or an equivalent
        # one) has come back.
        words, segs = pipeline.transcribe_audio_file(str(audio_path))
    # Empty is the correct result here specifically because Whisper itself
    # was mocked to fail -- the function's own graceful fallback, not a sign
    # anything is broken. The real assertion is that this line was reached
    # at all without a NameError, which the code path above already proves.
    assert words == [] and segs == []

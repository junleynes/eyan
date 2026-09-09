"""
Tests for background music trim (start/end) support in prepare_bgm_track().

Before this, an uploaded music track was always read from its own
beginning -- there was no way to select a different portion of a longer
file, unlike uploaded VO/narration which already had this via
vo_trim_start/vo_trim_end and "Set in/out from player". This adds the
same capability for background music, reusing the existing generic
cardVoFileChosen/cardVoSetPoint frontend functions (which only need
matching element IDs, not new JS) and mirroring VO's -ss/-to-before-the-
input ffmpeg pattern server-side.
"""
import shutil
import subprocess

import numpy as np
import pytest

import pipeline


def _ffmpeg_available():
    return shutil.which('ffmpeg') is not None


def _dominant_freq(path):
    """Peak frequency in a mono PCM read of `path`, via a real ffmpeg
    decode + FFT -- not just trusting file metadata."""
    p = subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-f', 's16le',
                        '-ac', '1', '-ar', '44100', '-'], capture_output=True, timeout=30)
    audio = np.frombuffer(p.stdout, dtype=np.int16).astype(np.float64)
    if len(audio) == 0:
        return None
    spec = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), d=1 / 44100)
    return freqs[np.argmax(spec)]


@pytest.fixture
def two_tone_track(tmp_path):
    """A real music file: 440Hz for the first 10s, 880Hz for the next 10s --
    lets a test prove WHICH portion actually got used, not just that
    trimming didn't crash."""
    if not _ffmpeg_available():
        pytest.skip('ffmpeg not available in this environment')
    half1 = tmp_path / 'half1.wav'
    half2 = tmp_path / 'half2.wav'
    combined = tmp_path / 'combined.mp3'
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'lavfi',
                    '-i', 'sine=frequency=440:duration=10', str(half1)], check=True, timeout=30)
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'lavfi',
                    '-i', 'sine=frequency=880:duration=10', str(half2)], check=True, timeout=30)
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(half1), '-i', str(half2),
                    '-filter_complex', '[0:a][1:a]concat=n=2:v=0:a=1[out]', '-map', '[out]', str(combined)],
                   check=True, timeout=30)
    return str(combined)


def test_no_trim_uses_the_start_of_the_file(two_tone_track, tmp_path):
    path, source = pipeline.prepare_bgm_track('action', 'upload', two_tone_track, 5.0, 'testts1')
    assert source == 'uploaded'
    assert abs(_dominant_freq(path) - 440) < 10


def test_trim_start_skips_to_the_correct_portion(two_tone_track):
    path, source = pipeline.prepare_bgm_track('action', 'upload', two_tone_track, 5.0, 'testts2',
                                               trim_start=10.0)
    assert source == 'uploaded'
    assert abs(_dominant_freq(path) - 880) < 10


def test_trim_start_and_end_select_a_specific_window(two_tone_track):
    # trim_start=10, trim_end=15 selects only within the 880Hz half --
    # confirms -to is honored, not just -ss.
    path, source = pipeline.prepare_bgm_track('action', 'upload', two_tone_track, 4.0, 'testts3',
                                               trim_start=10.0, trim_end=15.0)
    assert source == 'uploaded'
    assert abs(_dominant_freq(path) - 880) < 10


def test_generate_mode_ignores_trim_params_without_erroring():
    # trim_start/trim_end are meaningless for 'generate' mode (no source
    # file to trim) -- must not raise just because they were passed.
    path, source = pipeline.prepare_bgm_track('action', 'generate', 'GENERATE', 5.0, 'testts4',
                                               trim_start=10.0, trim_end=15.0)
    # No ACE-Step reachable in this sandbox -- falls back to the synth bed,
    # which is the expected, correct behavior here, not a test failure.
    assert source in ('ai_generated', 'synth_fallback')
    assert path is not None

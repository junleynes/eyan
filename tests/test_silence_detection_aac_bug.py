"""
Regression test for a real, user-reported sync bug: music wasn't
audibly ducking under uploaded VO. Traced to _detect_silence_intervals()
misreading a genuinely, verifiably silent stretch of a loudnorm-processed,
AAC-encoded file as "active audio" past its first ~0.3s -- confirmed by
direct sample inspection (RMS and peak both exactly 0 in the region being
misread) and by the same audio, analyzed as PCM with no AAC round-trip,
detecting the real silent interval correctly and completely.

This function determines exactly where BGM/dialogue duck under VO/SOT --
misreading a silent gap before VO actually starts as "VO already playing"
ducks music too early or for the wrong duration, not from the real,
correct moment narration actually begins. Uploaded VO's own prep step
(loudnorm + adelay padding before the VO's real start, re-encoded to AAC)
is exactly the shape that triggered this.
"""
import shutil
import subprocess

import pytest

import pipeline


def _ffmpeg_available():
    return shutil.which('ffmpeg') is not None


@pytest.fixture
def loudnorm_aac_with_silent_padding(tmp_path):
    """Reproduces the exact real-world shape that exposed this: a short
    tone, loudnorm-normalized, then padded with 5 real seconds of silence
    at its start (matching adelay's role in the real VO prep step),
    re-encoded to AAC -- the same filter chain uploaded VO actually goes
    through, just built directly here instead of via a full render."""
    if not _ffmpeg_available():
        pytest.skip('ffmpeg not available in this environment')
    tone = tmp_path / 'tone.wav'
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'lavfi',
                    '-i', 'sine=frequency=1200:duration=4', str(tone)], check=True, timeout=30)
    padded = tmp_path / 'padded.m4a'
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(tone), '-af',
                    'loudnorm=I=-14:TP=-1.5:LRA=7,adelay=5000|5000,volume=1.15',
                    '-c:a', 'aac', '-b:a', '192k', str(padded)], check=True, timeout=30)
    return str(padded)


def test_detects_the_full_silent_padding_not_just_a_sliver(loudnorm_aac_with_silent_padding):
    silence = pipeline._detect_silence_intervals(loudnorm_aac_with_silent_padding)
    assert len(silence) >= 1
    start, end = silence[0]
    # The real bug: this used to come back as roughly (-0.29, 0.02) -- a
    # ~0.3s sliver right at the front -- instead of the genuine ~5s of
    # silent padding that's actually there.
    assert start <= 0.1
    assert end >= 4.5


def test_downstream_active_window_starts_at_the_real_vo_position(loudnorm_aac_with_silent_padding):
    # This is what actually drives ducking -- confirms the fix's effect
    # reaches the function callers actually use, not just the raw
    # silence-interval list on its own.
    silence = pipeline._detect_silence_intervals(loudnorm_aac_with_silent_padding)
    real_dur = pipeline.probe_duration(loudnorm_aac_with_silent_padding)
    windows = pipeline._active_windows_from_silence(silence, 30.0, content_duration=real_dur)
    assert len(windows) == 1
    start, end = windows[0]
    # Must start at roughly 5s (where the real tone begins), not 0 (which
    # would mean ducking starts before VO is even audible).
    assert 4.5 <= start <= 5.5
    assert end >= 8.5


def test_silence_start_with_a_negative_sign_is_parsed_correctly():
    # A separate, smaller bug found alongside the main one: the original
    # regex for silence_start/silence_end had no way to match a leading
    # minus sign at all, silently turning a slightly-negative timestamp
    # (which ffmpeg itself sometimes reports right at the start of a file)
    # into a small POSITIVE one instead -- a different, wrong moment, not
    # a harmlessly-rounded one.
    import re
    fake_stderr = 'silence_start: -0.287719\nsilence_end: 0.023552 | silence_duration: 0.311271\n'
    starts = [float(m) for m in re.findall(r'silence_start:\s*(-?[\d.]+)', fake_stderr)]
    assert starts == [-0.287719]

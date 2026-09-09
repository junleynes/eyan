"""
Tests for /api/network/combine's segment_durations return value -- added so
a script's "M1"/"M2" cues (see tests/test_script_parsing.py's
TestSegmentOffsets) can be offset to the right point in the single combined
video the render actually works with. Each source file's own duration,
probed before combining, in the same order they're being combined.
"""
import os
import re
import shutil
import subprocess

import pytest

import unittest.mock as mock

with mock.patch('requests.post'), mock.patch('requests.get'):
    import main


def _ffmpeg_available():
    return shutil.which('ffmpeg') is not None


@pytest.fixture
def two_staged_files(tmp_path, monkeypatch):
    """Two real video files with known, distinct durations, placed as if
    already fetched from a network share (net_*-prefixed, matching what
    /api/network/fetch actually produces)."""
    if not _ffmpeg_available():
        pytest.skip('ffmpeg not available in this environment')
    app = main.app
    upload_dir = tmp_path / 'uploads'
    upload_dir.mkdir()
    monkeypatch.setitem(app.config, 'UPLOAD_FOLDER', str(upload_dir))

    seg1 = upload_dir / 'net_1000_seg1.mp4'
    seg2 = upload_dir / 'net_1000_seg2.mp4'
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'lavfi',
                    '-i', 'color=c=red:s=320x240:d=8:r=25', '-f', 'lavfi',
                    '-i', 'sine=frequency=440:duration=8', '-c:v', 'libx264',
                    '-c:a', 'aac', '-shortest', str(seg1)], check=True, timeout=30)
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'lavfi',
                    '-i', 'color=c=blue:s=320x240:d=5:r=25', '-f', 'lavfi',
                    '-i', 'sine=frequency=880:duration=5', '-c:v', 'libx264',
                    '-c:a', 'aac', '-shortest', str(seg2)], check=True, timeout=30)

    client = app.test_client()
    csrf_token = 'test-csrf-combine'
    with client.session_transaction() as sess:
        sess['authed'] = True
        sess['user_id'] = 1
        sess['username'] = 'admin'
        sess['role'] = 'admin'
        sess['csrf_token'] = csrf_token
    return client, {'X-CSRF-Token': csrf_token}


def test_combine_returns_each_segments_own_duration_in_order(two_staged_files):
    client, headers = two_staged_files
    r = client.post('/api/network/combine',
                    json={'files': ['net_1000_seg1.mp4', 'net_1000_seg2.mp4']}, headers=headers)
    assert r.status_code == 200, r.get_data(as_text=True)
    d = r.get_json()
    assert d['ok'] is True
    durations = d['segment_durations']
    assert len(durations) == 2
    assert abs(durations[0] - 8.0) < 0.5
    assert abs(durations[1] - 5.0) < 0.5


def test_combined_file_duration_matches_the_sum_of_segments(two_staged_files):
    # The combined output's own real duration should match the sum of what
    # segment_durations reports -- confirms these numbers describe the
    # actual file produced, not just independently-probed originals that
    # might not reflect what concat actually did.
    client, headers = two_staged_files
    r = client.post('/api/network/combine',
                    json={'files': ['net_1000_seg1.mp4', 'net_1000_seg2.mp4']}, headers=headers)
    d = r.get_json()
    app = main.app
    combined_path = os.path.join(app.config['UPLOAD_FOLDER'], d['filename'])
    p = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                        '-of', 'csv=p=0', combined_path], capture_output=True, text=True, timeout=10)
    actual_duration = float(p.stdout.strip())
    assert abs(actual_duration - sum(d['segment_durations'])) < 0.5

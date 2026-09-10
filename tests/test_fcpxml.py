"""
Tests for build_fcpxml() and its route -- exporting a render's selected
scenes as a legacy "Final Cut Pro XML Interchange Format" (xmeml version
5) sequence, deliberately NOT modern FCPXML (Final Cut Pro X's own,
incompatible format): the old dialect is what Premiere Pro and DaVinci
Resolve both actually read, while Premiere cannot import modern FCPXML
at all without a paid third-party converter -- for a facility with
editors on different NLEs, that's the format that's actually broadly
useful.

Frame-rate/timecode handling is the single most fragile part of this
format (an easy, well-documented, silent way to get every timecode in
a file wrong by mixing up true-24fps vs 23.976fps), so that gets
proportionally more coverage here than the rest.
"""
import json
import os
import shutil
import subprocess
import time
import unittest.mock as mock
import xml.etree.ElementTree as ET
from io import BytesIO

import pytest

import core
import pipeline

with mock.patch('requests.post'), mock.patch('requests.get'):
    import main


def _ffmpeg_available():
    return shutil.which('ffmpeg') is not None


# ---- _fps_to_timebase_ntsc ----

@pytest.mark.parametrize('fps,expected_timebase,expected_ntsc', [
    (23.976, 24, True),   # the classic, easy-to-get-wrong case
    (24.0, 24, False),
    (25.0, 25, False),
    (29.97, 30, True),
    (30.0, 30, False),
    (59.94, 60, True),
    (60.0, 60, False),
])
def test_fps_to_timebase_ntsc_exact_matches(fps, expected_timebase, expected_ntsc):
    timebase, ntsc = pipeline._fps_to_timebase_ntsc(fps)
    assert (timebase, ntsc) == (expected_timebase, expected_ntsc)


def test_fps_to_timebase_ntsc_snaps_a_slightly_off_measurement():
    # A real-world measured fps is rarely bit-exact (23.9760104..., etc.)
    # -- confirms a close-but-not-exact value still snaps to the right
    # standard rather than being left unmapped.
    timebase, ntsc = pipeline._fps_to_timebase_ntsc(23.976023976)
    assert (timebase, ntsc) == (24, True)


def test_fps_to_timebase_ntsc_zero_fallback_does_not_crash():
    timebase, ntsc = pipeline._fps_to_timebase_ntsc(0)
    assert timebase > 0


# ---- _path_to_file_url ----

def test_path_to_file_url_windows_style():
    assert pipeline._path_to_file_url(r'C:\AI-Tools\Prism\uploads\net_1_x.mp4') == \
        'file://localhost/C:/AI-Tools/Prism/uploads/net_1_x.mp4'


def test_path_to_file_url_posix_style():
    assert pipeline._path_to_file_url('/srv/uploads/net_1_x.mp4') == \
        'file://localhost/srv/uploads/net_1_x.mp4'


# ---- build_fcpxml: error cases ----

def test_no_scenes_raises_clear_error():
    row = {'filename': 'x.mp4', 'orig_name': 'x.mp4',
           'result_json': json.dumps({'source_video_path': '/tmp/x.mp4', 'scenes': []})}
    with pytest.raises(ValueError, match='no recorded scene selection'):
        pipeline.build_fcpxml(row)


def test_no_source_path_raises_clear_error():
    row = {'filename': 'x.mp4', 'orig_name': 'x.mp4',
           'result_json': json.dumps({'scenes': [{'scene': 1, 'start': 0, 'end': 5}]})}
    with pytest.raises(ValueError, match='no recorded source video'):
        pipeline.build_fcpxml(row)


def test_missing_source_file_raises_clear_error():
    row = {'filename': 'x.mp4', 'orig_name': 'x.mp4',
           'result_json': json.dumps({'source_video_path': '/tmp/does_not_exist_xyz123.mp4',
                                       'scenes': [{'scene': 1, 'start': 0, 'end': 5}]})}
    with pytest.raises(ValueError, match='no longer available'):
        pipeline.build_fcpxml(row)


# ---- build_fcpxml: real video, real math ----

@pytest.fixture
def test_source_25fps(tmp_path):
    if not _ffmpeg_available():
        pytest.skip('ffmpeg not available in this environment')
    parts = []
    for i, color in enumerate(['red', 'blue', 'green']):
        part = tmp_path / f'part{i}.mp4'
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'lavfi',
                        '-i', f'color=c={color}:s=640x360:d=6:r=25',
                        '-f', 'lavfi', '-i', f'sine=frequency={200 + i * 100}:duration=6',
                        '-c:v', 'libx264', '-c:a', 'aac', '-shortest', str(part)], check=True, timeout=30)
        parts.append(part)
    list_file = tmp_path / 'list.txt'
    list_file.write_text('\n'.join(f"file '{p}'" for p in parts))
    src = tmp_path / 'src.mp4'
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0',
                    '-i', str(list_file), '-c', 'copy', str(src)], check=True, timeout=30)
    return src


def test_well_formed_xml_with_correct_clip_count(test_source_25fps):
    row = {'filename': 'x.mp4', 'orig_name': 'src.mp4', 'result_json': json.dumps({
        'source_video_path': str(test_source_25fps),
        'scenes': [
            {'scene': 1, 'start': 1.0, 'end': 4.0, 'description': 'Opening'},
            {'scene': 2, 'start': 8.0, 'end': 12.0, 'description': 'Middle'},
            {'scene': 3, 'start': 14.0, 'end': 17.5, 'description': 'Closing'},
        ],
    })}
    xml_text = pipeline.build_fcpxml(row)
    root = ET.fromstring(xml_text)  # raises if not well-formed
    assert root.tag == 'xmeml'
    assert root.get('version') == '5'
    assert len(root.findall('.//clipitem')) == 3


def test_frame_math_is_exact_at_25fps(test_source_25fps):
    row = {'filename': 'x.mp4', 'orig_name': 'src.mp4', 'result_json': json.dumps({
        'source_video_path': str(test_source_25fps),
        'scenes': [
            {'scene': 1, 'start': 1.0, 'end': 4.0, 'description': ''},
            {'scene': 2, 'start': 8.0, 'end': 12.0, 'description': ''},
        ],
    })}
    root = ET.fromstring(pipeline.build_fcpxml(row))
    assert root.find('.//timebase').text == '25'
    assert root.find('.//ntsc').text == 'FALSE'
    clips = root.findall('.//clipitem')
    # scene 1: 1.0s-4.0s @ 25fps -> in=25, out=100
    assert clips[0].find('in').text == '25'
    assert clips[0].find('out').text == '100'
    assert clips[0].find('start').text == '0'    # first clip starts at 0 on the sequence timeline
    assert clips[0].find('end').text == '75'      # 100-25 = 75 frames long
    # scene 2: 8.0s-12.0s @ 25fps -> in=200, out=300, placed right after scene 1 on the sequence
    assert clips[1].find('in').text == '200'
    assert clips[1].find('out').text == '300'
    assert clips[1].find('start').text == '75'    # continues immediately after scene 1's end
    assert clips[1].find('end').text == '175'


def test_only_first_clipitem_carries_the_full_file_definition(test_source_25fps):
    # The id-attribute inheritance convention: repeating the same file's
    # full metadata (name/pathurl/rate/dimensions) on every single
    # clipitem would work, but a real FCP/Premiere export shares it via
    # one definition plus references -- confirms this implementation
    # does the same, not just "technically valid XML".
    row = {'filename': 'x.mp4', 'orig_name': 'src.mp4', 'result_json': json.dumps({
        'source_video_path': str(test_source_25fps),
        'scenes': [{'scene': 1, 'start': 0, 'end': 3, 'description': ''},
                   {'scene': 2, 'start': 4, 'end': 5, 'description': ''}],
    })}
    root = ET.fromstring(pipeline.build_fcpxml(row))
    files = root.findall('.//file')
    assert len(files) == 2
    assert files[0].get('id') == files[1].get('id') == 'file-1'
    assert files[0].find('name') is not None
    assert len(list(files[1])) == 0  # second is a bare, self-closing reference


def test_xml_special_characters_in_description_do_not_break_parsing(test_source_25fps):
    row = {'filename': 'x.mp4', 'orig_name': 'src.mp4', 'result_json': json.dumps({
        'source_video_path': str(test_source_25fps),
        'scenes': [{'scene': 1, 'start': 0, 'end': 3,
                    'description': 'A "quoted" & tricky <description>'}],
    })}
    root = ET.fromstring(pipeline.build_fcpxml(row))  # raises if escaping is wrong
    assert root.find('.//description').text == 'A "quoted" & tricky <description>'


def test_23_976_fps_is_correctly_flagged_ntsc_true(tmp_path):
    # The single highest-risk case this feature has: true 24fps and
    # 23.976fps share the same <timebase>24</timebase> and are
    # distinguished ONLY by <ntsc>, so this gets its own dedicated test
    # against a source built at an exact, verified 24000/1001 rate.
    if not _ffmpeg_available():
        pytest.skip('ffmpeg not available in this environment')
    src = tmp_path / 'ntsc_src.mp4'
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'lavfi',
                    '-i', 'color=c=red:s=640x360:d=6:r=24000/1001',
                    '-f', 'lavfi', '-i', 'sine=frequency=440:duration=6',
                    '-c:v', 'libx264', '-c:a', 'aac', '-shortest', str(src)], check=True, timeout=30)
    row = {'filename': 'x.mp4', 'orig_name': 'ntsc_src.mp4', 'result_json': json.dumps({
        'source_video_path': str(src),
        'scenes': [{'scene': 1, 'start': 1.0, 'end': 4.0, 'description': ''}],
    })}
    root = ET.fromstring(pipeline.build_fcpxml(row))
    assert root.find('.//timebase').text == '24'
    assert root.find('.//ntsc').text == 'TRUE'
    # Frame math must use the REAL 23.976 rate, not a naive 24.0 --
    # 1.0s * 23.976 rounds to 24, not 24.0 exactly.
    assert root.find('.//clipitem/in').text == '24'


# ---- The real route, end to end ----

@pytest.fixture
def client_and_headers(tmp_path, monkeypatch):
    if not _ffmpeg_available():
        pytest.skip('ffmpeg not available in this environment')
    app = main.app
    upload_dir = tmp_path / 'uploads'
    upload_dir.mkdir()
    monkeypatch.setitem(app.config, 'UPLOAD_FOLDER', str(upload_dir))
    monkeypatch.setattr(pipeline, 'ALLOW_LOCAL_MEDIA_UPLOAD', True)
    client = app.test_client()
    csrf_token = 'test-csrf-fcpxml'
    with client.session_transaction() as sess:
        sess['authed'] = True
        sess['user_id'] = 1
        sess['username'] = 'admin'
        sess['role'] = 'admin'
        sess['csrf_token'] = csrf_token
    return client, {'X-CSRF-Token': csrf_token}, str(upload_dir)


def test_full_route_after_a_real_network_staged_render(client_and_headers, tmp_path):
    # A direct upload is cleaned up after rendering (see build_fcpxml's
    # own docstring) -- only a network-staged source survives, so this
    # simulates that path specifically (matching what fetch_network_file
    # actually produces: net_<timestamp>_<name> already sitting in
    # UPLOAD_FOLDER) rather than a plain multipart upload.
    client, headers, upload_dir = client_and_headers
    parts = []
    for i, color in enumerate(['red', 'blue', 'green', 'yellow', 'purple', 'cyan']):
        part = tmp_path / f'part{i}.mp4'
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'lavfi',
                        '-i', f'color=c={color}:s=640x360:d=6:r=25',
                        '-f', 'lavfi', '-i', f'sine=frequency={200 + i * 100}:duration=6',
                        '-c:v', 'libx264', '-c:a', 'aac', '-shortest', str(part)], check=True, timeout=30)
        parts.append(part)
    list_file = tmp_path / 'list.txt'
    list_file.write_text('\n'.join(f"file '{p}'" for p in parts))
    src = tmp_path / 'src.mp4'
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0',
                    '-i', str(list_file), '-c', 'copy', str(src)], check=True, timeout=30)

    staged_name = 'net_1789041900_src.mp4'
    shutil.copy(str(src), os.path.join(upload_dir, staged_name))

    core._job_submit_limiter.buckets.clear()
    r = client.post('/api/trailer/generate', data={
        'network_file': staged_name,
        'genre': '', 'trailer_length': '15', 'scoring_mode': 'none',
        'sfx_mode': 'none', 'vo_mode': 'none', 'transition': 'cut',
    }, headers=headers, content_type='multipart/form-data')
    assert r.status_code == 200, r.get_data(as_text=True)

    deadline = time.time() + 60
    d = None
    while time.time() < deadline:
        d = client.get(f'/api/trailer/progress/{r.get_json()["job_id"]}', headers=headers).get_json()
        if d.get('done'):
            break
        time.sleep(1)
    assert d and d.get('error') is None
    library_id = (d.get('result') or {}).get('library_id')
    assert library_id

    resp = client.get(f'/library/{library_id}/fcpxml', headers=headers)
    assert resp.status_code == 200
    assert resp.headers['Content-Type'].startswith('application/xml')
    assert 'attachment' in resp.headers.get('Content-Disposition', '')
    root = ET.fromstring(resp.get_data())
    assert root.tag == 'xmeml'
    assert len(root.findall('.//clipitem')) == (d.get('result') or {}).get('selected_scenes')


def test_route_404s_for_a_nonexistent_trailer(client_and_headers):
    client, headers, _ = client_and_headers
    resp = client.get('/library/999999/fcpxml', headers=headers)
    assert resp.status_code == 404

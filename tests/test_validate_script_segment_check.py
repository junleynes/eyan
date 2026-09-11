"""
Regression test for a real bug found while merging two commits that both
touched parse_script_cues() and its validation-preview counterpart,
api_validate_script(): the segment-name safety check in both places was
gated on `if segment_offsets:` (or `if seg_match and segment_offsets:`),
so a script line naming a segment (e.g. "M2") on an ordinary single-file
job -- where segment_offsets is empty/absent entirely, not just missing
that one segment -- fell through the check and was kept/reported as
usable with its raw, unadjusted timecode, instead of being dropped.

test_script_parsing.py already covers parse_script_cues() itself
directly; this covers api_validate_script(), which had an independent
copy of the same logic (not shared code) and needed its own, separate
fix to stay consistent with what parsing actually does -- otherwise the
validation preview would show a cue as "usable" that a real render
would silently drop.
"""
import json
import unittest.mock as mock
from io import BytesIO

with mock.patch('requests.post'), mock.patch('requests.get'):
    import main


def _client_with_session():
    app = main.app
    client = app.test_client()
    csrf_token = 'test-csrf-validate-script'
    with client.session_transaction() as sess:
        sess['authed'] = True
        sess['user_id'] = 1
        sess['username'] = 'admin'
        sess['role'] = 'admin'
        sess['csrf_token'] = csrf_token
    return client, {'X-CSRF-Token': csrf_token}


def test_named_segment_with_no_combine_offsets_is_reported_as_skipped():
    client, headers = _client_with_session()
    script_text = "M2 0:05 a line naming a segment that was never combined"
    r = client.post('/api/trailer/validate_script', data={
        'script_file': (BytesIO(script_text.encode()), 'script.txt'),
    }, headers=headers, content_type='multipart/form-data')
    assert r.status_code == 200, r.get_data(as_text=True)
    body = r.get_json()
    # The cue must NOT show up as usable/kept -- it must be reported as
    # skipped, matching what parse_script_cues() itself actually does
    # with the same line (drops it entirely).
    kept_times = [c['raw_time'] for c in (body.get('cues') or [])]
    assert 5.0 not in kept_times
    assert any('not combined' in (s.get('reason') or '') for s in (body.get('skipped') or []))


def test_segment_1_with_no_combine_offsets_is_still_usable():
    # Segment 1 always maps to offset 0 regardless of whether any combine
    # happened at all -- confirms the fix didn't overcorrect into treating
    # every named segment as unusable without offsets.
    client, headers = _client_with_session()
    script_text = "M1 0:05 the first segment, always fine on its own"
    r = client.post('/api/trailer/validate_script', data={
        'script_file': (BytesIO(script_text.encode()), 'script.txt'),
    }, headers=headers, content_type='multipart/form-data')
    assert r.status_code == 200, r.get_data(as_text=True)
    body = r.get_json()
    kept_times = [c['raw_time'] for c in (body.get('cues') or [])]
    assert 5.0 in kept_times


def test_named_segment_with_real_combine_offsets_is_shifted_correctly():
    client, headers = _client_with_session()
    script_text = "M2 0:05 a line naming a segment that WAS combined"
    r = client.post('/api/trailer/validate_script', data={
        'script_file': (BytesIO(script_text.encode()), 'script.txt'),
        'file_segment_durations': json.dumps([10.0, 8.0]),  # segment 2 starts at 10.0
    }, headers=headers, content_type='multipart/form-data')
    assert r.status_code == 200, r.get_data(as_text=True)
    body = r.get_json()
    kept = body.get('cues') or []
    matching = [c for c in kept if c['raw_time'] == 5.0]
    assert len(matching) == 1
    assert matching[0]['time'] == 15.0  # 5.0 + segment 2's 10.0s offset

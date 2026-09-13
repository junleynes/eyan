"""
Tests for Timecode Priority's "Manual Entry" mode: typing one or more
(material, start, end) timecode rows directly in the UI, with no script
file involved at all, as an alternative to Script Upload.

manual_cues_to_script_text() converts these structured rows into
synthetic script-line text ("M1 3:33 3:37") and hands them to the exact
same parse_script_cues() a real uploaded script goes through -- so these
tests focus on that conversion and its own validation; parse_script_cues
itself already has its own dedicated test coverage.
"""
import json
import unittest.mock as mock

with mock.patch('requests.post'), mock.patch('requests.get'):
    import main
import pipeline


def test_single_entry_converts_and_parses_correctly():
    entries = [{'material': 1, 'start': '3:33', 'end': '3:37'}]
    text, err = pipeline.manual_cues_to_script_text(json.dumps(entries))
    assert err is None
    cues = pipeline.parse_script_cues(text, available_materials={1})
    assert len(cues) == 1
    assert cues[0]['time'] == 213  # 3*60+33
    assert cues[0]['material'] == 1
    assert cues[0]['out'] == 217  # 3*60+37


def test_multiple_entries_across_different_materials():
    entries = [
        {'material': 1, 'start': '3:33', 'end': '3:37'},
        {'material': 2, 'start': '00:35', 'end': '00:42'},
    ]
    text, err = pipeline.manual_cues_to_script_text(json.dumps(entries))
    assert err is None
    cues = pipeline.parse_script_cues(text, available_materials={1, 2})
    assert len(cues) == 2
    assert {c['material'] for c in cues} == {1, 2}


def test_end_timecode_is_optional():
    entries = [{'material': 1, 'start': '3:33', 'end': ''}]
    text, err = pipeline.manual_cues_to_script_text(json.dumps(entries))
    assert err is None
    cues = pipeline.parse_script_cues(text, available_materials={1})
    assert len(cues) == 1
    assert cues[0].get('out') is None


def test_entry_naming_a_material_that_was_not_loaded_is_dropped_by_parse_script_cues():
    # manual_cues_to_script_text itself doesn't know what was loaded --
    # that check is parse_script_cues' own job, exactly as it already is
    # for a real script upload, so this is only confirming the conversion
    # doesn't short-circuit or duplicate that check.
    entries = [{'material': 3, 'start': '1:00', 'end': ''}]
    text, err = pipeline.manual_cues_to_script_text(json.dumps(entries))
    assert err is None
    cues = pipeline.parse_script_cues(text, available_materials={1, 2})
    assert cues == []


def test_empty_list_is_an_error():
    text, err = pipeline.manual_cues_to_script_text('[]')
    assert text is None
    assert 'No manual timecode entries' in err


def test_missing_or_empty_json_is_an_error():
    text, err = pipeline.manual_cues_to_script_text('')
    assert text is None
    assert err is not None


def test_unparseable_json_is_an_error():
    text, err = pipeline.manual_cues_to_script_text('not valid json')
    assert text is None
    assert 'readable format' in err


def test_entry_missing_material_is_a_specific_error():
    text, err = pipeline.manual_cues_to_script_text(json.dumps([{'start': '1:30'}]))
    assert text is None
    assert 'entry #1' in err
    assert 'material' in err


def test_entry_missing_start_timecode_is_a_specific_error():
    text, err = pipeline.manual_cues_to_script_text(json.dumps([{'material': 1}]))
    assert text is None
    assert 'entry #1' in err
    assert 'start timecode' in err


def test_entry_with_unrecognisable_start_timecode_is_a_specific_error():
    text, err = pipeline.manual_cues_to_script_text(
        json.dumps([{'material': 1, 'start': 'not a timecode'}]))
    assert text is None
    assert 'entry #1' in err
    assert 'not a timecode' in err


def test_second_bad_entry_is_identified_by_its_own_number_not_the_first():
    entries = [
        {'material': 1, 'start': '1:00', 'end': ''},
        {'material': 2, 'start': ''},
    ]
    text, err = pipeline.manual_cues_to_script_text(json.dumps(entries))
    assert text is None
    assert 'entry #2' in err


def test_non_list_json_is_an_error():
    text, err = pipeline.manual_cues_to_script_text(json.dumps({'material': 1}))
    assert text is None


def test_non_dict_row_is_a_specific_error():
    text, err = pipeline.manual_cues_to_script_text(json.dumps(["not a dict"]))
    assert text is None
    assert 'entry #1' in err


def test_full_validate_script_endpoint_with_manual_entries():
    app = main.app
    client = app.test_client()
    csrf_token = 'test-csrf-manual'
    with client.session_transaction() as sess:
        sess['authed'] = True
        sess['user_id'] = 1
        sess['username'] = 'admin'
        sess['role'] = 'admin'
        sess['csrf_token'] = csrf_token
    headers = {'X-CSRF-Token': csrf_token}

    entries = [
        {'material': 1, 'start': '3:33', 'end': '3:37'},
        {'material': 2, 'start': '00:35', 'end': '00:42'},
    ]
    r = client.post('/api/trailer/validate_script', data={
        'manual_cues': json.dumps(entries),
        'materials_network': json.dumps(['a.mp4', 'b.mp4']),
    }, headers=headers)
    assert r.status_code == 200
    body = r.get_json()
    assert body['ok'] is True
    assert body['cues_count'] == 2
    materials = {c['material'] for c in body['cues']}
    assert materials == {'M1', 'M2'}


def test_full_validate_script_endpoint_rejects_bad_manual_entries():
    app = main.app
    client = app.test_client()
    csrf_token = 'test-csrf-manual-bad'
    with client.session_transaction() as sess:
        sess['authed'] = True
        sess['user_id'] = 1
        sess['username'] = 'admin'
        sess['role'] = 'admin'
        sess['csrf_token'] = csrf_token
    headers = {'X-CSRF-Token': csrf_token}

    r = client.post('/api/trailer/validate_script', data={
        'manual_cues': json.dumps([{'material': 1, 'start': ''}]),
    }, headers=headers)
    assert r.status_code == 400
    assert r.get_json()['ok'] is False

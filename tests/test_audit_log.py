"""
Tests for audit_log/audit_log_list -- the accountability trail behind
Config > Audit Log.
"""
import os
import sys
import unittest.mock as mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

with mock.patch('requests.post'), mock.patch('requests.get'):
    import library_db


def _use_temp_lib_db(tmp_path, monkeypatch):
    db_path = str(tmp_path / 'test_library.db')
    monkeypatch.setattr(library_db, 'LIBRARY_DB_PATH', db_path)
    library_db.library_db_init()


def test_log_entry_round_trips(tmp_path, monkeypatch):
    _use_temp_lib_db(tmp_path, monkeypatch)
    library_db.audit_log('user_create', target='alice', detail='role=user',
                          user_id=1, username='admin', ip='127.0.0.1')
    items = library_db.audit_log_list()
    assert len(items) == 1
    assert items[0]['action'] == 'user_create'
    assert items[0]['target'] == 'alice'
    assert items[0]['detail'] == 'role=user'
    assert items[0]['username'] == 'admin'
    assert items[0]['ip'] == '127.0.0.1'


def test_optional_fields_default_to_none(tmp_path, monkeypatch):
    _use_temp_lib_db(tmp_path, monkeypatch)
    library_db.audit_log('trailer_generate')
    items = library_db.audit_log_list()
    assert items[0]['target'] is None
    assert items[0]['detail'] is None
    assert items[0]['user_id'] is None


def test_list_is_newest_first(tmp_path, monkeypatch):
    _use_temp_lib_db(tmp_path, monkeypatch)
    library_db.audit_log('first_action')
    library_db.audit_log('second_action')
    library_db.audit_log('third_action')
    items = library_db.audit_log_list()
    assert [it['action'] for it in items] == ['third_action', 'second_action', 'first_action']


def test_list_respects_limit(tmp_path, monkeypatch):
    _use_temp_lib_db(tmp_path, monkeypatch)
    for i in range(10):
        library_db.audit_log(f'action_{i}')
    items = library_db.audit_log_list(limit=3)
    assert len(items) == 3
    # Still newest-first even when truncated
    assert items[0]['action'] == 'action_9'

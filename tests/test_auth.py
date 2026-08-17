"""
Tests for auth.py's password policy (pure) and the account/TOTP lifecycle
(backed by a real temp SQLite DB per test, via monkeypatching USERS_DB_PATH --
these functions genuinely touch the database, so a pure-function test would
not actually exercise the real code path).
"""
import os
import sys
import unittest.mock as mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

with mock.patch('requests.post'), mock.patch('requests.get'):
    import auth


class TestPasswordPolicy:
    def test_too_short_is_rejected(self):
        ok, reason = auth._password_policy_error('short1')
        assert ok is False
        assert '12' in reason

    def test_exactly_minimum_length_is_accepted(self):
        ok, reason = auth._password_policy_error('a' * 12)
        assert ok is True

    def test_common_password_rejected_even_if_long_enough(self):
        ok, reason = auth._password_policy_error('password123')
        assert ok is False

    def test_genuinely_strong_password_accepted(self):
        ok, reason = auth._password_policy_error('correct horse battery staple 42')
        assert ok is True

    def test_empty_password_rejected(self):
        ok, reason = auth._password_policy_error('')
        assert ok is False

    def test_none_password_rejected_not_crashed(self):
        ok, reason = auth._password_policy_error(None)
        assert ok is False


class TestAccountAndTotpLifecycle:
    """Each test gets its own throwaway users.db via the users_db fixture in
    conftest.py, so these can freely create/modify accounts without any
    cross-test interference or touching a real deployment's data."""

    def test_user_create_enforces_password_policy(self, users_db):
        ok, err = auth.user_create('newuser', 'short')
        assert ok is False
        assert auth.user_get_by_username('newuser') is None

    def test_user_create_success(self, users_db):
        ok, err = auth.user_create('newuser', 'a-genuinely-long-password-1', role='user')
        assert ok is True
        user = auth.user_get_by_username('newuser')
        assert user is not None
        assert user['role'] == 'user'
        assert bool(user['is_active']) is True

    def test_self_registration_creates_inactive_account(self, users_db):
        ok, err = auth.user_create('pending', 'a-genuinely-long-password-1', role='user', active=False)
        assert ok is True
        user = auth.user_get_by_username('pending')
        assert bool(user['is_active']) is False

    def test_totp_not_enabled_until_confirmed(self, users_db):
        auth.user_create('totpuser', 'a-genuinely-long-password-1')
        user = auth.user_get_by_username('totpuser')
        secret, uri = auth.user_totp_begin_enrolment(user['id'])
        assert secret is not None
        assert uri.startswith('otpauth://totp/')
        # Enrolment started but not confirmed -- must still read as disabled.
        still_user = auth.user_get(user['id'])
        assert bool(still_user['totp_enabled']) is False

    def test_totp_wrong_code_does_not_enable(self, users_db):
        auth.user_create('totpuser2', 'a-genuinely-long-password-1')
        user = auth.user_get_by_username('totpuser2')
        auth.user_totp_begin_enrolment(user['id'])
        codes, err = auth.user_totp_confirm(user['id'], '000000')
        assert codes is None
        assert err is not None
        assert bool(auth.user_get(user['id'])['totp_enabled']) is False

    def test_totp_correct_code_enables_and_issues_backup_codes(self, users_db):
        import pyotp
        auth.user_create('totpuser3', 'a-genuinely-long-password-1')
        user = auth.user_get_by_username('totpuser3')
        secret, uri = auth.user_totp_begin_enrolment(user['id'])
        valid_code = pyotp.TOTP(secret).now()
        codes, err = auth.user_totp_confirm(user['id'], valid_code)
        assert err is None
        assert len(codes) == 10
        assert bool(auth.user_get(user['id'])['totp_enabled']) is True

    def test_backup_code_works_once_only(self, users_db):
        import pyotp
        auth.user_create('totpuser4', 'a-genuinely-long-password-1')
        user = auth.user_get_by_username('totpuser4')
        secret, uri = auth.user_totp_begin_enrolment(user['id'])
        codes, err = auth.user_totp_confirm(user['id'], pyotp.TOTP(secret).now())
        one_code = codes[0]
        fresh_user = auth.user_get(user['id'])
        assert auth.user_totp_verify(fresh_user, one_code) is True
        # Re-fetch: the used code must be consumed, not reusable
        fresh_user2 = auth.user_get(user['id'])
        assert auth.user_totp_verify(fresh_user2, one_code) is False


class TestAccountLockout:
    def test_lockout_triggers_after_max_failures(self, users_db):
        auth.user_create('lockme', 'a-genuinely-long-password-1')
        user = auth.user_get_by_username('lockme')
        locked_until = None
        for _ in range(auth.LOGIN_MAX_FAILURES):
            locked_until = auth.user_note_failed_login(user['id'])
        assert locked_until is not None
        fresh = auth.user_get(user['id'])
        assert auth.user_lockout_remaining(fresh) > 0

    def test_not_locked_before_reaching_the_threshold(self, users_db):
        auth.user_create('notyet', 'a-genuinely-long-password-1')
        user = auth.user_get_by_username('notyet')
        for _ in range(auth.LOGIN_MAX_FAILURES - 1):
            auth.user_note_failed_login(user['id'])
        fresh = auth.user_get(user['id'])
        assert auth.user_lockout_remaining(fresh) == 0

    def test_successful_login_clears_the_failure_count(self, users_db):
        auth.user_create('recovers', 'a-genuinely-long-password-1')
        user = auth.user_get_by_username('recovers')
        for _ in range(auth.LOGIN_MAX_FAILURES - 1):
            auth.user_note_failed_login(user['id'])
        auth.user_touch_login(user['id'])
        fresh = auth.user_get(user['id'])
        assert fresh['failed_logins'] == 0
        assert fresh['locked_until'] is None

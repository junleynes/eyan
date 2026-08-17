"""
Shared pytest fixtures for the whole suite.

pipeline.py makes a couple of module-level calls out to local services when
imported directly (checking Ollama/Whisper/etc. reachability for the
service-status banner). Patching requests before importing means the test
run doesn't depend on any of those actually being up, and doesn't hang or
error out on a machine that doesn't have them installed at all -- exactly
the situation this sandbox and CI both are in.
"""
import os
import sys
import unittest.mock as mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault('LIBRARY_DIR', '/tmp/prism_test_library')
os.environ.setdefault('SECRET_KEY_FILE', '/tmp/prism_test_library/.secret_key')

with mock.patch('requests.post'), mock.patch('requests.get'):
    import pipeline  # noqa: F401 -- imported once here so every test module can just `import pipeline`

import pytest


@pytest.fixture
def users_db(tmp_path, monkeypatch):
    """A throwaway users.db for one test, so account/TOTP/lockout tests can
    freely create and modify accounts with no risk of touching a real
    deployment's data and no cross-test interference. Points auth.py's
    module-level USERS_DB_PATH at a temp file, then runs the real schema
    init against it -- exercising the actual init path (including its
    bootstrap-admin-if-empty behavior) rather than hand-building a schema
    that could drift from the real one."""
    with mock.patch('requests.post'), mock.patch('requests.get'):
        import auth
    db_path = str(tmp_path / 'test_users.db')
    monkeypatch.setattr(auth, 'USERS_DB_PATH', db_path)
    monkeypatch.setenv('ADMIN_USERNAME', 'admin')
    monkeypatch.setenv('ADMIN_PASSWORD', 'TestBootstrapPass123')
    auth.users_db_init()
    yield db_path


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


@pytest.fixture(autouse=True)
def _assume_ai_services_reachable(monkeypatch):
    """_run_trailer_job now refuses to start a job at all when AI Vision
    (Ollama) or faster-whisper is unreachable (see the module docstring
    above for why this sandbox never has either actually running, and
    test_service_preflight_check.py for the dedicated coverage of that
    refusal itself). Every OTHER test that exercises generation end-to-end
    predates that gate and depends on the *previous* behavior -- a request
    to an unreachable service failing individually and the pipeline
    degrading gracefully (a neutral AI score, no dialogue-aware cutting) --
    to actually reach the scene-selection/rendering logic it's testing.
    Autouse, so every test gets the preflight check itself stubbed out
    (reporting every service 'up') without needing to remember to opt in
    file by file; the *actual* per-call requests.post/get still go out
    unmocked (or mocked per-test) exactly as before, so the graceful
    degradation this suite already relies on is unaffected."""
    monkeypatch.setattr(pipeline, '_check_service',
                        lambda name, url, path='/', timeout=3: {'name': name, 'url': url, 'status': 'up'})


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


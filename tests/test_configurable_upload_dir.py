"""
Tests for UPLOAD_TEMP_DIR -- lets a deployment redirect all working/temp
storage (uploaded/staged source files, every intermediate processing file
for the duration of a render) to a chosen drive/folder instead of wherever
the OS default temp location happens to be. On Windows that default is
normally on the system drive regardless of where the app itself is
installed, and this folder alone can genuinely fill a system drive over
time -- user-reported low disk space was the reason this exists.

These import core.py fresh in a subprocess per test (rather than reloading
the already-imported module in-process) since app.config['UPLOAD_FOLDER']
is set once, at module import time, as a side effect of importing core --
reimporting cleanly per test is more reliable than trying to undo that
side effect through importlib.reload.
"""
import subprocess
import sys
import tempfile
import textwrap


def _run_core_import(env_extra):
    """Runs a small script in a fresh subprocess that imports core.py with
    the given extra environment variables, and prints app.config['UPLOAD_FOLDER'].
    Returns (stdout, stderr)."""
    script = textwrap.dedent('''
        import sys
        sys.path.insert(0, {repo_dir!r})
        import core
        print("UPLOAD_FOLDER=" + core.app.config['UPLOAD_FOLDER'])
    ''').format(repo_dir=__file__.rsplit('/tests/', 1)[0])
    import os
    env = dict(os.environ)
    env.pop('UPLOAD_TEMP_DIR', None)
    env.update(env_extra)
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True,
                            env=env, timeout=30)
    return result.stdout, result.stderr


def test_no_setting_uses_the_system_temp_location_as_before():
    stdout, stderr = _run_core_import({})
    line = [l for l in stdout.splitlines() if l.startswith('UPLOAD_FOLDER=')][0]
    folder = line.split('=', 1)[1]
    assert folder.startswith(tempfile.gettempdir())


def test_valid_custom_dir_is_used(tmp_path):
    chosen = tmp_path / 'my_custom_temp'
    stdout, stderr = _run_core_import({'UPLOAD_TEMP_DIR': str(chosen)})
    line = [l for l in stdout.splitlines() if l.startswith('UPLOAD_FOLDER=')][0]
    folder = line.split('=', 1)[1]
    assert folder.startswith(str(chosen))


def test_custom_dir_is_created_if_it_does_not_exist_yet(tmp_path):
    chosen = tmp_path / 'does_not_exist_yet' / 'nested'
    assert not chosen.exists()
    stdout, stderr = _run_core_import({'UPLOAD_TEMP_DIR': str(chosen)})
    assert chosen.exists()


def test_inaccessible_dir_falls_back_to_system_temp_with_a_warning():
    # /proc is a real, always-present read-only-for-writes location on any
    # Linux test runner -- a reliable way to force the failure path without
    # depending on filesystem permissions that vary by environment.
    stdout, stderr = _run_core_import({'UPLOAD_TEMP_DIR': '/proc/cannot_write_here'})
    line = [l for l in stdout.splitlines() if l.startswith('UPLOAD_FOLDER=')][0]
    folder = line.split('=', 1)[1]
    assert folder.startswith(tempfile.gettempdir())
    assert 'could not be' in stdout
    assert 'Falling back' in stdout


def test_each_process_still_gets_its_own_fresh_subfolder(tmp_path):
    # The existing "clean slate every process start" behavior (a fresh,
    # uniquely-named folder, not the chosen folder used directly) must be
    # preserved -- two separate imports should get two different subfolders
    # under the same chosen parent.
    chosen = tmp_path / 'shared_parent'
    stdout1, _ = _run_core_import({'UPLOAD_TEMP_DIR': str(chosen)})
    stdout2, _ = _run_core_import({'UPLOAD_TEMP_DIR': str(chosen)})
    folder1 = [l for l in stdout1.splitlines() if l.startswith('UPLOAD_FOLDER=')][0].split('=', 1)[1]
    folder2 = [l for l in stdout2.splitlines() if l.startswith('UPLOAD_FOLDER=')][0].split('=', 1)[1]
    assert folder1 != folder2
    assert folder1.startswith(str(chosen)) and folder2.startswith(str(chosen))

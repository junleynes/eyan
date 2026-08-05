"""Persistent trailer library (SQLite): saved trailers survive restarts.

Split out of the original monolith -- self-contained aside from `app`
(for the upload folder path) and stdlib.
"""
import os, sqlite3, uuid, time, json, shutil
from core import app

# ---- Persistent trailer library (SQLite) ----
# UPLOAD_FOLDER above is a fresh tempdir every process start, so anything that
# needs to survive a restart -- completed trailers the user wants to come back
# to later -- gets copied here instead, with metadata tracked in a small
# SQLite DB alongside it. Override LIBRARY_DIR to point this at persistent
# storage (a mounted volume, etc.) in production.
LIBRARY_DIR = os.environ.get('LIBRARY_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trailer_library'))
os.makedirs(LIBRARY_DIR, exist_ok=True)
LIBRARY_DB_PATH = os.path.join(LIBRARY_DIR, 'library.db')

def _sqlite_connect(db_path):
    """Opens a SQLite connection tuned for this app's access pattern: several
    concurrent job threads writing while the monitor endpoint polls for reads.

    The stock settings give a 5-second busy timeout and rollback-journal locking,
    under which a reader blocks writers and a slow write surfaces as
    'database is locked'. WAL lets readers and one writer proceed concurrently,
    and the longer timeout absorbs the rest."""
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute('PRAGMA synchronous=NORMAL')
        conn.execute('PRAGMA busy_timeout=30000')
    except sqlite3.Error as e:
        # A network/SMB-mounted DB path can refuse WAL; rollback journal still works.
        print(f'SQLite pragma setup failed for {db_path} (continuing): {e}')
    return conn

def _lib_db():
    return _sqlite_connect(LIBRARY_DB_PATH)

def library_db_init():
    conn = _lib_db()
    conn.execute('''CREATE TABLE IF NOT EXISTS trailers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        orig_name TEXT,
        filename TEXT NOT NULL,
        created_at REAL NOT NULL,
        trailer_duration REAL,
        video_duration REAL,
        trailer_length TEXT,
        bgm_source TEXT,
        sfx_source TEXT,
        vo_source TEXT,
        result_json TEXT
    )''')
    # Added after the table already existed in the wild -- ALTER TABLE guard
    # rather than baking these into the CREATE above, same pattern used for
    # show_templates' settings_json column. Existing rows get NULL for both,
    # which library_list()/ownership checks treat as admin-only (nothing to
    # honestly attribute them to -- see _owns_or_admin in pipeline.py).
    have = {r[1] for r in conn.execute('PRAGMA table_info(trailers)')}
    if 'user_id' not in have:
        conn.execute('ALTER TABLE trailers ADD COLUMN user_id INTEGER')
    if 'username' not in have:
        conn.execute('ALTER TABLE trailers ADD COLUMN username TEXT')
    conn.commit()
    conn.close()

def library_add(upload_filename, result, user_id=None, username=None):
    """Copies the just-finished trailer (currently sitting in the ephemeral
    UPLOAD_FOLDER as `upload_filename`) into LIBRARY_DIR under a permanent
    name, records it in SQLite, and returns the new row id. The saved
    result_json has its trailer_url rewritten to the persistent /library/
    route so re-opening it later (even after a restart) works regardless of
    whether the original UPLOAD_FOLDER file still exists."""
    ext = os.path.splitext(upload_filename)[1] or '.mp4'
    persist_name = f'{uuid.uuid4().hex}{ext}'
    src = os.path.join(app.config['UPLOAD_FOLDER'], upload_filename)
    dst = os.path.join(LIBRARY_DIR, persist_name)
    shutil.copy2(src, dst)
    conn = _lib_db()
    cur = conn.execute(
        'INSERT INTO trailers (orig_name, filename, created_at, trailer_duration, video_duration, trailer_length, bgm_source, sfx_source, vo_source, result_json, user_id, username) '
        'VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
        (result.get('orig_name'), persist_name, time.time(), result.get('trailer_duration'), result.get('video_duration'),
         result.get('trailer_length'), result.get('bgm_source'), result.get('sfx_source'), result.get('vo_source'), None,
         user_id, username))
    tid = cur.lastrowid
    saved_result = dict(result, trailer_url=f'/library/{tid}/file', library_id=tid)
    conn.execute('UPDATE trailers SET result_json=? WHERE id=?', (json.dumps(saved_result), tid))
    conn.commit()
    conn.close()
    return tid

def library_list(limit=50, user_id=None, is_admin=False):
    """Most recent saved trailers. An admin sees everything; anyone else sees
    only what they own (user_id must match) -- rows with no owner at all
    (user_id IS NULL, from before this app had accounts, or a save that
    somehow didn't capture one) are admin-only, same reasoning as
    _owns_or_admin in pipeline.py."""
    conn = _lib_db()
    if is_admin:
        rows = conn.execute(
            'SELECT id, orig_name, filename, created_at, trailer_duration, video_duration, trailer_length, bgm_source, sfx_source, vo_source, user_id, username '
            'FROM trailers ORDER BY created_at DESC LIMIT ?', (limit,)).fetchall()
    else:
        rows = conn.execute(
            'SELECT id, orig_name, filename, created_at, trailer_duration, video_duration, trailer_length, bgm_source, sfx_source, vo_source, user_id, username '
            'FROM trailers WHERE user_id=? ORDER BY created_at DESC LIMIT ?', (user_id, limit)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def library_get_row(tid):
    conn = _lib_db()
    row = conn.execute('SELECT * FROM trailers WHERE id=?', (tid,)).fetchone()
    conn.close()
    return dict(row) if row else None

def library_delete(tid):
    row = library_get_row(tid)
    if not row:
        return False
    path = os.path.join(LIBRARY_DIR, row['filename'])
    if os.path.exists(path):
        os.remove(path)
    # Also drop any cached format-converted copies of it (see library_download()).
    base = os.path.splitext(row['filename'])[0]
    for f in os.listdir(LIBRARY_DIR):
        if f.startswith(base + '_'):
            try:
                os.remove(os.path.join(LIBRARY_DIR, f))
            except OSError:
                pass
    conn = _lib_db()
    conn.execute('DELETE FROM trailers WHERE id=?', (tid,))
    conn.commit()
    conn.close()
    return True

library_db_init()


"""Persistent trailer library (SQLite): saved trailers survive restarts.

Split out of the original monolith -- self-contained aside from `app`
(for the upload folder path) and stdlib.
"""
import os, sqlite3, uuid, time, json, shutil, re
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

# ---- Branding: app name, tagline, accent color, logo, favicon, editable from
# Config > Branding ----
# Lives here (not in pipeline.py, where it started) rather than off in its own
# module because it needs to be importable by BOTH pipeline.py (the Config >
# Branding tab's routes) and auth.py (the login page, which now shows the
# same name/tagline/logo/accent pre-auth) -- library_db.py is a dependency
# both of those already sit on top of, so this is the one shared spot that
# doesn't introduce a circular import. Same persistence shape as the AI-service
# config in pipeline.py (a small JSON file next to the script), but kept
# separate since this one also manages uploaded image files, not just text.
# Uploaded logo/favicon files live under LIBRARY_DIR (survives restarts the
# same way the trailer library does) rather than next to the script, so they
# don't need separate backup/deploy handling from everything else this app
# persists.
BRANDING_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'branding_config.json')
BRANDING_DIR = os.path.join(LIBRARY_DIR, 'branding')
os.makedirs(BRANDING_DIR, exist_ok=True)
DEFAULT_BRAND_NAME = 'AIMP'
DEFAULT_BRAND_TAGLINE = 'AI Media Provider'
# Matches the dark-theme --accent default baked into templates/index.html, so
# an unconfigured install renders identically to before this existed.
DEFAULT_BRAND_ACCENT = '#4f8cff'
_BRANDING_IMAGE_EXTS = {'.svg', '.png', '.jpg', '.jpeg', '.webp', '.gif'}
_BRANDING_FAVICON_EXTS = {'.ico', '.svg', '.png'}
_HEX_COLOR_RE = re.compile(r'^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$')

def load_branding():
    """Current brand name/tagline/accent color/logo/favicon. Falls back to the
    built-in AIMP defaults for anything never configured -- this always
    returns a complete dict, never partial, so callers don't each need their
    own fallback logic."""
    cfg = {'name': DEFAULT_BRAND_NAME, 'tagline': DEFAULT_BRAND_TAGLINE,
           'accent_color': DEFAULT_BRAND_ACCENT, 'logo_filename': None, 'favicon_filename': None}
    if os.path.exists(BRANDING_FILE):
        try:
            with open(BRANDING_FILE) as f:
                saved = json.load(f)
            for k in cfg:
                if saved.get(k):
                    cfg[k] = saved[k]
        except Exception as e:
            print(f'Branding config load error ({BRANDING_FILE}): {e}')
    return cfg

def _write_branding(cfg):
    with open(BRANDING_FILE, 'w') as f:
        json.dump(cfg, f, indent=2)

def save_branding_text(name=None, tagline=None):
    cfg = load_branding()
    if name is not None:
        cfg['name'] = name.strip() or DEFAULT_BRAND_NAME
    if tagline is not None:
        cfg['tagline'] = tagline.strip() or DEFAULT_BRAND_TAGLINE
    _write_branding(cfg)
    return cfg

def save_branding_color(accent_color):
    """Sets the accent color used app-wide (buttons, links, highlights, the
    login screen) in both the light and dark themes -- overrides
    templates/index.html's default --accent CSS variable for this
    install. Rejects anything that isn't a real #rgb/#rrggbb hex value rather
    than silently falling back, so a typo in the Config tab surfaces as an
    error there instead of quietly reverting to the default."""
    color = (accent_color or '').strip()
    if not _HEX_COLOR_RE.match(color):
        return None, f'"{color}" isn\'t a valid hex color -- use a format like #4f8cff.'
    cfg = load_branding()
    cfg['accent_color'] = color
    _write_branding(cfg)
    return cfg, None

def clear_branding_color():
    cfg = load_branding()
    cfg['accent_color'] = DEFAULT_BRAND_ACCENT
    _write_branding(cfg)
    return cfg

def _replace_branding_file(cfg, key, file_storage, allowed_exts, prefix):
    """Shared by save_branding_logo/favicon below: validates the new upload's
    extension BEFORE touching anything, then removes whatever custom file was
    previously set for `key` and saves the new one under a fixed name (so
    there's never more than one live file per slot to track). Validating
    first matters: an upload with a bad extension must leave the existing
    custom logo/favicon untouched, not delete it and then fail -- silently
    reverting to the default because of a rejected upload would be a much
    more confusing failure mode than the plain error message this returns.
    Returns (cfg, error)."""
    ext = os.path.splitext(file_storage.filename or '')[1].lower()
    if ext not in allowed_exts:
        return None, f'Unsupported file type "{ext or "(none)"}" -- use one of: {", ".join(sorted(allowed_exts))}'
    old = cfg.get(key)
    if old:
        old_path = os.path.join(BRANDING_DIR, old)
        if os.path.exists(old_path):
            try:
                os.remove(old_path)
            except OSError:
                pass
    fname = f'{prefix}{ext}'
    file_storage.save(os.path.join(BRANDING_DIR, fname))
    cfg[key] = fname
    _write_branding(cfg)
    return cfg, None

def save_branding_logo(file_storage):
    return _replace_branding_file(load_branding(), 'logo_filename', file_storage, _BRANDING_IMAGE_EXTS, 'logo')

def save_branding_favicon(file_storage):
    return _replace_branding_file(load_branding(), 'favicon_filename', file_storage, _BRANDING_FAVICON_EXTS, 'favicon')

def _clear_branding_file(key):
    cfg = load_branding()
    old = cfg.get(key)
    if old:
        old_path = os.path.join(BRANDING_DIR, old)
        if os.path.exists(old_path):
            try:
                os.remove(old_path)
            except OSError:
                pass
    cfg[key] = None
    _write_branding(cfg)
    return cfg

def clear_branding_logo():
    return _clear_branding_file('logo_filename')

def clear_branding_favicon():
    return _clear_branding_file('favicon_filename')

library_db_init()


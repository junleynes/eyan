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
    conn.execute('''CREATE TABLE IF NOT EXISTS audit_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at REAL NOT NULL,
        user_id INTEGER,
        username TEXT,
        action TEXT NOT NULL,
        target TEXT,
        detail TEXT,
        ip TEXT
    )''')
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

# ---- Audit log ----
# Deliberately a short, fixed set of columns rather than a free-form blob --
# action/target/detail are plain strings so the log stays queryable and
# readable without needing to parse anything, and so a future column can be
# added the same ALTER TABLE way the trailers table already handles it.
def audit_log(action, target=None, detail=None, user_id=None, username=None, ip=None):
    """Records one audit entry. user_id/username/ip are accepted as explicit
    params rather than this function reaching into `session`/`request`
    itself -- library_db.py stays free of any Flask dependency (it's
    imported by things that aren't always inside a request), and the
    handful of call sites that DO have session/request access can supply
    them directly."""
    conn = _lib_db()
    conn.execute('INSERT INTO audit_log (created_at, user_id, username, action, target, detail, ip) '
                 'VALUES (?,?,?,?,?,?,?)',
                 (time.time(), user_id, username, action, target, detail, ip))
    conn.commit()
    conn.close()

def audit_log_list(limit=200):
    """Most recent audit entries, newest first. Admin-only at the route
    level (this function itself doesn't check permissions -- same pattern
    as library_stats(), which the dashboard's Library card already uses the
    same way)."""
    conn = _lib_db()
    rows = conn.execute('SELECT * FROM audit_log ORDER BY created_at DESC LIMIT ?', (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

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

def library_stats():
    """Total saved-trailer count and their combined size on disk, for the
    dashboard's admin-only Library card. Deliberately a separate, lighter
    query rather than reusing library_list() -- that one caps at 50 rows by
    default and pulls every column including result_json (the full per-scene
    breakdown, potentially large), neither of which this needs. All this
    needs is every filename, to total their real size with os.path.getsize
    rather than trusting a stored size that could drift from the actual file."""
    conn = _lib_db()
    rows = conn.execute('SELECT filename FROM trailers').fetchall()
    conn.close()
    total_bytes = 0
    missing = 0
    for r in rows:
        p = os.path.join(LIBRARY_DIR, r['filename'])
        try:
            total_bytes += os.path.getsize(p)
        except OSError:
            missing += 1  # DB row survives its file being deleted out from under it -- don't crash the count over it
    return {'count': len(rows), 'total_bytes': total_bytes, 'missing_files': missing}

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
DEFAULT_BRAND_NAME = 'PRISM'
DEFAULT_BRAND_TAGLINE = 'AI-Powered Media Creation'
# Matches the dark-theme --accent default baked into templates/index.html, so
# an unconfigured install renders identically to before this existed.
DEFAULT_BRAND_ACCENT = '#4f8cff'

# A single accent color barely changes how the app looks: --phosphor and
# --amber (the active-tab highlight, most buttons, badges, status dots) are
# used far more throughout the stylesheet than --accent is, and swapping only
# --accent left the app looking almost the same as before. A real "color
# theme" needs to be a coordinated set, not one hex value -- these five keys
# are the same semantic roles templates/index.html's own :root already
# defines (see the CSS comments there for what each one means), just made
# swappable as a matched set rather than piecemeal.
#
# Curated combinations rather than generating a palette algorithmically from
# one picked color: an arbitrary hue run through a generator can produce
# genuinely clashing or illegible combinations (a phosphor and tally that are
# too close in value, an amber that reads as brown), where a hand-picked set
# is guaranteed to look intentional. 'custom' is the escape hatch for anyone
# who wants to pick their own accent instead of a preset -- it keeps the
# original single-accent behavior (default phosphor/tally/amber, custom
# accent only) rather than removing that option.
THEME_PRESETS = {
    'teal_amber': {'label': 'Teal & Amber',
                   'phosphor': '#34e6c5', 'phosphor_dim': '#1d8f7c',
                   'tally': '#ff5470', 'amber': '#ffb545', 'accent': '#4f8cff',
                   # Unchanged from the app's original neutral navy/white --
                   # the default theme should render identically to before
                   # this feature existed.
                   'bg_dark': '#0b1220', 'panel_dark': '#121a2b', 'panel_2_dark': '#1a2233',
                   'elevated_dark': '#1a2436', 'sunken_dark': '#03070d',
                   'bg_light': '#f3f5f9', 'panel_light': '#ffffff', 'panel_2_light': '#eef1f6',
                   'elevated_light': '#ffffff', 'sunken_light': '#e4e8f0', 'rail_bg_dark': 'rgba(13,19,32,.72)', 'rail_bg_light': 'rgba(255,255,255,.75)'},
    'ocean': {'label': 'Ocean Blue',
             'phosphor': '#38bdf8', 'phosphor_dim': '#0284c7',
             'tally': '#f43f5e', 'amber': '#fbbf24', 'accent': '#6366f1',
             'bg_dark': '#0d1b2b', 'panel_dark': '#142235', 'panel_2_dark': '#1c2a3d',
             'elevated_dark': '#1c2c40', 'sunken_dark': '#061019',
             'bg_light': '#eaf2f9', 'panel_light': '#fbfeff', 'panel_2_light': '#e5eef6',
             'elevated_light': '#fbfeff', 'sunken_light': '#dbe6f0', 'rail_bg_dark': 'rgba(13,27,43,.72)', 'rail_bg_light': 'rgba(234,242,249,.75)'},
    'sunset': {'label': 'Sunset',
              'phosphor': '#fb923c', 'phosphor_dim': '#c2410c',
              'tally': '#ec4899', 'amber': '#fbbf24', 'accent': '#f97316',
              'bg_dark': '#171821', 'panel_dark': '#1e202c', 'panel_2_dark': '#252833',
              'elevated_dark': '#252a36', 'sunken_dark': '#0f0e0f',
              'bg_light': '#f3f0f0', 'panel_light': '#fffdfb', 'panel_2_light': '#efeced',
              'elevated_light': '#fffdfb', 'sunken_light': '#e5e4e7', 'rail_bg_dark': 'rgba(23,24,33,.72)', 'rail_bg_light': 'rgba(243,240,240,.75)'},
    'forest': {'label': 'Forest',
              'phosphor': '#4ade80', 'phosphor_dim': '#16a34a',
              'tally': '#ef4444', 'amber': '#eab308', 'accent': '#22c55e',
              'bg_dark': '#0e1c25', 'panel_dark': '#15242f', 'panel_2_dark': '#1c2b37',
              'elevated_dark': '#1c2d3a', 'sunken_dark': '#071213',
              'bg_light': '#ebf4f3', 'panel_light': '#fbfefc', 'panel_2_light': '#e6f0f0',
              'elevated_light': '#fbfefc', 'sunken_light': '#dce8ea', 'rail_bg_dark': 'rgba(14,28,37,.72)', 'rail_bg_light': 'rgba(235,244,243,.75)'},
    'crimson': {'label': 'Crimson (default)',
               'phosphor': '#f87171', 'phosphor_dim': '#b91c1c',
               'tally': '#fb7185', 'amber': '#fbbf24', 'accent': '#ef4444',
               'bg_dark': '#171724', 'panel_dark': '#1e1e2e', 'panel_2_dark': '#252636',
               'elevated_dark': '#252839', 'sunken_dark': '#0f0c12',
               'bg_light': '#f3eef2', 'panel_light': '#fffcfc', 'panel_2_light': '#eeebef',
               'elevated_light': '#fffcfc', 'sunken_light': '#e5e2ea', 'rail_bg_dark': 'rgba(23,23,36,.72)', 'rail_bg_light': 'rgba(243,238,242,.75)'},
    'violet': {'label': 'Violet',
              'phosphor': '#a78bfa', 'phosphor_dim': '#7c3aed',
              'tally': '#f472b6', 'amber': '#fbbf24', 'accent': '#8b5cf6',
              'bg_dark': '#13182b', 'panel_dark': '#192035', 'panel_2_dark': '#21273d',
              'elevated_dark': '#212940', 'sunken_dark': '#0b0e19',
              'bg_light': '#eff0f9', 'panel_light': '#fdfdff', 'panel_2_light': '#eaecf6',
              'elevated_light': '#fdfdff', 'sunken_light': '#e1e3f0', 'rail_bg_dark': 'rgba(19,24,43,.72)', 'rail_bg_light': 'rgba(239,240,249,.75)'},
}
DEFAULT_THEME_NAME = 'crimson'
_BRANDING_IMAGE_EXTS = {'.svg', '.png', '.jpg', '.jpeg', '.webp', '.gif'}
_BRANDING_FAVICON_EXTS = {'.ico', '.svg', '.png'}
_HEX_COLOR_RE = re.compile(r'^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$')

def resolve_theme_colors(theme_name, custom_accent):
    """The five hex values actually applied for the given theme choice.
    'custom' (or any unrecognized name -- e.g. a preset removed in a future
    version) falls back to the default preset's phosphor/tally/amber with
    only accent swapped, preserving the original single-accent-color
    behavior for anyone using it that way."""
    preset = THEME_PRESETS.get(theme_name)
    if preset:
        return {k: v for k, v in preset.items() if k != 'label'}
    base = dict(THEME_PRESETS[DEFAULT_THEME_NAME])
    base.pop('label', None)
    base['accent'] = custom_accent or DEFAULT_BRAND_ACCENT
    return base

def load_branding():
    """Current brand name/tagline/footer/theme/logo/favicon. Falls back to
    the built-in PRISM defaults for anything never configured -- footer's
    default is an empty string (no footer shown at all) rather than built-in
    text, since unlike name/tagline there's no sensible non-empty default to
    fall back to. This always returns a complete dict, never partial, so
    callers don't each need their own fallback logic. Includes a resolved
    'theme_colors' dict (see resolve_theme_colors) so callers never need to
    re-derive it themselves."""
    cfg = {'name': DEFAULT_BRAND_NAME, 'tagline': DEFAULT_BRAND_TAGLINE, 'footer': '',
           'accent_color': DEFAULT_BRAND_ACCENT, 'theme_name': DEFAULT_THEME_NAME,
           'logo_filename': None, 'favicon_filename': None}
    if os.path.exists(BRANDING_FILE):
        try:
            with open(BRANDING_FILE) as f:
                saved = json.load(f)
            for k in cfg:
                # 'footer' deliberately allows an explicit '' to be read back
                # (clearing it is a real, valid state, not "unset" the way a
                # blank name/tagline falling back to the default is) -- every
                # other key here keeps the original any-truthy-value check.
                if k == 'footer':
                    if 'footer' in saved:
                        cfg['footer'] = saved['footer']
                elif saved.get(k):
                    cfg[k] = saved[k]
        except Exception as e:
            print(f'Branding config load error ({BRANDING_FILE}): {e}')
    cfg['theme_colors'] = resolve_theme_colors(cfg['theme_name'], cfg['accent_color'])
    return cfg

def _write_branding(cfg):
    # theme_colors is derived (see resolve_theme_colors), not stored -- keeps
    # the presets themselves editable in code without stale resolved values
    # lingering in old branding_config.json files.
    to_write = {k: v for k, v in cfg.items() if k != 'theme_colors'}
    with open(BRANDING_FILE, 'w') as f:
        json.dump(to_write, f, indent=2)

def save_branding_text(name=None, tagline=None, footer=None):
    cfg = load_branding()
    if name is not None:
        cfg['name'] = name.strip() or DEFAULT_BRAND_NAME
    if tagline is not None:
        cfg['tagline'] = tagline.strip() or DEFAULT_BRAND_TAGLINE
    if footer is not None:
        # No fallback-to-default here on purpose -- an empty footer is a
        # legitimate, common choice (most installs won't want one at all),
        # not a mistake to silently correct the way a blank name would be.
        cfg['footer'] = footer.strip()
    _write_branding(cfg)
    return cfg

def save_branding_theme(theme_name):
    """Sets the coordinated color theme (see THEME_PRESETS) for this
    install. 'custom' is valid and just means "use accent_color with the
    default preset's other colors" -- see resolve_theme_colors."""
    theme_name = (theme_name or '').strip()
    if theme_name != 'custom' and theme_name not in THEME_PRESETS:
        return None, f'Unknown theme "{theme_name}".'
    cfg = load_branding()
    cfg['theme_name'] = theme_name
    _write_branding(cfg)
    return cfg, None

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

# ---- Manual per-service disable (admin override, independent of live health) ----
# The health check (/api/health) already detects a service being unreachable
# and the UI greys things out accordingly -- this is for the other case: an
# admin who KNOWS a service is flaky, being maintained, or just not meant to
# be offered right now, and wants it treated as unavailable everywhere a live
# health check would treat it that way, without waiting for it to actually
# fail a probe (or continuing to offer it during a maintenance window where it
# might flicker back "up" between checks).
SERVICE_OVERRIDES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'service_overrides.json')
# Same five names /api/health already checks -- kept as the single source of
# truth for "which services exist" rather than letting this file invent its
# own list that could drift out of sync with it.
KNOWN_SERVICES = {'ollama', 'fish_audio', 'whisper', 'ace_step', 'woosh'}

def load_disabled_services():
    """Set of service names an admin has manually disabled. Never raises --
    a missing or corrupt file just means nothing is disabled, same as a
    fresh install."""
    if not os.path.exists(SERVICE_OVERRIDES_FILE):
        return set()
    try:
        with open(SERVICE_OVERRIDES_FILE) as f:
            data = json.load(f)
        return {s for s in data.get('disabled', []) if s in KNOWN_SERVICES}
    except Exception as e:
        print(f'Service overrides load error ({SERVICE_OVERRIDES_FILE}): {e}')
        return set()

def set_service_disabled(name, disabled):
    if name not in KNOWN_SERVICES:
        return False, f'Unknown service "{name}".'
    current = load_disabled_services()
    if disabled:
        current.add(name)
    else:
        current.discard(name)
    with open(SERVICE_OVERRIDES_FILE, 'w') as f:
        json.dump({'disabled': sorted(current)}, f, indent=2)
    return True, None

# ---- Per-folder network paths ----
# HIRES, title/end cards, music, VO, and SFX often live on genuinely
# different network volumes with different credentials in a real broadcast
# setup. Rather than a "default share + per-category override" system
# (an earlier version of this), each folder is independently and directly
# configured with its own full UNC path and its own credentials -- no
# shared default, no inherit-unless-overridden logic. Simpler to reason
# about and matches how these are actually assigned in practice: someone
# filling this in already knows the exact path and login for each folder,
# not a "base share" they're subdividing.
NETWORK_FOLDERS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'network_folders.json')
NETWORK_CATEGORY_KEYS = ['hires', 'tcard', 'endcard', 'music', 'vo', 'sfx']
_NETWORK_FOLDER_FIELDS = ('path', 'username', 'password')

def load_network_folders():
    """{category: {path, username, password}} for every configured folder.
    A category absent from this dict simply hasn't been set up yet --
    callers should treat that the same as one present with every field
    blank (not reachable until configured), there is no fallback source to
    pull from."""
    if not os.path.exists(NETWORK_FOLDERS_FILE):
        return {}
    try:
        with open(NETWORK_FOLDERS_FILE) as f:
            data = json.load(f)
        return {cat: {k: v.get(k, '') for k in _NETWORK_FOLDER_FIELDS}
                for cat, v in data.items() if cat in NETWORK_CATEGORY_KEYS and isinstance(v, dict)}
    except Exception as e:
        print(f'Network folder config load error ({NETWORK_FOLDERS_FILE}): {e}')
        return {}

def save_network_folder(category, fields):
    """Sets category's path/username/password. A field not present in
    `fields` is left as its current saved value; an explicitly empty string
    DOES clear that one field -- same is-present-vs-is-empty distinction
    load_branding uses for footer, for the same reason: clearing a field on
    purpose (e.g. removing a password without touching the path) is a real,
    valid state here too."""
    if category not in NETWORK_CATEGORY_KEYS:
        return False, f'Unknown category "{category}".'
    current = load_network_folders()
    row = current.get(category, {k: '' for k in _NETWORK_FOLDER_FIELDS})
    for k in _NETWORK_FOLDER_FIELDS:
        if k in fields:
            row[k] = fields[k] or ''
    current[category] = row
    with open(NETWORK_FOLDERS_FILE, 'w') as f:
        json.dump(current, f, indent=2)
    return True, None

library_db_init()


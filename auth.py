"""User accounts, login/logout, and the /admin/users management page.

Depends on: core (app, rate limiter, dummy-hash, default password) and
library_db (LIBRARY_DIR, _sqlite_connect) for where users.db lives.
"""
import os, time, sqlite3, functools
from flask import request, session, redirect, jsonify
from markupsafe import escape
from werkzeug.security import generate_password_hash, check_password_hash
from core import app, _client_ip, _login_limiter, _DUMMY_PW_HASH, DEFAULT_ADMIN_PASSWORD
from library_db import LIBRARY_DIR, _sqlite_connect, load_branding

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        if not _login_limiter.allow(_client_ip()):
            error = 'Too many attempts. Wait a few minutes and try again.'
        else:
            username = request.form.get('username', '').strip()
            password = request.form.get('password', '')
            user = user_get_by_username(username)
            # Run check_password_hash even on a miss (against a dummy hash) so
            # a nonexistent username doesn't respond measurably faster than a
            # wrong password -- that timing gap would otherwise leak which
            # usernames exist.
            pw_ok = check_password_hash(user['password_hash'] if user else _DUMMY_PW_HASH, password)
            if user and pw_ok and user['is_active']:
                session.permanent = True
                session['authed'] = True
                session['user_id'] = user['id']
                session['username'] = user['username']
                session['role'] = user['role']
                user_touch_login(user['id'])
                dest = request.form.get('next') or '/'
                # Only ever redirect to a path on this app -- an absolute or
                # protocol-relative 'next' would be an open-redirect vector.
                if not dest.startswith('/') or dest.startswith('//'):
                    dest = '/'
                return redirect(dest)
            elif user and not user['is_active']:
                error = 'This account has been disabled.'
            else:
                error = 'Incorrect username or password.'
    nxt = request.args.get('next', '/')
    # 'next' and any error text land inside HTML attribute/element content
    # below -- escape both. 'next' is attacker-controlled (it's a query
    # param), so leaving it raw would be a reflected-XSS hole via a crafted
    # login link; error is currently always one of this function's own fixed
    # strings, but escaping costs nothing and keeps that true by construction
    # rather than by convention.
    nxt = escape(nxt)
    error = escape(error) if error else None
    brand = load_branding()
    brand_name = escape(brand['name'])
    brand_tagline = escape(brand['tagline'])
    accent = brand['accent_color']
    # /branding/logo always resolves to something displayable (falls back to
    # the built-in mark itself when no custom logo is configured -- see
    # branding_logo() in pipeline.py), so this <img> never needs an
    # onerror/placeholder fallback of its own.
    return f'''<!doctype html><html><head><meta charset="utf-8">
<title>{brand_name} &mdash; {brand_tagline}</title>
<link rel="icon" href="/branding/favicon">
<style>:root{{--accent:{accent}}}
body{{font-family:system-ui,sans-serif;background:#0b1220;color:#e6e9ef;
display:flex;align-items:center;justify-content:center;height:100vh;margin:0}}
.login-card{{display:flex;flex-direction:column;align-items:center;width:280px}}
.login-logo{{width:52px;height:52px;border-radius:12px;margin-bottom:14px;object-fit:contain}}
.login-name{{font-size:20px;font-weight:600;margin:0 0 2px}}
.login-tagline{{font-size:13px;color:#8b98ad;margin:0 0 22px;text-align:center}}
form{{background:#141b2d;padding:28px 32px;border-radius:10px;border:1px solid #232b41;width:100%;
box-sizing:border-box}}
h1{{font-size:16px;margin:0 0 16px}}
input{{width:100%;box-sizing:border-box;padding:9px 10px;border-radius:6px;border:1px solid #232b41;
background:#0b1220;color:#e6e9ef;font-size:14px;margin-bottom:12px}}
button{{width:100%;padding:9px;border-radius:6px;border:none;background:var(--accent);color:#fff;
font-size:14px;cursor:pointer}}
.err{{color:#e08a3c;font-size:12px;margin-bottom:12px}}</style></head><body>
<div class="login-card">
<img class="login-logo" src="/branding/logo" alt="{brand_name} logo">
<p class="login-name">{brand_name}</p>
<p class="login-tagline">{brand_tagline}</p>
<form method=post>
<h1>Sign in</h1>
{f'<div class="err">{error}</div>' if error else ''}
<input type=hidden name=next value="{nxt}">
<input type=text name=username placeholder="Username" autofocus autocomplete="username">
<input type=password name=password placeholder="Password" autocomplete="current-password">
<button type=submit>Continue</button>
</form>
</div></body></html>'''

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')


# ---- User accounts (SQLite) ----
# Simple username/password accounts with an admin/user role, replacing the
# old shared passphrase. Stored in its own DB file alongside the trailer
# library so it survives restarts the same way.
USERS_DB_PATH = os.path.join(LIBRARY_DIR, 'users.db')

def _users_db():
    return _sqlite_connect(USERS_DB_PATH)

# Every permission a group can be granted, one per gated tab. Deliberately
# does NOT include Docs (informational, no access implications), or
# Config/API/admin-users (those stay exclusively role='admin', unaffected by
# groups, same as before this feature existed -- a group granting every
# permission below still isn't the same thing as being an admin).
#
# 'text_to_speech' covers both the standalone Text to Speech tab AND the
# promo generator's narration section, which share the same underlying
# /api/vo/preview endpoint -- see require_permission's OR-semantics below.
# The Manage Voices sub-tab (registering/deleting shared voices) stays
# admin-only regardless of this permission, same reasoning as Config: it's a
# shared global resource, not a per-job action.
AVAILABLE_PERMISSIONS = [
    ('promo_generation', 'Generate Promo Plug'),
    ('music_generation', 'Music Generation'),
    ('text_to_sfx', 'Text to SFX'),
    ('text_to_speech', 'Text to Speech (voice cloning/narration)'),
    ('speech_to_text', 'Speech to Text'),
    ('scene_detection', 'Scene Detection'),
    ('ai_chat', 'AI Chat'),
    ('player', 'Player'),
]
_PERMISSION_KEYS = {k for k, _ in AVAILABLE_PERMISSIONS}

def users_db_init():
    conn = _users_db()
    conn.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL UNIQUE,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL DEFAULT 'user',
        is_active INTEGER NOT NULL DEFAULT 1,
        created_at REAL NOT NULL,
        last_login REAL
    )''')
    conn.execute('''CREATE TABLE IF NOT EXISTS groups (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE,
        created_at REAL NOT NULL
    )''')
    conn.execute('''CREATE TABLE IF NOT EXISTS group_permissions (
        group_id INTEGER NOT NULL,
        permission TEXT NOT NULL,
        PRIMARY KEY (group_id, permission)
    )''')
    # Added after `users` already existed in the wild -- ALTER TABLE guard,
    # same pattern as the earlier user_id/username columns on trailers.
    # NULL (the default for every existing account) means "no group
    # assigned" -- see user_can() below for why that means unrestricted
    # rather than locked out. Restricting an account is an opt-in action an
    # admin takes (create a group, check only the permissions it should
    # have, assign the account to it), not something that silently happens
    # to existing accounts when this feature is deployed.
    #
    # No REFERENCES/FOREIGN KEY clause here on purpose: this app never runs
    # `PRAGMA foreign_keys = ON` (see _sqlite_connect), so a FK constraint
    # here would be silently inert -- SQLite accepts the syntax but enforces
    # nothing without that pragma, including any ON DELETE behavior. Group
    # deletion's cleanup (clearing group_id off members, removing the
    # group's permission rows) is therefore done explicitly in
    # group_delete() below rather than relied on at the schema level.
    have = {r[1] for r in conn.execute('PRAGMA table_info(users)')}
    if 'group_id' not in have:
        conn.execute('ALTER TABLE users ADD COLUMN group_id INTEGER')
    conn.commit()
    count = conn.execute('SELECT COUNT(*) AS c FROM users').fetchone()['c']
    if count == 0:
        # Bootstrap: no accounts exist yet, so create a default admin rather
        # than locking the first deploy out entirely. Override with
        # ADMIN_USERNAME/ADMIN_PASSWORD env vars; otherwise this falls back to
        # the hardcoded DEFAULT_ADMIN_PASSWORD above.
        username = (os.environ.get('ADMIN_USERNAME', '').strip() or 'admin')
        password = os.environ.get('ADMIN_PASSWORD', '').strip() or DEFAULT_ADMIN_PASSWORD
        conn.execute(
            'INSERT INTO users (username, password_hash, role, is_active, created_at) VALUES (?,?,?,1,?)',
            (username, generate_password_hash(password), 'admin', time.time()))
        conn.commit()
        print('=' * 64)
        print(' * No user accounts found -- created a default admin account:')
        print(f'     username: {username}')
        print(f'     password: {password}')
        print('   Sign in, add real accounts from Admin > Users, and change')
        print('   this admin password (Admin > Users > Set) afterward --')
        print('   the default is hardcoded in this file, not a secret.')
        print('=' * 64)
    conn.close()

def user_get_by_username(username):
    if not username:
        return None
    conn = _users_db()
    row = conn.execute('SELECT * FROM users WHERE username=?', (username,)).fetchone()
    conn.close()
    return dict(row) if row else None

def user_get(uid):
    conn = _users_db()
    row = conn.execute('SELECT * FROM users WHERE id=?', (uid,)).fetchone()
    conn.close()
    return dict(row) if row else None

def user_list():
    conn = _users_db()
    rows = conn.execute(
        'SELECT u.id, u.username, u.role, u.is_active, u.created_at, u.last_login, u.group_id, g.name AS group_name '
        'FROM users u LEFT JOIN groups g ON g.id = u.group_id ORDER BY u.username COLLATE NOCASE'
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def user_count_active_admins(exclude_id=None):
    conn = _users_db()
    rows = conn.execute("SELECT id FROM users WHERE role='admin' AND is_active=1").fetchall()
    conn.close()
    return len([r for r in rows if r['id'] != exclude_id])

def user_create(username, password, role='user'):
    username = (username or '').strip()
    if not username:
        return False, 'Username is required.'
    if len(password or '') < 6:
        return False, 'Password must be at least 6 characters.'
    if role not in ('admin', 'user'):
        role = 'user'
    conn = _users_db()
    try:
        conn.execute(
            'INSERT INTO users (username, password_hash, role, is_active, created_at) VALUES (?,?,?,1,?)',
            (username, generate_password_hash(password), role, time.time()))
        conn.commit()
        return True, None
    except sqlite3.IntegrityError:
        return False, 'That username is already taken.'
    finally:
        conn.close()

def user_set_password(uid, password):
    if len(password or '') < 6:
        return False, 'Password must be at least 6 characters.'
    conn = _users_db()
    conn.execute('UPDATE users SET password_hash=? WHERE id=?', (generate_password_hash(password), uid))
    conn.commit()
    conn.close()
    return True, None

def user_set_role(uid, role):
    if role not in ('admin', 'user'):
        return False, 'Invalid role.'
    if role == 'user' and user_count_active_admins(exclude_id=uid) == 0:
        target = user_get(uid)
        if target and target['role'] == 'admin' and target['is_active']:
            return False, "Can't remove the last active admin."
    conn = _users_db()
    conn.execute('UPDATE users SET role=? WHERE id=?', (role, uid))
    conn.commit()
    conn.close()
    return True, None

def user_set_active(uid, active):
    if not active and user_count_active_admins(exclude_id=uid) == 0:
        target = user_get(uid)
        if target and target['role'] == 'admin':
            return False, "Can't disable the last active admin."
    conn = _users_db()
    conn.execute('UPDATE users SET is_active=? WHERE id=?', (1 if active else 0, uid))
    conn.commit()
    conn.close()
    return True, None

def user_delete(uid):
    target = user_get(uid)
    if target and target['role'] == 'admin' and target['is_active'] and user_count_active_admins(exclude_id=uid) == 0:
        return False, "Can't delete the last active admin."
    conn = _users_db()
    conn.execute('DELETE FROM users WHERE id=?', (uid,))
    conn.commit()
    conn.close()
    return True, None

def user_touch_login(uid):
    conn = _users_db()
    conn.execute('UPDATE users SET last_login=? WHERE id=?', (time.time(), uid))
    conn.commit()
    conn.close()

def user_set_group(uid, group_id):
    """group_id=None (or 0/'') clears the assignment -- back to unrestricted,
    same as an account that's never been assigned one."""
    group_id = group_id or None
    conn = _users_db()
    if group_id is not None:
        exists = conn.execute('SELECT 1 FROM groups WHERE id=?', (group_id,)).fetchone()
        if not exists:
            conn.close()
            return False, 'No such group.'
    conn.execute('UPDATE users SET group_id=? WHERE id=?', (group_id, uid))
    conn.commit()
    conn.close()
    return True, None

# ---- Groups (per-tab access control) ----
def group_list():
    """Every group with its permission set and member count, for the
    Groups admin panel."""
    conn = _users_db()
    groups = [dict(r) for r in conn.execute('SELECT id, name, created_at FROM groups ORDER BY name COLLATE NOCASE')]
    for g in groups:
        perms = conn.execute('SELECT permission FROM group_permissions WHERE group_id=?', (g['id'],)).fetchall()
        g['permissions'] = sorted(p['permission'] for p in perms)
        g['member_count'] = conn.execute('SELECT COUNT(*) AS c FROM users WHERE group_id=?', (g['id'],)).fetchone()['c']
    conn.close()
    return groups

def group_get(gid):
    conn = _users_db()
    row = conn.execute('SELECT id, name, created_at FROM groups WHERE id=?', (gid,)).fetchone()
    if not row:
        conn.close()
        return None
    g = dict(row)
    perms = conn.execute('SELECT permission FROM group_permissions WHERE group_id=?', (gid,)).fetchall()
    g['permissions'] = sorted(p['permission'] for p in perms)
    conn.close()
    return g

def group_create(name):
    name = (name or '').strip()
    if not name:
        return False, 'Enter a name for the group.', None
    conn = _users_db()
    try:
        cur = conn.execute('INSERT INTO groups (name, created_at) VALUES (?,?)', (name, time.time()))
        conn.commit()
        return True, None, cur.lastrowid
    except sqlite3.IntegrityError:
        return False, 'A group with that name already exists.', None
    finally:
        conn.close()

def group_set_permissions(gid, permissions):
    """Replaces a group's whole permission set with `permissions` (any
    subset of AVAILABLE_PERMISSIONS' keys -- unknown keys are dropped rather
    than rejected outright, so a stale checkbox from a future/past version
    of this list can't wedge the save)."""
    perms = sorted(set(permissions or []) & _PERMISSION_KEYS)
    conn = _users_db()
    if not conn.execute('SELECT 1 FROM groups WHERE id=?', (gid,)).fetchone():
        conn.close()
        return False, 'No such group.'
    conn.execute('DELETE FROM group_permissions WHERE group_id=?', (gid,))
    conn.executemany('INSERT INTO group_permissions (group_id, permission) VALUES (?,?)',
                      [(gid, p) for p in perms])
    conn.commit()
    conn.close()
    return True, None

def group_rename(gid, name):
    name = (name or '').strip()
    if not name:
        return False, 'Enter a name for the group.'
    conn = _users_db()
    try:
        conn.execute('UPDATE groups SET name=? WHERE id=?', (name, gid))
        conn.commit()
        return True, None
    except sqlite3.IntegrityError:
        return False, 'A group with that name already exists.'
    finally:
        conn.close()

def group_delete(gid):
    """Members of a deleted group fall back to group_id=NULL -- unrestricted,
    not locked out -- rather than left pointing at a group row that no
    longer exists. Done explicitly here (not via a DB-level FK cascade,
    which this app doesn't have enabled -- see users_db_init's comment on
    the users.group_id column) so it happens reliably regardless of that."""
    conn = _users_db()
    if not conn.execute('SELECT 1 FROM groups WHERE id=?', (gid,)).fetchone():
        conn.close()
        return False, 'No such group.'
    conn.execute('UPDATE users SET group_id=NULL WHERE group_id=?', (gid,))
    conn.execute('DELETE FROM group_permissions WHERE group_id=?', (gid,))
    conn.execute('DELETE FROM groups WHERE id=?', (gid,))
    conn.commit()
    conn.close()
    return True, None

def user_can(user_id, role, *permissions):
    """True if the session identified by (user_id, role) may use a feature
    gated behind ANY of the given permission keys (multiple keys = OR, for
    the handful of endpoints two different tabs both depend on -- see
    require_permission's docstring). Admins always pass. An account with no
    group assigned (group_id IS NULL) always passes too -- see
    users_db_init's comment on why that's the safe default rather than
    locking out every existing account the moment this feature ships."""
    if role == 'admin':
        return True
    if not user_id:
        return False
    conn = _users_db()
    row = conn.execute('SELECT group_id FROM users WHERE id=?', (user_id,)).fetchone()
    if not row or not row['group_id']:
        conn.close()
        return True
    granted = conn.execute('SELECT permission FROM group_permissions WHERE group_id=?',
                           (row['group_id'],)).fetchall()
    conn.close()
    granted = {p['permission'] for p in granted}
    return any(p in granted for p in permissions)

def user_permissions(user_id, role):
    """The set of permission keys the current session actually has, for
    handing to the template so it can hide tabs a user's group doesn't
    grant. Admins and ungrouped accounts get the full set (matching
    user_can's semantics above) rather than an empty one, since neither is
    actually restricted."""
    if role == 'admin' or not user_id:
        return set(_PERMISSION_KEYS) if role == 'admin' else set()
    conn = _users_db()
    row = conn.execute('SELECT group_id FROM users WHERE id=?', (user_id,)).fetchone()
    if not row or not row['group_id']:
        conn.close()
        return set(_PERMISSION_KEYS)
    granted = conn.execute('SELECT permission FROM group_permissions WHERE group_id=?',
                           (row['group_id'],)).fetchall()
    conn.close()
    return {p['permission'] for p in granted}

def require_permission(*permissions):
    """Route decorator: 403s (JSON for /api/, redirect to / otherwise)
    unless the session has at least one of the given permissions per
    user_can() above. Multiple permissions are OR'd together -- needed for
    the couple of endpoints two different tabs both genuinely depend on
    (e.g. /api/vo/preview is the real 'Render narration' action for both
    the standalone Text to Speech tab and the promo generator's narration
    section, so it's gated with require_permission('text_to_speech',
    'promo_generation') rather than picking one and quietly breaking the
    other)."""
    def decorator(f):
        @functools.wraps(f)
        def wrapped(*args, **kwargs):
            if not user_can(session.get('user_id'), session.get('role'), *permissions):
                if request.path.startswith('/api/'):
                    return jsonify(error="Your account doesn't have access to this feature."), 403
                return redirect('/')
            return f(*args, **kwargs)
        return wrapped
    return decorator

def admin_reset_if_requested():
    """Recovery path for a lost admin password, independent of whatever's
    already in users.db -- set RESET_ADMIN=1 (plus optionally ADMIN_USERNAME/
    ADMIN_PASSWORD) and restart. Unlike the first-run bootstrap in
    users_db_init(), this runs every startup and always takes effect when the
    flag is set, whether or not accounts already exist: if the named account
    exists it's reset to an active admin with a new password; if not, it's
    created. Meant to be turned back off after use.
    """
    if os.environ.get('RESET_ADMIN', '').strip().lower() not in ('1', 'true', 'yes'):
        return
    username = (os.environ.get('ADMIN_USERNAME', '').strip() or 'admin')
    password = os.environ.get('ADMIN_PASSWORD', '').strip() or DEFAULT_ADMIN_PASSWORD
    existing = user_get_by_username(username)
    conn = _users_db()
    if existing:
        conn.execute('UPDATE users SET password_hash=?, role=?, is_active=1 WHERE id=?',
                     (generate_password_hash(password), 'admin', existing['id']))
    else:
        conn.execute(
            'INSERT INTO users (username, password_hash, role, is_active, created_at) VALUES (?,?,?,1,?)',
            (username, generate_password_hash(password), 'admin', time.time()))
    conn.commit()
    conn.close()
    print('=' * 64)
    print(f' * RESET_ADMIN was set -- {"reset" if existing else "created"} admin account:')
    print(f'     username: {username}')
    print(f'     password: {password}')
    print('   Remove RESET_ADMIN from the environment and restart, or it will')
    print('   reset this account back to this password on every future boot.')
    print('=' * 64)

users_db_init()
admin_reset_if_requested()

# ---- Admin: user management ----
# Simple server-rendered page; only reachable by an authenticated admin (see
# _access_control's _ADMIN_PREFIX check). Anyone else gets a redirect/403.

def _admin_users_page(error=None, notice=None):
    users = user_list()
    groups = group_list()
    group_options = ''.join(f'<option value={g["id"]}>{g["name"]}</option>' for g in groups)
    rows = ''
    for u in users:
        created = time.strftime('%Y-%m-%d', time.localtime(u['created_at'])) if u['created_at'] else '—'
        last_login = time.strftime('%Y-%m-%d %H:%M', time.localtime(u['last_login'])) if u['last_login'] else 'never'
        is_you = (u['id'] == session.get('user_id'))
        this_group_options = f'<option value="">\u2014 none (unrestricted) \u2014</option>' + ''.join(
            f'<option value={g["id"]} {"selected" if u["group_id"] == g["id"] else ""}>{g["name"]}</option>' for g in groups)
        rows += f'''<tr>
<td>{u['username']}{' <span class="you">(you)</span>' if is_you else ''}</td>
<td>
  <form method=post action="/admin/users/{u['id']}/role" class=inline-form>
    <select name=role onchange="this.form.submit()" {'disabled' if is_you else ''}>
      <option value=user {'selected' if u['role']=='user' else ''}>user</option>
      <option value=admin {'selected' if u['role']=='admin' else ''}>admin</option>
    </select>
  </form>
</td>
<td>
  <form method=post action="/admin/users/{u['id']}/group" class=inline-form>
    <select name=group_id onchange="this.form.submit()">{this_group_options}</select>
  </form>
</td>
<td>{'active' if u['is_active'] else 'disabled'}</td>
<td>{created}</td>
<td>{last_login}</td>
<td class=actions>
  <form method=post action="/admin/users/{u['id']}/toggle" class=inline-form>
    <button type=submit class=btn-sm {'disabled' if is_you else ''}>{'Disable' if u['is_active'] else 'Enable'}</button>
  </form>
  <form method=post action="/admin/users/{u['id']}/password" class=inline-form pw-form>
    <input type=password name=password placeholder="New password" minlength=6 required>
    <button type=submit class=btn-sm>Set</button>
  </form>
  <form method=post action="/admin/users/{u['id']}/delete" class=inline-form
        onsubmit="return confirm('Delete user {u['username']}? This cannot be undone.')">
    <button type=submit class="btn-sm btn-danger" {'disabled' if is_you else ''}>Delete</button>
  </form>
</td>
</tr>'''

    group_rows = ''
    for g in groups:
        checks = ''.join(
            f'<label class=perm-check><input type=checkbox name=permission value={key} '
            f'{"checked" if key in g["permissions"] else ""}> {label}</label>'
            for key, label in AVAILABLE_PERMISSIONS)
        group_rows += f'''<div class=group-card>
<form method=post action="/admin/groups/{g['id']}/permissions">
  <div class=group-head>
    <strong>{g['name']}</strong>
    <span class=group-meta>{g['member_count']} member{'s' if g['member_count'] != 1 else ''}</span>
    <button type=submit class="btn-sm primary-sm">Save</button>
    <button type=submit form=delgroup-{g['id']} class="btn-sm btn-danger"
            onclick="return confirm('Delete group {g['name']}? Members fall back to unrestricted access, not locked out.')">Delete</button>
  </div>
  <div class=perm-grid>{checks}</div>
</form>
<form id=delgroup-{g['id']} method=post action="/admin/groups/{g['id']}/delete"></form>
</div>'''
    if not groups:
        group_rows = '<p style="font-size:12px;opacity:.7;margin:4px 0 0">No groups yet. Accounts with no group assigned have unrestricted access to every tab.</p>'

    return f'''<!doctype html><html><head><meta charset="utf-8">
<title>Manage users</title>
<style>
body{{font-family:system-ui,sans-serif;background:#0b1220;color:#e6e9ef;margin:0;padding:32px}}
.wrap{{max-width:960px;margin:0 auto}}
h1{{font-size:20px;margin:0 0 4px}}
.back{{color:#7fa4ff;text-decoration:none;font-size:13px}}
.card{{background:#141b2d;border:1px solid #232b41;border-radius:10px;padding:20px;margin-top:20px}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th,td{{text-align:left;padding:8px 10px;border-bottom:1px solid #232b41;vertical-align:middle}}
th{{color:#8b94a8;font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.03em}}
.you{{color:#7fa4ff;font-size:11px}}
.inline-form{{display:inline-flex;gap:4px;align-items:center;margin:0 4px 0 0}}
.actions{{display:flex;flex-wrap:wrap;gap:4px;align-items:center}}
.pw-form input{{width:110px}}
select,input{{background:#0b1220;border:1px solid #232b41;color:#e6e9ef;border-radius:5px;padding:5px 7px;font-size:12px}}
.btn-sm{{background:#232b41;border:1px solid #2d3752;color:#e6e9ef;border-radius:5px;padding:5px 9px;font-size:12px;cursor:pointer}}
.btn-sm:hover{{background:#2d3752}}
.btn-sm:disabled{{opacity:.4;cursor:not-allowed}}
.btn-danger{{background:#3c1f24;border-color:#5c2a30;color:#e08a8a}}
.btn-danger:hover{{background:#4a2429}}
.primary-sm{{background:#1a3a33;border-color:#245144;color:#7fd9c5}}
.primary-sm:hover{{background:#204a40}}
.add-form{{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-top:6px}}
.add-form input{{flex:1;min-width:140px}}
.primary{{background:#4f8cff;border:none;color:#fff;border-radius:6px;padding:8px 16px;font-size:13px;cursor:pointer}}
.err{{color:#e08a3c;font-size:13px;margin-bottom:12px}}
.notice{{color:#7fd99a;font-size:13px;margin-bottom:12px}}
.group-card{{border:1px solid #232b41;border-radius:8px;padding:12px 14px;margin-top:12px}}
.group-head{{display:flex;align-items:center;gap:10px;margin-bottom:10px}}
.group-meta{{color:#8b94a8;font-size:11px;flex:1}}
.perm-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:6px 14px}}
.perm-check{{display:flex;align-items:center;gap:6px;font-size:12px;cursor:pointer}}
</style></head><body>
<div class=wrap>
<a class=back href="/">&larr; Back to app</a>
<h1>Manage users</h1>
{f'<div class="err">{error}</div>' if error else ''}
{f'<div class="notice">{notice}</div>' if notice else ''}
<div class=card>
<table>
<tr><th>Username</th><th>Role</th><th>Group</th><th>Status</th><th>Created</th><th>Last login</th><th>Actions</th></tr>
{rows}
</table>
</div>
<div class=card>
<h2 style="font-size:14px;margin:0">Add user</h2>
<form method=post action="/admin/users/create" class=add-form>
<input type=text name=username placeholder="Username" required>
<input type=password name=password placeholder="Password (min 6 chars)" minlength=6 required>
<select name=role><option value=user selected>user</option><option value=admin>admin</option></select>
<button type=submit class=primary>Create</button>
</form>
</div>
<div class=card>
<h2 style="font-size:14px;margin:0 0 4px">Groups</h2>
<p style="font-size:12px;opacity:.7;margin:0">A group grants access to only the checked tabs. An account with no group assigned (the default) is unrestricted -- creating and assigning a group is an opt-in way to <em>narrow</em> access, not something that happens automatically. Admins always have full access regardless of group.</p>
{group_rows}
<form method=post action="/admin/groups/create" class=add-form style="margin-top:14px">
<input type=text name=name placeholder="New group name (e.g. Editors)" required>
<button type=submit class=primary>Create group</button>
</form>
</div>
</div>
</body></html>'''

@app.route('/admin/users')
def admin_users():
    return _admin_users_page()

@app.route('/admin/users/create', methods=['POST'])
def admin_users_create():
    ok, err = user_create(request.form.get('username', ''), request.form.get('password', ''),
                           request.form.get('role', 'user'))
    return _admin_users_page(error=None if ok else err, notice='User created.' if ok else None)

@app.route('/admin/users/<int:uid>/role', methods=['POST'])
def admin_users_role(uid):
    ok, err = user_set_role(uid, request.form.get('role', 'user'))
    return _admin_users_page(error=None if ok else err, notice='Role updated.' if ok else None)

@app.route('/admin/users/<int:uid>/toggle', methods=['POST'])
def admin_users_toggle(uid):
    target = user_get(uid)
    if not target:
        return _admin_users_page(error='User not found.')
    ok, err = user_set_active(uid, not target['is_active'])
    return _admin_users_page(error=None if ok else err, notice='Status updated.' if ok else None)

@app.route('/admin/users/<int:uid>/password', methods=['POST'])
def admin_users_password(uid):
    ok, err = user_set_password(uid, request.form.get('password', ''))
    return _admin_users_page(error=None if ok else err, notice='Password updated.' if ok else None)

@app.route('/admin/users/<int:uid>/delete', methods=['POST'])
def admin_users_delete(uid):
    if uid == session.get('user_id'):
        return _admin_users_page(error="You can't delete your own account while signed in as it.")
    ok, err = user_delete(uid)
    return _admin_users_page(error=None if ok else err, notice='User deleted.' if ok else None)

@app.route('/admin/users/<int:uid>/group', methods=['POST'])
def admin_users_group(uid):
    raw = request.form.get('group_id', '').strip()
    gid = int(raw) if raw.isdigit() else None
    ok, err = user_set_group(uid, gid)
    return _admin_users_page(error=None if ok else err, notice='Group updated.' if ok else None)

@app.route('/admin/groups/create', methods=['POST'])
def admin_groups_create():
    ok, err, _gid = group_create(request.form.get('name', ''))
    return _admin_users_page(error=None if ok else err, notice='Group created -- check its permissions below, then assign accounts to it.' if ok else None)

@app.route('/admin/groups/<int:gid>/permissions', methods=['POST'])
def admin_groups_permissions(gid):
    perms = request.form.getlist('permission')
    ok, err = group_set_permissions(gid, perms)
    return _admin_users_page(error=None if ok else err, notice='Permissions saved.' if ok else None)

@app.route('/admin/groups/<int:gid>/delete', methods=['POST'])
def admin_groups_delete(gid):
    ok, err = group_delete(gid)
    return _admin_users_page(error=None if ok else err, notice='Group deleted. Its members are now unrestricted.' if ok else None)


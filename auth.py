"""User accounts, login/logout, and the /admin/users management page.

Depends on: core (app, rate limiter, dummy-hash, default password) and
library_db (LIBRARY_DIR, _sqlite_connect) for where users.db lives.
"""
import os, time, sqlite3
from flask import request, session, redirect, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from core import app, _client_ip, _login_limiter, _DUMMY_PW_HASH, DEFAULT_ADMIN_PASSWORD
from library_db import LIBRARY_DIR, _sqlite_connect

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
    return f'''<!doctype html><html><head><meta charset="utf-8">
<title>Sign in</title>
<style>body{{font-family:system-ui,sans-serif;background:#0b1220;color:#e6e9ef;
display:flex;align-items:center;justify-content:center;height:100vh;margin:0}}
form{{background:#141b2d;padding:28px 32px;border-radius:10px;border:1px solid #232b41;width:280px}}
h1{{font-size:16px;margin:0 0 16px}}
input{{width:100%;box-sizing:border-box;padding:9px 10px;border-radius:6px;border:1px solid #232b41;
background:#0b1220;color:#e6e9ef;font-size:14px;margin-bottom:12px}}
button{{width:100%;padding:9px;border-radius:6px;border:none;background:#4f8cff;color:#fff;
font-size:14px;cursor:pointer}}
.err{{color:#e08a3c;font-size:12px;margin-bottom:12px}}</style></head><body>
<form method=post>
<h1>Sign in</h1>
{f'<div class="err">{error}</div>' if error else ''}
<input type=hidden name=next value="{nxt}">
<input type=text name=username placeholder="Username" autofocus autocomplete="username">
<input type=password name=password placeholder="Password" autocomplete="current-password">
<button type=submit>Continue</button>
</form></body></html>'''

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
        'SELECT id, username, role, is_active, created_at, last_login FROM users ORDER BY username COLLATE NOCASE'
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
    rows = ''
    for u in users:
        created = time.strftime('%Y-%m-%d', time.localtime(u['created_at'])) if u['created_at'] else '—'
        last_login = time.strftime('%Y-%m-%d %H:%M', time.localtime(u['last_login'])) if u['last_login'] else 'never'
        is_you = (u['id'] == session.get('user_id'))
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
    return f'''<!doctype html><html><head><meta charset="utf-8">
<title>Manage users</title>
<style>
body{{font-family:system-ui,sans-serif;background:#0b1220;color:#e6e9ef;margin:0;padding:32px}}
.wrap{{max-width:920px;margin:0 auto}}
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
.add-form{{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-top:6px}}
.add-form input{{flex:1;min-width:140px}}
.primary{{background:#4f8cff;border:none;color:#fff;border-radius:6px;padding:8px 16px;font-size:13px;cursor:pointer}}
.err{{color:#e08a3c;font-size:13px;margin-bottom:12px}}
.notice{{color:#7fd99a;font-size:13px;margin-bottom:12px}}
</style></head><body>
<div class=wrap>
<a class=back href="/">&larr; Back to app</a>
<h1>Manage users</h1>
{f'<div class="err">{error}</div>' if error else ''}
{f'<div class="notice">{notice}</div>' if notice else ''}
<div class=card>
<table>
<tr><th>Username</th><th>Role</th><th>Status</th><th>Created</th><th>Last login</th><th>Actions</th></tr>
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


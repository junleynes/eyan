"""User accounts, login/logout, and the /admin/users management page.

Depends on: core (app, rate limiter, dummy-hash, default password) and
library_db (LIBRARY_DIR, _sqlite_connect) for where users.db lives.
"""
import os, time, sqlite3, functools, secrets
from flask import request, session, redirect, jsonify, render_template
from markupsafe import escape
from werkzeug.security import generate_password_hash, check_password_hash
from core import app, _client_ip, _login_limiter, _DUMMY_PW_HASH, DEFAULT_ADMIN_PASSWORD, ensure_csrf_token
from library_db import LIBRARY_DIR, _sqlite_connect, load_branding

def _safe_next(dest):
    """Only ever redirect to a path on this app -- an absolute or
    protocol-relative 'next' would be an open-redirect vector."""
    dest = dest or '/'
    if not dest.startswith('/') or dest.startswith('//'):
        return '/'
    return dest

def _establish_session(user):
    """Promotes a verified user to a fully signed-in session. Called only
    after EVERY required factor has passed -- password alone for an account
    without 2FA, password + code for one with it."""
    session.pop('pending_2fa_uid', None)
    session.pop('pending_2fa_next', None)
    session.permanent = True
    session['authed'] = True
    session['user_id'] = user['id']
    session['username'] = user['username']
    session['role'] = user['role']
    # Fresh CSRF token per sign-in, and only after every factor has passed --
    # so a token handed out during the half-authenticated 2FA step can't be
    # reused against the fully-authenticated session.
    session['csrf_token'] = secrets.token_urlsafe(32)
    user_touch_login(user['id'])

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    # A pending 2FA challenge: the password was already accepted, and we're
    # waiting on the code. Held in the session (not a hidden form field) so the
    # "password was correct" fact can't be forged by posting a crafted form
    # straight to the second step.
    pending_uid = session.get('pending_2fa_uid')
    if request.method == 'POST':
        if not _login_limiter.allow(_client_ip()):
            error = 'Too many attempts. Wait a few minutes and try again.'
        elif pending_uid:
            # ---- Step 2: verify the TOTP (or backup) code ----
            user = user_get(pending_uid)
            if not user or not user['is_active']:
                session.pop('pending_2fa_uid', None)
                error = 'That account is no longer available. Sign in again.'
            elif user_totp_verify(user, request.form.get('totp_code')):
                session.pop('pending_2fa_uid', None)
                _establish_session(user)
                return redirect(_safe_next(request.form.get('next')))
            else:
                locked_until = user_note_failed_login(user['id'])
                if locked_until:
                    session.pop('pending_2fa_uid', None)
                    error = f'Too many failed codes. This account is locked for {LOGIN_LOCKOUT_MINUTES} minutes.'
                else:
                    error = 'That code was not correct. Try again, or use one of your backup codes.'
        else:
            # ---- Step 1: username + password ----
            username = request.form.get('username', '').strip()
            password = request.form.get('password', '')
            user = user_get_by_username(username)
            # Run check_password_hash even on a miss (against a dummy hash) so
            # a nonexistent username doesn't respond measurably faster than a
            # wrong password -- that timing gap would otherwise leak which
            # usernames exist.
            pw_ok = check_password_hash(user['password_hash'] if user else _DUMMY_PW_HASH, password)
            locked_secs = user_lockout_remaining(user) if user else 0
            if locked_secs:
                # Deliberately the same message whether or not the password was
                # right -- confirming "correct password, but locked" would tell
                # an attacker they've found valid credentials.
                error = f'Too many failed attempts. Try again in {max(1, locked_secs // 60)} minute(s).'
            elif user and pw_ok and user['is_active']:
                if user['totp_enabled']:
                    # Hold the session in a half-authenticated state: enough to
                    # remember WHO is signing in, not enough to reach anything.
                    session['pending_2fa_uid'] = user['id']
                    session['pending_2fa_next'] = _safe_next(request.form.get('next'))
                    pending_uid = user['id']
                else:
                    _establish_session(user)
                    return redirect(_safe_next(request.form.get('next')))
            elif user and not user['is_active']:
                error = 'This account is disabled or waiting for admin approval.'
            else:
                if user:
                    user_note_failed_login(user['id'])
                error = 'Incorrect username or password.'
    # GET /login → send people to the landing page (login lives there now).
    # POST failures and the 2FA step re-render the landing with the form state
    # so the user never leaves the marketing + sign-in experience.
    if request.method == 'GET' and not pending_uid and not error:
        nxt = request.args.get('next', '/')
        if nxt and nxt != '/':
            return redirect('/?next=' + nxt + '#signin')
        return redirect('/#signin')

    nxt = request.form.get('next') or request.args.get('next', '/')
    nxt = _safe_next(nxt)
    brand = load_branding()
    return render_template('landing.html',
                           brand_name=brand['name'],
                           brand_tagline=brand['tagline'],
                           brand_theme=brand['theme_colors'],
                           login_error=error,
                           register_error=None,
                           register_success=None,
                           pending_2fa=bool(pending_uid),
                           next_url=nxt,
                           auth_panel='signin' if (error or pending_uid) else None)


@app.route('/register', methods=['GET', 'POST'])
def register():
    """Self-service registration. Accounts are created inactive until an admin enables them."""
    from pipeline import load_branding
    brand = load_branding()
    if request.method == 'GET':
        return redirect('/#register')

    username = (request.form.get('username') or '').strip()
    password = request.form.get('password') or ''
    password2 = request.form.get('password2') or ''
    error = None
    success = None
    if password != password2:
        error = 'Passwords do not match.'
    else:
        ok, err = user_create(username, password, role='user', active=False)
        if ok:
            success = ('Account requested. An administrator must approve it before you can sign in.')
        else:
            error = err

    return render_template(
        'landing.html',
        brand_name=brand['name'],
        brand_tagline=brand['tagline'],
        brand_theme=brand['theme_colors'],
        login_error=error if not success else None,
        register_error=error if not success else None,
        register_success=success,
        pending_2fa=False,
        next_url='/',
        auth_panel='register' if not success else 'register-done',
    )


@app.route('/logout')
def logout():
    session.clear()
    return redirect('/#signin')

# ---- Two-factor enrolment (self-service, any signed-in account) ----
# Deliberately not admin-gated: 2FA protects the individual account, so every
# user manages their own. An admin can only turn someone else's OFF (see
# /admin/users/<uid>/2fa/reset below) for the lost-phone case -- they can't
# enrol on another user's behalf, since that would mean the admin holding a
# secret that's supposed to be the user's alone.
@app.route('/api/2fa/status')
def api_2fa_status():
    if not session.get('authed'):
        return jsonify(ok=False, error='Not signed in.'), 403
    user = user_get(session['user_id'])
    return jsonify(ok=True, enabled=bool(user and user['totp_enabled']),
                   available=totp_available(),
                   backup_remaining=len([c for c in (user['totp_backup_codes'] or '').split('\n') if c]) if user else 0)

@app.route('/api/2fa/begin', methods=['POST'])
def api_2fa_begin():
    """Generates a fresh secret and returns it plus a scannable QR. Does NOT
    enable 2FA -- see api_2fa_confirm. Re-running this before confirming just
    replaces the pending secret, so restarting a half-finished setup is safe."""
    if not session.get('authed'):
        return jsonify(ok=False, error='Not signed in.'), 403
    if not totp_available():
        return jsonify(ok=False, error='2FA support is not installed on this server (pip install pyotp qrcode).'), 501
    secret, uri = user_totp_begin_enrolment(session['user_id'])
    if not secret:
        return jsonify(ok=False, error='Could not start 2FA setup.'), 500
    qr_data_uri = None
    try:
        import qrcode, io, base64
        img = qrcode.make(uri)
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        qr_data_uri = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        # A missing/broken qrcode lib shouldn't block enrolment -- the secret
        # can still be typed into the app by hand, so this degrades to
        # manual entry rather than failing outright.
        print(f'2FA QR render failed (falling back to manual entry): {e}')
    return jsonify(ok=True, secret=secret, uri=uri, qr=qr_data_uri)

@app.route('/api/2fa/confirm', methods=['POST'])
def api_2fa_confirm():
    """Verifies a code against the pending secret and, on success, switches 2FA
    on and returns the one-time backup codes. These are the ONLY time the
    plaintext codes exist -- only their hashes are stored."""
    if not session.get('authed'):
        return jsonify(ok=False, error='Not signed in.'), 403
    data = request.get_json(silent=True) or {}
    codes, err = user_totp_confirm(session['user_id'], data.get('code'))
    if err:
        return jsonify(ok=False, error=err), 400
    return jsonify(ok=True, backup_codes=codes)

@app.route('/api/2fa/disable', methods=['POST'])
def api_2fa_disable():
    """Turns 2FA off for the signed-in account. Requires a currently-valid
    code (or backup code) rather than just a session -- otherwise anyone who
    got hold of an already-signed-in browser could quietly strip the second
    factor off the account."""
    if not session.get('authed'):
        return jsonify(ok=False, error='Not signed in.'), 403
    user = user_get(session['user_id'])
    if not user or not user['totp_enabled']:
        return jsonify(ok=True)   # already off; nothing to do
    data = request.get_json(silent=True) or {}
    if not user_totp_verify(user, data.get('code')):
        return jsonify(ok=False, error='Enter a current code from your authenticator app (or a backup code) to turn 2FA off.'), 400
    user_totp_disable(session['user_id'])
    return jsonify(ok=True)


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
    # ---- Two-factor authentication (TOTP, Google Authenticator compatible) ----
    # totp_secret holds the base32 shared secret; totp_enabled is only set to 1
    # AFTER the user has proved they can generate a valid code from it, so an
    # interrupted enrolment can never lock someone out of their own account with
    # a secret they never successfully scanned. totp_backup_codes holds
    # newline-separated HASHES of one-time recovery codes -- hashed rather than
    # stored plainly for the same reason passwords are: a database read
    # shouldn't hand over a working second factor.
    if 'totp_secret' not in have:
        conn.execute('ALTER TABLE users ADD COLUMN totp_secret TEXT')
    if 'totp_enabled' not in have:
        conn.execute('ALTER TABLE users ADD COLUMN totp_enabled INTEGER NOT NULL DEFAULT 0')
    if 'totp_backup_codes' not in have:
        conn.execute('ALTER TABLE users ADD COLUMN totp_backup_codes TEXT')
    if 'failed_logins' not in have:
        # Per-ACCOUNT failure tracking, alongside the existing per-IP rate
        # limit. The IP limiter alone is bypassed entirely by a distributed
        # attempt (many IPs, one target account), which is exactly the shape
        # a public-facing deployment attracts.
        conn.execute('ALTER TABLE users ADD COLUMN failed_logins INTEGER NOT NULL DEFAULT 0')
    if 'locked_until' not in have:
        conn.execute('ALTER TABLE users ADD COLUMN locked_until REAL')
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
        'SELECT u.id, u.username, u.role, u.is_active, u.created_at, u.last_login, u.group_id, '
        'u.totp_enabled, g.name AS group_name '
        'FROM users u LEFT JOIN groups g ON g.id = u.group_id ORDER BY u.username COLLATE NOCASE'
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def user_count_active_admins(exclude_id=None):
    conn = _users_db()
    rows = conn.execute("SELECT id FROM users WHERE role='admin' AND is_active=1").fetchall()
    conn.close()
    return len([r for r in rows if r['id'] != exclude_id])

# ---- Password policy ----
# 12 characters rather than the previous 6. Six is trivially brute-forcible
# and was only ever defensible on a LAN-only deployment; anything reachable
# from outside needs a length that actually costs something to guess. Length
# is deliberately the only hard requirement -- forced character-class rules
# reliably produce "Password1!" rather than genuinely stronger secrets, and
# NIST's own guidance now recommends against them.
MIN_PASSWORD_LENGTH = int(os.environ.get('MIN_PASSWORD_LENGTH', 12))

def _password_policy_error(password):
    """Returns (ok, reason)."""
    pw = password or ''
    if len(pw) < MIN_PASSWORD_LENGTH:
        return False, f'Password must be at least {MIN_PASSWORD_LENGTH} characters.'
    if pw.lower() in ('password', 'password123', '123456789012', 'administrator'):
        return False, 'That password is too common -- pick something less guessable.'
    return True, None

def user_create(username, password, role='user', active=True):
    """Create an account. `active=False` is used for self-registration (pending admin approval)."""
    username = (username or '').strip()
    if not username:
        return False, 'Username is required.'
    if len(username) < 3:
        return False, 'Username must be at least 3 characters.'
    if len(username) > 64:
        return False, 'Username is too long.'
    ok, why = _password_policy_error(password)
    if not ok:
        return False, why
    if role not in ('admin', 'user'):
        role = 'user'
    conn = _users_db()
    try:
        conn.execute(
            'INSERT INTO users (username, password_hash, role, is_active, created_at) VALUES (?,?,?,?,?)',
            (username, generate_password_hash(password), role, 1 if active else 0, time.time()))
        conn.commit()
        return True, None
    except sqlite3.IntegrityError:
        return False, 'That username is already taken.'
    finally:
        conn.close()

def user_set_password(uid, password):
    ok, why = _password_policy_error(password)
    if not ok:
        return False, why
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
    # A successful sign-in clears any accumulated failure count and lockout --
    # otherwise a user who fumbled their password a few times would stay one
    # mistake away from a lockout indefinitely.
    conn.execute('UPDATE users SET last_login=?, failed_logins=0, locked_until=NULL WHERE id=?',
                 (time.time(), uid))
    conn.commit()
    conn.close()

# ---- Per-account lockout (complements the per-IP rate limit in core.py) ----
LOGIN_MAX_FAILURES = int(os.environ.get('LOGIN_MAX_FAILURES', 10))
LOGIN_LOCKOUT_MINUTES = int(os.environ.get('LOGIN_LOCKOUT_MINUTES', 15))

def user_note_failed_login(uid):
    """Counts a failed attempt against the ACCOUNT and locks it temporarily
    once the threshold is hit. The existing per-IP limiter can't see a
    distributed attempt (many source IPs, one target account), which is the
    realistic shape of an attack on a public-facing login."""
    conn = _users_db()
    row = conn.execute('SELECT failed_logins FROM users WHERE id=?', (uid,)).fetchone()
    fails = (row['failed_logins'] if row else 0) + 1
    locked_until = None
    if fails >= LOGIN_MAX_FAILURES:
        locked_until = time.time() + LOGIN_LOCKOUT_MINUTES * 60
        fails = 0   # reset the counter so the next lockout needs a fresh run of failures
    conn.execute('UPDATE users SET failed_logins=?, locked_until=? WHERE id=?', (fails, locked_until, uid))
    conn.commit()
    conn.close()
    return locked_until

def user_lockout_remaining(user):
    """Seconds left on an active lockout, or 0 if not locked."""
    if not user or not user['locked_until']:
        return 0
    return max(0, int(user['locked_until'] - time.time()))

# ---- TOTP two-factor (Google Authenticator / Authy / 1Password compatible) ----
def _totp_lib():
    """pyotp imported lazily so the app still starts (with 2FA simply
    unavailable) on a deployment that hasn't installed it yet, rather than
    failing at import time and taking the whole service down."""
    try:
        import pyotp
        return pyotp
    except ImportError:
        return None

def totp_available():
    return _totp_lib() is not None

def user_totp_begin_enrolment(uid):
    """Generates and stores a fresh secret WITHOUT enabling 2FA yet, and
    returns (secret, otpauth_uri). Enabling only happens once the user proves
    they can produce a valid code from it (see user_totp_confirm) -- so
    abandoning enrolment halfway leaves the account exactly as it was."""
    pyotp = _totp_lib()
    if not pyotp:
        return None, None
    secret = pyotp.random_base32()
    conn = _users_db()
    row = conn.execute('SELECT username FROM users WHERE id=?', (uid,)).fetchone()
    username = row['username'] if row else 'user'
    conn.execute('UPDATE users SET totp_secret=?, totp_enabled=0 WHERE id=?', (secret, uid))
    conn.commit()
    conn.close()
    try:
        issuer = load_branding()['name']
    except Exception:
        issuer = 'PRISM'
    uri = pyotp.totp.TOTP(secret).provisioning_uri(name=username, issuer_name=issuer)
    return secret, uri

def _hash_backup_code(code):
    return generate_password_hash(code.strip().upper())

def user_totp_confirm(uid, code):
    """Verifies `code` against the pending secret and, on success, switches 2FA
    on and returns freshly generated one-time backup codes. Returns (codes,
    error) -- codes are shown to the user exactly once here, since only their
    hashes are stored."""
    pyotp = _totp_lib()
    if not pyotp:
        return None, '2FA support is not installed on this server (pip install pyotp).'
    conn = _users_db()
    row = conn.execute('SELECT totp_secret FROM users WHERE id=?', (uid,)).fetchone()
    if not row or not row['totp_secret']:
        conn.close()
        return None, 'Start setup again -- no pending 2FA secret for this account.'
    # valid_window=1 accepts the adjacent 30s step either side, which absorbs
    # ordinary clock drift between the phone and the server without meaningfully
    # widening the guess space.
    if not pyotp.TOTP(row['totp_secret']).verify((code or '').strip(), valid_window=1):
        conn.close()
        return None, 'That code was not correct. Check your phone and try again.'
    codes = [secrets.token_hex(4).upper() for _ in range(10)]
    hashed = '\n'.join(_hash_backup_code(c) for c in codes)
    conn.execute('UPDATE users SET totp_enabled=1, totp_backup_codes=? WHERE id=?', (hashed, uid))
    conn.commit()
    conn.close()
    return codes, None

def user_totp_verify(user, code):
    """Checks `code` against the account's TOTP secret, then against its unused
    backup codes. A matching backup code is CONSUMED (removed from the stored
    list) so it genuinely works only once."""
    pyotp = _totp_lib()
    code = (code or '').strip()
    if not code or not user or not user['totp_secret']:
        return False
    if pyotp and pyotp.TOTP(user['totp_secret']).verify(code, valid_window=1):
        return True
    stored = (user['totp_backup_codes'] or '').split('\n')
    for h in stored:
        if h and check_password_hash(h, code.upper()):
            remaining = '\n'.join(x for x in stored if x != h)
            conn = _users_db()
            conn.execute('UPDATE users SET totp_backup_codes=? WHERE id=?', (remaining, user['id']))
            conn.commit()
            conn.close()
            return True
    return False

def user_totp_disable(uid):
    conn = _users_db()
    conn.execute('UPDATE users SET totp_secret=NULL, totp_enabled=0, totp_backup_codes=NULL WHERE id=?', (uid,))
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

def _twofa_cell(u):
    """The 2FA column for one row: status, plus a Reset button when it's on.
    Reset is the lost-phone escape hatch -- an admin can clear someone's 2FA
    but never enrol it for them (see admin_users_2fa_reset)."""
    if not u['totp_enabled']:
        return '<span style="opacity:.5;font-size:11px">off</span>'
    return (f'<span class=you style="font-size:11px">on</span> '
            f'<form method=post action="/admin/users/{u["id"]}/2fa/reset" class=inline-form '
            f'onsubmit="return confirm(\'Clear 2FA for {escape(u["username"])}? '
            f'They will sign in with just their password until they set it up again.\')">'
            f'<button type=submit class="btn-sm btn-danger" style="padding:3px 7px;font-size:10px">Reset</button></form>')

def _admin_users_page(error=None, notice=None):
    # This page's forms post to state-changing routes, so they need the
    # session's CSRF token (see _csrf_protect in core.py). ensure_csrf_token
    # rather than reading session directly, so a session predating the CSRF
    # feature gets one here instead of being unable to submit anything.
    csrf_js = ensure_csrf_token()
    users = user_list()
    groups = group_list()
    brand = load_branding()
    brand_name = escape(brand['name'])
    brand_tagline = escape(brand['tagline'])
    theme = brand['theme_colors']
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
<td>{'active' if u['is_active'] else 'pending / disabled'}</td>
<td>{_twofa_cell(u)}</td>
<td>{created}</td>
<td>{last_login}</td>
<td class=actions>
  <form method=post action="/admin/users/{u['id']}/toggle" class=inline-form>
    <button type=submit class=btn-sm {'disabled' if is_you else ''}>{'Disable' if u['is_active'] else 'Enable'}</button>
  </form>
  <form method=post action="/admin/users/{u['id']}/password" class=inline-form pw-form>
    <input type=password name=password placeholder="New password" minlength=12 required>
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
<title>Manage users &mdash; {brand_name}</title>
<link rel="icon" href="/branding/favicon">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root{{
  --bg:{theme['bg_dark']}; --panel:{theme['panel_dark']}; --panel-2:{theme['panel_2_dark']};
  --elevated:{theme['elevated_dark']}; --line:#263149; --ink:#e7edf6; --ink-dim:#8b98ad;
  --phosphor:{theme['phosphor']}; --amber:{theme['amber']}; --tally:{theme['tally']}; --accent:{theme['accent']};
  --rail-bg:{theme['rail_bg_dark']};
}}
*{{box-sizing:border-box}}
body{{
  font-family:'IBM Plex Sans',system-ui,-apple-system,sans-serif; margin:0; color:var(--ink);
  background:
    radial-gradient(ellipse 900px 480px at 12% -12%, color-mix(in srgb, var(--phosphor) 7%, transparent), transparent 60%),
    radial-gradient(ellipse 700px 420px at 100% 0%, color-mix(in srgb, var(--amber) 5%, transparent), transparent 55%),
    var(--bg);
  display:flex; min-height:100vh; -webkit-font-smoothing:antialiased;
}}
/* ---- Left rail -- same structural role as the main app's sidebar (brand
   identity up top, a short nav, signed-in-as at the bottom), so this admin
   page reads as part of the app rather than a separate unstyled backend
   screen you can only leave via the browser's Back button. Not a byte-for-
   byte copy of the real one (that's SPA-tab-driven; this is a handful of
   plain links), but the same visual weight and position. ---- */
.rail{{
  width:220px; flex-shrink:0; background:var(--rail-bg); backdrop-filter:blur(12px);
  border-right:1px solid var(--line); padding:20px 16px; display:flex; flex-direction:column;
  position:sticky; top:0; height:100vh; overflow-y:auto;
}}
.rail-brand{{display:flex; align-items:center; gap:10px; margin-bottom:22px}}
.rail-logo{{width:28px; height:28px; border-radius:7px; object-fit:contain; flex-shrink:0}}
.rail-name{{font-family:'JetBrains Mono',monospace; font-size:14px; font-weight:700;
  letter-spacing:.04em; text-transform:uppercase; line-height:1.2}}
.rail-tagline{{font-family:'JetBrains Mono',monospace; font-size:9px; color:var(--ink-dim);
  letter-spacing:.02em; margin-top:2px}}
.rail-nav{{display:flex; flex-direction:column; gap:2px}}
.rail-link{{
  font-family:'JetBrains Mono',monospace; font-size:11px; letter-spacing:.03em;
  color:var(--ink-dim); text-decoration:none; padding:9px 10px; border-radius:7px;
  display:flex; align-items:center; gap:8px; transition:background .15s,color .15s;
}}
.rail-link:hover{{background:var(--wash-4,rgba(255,255,255,.06)); color:var(--ink)}}
.rail-link.active{{background:color-mix(in srgb, var(--phosphor) 12%, transparent); color:var(--phosphor)}}
.rail-sep{{height:1px; background:var(--line); margin:12px 0; opacity:.6}}
.rail-foot{{margin-top:auto; padding-top:16px; font-size:11px; color:var(--ink-dim); line-height:1.6}}
.rail-foot a{{color:var(--ink-dim)}}
.main{{flex:1; padding:32px 40px; min-width:0}}
.wrap{{max-width:960px}}
h1{{font-family:'JetBrains Mono',monospace; font-size:13px; font-weight:600; text-transform:uppercase;
  letter-spacing:.06em; margin:0 0 4px; color:var(--ink-dim); display:flex; align-items:center; gap:8px}}
h1::before{{content:'\u25b6'; color:var(--phosphor); font-size:11px}}
h2{{font-family:'JetBrains Mono',monospace; font-size:12px; text-transform:uppercase; letter-spacing:.05em;
  color:var(--ink-dim); margin:0}}
.card{{background:var(--elevated); border:1px solid var(--line); border-radius:12px; padding:20px; margin-top:20px}}
table{{width:100%; border-collapse:collapse; font-size:13px}}
th,td{{text-align:left; padding:9px 10px; border-bottom:1px solid var(--line); vertical-align:middle}}
th{{color:var(--ink-dim); font-family:'JetBrains Mono',monospace; font-weight:600; font-size:10px;
  text-transform:uppercase; letter-spacing:.05em}}
.you{{color:var(--phosphor); font-size:11px}}
.inline-form{{display:inline-flex; gap:4px; align-items:center; margin:0 4px 0 0}}
.actions{{display:flex; flex-wrap:wrap; gap:4px; align-items:center}}
.pw-form input{{width:110px}}
select,input{{background:var(--panel); border:1px solid var(--line); color:var(--ink);
  border-radius:7px; padding:6px 8px; font-size:12px; font-family:inherit}}
.btn-sm{{background:transparent; border:1px solid var(--phosphor); color:var(--phosphor);
  font-family:'JetBrains Mono',monospace; text-transform:uppercase; letter-spacing:.03em;
  border-radius:7px; padding:6px 10px; font-size:11px; cursor:pointer; transition:background .15s}}
.btn-sm:hover{{background:color-mix(in srgb, var(--phosphor) 12%, transparent)}}
.btn-sm:disabled{{opacity:.35; cursor:not-allowed}}
.btn-danger{{color:var(--tally); border-color:var(--tally)}}
.btn-danger:hover{{background:color-mix(in srgb, var(--tally) 12%, transparent)}}
.primary-sm{{color:var(--phosphor); border-color:var(--phosphor); font-weight:600}}
.add-form{{display:flex; gap:8px; flex-wrap:wrap; align-items:center; margin-top:10px}}
.add-form input{{flex:1; min-width:140px}}
.primary{{background:var(--accent); border:none; color:#fff; font-family:'JetBrains Mono',monospace;
  text-transform:uppercase; letter-spacing:.04em; border-radius:8px; padding:9px 18px; font-size:12px;
  cursor:pointer; transition:opacity .15s}}
.primary:hover{{opacity:.9}}
.err{{background:color-mix(in srgb, var(--tally) 14%, transparent); border:1px solid color-mix(in srgb, var(--tally) 40%, transparent);
  color:var(--tally); font-size:12px; padding:10px 12px; border-radius:8px; margin-bottom:16px}}
.notice{{background:color-mix(in srgb, var(--phosphor) 14%, transparent); border:1px solid color-mix(in srgb, var(--phosphor) 40%, transparent);
  color:var(--phosphor); font-size:12px; padding:10px 12px; border-radius:8px; margin-bottom:16px}}
.group-card{{border:1px solid var(--line); border-radius:10px; padding:14px 16px; margin-top:12px}}
.group-head{{display:flex; align-items:center; gap:10px; margin-bottom:10px}}
.group-meta{{color:var(--ink-dim); font-size:11px; flex:1}}
.perm-grid{{display:grid; grid-template-columns:repeat(auto-fill,minmax(210px,1fr)); gap:6px 14px}}
.group-check{{display:flex; align-items:center; gap:6px; font-size:12px; cursor:pointer; color:var(--ink)}}
</style></head><body>
<div class="main"><div class=wrap>
{f'<div class="err">{error}</div>' if error else ''}
{f'<div class="notice">{notice}</div>' if notice else ''}
<div class=card>
<table>
<tr><th>Username</th><th>Role</th><th>Group</th><th>Status</th><th>2FA</th><th>Created</th><th>Last login</th><th>Actions</th></tr>
{rows}
</table>
</div>
<div class=card>
<h2>Add user</h2>
<form method=post action="/admin/users/create" class=add-form>
<input type=text name=username placeholder="Username" required>
<input type=password name=password placeholder="Password (min 12 chars)" minlength=12 required>
<select name=role><option value=user selected>user</option><option value=admin>admin</option></select>
<button type=submit class=primary>Create</button>
</form>
</div>
<div class=card>
<h2>Groups</h2>
<p style="font-size:12px;opacity:.7;margin:8px 0 0">A group grants access to only the checked tabs. An account with no group assigned (the default) is unrestricted -- creating and assigning a group is an opt-in way to <em>narrow</em> access, not something that happens automatically. Admins always have full access regardless of group.</p>
{group_rows}
<form method=post action="/admin/groups/create" class=add-form style="margin-top:14px">
<input type=text name=name placeholder="New group name (e.g. Editors)" required>
<button type=submit class=primary>Create group</button>
</form>
</div>
</div></div>
<script>
// Every form on this page posts to a state-changing route, so each needs the
// session's CSRF token. Tagging them once here rather than adding a hidden
// input to ~10 separate form templates -- and, more usefully, a form added
// later is covered automatically instead of silently 403ing.
(function(){{
  var token = {csrf_js!r};
  if(!token) return;
  document.querySelectorAll('form').forEach(function(f){{
    var i = document.createElement('input');
    i.type = 'hidden'; i.name = '_csrf'; i.value = token;
    f.appendChild(i);
  }});
}})()
</script>
</body></html>'''

@app.route('/admin/users')
def admin_users():
    # Direct navigation here (bookmarked, typed in, an old link) now redirects
    # into the app instead of showing the old standalone page -- user
    # management is only meant to be reached via Config > Users from here on.
    # The Config > Users iframe itself loads this exact URL with ?embed=1, so
    # that specific request still renders the real page; everything else
    # bounces to '/', where Config > Users is one click away.
    if request.args.get('embed') != '1':
        return redirect('/')
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

@app.route('/admin/users/<int:uid>/2fa/reset', methods=['POST'])
def admin_users_2fa_reset(uid):
    """Clears another account's 2FA entirely -- the lost-phone escape hatch.
    Admins can only turn it OFF, never enrol on someone's behalf: enrolling
    would mean an admin holding a secret that's meant to be the user's alone.
    The user then re-enrols themselves from Config > Security."""
    target = user_get(uid)
    if not target:
        return _admin_users_page(error='No such account.')
    user_totp_disable(uid)
    return _admin_users_page(notice=f"Two-factor authentication cleared for {target['username']} — "
                                    "they can set it up again from Config > Security.")

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

# ---- Admin: user management, JSON API ----
# Same underlying functions as the form-posting routes above, but returning
# JSON instead of a re-rendered page -- so Config > Users can be a normal
# fetch-driven tab embedded in the main app shell (matching every other
# Config tab: Network, Production, Security) instead of an <iframe> to a
# separate server-rendered page. Left the HTML routes above in place rather
# than deleting them -- no other caller depends on them, but there's no
# reason to break a bookmark or script hitting them directly either.
#
# All under /admin/ so _access_control's existing admin-only gate covers
# these automatically, same as everything else in this section.
def _user_json(u, current_uid):
    return {
        'id': u['id'], 'username': u['username'], 'role': u['role'],
        'is_active': bool(u['is_active']), 'group_id': u['group_id'],
        'group_name': u['group_name'], 'totp_enabled': bool(u['totp_enabled']),
        'created_at': u['created_at'], 'last_login': u['last_login'],
        'is_you': u['id'] == current_uid,
    }

def _group_json(g):
    return {'id': g['id'], 'name': g['name'], 'member_count': g['member_count'],
            'permissions': sorted(g['permissions'])}

@app.route('/admin/api/users')
def admin_api_users():
    return jsonify(ok=True,
                    users=[_user_json(u, session.get('user_id')) for u in user_list()],
                    groups=[_group_json(g) for g in group_list()],
                    permissions=[{'key': k, 'label': label} for k, label in AVAILABLE_PERMISSIONS],
                    current_user_id=session.get('user_id'))

@app.route('/admin/api/users/create', methods=['POST'])
def admin_api_users_create():
    d = request.get_json(silent=True) or {}
    ok, err = user_create(d.get('username', ''), d.get('password', ''), d.get('role', 'user'))
    return jsonify(ok=ok, error=None if ok else err, notice='User created.' if ok else None)

@app.route('/admin/api/users/<int:uid>/role', methods=['POST'])
def admin_api_users_role(uid):
    d = request.get_json(silent=True) or {}
    ok, err = user_set_role(uid, d.get('role', 'user'))
    return jsonify(ok=ok, error=None if ok else err, notice='Role updated.' if ok else None)

@app.route('/admin/api/users/<int:uid>/toggle', methods=['POST'])
def admin_api_users_toggle(uid):
    target = user_get(uid)
    if not target:
        return jsonify(ok=False, error='User not found.')
    ok, err = user_set_active(uid, not target['is_active'])
    return jsonify(ok=ok, error=None if ok else err, notice='Status updated.' if ok else None)

@app.route('/admin/api/users/<int:uid>/2fa/reset', methods=['POST'])
def admin_api_users_2fa_reset(uid):
    target = user_get(uid)
    if not target:
        return jsonify(ok=False, error='No such account.')
    user_totp_disable(uid)
    return jsonify(ok=True, notice=f"Two-factor authentication cleared for {target['username']} — "
                                    "they can set it up again from Config > Security.")

@app.route('/admin/api/users/<int:uid>/password', methods=['POST'])
def admin_api_users_password(uid):
    d = request.get_json(silent=True) or {}
    ok, err = user_set_password(uid, d.get('password', ''))
    return jsonify(ok=ok, error=None if ok else err, notice='Password updated.' if ok else None)

@app.route('/admin/api/users/<int:uid>/delete', methods=['POST'])
def admin_api_users_delete(uid):
    if uid == session.get('user_id'):
        return jsonify(ok=False, error="You can't delete your own account while signed in as it.")
    ok, err = user_delete(uid)
    return jsonify(ok=ok, error=None if ok else err, notice='User deleted.' if ok else None)

@app.route('/admin/api/users/<int:uid>/group', methods=['POST'])
def admin_api_users_group(uid):
    d = request.get_json(silent=True) or {}
    raw = str(d.get('group_id', '') or '').strip()
    gid = int(raw) if raw.isdigit() else None
    ok, err = user_set_group(uid, gid)
    return jsonify(ok=ok, error=None if ok else err, notice='Group updated.' if ok else None)

@app.route('/admin/api/groups/create', methods=['POST'])
def admin_api_groups_create():
    d = request.get_json(silent=True) or {}
    ok, err, _gid = group_create(d.get('name', ''))
    return jsonify(ok=ok, error=None if ok else err,
                    notice='Group created -- check its permissions below, then assign accounts to it.' if ok else None)

@app.route('/admin/api/groups/<int:gid>/permissions', methods=['POST'])
def admin_api_groups_permissions(gid):
    d = request.get_json(silent=True) or {}
    ok, err = group_set_permissions(gid, d.get('permissions') or [])
    return jsonify(ok=ok, error=None if ok else err, notice='Permissions saved.' if ok else None)

@app.route('/admin/api/groups/<int:gid>/delete', methods=['POST'])
def admin_api_groups_delete(gid):
    ok, err = group_delete(gid)
    return jsonify(ok=ok, error=None if ok else err,
                    notice='Group deleted. Its members are now unrestricted.' if ok else None)


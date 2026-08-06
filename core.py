"""App instance, session/cookie config, and the request-level access gate.

Everything else in this app (auth, the trailer library, the media pipeline)
imports `app` from here. Kept dependency-free of the other modules so there's
no risk of circular imports.
"""
import os, cv2, numpy as np, tempfile, threading, time, pathlib, base64, json, requests, subprocess, shutil, re, sqlite3, uuid, secrets, hmac, io
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
from flask import Flask, render_template_string, request, send_from_directory, jsonify, Response, session, redirect, url_for
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import smbclient  # pip install smbprotocol -- lets the upload panels browse a Windows/SMB network share directly

# Loads .env (if present) into the real process environment, BEFORE any of
# this file's own os.environ.get() calls below and before any other module
# gets imported -- so LIBRARY_DIR, ADMIN_PASSWORD, FISH_AUDIO_URL, etc. in a
# .env file next to this script take effect the same as if they'd been
# exported in the shell. Silently does nothing if python-dotenv isn't
# installed or there's no .env file, so this is optional, not required --
# every var can still be set the old way (real env vars / your process
# manager) with no .env file at all.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024 * 1024  # 5GB

# ---- Session secret ----
# Flask needs this to sign the session cookie. Generating it fresh at boot would
# silently invalidate every session (and log everyone out) on each pm2 restart,
# so it's persisted to disk the first time the app runs and reused after that.
_SECRET_KEY_FILE = os.environ.get('SECRET_KEY_FILE',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '.secret_key'))
try:
    with open(_SECRET_KEY_FILE, 'rb') as _f:
        app.secret_key = _f.read().strip()
    if not app.secret_key:
        raise ValueError('empty key file')
except (FileNotFoundError, ValueError):
    app.secret_key = secrets.token_hex(32).encode()
    try:
        with open(_SECRET_KEY_FILE, 'wb') as _f:
            _f.write(app.secret_key)
        try:
            os.chmod(_SECRET_KEY_FILE, 0o600)
        except OSError:
            pass
    except OSError as e:
        print(f'Could not persist {_SECRET_KEY_FILE} ({e}); sessions will not '
              'survive a restart until this is writable.')
app.config['PERMANENT_SESSION_LIFETIME'] = int(os.environ.get('SESSION_LIFETIME_DAYS', 30)) * 86400
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
# Only set Secure on the cookie if the app is actually served over HTTPS -- on a
# plain-HTTP LAN deployment (this app's default) a Secure cookie would never be
# sent at all, silently breaking the gate rather than protecting anything.
app.config['SESSION_COOKIE_SECURE'] = os.environ.get('FORCE_HTTPS', '').lower() in ('1', 'true', 'yes')

# ---- Access gate ----
# Per-user accounts (see users_db_init() below) replace the old single shared
# passphrase -- every request needs a signed-in session, and /admin/users is
# further restricted to the 'admin' role. This closes the standing hole where
# every trailer, template asset, and upload was servable to anyone who could
# reach the port, via predictable filenames and sequential library IDs.
_PUBLIC_PATHS = {'/login', '/logout', '/branding/logo', '/branding/favicon'}
_API_PREFIXES = ('/api/', '/uploads/', '/library/', '/download/')
_ADMIN_PREFIX = '/admin/'

def _client_ip():
    """Best-effort client IP for rate limiting.

    Trusts X-Forwarded-For only because this app is documented as running
    behind an Apache reverse proxy (see R.I.M.S notes); on a direct deployment
    that header is attacker-controlled and this falls back to remote_addr."""
    fwd = request.headers.get('X-Forwarded-For', '')
    return (fwd.split(',')[0].strip() if fwd else None) or request.remote_addr or 'unknown'

class _RateLimiter:
    """Simple in-memory sliding-window limiter: `limit` events per `window`
    seconds, per key. No external dependency (no redis) since this is a
    single-process app; resets on restart, which is an acceptable trade-off
    for slowing enumeration/brute-force rather than a hard guarantee."""
    def __init__(self, limit, window):
        self.limit = limit
        self.window = window
        self.buckets = {}
        self.lock = threading.Lock()

    def allow(self, key):
        now = time.time()
        with self.lock:
            dq = self.buckets.setdefault(key, deque())
            while dq and now - dq[0] > self.window:
                dq.popleft()
            if len(dq) >= self.limit:
                return False
            dq.append(now)
            # Bound memory: drop buckets nobody has touched recently.
            if len(self.buckets) > 5000:
                stale = [k for k, v in self.buckets.items() if not v or now - v[-1] > self.window * 4]
                for k in stale:
                    self.buckets.pop(k, None)
            return True

# Always on, regardless of whether the passphrase gate is configured: an
# unauthenticated deployment still shouldn't let one client rip through
# sequential library IDs or guessed upload filenames at full speed.
_file_route_limiter = _RateLimiter(limit=int(os.environ.get('FILE_ROUTE_RATE_LIMIT', 40)), window=60)
_login_limiter = _RateLimiter(limit=int(os.environ.get('LOGIN_RATE_LIMIT', 8)), window=300)
# Fixed dummy hash checked when a username doesn't exist, so a login attempt
# against a bad username costs the same CPU time as one against a real user
# with a wrong password (see login()).
_DUMMY_PW_HASH = generate_password_hash(secrets.token_hex(16))
# Fallback admin password used when ADMIN_PASSWORD isn't set in the
# environment (first-run bootstrap and RESET_ADMIN both fall back to this).
# Hardcoded at JUN's request for convenience -- this is a real security
# tradeoff: anyone with read access to this file (git history, a backup, a
# shared drive) effectively has admin access to the app. Prefer setting
# ADMIN_PASSWORD in the environment instead where possible, and change this
# account's password from Admin > Users right after first login either way.
DEFAULT_ADMIN_PASSWORD = 'Aimp#Admin2026!'

@app.before_request
def _access_control():
    path = request.path
    if path in _PUBLIC_PATHS or path.startswith('/static/'):
        return None

    # Baseline throttle on the direct file/asset routes -- applies even with the
    # gate disabled, since that's the default and these are exactly the routes a
    # scanner would hit to enumerate trailers or templates.
    if path.startswith(('/uploads/', '/library/', '/download/')) or '/asset/' in path:
        if not _file_route_limiter.allow(_client_ip()):
            return jsonify(error='Too many requests. Slow down.'), 429

    if session.get('authed'):
        if path.startswith(_ADMIN_PREFIX) and session.get('role') != 'admin':
            if path.startswith('/api/'):
                return jsonify(error='Admin access required.'), 403
            return redirect('/')
        return None

    if path.startswith(_API_PREFIXES) or path.startswith(_ADMIN_PREFIX):
        return jsonify(error='Not authenticated. Sign in at /login.'), 401
    return redirect(url_for('login', next=request.full_path if request.query_string else path))


# Shared upload-validation constant (kept here rather than in a specific
# feature module since several modules -- pipeline, network browsing --
# all validate against it).
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm'}

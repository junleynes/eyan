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

# ---- Reverse-proxy awareness ----
# This deployment terminates TLS at a reverse proxy, so from Flask's own point
# of view every request arrives as plain HTTP. Without this, request.is_secure
# is always False (so HSTS would never be sent) and _client_ip() sees the
# proxy's address for every request -- which would quietly collapse the
# per-IP rate limiters into one shared bucket for the whole internet.
#
# Only enabled when TRUST_PROXY_HEADERS is set, because X-Forwarded-* headers
# are trivially forged by a client talking to the app DIRECTLY. Trusting them
# unconditionally would let anyone spoof their source IP and sidestep rate
# limiting entirely. Turn this on only when the app is genuinely reachable
# solely through a proxy that overwrites those headers.
if os.environ.get('TRUST_PROXY_HEADERS', '').lower() in ('1', 'true', 'yes'):
    from werkzeug.middleware.proxy_fix import ProxyFix
    _proxy_hops = int(os.environ.get('TRUST_PROXY_HOPS', 1))
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=_proxy_hops, x_proto=_proxy_hops,
                             x_host=_proxy_hops, x_prefix=_proxy_hops)
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
# Sessions now default to 1 day rather than 30. A month-long session is a
# month-long window for a stolen cookie or an unattended browser to stay
# valid, which is a poor trade for a tool that people sign into from a
# workstation each shift. Still configurable for a LAN-only deployment where
# the convenience genuinely outweighs it.
app.config['PERMANENT_SESSION_LIFETIME'] = int(os.environ.get('SESSION_LIFETIME_DAYS', 1)) * 86400
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

# Render jobs are expensive (ffmpeg, AI scoring, potentially minutes of CPU
# each), so an authenticated client -- or a stolen session -- queueing them in
# a loop is a cheap self-inflicted denial of service. This is separate from
# MAX_CONCURRENT_JOBS, which caps how many run AT ONCE but happily accepts an
# unbounded queue behind it.
_job_submit_limiter = _RateLimiter(limit=int(os.environ.get('JOB_SUBMIT_RATE_LIMIT', 12)), window=300)
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

@app.after_request
def _security_headers(resp):
    """Defence-in-depth response headers. None of these fix a vulnerability on
    their own -- they limit what an attacker can do with one that exists.

    A note on the CSP, so nobody reads it as stronger than it is: this app
    uses ~98 inline onclick handlers and extensive inline styles, so
    'unsafe-inline' is required for script-src and style-src. That
    significantly weakens CSP's XSS protection -- a genuinely strict policy
    would need those refactored into addEventListener bindings and CSS
    classes first, which is a large change and not one to make quietly
    alongside security headers. What this policy still buys: default-src
    'self' blocks loading scripts/objects from arbitrary external origins,
    frame-ancestors blocks clickjacking, and form-action stops a injected
    form posting credentials off-site.

    frame-ancestors/X-Frame-Options are SAMEORIGIN rather than DENY because
    Config > Users legitimately embeds /admin/users in an iframe."""
    resp.headers.setdefault('X-Content-Type-Options', 'nosniff')
    resp.headers.setdefault('X-Frame-Options', 'SAMEORIGIN')
    resp.headers.setdefault('Referrer-Policy', 'same-origin')
    # Nothing here needs a camera, mic, or geolocation -- deny them outright so
    # an injected script can't prompt for them.
    resp.headers.setdefault('Permissions-Policy', 'geolocation=(), microphone=(), camera=(), payment=()')
    resp.headers.setdefault('Content-Security-Policy', '; '.join([
        "default-src 'self'",
        # data: covers the 2FA QR code and the inline SVG favicon; blob: covers
        # locally-previewed media before upload.
        "img-src 'self' data: blob:",
        "media-src 'self' blob:",
        "script-src 'self' 'unsafe-inline'",
        # Google Fonts serves the stylesheet from googleapis and the actual
        # font files from gstatic -- both are needed or the UI falls back to
        # system fonts.
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
        "font-src 'self' https://fonts.gstatic.com data:",
        "connect-src 'self'",
        "object-src 'none'",
        "base-uri 'self'",
        "form-action 'self'",
        "frame-ancestors 'self'",
    ]))
    # HSTS only when the request actually arrived over HTTPS. Sending it on a
    # plain-HTTP LAN deployment would tell browsers to force HTTPS against a
    # server that may not speak it, locking users out. request.is_secure
    # honours X-Forwarded-Proto when running behind a reverse proxy that sets
    # it, which is this deployment's setup.
    if request.is_secure:
        resp.headers.setdefault('Strict-Transport-Security', 'max-age=31536000; includeSubDomains')
    return resp

def ensure_csrf_token():
    """Returns this session's CSRF token, creating one if it doesn't have a
    token yet. Called when rendering any page that will make state-changing
    requests, so sessions established before this feature existed pick one up
    on their next page load rather than being locked out of every POST until
    they sign in again."""
    tok = session.get('csrf_token')
    if not tok:
        tok = secrets.token_urlsafe(32)
        session['csrf_token'] = tok
    return tok

@app.before_request
def _csrf_protect():
    """Rejects state-changing requests that don't carry this session's CSRF
    token. SameSite=Lax already blocks the classic cross-site form POST, but
    it isn't complete coverage -- it doesn't help against an attack originating
    from the same site (a subdomain, or any injected content), and older
    browsers vary in how they enforce it. This is the actual control.

    The token is bound to the session, so it's only obtainable by someone who
    already has the session -- which is the point.

    Accepted from either the X-CSRF-Token header (used by the app's fetch()
    calls, added centrally in the page's JS) or a _csrf form field (used by the
    plain HTML forms on the admin users page). GET/HEAD/OPTIONS are exempt as
    they shouldn't change state; /login is exempt because there's no session
    to carry a token yet at that point, and it has its own rate limiting and
    lockout."""
    if request.method in ('GET', 'HEAD', 'OPTIONS'):
        return None
    if request.path in ('/login', '/logout'):
        return None
    if not session.get('authed'):
        # Unauthenticated POSTs are already rejected by _access_control below;
        # no session means no token to compare against anyway.
        return None
    sent = request.headers.get('X-CSRF-Token') or request.form.get('_csrf') or ''
    expected = session.get('csrf_token') or ''
    # compare_digest rather than == so a token isn't recoverable a character at
    # a time by timing the comparison.
    if not expected or not hmac.compare_digest(str(sent), str(expected)):
        return jsonify(ok=False, error='Your session token was missing or stale. Reload the page and try again.'), 403
    return None

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

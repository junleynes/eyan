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
import smbclient

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = Flask(__name__)

if os.environ.get('TRUST_PROXY_HEADERS', '').lower() in ('1', 'true', 'yes'):
    from werkzeug.middleware.proxy_fix import ProxyFix
    _proxy_hops = int(os.environ.get('TRUST_PROXY_HOPS', 1))
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=_proxy_hops, x_proto=_proxy_hops,
                             x_host=_proxy_hops, x_prefix=_proxy_hops)
app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024 * 1024

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
        print(f'Could not persist {_SECRET_KEY_FILE} ({e}); sessions will not survive a restart until this is writable.')
app.config['PERMANENT_SESSION_LIFETIME'] = int(os.environ.get('SESSION_LIFETIME_DAYS', 1)) * 86400
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = os.environ.get('FORCE_HTTPS', '').lower() in ('1', 'true', 'yes')

_PUBLIC_PATHS = {'/', '/login', '/logout', '/register', '/branding/logo', '/branding/logo-dark', '/branding/favicon', '/branding/logo-mark'}
_API_PREFIXES = ('/api/', '/uploads/', '/library/', '/download/')
_ADMIN_PREFIX = '/admin/'

def _client_ip():
    fwd = request.headers.get('X-Forwarded-For', '')
    return (fwd.split(',')[0].strip() if fwd else None) or request.remote_addr or 'unknown'

class _RateLimiter:
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
            if len(self.buckets) > 5000:
                stale = [k for k, v in self.buckets.items() if not v or now - v[-1] > self.window * 4]
                for k in stale:
                    self.buckets.pop(k, None)
            return True

_file_route_limiter = _RateLimiter(limit=int(os.environ.get('FILE_ROUTE_RATE_LIMIT', 40)), window=60)
_job_submit_limiter = _RateLimiter(limit=int(os.environ.get('JOB_SUBMIT_RATE_LIMIT', 12)), window=300)
_login_limiter = _RateLimiter(limit=int(os.environ.get('LOGIN_RATE_LIMIT', 8)), window=300)
_DUMMY_PW_HASH = generate_password_hash(secrets.token_hex(16))
DEFAULT_ADMIN_PASSWORD = 'Aimp#Admin2026!'

@app.after_request
def _security_headers(resp):
    # Inject one shared, versioned stylesheet into every HTML page. Keeping
    # this at the app layer lets the login page and studio shell share the
    # same visual language without duplicating large CSS blocks in templates.
    if resp.content_type and resp.content_type.startswith('text/html'):
        try:
            body = resp.get_data(as_text=True)
            if '</head>' in body and '/static/ui-modern.css' not in body:
                body = body.replace('</head>', '<link rel="stylesheet" href="/static/ui-modern.css?v=7">\n</head>', 1)
                resp.set_data(body)
        except Exception:
            pass
    resp.headers.setdefault('X-Content-Type-Options', 'nosniff')
    resp.headers.setdefault('X-Frame-Options', 'SAMEORIGIN')
    resp.headers.setdefault('Referrer-Policy', 'same-origin')
    resp.headers.setdefault('Permissions-Policy', 'geolocation=(), microphone=(), camera=(), payment=()')
    resp.headers.setdefault('Content-Security-Policy', '; '.join([
        "default-src 'self'", "img-src 'self' data: blob:", "media-src 'self' blob:",
        "script-src 'self' 'unsafe-inline'", "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
        "font-src 'self' https://fonts.gstatic.com data:", "connect-src 'self'", "object-src 'none'",
        "base-uri 'self'", "form-action 'self'", "frame-ancestors 'self'",
    ]))
    if request.is_secure:
        resp.headers.setdefault('Strict-Transport-Security', 'max-age=31536000; includeSubDomains')
    return resp

def ensure_csrf_token():
    tok = session.get('csrf_token')
    if not tok:
        tok = secrets.token_urlsafe(32)
        session['csrf_token'] = tok
    return tok

@app.before_request
def _csrf_protect():
    if request.method in ('GET', 'HEAD', 'OPTIONS') or request.path in ('/login', '/logout'):
        return None
    if not session.get('authed'):
        return None
    sent = request.headers.get('X-CSRF-Token') or request.form.get('_csrf') or ''
    expected = session.get('csrf_token') or ''
    if not expected or not hmac.compare_digest(str(sent), str(expected)):
        return jsonify(ok=False, error='Your session token was missing or stale. Reload the page and try again.'), 403
    return None

@app.before_request
def _access_control():
    path = request.path
    if path in _PUBLIC_PATHS or path.startswith('/static/'):
        return None
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

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm'}

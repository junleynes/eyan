"""The media/trailer-generation engine: scene detection, AI vision scoring,
TTS/STT, music & SFX generation, the ffmpeg render pipeline, per-show asset
templates, and their API routes.

This is intentionally kept as one module rather than split further. Internally
it has several places (live-reloadable AI service URLs edited from the Config
tab, a shared face-detector instance, a structured-output-support flag) that
use `global` to let a nested closure deep in one route update state that
other routes read on their next call -- moving those definitions and their
readers into different files would silently break the live-reload behavior
unless every read site were rewritten to a qualified module.attr lookup.
Splitting *within* this module further is a reasonable next step, but it's
safer done incrementally against a real render on the actual server, where
the AI services, ffmpeg, and real footage can actually be exercised --
not blind in a sandbox that has none of those available.
"""
import os, cv2, numpy as np, tempfile, threading, time, pathlib, base64, json, requests, subprocess, shutil, re, sqlite3, uuid, secrets, io, mimetypes
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector, AdaptiveDetector
from flask import request, jsonify, redirect, url_for, Response, send_from_directory, session
from werkzeug.utils import secure_filename
import smbclient  # pip install smbprotocol -- lets the upload panels browse a Windows/SMB network share directly

from core import app, ALLOWED_EXTENSIONS
from library_db import (LIBRARY_DIR, _sqlite_connect, library_add, library_list, library_get_row, library_delete,
    load_branding, save_branding_text, save_branding_color, save_branding_logo, save_branding_favicon,
    clear_branding_logo, clear_branding_favicon, clear_branding_color, BRANDING_DIR)
from auth import require_permission

# ---- Per-show asset templates (SQLite) ----
# A "template" is a named bundle of the reusable assets a specific show always
# uses -- its background music bed, SFX one-shot, VO track, title card and end
# card (each card with an optional card VO + in/out points). Picking a template
# at generate time fills all of those slots at once, as an alternative to picking
# a Genre (which only presets transition style and the AI music/SFX *prompts*,
# and supplies no actual files).
#
# Like LIBRARY_DIR above, this has to survive process restarts -- UPLOAD_FOLDER is
# a fresh tempdir every boot -- so the master copy of each asset lives here and is
# only ever *copied* into UPLOAD_FOLDER per job (see template_stage_asset). That
# copy is mandatory, not defensive: _run_trailer_job() deletes the SFX, VO and
# card-VO files it is handed once it's finished with them.
TEMPLATES_DIR = os.environ.get('TEMPLATES_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'show_templates'))
os.makedirs(TEMPLATES_DIR, exist_ok=True)
TEMPLATES_DB_PATH = os.path.join(TEMPLATES_DIR, 'templates.db')

# Each slot maps a template asset to the form field the trailer form already uses
# for it. Note the two legacy field names: 'end_card_video' is really the *title*
# card and 'schedule_video' is really the *end* card -- the slot keys below use the
# accurate names, and the mapping is kept in this one place so nothing else has to
# know about the mismatch.
TEMPLATE_SLOTS = {
    'bgm':           {'field': 'scoring_audio',  'kind': 'audio', 'label': 'Background music'},
    'sfx':           {'field': 'sfx_upload',     'kind': 'audio', 'label': 'SFX one-shot'},
    'vo':            {'field': 'vo_upload',      'kind': 'audio', 'label': 'Voiceover'},
    'title_card':    {'field': 'end_card_video', 'kind': 'video', 'label': 'Title card'},
    'title_card_vo': {'field': 'title_card_vo',  'kind': 'audio', 'label': 'Title card VO'},
    'end_card':      {'field': 'schedule_video', 'kind': 'video', 'label': 'End card'},
    'end_card_vo':   {'field': 'end_card_vo',    'kind': 'audio', 'label': 'End card VO'},
}
TEMPLATE_SLOT_KEYS = list(TEMPLATE_SLOTS.keys())

# A template is the whole configuration for a show, not just its files: genre and
# transition, the asset slots above, and every other generator setting. These are
# captured verbatim from the generate form into a single settings_json column
# rather than one column each, so adding a control to the form doesn't need a
# schema migration.
TEMPLATE_SETTING_FIELDS = [
    'genre', 'transition', 'xfade_dur', 'transition_matte',
    'trailer_length', 'max_scene_dur', 'scene_threshold', 'min_scene_len_sec',
    'detector', 'adaptive_threshold',
    'mode', 'model', 'prompt',
    'scoring_mode', 'sfx_mode', 'sfx_source', 'vo_mode', 'vo_engine',
    'vo_voice', 'vo_language', 'vo_rate', 'vo_start', 'vo_volume',
    'vo_trim_start', 'vo_trim_end', 'vo_text',
    'title_card_vo_start', 'title_card_vo_end',
    'end_card_vo_start', 'end_card_vo_end',
    'target_loudness', 'true_peak', 'music_duck_db', 'duck_depth_db',
    'duck_release_hold', 'beat_match', 'broadcast_stereo',
    'sync_beats', 'whisper_enhance',
]
# Checkbox-style fields: absent from a form POST means "off", so they must be
# recorded as off rather than left at whatever the previous value was.
TEMPLATE_BOOL_FIELDS = {'beat_match', 'broadcast_stereo', 'sync_beats', 'whisper_enhance'}

def _tpl_db():
    return _sqlite_connect(TEMPLATES_DB_PATH)

def templates_db_init():
    cols = []
    for slot in TEMPLATE_SLOT_KEYS:
        cols.append(f'{slot}_file TEXT')   # stored filename inside TEMPLATES_DIR
        cols.append(f'{slot}_name TEXT')   # original filename, for display only
    conn = _tpl_db()
    conn.execute('CREATE TABLE IF NOT EXISTS show_templates ('
                 'id INTEGER PRIMARY KEY AUTOINCREMENT,'
                 'name TEXT NOT NULL UNIQUE,'
                 'notes TEXT,'
                 'created_at REAL NOT NULL,'
                 'updated_at REAL NOT NULL,'
                 'genre TEXT,'
                 'transition TEXT,'
                 'xfade_dur REAL,'
                 'settings_json TEXT,'
                 'vo_start REAL, vo_volume REAL, vo_trim_start REAL, vo_trim_end REAL,'
                 'title_card_vo_start REAL, title_card_vo_end REAL,'
                 'end_card_vo_start REAL, end_card_vo_end REAL,'
                 + ','.join(cols) + ')')
    # Migration for databases created before settings_json existed.
    have = {r[1] for r in conn.execute('PRAGMA table_info(show_templates)')}
    if 'settings_json' not in have:
        conn.execute('ALTER TABLE show_templates ADD COLUMN settings_json TEXT')
        print('show_templates: added settings_json column (existing templates keep their assets).')
    conn.commit()
    conn.close()

def template_settings(row):
    """The saved form configuration for a template, as a plain dict."""
    if not row:
        return {}
    try:
        return json.loads(row.get('settings_json') or '{}')
    except (ValueError, TypeError):
        return {}

def _template_public(row):
    """Shapes a DB row for the UI: hides on-disk filenames, exposes which slots are
    actually filled plus the display name of each."""
    out = {k: row.get(k) for k in ('id', 'name', 'notes', 'created_at', 'updated_at', 'genre',
                                   'transition', 'xfade_dur', 'vo_start', 'vo_volume',
                                   'vo_trim_start', 'vo_trim_end',
                                   'title_card_vo_start', 'title_card_vo_end',
                                   'end_card_vo_start', 'end_card_vo_end')}
    out['slots'] = {slot: {'filled': bool(row.get(f'{slot}_file')),
                           'name': row.get(f'{slot}_name'),
                           'label': TEMPLATE_SLOTS[slot]['label']}
                    for slot in TEMPLATE_SLOT_KEYS}
    out['settings'] = template_settings(row)
    return out

def template_list():
    conn = _tpl_db()
    rows = conn.execute('SELECT * FROM show_templates ORDER BY name COLLATE NOCASE').fetchall()
    conn.close()
    return [_template_public(dict(r)) for r in rows]

def template_get(tid):
    conn = _tpl_db()
    row = conn.execute('SELECT * FROM show_templates WHERE id=?', (tid,)).fetchone()
    conn.close()
    return dict(row) if row else None

def template_get_by_name(name):
    conn = _tpl_db()
    row = conn.execute('SELECT * FROM show_templates WHERE name=? COLLATE NOCASE', (name,)).fetchone()
    conn.close()
    return dict(row) if row else None

def template_store_asset(src_path, slot, orig_name=None):
    """Copies a resolved upload into TEMPLATES_DIR under a collision-proof name.
    Returns (stored_filename, display_name)."""
    ext = os.path.splitext(src_path)[1] or ''
    stored = f'{slot}_{uuid.uuid4().hex}{ext}'
    shutil.copy2(src_path, os.path.join(TEMPLATES_DIR, stored))
    return stored, (orig_name or os.path.basename(src_path))

def template_asset_abspath(row, slot):
    fn = row.get(f'{slot}_file') if row else None
    if not fn:
        return None
    p = os.path.join(TEMPLATES_DIR, os.path.basename(fn))
    return p if os.path.exists(p) else None

def template_stage_asset(row, slot):
    """Copies the template's master asset for `slot` into UPLOAD_FOLDER and returns
    the job-local path, or None if that slot is empty / missing on disk.

    The copy is REQUIRED -- _run_trailer_job() deletes sfx_upload_path,
    vo_upload_path, title_card_vo_path and end_card_vo_path once it's done with
    them, so handing it a TEMPLATES_DIR path directly would destroy the template
    after a single use."""
    src = template_asset_abspath(row, slot)
    if not src:
        return None
    ext = os.path.splitext(src)[1] or ''
    dest = os.path.join(app.config['UPLOAD_FOLDER'],
                        f'tpl{row["id"]}_{slot}_{int(time.time()*1000)}_{threading.get_ident()}{ext}')
    shutil.copy2(src, dest)
    return dest

def template_delete_asset_file(row, slot):
    p = template_asset_abspath(row, slot)
    if p:
        try:
            os.remove(p)
        except OSError:
            pass

def template_delete(tid):
    row = template_get(tid)
    if not row:
        return False
    for slot in TEMPLATE_SLOT_KEYS:
        template_delete_asset_file(row, slot)
    conn = _tpl_db()
    conn.execute('DELETE FROM show_templates WHERE id=?', (tid,))
    conn.commit()
    conn.close()
    return True

templates_db_init()

# ---- Pipeline stages (for the progress checklist in the UI) ----
# The job reports 16 distinct named steps but the UI used to collapse them into
# one bar and one line, so long stages looked frozen and short ones flew past.
# Each entry is (percent_at_start, label); the UI marks everything below the
# current percent as done and highlights the stage the job is actually in.
PIPELINE_STAGES = [
    (2,   'Reading video'),
    (8,   'Detecting cuts'),
    (15,  'Rating scenes'),
    (18,  'AI vision rating'),
    (28,  'Selecting scenes'),
    (38,  'Extracting clips'),
    (50,  'Transitions'),
    (58,  'Audio levels'),
    (62,  'Sound effects'),
    (80,  'Music'),
    (90,  'Narration'),
    (100, 'Done'),
]

# ---- Preview checkpoint ----
# A preview job runs the expensive analysis half of the pipeline (detect, score,
# select) and then STOPS, handing back the chosen cut with thumbnails. The user
# approves or drops scenes, then renders — and the render reuses this stored
# selection instead of redoing detection and AI scoring. Without it the only way
# to see what got picked was to sit through a full render.
PREVIEWS = {}
PREVIEWS_LOCK = threading.Lock()
PREVIEW_TTL = int(os.environ.get('PREVIEW_TTL', 2 * 3600))
# How many runner-up scenes a preview offers as swap-in alternatives ("show more").
PREVIEW_ALTERNATES = int(os.environ.get('PREVIEW_ALTERNATES', 12))

def preview_store(pid, data):
    with PREVIEWS_LOCK:
        now = time.time()
        for k in [k for k, v in PREVIEWS.items() if now - v.get('created', 0) > PREVIEW_TTL]:
            PREVIEWS.pop(k, None)
        data['created'] = now
        PREVIEWS[pid] = data

def preview_get(pid):
    with PREVIEWS_LOCK:
        p = PREVIEWS.get(pid)
        return dict(p) if p else None

ACE_STEP_URL = os.environ.get('ACE_STEP_URL', 'http://localhost:8001')
# Diffusion steps for music generation. The previous hardcoded 8 was far below
# ACE-Step's documented default and produced noticeably thin, smeared beds.
ACE_STEP_STEPS = int(os.environ.get('ACE_STEP_STEPS', 27))
ACE_STEP_NEGATIVE_PROMPT = os.environ.get(
    'ACE_STEP_NEGATIVE_PROMPT',
    'vocals, singing, voice, lyrics, choir, spoken word, low quality, distorted, clipping')
WOOSH_URL = os.environ.get('WOOSH_URL', 'http://localhost:8030')  # local API server for Sony AI's Woosh SFX foundation model (github.com/SonyResearch/Woosh); tried first for genre SFX, falls back to a procedural synth if unreachable (see woosh_sfx())
# Local speech-to-text service (e.g. fedirz/faster-whisper-server or any server exposing
# an OpenAI-compatible POST /v1/audio/transcriptions endpoint). Dialogue transcription
# calls out to this service instead of loading faster-whisper in-process, same as the
# other engines (Fish Audio, Ollama, ACE-Step, Woosh) — no local Python whisper
# dependency required. Override WHISPER_MODEL if your server needs a model name passed.
WHISPER_URL = os.environ.get('WHISPER_URL', 'http://localhost:8000')
WHISPER_MODEL = os.environ.get('WHISPER_MODEL', 'large-v2')
# Self-hosted Fish Audio S2 (SGLang-based server you run yourself) — no API key needed.
# To use Fish Audio's hosted cloud API instead, set FISH_AUDIO_URL=https://api.fish.audio/v1/tts
# and FISH_AUDIO_API_KEY to your key.
FISH_AUDIO_URL = os.environ.get('FISH_AUDIO_URL', 'http://localhost:8080/v1/tts')
FISH_AUDIO_API_KEY = os.environ.get('FISH_AUDIO_API_KEY', '')  # leave blank for a self-hosted server
FISH_AUDIO_MODEL = os.environ.get('FISH_AUDIO_MODEL', 's2.1-pro-free')  # ignored by most self-hosted servers, which just run whichever checkpoint they were started with

# ---- Fish Audio inline delivery tags ----
# Fish Audio interprets markers embedded in the narration text to control
# emotion, delivery and non-speech sounds. The syntax differs by generation:
# S2 uses [square brackets] and accepts free-form natural language; the older S1
# uses (parentheses) and a fixed tag set. FISH_TAG_STYLE picks which to emit --
# 'auto' infers it from FISH_AUDIO_MODEL, which is what a self-hosted server is
# usually named after.
FISH_TAG_STYLE = os.environ.get('FISH_TAG_STYLE', 'auto')

def fish_tag_syntax():
    """Returns ('[', ']') for S2-style tags or ('(', ')') for legacy S1."""
    style = (FISH_TAG_STYLE or 'auto').lower()
    if style in ('s1', 'paren', 'parentheses'):
        return '(', ')'
    if style in ('s2', 'bracket', 'brackets'):
        return '[', ']'
    model = (FISH_AUDIO_MODEL or '').lower()
    # Anything explicitly S1 gets parentheses; everything else (s2*, unknown,
    # blank) gets brackets, which is the current default generation.
    return ('(', ')') if ('s1' in model and 's2' not in model) else ('[', ']')

# Curated from Fish Audio's emotion-control reference. Not the full 64+ list --
# these are the ones that actually earn their place in a broadcast promo script;
# S2 accepts free-form descriptions anyway, so anything missing can be typed.
FISH_TAG_GROUPS = [
    ('Delivery', ['emphasis', 'whispering', 'shouting', 'soft tone',
                  'in a hurry tone', 'screaming']),
    ('Tone', ['confident', 'excited', 'calm', 'serious', 'friendly',
              'empathetic', 'curious', 'determined', 'hopeful', 'nostalgic',
              'sarcastic', 'proud']),
    ('Emotion', ['happy', 'sad', 'angry', 'scared', 'worried', 'surprised',
                 'frustrated', 'delighted', 'grateful', 'moved', 'relaxed',
                 'disappointed']),
    ('Pauses', ['break', 'long-break']),
    ('Sounds', ['laughing', 'chuckling', 'sighing', 'gasping', 'panting',
                'clear throat', 'audience laughing', 'crowd laughing']),
]

def fish_tag_catalogue():
    """Tag groups rendered in the syntax the configured model expects."""
    lo, hi = fish_tag_syntax()
    return {
        'open': lo, 'close': hi,
        'style': 's1' if lo == '(' else 's2',
        'model': FISH_AUDIO_MODEL,
        'groups': [{'name': name, 'tags': [{'name': t, 'tag': f'{lo}{t}{hi}'} for t in tags]}
                   for name, tags in FISH_TAG_GROUPS],
    }
# Reference WAV used for Fish Audio voice cloning (zero-shot): the narration voice is
# cloned from this sample on every request rather than requiring a pre-registered
# reference_id. Looked up next to this script by default; override with the env var.

# ---- Network (SMB) folder browsing — alternative to local file upload ----
# Lets the upload panels list and pull media straight from a Windows network
# share instead of requiring a local drag-and-drop/browse. Override any of
# these with env vars; not exposed through the Config tab since it holds a
# plaintext password.
NETWORK_SHARE_HOST = os.environ.get('NETWORK_SHARE_HOST', '')
NETWORK_SHARE_NAME = os.environ.get('NETWORK_SHARE_NAME', '')
NETWORK_SHARE_SUBDIR = os.environ.get('NETWORK_SHARE_SUBDIR', '')
NETWORK_SHARE_USERNAME = os.environ.get('NETWORK_SHARE_USERNAME', '')
NETWORK_SHARE_PASSWORD = os.environ.get('NETWORK_SHARE_PASSWORD', '')

AUDIO_EXTENSIONS = {'mp3', 'wav', 'm4a', 'flac', 'ogg', 'aac', 'wma'}

# Each browsable panel gets its own subfolder under the share root, and its own
# allowed extension set.
#   \\<share-host>\<share-name>\<subdir>\HIRES    -> raw video mats
#   \\<share-host>\<share-name>\<subdir>\TCARD    -> title cards
#   \\<share-host>\<share-name>\<subdir>\ENDCARD  -> end cards
#   \\<share-host>\<share-name>\<subdir>\MUSIC  -> background music
#   \\<share-host>\<share-name>\<subdir>\VO     -> narration / voiceover
#   \\<share-host>\<share-name>\<subdir>\SFX    -> sound effects
NETWORK_CATEGORIES = {
    'hires': {'folder': 'HIRES', 'exts': ALLOWED_EXTENSIONS, 'label': 'Video (HIRES)'},
    # Cards live in their own delivery folders, not with the raw video mats.
    # NOTE the legacy form-field names: 'end_card_video' is the TITLE card and
    # 'schedule_video' is the END card (see TEMPLATE_SLOTS for the same mapping).
    'tcard': {'folder': 'TCARD',   'exts': ALLOWED_EXTENSIONS, 'label': 'Title card (TCARD)'},
    'endcard': {'folder': 'ENDCARD', 'exts': ALLOWED_EXTENSIONS, 'label': 'End card (ENDCARD)'},
    'music': {'folder': 'MUSIC', 'exts': AUDIO_EXTENSIONS, 'label': 'Music'},
    'vo':    {'folder': 'VO',    'exts': AUDIO_EXTENSIONS, 'label': 'VO'},
    'sfx':   {'folder': 'SFX',   'exts': AUDIO_EXTENSIONS, 'label': 'SFX'},
}
DEFAULT_NETWORK_CATEGORY = 'hires'

def _network_category(category):
    """Validates/normalizes a category key, falling back to 'hires'."""
    return NETWORK_CATEGORIES.get(category, NETWORK_CATEGORIES[DEFAULT_NETWORK_CATEGORY])

def _network_share_root(category=DEFAULT_NETWORK_CATEGORY):
    """UNC path of the folder we browse for `category`, e.g.
    \\\\<share-host>\\<share-name>\\MUSIC"""
    cat = _network_category(category)
    root = f'\\\\{NETWORK_SHARE_HOST}\\{NETWORK_SHARE_NAME}'
    if NETWORK_SHARE_SUBDIR:
        root += f'\\{NETWORK_SHARE_SUBDIR}'
    if cat['folder']:
        root += f'\\{cat["folder"]}'
    return root

def _network_session():
    """(Re)registers the SMB session for the configured share. smbclient caches
    connections per-server, so calling this repeatedly is cheap once logged in."""
    smbclient.register_session(NETWORK_SHARE_HOST, username=NETWORK_SHARE_USERNAME,
                                password=NETWORK_SHARE_PASSWORD, connection_timeout=10)

def list_network_files(category=DEFAULT_NETWORK_CATEGORY):
    """Returns the files (name/size/modified) in the network folder for `category`,
    filtered to that category's allowed extensions."""
    cat = _network_category(category)
    _network_session()
    root = _network_share_root(category)
    out = []
    for entry in smbclient.scandir(root):
        if not entry.is_file():
            continue
        if not allowed_file(entry.name, cat['exts']):
            continue
        st = entry.stat()
        out.append({'name': entry.name, 'size': st.st_size, 'mtime': st.st_mtime})
    out.sort(key=lambda e: e['name'].lower())
    return root, out

def fetch_network_file(name, category=DEFAULT_NETWORK_CATEGORY):
    """Copies `name` from the network folder for `category` into UPLOAD_FOLDER and
    returns the local staged filename (prefixed net_<ts>_ so load_video() /
    _resolve_upload() can recognize and trust it)."""
    cat = _network_category(category)
    if os.path.basename(name) != name or not allowed_file(name, cat['exts']):
        raise ValueError('Invalid filename')
    _network_session()
    remote_path = _network_share_root(category) + '\\' + name
    local_name = f'net_{int(time.time())}_{secure_filename(name)}'
    local_path = os.path.join(app.config['UPLOAD_FOLDER'], local_name)
    with smbclient.open_file(remote_path, mode='rb') as rf, open(local_path, 'wb') as lf:
        shutil.copyfileobj(rf, lf)
    return local_name

# Back-compat aliases (old names, always the 'hires'/video category).
def list_network_videos():
    return list_network_files('hires')

def fetch_network_video(name):
    return fetch_network_file(name, 'hires')

# ---- Config tab persistence: lets the AI service URLs above be edited from the UI ----
# instead of only via environment variables. Overrides are saved to a small JSON file
# next to this script and re-applied on every startup, on top of the env-var defaults
# set above. Env vars still win at process start; the Config tab wins after that until
# the file is deleted or a value is cleared back to blank.
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ai_services_config.json')
# name -> (module-level global to update, human label, help text)
CONFIGURABLE_SERVICES = {
    'FISH_AUDIO_URL':    ('Fish Audio S2', 'Full TTS endpoint URL, e.g. http://host:8080/v1/tts'),
    'FISH_AUDIO_API_KEY':('Fish Audio API key', 'Only needed for the hosted fish.audio cloud API — leave blank for a self-hosted server'),
    'WHISPER_URL':       ('faster-whisper', 'Base server URL, e.g. http://localhost:8000'),
    'OLLAMA_URL':        ('Ollama', 'Base server URL, e.g. http://localhost:11434'),
    'ACE_STEP_URL':      ('ACE-Step', 'Base server URL, e.g. http://localhost:8001'),
    'WOOSH_URL':         ('Woosh', 'Base server URL, e.g. http://localhost:8030'),
}

def load_config_overrides():
    """Applies any saved Config-tab overrides on top of the env-var defaults above.
    Called once at startup, after every constant it might touch is already defined."""
    global FISH_AUDIO_URL, FISH_AUDIO_API_KEY, WHISPER_URL, OLLAMA_URL, ACE_STEP_URL, WOOSH_URL
    if not os.path.exists(CONFIG_FILE):
        return
    try:
        with open(CONFIG_FILE) as f:
            cfg = json.load(f)
    except Exception as e:
        print(f'Config file load error ({CONFIG_FILE}): {e}')
        return
    if 'FISH_AUDIO_URL' in cfg and cfg['FISH_AUDIO_URL']: FISH_AUDIO_URL = cfg['FISH_AUDIO_URL']
    if 'FISH_AUDIO_API_KEY' in cfg: FISH_AUDIO_API_KEY = cfg['FISH_AUDIO_API_KEY']
    if 'WHISPER_URL' in cfg and cfg['WHISPER_URL']: WHISPER_URL = cfg['WHISPER_URL']
    if 'OLLAMA_URL' in cfg and cfg['OLLAMA_URL']: OLLAMA_URL = cfg['OLLAMA_URL']
    if 'ACE_STEP_URL' in cfg and cfg['ACE_STEP_URL']: ACE_STEP_URL = cfg['ACE_STEP_URL']
    if 'WOOSH_URL' in cfg and cfg['WOOSH_URL']: WOOSH_URL = cfg['WOOSH_URL']

def save_config_overrides(updates):
    """Merges `updates` (dict of the CONFIGURABLE_SERVICES keys) into the config file
    and applies them to the live module globals immediately — no restart needed."""
    global FISH_AUDIO_URL, FISH_AUDIO_API_KEY, WHISPER_URL, OLLAMA_URL, ACE_STEP_URL, WOOSH_URL
    existing = {}
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE) as f:
                existing = json.load(f)
        except Exception:
            existing = {}
    existing.update({k: v for k, v in updates.items() if k in CONFIGURABLE_SERVICES})
    with open(CONFIG_FILE, 'w') as f:
        json.dump(existing, f, indent=2)
    if 'FISH_AUDIO_URL' in updates: FISH_AUDIO_URL = updates['FISH_AUDIO_URL'] or FISH_AUDIO_URL
    if 'FISH_AUDIO_API_KEY' in updates: FISH_AUDIO_API_KEY = updates['FISH_AUDIO_API_KEY']
    if 'WHISPER_URL' in updates: WHISPER_URL = updates['WHISPER_URL'] or WHISPER_URL
    if 'OLLAMA_URL' in updates: OLLAMA_URL = updates['OLLAMA_URL'] or OLLAMA_URL
    if 'ACE_STEP_URL' in updates: ACE_STEP_URL = updates['ACE_STEP_URL'] or ACE_STEP_URL
    if 'WOOSH_URL' in updates: WOOSH_URL = updates['WOOSH_URL'] or WOOSH_URL

def current_config_values():
    return {
        'FISH_AUDIO_URL': FISH_AUDIO_URL, 'FISH_AUDIO_API_KEY': FISH_AUDIO_API_KEY,
        'WHISPER_URL': WHISPER_URL,
        'OLLAMA_URL': OLLAMA_URL, 'ACE_STEP_URL': ACE_STEP_URL, 'WOOSH_URL': WOOSH_URL,
    }

# ---- Branding ----
# Moved to library_db.py (see there for load_branding/save_branding_text/etc.)
# so both this module's Config > Branding routes AND auth.py's login page can
# import it without a circular import -- auth.py already imports from
# library_db, and pipeline.py imports from auth, so branding couldn't live
# here if the login page needed it too. Re-imported below under the same
# names so nothing else in this file has to change.

# ---- Fish Audio S2 (fish.audio) — primary voiceover engine (self-hosted or cloud REST API) ----
# How long to wait for a TTS server to synthesize. The old hardcoded 30s was
# fine for a preview but too tight for a full narration script on a self-hosted
# CPU instance -- when it expired the VO was silently dropped and the trailer
# rendered mute, with the reason buried in the server log.
TTS_TIMEOUT = int(os.environ.get('TTS_TIMEOUT', 180))

def _looks_like_audio(data, content_type=''):
    """True if `data` starts with a container signature we'd expect from a TTS
    server. Used to reject the very common failure where a server answers HTTP
    200 with a JSON or HTML error body instead of audio -- without this check
    the error text gets written to disk as a .wav, passes a size>0 test, and is
    reported as a successful render right up until ffmpeg chokes on it later."""
    if not data:
        return False
    ct = (content_type or '').lower()
    if ct.startswith(('application/json', 'text/html', 'text/plain')):
        return False
    sigs = (
        b'RIFF',      # wav
        b'ID3',       # mp3 with tag
        b'OggS',      # ogg/opus
        b'fLaC',      # flac
        b'\xff\xfb', b'\xff\xf3', b'\xff\xf2',  # raw mp3 frame
        b'\xff\xf1', b'\xff\xf9',               # adts aac
    )
    if data[:4] in (b'RIFF', b'OggS', b'fLaC') or data[:3] == b'ID3':
        return True
    if any(data.startswith(s) for s in sigs):
        return True
    # ISO-BMFF (m4a/mp4): 'ftyp' at offset 4
    if len(data) > 12 and data[4:8] == b'ftyp':
        return True
    return False

def _write_tts_response(r, output_path, engine_label):
    """Validates a TTS HTTP response and writes it out. Returns (ok, error)."""
    if not r.ok:
        detail = (r.text or '')[:300]
        try:
            j = r.json()
            detail = j.get('error') or j.get('detail') or j.get('message') or detail
        except Exception:
            pass
        return False, f'{engine_label} API error {r.status_code}: {detail}'
    if not r.content:
        return False, f'{engine_label} returned an empty response'
    if not _looks_like_audio(r.content, r.headers.get('Content-Type', '')):
        # 200 OK but the payload isn't audio -- surface whatever the server
        # actually said rather than writing it to disk as a fake .wav.
        detail = ''
        try:
            j = r.json()
            detail = j.get('error') or j.get('detail') or j.get('message') or ''
        except Exception:
            detail = (r.text or '')[:200]
        return False, (f'{engine_label} returned a non-audio response'
                       + (f': {detail}' if detail else
                          f' (Content-Type: {r.headers.get("Content-Type", "unknown")})'))
    try:
        with open(output_path, 'wb') as f:
            f.write(r.content)
    except OSError as e:
        return False, f'{engine_label}: could not write output file: {e}'
    if not (os.path.exists(output_path) and os.path.getsize(output_path) > 0):
        return False, f'{engine_label} returned an empty response'
    return True, None

def fish_audio_tts(text, output_wav_path, voice_id=None, rate=175, reference_audio_path=None, language=None):
    """Generate a voiceover WAV using a Fish Audio S2-compatible TTS server (POST /v1/tts) —
    either your own self-hosted instance or Fish Audio's hosted cloud API. Returns
    (ok, error_message). `voice_id`, if set, is passed as `reference_id` to reuse a
    pre-registered voice. Otherwise, if `reference_audio_path` points at an existing WAV,
    that sample is base64-encoded and sent as a `references` entry so the server clones
    the voice from it directly (zero-shot cloning) — no pre-registration needed. `rate` is
    the same wpm-style value used by the UI (default 175); mapped onto Fish Audio's
    prosody.speed multiplier (1.0 = normal) so faster/slower selections keep the same
    behavior as before. Language does not need to be specified — S2 auto-detects it
    from the text (83 languages, including Tagalog)."""
    speed = max(0.5, min(2.0, (rate or 175) / 175.0))
    body = {
        'text': text,
        'format': 'wav',
        'prosody': {'speed': speed, 'volume': 0, 'normalize_loudness': True},
    }
    if voice_id:
        body['reference_id'] = voice_id
    elif reference_audio_path and os.path.exists(reference_audio_path):
        # Normalized to 16k mono PCM. The upload field
        # accepts any audio/*, so this is routinely an MP3/M4A or a 48k stereo
        # WAV; previously those bytes were base64-ed straight through as a
        # "reference WAV" and the server either mis-cloned or failed at synthesis.
        normalized_path = _normalize_reference_audio(reference_audio_path)
        try:
            with open(normalized_path, 'rb') as rf:
                ref_b64 = base64.b64encode(rf.read()).decode('ascii')
            body['references'] = [{'audio': ref_b64, 'text': ''}]
        except Exception as e:
            return False, f'Failed to read voice reference audio ({reference_audio_path}): {e}'
        finally:
            if normalized_path != reference_audio_path and os.path.exists(normalized_path):
                try:
                    os.remove(normalized_path)
                except OSError:
                    pass
    if language and language != 'auto':
        # Best-effort hint only: S2 normally auto-detects language from the text
        # itself, but passing it explicitly helps disambiguate short/ambiguous
        # scripts. A server that doesn't recognize this field just ignores it.
        body['language'] = language
    headers = {'Content-Type': 'application/json'}
    if FISH_AUDIO_API_KEY:
        # Only the hosted api.fish.audio endpoint needs these; a self-hosted server
        # with no key configured just ignores their absence.
        headers['Authorization'] = f'Bearer {FISH_AUDIO_API_KEY}'
        headers['model'] = FISH_AUDIO_MODEL
    try:
        r = requests.post(FISH_AUDIO_URL, headers=headers, json=body, timeout=TTS_TIMEOUT)
        return _write_tts_response(r, output_wav_path, 'Fish Audio')
    except Exception as e:
        return False, f'Fish Audio request failed: {e}'

# ---- Shared voice-clone reference handling ----
def _normalize_reference_audio(src_path):
    """Re-encodes a reference/voice-clone sample to 16kHz mono 16-bit PCM WAV.

    Used by BOTH engines. The upload field accepts any audio/*, so this is
    routinely an MP3, M4A or a 48k stereo WAV. Voice-cloning servers can usually
    fingerprint a voice from almost any input but then fail during the actual
    synthesis pass that expects a specific format -- which is exactly the shape
    of a 'voice is detected but can't generate speech' error. Returns the
    normalized path, or the original path unchanged if ffmpeg isn't available or
    the conversion fails (caller falls back to sending the original bytes rather
    than hard-failing here)."""
    if not FFMPEG or not os.path.exists(src_path):
        return src_path
    norm_path = os.path.join(tempfile.gettempdir(), f'ttsref_{uuid.uuid4().hex}.wav')
    try:
        r = subprocess.run([FFMPEG, '-y', '-i', src_path, '-ac', '1', '-ar', '16000',
                             '-sample_fmt', 's16', norm_path],
                            capture_output=True, text=True, timeout=60)
        if os.path.exists(norm_path) and os.path.getsize(norm_path) > 0:
            return norm_path
        print(f'TTS: reference audio normalization failed, sending original file: {r.stderr[-300:]}')
    except Exception as e:
        print(f'TTS: reference audio normalization error, sending original file: {e}')
    return src_path


# Language codes offered in the narration UI. Fish Audio S2 auto-detects the
# script's language, so this is an optional override rather than a requirement.
FISH_AUDIO_LANGUAGES = [
    {'code': 'auto', 'label': 'Auto-detect (recommended)'},
    {'code': 'en', 'label': 'English'},
    {'code': 'tl', 'label': 'Tagalog / Filipino'},
    {'code': 'zh', 'label': 'Chinese'},
    {'code': 'ja', 'label': 'Japanese'},
    {'code': 'ko', 'label': 'Korean'},
    {'code': 'es', 'label': 'Spanish'},
    {'code': 'fr', 'label': 'French'},
    {'code': 'de', 'label': 'German'},
    {'code': 'ar', 'label': 'Arabic'},
    {'code': 'pt', 'label': 'Portuguese'},
    {'code': 'ru', 'label': 'Russian'},
    {'code': 'id', 'label': 'Indonesian'},
    {'code': 'vi', 'label': 'Vietnamese'},
    {'code': 'th', 'label': 'Thai'},
]

# Voice lists are fetched from the TTS server and cached briefly: the dropdown is
# reloaded on several UI events, and a self-hosted server can be slow to answer.
_VOICES_CACHE_TTL = int(os.environ.get('VOICES_CACHE_TTL', 300))
_VOICES_CACHE = {
    'fish_audio': {'voices': None, 'source': 'none', 'error': None, 'fetched_at': 0.0},
}

def fish_audio_list_voices(force=False):
    """List voices registered for narration via Fish Audio: registered voice
    models from Fish Audio's cloud API (if FISH_AUDIO_API_KEY is set), or a
    best-effort probe of a self-hosted server's own model-listing endpoint if
    it has one. There's no fallback "default" entry — if nothing is
    registered/listable, this returns an empty voices list, and the caller is
    expected to fall back to "upload a reference sample" for zero-shot
    cloning instead. Returns (voices, source, error) where source is
    'cloud' | 'self_hosted' | 'none' | 'error', and each voice is
    {'id': <reference_id>, 'title': <display name>, 'languages': [...]}."""
    now = time.time()
    cache = _VOICES_CACHE['fish_audio']
    if not force and cache['voices'] is not None and now - cache['fetched_at'] < _VOICES_CACHE_TTL:
        return cache['voices'], cache['source'], cache['error']

    voices = []
    source = 'none'
    error = None

    if FISH_AUDIO_API_KEY:
        # Hosted cloud API: list voice models registered under this account.
        try:
            r = requests.get('https://api.fish.audio/model',
                              headers={'Authorization': f'Bearer {FISH_AUDIO_API_KEY}'},
                              params={'self_only': 'true', 'page_size': 100}, timeout=8)
            if r.ok:
                data = r.json()
                items = data.get('items', data if isinstance(data, list) else [])
                for it in items:
                    vid = it.get('_id') or it.get('id')
                    if not vid:
                        continue
                    voices.append({'id': vid, 'title': it.get('title') or vid,
                                    'languages': it.get('languages') or []})
                source = 'cloud'
            else:
                error = f'Fish Audio model list error {r.status_code}: {r.text[:200]}'
        except Exception as e:
            error = f'Fish Audio model list request failed: {e}'
    else:
        # Self-hosted probe. Fish Speech exposes its registered reference voices at
        # /v1/references/list, NOT /v1/models (which 404s) -- probing only the
        # latter silently produced an empty list, so the voice dropdown had nothing
        # in it and generate_tts() then refused to run for want of a voice_id.
        #
        # Two quirks that matter:
        #   * the endpoint defaults to application/msgpack, so Accept must ask for
        #     JSON explicitly or r.json() fails on binary content;
        #   * the response is {"reference_ids": ["name", ...]} -- bare strings, not
        #     the {"items": [{"id": ...}]} objects the cloud API returns.
        # Older/alternative builds are still tried afterwards, and both object and
        # string entries are accepted, so this works across server versions.
        base = FISH_AUDIO_URL.rsplit('/v1/', 1)[0] if '/v1/' in FISH_AUDIO_URL else FISH_AUDIO_URL
        base = base.rstrip('/')
        headers = {'Accept': 'application/json'}
        for path in ('/v1/references/list', '/v1/models'):
            try:
                r = requests.get(base + path, headers=headers, timeout=4)
            except Exception:
                continue  # server down or path refused — try the next one
            if not r.ok:
                continue
            try:
                data = r.json()
            except ValueError:
                # Still msgpack (or otherwise not JSON) despite the Accept header.
                error = (f'{path} returned {r.headers.get("Content-Type", "an unreadable format")} '
                         'rather than JSON, so its voice list could not be parsed.')
                continue
            if isinstance(data, dict):
                items = data.get('reference_ids') or data.get('items') or data.get('models') or []
            elif isinstance(data, list):
                items = data
            else:
                items = []
            for it in items:
                if isinstance(it, str):
                    # /v1/references/list form: the id IS the display name.
                    vid, title, langs = it, it, []
                elif isinstance(it, dict):
                    vid = it.get('id') or it.get('_id') or it.get('reference_id')
                    title = it.get('title') or it.get('name') or vid
                    langs = it.get('languages') or []
                else:
                    continue
                if vid:
                    voices.append({'id': vid, 'title': title, 'languages': langs})
            # Set outside the loop so an endpoint that exists but returns nothing
            # still reports 'self_hosted' rather than 'none', matching
            # a server that exists but lists nothing.
            source = 'self_hosted'
            if voices:
                error = None
                break

    _VOICES_CACHE['fish_audio'].update(voices=voices, source=source, error=error, fetched_at=now)
    return voices, source, error

def list_voices_for_engine(engine, force=False):
    """Voice list for the requested engine. Fish Audio is the only narration
    engine now; the parameter is kept so existing API callers and saved templates
    that still pass vo_engine keep working."""
    return fish_audio_list_voices(force=force)

def _fish_audio_self_hosted_base():
    """Server root derived from FISH_AUDIO_URL (which points at .../v1/tts),
    e.g. http://localhost:8080 -- shared by list/clone/delete so they always
    agree on which server they're talking to."""
    base = FISH_AUDIO_URL.rsplit('/v1/', 1)[0] if '/v1/' in FISH_AUDIO_URL else FISH_AUDIO_URL
    return base.rstrip('/')

def fish_audio_add_reference(voice_id, audio_path, text, timeout=60):
    """Registers a new named, reusable voice on a self-hosted Fish Speech
    server via POST /v1/references/add -- confirmed against this server's own
    /docs (multipart/form-data: id, audio, text), not assumed.

    Unlike the zero-shot reference-audio upload elsewhere in this app (attach
    a sample to a single TTS request, nothing saved), this one-time
    registration makes the voice show up in the Voice dropdown for everyone
    from then on, with no need to re-upload a sample per generation. `text`
    must be the reference audio's exact transcript -- the model uses it to
    align pronunciation to the audio, the same purpose as fish-speech's own
    .lab sidecar files.

    Only supports self-hosted servers (no FISH_AUDIO_API_KEY set). The hosted
    cloud API's voice-management endpoints are a different, separate surface
    this hasn't been verified against -- see fish_audio_list_voices' cloud
    branch for the one hosted endpoint that has been."""
    if FISH_AUDIO_API_KEY:
        return False, ('Voice registration isn\u2019t implemented for the hosted Fish Audio cloud API '
                       '(FISH_AUDIO_API_KEY is set) -- only for a self-hosted server. '
                       'Manage cloud voices at fish.audio directly.')
    voice_id = (voice_id or '').strip()
    text = (text or '').strip()
    if not voice_id:
        return False, 'Enter a name for this voice.'
    if not text:
        return False, 'Enter the exact transcript of the reference audio.'
    if not (audio_path and os.path.exists(audio_path)):
        return False, 'No reference audio file.'
    base = _fish_audio_self_hosted_base()
    try:
        with open(audio_path, 'rb') as f:
            r = requests.post(f'{base}/v1/references/add',
                              data={'id': voice_id, 'text': text},
                              files={'audio': (os.path.basename(audio_path), f,
                                               mimetypes.guess_type(audio_path)[0] or 'audio/wav')},
                              timeout=timeout)
    except Exception as e:
        return False, f'Could not reach Fish Audio at {base}: {e}'
    if not r.ok:
        detail = (r.text or '')[:300]
        try:
            j = r.json()
            detail = j.get('error') or j.get('detail') or j.get('message') or detail
        except Exception:
            pass
        return False, f'Fish Audio API error {r.status_code}: {detail}'
    return True, None

def fish_audio_delete_reference(voice_id, timeout=15):
    """Deletes a registered voice from a self-hosted Fish Speech server via
    DELETE /v1/references/delete. Confirmed against this server's own /docs:
    the request body is msgpack-encoded (Content-Type: application/msgpack),
    not JSON -- {'reference_id': voice_id} -- matching the same
    defaults-to-msgpack behavior fish_audio_list_voices already had to work
    around for /v1/references/list's response. Sent as real encoded msgpack
    bytes here, not JSON with a relabeled header, since there's no
    confirmation this endpoint accepts JSON at all.

    Only supports self-hosted servers, same reasoning as
    fish_audio_add_reference above."""
    if FISH_AUDIO_API_KEY:
        return False, ('Voice deletion isn\u2019t implemented for the hosted Fish Audio cloud API '
                       '(FISH_AUDIO_API_KEY is set) -- only for a self-hosted server. '
                       'Manage cloud voices at fish.audio directly.')
    voice_id = (voice_id or '').strip()
    if not voice_id:
        return False, 'No voice id given.'
    base = _fish_audio_self_hosted_base()
    try:
        import msgpack
        body = msgpack.packb({'reference_id': voice_id})
        r = requests.delete(f'{base}/v1/references/delete', data=body,
                            headers={'Content-Type': 'application/msgpack'}, timeout=timeout)
    except ImportError:
        return False, 'The msgpack package is required for voice deletion but is not installed (pip install msgpack).'
    except Exception as e:
        return False, f'Could not reach Fish Audio at {base}: {e}'
    if not r.ok:
        detail = (r.text or '')[:300]
        try:
            j = r.json()
            detail = j.get('error') or j.get('detail') or j.get('message') or detail
        except Exception:
            pass
        return False, f'Fish Audio API error {r.status_code}: {detail}'
    return True, None

# ---- Background job tracking (progress reporting for long-running trailer jobs) ----
JOBS = {}
JOBS_LOCK = threading.Lock()
JOB_TTL = 60 * 60  # drop finished jobs after an hour so JOBS doesn't grow forever

class JobCancelled(Exception):
    pass

def job_new(user_id=None, username=None):
    jid = f'{int(time.time()*1000)}_{threading.get_ident()}'
    with JOBS_LOCK:
        JOBS[jid] = {'percent': 0, 'step': 'Queued', 'done': False, 'error': None,
                     'result': None, 'created': time.time(), 'cancel_requested': False,
                     'status': 'queued', 'user_id': user_id, 'username': username}
        stale = [k for k, v in JOBS.items() if v.get('done') and time.time() - v.get('created', 0) > JOB_TTL]
        for k in stale:
            JOBS.pop(k, None)
    return jid

def job_set(jid, percent=None, step=None, error=None, done=None, result=None, status=None):
    cancel_now = False
    with JOBS_LOCK:
        j = JOBS.get(jid)
        if not j:
            return
        if percent is not None:
            j['percent'] = percent
        if step is not None:
            j['step'] = step
        if status is not None:
            j['status'] = status
        if error is not None:
            j['error'] = error
            j['done'] = True
            j['status'] = 'error'
        if done is not None:
            j['done'] = done
        if result is not None:
            j['result'] = result
        # Any progress update after a cancellation request raises, so the running
        # job unwinds at its next checkpoint — this call itself (reporting the
        # cancellation) is exempt so it doesn't recursively raise.
        if j.get('cancel_requested') and error is None and not done:
            cancel_now = True
    if cancel_now:
        raise JobCancelled(jid)

def job_cancel(jid):
    """Request cancellation of a queued or running job. Queued jobs are removed
    from the wait line immediately; running jobs unwind at their next progress
    checkpoint (best-effort — an in-flight ffmpeg/API call still finishes first)."""
    with JOBS_LOCK:
        j = JOBS.get(jid)
        if not j or j.get('done'):
            return False
        j['cancel_requested'] = True
    with JOB_QUEUE_LOCK:
        if jid in JOB_QUEUE:
            JOB_QUEUE.remove(jid)
            job_set(jid, error='Cancelled', status='cancelled')
    return True

def job_get(jid):
    with JOBS_LOCK:
        j = JOBS.get(jid)
        return dict(j) if j else None

def _owns_or_admin(owner_user_id):
    """True if the current session is the owner of a job/trailer, or an
    admin. Admins see and manage everything; everyone else sees only their
    own. A record with no owner (user_id=None -- either created before this
    app had accounts, or something went wrong capturing it) is treated as
    admin-only rather than shown to whoever happens to ask, since there's no
    one it can honestly be attributed to."""
    if session.get('role') == 'admin':
        return True
    uid = session.get('user_id')
    return uid is not None and owner_user_id == uid

class JobGate:
    """Caps how many trailer jobs run at once. Limit is adjustable at runtime
    (e.g. via the /api/queue/limit endpoint) without needing to restart anything —
    waiting jobs just re-check the current limit each time they're woken up."""
    def __init__(self, limit):
        self.limit = max(1, limit)
        self.running = 0
        self.cond = threading.Condition()

    def set_limit(self, new_limit):
        with self.cond:
            self.limit = max(1, int(new_limit))
            self.cond.notify_all()

    def status(self):
        with self.cond:
            return {'running': self.running, 'limit': self.limit}

MAX_CONCURRENT_JOBS = int(os.environ.get('MAX_CONCURRENT_JOBS', 2))
# Parallel ffmpeg processes used for clip extraction within a single job. Kept
# modest by default because MAX_CONCURRENT_JOBS jobs can each run this many at
# once -- the effective ceiling is MAX_CONCURRENT_JOBS * EXTRACT_WORKERS.
EXTRACT_WORKERS = int(os.environ.get('EXTRACT_WORKERS', 4))
# Cap on how many scenes get an AI vision call per job. A long episode can detect
# hundreds of scenes but only ~12 reach the trailer, so scoring all of them is
# almost entirely wasted work; see the shortlist logic in _run_trailer_job().
AI_SCORE_LIMIT = int(os.environ.get('AI_SCORE_LIMIT', 60))
AI_SCORE_WORKERS = int(os.environ.get('AI_SCORE_WORKERS', 4))
# Token budget for one vision reply. Must comfortably exceed the answer length,
# because reasoning models count their (hidden) chain-of-thought against it -- a
# tight cap returns an empty `response` and every scene silently falls back to
# the neutral score.
AI_NUM_PREDICT = int(os.environ.get('AI_NUM_PREDICT', 400))
# Set to False the first time a server rejects the structured-output `format`
# field, so older Ollama builds pay the cost of that discovery only once.
AI_STRUCTURED_OK = True
AI_NEUTRAL_SCORE = 3  # mid-range prior for scenes that weren't or couldn't be scored
GATE = JobGate(MAX_CONCURRENT_JOBS)
JOB_QUEUE = []  # job_ids waiting for a free slot, in submission order
JOB_QUEUE_LOCK = threading.Lock()

def run_trailer_job_gated(jid, params):
    """Entry point used for every submitted job: waits for a free concurrency
    slot (reporting queue position while it waits), then runs the job, then
    frees the slot for the next one in line."""
    with JOB_QUEUE_LOCK:
        JOB_QUEUE.append(jid)
    try:
        with GATE.cond:
            while GATE.running >= GATE.limit:
                with JOB_QUEUE_LOCK:
                    if jid not in JOB_QUEUE:  # cancelled while waiting
                        return
                    ahead = JOB_QUEUE.index(jid)
                job_set(jid, step=(f'Queued — {ahead} job(s) ahead' if ahead else 'Queued — starting shortly'),
                        percent=0, status='queued')
                GATE.cond.wait(timeout=2)
            GATE.running += 1
        with JOB_QUEUE_LOCK:
            if jid in JOB_QUEUE:
                JOB_QUEUE.remove(jid)
        job_set(jid, step='Starting', percent=1, status='running')
        run_trailer_job(jid, params)
    finally:
        with GATE.cond:
            GATE.running = max(0, GATE.running - 1)
            GATE.cond.notify_all()
        with JOB_QUEUE_LOCK:
            if jid in JOB_QUEUE:
                JOB_QUEUE.remove(jid)

GENRE_PROMPTS = {
    'action': 'Epic cinematic action trailer score, 150 BPM, powerful orchestral percussion, aggressive taiko drums, bold brass stabs, driving string ostinatos, rising tension, heroic energy, blockbuster soundtrack, high impact dynamics, instrumental only, no vocals',
    'drama': 'Emotional cinematic drama score, 80 BPM, expressive piano melody, warm string ensemble, gentle emotional build, heartfelt atmosphere, reflective storytelling, film soundtrack style, instrumental only, no vocals',
    'horror': 'Dark atmospheric horror soundtrack, 65 BPM, eerie drones, unsettling textures, dissonant strings, distant impacts, creeping suspense, psychological tension, cinematic dread, instrumental only, no vocals',
    'comedy': 'Playful comedy soundtrack, 120 BPM, cheerful pizzicato strings, quirky woodwinds, light percussion, whimsical melodies, upbeat and humorous mood, family entertainment style, instrumental only, no vocals',
    'documentary': 'Inspiring documentary score, 90 BPM, soft piano and strings, ambient orchestral textures, thoughtful emotional tone, uplifting cinematic storytelling, modern documentary soundtrack, instrumental only, no vocals',
    'thriller': 'Suspenseful thriller soundtrack, 110 BPM, pulsing rhythmic patterns, dark atmospheric pads, subtle electronic elements, escalating tension, cinematic urgency, investigative mood, instrumental only, no vocals',
    'scifi': 'Futuristic science fiction soundtrack, 120 BPM, atmospheric synthesizers, cosmic pads, electronic pulses, cinematic space exploration mood, advanced technology theme, immersive and expansive, instrumental only, no vocals',
    'fantasy': 'Enchanting fantasy orchestral score, 105 BPM, magical strings and woodwinds, mystical choir textures, adventurous melodies, wonder and discovery, cinematic fantasy realm atmosphere, instrumental only, no vocals',
    'romance': 'Romantic cinematic soundtrack, 75 BPM, tender piano melodies, warm strings, emotional intimacy, gentle orchestral swells, heartfelt and elegant atmosphere, instrumental only, no vocals',
    'adventure': 'Epic adventure soundtrack, 130 BPM, heroic brass, sweeping strings, driving percussion, exploration and discovery theme, uplifting cinematic energy, triumphant orchestral score, instrumental only, no vocals',
    'mystery': 'Intriguing mystery soundtrack, 90 BPM, subtle piano motifs, atmospheric strings, investigative mood, gradual tension build, enigmatic cinematic atmosphere, suspenseful yet elegant, instrumental only, no vocals',
    'western': 'Classic western cinematic score, 105 BPM, acoustic guitar, harmonica, sparse percussion, dusty frontier atmosphere, rugged adventure mood, expansive desert landscapes, instrumental only, no vocals',
    'sports': 'High-energy sports anthem, 155 BPM, driving drums, motivational brass, uplifting orchestral and modern hybrid elements, victory and competition theme, powerful momentum, instrumental only, no vocals',
    'noir': 'Film noir jazz soundtrack, 85 BPM, smoky saxophone, upright bass, brushed drums, moody piano, mysterious detective atmosphere, dark urban night setting, instrumental only, no vocals',
    'war': 'Epic war drama soundtrack, 115 BPM, military drums, emotional strings, heroic brass, sacrifice and courage theme, cinematic battlefield atmosphere, tragic yet triumphant, instrumental only, no vocals',
}

GENRE_PRESETS = {
    # transition choices are deliberately genre-signature — 'fade' is only reused
    # across drama/documentary/romance, the tonally-similar "soft/emotional" cluster,
    # which stay differentiated from each other via xfade_dur instead.
    'action': {'transition': 'zoomin', 'xfade_dur': 0.2, 'sfx': True},
    'drama': {'transition': 'fade', 'xfade_dur': 0.6, 'sfx': False},
    'horror': {'transition': 'wipeleft', 'xfade_dur': 0.3, 'sfx': True},
    'comedy': {'transition': 'squeezev', 'xfade_dur': 0.25, 'sfx': True},
    'documentary': {'transition': 'fade', 'xfade_dur': 0.5, 'sfx': False},
    'thriller': {'transition': 'radial', 'xfade_dur': 0.2, 'sfx': True},
    'scifi': {'transition': 'pixelize', 'xfade_dur': 0.3, 'sfx': True},
    'fantasy': {'transition': 'dissolve', 'xfade_dur': 0.5, 'sfx': True},
    'romance': {'transition': 'fade', 'xfade_dur': 0.8, 'sfx': False},
    'adventure': {'transition': 'smoothright', 'xfade_dur': 0.2, 'sfx': True},
    'mystery': {'transition': 'fadeblack', 'xfade_dur': 0.4, 'sfx': False},
    'western': {'transition': 'diagbr', 'xfade_dur': 0.3, 'sfx': True},
    'sports': {'transition': 'slideup', 'xfade_dur': 0.15, 'sfx': True},
    'noir': {'transition': 'circleclose', 'xfade_dur': 0.6, 'sfx': False},
    'war': {'transition': 'distance', 'xfade_dur': 0.3, 'sfx': True},
}
# NOTE: values above were already identical to the uploaded CSV's transition/crossfade/sfx_at_cuts
# columns for all 15 genres — no changes were needed here.

GENRE_NAMES = list(GENRE_PRESETS.keys())

# Module-level so both api_trailer() and the show-template save route can validate
# against the same list (it used to be a local inside api_trailer()).
VALID_TRANSITIONS = {'fade','fadeblack','fadewhite','fadefast','fadegrays',
    'wipeleft','wiperight','wipeup','wipedown',
    'slideleft','slideright','slideup','slidedown',
    'smoothleft','smoothright','smoothup','smoothdown',
    'circlecrop','rectcrop','circleopen','circleclose',
    'distance','pixelize','diagtl','diagtr','diagbl','diagbr',
    'hlslice','hrslice','vuslice','vdslice',
    'radial','zoomin','dissolve','hblur','squeezev','squeezeh',
    'horzopen','horzclose','vertopen','vertclose','custom_matte'}

GENRE_SFX_PROMPTS = {
    'action': 'Massive cinematic impact, deep explosion boom, trailer hit, powerful transient, sound effect only',
    'horror': 'Eerie horror sting, dark whoosh, unsettling impact, suspense accent, sound effect only',
    'comedy': 'Cartoon boing, comedic pop, playful bounce, humorous sting, sound effect only',
    'thriller': 'Tension hit, dark pulse, suspense sting, cinematic impact, sound effect only',
    'scifi': 'Futuristic whoosh, electronic glitch impact, cybernetic sweep, sci-fi accent, sound effect only',
    'fantasy': 'Magical sparkle chime, enchanted shimmer, mystical twinkle, fantasy accent, sound effect only',
    'adventure': 'Heroic orchestral hit, cinematic impact, adventure accent, sound effect only',
    'western': 'Whip crack, dusty impact, western accent, sound effect only',
    'sports': 'Stadium crowd hit, whistle blast, energetic impact, sports accent, sound effect only',
    'war': 'Battlefield explosion hit, military impact, distant artillery boom, sound effect only',
}

GENRE_LAVFI = {
    'default': 'sin(261.63*t)*0.25+sin(329.63*t)*0.18+sin(392.00*t)*0.14+sin(523.25*t)*0.1+sin(130.81*t)*0.12',
    'action': 'sin(55*t)*(1+0.3*sin(4*t))+sin(110*t)*0.4+sin(220*t)*0.2+sin(440*t)*0.1+sin(880*t)*0.05',
    'drama': 'sin(130.81*t)*0.3+sin(196*t)*0.2+sin(261.63*t)*0.15+sin(392*t)*0.08',
    'horror': 'sin(30*t)*0.5+sin(35*t)*0.3+sin(2000*t)*0.05+sin(2100*t)*0.04+random(t)*0.02',
    'comedy': 'sin(523.25*t)*0.3+sin(659.25*t)*0.25+sin(783.99*t)*0.2+sin(1046.5*t)*0.1+sin(1318.5*t)*0.05',
    'documentary': 'sin(261.63*t)*0.2+sin(329.63*t)*0.15+sin(392*t)*0.12+sin(523.25*t)*0.08',
    'thriller': 'sin(50*t)*0.4+sin(100*t)*0.2+sin(150*t)*0.1+sin(800*t)*0.05+sin(1200*t)*0.03',
    'scifi': 'sin(220*t)*0.2+sin(440*t)*0.15+sin(880*t)*0.1+sin(1760*t)*0.05+sin(200*t*(1+0.1*sin(0.5*t)))*0.15',
    'fantasy': 'sin(261.63*t)*0.2+sin(392*t)*0.15+sin(523.25*t)*0.12+sin(659.25*t)*0.08+sin(783.99*t)*0.05',
    'romance': 'sin(261.63*t)*0.25+sin(329.63*t)*0.2+sin(392*t)*0.15+sin(523.25*t)*0.08',
    'adventure': 'sin(65.41*t)*0.3+sin(130.81*t)*0.2+sin(261.63*t)*0.15+sin(392*t)*0.1+sin(523.25*t)*0.08',
    'mystery': 'sin(100*t)*0.3+sin(150*t)*0.15+sin(1200*t)*0.05+sin(1800*t)*0.03',
    'western': 'sin(196*t)*0.25+sin(220*t)*0.15+sin(261.63*t)*0.12+sin(329.63*t)*0.08+sin(392*t)*0.05',
    'sports': 'sin(110*t)*(0.5+0.3*lt(sin(2*t),0))+sin(220*t)*0.2+sin(440*t)*0.1+sin(880*t)*0.05',
    'noir': 'sin(98*t)*0.25+sin(130.81*t)*0.2+sin(196*t)*0.15+sin(246.94*t)*0.1',
    'war': 'sin(55*t)*0.3+sin(65.41*t)*0.2+sin(110*t)*0.15+sin(200*t*(1+0.2*sin(2*t)))*0.1+sin(440*t)*0.05',
}

FFMPEG = shutil.which('ffmpeg') or 'C:\\ffmpeg\\bin\\ffmpeg.exe'
FFPROBE = shutil.which('ffprobe') or 'C:\\ffmpeg\\bin\\ffprobe.exe'

# ---- Hard timeouts on every external media call ----
# A subprocess.run() without a timeout can block its worker thread forever if
# ffmpeg wedges on a malformed stream. That thread holds a GATE slot (see
# JobGate), so with MAX_CONCURRENT_JOBS=2 two hung jobs stall the whole server
# with no way to recover short of a restart. Every ffmpeg/ffprobe call therefore
# goes through run_ffmpeg()/run_ffprobe() below, which always pass a timeout and
# always kill the whole process group on expiry.
FFPROBE_TIMEOUT = int(os.environ.get('FFPROBE_TIMEOUT', 30))
FFMPEG_TIMEOUT = int(os.environ.get('FFMPEG_TIMEOUT', 300))       # per encode/mix step
FFMPEG_LONG_TIMEOUT = int(os.environ.get('FFMPEG_LONG_TIMEOUT', 900))  # full-source passes

class MediaToolTimeout(RuntimeError):
    """Raised when ffmpeg/ffprobe exceeded its timeout and was killed."""

def _run_media_tool(cmd, timeout, label):
    """subprocess.run with a guaranteed timeout and a guaranteed kill.

    subprocess.run(timeout=...) sends SIGKILL to the direct child only; ffmpeg
    rarely spawns children, but Popen.kill() can still leave a stuck process if
    it's blocked in an uninterruptible read, so we kill then reap with a short
    second wait and surface a clean exception either way."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        print(f'{label} timed out after {timeout}s and was killed: {" ".join(str(c) for c in cmd[:6])}...')
        raise MediaToolTimeout(f'{label} exceeded {timeout}s') from e

def run_ffmpeg(cmd, timeout=None, label='ffmpeg'):
    return _run_media_tool(cmd, timeout or FFMPEG_TIMEOUT, label)

def run_ffprobe(cmd, timeout=None, label='ffprobe'):
    return _run_media_tool(cmd, timeout or FFPROBE_TIMEOUT, label)

def probe_duration(path, default=None):
    """Duration of `path` in seconds, or `default` if it can't be determined.

    Returning None (the default default) lets callers distinguish "unknown" from
    a real value instead of silently substituting a guess -- a wrong duration
    here feeds the xfade offset maths and desynchronises the entire concat."""
    try:
        r = run_ffprobe([FFPROBE, '-v', 'error', '-show_entries', 'format=duration',
                         '-of', 'default=noprint_wrappers=1:nokey=1', path])
        return float(r.stdout.strip())
    except (MediaToolTimeout, ValueError, AttributeError):
        return default

def probe_media_info(path):
    """One ffprobe call returning {'duration': float|None, 'has_audio': bool}.

    Replaces the previous pattern of three separate ffprobe spawns per input
    (duration, audio-stream check, duration again after normalisation) -- on a
    15-input job that was ~46 process spawns just to read metadata."""
    info = {'duration': None, 'has_audio': False}
    try:
        r = run_ffprobe([FFPROBE, '-v', 'error', '-show_entries',
                         'format=duration:stream=codec_type', '-of', 'json', path])
        data = json.loads(r.stdout or '{}')
    except (MediaToolTimeout, ValueError):
        return info
    try:
        info['duration'] = float(data.get('format', {}).get('duration'))
    except (TypeError, ValueError):
        info['duration'] = None
    info['has_audio'] = any(s.get('codec_type') == 'audio' for s in data.get('streams', []))
    return info

# Precomputed once at import time for the Docs tab's genre reference table —
# pulls straight from GENRE_PRESETS/GENRE_PROMPTS/GENRE_SFX_PROMPTS so the docs
# can never drift out of sync with the actual per-genre behavior.
GENRE_DOCS_ROWS = [{
    'genre': g,
    'transition': GENRE_PRESETS[g]['transition'],
    'xfade_dur': GENRE_PRESETS[g]['xfade_dur'],
    'sfx': GENRE_PRESETS[g]['sfx'],
    'music_theme': GENRE_PROMPTS.get(g, ''),
    'sfx_theme': GENRE_SFX_PROMPTS.get(g, '—'),
} for g in GENRE_NAMES]

# ---- Download/export format options ----
# 'mp4_high' is a genuine, standard H.264 High Profile MP4 — no caveats.
# The two ProRes options use ffmpeg's real prores_ks encoder, which is a
# legitimate, widely-used open implementation (not an approximation).
# 'avci100i' is NOT a certified Panasonic AVC-Intra stream — ffmpeg has no
# actual AVC-Intra encoder. It's a best-effort approximation built from
# all-intra libx264 at a similar spec (1080i, 4:2:2 10-bit, ~100Mb/s CBR),
# meant to be visually/structurally similar, not a guaranteed match for
# equipment that specifically validates the AVC-Intra codec ID.
EXPORT_FORMATS = {
    'mp4_high':        {'ext': 'mp4', 'label': 'MP4 (H.264 High Profile)'},
    'prores_hq_2997':  {'ext': 'mov', 'label': 'Apple ProRes 422 HQ — 29.97fps'},
    'prores_hq_2398':  {'ext': 'mov', 'label': 'Apple ProRes 422 HQ — 23.976fps'},
    'avci100i':        {'ext': 'mov', 'label': 'AVC-Intra 100i (H.264 Intra approximation)'},
}

def _detect_silence_intervals(audio_path, noise_db=-30, min_dur=0.3, timeout=120):
    """Runs ffmpeg's silencedetect filter and parses stderr for silence_start/silence_end
    pairs. Returns a list of (start, end) SILENT intervals in audio_path. A silence_start
    with no matching silence_end (file ends mid-silence) is dropped rather than guessed at —
    the caller treats "not explicitly silent" as active, which is the safe default."""
    try:
        r = subprocess.run([FFMPEG, '-i', audio_path, '-af',
                             f'silencedetect=noise={noise_db}dB:d={min_dur}', '-f', 'null', '-'],
                            capture_output=True, text=True, timeout=timeout)
    except Exception as e:
        print(f'Silence detection error ({audio_path}): {e}')
        return []
    starts = [float(m) for m in re.findall(r'silence_start:\s*([\d.]+)', r.stderr)]
    ends = [float(m) for m in re.findall(r'silence_end:\s*([\d.]+)', r.stderr)]
    return list(zip(starts, ends))

def _active_windows_from_silence(silence_intervals, total_duration, content_duration=None):
    """Inverts silent intervals into the (start, end) windows where audio IS
    present.

    `content_duration` is the real length of the file that was analyzed. It
    defaults to `total_duration` for the common case (e.g. SOT, where the
    analyzed file genuinely spans the whole trailer) -- but when the source is
    SHORTER than `total_duration` (a VO clip placed early in a longer
    trailer), the trailing "active" extension must stop at the real end of
    that file, not run all the way out to total_duration. Without this bound:
    a short VO clip with no detected internal silence (nothing for
    silencedetect to report) gets treated as "playing" for the entire
    remainder of the trailer, which then ducks BGM/SOT under a VO that
    actually stopped minutes earlier -- including under the cards."""
    if content_duration is None:
        content_duration = total_duration
    silence_intervals = sorted(silence_intervals)
    windows = []
    cursor = 0.0
    for s, e in silence_intervals:
        if s > cursor:
            windows.append((cursor, s))
        cursor = max(cursor, e)
    end_bound = min(content_duration, total_duration)
    if cursor < end_bound:
        windows.append((cursor, end_bound))
    return windows

def _union_windows(window_lists):
    """Unions multiple lists of (start, end) windows (e.g. SOT-active ∪ VO-active) into
    one merged, sorted, non-overlapping list."""
    all_w = sorted(w for lst in window_lists for w in lst)
    if not all_w:
        return []
    merged = [list(all_w[0])]
    for s, e in all_w[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]

def _merge_windows_with_hold(windows, hold_sec):
    """Bridges any gap between consecutive windows that's shorter than hold_sec — this is
    the actual 'minimum gap of no VO/SOT before the duck releases and music is heard again'
    behavior: a brief pause in dialogue no longer lets BGM swell back up and immediately
    duck again, since the gap has to be at least hold_sec long to count as a real release."""
    if not windows:
        return []
    windows = sorted(windows)
    merged = [list(windows[0])]
    for s, e in windows[1:]:
        if s - merged[-1][1] <= hold_sec:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]

DUCK_ATTACK = float(os.environ.get('DUCK_ATTACK', 0.08))    # seconds to duck down
DUCK_RELEASE = float(os.environ.get('DUCK_RELEASE', 0.45))  # seconds to come back up

def _build_duck_volume_expr(duck_windows, duck_depth_db, attack=None, release=None):
    """ffmpeg volume expression that ducks by duck_depth_db across duck_windows,
    with real attack/release ramps.

    This used to emit a bare step -- if(between(t,s,e), gain, 1) -- which changes
    gain by the full depth between one sample and the next. A 15 dB discontinuity
    is a broadband click, and there were typically 9-13 of them in a 30s promo, so
    it read as a fault rather than as ducking.

    The envelope is built as a FLAT sum of trapezoids rather than a nested if
    chain, which matters as much as the ramps do: the old form nested one if()
    per window, so expression depth grew with the amount of dialogue. Here depth
    is constant and only the length grows.

        gain(t) = 1 - (1 - g) * min(1, SUM_i trapezoid_i(t))
        trapezoid_i(t) = clip((t - s_i + a)/a, 0, 1) * clip((e_i + r - t)/r, 0, 1)

    Each trapezoid ramps 0->1 over the attack window ending at s_i, holds at 1
    through the window, then ramps 1->0 over the release after e_i. min(1, ...)
    keeps overlapping ramps from over-ducking: two windows closer together than
    attack+release simply stay ducked through the gap, which is what you'd want
    anyway."""
    if not duck_windows:
        return None
    a = max(0.01, attack if attack is not None else DUCK_ATTACK)
    r = max(0.01, release if release is not None else DUCK_RELEASE)
    gain = 10 ** (duck_depth_db / 20)
    terms = [f'clip((t-{s:.3f}+{a})/{a},0,1)*clip(({e:.3f}+{r}-t)/{r},0,1)'
             for s, e in duck_windows]
    return f'1-{1 - gain:.5f}*min(1,{"+".join(terms)})'


def build_export_cmd(src, dst, fmt_key):
    """Build the ffmpeg command that transcodes a finished trailer (src) into the
    requested delivery format (dst). Returns None for an unknown fmt_key."""
    if fmt_key == 'mp4_high':
        # A distinct higher-quality delivery pass vs. the crf22/fast preset used
        # internally while assembling the trailer.
        return [FFMPEG, '-y', '-i', src,
                '-c:v', 'libx264', '-profile:v', 'high', '-pix_fmt', 'yuv420p',
                '-preset', 'slow', '-crf', '16',
                '-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart', dst]
    if fmt_key == 'prores_hq_2997':
        return [FFMPEG, '-y', '-i', src, '-vf', 'fps=30000/1001',
                '-c:v', 'prores_ks', '-profile:v', '3', '-vendor', 'apl0',
                '-pix_fmt', 'yuv422p10le', '-c:a', 'pcm_s16le', dst]
    if fmt_key == 'prores_hq_2398':
        return [FFMPEG, '-y', '-i', src, '-vf', 'fps=24000/1001',
                '-c:v', 'prores_ks', '-profile:v', '3', '-vendor', 'apl0',
                '-pix_fmt', 'yuv422p10le', '-c:a', 'pcm_s16le', dst]
    if fmt_key == 'avci100i':
        return [FFMPEG, '-y', '-i', src, '-vf', 'fps=30000/1001,format=yuv422p10le',
                '-c:v', 'libx264', '-profile:v', 'high422',
                '-x264-params', 'keyint=1:bframes=0:cabac=1:interlaced=1',
                '-b:v', '100M', '-minrate', '100M', '-maxrate', '100M', '-bufsize', '100M',
                '-flags', '+ildct+ilme', '-pix_fmt', 'yuv422p10le',
                '-c:a', 'pcm_s16le', dst]
    return None

# Find ONNX model for face detection
ONNX_PATH = next((p for p in [
    os.path.join(os.environ.get('TEMP', '/tmp'), 'face_detection_yunet.onnx'),
    os.path.join(os.environ.get('TMP', '/tmp'), 'face_detection_yunet.onnx'),
    os.path.join(str(pathlib.Path.home()), 'face_detection_yunet.onnx'),
    os.path.join(os.path.dirname(cv2.__file__), 'face_detection_yunet.onnx'),
    os.path.join(os.path.dirname(cv2.__file__), 'data', 'face_detection_yunet.onnx'),
] if os.path.exists(p)), None)

# Cache the face detector (reused across requests)
_fd_lock = threading.Lock()
_fd = None

def get_fd(w, h):
    global _fd
    if _fd is None:
        with _fd_lock:
            if _fd is None:
                _fd = cv2.FaceDetectorYN.create(model=ONNX_PATH, config='', input_size=(320, 320),
                                                score_threshold=0.8, nms_threshold=0.3, top_k=5000)
    _fd.setInputSize((w, h))
    return _fd

def allowed_file(filename, exts=None):
    exts = ALLOWED_EXTENSIONS if exts is None else exts
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in exts

def load_video(req):
    if 'file' in req.files and req.files['file'].filename != '':
        f = req.files['file']
        if not allowed_file(f.filename):
            return None, 'File type not allowed'
        fn = secure_filename(f.filename)
        if not fn:
            return None, 'Invalid filename'
        # Two people can easily upload same-named files at once (e.g. two
        # different "episode1.mp4"s) while both jobs run concurrently (see
        # MAX_CONCURRENT_JOBS / GATE below) -- without a per-request-unique
        # disk name, the second upload would silently overwrite the first
        # one's staged file out from under its still-running job. orig_name
        # (the display name shown in results/history) stays the clean one.
        disk_name = f'src_{int(time.time()*1000)}_{threading.get_ident()}_{fn}'
        path = os.path.join(app.config['UPLOAD_FOLDER'], disk_name)
        f.save(path)
        return path, fn
    # The main dropzone's own richer browser posts 'network_file'; the shared
    # "Browse library" modal used elsewhere (title/end card, BGM, SFX, VO, and
    # now this Vision form) posts '<field>_network' -- for a field literally
    # named 'file' that's 'file_network'. Accept either so the shared modal
    # works here without needing its own bespoke field-naming convention.
    staged = req.form.get('network_file') or req.form.get('file_network')
    if staged:
        # Must be a name we generated ourselves in fetch_network_video() (net_<ts>_<name>)
        # and that still exists in UPLOAD_FOLDER -- never trust an arbitrary path here.
        safe = os.path.basename(staged)
        path = os.path.join(app.config['UPLOAD_FOLDER'], safe)
        if safe.startswith('net_') and os.path.exists(path):
            return path, safe
        return None, 'Selected network file is no longer available -- please re-select it'
    return None, 'No video provided'

def _resolve_upload(field_name, exts=None):
    """Resolves `field_name` to a local file path, from either a normal file
    upload (request.files[field_name]) or a network-folder file already staged
    into UPLOAD_FOLDER by /api/network/fetch (request.form[field_name + '_network']).
    A direct upload always takes priority if both are present. Returns None if
    neither is present/valid. Used for every optional media field that now
    supports "browse library" (title/end card video, BG music, VO, SFX)."""
    if field_name in request.files and request.files[field_name].filename:
        f = request.files[field_name]
        if exts is not None and not allowed_file(f.filename, exts):
            return None
        fn = secure_filename(f.filename)
        if not fn:
            return None
        dest = os.path.join(app.config['UPLOAD_FOLDER'], f'{field_name}_{int(time.time()*1000)}_{threading.get_ident()}{os.path.splitext(fn)[1]}')
        f.save(dest)
        return dest
    staged = (request.form.get(field_name + '_network') or '').strip()
    if staged:
        # Must be a name we generated ourselves in fetch_network_file() (net_<ts>_<name>)
        # and that still exists in UPLOAD_FOLDER -- never trust an arbitrary path here.
        safe = os.path.basename(staged)
        path = os.path.join(app.config['UPLOAD_FOLDER'], safe)
        if safe.startswith('net_') and os.path.exists(path):
            return path
    return None

def _upload_display_name(field_name):
    """The human-readable original filename behind whatever _resolve_upload() would
    return for `field_name` -- a browser upload's own name, or the network file's
    name with the net_<ts>_ staging prefix stripped back off. Used so saved show
    templates list "PLUG_BED_2024.wav" rather than an internal storage name."""
    if field_name in request.files and request.files[field_name].filename:
        return os.path.basename(request.files[field_name].filename)
    staged = (request.form.get(field_name + '_network') or '').strip()
    if staged:
        return re.sub(r'^net_\d+_', '', os.path.basename(staged))
    return None

def _clean_ai_desc(raw):
    """Tidies a vision model's DESC field for display in the scene table: single
    line, no trailing punctuation, sentence-cased, and capped so one rambling
    response can't stretch the column."""
    t = ' '.join((raw or '').split())
    t = re.sub(r'^(the\s+)?(image|frame|shot|scene)\s+(shows|depicts|features)\s+', '', t, flags=re.I)
    t = t.strip(' .;:,-')
    if len(t) > 120:
        t = t[:117].rsplit(' ', 1)[0] + '…'
    return (t[:1].upper() + t[1:]) if t else ''

def _scene_desc(s):
    """One human-readable line describing the shot, for the scene table.

    Prefers the vision model's literal description of what's in frame. Without
    AI rating there's no semantic information available at all, so the fallback
    describes the measurable properties in plain words -- a close-up vs a wide,
    how bright, how busy -- rather than emitting raw metric tags."""
    ai = s.get('ai_desc', '') or ''
    if ai:
        return ai

    sat = s.get('mean_sat', 0)
    val = s.get('mean_val', 0)
    edge = s.get('edge_ratio', 0)
    hue = s.get('mean_hue', 0)
    dur = s.get('duration', 0)

    subject = 'Person in shot' if s.get('has_face') else (
        'Busy, detailed shot' if edge > 0.15 else
        'Wide, open shot' if edge < 0.06 else 'Medium shot')

    light = ('very dark' if val < 40 else
             'dim' if val < 90 else
             'bright' if val > 200 else 'well lit')

    colour = ('almost monochrome' if sat < 30 else
              'strongly coloured' if sat > 100 else None)

    palette = ('greens/outdoors' if 90 < hue < 150 else
               'warm tones' if (0 < hue < 30 or 160 < hue < 180) else None)

    bits = [light]
    if colour: bits.append(colour)
    if palette: bits.append(palette)
    tail = f' — {dur:.1f}s take' if dur > 5 else ''
    return f'{subject}, {", ".join(bits)}{tail}'

def _to_scalar(x):
    """Robustly converts a value that may already be a Python/numpy scalar, or
    may be a 0-d or 1-d numpy array, into a plain Python float. Needed because
    librosa.beat.beat_track's `tempo` return value changed from a plain float
    to a numpy array (shape (1,), one entry per audio channel) in librosa
    0.10+ -- calling float() directly on that array is what raises numpy's
    'only 0-dimensional arrays can be converted to Python scalars' error.
    np.ravel(x)[0] normalizes every shape (scalar, 0-d, 1-d) to a single
    value before the float() conversion."""
    return float(np.ravel(x)[0])

def beat_match_audio(video_path, bgm_path, target_dur, output_path):
    try:
        import librosa
        # Extract audio from source video, resample to consistent rate
        audio_tmp = os.path.join(app.config['UPLOAD_FOLDER'], f'beat_video_{int(time.time())}.wav')
        subprocess.run([FFMPEG, '-y', '-i', video_path, '-vn', '-ar', '22050', '-ac', '1', audio_tmp],
                       capture_output=True, text=True, timeout=60)
        if not os.path.exists(audio_tmp) or os.path.getsize(audio_tmp) == 0:
            return False
        y_vid, sr = librosa_load(audio_tmp, sr=22050)
        os.remove(audio_tmp)
        tempo_vid, _ = librosa.beat.beat_track(y=y_vid, sr=sr)
        tempo_vid = _to_scalar(tempo_vid)
        if tempo_vid < 30 or tempo_vid > 300:
            tempo_vid = 120
    except Exception as e:
        print(f'Beat detection error: {e}')
        return False

    try:
        y_bgm, sr_bgm = librosa_load(bgm_path, sr=22050)
        orig_len = len(y_bgm)
        # Detect BGM tempo
        tempo_bgm, _ = librosa.beat.beat_track(y=y_bgm, sr=sr_bgm)
        tempo_bgm = _to_scalar(tempo_bgm)
        if tempo_bgm < 30 or tempo_bgm > 300:
            tempo_bgm = tempo_vid

        # Time-stretch to match video tempo (preserves pitch)
        stretch = tempo_vid / tempo_bgm
        if abs(stretch - 1.0) > 0.01:
            y_bgm = librosa.effects.time_stretch(y=y_bgm, rate=stretch)

        # Loop or trim to fill target_dur seconds
        target_samples = int(target_dur * sr_bgm)
        if len(y_bgm) < target_samples:
            repeats = int(np.ceil(target_samples / len(y_bgm)))
            y_bgm = np.tile(y_bgm, repeats)
        y_bgm = y_bgm[:target_samples]

        # Write processed BGM with proper sample rate
        import soundfile as sf
        sf.write(output_path, y_bgm, sr_bgm)
        return os.path.exists(output_path) and os.path.getsize(output_path) > 0
    except Exception as e:
        print(f'Beat match processing error: {e}')
        return False

def apply_filter(frame, mode, prev_gray=None):
    if mode == 'gray':
        return cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR), prev_gray
    if mode == 'edges':
        return cv2.cvtColor(cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 100, 200), cv2.COLOR_GRAY2BGR), prev_gray
    if mode == 'hsv':
        return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), prev_gray
    if mode == 'blur':
        return cv2.GaussianBlur(frame, (15, 15), 0), prev_gray
    if mode == 'face':
        if ONNX_PATH is None:
            return frame, prev_gray
        h, w = frame.shape[:2]
        _, faces = get_fd(w, h).detect(frame)
        if faces is not None:
            for f in faces:
                x, y, fw, fh = map(int, f[:4])
                cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 0), 3)
        return frame, prev_gray
    if mode == 'motion':
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (21, 21), 0)
        if prev_gray is None:
            return frame, gray
        diff = cv2.absdiff(prev_gray, gray)
        thresh = cv2.dilate(cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1], None, iterations=2)
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        n = 0
        for c in cnts:
            if cv2.contourArea(c) > 500:
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                n += 1
        if n:
            cv2.putText(frame, f'Motion: {n}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frame, gray
    return frame, prev_gray

def synth_sfx_waveform(genre, sample_rate=22050, sfx_dur=0.6):
    """Synthesize a single procedural 'hit' waveform for a genre. Returns a
    1-D float array in [-1, 1], or None if the genre has no synth SFX defined."""
    n_sfx = int(sfx_dur * sample_rate)
    t = np.arange(n_sfx) / sample_rate
    if genre == 'action' or genre == 'war':
        # low sine "thump" for body + short filtered-feeling noise burst for texture,
        # instead of pure white noise (which reads as hiss, not boom)
        thump = np.sin(2*np.pi * 65 * t) * np.exp(-t * 14)
        crackle = np.random.randn(n_sfx) * np.exp(-t * 22) * 0.35
        sfx = thump * 0.8 + crackle
    elif genre == 'horror':
        # lower, narrower sweep than before (was up to 8kHz — shrill/painful);
        # add a sub-bass rumble underneath for dread rather than a piercing shriek
        sweep = np.sin(2*np.pi * (1200 + t * 3000) * t) * np.exp(-t * 8) * 0.25
        rumble = np.sin(2*np.pi * 42 * t) * np.exp(-t * 6) * 0.3
        sfx = sweep + rumble
    elif genre == 'comedy':
        sfx = np.sin(2*np.pi * (600 - t * 1200) * t) * np.exp(-t * 5) * 0.4
    elif genre in ('thriller', 'adventure', 'scifi'):
        sfx = np.sin(2*np.pi * (100 + t * 3000) * t) * np.exp(-t * 6) * 0.3
    elif genre == 'western':
        # sharp transient crack plus a touch of low-mid body so it reads as a
        # gunshot/whip-crack rather than a thin, bodyless tick
        crack = np.random.randn(n_sfx) * np.exp(-t * 35) * 0.55
        body = np.sin(2*np.pi * 180 * t) * np.exp(-t * 16) * 0.3
        sfx = crack + body
    elif genre == 'sports':
        # sine-based whistle tone in real whistle range (~2.8-3.1kHz) instead of a
        # raw square wave, which aliases into a harsh electronic buzz
        sfx = (np.sin(2*np.pi * 2800 * t) * 0.35 + np.sin(2*np.pi * 3100 * t) * 0.2) * np.exp(-t * 4)
    elif genre == 'fantasy':
        sfx = (np.sin(2*np.pi * 528 * t) * 0.3 + np.sin(2*np.pi * 1056 * t) * 0.15 +
               np.sin(2*np.pi * 1584 * t) * 0.08) * np.exp(-t * 4)
    else:
        return None
    peak = np.max(np.abs(sfx))
    if peak > 0:
        sfx = sfx / peak * 0.85
    return sfx

def load_hit_waveform(path, sample_rate=22050, max_dur=1.2):
    """Load an uploaded or AI-generated one-shot SFX file (any format ffmpeg/librosa
    can read) as a short mono waveform, trimmed and fade-tailed so it behaves like
    a 'hit' when stamped at multiple cut points. Returns None on failure."""
    try:
        import librosa
        y, sr = librosa_load(path, sr=sample_rate, mono=True, duration=max_dur)
        if y is None or len(y) == 0:
            return None
        # short fade-out so repeated stamping never clicks at the tail
        fade_n = min(len(y), int(0.05 * sample_rate))
        if fade_n > 0:
            y[-fade_n:] *= np.linspace(1, 0, fade_n)
        peak = np.max(np.abs(y))
        if peak > 0:
            y = y / peak * 0.85
        return y
    except Exception as e:
        print(f'SFX load error: {e}')
        return None

def write_wav_pcm16(track, output_path, sample_rate=22050):
    track = np.clip(track, -1, 1)
    track_int = (track * 32767).astype(np.int16)
    import struct
    with open(output_path, 'wb') as f:
        data_len = len(track_int) * 2
        f.write(b'RIFF')
        f.write(struct.pack('<I', 36 + data_len))
        f.write(b'WAVE')
        f.write(b'fmt ')
        f.write(struct.pack('<I', 16))
        f.write(struct.pack('<H', 1))
        f.write(struct.pack('<H', 1))
        f.write(struct.pack('<I', sample_rate))
        f.write(struct.pack('<I', sample_rate * 2))
        f.write(struct.pack('<H', 2))
        f.write(struct.pack('<H', 16))
        f.write(b'data')
        f.write(struct.pack('<I', data_len))
        f.write(track_int.tobytes())
    return os.path.exists(output_path) and os.path.getsize(output_path) > 0

def stamp_hits(hit_wave, timestamps, output_path, sample_rate=22050):
    """Place a copy of `hit_wave` at every timestamp (seconds) into one continuous
    track and write it as a WAV. This is what makes SFX land on *every* cut,
    regardless of whether the hit sound came from synth, upload, or ACE-Step."""
    try:
        if hit_wave is None or len(hit_wave) == 0 or not timestamps:
            return False
        total_dur = max(timestamps) + (len(hit_wave) / sample_rate) + 0.5
        total_samples = int(total_dur * sample_rate)
        track = np.zeros(total_samples)
        for i, ts in enumerate(timestamps):
            # slight per-hit variation (pitch + level) so repeated cuts on a long
            # trailer don't sound like an obviously copy-pasted stock sound
            rng = np.random.RandomState(int(ts * 1000) & 0xffffffff)
            pitch = 1.0 + rng.uniform(-0.08, 0.08)
            amp = 0.9 + rng.uniform(-0.1, 0.1)
            if abs(pitch - 1.0) > 1e-6:
                idx = np.clip((np.arange(len(hit_wave)) * pitch).astype(int), 0, len(hit_wave) - 1)
                hit = hit_wave[idx] * amp
            else:
                hit = hit_wave * amp
            start = int(ts * sample_rate)
            end = min(start + len(hit), total_samples)
            if end > start:
                track[start:end] += hit[:end - start]
        return write_wav_pcm16(track, output_path, sample_rate)
    except Exception as e:
        print(f'SFX stamping error: {e}')
        return False

def _woosh_generate_raw(prompt, timeout=15):
    """One call to Woosh's /generate. Returns (flac_bytes, error).

    Request shape confirmed directly against this server's own /openapi.json
    (an earlier version of this comment claimed the same rigor for a flatter
    {prompt, token} body and was wrong -- that shape 500'd on '/generate' with
    a Pydantic "extra_forbidden" error on the field it lives on now,
    'args.prompt'). The real schema wraps every generation parameter inside
    a nested `args` object, alongside top-level `version`/`token`:

        {"version": "0.1", "token": "string", "args": {"prompt": ..., "cfg": 1,
         "sampler": "heun", "num_steps": 100, "sigma_min": 0.00001,
         "sigma_max": 80, "rho": 7, "S_churn": 1, "S_min": 0, "S_noise": 1,
         "guidance_scale": 7.5, "noise_scheduler": "karras", "model": "Woosh-DFlow"}}

    `token` still isn't read/validated server-side -- "string" (the schema's
    own example placeholder) satisfies it with nothing to configure. Every
    `args` field besides `prompt` is left at the schema's own example/default
    value rather than guessed at, since there's no way to verify what
    changing e.g. `sampler` or `guidance_scale` would actually do without
    access to the model itself.

    The response is FLAC (media_type="audio/flac"), not WAV -- this was
    previously saved with a .wav extension and served as-is, which lies about
    the container in both the filename and the Content-Type Flask would
    derive from it, and could fail to play in a browser depending on how
    strictly it trusts that mismatch.

    `duration` is not a field anywhere in this schema. Callers that need a
    specific length enforce it themselves by trimming the response (see
    _woosh_generate below), since the API has no way to ask for one."""
    try:
        r = requests.post(f'{WOOSH_URL}/generate', json={
            'version': '0.1',
            'token': 'string',
            'args': {
                'prompt': prompt,
                'cfg': 1,
                'sampler': 'heun',
                'num_steps': 100,
                'sigma_min': 0.00001,
                'sigma_max': 80,
                'rho': 7,
                'S_churn': 1,
                'S_min': 0,
                'S_noise': 1,
                'guidance_scale': 7.5,
                'noise_scheduler': 'karras',
                'model': 'Woosh-DFlow',
            },
        }, timeout=timeout)
    except Exception as e:
        return None, f'Could not reach Woosh at {WOOSH_URL}: {e}'
    if not r.ok:
        return None, f'Woosh API error {r.status_code}: {(r.text or "")[:300]}'
    if not r.content:
        return None, 'Woosh returned an empty response.'
    return r.content, None

def _woosh_generate(prompt, dest_flac_path, duration=None, timeout=15):
    """Fetches one Woosh generation and writes it to `dest_flac_path` (must end
    in .flac), trimming to `duration` seconds if given. Returns (ok, error).

    Trimming is done here, client-side, because the API has no duration
    parameter of its own -- this is the only way "duration" means anything."""
    raw, err = _woosh_generate_raw(prompt, timeout=timeout)
    if err:
        return False, err
    if duration is None:
        try:
            with open(dest_flac_path, 'wb') as f:
                f.write(raw)
        except OSError as e:
            return False, f'Could not write output file: {e}'
        return os.path.exists(dest_flac_path) and os.path.getsize(dest_flac_path) > 0, None

    raw_path = dest_flac_path + '.raw.flac'
    try:
        with open(raw_path, 'wb') as f:
            f.write(raw)
    except OSError as e:
        return False, f'Could not write output file: {e}'
    try:
        run_ffmpeg([FFMPEG, '-y', '-i', raw_path, '-t', str(duration), '-c:a', 'flac', dest_flac_path],
                   timeout=30, label='woosh trim')
    except MediaToolTimeout as e:
        return False, f'Trimming the Woosh output timed out: {e}'
    finally:
        if os.path.exists(raw_path):
            try:
                os.remove(raw_path)
            except OSError:
                pass
    return os.path.exists(dest_flac_path) and os.path.getsize(dest_flac_path) > 0, None

def woosh_sfx(genre, output_path, duration=0.8):
    """Generate a one-shot SFX using Sony AI's Woosh text-to-audio model via its
    local API server. `output_path` should end in .flac (Woosh's real output
    format -- see _woosh_generate_raw). Returns True/False — failures fall
    through silently so the caller can fall back to the procedural synth
    (ACE-Step is a music model, not used for SFX)."""
    prompt = GENRE_SFX_PROMPTS.get(genre)
    if not prompt:
        return False
    ok, err = _woosh_generate(prompt, output_path, duration=duration, timeout=15)
    if not ok and err:
        print(f'Woosh SFX error: {err}')
    return ok

WOOSH_MAX_SAMPLES = int(os.environ.get('WOOSH_MAX_SAMPLES', 4))

def woosh_sfx_generate(prompt, duration=None, samples=1, base_ts=None):
    """Generate SFX from free-text prompts, for the Tools tab. Returns (paths, error).

    Unlike woosh_sfx() above (which the trailer pipeline calls with a fixed
    genre-derived prompt and silently falls back to a procedural synth on
    failure, since *some* sound must occupy that slot), this is the raw
    generator behind the Text to SFX tool: any prompt the user types, and no
    fallback -- if Woosh is down the tool should say so rather than hand back
    a synth click the user didn't ask for.

    duration=None (the default, and the only option the Tools tab UI offers)
    returns Woosh's own natural-length output untouched. A caller can still
    pass a number to get it trimmed to that length client-side, since Woosh's
    real API has no duration parameter of its own -- see _woosh_generate.

    Multiple samples are produced by calling _woosh_generate repeatedly rather
    than a batch parameter, since the real request schema has no such field
    either (see _woosh_generate_raw)."""
    prompt = (prompt or '').strip()
    if not prompt:
        return [], 'Enter a description of the sound you want.'
    if duration is not None:
        duration = max(0.2, min(10.0, float(duration)))
    samples = max(1, min(WOOSH_MAX_SAMPLES, int(samples or 1)))
    base_ts = base_ts or f'tool{int(time.time()*1000)}'

    paths, last_err = [], None
    for i in range(samples):
        dest = os.path.join(app.config['UPLOAD_FOLDER'], f'sfx_{base_ts}_{i}.flac')
        ok, err = _woosh_generate(prompt, dest, duration=duration, timeout=30)
        if ok:
            paths.append(dest)
        else:
            last_err = err

    if not paths:
        return [], (last_err or 'Generation failed.')
    return paths, None

def generate_tts(text, output_wav_path, rate=175, voice_id=None, reference_audio_path=None, language=None, engine='fish_audio'):
    """Generate a narration WAV from text, via whichever engine the user picked:
    'fish_audio' (Fish Audio S2, voice cloning, auto-detects language including
    Tagalog). Returns (ok, error_message). There's
    no bundled default voice — the caller must pass either `voice_id` (a voice
    picked from list_voices_for_engine()) or `reference_audio_path` (an uploaded
    sample to clone zero-shot); if neither is given, this returns an error
    rather than silently falling back to some fixed voice file."""
    text = (text or '').strip()
    if not text:
        return False, 'No text provided'
    if not voice_id and not (reference_audio_path and os.path.exists(reference_audio_path)):
        return False, 'No voice selected — choose a voice from the list or upload a reference sample to clone.'
    # Fish Audio is the only narration engine. Anything else (an old template
    # or API call naming a removed engine) falls through to it rather than failing.
    engine_fn, engine_label = fish_audio_tts, 'Fish Audio'
    try:
        ok, err = engine_fn(text, output_wav_path, voice_id=voice_id, rate=rate,
                             reference_audio_path=reference_audio_path, language=language)
        if ok:
            return True, None
        print(f'{engine_label} TTS error: {err}')
        return False, f'{engine_label}: {err}'
    except Exception as e:
        print(f'{engine_label} unavailable: {e}')
        return False, f'{engine_label} unavailable: {e}'

def prepare_bgm_track(genre, scoring_mode, scoring_audio_path, duration, base_ts, fade_in=2.0, fade_out=3.0):
    """Produce a ready-to-mix BGM track (AAC .m4a, faded, trimmed to `duration`).
    Shared by the early beat-sync pass (approximate target duration) and the
    final mix pass (reused as-is if already prepared, else generated fresh).
    Returns (path_or_None, source) where source is 'uploaded' | 'ai_generated' | 'synth_fallback' | 'none'."""
    bgm_source = 'none'
    if not scoring_audio_path:
        return None, bgm_source
    if scoring_audio_path == 'GENERATE':
        gen_audio = os.path.join(app.config['UPLOAD_FOLDER'], f'gen_{base_ts}_{int(time.time()*1000)%100000}.m4a')
        acestep_ok = False
        try:
            prompt = GENRE_PROMPTS.get(genre, 'Cinematic background music, instrumental, no vocals')
            payload = {
                'prompt': prompt,
                'audio_duration': duration,
                'thinking': False,
                # 8 steps was well below ACE-Step's usable range and was the main
                # reason generated beds sounded thin/smeared. 27 is the model's
                # own documented default; raise ACE_STEP_STEPS for more quality
                # at proportionally more GPU time.
                'inference_steps': ACE_STEP_STEPS,
                'batch_size': 1,
                # "no vocals" in the prompt text is only a soft hint. ACE-Step
                # takes a dedicated lyrics field, and [inst] is its explicit
                # instrumental marker -- a far stronger guarantee of no vocals.
                'lyrics': '[inst]',
            }
            if ACE_STEP_NEGATIVE_PROMPT:
                payload['negative_prompt'] = ACE_STEP_NEGATIVE_PROMPT
            r = requests.post(f'{ACE_STEP_URL}/release_task', json=payload, timeout=10)
            data = r.json()
            task_id = data.get('data', {}).get('task_id')
            if task_id:
                for _ in range(60):
                    time.sleep(2)
                    q = requests.post(f'{ACE_STEP_URL}/query_result', json={'task_id_list': [task_id]}, timeout=5)
                    qd = q.json()
                    items = qd.get('data', [])
                    if items and items[0].get('status') == 1:
                        result = json.loads(items[0]['result'])
                        audio_path = result[0]['file'] if isinstance(result, list) else result.get('file', '')
                        if audio_path:
                            dl_url = f'{ACE_STEP_URL}{audio_path}'
                            resp = requests.get(dl_url, timeout=60)
                            with open(gen_audio, 'wb') as f:
                                f.write(resp.content)
                            if os.path.getsize(gen_audio) > 0:
                                acestep_ok = True
                        break
                    elif items and items[0].get('status') == 2:
                        break
        except Exception as e:
            print(f'ACE-Step error: {e}')
        if acestep_ok:
            bgm_source = 'ai_generated'
        else:
            lavfi_src = GENRE_LAVFI.get(genre, GENRE_LAVFI['default'])
            subprocess.run([FFMPEG, '-y', '-f', 'lavfi', '-i',
                            f'aevalsrc=exprs=\'{lavfi_src}\':d={duration}:s=44100:c=stereo',
                            '-af',
                            f'lowpass=f=6000,tremolo=f=0.15:d=0.4,volume=1.5,'
                            f'aecho=0.8:0.7:60:0.25,'
                            f'afade=t=in:d={fade_in},'
                            f'afade=t=out:st={max(duration - fade_out, 0)}:d={min(fade_out, duration)}',
                            '-c:a', 'aac', '-b:a', '192k', gen_audio],
                           capture_output=True, text=True, timeout=30)
            bgm_source = 'synth_fallback'
        if os.path.exists(gen_audio) and os.path.getsize(gen_audio) > 0:
            return gen_audio, bgm_source
        return None, 'none'
    else:
        processed_audio = os.path.join(app.config['UPLOAD_FOLDER'], f'score_{base_ts}_{int(time.time()*1000)%100000}.m4a')
        r = subprocess.run([FFMPEG, '-y', '-i', scoring_audio_path,
                            '-af', (f'atrim=duration={duration},'
                                    f'afade=t=in:d={fade_in},'
                                    f'afade=t=out:st={max(duration - fade_out, 0)}:d={min(fade_out, duration)}'),
                            '-c:a', 'aac', '-b:a', '192k', '-vn', processed_audio],
                           capture_output=True, text=True, timeout=60)
        if os.path.exists(processed_audio) and os.path.getsize(processed_audio) > 0:
            return processed_audio, 'uploaded'
        return None, 'none'

def finalize_bgm_duration(src_path, duration, base_ts, fade_in=2.0, fade_out=3.0):
    """Re-trim/pad + re-fade an already-generated BGM track to an exact final
    duration (used when the early beat-sync pass generated it against an
    estimate that ended up slightly off from the final trailer length)."""
    out = os.path.join(app.config['UPLOAD_FOLDER'], f'bgmfit_{base_ts}_{int(time.time()*1000)%100000}.m4a')
    r = subprocess.run([FFMPEG, '-y', '-i', src_path,
                        '-af', (f'atrim=duration={duration},apad=whole_dur={duration},'
                                f'afade=t=in:d={fade_in},'
                                f'afade=t=out:st={max(duration - fade_out, 0)}:d={min(fade_out, duration)}'),
                        '-c:a', 'aac', '-b:a', '192k', out],
                       capture_output=True, text=True, timeout=30)
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return out
    return None

def mux_card_vo(video_path, vo_path, trim_start, trim_end, output_path):
    """Replace a title/end card video's audio with a trimmed window of an uploaded
    VO file: [trim_start, trim_end) seconds of vo_path (trim_end=None means to the
    end of the file). The VO is padded with silence if shorter than the card video
    so the card keeps its full original length either way. Returns output_path on
    success, or None if the mux failed (caller should keep the card's original
    audio/video untouched in that case)."""
    cmd = [FFMPEG, '-y', '-i', video_path, '-ss', str(trim_start)]
    if trim_end is not None:
        cmd.extend(['-to', str(trim_end)])
    cmd.extend(['-i', vo_path,
                '-map', '0:v', '-map', '1:a',
                '-af', 'apad', '-shortest',
                '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', output_path])
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return output_path
    print(f'Card VO mux error ({video_path}): {r.stderr[:500]}')
    return None

def detect_beat_times(audio_path, duration):
    """Return a sorted list of beat timestamps (seconds) within an audio file, or
    an empty list if librosa/beat detection isn't available. Used to snap cut
    points onto the music so edits land 'on the beat'."""
    try:
        import librosa
        y, sr = librosa_load(audio_path, sr=22050, duration=duration)
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        times = librosa.frames_to_time(beat_frames, sr=sr)
        return sorted(_to_scalar(t) for t in times)
    except Exception as e:
        print(f'Beat detection error: {e}')
        return []

def nearest_beat(target, beats, lo, hi):
    """Nearest beat time to `target` within [lo, hi]; falls back to `target` if none in range."""
    candidates = [b for b in beats if lo <= b <= hi]
    if not candidates:
        return target
    return min(candidates, key=lambda b: abs(b - target))

# ---- whisper service — dialogue transcription to improve scene selection ----
def transcribe_video(path):
    """Transcribe the source video's dialogue via the local whisper service
    (WHISPER_URL, an OpenAI-compatible /v1/audio/transcriptions endpoint), with
    word-level timestamps. Returns (words, segments):
      words:    [{'start','end','word'}, ...]
      segments: [{'start','end','text'}, ...]
    Returns ([], []) if the service is unreachable or transcription fails —
    callers should treat that as 'feature unavailable' and continue without it.

    The video's audio is extracted to 16 kHz mono WAV first. This used to POST
    the whole source container -- for a 45-minute episode that meant pushing
    several GB over HTTP so the service could demux and discard the video track
    anyway. Whisper resamples to 16 kHz mono internally regardless, so doing it
    here costs one cheap ffmpeg pass and cuts the upload by ~100x."""
    audio_path = None
    try:
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'],
                                  f'stt_{uuid.uuid4().hex}.wav')
        try:
            run_ffmpeg([FFMPEG, '-y', '-i', path, '-vn', '-ac', '1', '-ar', '16000',
                        '-c:a', 'pcm_s16le', audio_path],
                       timeout=FFMPEG_LONG_TIMEOUT, label='STT audio extract')
        except MediaToolTimeout as e:
            print(f'Whisper: audio extraction timed out ({e}); skipping transcription.')
            return [], []
        if not (os.path.exists(audio_path) and os.path.getsize(audio_path) > 0):
            print('Whisper: could not extract an audio track from the source; skipping transcription.')
            return [], []
        upload_name = os.path.splitext(os.path.basename(path))[0] + '.wav'
        with open(audio_path, 'rb') as f:
            r = requests.post(
                f'{WHISPER_URL}/v1/audio/transcriptions',
                files={'file': (upload_name, f, 'audio/wav')},
                data={
                    'model': WHISPER_MODEL,
                    'response_format': 'verbose_json',
                    'timestamp_granularities[]': 'word',
                },
                timeout=600,
            )
        r.raise_for_status()
        data = r.json()
        words, segments = [], []
        for seg in data.get('segments', []):
            text = (seg.get('text') or '').strip()
            if text:
                segments.append({'start': seg['start'], 'end': seg['end'], 'text': text})
        for w in data.get('words', []):
            word = (w.get('word') or '').strip()
            if word:
                words.append({'start': w['start'], 'end': w['end'], 'word': word})
        return words, segments
    except Exception as e:
        print(f'Whisper transcription error (service at {WHISPER_URL}): {e}')
        return [], []
    finally:
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except OSError:
                pass

def nearest_word_boundary(target, boundaries, max_snap=0.35):
    """Nearest timestamp in `boundaries` to `target`, but only if within
    `max_snap` seconds — otherwise returns `target` unchanged (no nearby word
    to snap to, e.g. a silent B-roll clip, so leave the cut point as-is)."""
    if not boundaries:
        return target
    candidates = [b for b in boundaries if abs(b - target) <= max_snap]
    if not candidates:
        return target
    return min(candidates, key=lambda b: abs(b - target))

def librosa_load(path, sr=22050, mono=True, duration=None):
    """librosa.load, but never via the deprecated audioread fallback.

    soundfile cannot open compressed/container formats (.mov, .m4a, .mp4), so
    librosa silently falls back to audioread -- which emits a deprecation warning,
    is removed in librosa 1.0, and is markedly slower. Decoding to a temporary
    PCM WAV with ffmpeg first keeps everything on the soundfile path."""
    import librosa as _lb
    ext = os.path.splitext(path)[1].lower()
    tmp = None
    try:
        if ext not in ('.wav', '.flac', '.ogg', '.aiff', '.aif'):
            tmp = os.path.join(tempfile.gettempdir(), f'lb_{uuid.uuid4().hex}.wav')
            cmd = [FFMPEG, '-y', '-i', path, '-vn', '-ac', '1' if mono else '2',
                   '-ar', str(int(sr)), '-c:a', 'pcm_s16le']
            if duration:
                cmd += ['-t', str(duration)]
            cmd.append(tmp)
            try:
                run_ffmpeg(cmd, timeout=180, label='librosa decode')
            except MediaToolTimeout:
                tmp = None
            if tmp and not (os.path.exists(tmp) and os.path.getsize(tmp) > 0):
                tmp = None
        return _lb.load(tmp or path, sr=sr, mono=mono, duration=duration)
    finally:
        if tmp and os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass

def tc_seconds(tc):
    """Seconds from a PySceneDetect FrameTimecode.

    .get_seconds() is deprecated in favour of the .seconds property, but the
    property does not exist on older releases -- so prefer it and fall back."""
    v = getattr(tc, 'seconds', None)
    return v if v is not None else tc_seconds(tc)

def tc_frames(tc):
    """Frame number from a PySceneDetect FrameTimecode (see tc_seconds)."""
    v = getattr(tc, 'frame_num', None)
    return v if v is not None else tc_frames(tc)

def detect_scenes(path, threshold=30.0, min_scene_len_sec=0.5, downscale=2,
                   detector='content', adaptive_threshold=3.0, min_content_val=15.0):
    """Run PySceneDetect over `path` and return the raw scene list
    [(start_timecode, end_timecode), ...].

    threshold, min_scene_len_sec, downscale, detector, and adaptive_threshold
    are all parameters (not hardcoded) so every caller -- the live preview
    endpoint, the vision-analyze endpoint, and the actual trailer-generation
    job -- can agree on the same detection settings. That agreement used to
    be incomplete: downscale defaulted to None here but the real job always
    passed downscale=2, so "the same threshold" in the preview didn't
    guarantee the same cuts, since downscaling changes the frame-to-frame
    content-difference PySceneDetect measures, not just the coordinate scale.
    downscale now defaults to 2 everywhere so preview, vision-analyze, and
    the real job detect identically unless a caller deliberately overrides it.

    min_scene_len_sec filters out sub-fragment "scenes" (whip-pans, motion
    blur, flash cuts) that PySceneDetect's frame-count default would otherwise
    let through; it's converted to frames using the source's actual frame rate.

    detector picks the underlying algorithm:
      - 'content' (default): PySceneDetect's ContentDetector, a per-frame
        hue/saturation/luma difference against a fixed threshold. Good for
        clean hard cuts; prone to false positives on fast pans, handheld
        motion, or zooms, since rapid camera movement produces a large
        frame-to-frame difference that looks like a cut.
      - 'adaptive': AdaptiveDetector, which compares each frame's content
        score against a rolling local average (window_width frames either
        side) rather than a fixed threshold, so a sustained change from
        panning/motion doesn't trip it the way a genuine cut does.
        adaptive_threshold is a ratio (current score vs. local average, not
        the 0-100 scale 'threshold' uses) -- PySceneDetect's own default of
        3.0 is a reasonable starting point. min_content_val is a floor below
        which a frame is never flagged as a cut regardless of the ratio,
        which keeps near-static footage (e.g. a held shot) from tripping on
        noise alone.
      Neither detector reliably catches cross-dissolves between two live
      scenes (a gradual blend has no single frame with a large difference to
      key on) -- that's a fundamentally different detection problem neither
      PySceneDetect algorithm targets. AdaptiveDetector's lower effective
      noise floor tends to catch *more* of a dissolve's ramp than
      ContentDetector does, but it is not a dedicated fix.
    """
    video = open_video(path)
    fps = video.frame_rate or 30.0
    min_scene_len = max(1, int(round(min_scene_len_sec * fps)))
    sm = SceneManager()
    if downscale:
        # video.set_downscale_factor() doesn't exist on the installed
        # PySceneDetect/backend combo -- it silently no-opped through the
        # try/except that used to wrap this, so `downscale` was never
        # actually being applied. Downscale lives on SceneManager in this
        # version, and SceneManager also defaults to auto-picking its own
        # factor (auto_downscale=True) which overrides an explicit one
        # unless it's turned off first -- so both parts are needed, not
        # just swapping which object the call is made on.
        sm.auto_downscale = False
        sm.downscale = downscale
    if detector == 'adaptive':
        sm.add_detector(AdaptiveDetector(adaptive_threshold=adaptive_threshold,
                                          min_scene_len=min_scene_len,
                                          min_content_val=min_content_val))
    else:
        sm.add_detector(ContentDetector(threshold=threshold, min_scene_len=min_scene_len))
    sm.detect_scenes(video)
    return sm.get_scene_list()

def _scene_detector_params(form):
    """Parses+clamps the shared scene-detection form fields (threshold,
    min_scene_len, detector, adaptive_threshold) the same way everywhere
    they're accepted -- preview, vision-analyze, and the real job -- so the
    three can't quietly drift out of sync the way preview/job downscale
    once did. Returns a kwargs dict ready to splat into detect_scenes()."""
    try:
        threshold = max(1.0, min(100.0, float(form.get('scene_threshold', form.get('threshold', 30.0)))))
    except (TypeError, ValueError):
        threshold = 30.0
    try:
        min_scene_len_sec = max(0.1, min(5.0, float(form.get('min_scene_len', 0.5))))
    except (TypeError, ValueError):
        min_scene_len_sec = 0.5
    detector = form.get('detector', 'content')
    if detector not in ('content', 'adaptive'):
        detector = 'content'
    try:
        adaptive_threshold = max(1.0, min(10.0, float(form.get('adaptive_threshold', 3.0))))
    except (TypeError, ValueError):
        adaptive_threshold = 3.0
    return dict(threshold=threshold, min_scene_len_sec=min_scene_len_sec,
                detector=detector, adaptive_threshold=adaptive_threshold)

def get_video_info(path):
    cap = cv2.VideoCapture(path)
    info = {
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': round(cap.get(cv2.CAP_PROP_FPS), 2),
        'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    info['duration_sec'] = round(info['total_frames'] / info['fps'], 2) if info['fps'] > 0 else 0
    cap.release()
    return info

# ---- API ----

@app.route('/api/opencv/info', methods=['POST'])
def api_info():
    path, err = load_video(request)
    if not path:
        return jsonify(error=err), 400
    return jsonify(video_info=get_video_info(path))

@app.route('/api/opencv/analyze', methods=['POST'])
def api_analyze():
    path, err = load_video(request)
    if not path:
        return jsonify(error=err), 400
    n = min(int(request.form.get('num_frames', 10)), 100)
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total // n, 1)
    frames = []
    for i in range(0, total, step):
        if len(frames) >= n:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, f = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=20, qualityLevel=0.01, minDistance=10)
        frames.append({'idx': i, 'shape': list(f.shape),
                       'mean_bgr': [round(float(c), 1) for c in cv2.mean(f)[:3]],
                       'brightness': round(float(np.mean(gray)), 1),
                       'edge_pixels': int(np.sum(edges > 0)),
                       'corners': len(corners) if corners is not None else 0})
    cap.release()
    return jsonify(frames=frames)

@app.route('/api/scenedetect/detect', methods=['POST'])
@require_permission('scene_detection')
def api_sd():
    path, err = load_video(request)
    if not path:
        return jsonify(error=err), 400
    # Shared parser + the same downscale=2 the real job uses (detect_scenes'
    # default) -- this is what makes "Preview scene cuts" an honest preview
    # of what generation will actually do, rather than a full-res detection
    # that can disagree with the (downscaled) real run at the margins.
    scene_list = detect_scenes(path, **_scene_detector_params(request.form))
    scenes = [{'scene': i+1, 'start': s.get_timecode(), 'end': e.get_timecode(),
               'start_sec': round(tc_seconds(s), 2), 'end_sec': round(tc_seconds(e), 2),
               'duration': round(tc_seconds(e) - tc_seconds(s), 2)}
              for i, (s, e) in enumerate(scene_list)]
    return jsonify(scenes=scenes)

def _ensure_readable(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.mov', '.mkv', '.flv', '.wmv', '.webm'):
        cap = cv2.VideoCapture(path)
        ret, _ = cap.read()
        cap.release()
        if not ret:
            mp4_path = os.path.splitext(path)[0] + '_converted.mp4'
            r = run_ffmpeg([FFMPEG, '-y', '-i', path, '-c:v', 'libx264', '-preset', 'ultrafast',
                                '-crf', '28', '-pix_fmt', 'yuv420p', '-an', mp4_path],
                           timeout=FFMPEG_LONG_TIMEOUT, label='preview transcode')
            if r.returncode == 0 and os.path.exists(mp4_path):
                return mp4_path
    return path

@app.route('/api/media/playable', methods=['POST'])
def api_media_playable():
    """Re-encodes a staged file to H.264/AAC MP4 for the browser Player.

    Distinct from _ensure_readable(): that one strips audio and only checks
    whether *OpenCV* can decode a frame, which is the wrong question here --
    OpenCV (via its ffmpeg backend) opens ProRes/DNxHD/MXF just fine, but no
    browser decodes them natively, so a HIRES mat in one of those often loads
    with duration/controls but a black frame, or an outright media error. Rather
    than guess server-side which containers a given browser supports, the Player
    calls this reactively -- only once its <video> element actually reports an
    error -- and gets back a guaranteed-playable copy with audio intact."""
    name = secure_filename(request.form.get('filename', ''))
    if not name or name != request.form.get('filename', ''):
        return jsonify(ok=False, error='Invalid filename.'), 400
    src = os.path.join(app.config['UPLOAD_FOLDER'], name)
    if not os.path.isfile(src):
        return jsonify(ok=False, error='That file is no longer staged -- pick it again.'), 404

    out_name = f'playable_{os.path.splitext(name)[0]}.mp4'
    out_path = os.path.join(app.config['UPLOAD_FOLDER'], out_name)
    if not (os.path.exists(out_path) and os.path.getsize(out_path) > 0):
        try:
            r = run_ffmpeg([FFMPEG, '-y', '-i', src,
                            '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20',
                            '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '192k',
                            '-movflags', '+faststart', out_path],
                           timeout=FFMPEG_LONG_TIMEOUT, label='player playback transcode')
        except MediaToolTimeout as e:
            return jsonify(ok=False, error=f'Conversion took too long: {e}'), 504
        if not (os.path.exists(out_path) and os.path.getsize(out_path) > 0):
            return jsonify(ok=False, error='This file could not be converted for browser playback. '
                                           f'ffmpeg error: {r.stderr[-400:]}'), 502
    return jsonify(ok=True, url=f'/uploads/{out_name}')

# ---- Ollama Vision ----

OLLAMA_URL = os.environ.get('OLLAMA_URL', 'http://localhost:11434')

@app.route('/api/vision/analyze', methods=['POST'])
@require_permission('scene_detection')
def api_vision():
    path, err = load_video(request)
    if not path:
        return jsonify(error=err), 400
    prompt = request.form.get('prompt', 'Describe what is happening in this video frame in 1-2 sentences.')
    num_frames = min(int(request.form.get('num_frames', 5)), 20)
    model = request.form.get('model', 'llama3.2-vision:11b')

    path = _ensure_readable(path)
    # Basename under UPLOAD_FOLDER, handed back so the frontend can play the
    # scene straight from /uploads/<name> -- lets "Analyze with AI" double as
    # a scene-cut preview you can actually watch, not just read timecodes for.
    video_filename = os.path.basename(path)

    # Was hardcoded to threshold=30.0 regardless of what the generator form
    # (or this tab) was set to, so this tab silently stopped matching what
    # generation would actually do the moment anyone tuned the threshold.
    # Same shared parser + default downscale as the preview and real job now.
    scene_list = detect_scenes(path, **_scene_detector_params(request.form))
    scenes = [{'scene': i+1, 'start': tc_seconds(s), 'end': tc_seconds(e),
               'start_tc': s.get_timecode(), 'end_tc': e.get_timecode(),
               'duration': round(tc_seconds(e) - tc_seconds(s), 2)}
              for i, (s, e) in enumerate(scene_list)]

    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    frames_to_analyze = []
    if scenes:
        for sc in scenes:
            mid_sec = (sc['start'] + sc['end']) / 2
            mid_frame = int(mid_sec * fps) if fps else 0
            frames_to_analyze.append({'frame_idx': mid_frame, 'time_sec': round(mid_sec, 2), 'scene': sc})
    else:
        step = max(total // num_frames, 1) if total > num_frames else 1
        for i in range(0, total, step):
            if len(frames_to_analyze) >= num_frames:
                break
            ts = round(i / fps, 2) if fps > 0 else 0
            frames_to_analyze.append({'frame_idx': i, 'time_sec': ts, 'scene': None})

    results = []
    for fa in frames_to_analyze[:num_frames]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fa['frame_idx'])
        ret, frame = cap.read()
        if not ret:
            continue
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64 = base64.b64encode(buf.tobytes()).decode()

        scene_ctx = ''
        if fa['scene']:
            s = fa['scene']
            scene_ctx = f' (Scene {s["scene"]}, {s["start_tc"]}-{s["end_tc"]}, {s["duration"]}s)'
        full_prompt = prompt + scene_ctx

        try:
            r = requests.post(f'{OLLAMA_URL}/api/generate', json={
                'model': model, 'prompt': full_prompt, 'stream': False,
                'images': [b64]
            }, timeout=300)
            data = r.json()
            resp = data.get('response', '')
        except Exception as e:
            resp = f'Error: {e}'

        entry = {'frame_idx': fa['frame_idx'], 'time_sec': fa['time_sec'], 'ollama_response': resp}
        if fa['scene']:
            entry['scene'] = fa['scene']['scene']
            entry['scene_start'] = fa['scene']['start_tc']
            entry['scene_end'] = fa['scene']['end_tc']
            entry['scene_duration'] = fa['scene']['duration']
            # Numeric seconds alongside the display timecodes above, so the
            # frontend can seek a <video> element directly (currentTime wants
            # a float, not a "00:00:12.500" string).
            entry['scene_start_sec'] = fa['scene']['start']
            entry['scene_end_sec'] = fa['scene']['end']
        results.append(entry)

    cap.release()
    return jsonify(frames_analyzed=len(results), total_scenes=len(scenes), results=results,
                    video_filename=video_filename)

@app.errorhandler(413)
def too_large(e):
    return jsonify(error='File too large (max 2GB).'), 413

def _model_supports_vision(name):
    # /api/tags does not include per-model capabilities, so we have to ask
    # /api/show for each model individually to find out if it's a vision model.
    try:
        r = requests.post(f'{OLLAMA_URL}/api/show', json={'model': name}, timeout=10)
        data = r.json()
        caps = data.get('capabilities', [])
        if caps:
            return 'vision' in caps
        # Older Ollama versions don't return `capabilities` at all - fall back
        # to checking the projector/family info that vision models expose.
        details = data.get('details', {}) or {}
        families = (details.get('families') or []) + [details.get('family', '')]
        if any('clip' in f.lower() or 'mllama' in f.lower() or 'vision' in f.lower() for f in families if f):
            return True
        return 'vision' in name.lower() or 'vl' in name.lower()
    except Exception:
        # If we can't introspect the model, guess from its name rather than
        # silently dropping it from the list.
        return 'vision' in name.lower() or 'vl' in name.lower()

@app.route('/api/vision/models')
@require_permission('scene_detection')
def api_vision_models():
    try:
        r = requests.get(f'{OLLAMA_URL}/api/tags', timeout=10)
        names = [m['name'] for m in r.json().get('models', [])]
    except Exception:
        return jsonify(models=[])
    if not names:
        return jsonify(models=[])
    with ThreadPoolExecutor(max_workers=min(8, len(names))) as ex:
        flags = list(ex.map(_model_supports_vision, names))
    models = [n for n, ok in zip(names, flags) if ok]
    return jsonify(models=models)

# ---- AI Chat ----
# A general-purpose chat panel over whatever models Ollama has installed --
# separate from the vision/rating pipeline, for testing a model's behavior,
# drafting text, or just asking it something directly.
CHAT_TIMEOUT = int(os.environ.get('CHAT_TIMEOUT', 180))
CHAT_MAX_HISTORY = int(os.environ.get('CHAT_MAX_HISTORY', 60))  # messages, not turns

# ---- Chat attachments (images + documents) ----
# Images ride directly in the /api/chat request body -- Ollama's own schema is
# messages[].images, a list of base64 strings with no data-URL prefix (see
# docs.ollama.com/capabilities/vision) -- so the browser encodes them and sends
# them straight through with no server round trip needed. Documents are
# different: a PDF can't be read as text in the browser, so those go through
# the extraction endpoint below first and the resulting text is folded into the
# message content client-side, the same as if the user had typed it.
CHAT_MAX_IMAGES_PER_MESSAGE = int(os.environ.get('CHAT_MAX_IMAGES_PER_MESSAGE', 4))
CHAT_MAX_IMAGE_BYTES = int(os.environ.get('CHAT_MAX_IMAGE_BYTES', 8 * 1024 * 1024))
CHAT_ATTACH_MAX_BYTES = int(os.environ.get('CHAT_ATTACH_MAX_BYTES', 8 * 1024 * 1024))
CHAT_ATTACH_MAX_CHARS = int(os.environ.get('CHAT_ATTACH_MAX_CHARS', 20000))
CHAT_TEXT_EXTS = {'.txt', '.md', '.markdown', '.csv', '.tsv', '.json', '.log', '.yaml', '.yml',
                  '.ini', '.cfg', '.conf', '.xml', '.html', '.htm', '.css', '.py', '.js', '.ts',
                  '.jsx', '.tsx', '.java', '.c', '.h', '.cpp', '.hpp', '.go', '.rs', '.rb', '.php',
                  '.sh', '.sql', '.srt', '.vtt'}

@app.route('/api/chat/extract_file', methods=['POST'])
@require_permission('ai_chat')
def api_chat_extract_file():
    """Extracts plain text from an uploaded document for a chat attachment.

    PDFs need pypdf (an optional dependency -- this degrades to a clear error
    rather than crashing the whole app if it isn't installed, since not every
    deployment needs PDF support). Anything else with a recognised text-ish
    extension is decoded as UTF-8 directly; everything else is rejected rather
    than silently attaching binary garbage as "context"."""
    f = request.files.get('file')
    if not f or not f.filename:
        return jsonify(ok=False, error='No file provided.'), 400
    name = secure_filename(f.filename)
    if not name:
        return jsonify(ok=False, error='Invalid filename.'), 400
    ext = os.path.splitext(name)[1].lower()

    raw = f.read(CHAT_ATTACH_MAX_BYTES + 1)
    if len(raw) > CHAT_ATTACH_MAX_BYTES:
        return jsonify(ok=False, error=f'That file is larger than the '
                       f'{CHAT_ATTACH_MAX_BYTES // (1024*1024)}MB attachment limit.'), 400
    if not raw:
        return jsonify(ok=False, error='That file is empty.'), 400

    if ext == '.pdf':
        try:
            import pypdf
        except ImportError:
            return jsonify(ok=False, error='PDF attachments need the "pypdf" package on the '
                           'server (pip install pypdf --break-system-packages), which is not '
                           'installed here. Plain text and code files work without it.'), 501
        try:
            reader = pypdf.PdfReader(io.BytesIO(raw))
            if reader.is_encrypted:
                return jsonify(ok=False, error='That PDF is password-protected.'), 400
            text = '\n\n'.join((page.extract_text() or '') for page in reader.pages)
        except Exception as e:
            return jsonify(ok=False, error=f'Could not read that PDF: {e}'), 400
    elif ext in CHAT_TEXT_EXTS or ext == '':
        try:
            text = raw.decode('utf-8')
        except UnicodeDecodeError:
            text = raw.decode('utf-8', errors='replace')
    else:
        return jsonify(ok=False, error=f'Unsupported file type "{ext}". Supported: PDF, plain '
                       'text, and common code/markup files. Images should be attached directly, '
                       'not through this button.'), 400

    text = text.strip()
    if not text:
        return jsonify(ok=False, error='No extractable text was found in that file '
                       '(a scanned/image-only PDF has no text layer to read).'), 400
    truncated = len(text) > CHAT_ATTACH_MAX_CHARS
    if truncated:
        text = text[:CHAT_ATTACH_MAX_CHARS]
    return jsonify(ok=True, filename=name, text=text, truncated=truncated, chars=len(text))

@app.route('/api/chat/models')
@require_permission('ai_chat')
def api_chat_models():
    """Every model Ollama has installed, unfiltered -- unlike /api/vision/models
    this doesn't restrict to vision-capable ones, since chat has no use for that
    distinction and the /api/show introspection per model is pure overhead here."""
    try:
        r = requests.get(f'{OLLAMA_URL}/api/tags', timeout=10)
        names = sorted(m['name'] for m in r.json().get('models', []))
        return jsonify(ok=True, models=names)
    except Exception as e:
        return jsonify(ok=False, models=[], error=f'Could not reach Ollama at {OLLAMA_URL}: {e}')

@app.route('/api/chat', methods=['POST'])
@require_permission('ai_chat')
def api_chat():
    data = request.get_json(silent=True) or {}
    model = (data.get('model') or '').strip()
    if not model:
        return jsonify(ok=False, error='Pick a model first.'), 400

    messages = data.get('messages')
    if not isinstance(messages, list) or not messages:
        return jsonify(ok=False, error='No conversation to send.'), 400
    # Only well-formed {role, content} pairs, and bounded -- this is a JSON body
    # built from the page's own running chat history, but treat it the same as
    # any other untrusted input rather than assuming the client behaved.
    clean = []
    for m in messages[-CHAT_MAX_HISTORY:]:
        if not isinstance(m, dict):
            continue
        role = m.get('role')
        content = m.get('content')
        if role not in ('user', 'assistant') or not isinstance(content, str) or not content.strip():
            continue
        entry = {'role': role, 'content': content}
        # Images only make sense on a user turn, and only up to a sane cap --
        # a single message with dozens of embedded base64 images would both
        # balloon the request and almost certainly exceed what a vision model
        # can usefully attend to at once.
        imgs = m.get('images')
        if role == 'user' and isinstance(imgs, list) and imgs:
            clean_imgs = []
            for img in imgs[:CHAT_MAX_IMAGES_PER_MESSAGE]:
                if not isinstance(img, str) or not img:
                    continue
                # Base64 expands input size by ~4/3 -- check the encoded string
                # length against that inflated bound rather than decoding every
                # image just to measure it.
                if len(img) > CHAT_MAX_IMAGE_BYTES * 4 // 3:
                    continue
                clean_imgs.append(img)
            if clean_imgs:
                entry['images'] = clean_imgs
        clean.append(entry)
    if not clean:
        return jsonify(ok=False, error='No conversation to send.'), 400

    system = (data.get('system') or '').strip()
    if system:
        clean = [{'role': 'system', 'content': system}] + clean

    # Reasoning models put their chain-of-thought in a separate `thinking` field
    # (see the AI-scoring fix earlier: an empty `content` with a full `thinking`
    # field is not a failure, it's the model reasoning silently). Chat has no
    # token budget cap the way scene rating does, so the specific failure mode
    # that caused there -- reasoning eating a *tight* budget -- doesn't apply,
    # but the fallback costs nothing to keep for the rare model that only ever
    # populates `thinking`.
    think = bool(data.get('think'))
    payload = {'model': model, 'messages': clean, 'stream': False, 'think': think}
    try:
        r = requests.post(f'{OLLAMA_URL}/api/chat', json=payload, timeout=CHAT_TIMEOUT)
        resp = r.json()
    except requests.exceptions.Timeout:
        return jsonify(ok=False, error=f'Ollama did not respond within {CHAT_TIMEOUT}s.'), 504
    except Exception as e:
        return jsonify(ok=False, error=f'Could not reach Ollama at {OLLAMA_URL}: {e}'), 502
    if resp.get('error'):
        return jsonify(ok=False, error=resp['error']), 502

    msg = resp.get('message') or {}
    content = (msg.get('content') or '').strip()
    thinking = (msg.get('thinking') or '').strip()
    if not content and thinking:
        content = thinking
    if not content:
        return jsonify(ok=False, error='The model returned an empty response.'), 502
    return jsonify(ok=True, content=content, thinking=thinking or None, model=resp.get('model') or model)

# ---- Trailer Generator (ffmpeg) ----

@app.route('/api/trailer/generate', methods=['POST'])
@require_permission('promo_generation')
def api_trailer():
    path, orig_name = load_video(request)
    if not path:
        return jsonify(error=orig_name), 400

    # Rating mode: 'ai' (OpenCV + Ollama Vision) or 'ai_stt' (adds faster-whisper
    # dialogue transcription on top of Vision scoring). 'ai_stt' is normalized down to
    # 'ai' for the scoring logic below, with whisper_enhance derived from it.
    mode = request.form.get('mode', 'ai')
    if mode not in ('ai', 'ai_stt'):
        mode = 'ai'
    mode_includes_stt = (mode == 'ai_stt')
    if mode_includes_stt:
        mode = 'ai'
    genre = request.form.get('genre', '').strip()
    scoring_mode = request.form.get('scoring_mode', 'generate')
    trailer_length = int(request.form.get('trailer_length', 15))
    if trailer_length not in (15, 30, 45, 60):
        trailer_length = 30
    try:
        max_scene_dur = float(request.form.get('max_scene_dur', '') or 0)
        max_scene_dur = max_scene_dur if max_scene_dur > 0 else None
    except ValueError:
        max_scene_dur = None
    try:
        scene_threshold = max(1.0, min(100.0, float(request.form.get('scene_threshold', 30.0))))
    except ValueError:
        scene_threshold = 30.0
    try:
        min_scene_len_sec = max(0.1, min(5.0, float(request.form.get('min_scene_len', 0.5))))
    except ValueError:
        min_scene_len_sec = 0.5
    detector = request.form.get('detector', 'content')
    if detector not in ('content', 'adaptive'):
        detector = 'content'
    try:
        adaptive_threshold = max(1.0, min(10.0, float(request.form.get('adaptive_threshold', 3.0))))
    except ValueError:
        adaptive_threshold = 3.0
    transition = request.form.get('transition', 'fade')
    transition_matte_path = None
    if genre in GENRE_PRESETS:
        preset = GENRE_PRESETS[genre]
        transition = preset['transition']
        xfade_dur = preset['xfade_dur']
        if scoring_mode not in ('upload', 'generate'):
            scoring_mode = 'generate'
    else:
        if transition not in VALID_TRANSITIONS:
            transition = 'fade'
        xfade_dur = float(request.form.get('xfade_dur', 0.3))
        xfade_dur = max(0.1, min(2.0, xfade_dur))
        if transition == 'custom_matte':
            if 'transition_matte' in request.files and request.files['transition_matte'].filename:
                f = request.files['transition_matte']
                fn = secure_filename(f.filename)
                if fn:
                    transition_matte_path = os.path.join(
                        app.config['UPLOAD_FOLDER'], f'transmatte_{int(time.time())}{os.path.splitext(fn)[1]}')
                    f.save(transition_matte_path)
            if not transition_matte_path:
                # Selected "Custom" but didn't upload anything — fall back rather
                # than fail the whole job over a missing optional asset.
                transition = 'fade'
    target_loudness = float(request.form.get('target_loudness', -14))
    true_peak = float(request.form.get('true_peak', -1.5))
    music_duck_db = float(request.form.get('music_duck_db', -3))
    duck_depth_db = float(request.form.get('duck_depth_db', -15))
    duck_release_hold = float(request.form.get('duck_release_hold', 0.4))
    beat_match = request.form.get('beat_match') == 'on'
    broadcast_stereo = request.form.get('broadcast_stereo') == 'on'
    model = request.form.get('model', 'qwen3-vl:8b')

    # SFX source selection: 'genre' (AI-generate/synth from the genre preset),
    # 'upload' (stamp a user-supplied one-shot at every cut), or 'none'.
    default_sfx_mode = 'genre' if (genre in GENRE_PRESETS and GENRE_PRESETS[genre].get('sfx')) else 'none'
    sfx_mode = request.form.get('sfx_mode', default_sfx_mode)
    if sfx_mode not in ('genre', 'upload', 'none'):
        sfx_mode = 'none'
    sfx_upload_path = None
    if sfx_mode == 'upload':
        sfx_upload_path = _resolve_upload('sfx_upload', AUDIO_EXTENSIONS)
    if sfx_mode == 'upload' and not sfx_upload_path:
        sfx_mode = 'none'  # nothing usable was uploaded, don't silently fall back to genre SFX


    # Voiceover: 'none' (skip), 'upload' (use a supplied VO track as-is), or
    # 'tts' (generate from typed text via a local TTS engine). Whichever
    # source, music/SFX/original audio get ducked underneath it, same as
    # scoring audio ducks under the original dialogue.
    vo_mode = request.form.get('vo_mode', 'none')
    if vo_mode not in ('none', 'upload', 'tts'):
        vo_mode = 'none'
    vo_upload_path = None
    if vo_mode == 'upload':
        vo_upload_path = _resolve_upload('vo_upload', AUDIO_EXTENSIONS)
    if vo_mode == 'upload' and not vo_upload_path:
        vo_mode = 'none'
    vo_text = request.form.get('vo_text', '').strip()
    if vo_mode == 'tts' and not vo_text:
        vo_mode = 'none'
    vo_voice = request.form.get('vo_voice', '').strip() or None
    vo_language = request.form.get('vo_language', '').strip() or None
    vo_engine = request.form.get('vo_engine', 'fish_audio').strip()
    if vo_engine != 'fish_audio':
        vo_engine = 'fish_audio'
    vo_ref_upload_path = None
    if vo_mode == 'tts' and 'vo_ref_upload' in request.files and request.files['vo_ref_upload'].filename:
        f = request.files['vo_ref_upload']
        fn = secure_filename(f.filename)
        if fn:
            vo_ref_upload_path = os.path.join(app.config['UPLOAD_FOLDER'], f'vorefsrc_{int(time.time())}{os.path.splitext(fn)[1]}')
            f.save(vo_ref_upload_path)
            vo_voice = None  # an uploaded reference clone takes priority over a picked registered voice
    try:
        vo_rate = int(request.form.get('vo_rate', 175))
    except ValueError:
        vo_rate = 175
    try:
        vo_start = max(0.0, float(request.form.get('vo_start', 0)))
    except ValueError:
        vo_start = 0.0
    try:
        vo_volume = max(0.3, min(3.0, float(request.form.get('vo_volume', 1.15))))
    except ValueError:
        vo_volume = 1.15
    # Trim points *within the uploaded VO file itself* (which portion of that
    # source clip to use) — distinct from vo_start above, which places the
    # (already-trimmed) narration on the trailer's own timeline.
    try:
        vo_trim_start = max(0.0, float(request.form.get('vo_trim_start', 0) or 0))
    except ValueError:
        vo_trim_start = 0.0
    vo_trim_end_raw = request.form.get('vo_trim_end', '').strip()
    try:
        vo_trim_end = float(vo_trim_end_raw) if vo_trim_end_raw else None
    except ValueError:
        vo_trim_end = None
    if vo_trim_end is not None and vo_trim_end <= vo_trim_start:
        vo_trim_end = None

    # Sync cuts to the beat of the background music (only meaningful when a
    # music track is actually used). Requires prepping the BGM before scene
    # selection instead of after, so cut points can be nudged onto beats.
    sync_beats = request.form.get('sync_beats') == 'on' and scoring_mode != 'none'
    whisper_enhance = mode_includes_stt

    end_card_path = None
    schedule_card_path = None
    scoring_audio_path = None
    if scoring_mode == 'upload':
        scoring_audio_path = _resolve_upload('scoring_audio', AUDIO_EXTENSIONS)
    if scoring_mode == 'generate':
        scoring_audio_path = 'GENERATE'  # flag to generate ambient
    end_card_path = _resolve_upload('end_card_video', ALLOWED_EXTENSIONS)
    schedule_card_path = _resolve_upload('schedule_video', ALLOWED_EXTENSIONS)

    # Optional VO tracks for the title card ("end_card_video" field, despite the
    # name) and end card ("schedule_video" field) — each can have its own
    # uploaded narration audio, muxed on in place of whatever audio the card
    # video already has, trimmed to a chosen [start, end) window of the source file.
    def _parse_card_vo(file_key, start_key, end_key):
        path = _resolve_upload(file_key, AUDIO_EXTENSIONS)
        try:
            start = max(0.0, float(request.form.get(start_key, 0) or 0))
        except ValueError:
            start = 0.0
        end_raw = request.form.get(end_key, '').strip()
        try:
            end = float(end_raw) if end_raw else None
        except ValueError:
            end = None
        if end is not None and end <= start:
            end = None
        return path, start, end

    title_card_vo_path, title_card_vo_start, title_card_vo_end = _parse_card_vo(
        'title_card_vo', 'title_card_vo_start', 'title_card_vo_end')
    end_card_vo_path, end_card_vo_start, end_card_vo_end = _parse_card_vo(
        'end_card_vo', 'end_card_vo_start', 'end_card_vo_end')

    # ---- Show template fill-in ----
    # A template IS the configuration for a programme: its genre, transition,
    # lengths, audio targets and voice choice, plus its music bed, SFX, VO and
    # cards. It is NOT an alternative to picking a genre -- genre is one of the
    # fields a template carries. Selecting a show therefore fills in everything
    # the request didn't specify, and nothing is mutually exclusive.
    #
    # Anything explicitly sent always wins: a template is a set of defaults, so a
    # one-off replacement bed or a different transition for a single episode works
    # without editing the show.
    #
    # In the browser the form is populated client-side the moment a show is
    # picked, so the request already carries these values and this block mostly
    # no-ops. It matters for API callers that just name a template.
    template_applied = None
    _tpl_pending = {}
    _tid_raw = (request.form.get('template_id') or '').strip()
    if _tid_raw.isdigit():
        tpl = template_get(int(_tid_raw))
        if tpl:
            filled = []
            tpl_settings = template_settings(tpl)

            def _tpl_number(current, key, default):
                """Uses the template's stored value only where the form left the
                field at its default -- an explicit form value always wins."""
                stored = tpl.get(key)
                return stored if (stored is not None and current == default) else current

            # Genre: only from the template if the request didn't name one. When it
            # does supply one, the genre preset block above has already applied that
            # genre's transition and xfade.
            if not genre:
                tpl_genre = tpl.get('genre') or tpl_settings.get('genre')
                if tpl_genre in GENRE_PRESETS:
                    genre = tpl_genre
                    preset = GENRE_PRESETS[genre]
                    if 'transition' not in request.form:
                        transition = preset['transition']
                    if 'xfade_dur' not in request.form:
                        xfade_dur = preset['xfade_dur']
                    filled.append('genre')

            # Transition/crossfade stored on the template win over a genre preset
            # only when the request named neither -- an explicit choice always wins.
            if 'transition' not in request.form and 'genre' not in request.form:
                if tpl.get('transition') in VALID_TRANSITIONS and tpl['transition'] != 'custom_matte':
                    transition = tpl['transition']
                    filled.append('transition')
                if tpl.get('xfade_dur') and 'xfade_dur' not in request.form:
                    xfade_dur = max(0.1, min(2.0, float(tpl['xfade_dur'])))

            # For the three mode-driven slots, an explicitly chosen mode is always
            # respected -- picking "None" for music means none, even if the show's
            # template has a bed. A slot is only filled when the mode field was
            # absent entirely (a bare API call that just names a template) or when
            # it says "upload" but no file actually came with it (the UI's state
            # after selecting a template).
            def _wants_template(mode_field, current_mode, have_file):
                # A real file always wins, full stop -- this used to be checked
                # only in the second branch below, so a bare API call that
                # attached a genuine file but omitted the mode field (a browser
                # form always sends it; a script easily might not) hit the first
                # branch instead and had its upload silently overwritten by the
                # template's asset.
                if have_file:
                    return False
                # Mode field absent entirely -> nothing was actually chosen (the
                # value in hand is just this function's own default), so the
                # template decides. This is what makes a bare API call that only
                # names a template work.
                if mode_field not in request.form:
                    return True
                # Otherwise only an explicit "upload" with nothing attached
                # defers -- which is exactly the state the UI is in after picking
                # a template, and is also how a deliberate None/Generate/TTS
                # choice gets respected.
                return current_mode == 'upload'

            def _skip_tpl(field):
                # Set by the UI when the user clicks the X on a template-sourced
                # chip, or touches that upload control at all -- an explicit
                # opt-out for this one job, distinct from the field simply being
                # empty (which still defers to the template).
                return request.form.get(f'{field}_skip_template') == '1'

            # Background music. Note scoring_audio_path is the sentinel 'GENERATE'
            # (not a path) when synthesis was selected, hence the explicit compare.
            if not _skip_tpl('scoring_audio') and _wants_template(
                    'scoring_mode', scoring_mode, scoring_audio_path not in (None, 'GENERATE')):
                staged = template_stage_asset(tpl, 'bgm')
                if staged:
                    scoring_audio_path, scoring_mode = staged, 'upload'
                    filled.append('bgm')

            # SFX one-shot ('genre' mode is an explicit choice; leave it alone).
            if not _skip_tpl('sfx_upload') and sfx_mode != 'genre' and _wants_template(
                    'sfx_mode', sfx_mode, bool(sfx_upload_path)):
                staged = template_stage_asset(tpl, 'sfx')
                if staged:
                    sfx_upload_path, sfx_mode = staged, 'upload'
                    filled.append('sfx')

            # Voiceover ('tts' is an explicit choice; leave it alone).
            if not _skip_tpl('vo_upload') and vo_mode != 'tts' and _wants_template(
                    'vo_mode', vo_mode, bool(vo_upload_path)):
                staged = template_stage_asset(tpl, 'vo')
                if staged:
                    vo_upload_path, vo_mode = staged, 'upload'
                    filled.append('vo')
                    vo_start = _tpl_number(vo_start, 'vo_start', 0.0)
                    vo_volume = _tpl_number(vo_volume, 'vo_volume', 1.15)
                    vo_trim_start = _tpl_number(vo_trim_start, 'vo_trim_start', 0.0)
                    vo_trim_end = _tpl_number(vo_trim_end, 'vo_trim_end', None)
                    if vo_trim_end is not None and vo_trim_end <= vo_trim_start:
                        vo_trim_end = None

            if not end_card_path and not _skip_tpl('end_card_video'):
                staged = template_stage_asset(tpl, 'title_card')
                if staged:
                    end_card_path = staged
                    filled.append('title_card')
            if not schedule_card_path and not _skip_tpl('schedule_video'):
                staged = template_stage_asset(tpl, 'end_card')
                if staged:
                    schedule_card_path = staged
                    filled.append('end_card')

            if not title_card_vo_path and not _skip_tpl('title_card_vo'):
                staged = template_stage_asset(tpl, 'title_card_vo')
                if staged:
                    title_card_vo_path = staged
                    filled.append('title_card_vo')
                    title_card_vo_start = _tpl_number(title_card_vo_start, 'title_card_vo_start', 0.0)
                    title_card_vo_end = _tpl_number(title_card_vo_end, 'title_card_vo_end', None)
                    if title_card_vo_end is not None and title_card_vo_end <= title_card_vo_start:
                        title_card_vo_end = None
            if not end_card_vo_path and not _skip_tpl('end_card_vo'):
                staged = template_stage_asset(tpl, 'end_card_vo')
                if staged:
                    end_card_vo_path = staged
                    filled.append('end_card_vo')
                    end_card_vo_start = _tpl_number(end_card_vo_start, 'end_card_vo_start', 0.0)
                    end_card_vo_end = _tpl_number(end_card_vo_end, 'end_card_vo_end', None)
                    if end_card_vo_end is not None and end_card_vo_end <= end_card_vo_start:
                        end_card_vo_end = None

            # Remaining scalar settings (lengths, thresholds, loudness targets,
            # voice choice...) are applied to the params dict below, once it
            # exists -- only for keys the request didn't send at all. A browser
            # POST carries every form field, so this is a no-op there; it matters
            # for an API call that just names a template.
            _tpl_pending = {k: v for k, v in tpl_settings.items()
                            if k not in request.form
                            and k not in ('genre', 'transition', 'xfade_dur')}

            template_applied = {'id': tpl['id'], 'name': tpl['name'], 'filled': filled}

    prompt = request.form.get('prompt',
        'Rate this frame 1-5 as a shot for a promo trailer, and describe what is '
        'actually visible in one short sentence (8-14 words): who or what is in '
        'shot, what they are doing, and where. Be concrete and literal. Do not use '
        'vague words like "cinematic", "dramatic" or "engaging".\n'
        'Answer with a single JSON object and nothing else: '
        '{"score": <1-5>, "desc": "<sentence>"}')

    jid = job_new(user_id=session.get('user_id'), username=session.get('username'))
    with JOBS_LOCK:
        if jid in JOBS:
            JOBS[jid]['orig_name'] = orig_name
    params = dict(path=path, orig_name=orig_name, mode=mode, genre=genre, scoring_mode=scoring_mode,
                  # Captured here (inside the request, where `session` exists) rather
                  # than inside the background thread that actually renders --
                  # threads don't have a Flask session/request context at all, so
                  # this is the only place ownership can be read from the login.
                  user_id=session.get('user_id'), username=session.get('username'),
                  trailer_length=trailer_length, max_scene_dur=max_scene_dur,
                  scene_threshold=scene_threshold, min_scene_len_sec=min_scene_len_sec,
                  detector=detector, adaptive_threshold=adaptive_threshold,
                  transition=transition, xfade_dur=xfade_dur, transition_matte_path=transition_matte_path,
                  target_loudness=target_loudness, true_peak=true_peak, music_duck_db=music_duck_db, duck_depth_db=duck_depth_db, duck_release_hold=duck_release_hold, beat_match=beat_match, broadcast_stereo=broadcast_stereo, model=model,
                  sfx_mode=sfx_mode, sfx_upload_path=sfx_upload_path,
                  vo_mode=vo_mode, vo_upload_path=vo_upload_path, vo_text=vo_text, vo_voice=vo_voice,
                  vo_language=vo_language, vo_engine=vo_engine, vo_ref_upload_path=vo_ref_upload_path,
                  vo_rate=vo_rate, vo_start=vo_start, vo_volume=vo_volume, sync_beats=sync_beats, whisper_enhance=whisper_enhance,
                  vo_trim_start=vo_trim_start, vo_trim_end=vo_trim_end,
                  end_card_path=end_card_path, schedule_card_path=schedule_card_path,
                  title_card_vo_path=title_card_vo_path, title_card_vo_start=title_card_vo_start, title_card_vo_end=title_card_vo_end,
                  end_card_vo_path=end_card_vo_path, end_card_vo_start=end_card_vo_start, end_card_vo_end=end_card_vo_end,
                  scoring_audio_path=scoring_audio_path, prompt=prompt,
                  template_applied=template_applied,
                  # Stop after scene selection and hand back a reviewable cut
                  # instead of rendering. See the preview block in _run_trailer_job.
                  preview_only=request.form.get('preview_only') in ('1', 'true', 'on'))

    # Overlay the template's saved configuration for anything the request left
    # out, coercing each stored string back to the type the param already holds.
    if _tpl_pending:
        for _k, _raw in _tpl_pending.items():
            if _k not in params:
                continue
            _cur = params[_k]
            try:
                if isinstance(_cur, bool):
                    _val = str(_raw).lower() in ('1', 'true', 'on', 'yes')
                elif isinstance(_cur, int) and not isinstance(_cur, bool):
                    _val = int(float(_raw))
                elif isinstance(_cur, float):
                    _val = float(_raw)
                else:
                    _val = _raw
            except (TypeError, ValueError):
                continue
            params[_k] = _val
            if template_applied:
                template_applied['filled'].append(_k)

    threading.Thread(target=run_trailer_job_gated, args=(jid, params), daemon=True).start()
    return jsonify(job_id=jid)

@app.route('/api/trailer/progress/<job_id>')
@require_permission('promo_generation')
def api_trailer_progress(job_id):
    j = job_get(job_id)
    if not j:
        return jsonify(error='Unknown job id'), 404
    if not _owns_or_admin(j.get('user_id')):
        return jsonify(error='That job belongs to a different account.'), 403
    created = j.pop('created', None)
    # Elapsed lets the UI show a running clock; the old response had no notion of
    # time at all, so a slow stage was indistinguishable from a hung one.
    if created:
        j['elapsed'] = round(time.time() - created, 1)
    j['stages'] = [{'percent': p, 'label': lbl} for p, lbl in PIPELINE_STAGES]
    return jsonify(**j)

@app.route('/api/trailer/preview/<preview_id>')
@require_permission('promo_generation')
def api_trailer_preview_get(preview_id):
    """Re-read a stored preview (thumbnails + chosen cut) without re-analysing."""
    p = preview_get(preview_id)
    if not p:
        return jsonify(ok=False, error='That preview has expired. Run the analysis again.'), 404
    if not _owns_or_admin(p.get('params', {}).get('user_id')):
        return jsonify(ok=False, error='That preview belongs to a different account.'), 403
    return jsonify(ok=True, preview_id=preview_id, total_scenes=p['total_scenes'],
                   video_filename=os.path.basename(p['params']['path']),
                   scenes=[{'scene': i + 1, 'start': round(s['start'], 1),
                            'end': round(s['end'], 1), 'quality': s['total_score'],
                            'duration': round(s['selected_dur'], 1),
                            'description': _scene_desc(s), 'thumb': p['thumbs'][i]}
                           for i, s in enumerate(p['selected'])],
                   alternates=[{'alt': i + 1, 'start': round(s['start'], 1),
                                'end': round(s['end'], 1), 'quality': s['total_score'],
                                'duration': round(s['selected_dur'], 1),
                                'description': _scene_desc(s), 'thumb': (p.get('alt_thumbs') or [None]*99)[i]}
                               for i, s in enumerate(p.get('alternates') or [])])

@app.route('/api/trailer/render', methods=['POST'])
@require_permission('promo_generation')
def api_trailer_render():
    """Render an approved preview cut. Reuses the preview's stored selection, so
    detection / quality scoring / AI vision scoring are not repeated.

    Optional `drop` is a JSON array of 1-based scene numbers (as shown in the
    preview) to leave out — the cheap way to fix a cut without re-running
    anything."""
    pid = (request.form.get('preview_id') or '').strip()
    p = preview_get(pid)
    if not p:
        return jsonify(error='That preview has expired or was never created. Run the analysis again.'), 404
    if not _owns_or_admin(p.get('params', {}).get('user_id')):
        return jsonify(error='That preview belongs to a different account.'), 403

    selected = p['selected']
    raw_drop = (request.form.get('drop') or '').strip()
    drop = set()
    if raw_drop:
        try:
            drop = {int(x) for x in json.loads(raw_drop)}
        except (ValueError, TypeError):
            return jsonify(error='`drop` must be a JSON array of scene numbers, e.g. [2,5].'), 400
        selected = [s for i, s in enumerate(selected) if (i + 1) not in drop]

    # `add` pulls in runner-up scenes the preview offered but didn't pick. Merged
    # back in timeline order so a swapped-in clip lands where it belongs rather
    # than at the end of the trailer.
    raw_add = (request.form.get('add') or '').strip()
    added = set()
    if raw_add:
        try:
            added = {int(x) for x in json.loads(raw_add)}
        except (ValueError, TypeError):
            return jsonify(error='`add` must be a JSON array of alternate numbers, e.g. [1,3].'), 400
        alts = p.get('alternates') or []
        bad = [n for n in added if not (1 <= n <= len(alts))]
        if bad:
            return jsonify(error=f'No alternate numbered {bad[0]} in this preview.'), 400
        selected = selected + [alts[n - 1] for n in sorted(added)]
        selected.sort(key=lambda s: s['start'])

    if not selected:
        return jsonify(error='You dropped every scene — keep at least one, or add an alternate.'), 400

    params = dict(p['params'])
    params['preview_only'] = False
    params['preselected'] = selected
    params['preview_total_scenes'] = p['total_scenes']
    # Re-stamped to the current session rather than left as whoever ran the
    # original preview -- normally the same person, but this is what actually
    # governs the resulting library entry's ownership, so it should reflect
    # who is triggering this render right now.
    params['user_id'] = session.get('user_id')
    params['username'] = session.get('username')
    if not (params.get('path') and os.path.exists(params['path'])):
        return jsonify(error='The source video for this preview is no longer on disk '
                             '(it may have been cleaned up). Re-upload and analyse again.'), 410

    jid = job_new(user_id=session.get('user_id'), username=session.get('username'))
    threading.Thread(target=run_trailer_job_gated, args=(jid, params), daemon=True).start()
    return jsonify(job_id=jid, dropped=sorted(drop), added=sorted(added), scenes=len(selected))

ACE_STEP_MAX_SAMPLES = int(os.environ.get('ACE_STEP_MAX_SAMPLES', 4))
# Where audio2audio reference files are written. ACE-Step's ref_audio_input is a
# path read by the ACE-Step process, not an upload, so both processes must be able
# to see the same file. Same machine: the default (UPLOAD_FOLDER) is fine. ACE-Step
# on another host: point this at a shared/NFS/SMB mount that resolves to the same
# path on both sides.
ACE_STEP_REF_DIR = os.environ.get('ACE_STEP_REF_DIR', '')

def acestep_generate(prompt, duration, lyrics=None, bpm=None, samples=1,
                     steps=None, seed=None, base_ts=None,
                     ref_audio_path=None, ref_strength=0.5,
                     keyscale=None, timesignature=None, thinking=False,
                     negative_prompt=None, model=None):
    """Generate music with ACE-Step directly. Returns (paths, error).

    Unlike prepare_bgm_track (which is shaped around the trailer pipeline: one
    track, faded, trimmed, transcoded to .m4a, with a synth fallback), this is the
    raw generator behind the Tools tab: N samples, optional sung lyrics, optional
    audio2audio reference, no fallback — if the service is down the caller should
    say so rather than hand back a sine drone the user didn't ask for.

    `bpm`, `keyscale`, and `timesignature` are all folded into the prompt as
    tags rather than sent as separate structured fields. Newer ACE-Step builds
    do accept dedicated bpm/keyscale/timesignature JSON fields, but this
    server's /release_task endpoint was tested against tempo specifically and
    only responds to it as a prompt tag ("124 bpm") — a plain field is
    silently ignored. Rather than send fields with unverified effect against
    this specific install, key and time signature follow the same
    known-working convention.

    `thinking` enables ACE-Step's LM planning pass: given the caption/lyrics,
    it auto-infers any of bpm/key/time-signature/etc you left unset. Slower
    (an extra LM inference before the DiT pass), often better when you don't
    have strong opinions on those specifics. This *is* sent as a real field
    (`thinking` in the payload below) — unlike tempo, this one's confirmed
    against this server since the payload already sent it, just hardcoded to
    False everywhere until now.

    `negative_prompt` is a real, already-used field (see ACE_STEP_NEGATIVE_PROMPT
    below) — this parameter just makes it caller-editable instead of only ever
    being the fixed default text auto-applied for instrumental generations.

    `ref_audio_path` enables audio2audio: the output follows the reference's
    structure, with `ref_strength` (0-1) controlling how closely. NOTE: ACE-Step's
    ref_audio_input takes a FILE PATH that the ACE-Step process must be able to
    read. That works when ACE-Step runs on this machine (the default
    localhost:8001), but on a separate host the path won't resolve there — see
    ACE_STEP_REF_DIR for pointing both sides at a shared mount.

    `model` selects between multiple DiT checkpoints on servers that expose
    more than one (ACE-Step 1.5's multi-model routing, or a wrapper's own
    aliases -- see api_music_models() for how the picker discovers what's
    available). Omitted from the payload entirely when blank, so a
    single-model server that doesn't recognize the field at all is
    unaffected either way."""
    tags = (prompt or '').strip() or 'cinematic, instrumental'
    if bpm:
        # Don't double up if the user already typed a bpm into the prompt.
        if not re.search(r'\b\d{2,3}\s*bpm\b', tags, re.I):
            tags = f'{tags}, {int(bpm)} bpm'
    keyscale = (keyscale or '').strip()
    if keyscale:
        tags = f'{tags}, {keyscale} key'
    timesignature = (timesignature or '').strip()
    if timesignature:
        # UI sends the beats-per-bar number (2/3/4/6); spelled out as "N/4"
        # or "6/8" matches how these are conventionally written as prompt tags
        # (and how ACE-Step's own docs describe them) far better than a bare
        # digit, which reads ambiguously next to a bpm tag right next to it.
        sig_label = '6/8' if timesignature == '6' else f'{timesignature}/4'
        tags = f'{tags}, {sig_label} time signature'
    lyrics = (lyrics or '').strip()
    instrumental = not lyrics
    samples = max(1, min(ACE_STEP_MAX_SAMPLES, int(samples or 1)))

    payload = {
        'prompt': tags,
        'audio_duration': float(duration),
        'thinking': bool(thinking),
        'inference_steps': int(steps or ACE_STEP_STEPS),
        'batch_size': samples,
        # '[inst]' is ACE-Step's explicit instrumental marker. When the user has
        # supplied real lyrics we must send those instead AND drop the
        # vocal-suppressing negative prompt, which would otherwise fight them.
        'lyrics': '[inst]' if instrumental else lyrics,
    }
    if seed not in (None, ''):
        try:
            payload['manual_seeds'] = str(int(seed))
        except (TypeError, ValueError):
            pass
    # An explicit negative_prompt from the caller always wins. Otherwise fall
    # back to the vocal-suppressing default, but only for instrumental takes —
    # applying it to a lyric take would fight the vocals the user just asked for.
    negative_prompt = (negative_prompt or '').strip()
    if negative_prompt:
        payload['negative_prompt'] = negative_prompt
    elif instrumental and ACE_STEP_NEGATIVE_PROMPT:
        payload['negative_prompt'] = ACE_STEP_NEGATIVE_PROMPT
    if ref_audio_path and os.path.exists(ref_audio_path):
        payload['audio2audio_enable'] = True
        payload['ref_audio_input'] = os.path.abspath(ref_audio_path)
        payload['ref_audio_strength'] = max(0.0, min(1.0, float(ref_strength)))
    model = (model or '').strip()
    if model:
        payload['model'] = model

    base_ts = base_ts or f'tool{int(time.time()*1000)}'
    try:
        r = requests.post(f'{ACE_STEP_URL}/release_task', json=payload, timeout=15)
        task_id = (r.json().get('data') or {}).get('task_id')
        if not task_id:
            return [], f'ACE-Step did not accept the request (no task id). Response: {r.text[:200]}'
        for _ in range(90):
            time.sleep(2)
            q = requests.post(f'{ACE_STEP_URL}/query_result',
                              json={'task_id_list': [task_id]}, timeout=10)
            items = q.json().get('data') or []
            if not items:
                continue
            status = items[0].get('status')
            if status == 2:
                return [], 'ACE-Step reported the generation task failed.'
            if status != 1:
                continue
            result = json.loads(items[0]['result'])
            entries = result if isinstance(result, list) else [result]
            paths = []
            for i, entry in enumerate(entries[:samples]):
                remote = entry.get('file') if isinstance(entry, dict) else None
                if not remote:
                    continue
                dest = os.path.join(app.config['UPLOAD_FOLDER'],
                                    f'music_{base_ts}_{i}{os.path.splitext(remote)[1] or ".wav"}')
                try:
                    resp = requests.get(f'{ACE_STEP_URL}{remote}', timeout=120)
                    with open(dest, 'wb') as f:
                        f.write(resp.content)
                except Exception as e:
                    print(f'ACE-Step sample {i} download failed: {e}')
                    continue
                if os.path.exists(dest) and os.path.getsize(dest) > 0:
                    paths.append(dest)
            if not paths:
                return [], 'ACE-Step finished but returned no usable audio.'
            return paths, None
        return [], 'Timed out waiting for ACE-Step (3 minutes). The service may be overloaded.'
    except Exception as e:
        return [], f'Could not reach ACE-Step at {ACE_STEP_URL}: {e}'

@app.route('/api/music/generate', methods=['POST'])
@require_permission('music_generation')
def api_music_generate():
    """Standalone ACE-Step music generation for the Tools tab.

    Exposes every control ACE-Step itself takes for a text2music-style
    generation: free-text prompt, sung lyrics (or instrumental), tempo, key,
    time signature, negative styles, LM "thinking" planning, and how many
    samples to generate in one go so alternatives can be auditioned side by
    side."""
    def _num(key, default, lo, hi, cast=float):
        raw = (request.form.get(key) or '').strip()
        if raw == '':
            return default
        try:
            return max(lo, min(hi, cast(float(raw))))
        except (TypeError, ValueError):
            return default

    duration = _num('duration', 30.0, 5.0, 300.0)
    samples = _num('samples', 1, 1, ACE_STEP_MAX_SAMPLES, int)
    bpm = _num('bpm', None, 40, 220, int)
    steps = _num('steps', ACE_STEP_STEPS, 8, 120, int)
    genre = (request.form.get('genre') or '').strip()
    prompt = (request.form.get('prompt') or '').strip() or GENRE_PROMPTS.get(genre, '')
    lyrics = (request.form.get('lyrics') or '').strip()
    seed = (request.form.get('seed') or '').strip()
    keyscale = (request.form.get('keyscale') or '').strip()
    timesignature = (request.form.get('timesignature') or '').strip()
    if timesignature not in ('2', '3', '4', '6'):
        timesignature = ''
    negative_prompt = (request.form.get('negative_prompt') or '').strip()
    thinking = (request.form.get('thinking') or '').strip().lower() in ('1', 'true', 'on', 'yes')
    model = (request.form.get('model') or '').strip()

    if not prompt:
        return jsonify(ok=False, error='Enter a prompt describing the style you want.'), 400

    base_ts = f'tool{int(time.time()*1000)}'

    # Optional audio2audio reference. Normalised to a plain 44.1k stereo WAV so
    # ACE-Step gets something it can definitely decode regardless of what the user
    # dropped in (m4a, mp3, a video's audio track, an odd sample rate).
    ref_path = None
    ref_strength = _num('ref_strength', 0.5, 0.0, 1.0)
    ref_src = _resolve_upload('ref_audio', AUDIO_EXTENSIONS)
    if ref_src:
        ref_dir = ACE_STEP_REF_DIR or app.config['UPLOAD_FOLDER']
        try:
            os.makedirs(ref_dir, exist_ok=True)
        except OSError as e:
            return jsonify(ok=False, error=f'Could not write to the reference audio directory ({ref_dir}): {e}'), 500
        ref_path = os.path.join(ref_dir, f'aceref_{base_ts}.wav')
        try:
            run_ffmpeg([FFMPEG, '-y', '-i', ref_src, '-vn', '-ac', '2', '-ar', '44100',
                        '-c:a', 'pcm_s16le', ref_path], timeout=120, label='ACE reference convert')
        except MediaToolTimeout:
            ref_path = None
        if not (ref_path and os.path.exists(ref_path) and os.path.getsize(ref_path) > 0):
            return jsonify(ok=False, error='Could not read that reference audio file — '
                                           'try a standard WAV or MP3.'), 400

    try:
        paths, err = acestep_generate(prompt, duration, lyrics=lyrics, bpm=bpm,
                                      samples=samples, steps=steps, seed=seed, base_ts=base_ts,
                                      ref_audio_path=ref_path, ref_strength=ref_strength,
                                      keyscale=keyscale, timesignature=timesignature,
                                      thinking=thinking, negative_prompt=negative_prompt, model=model)
    finally:
        # The reference only needs to survive the generation call itself.
        if ref_path and os.path.exists(ref_path):
            try:
                os.remove(ref_path)
            except OSError:
                pass
    if err:
        return jsonify(ok=False, error=err), 502

    return jsonify(ok=True, samples=[{
        'url': f'/uploads/{os.path.basename(p)}',
        'filename': os.path.basename(p),
        'duration': round(probe_duration(p) or duration, 1),
    } for p in paths],
        prompt=prompt, lyrics=lyrics or None, bpm=bpm, steps=steps,
        instrumental=not lyrics,
        keyscale=keyscale or None, timesignature=timesignature or None, thinking=thinking,
        negative_prompt=negative_prompt or None, model=model or None,
        reference=bool(ref_path), ref_strength=ref_strength if ref_path else None)

@app.route('/api/music/models')
@require_permission('music_generation')
def api_music_models():
    """Model/checkpoint aliases the ACE-Step server has loaded, for the Music
    Generation tab's model picker. ACE-Step 1.5's multi-model routing (and at
    least one hosted wrapper observed in the wild) exposes these at GET
    /models; older/single-model servers and other wrappers don't have this
    endpoint at all -- that's not an error, it just means there's nothing to
    switch between, so this returns an empty list rather than surfacing a
    failure. /api/music/generate still takes a free-typed model name either
    way (see acestep_generate's `model` param), this is purely to populate
    the picker's suggestions when the server can tell us what's available."""
    try:
        r = requests.get(f'{ACE_STEP_URL}/models', timeout=5)
        if not r.ok:
            return jsonify(ok=True, models=[])
        data = r.json()
    except Exception:
        return jsonify(ok=True, models=[])
    if isinstance(data, dict):
        items = data.get('models') or data.get('data') or data.get('aliases') or []
    elif isinstance(data, list):
        items = data
    else:
        items = []
    models = []
    for it in items:
        if isinstance(it, str):
            models.append(it)
        elif isinstance(it, dict):
            name = it.get('alias') or it.get('name') or it.get('id') or it.get('model')
            if name:
                models.append(name)
    return jsonify(ok=True, models=models)

@app.route('/api/music/genres')
@require_permission('music_generation')
def api_music_genres():
    """Genre -> music prompt map, so the Tools tab can show and pre-fill prompts."""
    return jsonify(ok=True, genres=[{'key': g, 'prompt': GENRE_PROMPTS.get(g, '')}
                                    for g in GENRE_NAMES])

@app.route('/api/sfx/generate', methods=['POST'])
@require_permission('text_to_sfx')
def api_sfx_generate():
    """Standalone Woosh SFX generation for the Tools tab: any text description,
    not just the fixed genre-derived prompts the trailer pipeline uses."""
    def _num(key, default, lo, hi, cast=float):
        raw = (request.form.get(key) or '').strip()
        if raw == '':
            return default
        try:
            return max(lo, min(hi, cast(float(raw))))
        except (TypeError, ValueError):
            return default

    prompt = (request.form.get('prompt') or '').strip()
    if not prompt:
        return jsonify(ok=False, error='Enter a description of the sound you want.'), 400
    # No UI control for this anymore -- Woosh's own API has no duration
    # parameter at all (see _woosh_generate's docstring); the only way
    # "duration" ever meant anything here was trimming the output afterward
    # with ffmpeg, which just chopped whatever natural length Woosh produced.
    # Removed rather than kept as a hidden default trim, so the tool always
    # returns Woosh's own take on how long the sound should be.
    samples = _num('samples', 1, 1, WOOSH_MAX_SAMPLES, int)

    paths, err = woosh_sfx_generate(prompt, duration=None, samples=samples)
    if err:
        return jsonify(ok=False, error=err), 502

    return jsonify(ok=True, prompt=prompt, samples=[{
        'url': f'/uploads/{os.path.basename(p)}',
        'filename': os.path.basename(p),
        'duration': round(probe_duration(p) or 0, 1),
    } for p in paths])

@app.route('/api/trailer/cancel/<job_id>', methods=['POST'])
@require_permission('promo_generation')
def api_trailer_cancel(job_id):
    """Cancel a queued or in-flight trailer job. Queued jobs stop immediately;
    running jobs unwind at their next progress checkpoint (best-effort)."""
    j = job_get(job_id)
    if not j:
        return jsonify(error='Unknown job id'), 404
    if not _owns_or_admin(j.get('user_id')):
        return jsonify(error='That job belongs to a different account.'), 403
    ok = job_cancel(job_id)
    if not ok:
        j = job_get(job_id)
        if not j:
            return jsonify(error='Unknown job id'), 404
        return jsonify(error='Job already finished'), 409
    return jsonify(cancelled=True, job_id=job_id)

@app.route('/api/trailer/library')
@require_permission('promo_generation')
def api_trailer_library():
    """Lists saved trailers (most recent first) for the History panel. A
    regular account sees only trailers it saved; an admin sees everyone's."""
    is_admin = session.get('role') == 'admin'
    return jsonify(ok=True, items=library_list(user_id=session.get('user_id'), is_admin=is_admin))

@app.route('/api/trailer/library/<int:tid>')
@require_permission('promo_generation')
def api_trailer_library_get(tid):
    """Returns one saved trailer's full result payload, ready to hand straight
    to the same renderer used for a just-completed job."""
    row = library_get_row(tid)
    if not row or not row.get('result_json'):
        return jsonify(ok=False, error='Not found'), 404
    if not _owns_or_admin(row.get('user_id')):
        return jsonify(ok=False, error='Not found'), 404
    return jsonify(ok=True, result=json.loads(row['result_json']), created_at=row['created_at'])

@app.route('/api/trailer/library/<int:tid>/delete', methods=['POST'])
@require_permission('promo_generation')
def api_trailer_library_delete(tid):
    row = library_get_row(tid)
    if not row:
        return jsonify(ok=False, error='Not found'), 404
    if not _owns_or_admin(row.get('user_id')):
        return jsonify(ok=False, error='Not found'), 404
    ok = library_delete(tid)
    if not ok:
        return jsonify(ok=False, error='Not found'), 404
    return jsonify(ok=True)

@app.route('/library/<int:tid>/file')
@require_permission('promo_generation')
def library_file(tid):
    row = library_get_row(tid)
    if not row:
        return jsonify(error='Not found'), 404
    if not _owns_or_admin(row.get('user_id')):
        return jsonify(error='Not found'), 404
    return send_from_directory(LIBRARY_DIR, row['filename'])

@app.route('/library/<int:tid>/download')
@require_permission('promo_generation')
def library_download(tid):
    row = library_get_row(tid)
    if not row:
        return jsonify(error='Not found'), 404
    if not _owns_or_admin(row.get('user_id')):
        return jsonify(error='Not found'), 404
    fmt_key = request.args.get('format', 'mp4_high')
    if fmt_key not in EXPORT_FORMATS:
        return jsonify(error=f'Unknown export format: {fmt_key}'), 400
    src_path = os.path.join(LIBRARY_DIR, row['filename'])
    if not os.path.exists(src_path):
        return jsonify(error='File not found'), 404
    ext = EXPORT_FORMATS[fmt_key]['ext']
    base_name, _ = os.path.splitext(row['orig_name'] or row['filename'])
    cache_name = f'{os.path.splitext(row["filename"])[0]}_{fmt_key}.{ext}'
    cache_path = os.path.join(LIBRARY_DIR, cache_name)
    if not (os.path.exists(cache_path) and os.path.getsize(cache_path) > 0):
        cmd = build_export_cmd(src_path, cache_path, fmt_key)
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if not (os.path.exists(cache_path) and os.path.getsize(cache_path) > 0):
            return jsonify(error=f'Export to {fmt_key} failed: {r.stderr[-800:]}'), 500
    resp = send_from_directory(LIBRARY_DIR, cache_name)
    resp.headers['Content-Disposition'] = f'attachment; filename="{base_name}.{ext}"'
    return resp


@app.route('/api/monitor')
@require_permission('promo_generation')
def api_monitor():
    """Live snapshot of trailer jobs the server currently knows about: running
    right now, waiting for a free concurrency slot, or finished
    (success/error/cancelled) within the last JOB_TTL. Per-user for a regular
    account (only jobs *they* started); whole-server for an admin. This is
    the transient, in-progress counterpart to the permanent Saved Trailers
    library: finished entries here age out after JOB_TTL regardless of
    whether they were also saved to the library."""
    is_admin = session.get('role') == 'admin'
    with JOB_QUEUE_LOCK:
        queued_ids = list(JOB_QUEUE)
    with JOBS_LOCK:
        snapshot = {jid: dict(j) for jid, j in JOBS.items() if _owns_or_admin(j.get('user_id'))}

    queued = [{'job_id': jid, 'position': i, 'orig_name': snapshot.get(jid, {}).get('orig_name'),
               **({'username': snapshot.get(jid, {}).get('username')} if is_admin else {})}
              for i, jid in enumerate(queued_ids) if jid in snapshot]

    active, finished = [], []
    for jid, j in snapshot.items():
        if jid in queued_ids:
            continue
        entry = {'job_id': jid, 'orig_name': j.get('orig_name'), 'percent': j.get('percent'),
                  'step': j.get('step'), 'status': j.get('status'), 'created': j.get('created')}
        if is_admin:
            entry['username'] = j.get('username')
        if j.get('done'):
            entry['error'] = j.get('error')
            finished.append(entry)
        else:
            active.append(entry)
    active.sort(key=lambda e: e['created'] or 0)
    finished.sort(key=lambda e: e['created'] or 0, reverse=True)
    return jsonify(active=active, queued=queued, finished=finished[:20], limit=GATE.status()['limit'])

@app.route('/api/queue/status')
def api_queue_status():
    """Overall queue state: how many jobs are running vs the current concurrency
    limit, plus every currently-queued job with its wait position."""
    gate_status = GATE.status()
    with JOB_QUEUE_LOCK:
        queued = list(JOB_QUEUE)
    queued_info = []
    for i, jid in enumerate(queued):
        j = job_get(jid)
        queued_info.append({'job_id': jid, 'position': i, 'orig_name': (j or {}).get('orig_name')})
    return jsonify(running=gate_status['running'], limit=gate_status['limit'], queued=queued_info)

@app.route('/api/queue/limit', methods=['GET', 'POST'])
def api_queue_limit():
    """View or change how many trailer jobs are allowed to run at once. Changing
    this takes effect immediately — queued jobs re-check the limit as soon as a
    slot frees up or the limit itself changes."""
    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        try:
            new_limit = int(data.get('limit', request.form.get('limit')))
        except (TypeError, ValueError):
            return jsonify(error='limit must be an integer'), 400
        if new_limit < 1:
            return jsonify(error='limit must be at least 1'), 400
        GATE.set_limit(new_limit)
    return jsonify(**GATE.status())

# ---- Health check: pings every external service this app depends on ----
# For each one we just check that *something* is listening and answers HTTP —
# we deliberately avoid hitting generation endpoints (Fish Audio /v1/tts,
# ACE-Step /release_task, Woosh /generate, Whisper /v1/audio/transcriptions)
# so a health check never costs GPU time or produces real output. Ollama is the
# only service with a cheap, purpose-built status endpoint (/api/tags), so that
# one's checked precisely; the rest are checked for bare reachability at their
# base URL, which is enough to tell "service is up" from "connection refused".
def _check_service(name, base_url, path='/', timeout=3):
    t0 = time.time()
    try:
        r = requests.get(base_url.rstrip('/') + path, timeout=timeout)
        latency_ms = round((time.time() - t0) * 1000)
        # Any HTTP response at all means something is listening and answering,
        # even a 404/405 for a path that server doesn't implement.
        return {'name': name, 'url': base_url, 'status': 'up',
                'http_status': r.status_code, 'latency_ms': latency_ms}
    except requests.exceptions.Timeout:
        return {'name': name, 'url': base_url, 'status': 'down', 'error': 'timeout'}
    except requests.exceptions.ConnectionError:
        return {'name': name, 'url': base_url, 'status': 'down', 'error': 'connection refused'}
    except Exception as e:
        return {'name': name, 'url': base_url, 'status': 'down', 'error': str(e)}

@app.route('/api/network/list')
def api_network_list():
    """Lists the files sitting in the network folder for ?category=hires|music|vo|sfx
    (defaults to hires/video). Each category maps to its own subfolder and its
    own allowed extensions -- see NETWORK_CATEGORIES."""
    category = request.args.get('category', DEFAULT_NETWORK_CATEGORY)
    try:
        root, files = list_network_files(category)
        return jsonify(ok=True, root=root, category=category, files=files)
    except Exception as e:
        return jsonify(ok=False, error=f'Could not reach network folder: {e}'), 500

@app.route('/api/network/fetch', methods=['POST'])
def api_network_fetch():
    """Copies one file from the network folder (for the given category) into the
    local upload folder so it can be used exactly like a drag-and-dropped file
    (see load_video() / _resolve_upload())."""
    data = request.get_json(silent=True) or request.form
    name = (data.get('name') or '').strip()
    category = (data.get('category') or DEFAULT_NETWORK_CATEGORY).strip()
    if not name:
        return jsonify(ok=False, error='No filename given'), 400
    try:
        local_name = fetch_network_file(name, category)
        local_path = os.path.join(app.config['UPLOAD_FOLDER'], local_name)
        # `url` lets the Player play the staged copy directly; callers that stage
        # a file for the generate form use `filename`.
        return jsonify(ok=True, filename=local_name, orig_name=name, category=category,
                        url=f'/uploads/{local_name}',
                        size=os.path.getsize(local_path))
    except ValueError as e:
        return jsonify(ok=False, error=str(e)), 400
    except Exception as e:
        return jsonify(ok=False, error=f'Could not fetch {name}: {e}'), 500

# ---- Show templates (saved per-show asset bundles) ----

def _tpl_num(key, default=None, lo=None, hi=None):
    """Reads an optional numeric form field. Blank/absent -> default (so the UI can
    leave a field untouched without wiping a previously saved value)."""
    raw = (request.form.get(key) or '').strip()
    if raw == '':
        return default
    try:
        v = float(raw)
    except ValueError:
        return default
    if lo is not None:
        v = max(lo, v)
    if hi is not None:
        v = min(hi, v)
    return v

@app.route('/api/templates', methods=['GET'])
@require_permission('promo_generation')
def api_templates_list():
    return jsonify(ok=True, templates=template_list(),
                   slots=[{'key': k, 'label': v['label'], 'kind': v['kind']} for k, v in TEMPLATE_SLOTS.items()])

@app.route('/api/templates/<int:tid>', methods=['GET'])
@require_permission('promo_generation')
def api_template_get(tid):
    row = template_get(tid)
    if not row:
        return jsonify(ok=False, error='Template not found'), 404
    return jsonify(ok=True, template=_template_public(row))

@app.route('/api/templates', methods=['POST'])
@require_permission('promo_generation')
def api_template_save():
    """Creates or updates a show template from whatever the trailer form currently
    has selected. Accepts the same multipart field names the generate form uses
    (including the `<field>_network` staged-file hidden inputs), so the "Add to
    template" button can just re-post the picked files without any new plumbing.

    Only slots that arrive with a file are written -- posting again with just a
    new music bed updates that one slot and leaves the show's cards alone. Send
    clear_<slot>=1 to deliberately empty a slot."""
    name = (request.form.get('name') or '').strip()
    if not name:
        return jsonify(ok=False, error='Give the template a show name first.'), 400
    if len(name) > 120:
        return jsonify(ok=False, error='Name is too long (max 120 characters).'), 400

    tid_raw = (request.form.get('template_id') or '').strip()
    existing = template_get(int(tid_raw)) if tid_raw.isdigit() else None
    if existing is None:
        existing = template_get_by_name(name)  # saving under an existing name updates it

    # Resolve whichever asset files were supplied this time round.
    incoming = {}
    for slot, meta in TEMPLATE_SLOTS.items():
        exts = AUDIO_EXTENSIONS if meta['kind'] == 'audio' else ALLOWED_EXTENSIONS
        p = _resolve_upload(meta['field'], exts)
        if p:
            incoming[slot] = p

    if not existing and not incoming:
        return jsonify(ok=False, error='Select at least one file (music, SFX, VO or a card), '
                                       'or fill in the form, before saving a show.'), 400

    # Capture the whole generator configuration, not just the files. A show
    # template is "how this programme's promos are made" -- genre, transition,
    # lengths, audio targets, voice choice -- with the assets as one part of that.
    settings = dict(template_settings(existing)) if existing else {}
    for key in TEMPLATE_SETTING_FIELDS:
        if key in request.form:
            settings[key] = (request.form.get(key) or '').strip()
        elif key in TEMPLATE_BOOL_FIELDS and request.form:
            # A checkbox absent from a submitted form means unticked, which is a
            # real value -- not "leave whatever was there before".
            settings[key] = ''
    settings = {k: v for k, v in settings.items() if v != ''}

    fields = {
        'name': name,
        'notes': (request.form.get('notes') or '').strip() or (existing or {}).get('notes'),
        'genre': (request.form.get('genre') or '').strip() or (existing or {}).get('genre'),
        'transition': (request.form.get('transition') or '').strip() or (existing or {}).get('transition'),
        'xfade_dur': _tpl_num('xfade_dur', (existing or {}).get('xfade_dur'), 0.1, 2.0),
        'vo_start': _tpl_num('vo_start', (existing or {}).get('vo_start'), 0),
        'vo_volume': _tpl_num('vo_volume', (existing or {}).get('vo_volume'), 0.3, 3.0),
        'vo_trim_start': _tpl_num('vo_trim_start', (existing or {}).get('vo_trim_start'), 0),
        'vo_trim_end': _tpl_num('vo_trim_end', (existing or {}).get('vo_trim_end'), 0),
        'title_card_vo_start': _tpl_num('title_card_vo_start', (existing or {}).get('title_card_vo_start'), 0),
        'title_card_vo_end': _tpl_num('title_card_vo_end', (existing or {}).get('title_card_vo_end'), 0),
        'end_card_vo_start': _tpl_num('end_card_vo_start', (existing or {}).get('end_card_vo_start'), 0),
        'end_card_vo_end': _tpl_num('end_card_vo_end', (existing or {}).get('end_card_vo_end'), 0),
        'settings_json': json.dumps(settings),
    }
    if fields['transition'] and fields['transition'] not in VALID_TRANSITIONS:
        fields['transition'] = None

    now = time.time()
    conn = _tpl_db()
    try:
        if existing:
            tid = existing['id']
            conn.execute('UPDATE show_templates SET ' + ','.join(f'{k}=?' for k in fields) + ', updated_at=? WHERE id=?',
                         list(fields.values()) + [now, tid])
        else:
            cols = list(fields.keys()) + ['created_at', 'updated_at']
            cur = conn.execute(f'INSERT INTO show_templates ({",".join(cols)}) VALUES ({",".join("?" * len(cols))})',
                               list(fields.values()) + [now, now])
            tid = cur.lastrowid

        for slot, meta in TEMPLATE_SLOTS.items():
            if slot in incoming:
                if existing:
                    template_delete_asset_file(existing, slot)  # replace: drop the old master
                stored, disp = template_store_asset(incoming[slot], slot,
                                                    _upload_display_name(meta['field']))
                conn.execute(f'UPDATE show_templates SET {slot}_file=?, {slot}_name=? WHERE id=?', (stored, disp, tid))
            elif existing and request.form.get(f'clear_{slot}') in ('1', 'on', 'true'):
                template_delete_asset_file(existing, slot)
                conn.execute(f'UPDATE show_templates SET {slot}_file=NULL, {slot}_name=NULL WHERE id=?', (tid,))
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        return jsonify(ok=False, error=f'A template named "{name}" already exists.'), 409
    finally:
        try:
            conn.close()
        except Exception:
            pass

    # The staged copies in UPLOAD_FOLDER have served their purpose -- the masters
    # now live in TEMPLATES_DIR. Leave them be if they were network-staged files,
    # since the same staged file may still be attached to the form for this job.
    return jsonify(ok=True, template=_template_public(template_get(tid)),
                   saved_slots=sorted(incoming.keys()), updated=bool(existing))

@app.route('/api/templates/<int:tid>', methods=['DELETE'])
@require_permission('promo_generation')
def api_template_delete(tid):
    if not template_delete(tid):
        return jsonify(ok=False, error='Template not found'), 404
    return jsonify(ok=True)

@app.route('/api/templates/<int:tid>/asset/<slot>')
@require_permission('promo_generation')
def api_template_asset(tid, slot):
    """Serves a template's stored asset so the UI can preview it (audio player /
    card-VO in-out scrubbing) without re-uploading anything."""
    if slot not in TEMPLATE_SLOTS:
        return jsonify(ok=False, error='Unknown slot'), 404
    row = template_get(tid)
    p = template_asset_abspath(row, slot)
    if not p:
        return jsonify(ok=False, error='That slot is empty'), 404
    return send_from_directory(TEMPLATES_DIR, os.path.basename(p))

@app.route('/api/config', methods=['GET'])
def api_config_get():
    """Current values of every configurable AI service URL, for the Config tab.
    Admin-only: exposes FISH_AUDIO_API_KEY, and lets you point the whole app's
    AI services somewhere else -- not something a regular account should see
    or touch."""
    if session.get('role') != 'admin':
        return jsonify(ok=False, error='Admin access required.'), 403
    return jsonify(ok=True, config=current_config_values(),
                   fields={k: {'label': v[0], 'help': v[1]} for k, v in CONFIGURABLE_SERVICES.items()})

@app.route('/api/config', methods=['POST'])
def api_config_post():
    """Saves Config-tab edits to disk and applies them immediately (no restart needed)."""
    if session.get('role') != 'admin':
        return jsonify(ok=False, error='Admin access required.'), 403
    data = request.get_json(silent=True) or {}
    unknown = [k for k in data if k not in CONFIGURABLE_SERVICES]
    if unknown:
        return jsonify(ok=False, error=f'Unknown config key(s): {", ".join(unknown)}'), 400
    try:
        save_config_overrides(data)
    except Exception as e:
        return jsonify(ok=False, error=f'Could not save config: {e}'), 500
    return jsonify(ok=True, config=current_config_values())

@app.route('/api/config/test', methods=['POST'])
def api_config_test():
    """Pings a single URL from the Config tab's edit fields (before saving), so a
    typo can be caught without committing it first. Body: {"name": "FISH_AUDIO_URL", "url": "..."}."""
    if session.get('role') != 'admin':
        return jsonify(ok=False, error='Admin access required.'), 403
    data = request.get_json(silent=True) or {}
    name = data.get('name', '')
    url = (data.get('url') or '').strip()
    if name not in CONFIGURABLE_SERVICES:
        return jsonify(ok=False, error='Unknown service'), 400
    if name == 'FISH_AUDIO_API_KEY':
        return jsonify(ok=False, error='Not a URL field'), 400
    if not url:
        return jsonify(ok=False, error='No URL to test'), 400
    base = url.rsplit('/v1/', 1)[0] if '/v1/' in url else url
    path = '/api/tags' if name == 'OLLAMA_URL' else '/'
    result = _check_service(name, base, path)
    return jsonify(ok=result['status'] == 'up', **result)

# ---- Branding routes ----
# /branding/logo and /branding/favicon are in _PUBLIC_PATHS (see core.py) so
# the login page -- rendered before any session exists -- can show the
# configured logo/favicon too, not just the signed-in app shell. Serving an
# image file to a logged-out client isn't a meaningful exposure (same content
# a favicon <link> would leak anyway), so this is a safe carve-out from the
# session gate.
@app.route('/branding/logo')
def branding_logo():
    cfg = load_branding()
    if cfg['logo_filename'] and os.path.exists(os.path.join(BRANDING_DIR, cfg['logo_filename'])):
        return send_from_directory(BRANDING_DIR, cfg['logo_filename'])
    return redirect('/static/logo-mark.svg')

@app.route('/branding/favicon')
def branding_favicon():
    cfg = load_branding()
    if cfg['favicon_filename'] and os.path.exists(os.path.join(BRANDING_DIR, cfg['favicon_filename'])):
        return send_from_directory(BRANDING_DIR, cfg['favicon_filename'])
    return redirect('/static/logo-mark.svg')

@app.route('/api/branding', methods=['GET'])
def api_branding_get():
    """Current brand name/tagline/accent color/logo/favicon state, for the
    Config > Branding tab to populate its fields and for the login/index
    pages' <title>/sidebar/theme. Read-only and not admin-gated -- knowing
    the current branding isn't sensitive, only changing it is (see the POST
    route below)."""
    cfg = load_branding()
    return jsonify(ok=True, name=cfg['name'], tagline=cfg['tagline'], accent_color=cfg['accent_color'],
                   has_logo=bool(cfg['logo_filename']), has_favicon=bool(cfg['favicon_filename']))

@app.route('/api/branding', methods=['POST'])
def api_branding_post():
    """Saves Branding-tab edits: name/tagline/accent_color (form fields)
    and/or a new logo/favicon (file uploads) in the same request.
    reset_logo=1/reset_favicon=1/reset_accent_color=1 clear that item back to
    the built-in default without needing to re-enter anything. Admin-only --
    this changes what every signed-in account (and the login page) sees, the
    same reasoning as /api/config."""
    if session.get('role') != 'admin':
        return jsonify(ok=False, error='Admin access required.'), 403
    name = request.form.get('name')
    tagline = request.form.get('tagline')
    if name is not None or tagline is not None:
        save_branding_text(name, tagline)
    accent_color = request.form.get('accent_color')
    if accent_color:
        _, err = save_branding_color(accent_color)
        if err:
            return jsonify(ok=False, error=err), 400
    logo_file = request.files.get('logo')
    if logo_file and logo_file.filename:
        _, err = save_branding_logo(logo_file)
        if err:
            return jsonify(ok=False, error=err), 400
    favicon_file = request.files.get('favicon')
    if favicon_file and favicon_file.filename:
        _, err = save_branding_favicon(favicon_file)
        if err:
            return jsonify(ok=False, error=err), 400
    if request.form.get('reset_logo') == '1':
        clear_branding_logo()
    if request.form.get('reset_favicon') == '1':
        clear_branding_favicon()
    if request.form.get('reset_accent_color') == '1':
        clear_branding_color()
    cfg = load_branding()
    return jsonify(ok=True, name=cfg['name'], tagline=cfg['tagline'], accent_color=cfg['accent_color'],
                   has_logo=bool(cfg['logo_filename']), has_favicon=bool(cfg['favicon_filename']))

@app.route('/api/health')
def api_health():
    """Reachability check for every local model/media service the app talks to
    (Ollama, Fish Audio S2, faster-whisper, ACE-Step, Woosh), checked in parallel
    so one slow/dead service doesn't stall the others. Returns per-service status
    plus an overall ok flag."""
    checks = [
        ('ollama', OLLAMA_URL, '/api/tags'),
        ('fish_audio', FISH_AUDIO_URL.rsplit('/v1/', 1)[0] if '/v1/' in FISH_AUDIO_URL else FISH_AUDIO_URL, '/'),
        ('whisper', WHISPER_URL, '/'),
        ('ace_step', ACE_STEP_URL, '/'),
        ('woosh', WOOSH_URL, '/'),
    ]
    results = {}
    threads = []
    def worker(name, url, path):
        results[name] = _check_service(name, url, path)
    for name, url, path in checks:
        th = threading.Thread(target=worker, args=(name, url, path))
        th.start()
        threads.append(th)
    # Joined against one shared deadline, not `timeout=5` per thread — since threads
    # already run in parallel, waiting up to 5s for each one *sequentially* could
    # take up to 5s × len(checks) in the worst case (multiple unreachable services)
    # instead of the ~5s total this is actually meant to cap at.
    deadline = time.time() + 5
    for th in threads:
        th.join(timeout=max(0, deadline - time.time()))
    ordered = [results.get(name, {'name': name, 'url': url, 'status': 'down', 'error': 'no response'})
               for name, url, path in checks]
    overall_ok = all(c['status'] == 'up' for c in ordered)
    return jsonify(ok=overall_ok, checked_at=time.time(), services=ordered)

@app.route('/api/stt/transcribe', methods=['POST'])
@require_permission('speech_to_text')
def api_stt_transcribe():
    """Standalone speech-to-text for the Speech to Text panel.

    Wraps the same transcribe_video() the promo pipeline uses for dialogue-aware
    cuts, so what you see here is exactly what the rating stage sees. Accepts any
    media the server can decode (video or audio) -- the audio is extracted to
    16 kHz mono before upload either way."""
    src = _resolve_upload('stt_file', ALLOWED_EXTENSIONS | AUDIO_EXTENSIONS)
    if not src:
        return jsonify(ok=False, error='Upload a video or audio file to transcribe.'), 400
    try:
        words, segments = transcribe_video(src)
    finally:
        if os.path.exists(src) and not os.path.basename(src).startswith('net_'):
            try:
                os.remove(src)
            except OSError:
                pass
    if not segments and not words:
        return jsonify(ok=False, error='No speech was transcribed. Check that the whisper service '
                                       f'at {WHISPER_URL} is reachable (see the Config tab) and '
                                       'that the file actually contains audio.'), 502

    full = ' '.join(sg['text'] for sg in segments).strip()

    def _srt_time(t):
        h, rem = divmod(max(0.0, t), 3600)
        m, sec = divmod(rem, 60)
        return f'{int(h):02d}:{int(m):02d}:{int(sec):02d},{int((sec % 1) * 1000):03d}'

    srt = '\n'.join(f'{i+1}\n{_srt_time(sg["start"])} --> {_srt_time(sg["end"])}\n{sg["text"]}\n'
                    for i, sg in enumerate(segments))
    return jsonify(ok=True, segments=segments, words=len(words), text=full,
                   srt=srt, duration=round(segments[-1]['end'], 1) if segments else 0,
                   model=WHISPER_MODEL)

@app.route('/api/voices/tags')
def api_voice_tags():
    """Inline delivery tags for Fish Audio, in the syntax the configured model
    expects. Shared by the Narration section of the generate form and the Fish
    Audio tool so both offer the same set."""
    return jsonify(ok=True, **fish_tag_catalogue())

@app.route('/api/voices')
def api_voices():
    """Lists narration voices and languages for the Narration dropdowns, for
    the narration engine (Fish Audio; the ?engine= parameter is retained for
    compatibility with saved templates and existing API callers)
    to fish_audio). There's no bundled default voice — if the list comes back
    empty, the UI should fall back to "upload a reference sample" for
    zero-shot cloning."""
    force = request.args.get('refresh') == '1'
    engine = request.args.get('engine', 'fish_audio')
    engine = 'fish_audio'   # only narration engine; parameter kept for compatibility
    voices, source, error = list_voices_for_engine(engine, force=force)
    return jsonify(ok=error is None, voices=voices, languages=FISH_AUDIO_LANGUAGES,
                    source=source, error=error, engine=engine)

@app.route('/api/voices/clone', methods=['POST'])
def api_voices_clone():
    """Registers a new named voice on the self-hosted Fish Audio server (see
    fish_audio_add_reference). Admin-only: this is a shared, global resource
    -- every account picks from the same Voice dropdown -- same reasoning as
    /api/config being admin-only."""
    if session.get('role') != 'admin':
        return jsonify(ok=False, error='Admin access required.'), 403
    voice_id = (request.form.get('id') or '').strip()
    text = (request.form.get('text') or '').strip()
    audio = request.files.get('audio')
    if not audio or not audio.filename:
        return jsonify(ok=False, error='Choose a reference audio file.'), 400
    fn = secure_filename(audio.filename)
    if not fn:
        return jsonify(ok=False, error='Invalid audio filename.'), 400
    tmp_path = os.path.join(app.config['UPLOAD_FOLDER'], f'voiceclone_{int(time.time()*1000)}{os.path.splitext(fn)[1]}')
    audio.save(tmp_path)
    try:
        ok, err = fish_audio_add_reference(voice_id, tmp_path, text)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
    if not ok:
        return jsonify(ok=False, error=err), 502
    # The voice list is cached (_VOICES_CACHE_TTL) -- force a refresh so the
    # new voice shows up immediately rather than after the cache expires.
    list_voices_for_engine('fish_audio', force=True)
    return jsonify(ok=True, id=voice_id)

@app.route('/api/voices/delete', methods=['POST'])
def api_voices_delete():
    """Deletes a registered voice from the self-hosted Fish Audio server (see
    fish_audio_delete_reference). Admin-only, same reasoning as clone above."""
    if session.get('role') != 'admin':
        return jsonify(ok=False, error='Admin access required.'), 403
    voice_id = (request.form.get('id') or '').strip()
    ok, err = fish_audio_delete_reference(voice_id)
    if not ok:
        return jsonify(ok=False, error=err), 502
    list_voices_for_engine('fish_audio', force=True)
    return jsonify(ok=True, id=voice_id)

@app.route('/api/vo/preview', methods=['POST'])
@require_permission('text_to_speech', 'promo_generation')
def api_vo_preview():
    """Generates a short narration clip so the script/voice/rate/language can be
    checked by ear before it's used in an actual trailer job. Runs synchronously
    (previews are short) and returns a URL to the generated WAV, served from the
    same /uploads/<filename> route as everything else in UPLOAD_FOLDER."""
    text = (request.form.get('text') or '').strip()
    if not text:
        return jsonify(ok=False, error='No narration text to preview'), 400
    # In the Narration box this is an audition, so a hard cap keeps it snappy.
    # The Tools tab passes full=1 to render a complete script as a deliverable.
    cap = 5000 if request.form.get('full') in ('1', 'true', 'on') else 800
    truncated = len(text) > cap
    if truncated:
        text = text[:cap]
    try:
        rate = int(request.form.get('rate', 175))
    except ValueError:
        rate = 175
    voice_id = (request.form.get('voice') or '').strip() or None
    language = (request.form.get('language') or '').strip() or None
    engine = (request.form.get('engine') or 'fish_audio').strip()
    engine = 'fish_audio'   # only narration engine; parameter kept for compatibility

    ref_path = None
    if 'ref_upload' in request.files and request.files['ref_upload'].filename:
        f = request.files['ref_upload']
        fn = secure_filename(f.filename)
        if fn:
            ref_path = os.path.join(app.config['UPLOAD_FOLDER'], f'voprevref_{int(time.time()*1000)}{os.path.splitext(fn)[1]}')
            f.save(ref_path)
            voice_id = None  # an uploaded reference clone takes priority over a picked registered voice

    out_name = f'vopreview_{int(time.time()*1000)}.wav'
    out_path = os.path.join(app.config['UPLOAD_FOLDER'], out_name)
    ok, err = generate_tts(text, out_path, rate=rate, voice_id=voice_id,
                            reference_audio_path=ref_path, language=language, engine=engine)
    if not ok:
        return jsonify(ok=False, error=err or 'Preview generation failed'), 502
    return jsonify(ok=True, url=f'/uploads/{out_name}', filename=out_name,
                   engine=engine, truncated=truncated, characters=len(text),
                   duration=round(probe_duration(out_path) or 0, 1))

UPLOAD_TTL = int(os.environ.get('UPLOAD_TTL', 6 * 3600))  # seconds
_SWEEP_INTERVAL = int(os.environ.get('UPLOAD_SWEEP_INTERVAL', 900))

def sweep_upload_folder(ttl=None):
    """Deletes anything in UPLOAD_FOLDER older than `ttl` seconds.

    Backstop for the things per-job cleanup deliberately can't touch: shared
    net_* staging files, VO previews, and intermediates from a job whose process
    died mid-render. Finished trailers live on in LIBRARY_DIR, which this never
    touches, so reclaiming an aged /uploads/ copy only breaks a stale open tab.
    Returns bytes freed."""
    ttl = UPLOAD_TTL if ttl is None else ttl
    folder = app.config['UPLOAD_FOLDER']
    now = time.time()
    freed = 0
    try:
        entries = os.listdir(folder)
    except OSError:
        return 0
    for name in entries:
        p = os.path.join(folder, name)
        try:
            if not os.path.isfile(p):
                continue
            if now - os.path.getmtime(p) < ttl:
                continue
            size = os.path.getsize(p)
            os.remove(p)
            freed += size
        except OSError:
            pass
    if freed:
        print(f'Upload sweeper reclaimed {freed / (1024*1024):.1f} MB of files older than {ttl}s')
    return freed

def _sweeper_loop():
    while True:
        time.sleep(_SWEEP_INTERVAL)
        try:
            sweep_upload_folder()
        except Exception as e:
            print(f'Upload sweeper error: {e}')

def free_disk_mb(path=None):
    """Free space in MB on the filesystem holding `path` (UPLOAD_FOLDER by default)."""
    try:
        return shutil.disk_usage(path or app.config['UPLOAD_FOLDER']).free / (1024 * 1024)
    except OSError:
        return None

def _cleanup_job_temp(jid, params, keep_basename=None):
    """Removes every temp file a job could have created, whatever exit path it took.

    UPLOAD_FOLDER is a tempdir that only clears on process restart, so anything
    left behind here accumulates for the life of the server. Previously nothing
    was cleaned on any of the ~10 error returns in _run_trailer_job, and the
    uploaded source video was never deleted at all -- on a long-running LAN box
    that grows without bound until the disk fills.

    `keep_basename` is the finished trailer, which must survive (it's served from
    /uploads/ and copied into the library)."""
    folder = app.config['UPLOAD_FOLDER']
    victims = []
    # Per-job intermediates are all prefixed with the job id (base_ts == jid).
    try:
        for f in os.listdir(folder):
            if keep_basename and f == keep_basename:
                continue
            if f.endswith(f'_{jid}.mp4') or f.endswith(f'_{jid}.wav') or f.endswith(f'_{jid}.m4a'):
                victims.append(os.path.join(folder, f))
            elif f'_{jid}_' in f or f.startswith(f'seg_{jid}') or f.startswith(f'norm_{jid}'):
                victims.append(os.path.join(folder, f))
    except OSError as e:
        print(f'Job {jid} cleanup could not list {folder}: {e}')
    # Explicit per-job inputs (source video, staged template copies, uploads).
    for key in ('path', 'sfx_upload_path', 'vo_upload_path', 'vo_ref_upload_path',
                'scoring_audio_path', 'end_card_path', 'schedule_card_path',
                'title_card_vo_path', 'end_card_vo_path', 'transition_matte_path'):
        p = params.get(key)
        # scoring_audio_path carries the sentinel 'GENERATE' rather than a path.
        if not (isinstance(p, str) and p != 'GENERATE' and os.path.isabs(p)):
            continue
        # net_* files are the shared staging area for network-share picks: the
        # same staged file can legitimately be attached to two concurrent jobs,
        # so deleting it here could pull the source out from under the other one.
        # The age-based sweeper below reclaims those instead.
        if os.path.basename(p).startswith('net_'):
            continue
        victims.append(p)
    freed = 0
    for p in set(victims):
        try:
            if os.path.isfile(p):
                freed += os.path.getsize(p)
                os.remove(p)
        except OSError:
            pass
    if freed:
        print(f'Job {jid} cleanup freed {freed / (1024*1024):.1f} MB')

def run_trailer_job(jid, params):
    keep = None
    try:
        _run_trailer_job(jid, params)
        with JOBS_LOCK:
            res = (JOBS.get(jid) or {}).get('result') or {}
        url = res.get('trailer_url') or ''
        if url.startswith('/uploads/'):
            keep = os.path.basename(url)
    except JobCancelled:
        print(f'Trailer job {jid} cancelled')
        job_set(jid, error='Cancelled', status='cancelled')
    except MediaToolTimeout as e:
        print(f'Trailer job {jid} timed out: {e}')
        job_set(jid, error=f'A media processing step timed out and was stopped ({e}). '
                           'The source may be corrupt, or the server is overloaded.')
    except Exception as e:
        print(f'Trailer job {jid} crashed: {e}')
        job_set(jid, error=f'Unexpected error: {e}')
    finally:
        # Runs on success, failure, cancellation and timeout alike -- except for a
        # successful preview, whose source video and staged assets must survive
        # until the user renders (or the preview TTL expires and the age-based
        # sweeper reclaims them).
        holding = params.get('preview_only') and not (JOBS.get(jid) or {}).get('error')
        if not holding:
            _cleanup_job_temp(jid, params, keep_basename=keep)

def _run_trailer_job(jid, params):
    path = params['path']; orig_name = params['orig_name']; mode = params['mode']
    genre = params['genre']; scoring_mode = params['scoring_mode']; trailer_length = params['trailer_length']
    max_scene_dur = params.get('max_scene_dur')
    scene_threshold = params.get('scene_threshold', 30.0)
    min_scene_len_sec = params.get('min_scene_len_sec', 0.5)
    detector = params.get('detector', 'content')
    adaptive_threshold = params.get('adaptive_threshold', 3.0)
    transition = params['transition']; xfade_dur = params['xfade_dur']
    transition_matte_path = params.get('transition_matte_path')
    target_loudness = params['target_loudness']; true_peak = params['true_peak']
    music_duck_db = params.get('music_duck_db', -3)
    duck_depth_db = params.get('duck_depth_db', -15)
    duck_release_hold = params.get('duck_release_hold', 0.4)
    broadcast_stereo = params.get('broadcast_stereo', False)
    beat_match = params['beat_match']; model = params['model']
    sfx_mode = params['sfx_mode']; sfx_upload_path = params['sfx_upload_path']
    vo_mode = params['vo_mode']; vo_upload_path = params['vo_upload_path']; vo_text = params['vo_text']
    vo_voice = params['vo_voice']; vo_rate = params['vo_rate']; vo_start = params['vo_start']
    vo_language = params.get('vo_language'); vo_ref_upload_path = params.get('vo_ref_upload_path')
    vo_engine = params.get('vo_engine', 'fish_audio')
    vo_volume = params.get('vo_volume', 1.15)
    vo_trim_start = params.get('vo_trim_start', 0.0); vo_trim_end = params.get('vo_trim_end')
    sync_beats = params['sync_beats']
    whisper_enhance = params.get('whisper_enhance', False)
    end_card_path = params['end_card_path']; schedule_card_path = params['schedule_card_path']
    title_card_vo_path = params.get('title_card_vo_path'); title_card_vo_start = params.get('title_card_vo_start', 0.0); title_card_vo_end = params.get('title_card_vo_end')
    end_card_vo_path = params.get('end_card_vo_path'); end_card_vo_start = params.get('end_card_vo_start', 0.0); end_card_vo_end = params.get('end_card_vo_end')
    scoring_audio_path = params['scoring_audio_path']; prompt = params['prompt']

    job_set(jid, percent=2, step='Reading video info')
    last_ffmpeg_stderr = None
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps if fps else 0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    src_fps = fps if fps and fps > 0 else 30
    cap.release()

    trailer_duration = trailer_length
    min_required = trailer_length * 1.5
    if video_duration < min_required:
        job_set(jid, error=f'Video is only {video_duration:.1f}s long, but a {trailer_length}s episodic promo plug requires at least {min_required:.1f}s of raw video. Upload a longer video or select a shorter length.')
        return

    # Measure card durations before selecting scenes
    card_files = []
    card_durations = []
    _card_vo_ts = int(time.time() * 1000)
    if end_card_path and os.path.exists(end_card_path):
        if title_card_vo_path and os.path.exists(title_card_vo_path):
            muxed = os.path.join(app.config['UPLOAD_FOLDER'], f'titlecard_vo_{_card_vo_ts}.mp4')
            result = mux_card_vo(end_card_path, title_card_vo_path, title_card_vo_start, title_card_vo_end, muxed)
            if result:
                end_card_path = result
        card_files.append(end_card_path)
    if schedule_card_path and os.path.exists(schedule_card_path):
        if end_card_vo_path and os.path.exists(end_card_vo_path):
            muxed = os.path.join(app.config['UPLOAD_FOLDER'], f'endcard_vo_{_card_vo_ts}.mp4')
            result = mux_card_vo(schedule_card_path, end_card_vo_path, end_card_vo_start, end_card_vo_end, muxed)
            if result:
                schedule_card_path = result
        card_files.append(schedule_card_path)
    for cf in card_files:
        d = probe_duration(cf)
        if d is None or d <= 0:
            # Previously this silently substituted 5s. A wrong card duration feeds
            # the scene-budget maths and every xfade offset after it, so the whole
            # concat desyncs and the user gets a subtly broken trailer with no
            # error. Failing here is far better than guessing.
            job_set(jid, error=f'Could not read the duration of the card video "{os.path.basename(cf)}" — '
                               'it may be corrupt or in an unsupported format. Re-export it and try again.')
            return
        card_durations.append(d)
    total_card_dur = sum(card_durations)

    # Scene target starts at trailer_length, minus cards duration
    base_target = max(5, trailer_length - total_card_dur)

    # Shared by both paths below. base_ts is the per-job filename prefix for every
    # intermediate; it used to be assigned partway through the analysis block,
    # which the resume path skips entirely.
    base_ts = jid  # already unique per job (see job_new()) -- using it here too
                   # avoids two jobs finishing a step in the same second (quite
                   # possible with MAX_CONCURRENT_JOBS > 1) from both writing to
                   # the same trailer_<ts>.mp4 path at once.
    early_bgm_path = None
    early_bgm_source = 'none'

    preselected = params.get('preselected')
    if preselected:
        # Rendering an approved preview: detection, quality scoring, AI vision
        # scoring and selection were all done during the preview pass, so skip
        # straight to extraction. On a long episode that's the majority of the
        # job's wall-clock time, and repeating it could also produce a slightly
        # different cut than the one the user actually approved.
        job_set(jid, percent=34, step=f'Rendering approved cut ({len(preselected)} clips)')
        selected = [dict(s) for s in preselected]
        selected.sort(key=lambda x: x['start'])
        total_sel = sum(s['selected_dur'] for s in selected)
        scene_list = [None] * int(params.get('preview_total_scenes') or len(selected))
        word_starts = word_ends = []
        beat_times = []
    else:
        job_set(jid, percent=8, step='Detecting scene cuts')
        # Detect scenes via PySceneDetect. downscale=2 speeds up detection on large
        # source files (frames are only scaled down for the detector's own
        # analysis; returned timecodes are unaffected).
        scene_list = detect_scenes(path, threshold=scene_threshold,
                                    min_scene_len_sec=min_scene_len_sec, downscale=2,
                                    detector=detector, adaptive_threshold=adaptive_threshold)
        if not scene_list:
            job_set(jid, error='No scene changes detected. Try a video with clear cuts, or lower the detection threshold.')
            return
        if len(scene_list) == 1 and (tc_seconds(scene_list[0][1]) - tc_seconds(scene_list[0][0])) > video_duration * 0.95:
            # PySceneDetect's own fallback: no real cuts found, so it returned one
            # scene spanning the whole video. Selecting from a single "scene" isn't
            # meaningful — surface this clearly instead of silently treating the
            # entire source as one giant clip.
            job_set(jid, error='No distinct scene cuts were found — PySceneDetect sees this video as one continuous shot. Try lowering the detection threshold or upload footage with visible cuts.')
            return

        job_set(jid, percent=15, step=f'Rating {len(scene_list)} scenes (sharpness/brightness)')
        # Score scenes
        from statistics import median
        def _score_one_scene(start, end):
            # Each worker opens its own VideoCapture — cv2.VideoCapture is not safe to
            # share across threads (concurrent .set()/.read() calls on one handle can
            # corrupt each other's seeks), but independent handles on the same file
            # decode concurrently just fine and this is what actually lets scene
            # scoring use more than one CPU core.
            local_cap = cv2.VideoCapture(path)
            try:
                mid_f = int((tc_frames(start) + tc_frames(end)) / 2)
                local_cap.set(cv2.CAP_PROP_POS_FRAMES, mid_f)
                ret, frame = local_cap.read()
                if not ret:
                    return None
                dur = tc_seconds(end) - tc_seconds(start)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                lap = cv2.Laplacian(gray, cv2.CV_64F).var()
                bri = float(np.mean(gray))
                h, w = gray.shape
                edges = cv2.Canny(gray, 50, 150)
                edge_ratio = float(np.count_nonzero(edges)) / (h * w)
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                mean_hue = float(np.mean(hsv[:,:,0]))
                mean_sat = float(np.mean(hsv[:,:,1]))
                mean_val = float(np.mean(hsv[:,:,2]))
                has_face = False
                if ONNX_PATH is not None:
                    # get_fd() returns one cached, shared FaceDetectorYN instance — its
                    # setInputSize()+detect() pair isn't safe for concurrent callers, so
                    # this part alone stays serialized. It's cheap relative to decode +
                    # Laplacian/Canny/HSV, so the lock doesn't erase the parallel gain.
                    with _fd_lock:
                        _, faces = get_fd(w, h).detect(frame)
                    has_face = faces is not None and len(faces) > 0
                return {
                    'start': tc_seconds(start), 'end': tc_seconds(end),
                    'start_f': tc_frames(start), 'end_f': tc_frames(end),
                    'duration': dur, 'laplacian': round(lap, 2), 'brightness': round(bri, 1),
                    'edge_ratio': round(edge_ratio, 3), 'mean_hue': round(mean_hue, 1),
                    'mean_sat': round(mean_sat, 1), 'mean_val': round(mean_val, 1),
                    'has_face': has_face, 'frame': frame, 'frame_idx': mid_f,
                }
            finally:
                local_cap.release()

        with ThreadPoolExecutor(max_workers=min(8, len(scene_list) or 1)) as ex:
            scored = list(ex.map(lambda se: _score_one_scene(se[0], se[1]), scene_list))
        scenes_data = [r for r in scored if r is not None]
        if not scenes_data:
            job_set(jid, error='No frames could be read.')
            return

        med_lap = median([s['laplacian'] for s in scenes_data])
        med_bri = median([s['brightness'] for s in scenes_data])
        for s in scenes_data:
            score = 0
            if s['laplacian'] > med_lap * 1.2: score += 2
            elif s['laplacian'] > med_lap * 0.8: score += 1
            if 80 < s['brightness'] < 180: score += 2
            elif s['brightness'] > 30: score += 1
            if 1 < s['duration'] < 8: score += 2
            elif s['duration'] > 8: score += 1
            elif s['duration'] < 0.7:
                # Sub-fragment scenes (whip-pans, flash cuts) can still win on raw
                # sharpness/brightness alone with no duration credit at all —
                # penalize them explicitly so they don't out-rank a real scene of
                # similar visual quality and end up as a flicker cut in the output.
                score -= 1
            if s['has_face']:
                # A face/reaction shot is generally more useful in a trailer than
                # empty B-roll of similar sharpness/brightness.
                score += 1
            s['quality_score'] = score

        if mode == 'ai':
            # Only AI-score scenes that could realistically make the cut. Previously
            # every detected scene got a vision call -- on a 45-minute episode that's
            # 200-400 Ollama round trips to choose ~12 clips, and it dominated the
            # job's wall-clock time.
            #
            # Candidates are chosen two ways so the shortlist stays both good and
            # spread out: the globally highest quality_score scenes, plus the best
            # few from each time bucket across the source. Without the buckets a
            # shortlist can cluster in one well-lit stretch and then get thinned out
            # by the min_gap spacing rule during selection, leaving nothing to pick.
            ai_pool = scenes_data
            if len(scenes_data) > AI_SCORE_LIMIT:
                ranked = sorted(scenes_data, key=lambda s: s['quality_score'], reverse=True)
                chosen = {id(s): s for s in ranked[:max(1, AI_SCORE_LIMIT // 2)]}
                n_buckets = max(1, min(AI_SCORE_LIMIT // 4, 12))
                span = max(video_duration, 1e-6)
                buckets = {}
                for s in scenes_data:
                    b = min(n_buckets - 1, int(s['start'] / span * n_buckets))
                    buckets.setdefault(b, []).append(s)
                per_bucket = max(1, (AI_SCORE_LIMIT - len(chosen)) // n_buckets)
                for b in sorted(buckets):
                    for s in sorted(buckets[b], key=lambda x: x['quality_score'], reverse=True)[:per_bucket]:
                        if len(chosen) >= AI_SCORE_LIMIT:
                            break
                        chosen.setdefault(id(s), s)
                ai_pool = list(chosen.values())
                print(f'AI scoring shortlist: {len(ai_pool)} of {len(scenes_data)} scenes '
                      f'(cap AI_SCORE_LIMIT={AI_SCORE_LIMIT})')

            # Anything not shortlisted keeps a neutral AI prior -- the same value used
            # when a vision response can't be parsed -- so it stays selectable if
            # spacing rules exhaust the shortlist, just ranked below scored scenes.
            ai_pool_ids = {id(s) for s in ai_pool}
            for s in scenes_data:
                if id(s) not in ai_pool_ids:
                    s['total_score'] = s['quality_score'] + AI_NEUTRAL_SCORE
                    s['ai_desc'] = ''

            # Tell the model what it's selecting *for*. The prompt never mentioned the
            # genre before, so an action promo and a drama promo were ranked by the
            # same generic "good for a movie trailer" criterion.
            ai_prompt = prompt
            if genre in GENRE_PRESETS and 'DESC:' in prompt:
                ai_prompt = prompt.replace('for a movie trailer',
                                           f'for a {genre} promo trailer', 1)

            n_scenes_ai = len(ai_pool)
            _ai_progress = {'done': 0}
            _ai_progress_lock = threading.Lock()

            def _ask_vision(b64, budget, allow_thinking, structured=True):
                """One /api/generate call. Returns (text, response_json).

                Two things fight us on a chatty reasoning model like qwen3-vl:
                  * it puts chain-of-thought in a separate `thinking` field, and
                    Ollama counts those tokens against num_predict; and
                  * told to answer in one line, it writes a paragraph anyway and
                    gets truncated (done_reason='length') before the score.

                Prompt wording alone does not fix either. Ollama's structured
                output does: passing a JSON schema in `format` constrains decoding
                to that shape, so the model cannot ramble and stops as soon as the
                object closes. That both removes the truncation failure and makes
                each call much shorter -- which matters at ~60 frames a job.
                `structured=False` is the fallback for older Ollama builds that
                reject the `format` field."""
                payload = {
                    'model': model, 'prompt': ai_prompt, 'stream': False, 'images': [b64],
                    # Near-greedy decoding: scene ranking should be reproducible
                    # across runs of the same source, and the default sampling
                    # temperature made scores jitter between jobs.
                    'options': {'temperature': 0.1, 'num_predict': budget},
                }
                if structured and AI_STRUCTURED_OK:
                    payload['format'] = {
                        'type': 'object',
                        'properties': {
                            'score': {'type': 'integer', 'minimum': 1, 'maximum': 5},
                            'desc': {'type': 'string'},
                        },
                        'required': ['score', 'desc'],
                    }
                if not allow_thinking:
                    # Understood by newer Ollama for reasoning models; older
                    # builds ignore the unknown key rather than erroring.
                    payload['think'] = False
                r = requests.post(f'{OLLAMA_URL}/api/generate', json=payload, timeout=180)
                data = r.json()
                if data.get('error'):
                    raise RuntimeError(data['error'])
                txt = (data.get('response') or '').strip()
                if not txt:
                    txt = (data.get('thinking') or '').strip()
                return txt, data

            def _parse_vision(txt):
                # Structured output path: a JSON object is the expected shape now,
                # so try that before any of the text heuristics below.
                if txt.lstrip().startswith('{'):
                    try:
                        obj = json.loads(txt)
                        sc = int(obj.get('score'))
                        if 1 <= sc <= 5:
                            return _clean_ai_desc(str(obj.get('desc') or '')), sc
                    except (ValueError, TypeError, AttributeError):
                        pass  # malformed -- salvage below
                    # Truncated JSON is the common failure (done_reason='length'
                    # cuts the object mid-string), so pull the fields out by hand
                    # rather than discarding a reply that has both values in it.
                    j_sc = re.search(r'"score"\s*:\s*([1-5])\b', txt)
                    j_de = re.search(r'"desc"\s*:\s*"([^"]*)', txt)
                    if j_sc:
                        return (_clean_ai_desc(j_de.group(1)) if j_de else ''), int(j_sc.group(1))
                    if j_de:
                        # Score unusable/out of range, but the description is fine.
                        txt = j_de.group(1)

                """(description, score) from a model reply, or (desc, None) if no
                score could be found.

                The prompt asks for SCORE first precisely because chatty models run
                past the token budget mid-sentence: with the score leading, a reply
                truncated by `length` still yields a usable rating and whatever
                description made it out. Either field order is accepted, since
                templates saved with the old prompt still ask for DESC first."""
                # Strip markdown emphasis first: models often bold the labels,
                # which breaks a plain 'SCORE:' match.
                txt = re.sub(r'[*_`]+', '', txt or '')
                score_m = re.search(r'SCORE\s*:?\s*([1-5])', txt, re.I)
                if not score_m:
                    # Models that ignore the format usually still emit a bare digit.
                    score_m = (re.search(r'\b([1-5])\s*/\s*5\b', txt)
                               or re.search(r'\b([1-5])\s*$', txt.strip()))
                desc_m = re.search(r'DESC\s*:?\s*(.+?)(?:\s*\|\s*SCORE|$)', txt, re.S | re.I)
                if not desc_m:
                    # No DESC label (or it was cut off): fall back to the longest
                    # sentence-ish run of prose in the reply.
                    plain = re.sub(r'SCORE\s*:?\s*[1-5]\s*(?:/\s*5)?\s*\|?', ' ', txt, flags=re.I)
                    plain = re.sub(r'#+', ' ', plain)
                    cand = max((p.strip() for p in re.split(r'[.\n]', plain)),
                               key=len, default='')
                    # A bare rating digit at the end is the score, not prose.
                    cand = re.sub(r'\s*\b[1-5]\s*(?:/\s*5)?\s*$', '', cand).strip()
                    desc = _clean_ai_desc(cand) if len(cand) > 12 else ''
                else:
                    desc = _clean_ai_desc(desc_m.group(1))
                return desc, (int(score_m.group(1)) if score_m else None)

            def _score_one_ai(ai_i, s):
                _, buf = cv2.imencode('.jpg', s['frame'], [cv2.IMWRITE_JPEG_QUALITY, 85])
                b64 = base64.b64encode(buf.tobytes()).decode()
                global AI_STRUCTURED_OK
                try:
                    # Retry on an UNPARSEABLE reply, not merely an empty one: a
                    # reasoning model that ran out of budget leaves `response`
                    # empty but `thinking` full of chain-of-thought, which is
                    # non-empty yet useless. Treating that as an answer is what
                    # silently sent every scene to the neutral score.
                    try:
                        txt, data = _ask_vision(b64, AI_NUM_PREDICT, allow_thinking=False)
                    except RuntimeError as e:
                        if 'format' not in str(e).lower():
                            raise
                        # Ollama too old for structured output -- disable it for
                        # the rest of this process and carry on unstructured.
                        print('Ollama rejected the structured-output `format` field; '
                              'falling back to text parsing for this session. '
                              'Upgrade Ollama for more reliable scene rating.')
                        AI_STRUCTURED_OK = False
                        txt, data = _ask_vision(b64, AI_NUM_PREDICT, allow_thinking=False,
                                                structured=False)
                    desc, score = _parse_vision(txt)
                    if score is None:
                        print(f'AI vision reply unusable for scene {ai_i+1} '
                              f'(done_reason={data.get("done_reason")!r}, {len(txt)} chars); '
                              f'retrying without a token cap. First 200 chars: {txt[:200]!r}')
                        txt, data = _ask_vision(b64, AI_NUM_PREDICT * 3,
                                                allow_thinking=True, structured=False)
                        desc, score = _parse_vision(txt)
                    if score is None:
                        print(f'AI vision score parse failed for scene {ai_i+1}, defaulting to '
                              f'{AI_NEUTRAL_SCORE}. done_reason={data.get("done_reason")!r} '
                              f'raw={txt[:300]!r}')
                    s['ai_desc'] = desc
                    s['total_score'] = s['quality_score'] + (score if score is not None else AI_NEUTRAL_SCORE)
                except Exception as e:
                    print(f'AI vision request failed for scene {ai_i+1}, defaulting to {AI_NEUTRAL_SCORE}: {e}')
                    s['total_score'] = s['quality_score'] + AI_NEUTRAL_SCORE
                with _ai_progress_lock:
                    _ai_progress['done'] += 1
                    job_set(jid, percent=18 + int(12 * _ai_progress['done'] / max(n_scenes_ai, 1)),
                            step=f"AI-rating scene {_ai_progress['done']}/{n_scenes_ai}")

            # Bounded concurrency, not unbounded: these are independent HTTP calls (so
            # parallelizing them is the single biggest wall-clock win in the whole
            # pipeline when Ollama is slow), but firing all of them at once could
            # overwhelm a single Ollama instance's request queue or GPU memory. 4 is a
            # reasonable default for a typical self-hosted single-GPU setup.
            with ThreadPoolExecutor(max_workers=min(AI_SCORE_WORKERS, n_scenes_ai or 1)) as ex:
                list(ex.map(lambda pair: _score_one_ai(*pair), enumerate(ai_pool)))
        else:
            for s in scenes_data:
                s['total_score'] = s['quality_score']

        # Dialogue transcription (faster-whisper) — improves scene selection two ways:
        # 1. Scenes with actual quotable dialogue get a small scoring boost, so
        #    selection isn't purely based on visual sharpness/brightness/AI framing.
        # 2. Word-level timestamps let cut in/out points snap to word boundaries
        #    later, instead of landing mid-word.
        word_starts, word_ends = [], []
        if whisper_enhance:
            job_set(jid, percent=22, step='Transcribing dialogue (faster-whisper)')
            words, segments = transcribe_video(path)
            if words or segments:
                word_starts = [w['start'] for w in words]
                word_ends = [w['end'] for w in words]
                for s in scenes_data:
                    overlap_text = ' '.join(
                        sg['text'] for sg in segments if sg['start'] < s['end'] and sg['end'] > s['start']
                    ).strip()
                    s['dialogue'] = overlap_text
                    if overlap_text:
                        bonus = 1
                        if '?' in overlap_text or '!' in overlap_text:
                            bonus += 1
                        s['total_score'] += bonus
            else:
                whisper_enhance = False  # transcription unavailable/failed — skip the snapping logic below too

        # "Edit to music": prep the BGM *before* picking scenes so cut points can be
        # snapped onto its beat grid. Only worth the extra generation pass when the
        # user actually asked for it — otherwise BGM is prepared later as before.
        base_ts = jid  # already unique per job (see job_new()) -- using it here too
                       # avoids two jobs finishing this step in the same second
                       # (quite possible with MAX_CONCURRENT_JOBS > 1) from both
                       # writing to the same trailer_<ts>.mp4 path at once.
        beat_times = []
        if sync_beats:
            job_set(jid, percent=20, step='Preparing music for beat-synced cuts')
            early_bgm_path, early_bgm_source = prepare_bgm_track(genre, scoring_mode, scoring_audio_path,
                                                                  base_target, base_ts)
            if early_bgm_path:
                beat_times = detect_beat_times(early_bgm_path, base_target)
                if not beat_times:
                    sync_beats = False  # detection failed (e.g. librosa missing) — fall back silently
            else:
                sync_beats = False

        job_set(jid, percent=28, step='Selecting best scenes')
        # Pick top scenes by score to fill target, then sort by timecode
        # Iterative: xfade transitions shorten output, so compensate
        # Floor for how short a *budget-truncated* clip is allowed to be. Without this,
        # whichever scene happens to land last (in score order, not timeline order) just
        # gets clipped to "whatever duration is left" — which can be a fraction of a
        # second. That sliver then lands wherever its timecode falls once we re-sort by
        # start time, often producing a jarring near-invisible cut right before the end.
        # A scene's own *natural* PySceneDetect duration can still be shorter than this
        # (that's a legitimate quick cut in the source) — this floor only stops us from
        # truncating a longer scene down below it just to hit the target duration exactly.
        min_seg_dur = max(0.8, xfade_dur * 2.5)
        if max_scene_dur:
            min_seg_dur = min(min_seg_dur, max_scene_dur)
        # Minimum spacing (in source timeline seconds) required between two
        # selected scenes' start points. Without this, greedy score-based
        # selection can pick several scenes from the same high-scoring stretch of
        # the video (e.g. one well-lit dialogue scene) and leave the rest of the
        # source entirely unrepresented. Scaled to the source length but bounded
        # so it's meaningful on both short and long videos.
        base_min_gap = max(2.0, min(8.0, video_duration * 0.03))
        trailer_duration = base_target
        for pass_attempt in range(4):
            # Relax the gap requirement on later passes: if spacing is preventing
            # us from filling the duration budget, it's better to allow some
            # clustering than to ship a trailer that's noticeably short.
            min_gap = max(1.0, base_min_gap - pass_attempt * (base_min_gap / 4))
            scenes_data.sort(key=lambda x: x['total_score'], reverse=True)
            selected = []
            total_sel = 0
            for s in scenes_data:
                remaining = trailer_duration - total_sel
                if remaining < min_seg_dur:
                    # Not enough budget left for a decent-length clip — stop selecting
                    # rather than truncating the next scene into a sliver. The
                    # shortfall gets absorbed by nudging trailer_duration up on the
                    # next pass_attempt below.
                    break
                if any(abs(s['start'] - c['start']) < min_gap for c in selected):
                    # Too close in the source timeline to an already-selected scene
                    # — skip it in favor of spreading selections across the video,
                    # rather than over-sampling one stretch of it.
                    continue
                seg_dur = min(s['duration'], remaining)
                if max_scene_dur:
                    seg_dur = min(seg_dur, max_scene_dur)
                seg_start = s['start']
                if whisper_enhance and word_starts:
                    # Don't start playback mid-word — nudge the in-point forward to
                    # the start of the nearest word within this scene (capped so we
                    # never drift far from the original visual cut point).
                    snapped_start = nearest_word_boundary(seg_start, word_starts, max_snap=0.35)
                    if seg_start < snapped_start < seg_start + seg_dur:
                        drift = snapped_start - seg_start
                        seg_start = snapped_start
                        seg_dur = max(0.3, seg_dur - drift)
                scene_end = s['start'] + s['duration']
                if sync_beats and beat_times:
                    # Nudge this segment's end so the *cumulative* cut point lands on
                    # the nearest beat, within what this scene can actually supply.
                    target_cut = total_sel + seg_dur
                    snapped_cut = nearest_beat(target_cut, beat_times, total_sel + 0.3, total_sel + (scene_end - seg_start))
                    seg_dur = max(0.3, min(scene_end - seg_start, snapped_cut - total_sel))
                if whisper_enhance and word_ends and seg_dur < (scene_end - seg_start):
                    # This is a separate `if`, not `elif` -- beat-sync above (when
                    # enabled) picks a rhythmically-aligned out-point first, and this
                    # then refines THAT point for word safety, rather than being
                    # skipped whenever beat-sync is on. It used to be `elif`, which
                    # meant turning on "sync cuts to the beat" silently disabled
                    # "don't cut mid-word" for every clip's out-point.
                    target_end = seg_start + seg_dur
                    snapped_end = nearest_word_boundary(target_end, word_ends, max_snap=0.35)
                    if seg_start < snapped_end <= scene_end:
                        seg_dur = max(0.3, snapped_end - seg_start)
                s['trim_start'] = seg_start
                s['selected_dur'] = seg_dur
                selected.append(s)
                total_sel += seg_dur
            selected.sort(key=lambda x: x['start'])

            n_seg = len(selected) + len(card_files)
            xfade_loss = max(0, (n_seg - 1)) * xfade_dur
            expected_total = total_sel + total_card_dur - xfade_loss
            shortfall = trailer_length - expected_total
            if abs(shortfall) <= 0.5 or pass_attempt == 3:
                break
            trailer_duration = total_sel + shortfall * 1.15

        # A positive shortfall here means every pass_attempt ran out of usable
        # scenes (limited spacing/availability) before hitting the target, and
        # the "no tiny sliver clips" rule above refused to add one more to close
        # a small gap. Rather than ship a trailer noticeably under the requested
        # length, extend the chronologically LAST selected clip using slack it
        # already has within its own detected scene boundaries -- this grows an
        # existing cut instead of adding a new one, so it doesn't reintroduce
        # the flash-cut problem that rule exists to prevent.
        if selected and shortfall > 0.15:
            last = selected[-1]
            slack = last['duration'] - last['selected_dur']
            if slack > 0.05:
                grow = min(slack, shortfall)
                new_end = last['trim_start'] + last['selected_dur'] + grow
                if whisper_enhance and word_ends:
                    # Growing this clip to close the shortfall creates a new cut
                    # point too -- give it the same word-boundary safety the main
                    # truncation path gets above, or this top-up can reintroduce
                    # exactly the mid-word cut the rest of this mechanism exists
                    # to prevent.
                    scene_end_abs = last['start'] + last['duration']
                    snapped = nearest_word_boundary(new_end, word_ends, max_snap=0.35)
                    if last['trim_start'] < snapped <= scene_end_abs:
                        new_end = snapped
                grow = max(0.0, new_end - (last['trim_start'] + last['selected_dur']))
                last['selected_dur'] += grow
                total_sel += grow
                shortfall -= grow

    if not selected:
        job_set(jid, error='No scenes selected.')
        return

    if params.get('preview_only'):
        # Analysis is done; stop here instead of spending minutes on extraction,
        # transitions and mixing for a cut the user hasn't seen yet.
        pid = f'pv{jid}'
        job_set(jid, percent=34, step='Writing preview thumbnails')

        # Runner-ups: the next-best scoring scenes that didn't make the cut, so a
        # rejected clip can be swapped for a real alternative instead of forcing a
        # full re-analysis. Spaced by the same min_gap rule the selector uses, and
        # excluding anything already chosen.
        chosen_starts = {round(s['start'], 3) for s in selected}
        alternates = []
        if not preselected:
            for cand in sorted(scenes_data, key=lambda x: x['total_score'], reverse=True):
                if len(alternates) >= PREVIEW_ALTERNATES:
                    break
                if round(cand['start'], 3) in chosen_starts:
                    continue
                if any(abs(cand['start'] - o['start']) < min_gap for o in alternates):
                    continue
                if any(abs(cand['start'] - s['start']) < min_gap for s in selected):
                    continue
                cand = dict(cand)
                cand.setdefault('selected_dur', min(cand['duration'], max_scene_dur or cand['duration']))
                cand.setdefault('trim_start', cand['start'])
                alternates.append(cand)

        def _thumb(scene, tag, i):
            frame = scene.get('frame')
            if frame is None:
                return None
            try:
                tw = 320
                h, w = frame.shape[:2]
                small = cv2.resize(frame, (tw, max(1, int(h * tw / max(w, 1)))))
                tname = f'preview_{pid}_{tag}{i}.jpg'
                cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], tname), small,
                            [cv2.IMWRITE_JPEG_QUALITY, 78])
                return f'/uploads/{tname}'
            except Exception as e:
                print(f'Preview thumbnail {tag}{i} failed: {e}')
                return None

        thumbs = [_thumb(s, 's', i) for i, s in enumerate(selected)]
        alt_thumbs = [_thumb(s, 'a', i) for i, s in enumerate(alternates)]

        def _slim(rows):
            # Only the fields the render half consumes -- frames and other
            # non-serializable analysis state are deliberately dropped.
            return [{'start': s['start'], 'end': s['end'], 'duration': s['duration'],
                     'selected_dur': s['selected_dur'], 'trim_start': s.get('trim_start', s['start']),
                     'total_score': s['total_score'], 'quality_score': s.get('quality_score', 0),
                     'ai_desc': s.get('ai_desc', ''), 'has_face': s.get('has_face', False),
                     'edge_ratio': s.get('edge_ratio', 0), 'mean_hue': s.get('mean_hue', 0)}
                    for s in rows]

        slim = _slim(selected)
        slim_alt = _slim(alternates)
        # basename of the source video under UPLOAD_FOLDER, handed back so the
        # preview grid's Play buttons can play a selected scene straight from
        # /uploads/<name> -- same file the render step itself reads from.
        video_filename = os.path.basename(params['path'])
        preview_store(pid, {'params': params, 'selected': slim, 'thumbs': thumbs,
                            'alternates': slim_alt, 'alt_thumbs': alt_thumbs,
                            'total_scenes': len(scene_list), 'video_duration': video_duration,
                            'total_card_dur': total_card_dur})
        result = dict(
            status='preview', preview=True, preview_id=pid,
            orig_name=orig_name, total_scenes=len(scene_list), selected_scenes=len(selected),
            video_duration=round(video_duration, 1), trailer_length=trailer_length,
            scenes_duration=round(total_sel, 1),
            estimated_duration=round(total_sel + total_card_dur - max(0, len(selected) + len(card_files) - 1) * xfade_dur, 1),
            video_filename=video_filename,
            scenes=[{'scene': i + 1, 'start': round(s['start'], 1), 'end': round(s['end'], 1),
                     'quality': s['total_score'], 'duration': round(s['selected_dur'], 1),
                     'description': _scene_desc(s), 'thumb': thumbs[i]}
                    for i, s in enumerate(selected)],
            alternates=[{'alt': i + 1, 'start': round(s['start'], 1), 'end': round(s['end'], 1),
                         'quality': s['total_score'], 'duration': round(s['selected_dur'], 1),
                         'description': _scene_desc(s), 'thumb': alt_thumbs[i]}
                        for i, s in enumerate(alternates)])
        job_set(jid, percent=100, step='Preview ready', done=True, result=result)
        return

    job_set(jid, percent=38, step=f'Extracting {len(selected)} selected clips')
    # Extract selected segments + card videos as temp files. `extracted`
    # tracks which of `selected` actually produced a usable clip, so stats
    # reported to the user (selected_scenes, trailer_duration) reflect what's
    # really in the output rather than what was merely picked.
    #
    # Run in parallel: each clip is an independent ffmpeg process reading the same
    # source read-only, so there's nothing to serialize. This used to be a plain
    # sequential loop and was one of the longest single stages of the job.
    _extract_errors = []
    _extract_lock = threading.Lock()
    _extract_progress = {'done': 0}
    n_to_extract = len(selected)

    def _extract_one(seg_i, seg):
        out_seg = os.path.join(app.config['UPLOAD_FOLDER'], f'seg_{base_ts}_{seg_i}.mp4')
        trim_start = seg.get('trim_start', seg['start'])

        def _enc(fast_seek):
            # fast_seek puts -ss before -i (input seeking: quick, but can land
            # awkwardly relative to keyframes). The retry puts it after -i
            # (output seeking: decodes from the start, slower but exact).
            pre = ['-ss', str(trim_start), '-i', path] if fast_seek else ['-i', path, '-ss', str(trim_start)]
            return [FFMPEG, '-y'] + pre + [
                '-t', str(seg['selected_dur']),
                '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '28', '-pix_fmt', 'yuv420p',
                '-c:a', 'aac', '-b:a', '128k', out_seg]

        for attempt, fast in enumerate((True, False)):
            try:
                r = run_ffmpeg(_enc(fast), label=f'clip extract @{trim_start:.1f}s')
                stderr = r.stderr
            except MediaToolTimeout as e:
                stderr = str(e)
            if os.path.exists(out_seg) and os.path.getsize(out_seg) > 0:
                with _extract_lock:
                    _extract_progress['done'] += 1
                    job_set(jid, percent=38 + int(10 * _extract_progress['done'] / max(n_to_extract, 1)),
                            step=f"Extracting clips {_extract_progress['done']}/{n_to_extract}")
                return seg_i, out_seg, seg
            print(f'FFMPEG seg extraction {"error" if attempt == 0 else "retry also failed"} '
                  f'(scene at {trim_start}s): {stderr[:500]}')
            with _extract_lock:
                _extract_errors.append(stderr[-800:])
        with _extract_lock:
            _extract_progress['done'] += 1
        return None

    with ThreadPoolExecutor(max_workers=min(EXTRACT_WORKERS, n_to_extract or 1)) as ex:
        results = list(ex.map(lambda p: _extract_one(*p), enumerate(selected)))

    # Reassemble in the original order — ThreadPoolExecutor.map preserves input
    # order, but filter first so a dropped clip doesn't shift the rest.
    ok_results = [r for r in results if r is not None]
    seg_files = [r[1] for r in ok_results]
    extracted = [r[2] for r in ok_results]
    if _extract_errors:
        last_ffmpeg_stderr = _extract_errors[-1]

    if len(extracted) < len(selected):
        dropped = len(selected) - len(extracted)
        print(f'{dropped} selected scene(s) failed extraction and were dropped from the trailer.')
    selected = extracted
    total_sel = sum(s['selected_dur'] for s in selected)

    if not selected:
        job_set(jid, error='All selected scenes failed to extract.' + (f' Last ffmpeg error: {last_ffmpeg_stderr}' if last_ffmpeg_stderr else ''))
        return

    all_inputs = seg_files + card_files
    n_total = len(all_inputs)

    out_path = os.path.join(app.config['UPLOAD_FOLDER'], f'trailer_{base_ts}.mp4')
    sfx_timestamps = []

    norm = (f'scale={src_w}:{src_h}:force_original_aspect_ratio=decrease,'
            f'pad={src_w}:{src_h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={src_fps}')

    job_set(jid, percent=50, step='Building transitions & crossfades')
    if n_total == 1:
        try:
            r = run_ffmpeg([FFMPEG, '-y', '-i', all_inputs[0], '-vf', norm,
                            '-c:v', 'libx264', '-preset', 'fast', '-crf', '22',
                            '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '128k', out_path],
                           label='single-clip encode')
            if r.returncode != 0:
                print(f'FFMPEG single concat error: {r.stderr[:500]}')
                last_ffmpeg_stderr = r.stderr[-800:]
        except MediaToolTimeout as e:
            last_ffmpeg_stderr = str(e)
    else:
        # One metadata pass per input (duration + whether it carries audio),
        # replacing what used to be three separate ffprobe spawns per input.
        # Probed in parallel since these are independent read-only calls.
        with ThreadPoolExecutor(max_workers=min(8, n_total)) as ex:
            infos = list(ex.map(probe_media_info, all_inputs))

        # Normalize every input to ensure consistent video/audio before xfade.
        # Only re-encode if audio is missing (add silent audio as fallback).
        normed_inputs = []
        for i, (inp, info) in enumerate(zip(all_inputs, infos)):
            if info['has_audio']:
                normed_inputs.append(inp)
                continue
            normed = os.path.join(app.config['UPLOAD_FOLDER'], f'norm_{base_ts}_{i}.mp4')
            try:
                run_ffmpeg([FFMPEG, '-y', '-i', inp,
                            '-f', 'lavfi', '-i', 'anullsrc=channel_layout=stereo:sample_rate=44100',
                            '-c:v', 'copy', '-c:a', 'aac', '-b:a', '128k',
                            '-map', '0:v:0', '-map', '1:a:0', '-shortest', normed],
                           timeout=120, label='silent-audio mux')
            except MediaToolTimeout as e:
                print(f'Silent-audio mux timed out for input {i}: {e}')
            if os.path.exists(normed) and os.path.getsize(normed) > 0:
                normed_inputs.append(normed)
                # Adding an audio track can change the container duration
                # slightly, so this one input gets re-probed; the untouched
                # inputs keep the duration measured above.
                d = probe_duration(normed)
                if d and d > 0:
                    infos[i] = {'duration': d, 'has_audio': True}
            else:
                normed_inputs.append(inp)

        durations = [info['duration'] for info in infos]
        if any(d is None or d <= 0 for d in durations):
            # Every xfade offset is computed by summing these. A single bad value
            # (this used to silently become 5.0) shifts every transition after it
            # and desynchronises audio from video for the rest of the trailer, so
            # this fails the job instead of shipping a subtly broken render.
            bad = [os.path.basename(all_inputs[i]) for i, d in enumerate(durations) if d is None or d <= 0]
            job_set(jid, error='Could not read the duration of these clips, so transition timing '
                               f'could not be calculated reliably: {", ".join(bad[:5])}. '
                               'This usually means ffprobe is missing or a clip is corrupt.')
            return

        all_inputs = normed_inputs
        n_total = len(all_inputs)

        norm = (f'scale={src_w}:{src_h}:force_original_aspect_ratio=decrease,'
                f'pad={src_w}:{src_h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={src_fps}')
        filter_parts = [f'[{i}:v]{norm}[n{i}]' for i in range(n_total)]
        use_matte = transition == 'custom_matte' and transition_matte_path and os.path.exists(transition_matte_path)
        matte_input_args = []

        if use_matte:
            # Custom transition: blend each cut using a user-uploaded matte's luma as
            # an opacity mask (maskedmerge) instead of one of ffmpeg's built-in xfade
            # wipe shapes. xfade computes its own overlap windows internally from a
            # single offset per cut; maskedmerge has no such concept, so each clip is
            # split by hand into a unique middle portion plus the tail/head slivers
            # (xfade_dur long) that feed the transition either side of it, and the
            # whole thing is reassembled with one concat filter at the end.
            matte_ext = os.path.splitext(transition_matte_path)[1].lower()
            is_image = matte_ext in {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
            matte_input_idx = n_total  # the matte is appended as one extra -i after all_inputs
            matte_input_args = (['-loop', '1', '-i', transition_matte_path] if is_image
                                 else ['-stream_loop', '-1', '-i', transition_matte_path])
            filter_parts.append(
                f'[{matte_input_idx}:v]{norm},format=gray,trim=0:{xfade_dur},setpts=PTS-STARTPTS[mattebase]')
            n_trans = n_total - 1
            filter_parts.append('[mattebase]split=' + str(n_trans) + ''.join(f'[mt{i}]' for i in range(n_trans)))

            seg_labels = []
            for i in range(n_total):
                start = xfade_dur if i > 0 else 0
                end = durations[i] - (xfade_dur if i < n_total - 1 else 0)
                if end <= start:
                    # Clip too short to give a full xfade_dur to both neighboring
                    # transitions — keep a thin sliver instead of an empty/negative one.
                    end = start + 0.05
                filter_parts.append(f'[n{i}]trim=start={start}:end={end},setpts=PTS-STARTPTS[seg{i}]')
                seg_labels.append(f'seg{i}')
                if i < n_total - 1:
                    tail_start = max(0, durations[i] - xfade_dur)
                    filter_parts.append(f'[n{i}]trim=start={tail_start}:end={durations[i]},setpts=PTS-STARTPTS[tailA{i}]')
                    filter_parts.append(f'[n{i+1}]trim=start=0:end={xfade_dur},setpts=PTS-STARTPTS[headB{i}]')
                    filter_parts.append(f'[tailA{i}][headB{i}][mt{i}]maskedmerge[trans{i}]')
                    seg_labels.append(f'trans{i}')
                    offset = sum(durations[:i + 1]) - (i + 1) * xfade_dur
                    sfx_timestamps.append(max(offset, 0) + xfade_dur * 0.5)
            concat_inputs = ''.join(f'[{lbl}]' for lbl in seg_labels)
            filter_parts.append(f'{concat_inputs}concat=n={len(seg_labels)}:v=1:a=0[vout]')
            prev_label = 'vout'
        else:
            prev_label = 'n0'
            for i in range(n_total - 1):
                offset = sum(durations[:i + 1]) - (i + 1) * xfade_dur
                sfx_timestamps.append(max(offset, 0) + xfade_dur * 0.5)
                out_label = f'v{i+1}'
                filter_parts.append(
                    f'[{prev_label}][n{i+1}]xfade=transition={transition}:duration={xfade_dur}:offset={max(offset, 0)}[{out_label}]')
                prev_label = out_label

        # Audio acrossfade chain — matches video transition timing exactly either way
        audio_parts = []
        for i in range(n_total):
            audio_parts.append(f'[{i}:a]atrim=0:{durations[i]}[a{i}]')
        for i in range(1, n_total):
            prev = f'af{i-1}' if i > 1 else 'a0'
            audio_parts.append(f'[{prev}][a{i}]acrossfade=d={xfade_dur}:c1=tri[af{i}]')
        last_audio_label = f'af{n_total-1}'
        filter_parts.extend(audio_parts)

        cmd = [FFMPEG, '-y']
        for f in all_inputs:
            cmd.extend(['-i', f])
        cmd.extend(matte_input_args)
        cmd.extend(['-filter_complex', ';'.join(filter_parts)])
        last_vlabel = f'[{prev_label}]'
        cmd.extend(['-map', last_vlabel, '-map', f'[{last_audio_label}]'])
        cmd.extend(['-c:v', 'libx264', '-preset', 'fast', '-crf', '22', '-pix_fmt', 'yuv420p',
                     '-c:a', 'aac', '-b:a', '128k', out_path])
        try:
            r = run_ffmpeg(cmd, timeout=FFMPEG_LONG_TIMEOUT, label='xfade concat')
            if r.returncode != 0:
                print(f'FFMPEG xfade error: {r.stderr[:1000]}')
                last_ffmpeg_stderr = r.stderr[-800:]
        except MediaToolTimeout as e:
            last_ffmpeg_stderr = str(e)

    # Cleanup segment files
    for f in seg_files:
        if os.path.exists(f):
            os.remove(f)

    filename = os.path.basename(out_path)
    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        detail = ''
        if not seg_files:
            detail = ' No scene clips were successfully extracted from the source video — check that ffmpeg/ffprobe are installed and on PATH, and that the uploaded video isn\'t corrupt.'
        elif last_ffmpeg_stderr:
            detail = f' Last ffmpeg error: {last_ffmpeg_stderr.strip()}'
        job_set(jid, error=f'Episodic Promo Plug generation failed (ffmpeg output empty).{detail}')
        return
    # Verify it's a valid video, and capture the ACTUAL assembled duration.
    # This is scenes + cards - crossfade overlap, which is NOT the same as
    # total_sel (scenes only). Ducking windows and the reported trailer length
    # both used total_sel previously: the trailing duck window got clipped short,
    # so music never ducked under the end-card VO, and the duration shown to the
    # user understated the real file by the length of the cards.
    assembled_duration = probe_duration(out_path)
    if assembled_duration is None or assembled_duration <= 0:
        job_set(jid, error='Episodic Promo Plug generation failed (invalid/corrupt output).')
        return

    job_set(jid, percent=58, step='Normalizing audio levels')
    # Normalize the raw SOT (original dialogue/nat sound baked into the source clips)
    # to a consistent loudness up front. Source footage can come in recorded at very
    # different levels — this keeps everything downstream (SFX ducking, BGM
    # sidechain detection, VO ducking) working off a predictable baseline instead of
    # being thrown off by an unusually quiet or hot original recording.
    sot_norm = os.path.join(app.config['UPLOAD_FOLDER'], f'sotnorm_{base_ts}.mp4')
    r = subprocess.run([FFMPEG, '-y', '-i', out_path,
                        '-af', f'loudnorm=I={target_loudness}:TP={true_peak}:LRA=7',
                        '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', sot_norm],
                       capture_output=True, text=True, timeout=120)
    if os.path.exists(sot_norm) and os.path.getsize(sot_norm) > 0:
        os.replace(sot_norm, out_path)
    else:
        print(f'SOT normalization error: {r.stderr[:500]}')

    job_set(jid, percent=62, step='Adding sound effects at cuts')
    # Generate SFX and mix into trailer audio. Whatever the source (AI-generated,
    # uploaded, or synthesized), the *same* hit gets stamped at every cut via
    # stamp_hits() so it's never just a single one-shot lost at the front.
    sfx_ok = False
    sfx_source = 'none'  # 'woosh' | 'uploaded' | 'synth_fallback' | 'none'
    sfx_path = os.path.join(app.config['UPLOAD_FOLDER'], f'sfx_{base_ts}.wav')
    if sfx_mode != 'none' and sfx_timestamps:
        hit_wave = None
        if sfx_mode == 'upload' and sfx_upload_path:
            hit_wave = load_hit_waveform(sfx_upload_path)
            sfx_source = 'uploaded' if hit_wave is not None else 'none'
        elif sfx_mode == 'genre' and genre:
            woosh_sfx_path = os.path.join(app.config['UPLOAD_FOLDER'], f'woosh_sfx_{base_ts}.flac')
            if woosh_sfx(genre, woosh_sfx_path, duration=0.8) and os.path.getsize(woosh_sfx_path) > 0:
                hit_wave = load_hit_waveform(woosh_sfx_path)
                if os.path.exists(woosh_sfx_path):
                    os.remove(woosh_sfx_path)
                sfx_source = 'woosh' if hit_wave is not None else 'none'
            # No ACE-Step fallback here by design — ACE-Step is a music model, not an
            # SFX model, so a failed Woosh call goes straight to the procedural synth
            # fallback below rather than a mismatched-model attempt. The "From genre"
            # option is gated on Woosh alone (see data-requires on sfx_mode=genre) so
            # this branch is only reached when Woosh was expected to work.
            if hit_wave is None:
                hit_wave = synth_sfx_waveform(genre)
                sfx_source = 'synth_fallback' if hit_wave is not None else 'none'
        if hit_wave is not None:
            sfx_ok = stamp_hits(hit_wave, sfx_timestamps, sfx_path)
            if not sfx_ok:
                sfx_source = 'none'
        if sfx_ok:
            sfx_m4a = os.path.join(app.config['UPLOAD_FOLDER'], f'sfx_{base_ts}.m4a')
            sfx_cmd = [FFMPEG, '-y', '-i', sfx_path]
            if sfx_source == 'synth_fallback':
                # Light production polish so the procedurally-synthesized fallback
                # doesn't sound as bare/synthetic next to AI-generated SFX.
                sfx_cmd += ['-af', 'aecho=0.6:0.5:35:0.25,alimiter=limit=0.9']
            sfx_cmd += ['-c:a', 'aac', '-b:a', '192k', sfx_m4a]
            subprocess.run(sfx_cmd, capture_output=True, text=True, timeout=30)
            if os.path.exists(sfx_m4a) and os.path.getsize(sfx_m4a) > 0:
                # Mix SFX into the trailer audio
                with_sfx = os.path.join(app.config['UPLOAD_FOLDER'], f'with_sfx_{base_ts}.mp4')
                r = subprocess.run([FFMPEG, '-y', '-i', out_path, '-i', sfx_m4a,
                                    '-filter_complex',
                                    '[0:a]volume=1.0[a0];[1:a]volume=0.85[a1];'
                                    '[a0][a1]amix=inputs=2:duration=first:dropout_transition=2:normalize=0[outa]',
                                    '-map', '0:v', '-map', '[outa]',
                                    '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k',
                                    '-shortest', with_sfx],
                                   capture_output=True, text=True, timeout=120)
                if os.path.exists(with_sfx) and os.path.getsize(with_sfx) > 0:
                    os.replace(with_sfx, out_path)
            if os.path.exists(sfx_path):
                os.remove(sfx_path)
            if os.path.exists(sfx_m4a):
                os.remove(sfx_m4a)
    if sfx_upload_path and os.path.exists(sfx_upload_path):
        os.remove(sfx_upload_path)
    if title_card_vo_path and os.path.exists(title_card_vo_path):
        os.remove(title_card_vo_path)
    if end_card_vo_path and os.path.exists(end_card_vo_path):
        os.remove(end_card_vo_path)

    job_set(jid, percent=80, step='Generating/mixing background music')
    # Prepare background music (ducked under SOT) as its own stem — the actual
    # merge into out_path happens later in the unified final mix below, once we
    # also know whether a voiceover is being added.
    bgm_source = 'none'  # 'uploaded' | 'ai_generated' | 'synth_fallback' | 'none'
    bgm_ready_path = None
    if scoring_audio_path and total_sel > 0:
        # Fit the music to the WHOLE assembled trailer, not just the scene
        # segments. This used to be total_sel (scenes only), so the music ran out
        # the moment the title/end cards started and they played dry -- exactly
        # where a promo most wants a bed under the card VO.
        scenes_dur = assembled_duration
        prepared_bgm = None

        if early_bgm_path and os.path.exists(early_bgm_path):
            # Reuse the track generated for beat-sync instead of paying for a
            # second ACE-Step/generation pass; just re-fit it to the exact length.
            prepared_bgm = finalize_bgm_duration(early_bgm_path, scenes_dur, base_ts)
            bgm_source = early_bgm_source if prepared_bgm else 'none'
            if os.path.exists(early_bgm_path):
                os.remove(early_bgm_path)
        else:
            prepared_bgm, bgm_source = prepare_bgm_track(genre, scoring_mode, scoring_audio_path, scenes_dur, base_ts)

        if prepared_bgm and beat_match:
            job_set(jid, step='Beat-matching music to video tempo')
            matched = os.path.join(app.config['UPLOAD_FOLDER'], f'matched_{base_ts}.wav')
            if beat_match_audio(path, prepared_bgm, scenes_dur, matched):
                prepared_bgm_m4a = os.path.join(app.config['UPLOAD_FOLDER'], f'matched_{base_ts}.m4a')
                subprocess.run([FFMPEG, '-y', '-i', matched,
                                '-c:a', 'aac', '-b:a', '192k', prepared_bgm_m4a],
                               capture_output=True, text=True, timeout=30)
                if os.path.exists(prepared_bgm_m4a) and os.path.getsize(prepared_bgm_m4a) > 0:
                    prepared_bgm = prepared_bgm_m4a
                if os.path.exists(matched):
                    os.remove(matched)

        if prepared_bgm:
            # Normalize the BGM track's own loudness only, to its baseline "full swell"
            # level (music_duck_db under overall target — so it never competes with SOT
            # even at full volume). The actual ducking during dialogue is applied later,
            # in the final mix, using deterministic silence-detected windows (see
            # _build_duck_volume_expr below) rather than a reactive sidechain here — that
            # lets a single minimum-gap "hold" rule govern both BGM-vs-SOT and BGM-vs-VO
            # instead of two separately-tuned compressors that could disagree with each other.
            bgm_target = target_loudness + music_duck_db
            bgm_ready_path = os.path.join(app.config['UPLOAD_FOLDER'], f'bgmready_{base_ts}.m4a')
            r = subprocess.run([FFMPEG, '-y', '-i', prepared_bgm,
                                '-af', f'loudnorm=I={bgm_target}:TP={true_peak}:LRA=7',
                                '-c:a', 'aac', '-b:a', '192k', bgm_ready_path],
                               capture_output=True, text=True, timeout=120)
            if not (os.path.exists(bgm_ready_path) and os.path.getsize(bgm_ready_path) > 0):
                print(f'BGM prep error: {r.stderr[:500]}')
                bgm_ready_path = None
            if os.path.exists(prepared_bgm):
                os.remove(prepared_bgm)

    # Voiceover: upload or TTS. Prepared as its own stem here too — mixed in below,
    # in the same unified final mix, so it can duck SOT and BGM by different amounts.
    vo_source = 'none'  # 'uploaded' | 'tts' | 'none'
    vo_error = None
    vo_ready_path = None
    if vo_mode != 'none':
        if vo_mode == 'tts':
            engine_label = 'Fish Audio'
            if vo_ref_upload_path:
                vo_step_note = f' (cloning voice from uploaded reference sample via {engine_label})'
            elif vo_voice:
                vo_step_note = f' (voice: {vo_voice}, {engine_label})'
            else:
                vo_step_note = f' ({engine_label}, no voice selected)'
        else:
            vo_step_note = ''
        job_set(jid, percent=90, step='Adding narration' + vo_step_note)
        vo_raw_path = None
        if vo_mode == 'upload' and vo_upload_path and os.path.exists(vo_upload_path):
            vo_raw_path = vo_upload_path
            vo_source = 'uploaded'
        elif vo_mode == 'tts':
            tts_wav = os.path.join(app.config['UPLOAD_FOLDER'], f'tts_{base_ts}.wav')
            tts_kwargs = {'rate': vo_rate, 'voice_id': vo_voice, 'language': vo_language, 'engine': vo_engine}
            if vo_ref_upload_path and os.path.exists(vo_ref_upload_path):
                tts_kwargs['reference_audio_path'] = vo_ref_upload_path
            ok, err = generate_tts(vo_text, tts_wav, **tts_kwargs)
            if ok:
                vo_raw_path = tts_wav
                vo_source = 'tts'
            else:
                vo_error = err
                print(f'TTS error: {err}')

        if vo_raw_path:
            # Normalize the VO's own loudness first — an uploaded voiceover can be
            # recorded much quieter or hotter than expected, which would otherwise
            # throw off both how audible it ends up and how reliably it triggers
            # the ducking below. If this is an uploaded VO, trim it to
            # [vo_trim_start, vo_trim_end) first — that's a window within the
            # source file itself, separate from vo_start below (which places the
            # already-trimmed narration on the trailer's own timeline).
            ms = max(0, int(vo_start * 1000))
            vo_ready_path = os.path.join(app.config['UPLOAD_FOLDER'], f'voready_{base_ts}.m4a')
            cmd = [FFMPEG, '-y']
            if vo_source == 'uploaded' and (vo_trim_start > 0 or vo_trim_end is not None):
                cmd.extend(['-ss', str(vo_trim_start)])
                if vo_trim_end is not None:
                    cmd.extend(['-to', str(vo_trim_end)])
            cmd.extend(['-i', vo_raw_path,
                        '-af', f'loudnorm=I={target_loudness}:TP={true_peak}:LRA=7,adelay={ms}|{ms},volume={vo_volume}',
                        '-c:a', 'aac', '-b:a', '192k', vo_ready_path])
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if not (os.path.exists(vo_ready_path) and os.path.getsize(vo_ready_path) > 0):
                print(f'VO prep error: {r.stderr[:500]}')
                vo_ready_path = None
        if vo_upload_path and os.path.exists(vo_upload_path):
            os.remove(vo_upload_path)
        tts_wav = os.path.join(app.config['UPLOAD_FOLDER'], f'tts_{base_ts}.wav')
        if os.path.exists(tts_wav):
            os.remove(tts_wav)

    # Unified final mix: combine SOT (already in out_path) with whichever of
    # BGM/VO are present. If VO is playing, SOT gets ducked to near-silence (two
    # people talking over each other is unusable) via a live sidechain, since VO
    # presence is unambiguous there. BGM's duck is different: it needs to stay
    # ducked whenever EITHER SOT or VO has anything going on, and — per request —
    # must not swell back up on every brief pause, only after a real gap. A
    # reactive sidechain has no concept of a minimum "hold" before releasing (only
    # attack/release ramp times), so BGM ducking is computed deterministically
    # instead: silence-detect SOT and VO separately, union the two "not silent"
    # timelines, bridge any gap shorter than duck_release_hold seconds (so a short
    # dialogue pause doesn't count as a real release), then apply duck_depth_db
    # only within those merged windows via a per-frame volume expression.
    #
    # Broadcast delivery toggle: if enabled, every element is forced to be
    # identically audible in both channels (true center / dual-mono), so nothing
    # is lost if played back or checked on a single channel. This is applied as
    # the very last step, after all ducking/mixing — it only changes the stereo
    # image (L/R placement), never the levels or ducking decided above it.
    CENTER_FILTER = 'aformat=channel_layouts=stereo,pan=stereo|c0=0.5*c0+0.5*c1|c1=0.5*c0+0.5*c1'
    if bgm_ready_path or vo_ready_path or broadcast_stereo:
        job_set(jid, step='Finalizing audio mix')
        final_mixed = os.path.join(app.config['UPLOAD_FOLDER'], f'finalmix_{base_ts}.mp4')
        cmd = [FFMPEG, '-y', '-i', out_path]
        inputs = []
        if bgm_ready_path:
            cmd += ['-i', bgm_ready_path]
            inputs.append('bgm')
        if vo_ready_path:
            cmd += ['-i', vo_ready_path]
            inputs.append('vo')

        tail = (CENTER_FILTER + ',' if broadcast_stereo else '') + f'loudnorm=I={target_loudness}:TP={true_peak}:LRA=7'

        bgm_duck_expr = None
        if bgm_ready_path:
            job_set(jid, step='Detecting dialogue gaps for music ducking')
            sot_silence = _detect_silence_intervals(out_path)
            sot_windows = _active_windows_from_silence(sot_silence, assembled_duration)
            vo_windows = []
            if vo_ready_path:
                vo_silence = _detect_silence_intervals(vo_ready_path)
                # Bound to the VO clip's own real length -- see
                # _active_windows_from_silence's docstring: without this, a
                # short VO with no detected internal silence reads as "active"
                # for the rest of the trailer, well past where it actually ends.
                vo_real_dur = probe_duration(vo_ready_path) or assembled_duration
                vo_windows = _active_windows_from_silence(vo_silence, assembled_duration,
                                                           content_duration=vo_real_dur)
            combined = _union_windows([sot_windows, vo_windows])
            duck_windows = _merge_windows_with_hold(combined, duck_release_hold)
            bgm_duck_expr = _build_duck_volume_expr(duck_windows, duck_depth_db)
            if bgm_duck_expr is None:
                # Silence detection itself failed on every track (not just "no gaps
                # found") — fall back to a constant duck for the full duration rather
                # than accidentally leaving BGM unducked throughout.
                bgm_duck_expr = f'{10 ** (duck_depth_db / 20):.5f}'

        if vo_ready_path:
            vo_idx = 1 + inputs.index('vo')
            fc = []
            # sidechaincompress truncates its ENTIRE output to the shorter of its
            # two inputs, unconditionally -- confirmed directly against ffmpeg: a
            # 20s main track fed a 5s key track produces a 5s result even with
            # amix's duration=first downstream. That's the "trailer length gets
            # cut to match the VO" bug: any VO shorter than the assembled trailer
            # silently truncated the whole audio mix (and therefore, via -shortest
            # on the final mux, the whole video) to the VO's own length.
            # apad extends the key input with silence to the real trailer length
            # before it ever reaches sidechaincompress, so ducking still applies
            # correctly while the VO plays and simply has nothing left to duck
            # once the VO ends.
            fc.append(f'[{vo_idx}:a]asplit=2[vo_out][vokey_raw]')
            fc.append(f'[vokey_raw]apad=whole_dur={assembled_duration:.3f}[vokey1]')
            # SOT ducked to near-silence under VO (fast attack, low threshold, high
            # ratio, no makeup — this is meant to sit well below the VO, not just
            # lower than before). This one stays a live sidechain since "is VO
            # playing right now" is unambiguous and doesn't need a hold.
            fc.append('[0:a][vokey1]sidechaincompress=threshold=0.01:ratio=20:attack=5:release=250:makeup=1[sot_ducked]')
            mix_labels = ['sot_ducked']
            if bgm_ready_path:
                bgm_idx = 1 + inputs.index('bgm')
                fc.append(f"[{bgm_idx}:a]volume=eval=frame:volume='{bgm_duck_expr}'[bgm_ducked]")
                mix_labels.append('bgm_ducked')
            mix_labels.append('vo_out')
            fc.append('[' + ']['.join(mix_labels) + f']amix=inputs={len(mix_labels)}:duration=first:dropout_transition=2:normalize=0[premix]')
            fc.append(f'[premix]{tail}[outa]')
            filter_complex = ';'.join(fc)
        elif bgm_ready_path:
            # BGM only, no VO — same hold-based duck against SOT, applied directly here
            # since there's no separate VO-driven filter chain to fold it into.
            bgm_idx = 1
            filter_complex = (f"[{bgm_idx}:a]volume=eval=frame:volume='{bgm_duck_expr}'[bgm_ducked];"
                               f'[0:a][bgm_ducked]amix=inputs=2:duration=first:dropout_transition=2:normalize=0[premix];'
                               f'[premix]{tail}[outa]')
        else:
            # Neither BGM nor VO — only reached when the broadcast toggle needs
            # to center the SOT on its own.
            filter_complex = f'[0:a]{tail}[outa]'

        cmd += ['-filter_complex', filter_complex, '-map', '0:v', '-map', '[outa]',
                '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', '-shortest', final_mixed]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if os.path.exists(final_mixed) and os.path.getsize(final_mixed) > 0:
            os.replace(final_mixed, out_path)
        else:
            print(f'Final audio mix error: {r.stderr[:500]}')
        if bgm_ready_path and os.path.exists(bgm_ready_path):
            os.remove(bgm_ready_path)
        if vo_ready_path and os.path.exists(vo_ready_path):
            os.remove(vo_ready_path)

    result = dict(
        status='ok', trailer_url=f'/uploads/{filename}',
        orig_name=orig_name,
        total_scenes=len(scene_list), selected_scenes=len(selected),
        trailer_duration=round(assembled_duration, 1),
        scenes_duration=round(total_sel, 1),
        video_duration=round(video_duration, 1),
        trailer_length=trailer_length,
        bgm_source=bgm_source, sfx_source=sfx_source,
        vo_source=vo_source, vo_error=vo_error, sync_beats=sync_beats,
        whisper_enhance=whisper_enhance,
        template_applied=params.get('template_applied'),
        scenes=[{
            'scene': i+1, 'start': round(s['start'], 1), 'end': round(s['end'], 1),
            'quality': s['total_score'], 'duration': round(s['selected_dur'], 1),
            'description': _scene_desc(s)
        } for i, s in enumerate(selected)])
    job_set(jid, percent=100, step='Done', done=True, result=result)
    try:
        result['library_id'] = library_add(filename, result,
                                            user_id=params.get('user_id'), username=params.get('username'))
    except Exception as e:
        print(f'Trailer library save failed (job still succeeded): {e}')


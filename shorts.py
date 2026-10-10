"""Vertical Shorts tab: cuts a long-form programme into stand-alone 9:16 shorts.

A separate section from the promo generator on purpose -- it shares PRISM's
services (scene detection, faster-whisper, Ollama, ffmpeg, the job queue,
network shares and destinations) but none of the promo pipeline's own logic,
so nothing here can change how a plug is built.

Two jobs, with a person in between:

  1. Analyse  -- detect cuts, rate sampled frames with the vision model,
                 transcribe, have the story model propose stand-alone
                 moments, and hand back ranked candidates. Nothing is
                 rendered yet.
  2. Render   -- for the candidates the editor kept (and re-timed, retitled
                 or added by hand), reframe to 1080x1920, burn captions in,
                 and save the batch.

The decisions themselves -- what makes a candidate, where it starts and
ends, how a shot is reframed -- live in shorts_core.py, which has no Flask or
PRISM imports and can be driven by a watch-folder script just as well. This
module is the wiring: HTTP in, services called, files stored, progress
reported.

Service URLs are always read as pipeline.OLLAMA_URL / pipeline.WHISPER_URL at
call time, never imported by name: they are reassigned live from the Config
tab, and a by-name import would freeze whatever value was set at startup.
"""
import base64
import hashlib
import json
import math
import os
import re
import secrets
import shutil
import tempfile
import threading
import time
import traceback
import functools
import zipfile
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import requests
from flask import request, jsonify, session, send_from_directory
from werkzeug.utils import secure_filename

from core import app, _job_submit_limiter, _client_ip, _RateLimiter
from library_db import LIBRARY_DIR, audit_log, network_destination_get
from auth import require_permission
import pipeline
import shorts_core as sc
import shorts_projects as sp


def _env_num(name, default, cast=float):
    try:
        return cast(os.environ.get(name, default))
    except (TypeError, ValueError):
        return cast(default)


# Finished shorts live next to the trailer library (and its branding/
# subfolder), not in UPLOAD_FOLDER: that one is a fresh temp dir every start
# and is swept by age, so anything left there is gone after a restart.
SHORTS_DIR = os.path.abspath(os.environ.get('SHORTS_DIR') or os.path.join(LIBRARY_DIR, 'shorts'))
# Projects (see shorts_projects.py): the episode a set of shorts was cut from.
# Every batch is filed under one. Kept beside the batches, not among them.
SHORTS_PROJECTS_DIR = os.path.abspath(os.environ.get('SHORTS_PROJECTS_DIR')
                                      or os.path.join(LIBRARY_DIR, 'shorts_projects'))
# Analyses (the moments found in an episode, and the editor's review of
# them) are kept on disk, so a restart or a long lunch does not mean
# analysing a 45-minute episode again. Kept this many days after the last
# time anyone opened them.
SHORTS_ANALYSES_DIR = os.path.abspath(os.environ.get('SHORTS_ANALYSES_DIR')
                                      or os.path.join(LIBRARY_DIR, 'shorts_analyses'))

SHORTS_VISION_FRAMES = _env_num('SHORTS_VISION_FRAMES', 90, int)       # vision calls per analysis, whatever the source length
SHORTS_STORY_CHUNK_SEC = _env_num('SHORTS_STORY_CHUNK_SEC', 300)       # transcript handed to the story model per call
SHORTS_STORY_OVERLAP_SEC = _env_num('SHORTS_STORY_OVERLAP_SEC', 60)
SHORTS_STORY_NUM_CTX = _env_num('SHORTS_STORY_NUM_CTX', 8192, int)
SHORTS_CRF = _env_num('SHORTS_CRF', 18, int)
SHORTS_PRESET = os.environ.get('SHORTS_PRESET', 'medium')
# Its own loudness target rather than Config > Production's: that one is set
# for where promos air (often -23/-24 for broadcast), and a short mastered
# to a broadcast target plays noticeably quiet next to everything else in a
# social feed, where -14 LUFS is the norm.
SHORTS_LOUDNESS = _env_num('SHORTS_LOUDNESS', -14.0)
# How long the cliffhanger ending holds on its last picture before the cut to
# black, in seconds: the form's starting value, which an editor can change per render.
SHORTS_FADE = sc.clamp_fade(_env_num('SHORTS_FADE', 1.0)) or 1.0      # the fade-to-black ending's length, seconds
# The language the dialogue is in, when it is nearly always the same one
# (a station's own programmes): the form starts on it. Empty: Whisper decides.
SHORTS_LANGUAGE = pipeline.normalize_language(os.environ.get('SHORTS_LANGUAGE'))
SHORTS_CLIFF_HOLD = sc.clamp_hold(_env_num('SHORTS_CLIFF_HOLD', sc.CLIFFHANGER['hold'])) or sc.CLIFFHANGER['hold']
SHORTS_TRUE_PEAK = _env_num('SHORTS_TRUE_PEAK', -1.5)
SHORTS_SUB_FONT = os.environ.get('SHORTS_SUB_FONT', 'Arial')
SHORTS_FACE_MODEL = os.environ.get('SHORTS_FACE_MODEL', '')
# Whether "Follow the speaker" starts ticked in the tab. It is a per-render
# choice either way; this only sets where the checkbox begins. Off unless
# asked for: it is a judgement from mouth movement, and when it is wrong it
# crops out the person talking, which showing the whole frame never does.
SHORTS_SPEAKER_CROP = os.environ.get('SHORTS_SPEAKER_CROP', '').strip().lower() in ('1', 'true', 'yes', 'on')

SHORTS_MAX_ITEMS = 100     # moments per analysis, and shorts per render job
SHORTS_ANALYSIS_DAYS = _env_num('SHORTS_ANALYSIS_DAYS', 14)
# "Auto" in Moments to find: every distinct moment the story model scored at
# least this, out of 10 -- as many as are worth making, not a number picked
# in advance.
SHORTS_AUTO_MIN_STORY = _env_num('SHORTS_AUTO_MIN_STORY', 6, int)
# Once the moments are known, a few frames inside each are rated too (the
# whole-episode sample lands in some moments and misses others): this many
# in all, at most SHORTS_MOMENT_FRAMES_EACH per moment.
SHORTS_MOMENT_FRAMES = _env_num('SHORTS_MOMENT_FRAMES', 160, int)
SHORTS_MOMENT_FRAMES_EACH = _env_num('SHORTS_MOMENT_FRAMES_EACH', 4, int)
# Delivery formats: the ones the rest of PRISM exports, minus AVC-Intra 100,
# which is defined for a 1920x1080 picture -- a 1080x1920 file labelled as it
# is not something the equipment that asks for AVC-Intra will take.
SHORTS_FORMATS = [k for k in pipeline.EXPORT_FORMATS if k != 'avci100i']
# A short delivered as anything but MP4 is rendered to H.264 first (that copy
# is what plays in the browser) and the delivery file is made from it, by
# the same export step every other tab uses. That first render is therefore
# done at this quality or better, so the delivery file is not a copy of an
# ordinary web-quality encode.
SHORTS_MASTER_CRF = _env_num('SHORTS_MASTER_CRF', 12, int)
SHORTS_MIN_CLIP = 3.0      # seconds
SHORTS_MAX_CLIP = 300.0

ANALYZE_STAGES = [(2, 'Reading video'), (5, 'Detecting cuts'), (20, 'Rating frames'),
                  (46, 'Transcribing dialogue'), (54, 'Reading the sound'), (60, 'Finding story beats'),
                  (88, 'Building candidates'), (89, 'Rating frames inside the moments'), (100, 'Done')]
RENDER_STAGES = [(2, 'Preparing'), (5, 'Rendering shorts'), (100, 'Done')]
RECAPTION_STAGES = [(5, 'Preparing'), (10, 'Rendering with the new captions'), (100, 'Done')]
# "Generate without preview" is the two jobs above run back to back as one:
# the analysis fills the first AUTO_SPLIT percent of the bar and the render
# the rest. Its stage list is derived from theirs so the three can't drift.
AUTO_SPLIT = 55
AUTO_STAGES = ([(max(1, p * AUTO_SPLIT // 100), lbl) for p, lbl in ANALYZE_STAGES[:-1]]
               + [(AUTO_SPLIT + p * (100 - AUTO_SPLIT) // 100, lbl) for p, lbl in RENDER_STAGES[1:]])
STAGES_BY_KIND = {'analyze': ANALYZE_STAGES, 'render': RENDER_STAGES, 'auto': AUTO_STAGES,
                  'recaption': RECAPTION_STAGES}

# ---- Analyses awaiting review ----
# An analysis holds the whole transcript, the cut list and the moments
# found, which the render needs and the browser has no use for, and the
# editor's review of those moments as it stands (see api_shorts_review).
# Each is a folder in SHORTS_ANALYSES_DIR -- analysis.json and the moments'
# stills -- so it outlives a restart; the recently used ones are also held
# in memory. Keys starting with "_" are working state (a cache of framing
# plans) and are never written.
ANALYSES = {}
ANALYSES_LOCK = threading.Lock()
_ANALYSIS_FILE_LOCK = threading.Lock()
_JOB_KINDS = {}
_AID = re.compile(r'^[0-9a-f]{16}$')
_IN_MEMORY = 24


def _analysis_ttl():
    return float(SHORTS_ANALYSIS_DAYS) * 86400.0


def _analysis_dir(aid):
    """The folder for an analysis id, or None if it is not one this module
    made (the pattern check is the path-traversal guard)."""
    if not aid or not _AID.match(str(aid)):
        return None
    return os.path.join(SHORTS_ANALYSES_DIR, str(aid))


def _save_analysis(aid, data):
    """Writes analysis.json for `aid`, replacing it whole."""
    d = _analysis_dir(aid)
    if not d:
        return
    os.makedirs(d, exist_ok=True)
    body = {k: v for k, v in data.items() if not str(k).startswith('_')}
    tmp = os.path.join(d, f'analysis.json.{secrets.token_hex(3)}.tmp')
    with _ANALYSIS_FILE_LOCK:
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(body, f, ensure_ascii=False, separators=(',', ':'), default=float)
        for attempt in range(6):
            try:
                os.replace(tmp, os.path.join(d, 'analysis.json'))
                return
            except PermissionError:
                if attempt == 5:
                    raise
                time.sleep(0.25)


def _load_analysis(aid):
    d = _analysis_dir(aid)
    if not d:
        return None
    try:
        with open(os.path.join(d, 'analysis.json'), encoding='utf-8') as f:
            a = json.load(f)
    except (OSError, ValueError):
        return None
    return a if isinstance(a, dict) else None


def _expired(a, now=None):
    return (now or time.time()) - float(a.get('opened') or a.get('created') or 0) > _analysis_ttl()


def _sweep_analyses():
    """Removes analyses no one has opened for SHORTS_ANALYSIS_DAYS."""
    try:
        names = os.listdir(SHORTS_ANALYSES_DIR)
    except OSError:
        return
    now = time.time()
    for name in names:
        if not _AID.match(name):
            continue
        try:
            age = now - os.path.getmtime(os.path.join(SHORTS_ANALYSES_DIR, name, 'analysis.json'))
        except OSError:
            age = None
        if age is None or age > _analysis_ttl():
            a = _load_analysis(name)
            if a is None or _expired(a, now):
                shutil.rmtree(os.path.join(SHORTS_ANALYSES_DIR, name), ignore_errors=True)
                with ANALYSES_LOCK:
                    ANALYSES.pop(name, None)


def _remember(aid, data):
    with ANALYSES_LOCK:
        ANALYSES.pop(aid, None)
        ANALYSES[aid] = data
        while len(ANALYSES) > _IN_MEMORY:
            ANALYSES.pop(next(iter(ANALYSES)))


def analysis_store(aid, data):
    now = time.time()
    data['created'] = now
    data['opened'] = now
    _save_analysis(aid, data)
    _remember(aid, data)
    _sweep_analyses()


def analysis_save(aid, data):
    """Writes an analysis that has changed (its review, a source fetched
    again) back to disk."""
    _save_analysis(aid, data)


def analysis_get(aid):
    """The analysis, from memory or from disk, or None if there is none (or
    it has not been opened for SHORTS_ANALYSIS_DAYS)."""
    with ANALYSES_LOCK:
        a = ANALYSES.get(aid)
    if a is None:
        a = _load_analysis(aid)
        if a is None:
            return None
        _remember(aid, a)
    if _expired(a):
        with ANALYSES_LOCK:
            ANALYSES.pop(aid, None)
        d = _analysis_dir(aid)
        if d:
            shutil.rmtree(d, ignore_errors=True)
        return None
    # Opened: its days start again. Written at most once an hour for that alone.
    now = time.time()
    if now - float(a.get('opened') or 0) > 3600:
        a['opened'] = now
        try:
            _save_analysis(aid, a)
        except OSError:
            pass
    return a


def analysis_summaries(project_id=None):
    """Every analysis kept, newest first, as the list of earlier cuts shows
    them -- read from disk without holding them all in memory."""
    out = []
    try:
        names = os.listdir(SHORTS_ANALYSES_DIR)
    except OSError:
        return out
    now = time.time()
    for name in names:
        if not _AID.match(name):
            continue
        with ANALYSES_LOCK:
            a = ANALYSES.get(name)
        a = a or _load_analysis(name)
        if not a or _expired(a, now) or (project_id and a.get('project_id') != project_id):
            continue
        review = a.get('review') or []
        out.append({'analysis_id': name, 'orig_name': a.get('orig_name'), 'created': a.get('created'),
                    'opened': a.get('opened'), 'user_id': a.get('user_id'), 'username': a.get('username'),
                    'project_id': a.get('project_id'), 'candidates': len(a.get('candidates') or []),
                    'reviewed': bool(review), 'kept': sum(1 for c in review if c.get('keep'))
                    if review else len(a.get('candidates') or []),
                    'source_available': _sources_available(a), 'parts': len(_parts(a)),
                    'refetchable': all(P.get('origin') for P in _parts(a))})
    out.sort(key=lambda x: x['created'] or 0, reverse=True)
    return out


# ---- Parts ----
# An analysis is of one video, or of several parts of one episode laid end
# to end. A single video is stored the way it always was (path, info, cut
# list, transcript at the top level); several are stored with a `parts` list
# and the top level speaks for the whole episode (its info.duration is the
# sum). Everything the editor sees and sends -- moments, captions, framing --
# is on the joined timeline; each part's own time is that minus its offset.

def _parts(a):
    """The parts of an analysis, a one-file source being a single part."""
    if a.get('parts'):
        return a['parts']
    return [{'path': a.get('path'), 'name': a.get('orig_name'), 'info': a['info'], 'offset': 0.0,
             'duration': float(a['info']['duration']), 'cut_frames': a.get('cut_frames') or [],
             'words': a.get('words') or [], 'segments': a.get('segments') or [], 'origin': a.get('origin')}]


def _is_multi(a):
    return len(a.get('parts') or []) > 1


def _part_view(a, k):
    """The analysis as it is for part `k`: that part's file, frame rate,
    cuts and transcript, in the part's own time -- what everything that
    cuts, previews or plans a moment works on. A one-file analysis is its
    own view."""
    if not _is_multi(a):
        return a
    P = a['parts'][k]
    v = dict(a, path=P['path'], info=P['info'], cut_frames=P['cut_frames'], words=P['words'],
             segments=P['segments'], origin=P.get('origin'))
    v['_part'], v['_offset'] = k, P['offset']
    v['_plans'] = a.setdefault('_plans', {})
    return v


def _part_label(a, k):
    return f"Part {k + 1}" if _is_multi(a) else ''


def _locate(a, start, end):
    """(part index, error) for a moment from `start` to `end` on the
    analysis's timeline: the part it starts in, and an error if it runs
    across the break into the next."""
    parts = _parts(a)
    k = len(parts) - 1
    for i, P in enumerate(parts):
        if start < P['offset'] + P['duration'] - 1e-6:
            k = i
            break
    P = parts[k]
    if end > P['offset'] + P['duration'] + 0.05:
        return k, (f"This moment runs across the break between Part {k + 1} and Part {k + 2}. A short stays inside "
                   f"one part: end it before {fmt_clock(P['duration'])} in Part {k + 1}, or start it in Part {k + 2}.")
    return k, None


def fmt_clock(sec):
    sec = max(0.0, float(sec))
    h, m = int(sec // 3600), int(sec % 3600 // 60)
    return f'{h}:{m:02d}:{sec % 60:04.1f}' if h else f'{m}:{sec % 60:04.1f}'


def _localize_item(it, off):
    """A moment as the request made it (joined timeline) in its part's own
    time: its range, the range its edited captions were made for, and where
    its framing corrections are."""
    if not off:
        return it
    out = dict(it, start=max(0.0, it['start'] - off), end=max(0.0, it['end'] - off))
    if it.get('captions'):
        out['captions'] = dict(it['captions'], start=max(0.0, it['captions']['start'] - off),
                               end=max(0.0, it['captions']['end'] - off))
    if (it.get('framing') or {}).get('shots'):
        out['framing'] = {'shots': [dict(s_, at=max(0.0, s_['at'] - off)) for s_ in it['framing']['shots']]}
    return out


def _sources_available(a):
    return all(P.get('path') and os.path.exists(P['path']) for P in _parts(a))


def _ensure_source(a, aid=None, report=None):
    """The analysed episode -- every part of it -- on this server again, if
    it has gone: fetched from where it was first fetched from. (path of the
    first part, None) or (None, why)."""
    parts = _parts(a)
    changed = False
    for k, P in enumerate(parts):
        if P.get('path') and os.path.exists(P['path']):
            continue
        what = 'The episode' if len(parts) == 1 else f'Part {k + 1} ({P.get("name")})'
        origin = P.get('origin')
        if not origin:
            return None, (f'{what} is no longer on the server (staged files are cleared after a while, and on a '
                          'restart) and it was not taken from a network folder, so it cannot be fetched again. '
                          'Pick it again and re-run the analysis.')
        if report:
            report(step='Fetching the episode from the network folder again' if len(parts) == 1
                   else f'Fetching part {k + 1} from the network folder again')
        try:
            staged = pipeline.stage_network_file(origin['name'], origin.get('category') or 'shorts',
                                                 origin.get('subpath') or '')
        except Exception as e:
            return None, f"{what} could not be fetched again from the network folder ({e})."
        P['path'] = staged['path']
        changed = True
    if changed:
        a['path'] = parts[0]['path']
        if aid:
            try:
                analysis_save(aid, a)
            except OSError:
                pass
    return parts[0]['path'], None


def _touch(path):
    """Resets a staged source's age so the upload sweeper (which reclaims
    by mtime) doesn't delete it out from under a review that is still in
    progress. Best-effort. Only a staged copy: a file read where it lives
    on the network share is never ours to touch."""
    try:
        if os.path.dirname(os.path.abspath(path)) != os.path.abspath(app.config['UPLOAD_FOLDER']):
            return
        os.utime(path, None)
    except OSError:
        pass


# The YuNet face model, as published in the OpenCV model zoo (MIT licence).
# What is downloaded is checked against this before it is used.
YUNET_URL = ('https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/'
             'face_detection_yunet_2023mar.onnx')
YUNET_SHA256 = '8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4'
YUNET_SIZE = 232589
YUNET_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'face_detection_yunet_2023mar.onnx')


def _face_model_path():
    """YuNet model file if one is around: SHORTS_FACE_MODEL, then whatever
    the promo pipeline found, then a models/ folder next to the app.
    None means the built-in Haar cascades are used instead."""
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (SHORTS_FACE_MODEL, pipeline.ONNX_PATH,
              os.path.join(here, 'models', 'face_detection_yunet_2023mar.onnx'),
              os.path.join(here, 'models', 'face_detection_yunet.onnx')):
        if p and os.path.exists(p):
            return p
    return None


_ASS_FILTER = None


def _captions_available():
    global _ASS_FILTER
    if _ASS_FILTER is None:
        _ASS_FILTER = sc.has_filter(pipeline.FFMPEG, 'ass')
    return _ASS_FILTER


# --------------------------------------------------------------------------
# Batches on disk: SHORTS_DIR/<batch_id>/{batch.json, *.mp4, *.srt, *.jpg}
# --------------------------------------------------------------------------

_BATCH_ID = re.compile(r'^\d{10,}_[0-9a-f]{6}$')


def _batch_dir(bid):
    """Folder for a batch id, or None if the id isn't one this module made.
    The pattern check is the path-traversal guard: an id is only ever
    digits, an underscore and hex."""
    if not bid or not _BATCH_ID.match(str(bid)):
        return None
    return os.path.join(SHORTS_DIR, str(bid))


# Every read and write of a batch.json goes through this one lock. The app
# runs as a single process, so that is enough to guarantee the file is never
# being read while it is replaced -- which matters on Windows, where
# replacing a file another thread has open fails outright, and here would
# kill a render over a thumbnail request that happened to land mid-write.
_MANIFEST_LOCK = threading.Lock()


def _write_manifest(bdir, manifest):
    tmp = os.path.join(bdir, 'batch.json.tmp')
    dst = os.path.join(bdir, 'batch.json')
    with _MANIFEST_LOCK:
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, ensure_ascii=False, indent=1)
        for attempt in range(6):
            try:
                os.replace(tmp, dst)
                return
            except PermissionError:
                # Something outside this process (antivirus, a backup agent,
                # Explorer's preview) has the file open for a moment.
                if attempt == 5:
                    raise
                time.sleep(0.25)


def _read_manifest(bdir):
    with _MANIFEST_LOCK:
        try:
            with open(os.path.join(bdir, 'batch.json'), encoding='utf-8') as f:
                m = json.load(f)
        except (OSError, ValueError):
            return None
    return m if isinstance(m, dict) else None


def _load_manifest(bid):
    bdir = _batch_dir(bid)
    return _read_manifest(bdir) if bdir else None


def _batch_public(m):
    bid = m.get('batch_id')
    shorts = []
    # Always shown in episode order. `number` is what to call each one: the
    # number in its filename when that is already its place in the episode,
    # its place in this list for a batch made when files were numbered by
    # rank instead (there the two do not agree, and the order is what helps).
    by_episode = m.get('numbering') == 'episode'
    in_order = sorted(m.get('shorts') or [], key=lambda s: (float(s.get('start') or 0), s.get('index') or 0))
    for at, s in enumerate(in_order, 1):
        d = {k: s.get(k) for k in ('index', 'title', 'file', 'srt', 'start', 'end', 'duration', 'size',
                                   'layouts', 'captions')}
        d['number'] = s.get('index') if by_episode and s.get('index') else at
        # A short whose captions were changed keeps its names; `rev` in the
        # address is what stops a browser playing the copy it cached before.
        v = f"?v={int(s['rev'])}" if s.get('rev') else ''
        d['url'] = f"/api/shorts/file/{bid}/{s['file']}{v}"
        d['thumb_url'] = f"/api/shorts/file/{bid}/{s['thumb']}{v}" if s.get('thumb') else None
        d['srt_url'] = f"/api/shorts/file/{bid}/{s['srt']}{v}" if s.get('srt') else None
        # The file that is handed over. For MP4 it is the one that plays; for
        # any other format it is a second file beside it.
        d['delivery'] = s.get('delivery') or s['file']
        d['delivery_url'] = f"/api/shorts/file/{bid}/{d['delivery']}{v}"
        d['delivery_size'] = s.get('delivery_size') or s.get('size')
        d['caption_lines'] = len(s['cues']) if isinstance(s.get('cues'), list) else None
        d['captions_edited'] = bool(s.get('captions_edited'))
        for k in ('part', 'part_name', 'part_start', 'part_end'):
            if s.get(k) is not None:
                d[k] = s[k]
        shorts.append(d)
    proj = sp.load(SHORTS_PROJECTS_DIR, m.get('project_id')) if m.get('project_id') else None
    return {'batch_id': bid, 'orig_name': m.get('orig_name'), 'created': m.get('created'),
            'source_duration': m.get('source_duration'), 'parts': m.get('parts'),
            'username': m.get('username'), 'status': m.get('status'), 'options': m.get('options') or {},
            'shorts': shorts, 'errors': m.get('errors') or [], 'warnings': m.get('warnings') or [],
            'project_id': proj['project_id'] if proj else None,
            'project_name': sp.display_name(proj) if proj else None,
            'can_delete': _may_change(m)}


def _delivery_name(s):
    return s.get('delivery') or s['file']


def _may_change(m):
    """Delete a batch, or move it to another project: whoever made it, or an admin."""
    try:
        return bool(pipeline._owns_or_admin(m.get('user_id')))
    except RuntimeError:        # no request (a job thread building its result): the maker is the one asking
        return True


def _may_see(m):
    """A batch filed under a project is the team's: anyone with access to
    Vertical Shorts can play, download and send it. One that is not (made
    before projects, or its project has gone) is its maker's and an admin's,
    as every batch used to be."""
    if m.get('project_id') and sp.load(SHORTS_PROJECTS_DIR, m['project_id']):
        return True
    return _may_change(m)


def _batch_file_names(m):
    names = set()
    for s in m.get('shorts') or []:
        for k in ('file', 'srt', 'thumb', 'delivery'):
            if s.get(k):
                names.add(s[k])
    return names


# --------------------------------------------------------------------------
# Analyse job
# --------------------------------------------------------------------------

def _grab_frames(path, times, fps, max_w=768):
    """JPEG (base64) of the frame at each time, downscaled. The vision model
    resizes internally anyway; sending full 1080p only makes every call
    slower without changing what it sees."""
    out = []
    cap = cv2.VideoCapture(path)
    try:
        for t in times:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
            ok, frame = cap.read()
            if not ok:
                out.append((t, None))
                continue
            h, w = frame.shape[:2]
            if w > max_w:
                frame = cv2.resize(frame, (max_w, max(2, int(h * max_w / float(w)))),
                                   interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            out.append((t, base64.b64encode(buf.tobytes()).decode() if ok else None))
    finally:
        cap.release()
    return out


def _candidate_thumbs(path, cands, aid, fps, offset=0.0):
    """A still for each moment, taken from `path`, whose own time zero is
    `offset` on the analysis's timeline."""
    if not cands:
        return
    cap = cv2.VideoCapture(path)
    try:
        for c in cands:
            c['thumb'] = None
            at = c['start'] - offset + 0.35 * (c['end'] - c['start'])
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(at * fps)))
            ok, frame = cap.read()
            if not ok:
                continue
            h, w = frame.shape[:2]
            small = cv2.resize(frame, (360, max(2, int(h * 360 / float(max(w, 1))))),
                               interpolation=cv2.INTER_AREA)
            # Kept with the analysis, so they outlive the staged upload folder.
            d = _analysis_dir(aid)
            os.makedirs(d, exist_ok=True)
            name = f"{c['id']}.jpg"
            if cv2.imwrite(os.path.join(d, name), small, [cv2.IMWRITE_JPEG_QUALITY, 82]):
                c['thumb'] = f'/api/shorts/analysis/{aid}/thumb/{name}'
    finally:
        cap.release()


def _label(n, k, name):
    """What to call a part in a message: nothing for a one-file source."""
    return f'Part {k + 1} ({name}): ' if n > 1 else ''


def _grab_by_part(parts, times, max_w=768):
    """[(global time, JPEG b64 or None)] for global `times`, each taken from
    the part it falls in."""
    out = []
    for k, P in enumerate(parts):
        o, d = P['offset'], P['duration']
        last = k == len(parts) - 1
        mine = [t for t in times if o - 1e-6 <= t < o + d - 1e-6 or (last and t >= o + d - 1e-6)]
        if mine:
            out.extend((t + o, b) for t, b in _grab_frames(P['path'], [t - o for t in mine], P['fps'], max_w=max_w))
    return sorted(out, key=lambda x: x[0])


def _part_index(parts, t):
    """The part a time on the joined timeline falls in."""
    for k, P in enumerate(parts):
        if t < P['offset'] + P['duration'] - 1e-6:
            return k
    return len(parts) - 1


def _run_analysis(jid, params):
    """Finds the candidate moments. Returns the analysis id, or None when it
    ended with an error (already reported).

    The source is one video or several parts of one episode. Parts are laid
    end to end on a single timeline: each is probed, cut-detected,
    transcribed and listened to on its own, and then one story pass reads
    the whole episode. Every time in the analysis is on that joined
    timeline; a part's own times are its time minus its offset. No moment
    runs across the break between two parts."""
    report = params.get('_report') or functools.partial(pipeline.job_set, jid)
    srcs = params.get('parts') or [{'path': params['path'], 'name': params['orig_name']}]
    n_parts = len(srcs)
    vision_model, story_model = params['vision_model'], params['story_model']
    min_dur, max_dur, count = params['min_dur'], params['max_dur'], params['count']
    # "Auto": no number chosen in advance. Every moment the story model rates
    # SHORTS_AUTO_MIN_STORY or better is kept, up to the most a job handles.
    auto = count == 'auto'
    if auto:
        count = SHORTS_MAX_ITEMS

    # Both layers are load-bearing, so a service that's down fails the job
    # up front with the reason -- the same rule the promo generator follows
    # -- rather than quietly producing candidates picked on half the
    # evidence that look like the real thing.
    down = []
    chk = pipeline._check_service('ollama', pipeline.OLLAMA_URL, '/api/tags')
    if chk['status'] != 'up':
        down.append(f"AI Vision / story model (Ollama at {pipeline.OLLAMA_URL}): {chk.get('error', 'unreachable')}")
    chk = pipeline._check_service('whisper', pipeline.WHISPER_URL, '/')
    if chk['status'] != 'up':
        down.append(f"Speech-to-text (faster-whisper at {pipeline.WHISPER_URL}): {chk.get('error', 'unreachable')}")
    if down:
        report(error='Cannot look for shorts right now -- required service(s) unreachable: '
                         + '; '.join(down) + '. Picking moments needs both what is on screen and what is '
                         'said. Check Config > Services and try again once they are back.')
        return

    def at(lo, hi, k, frac=0.0):
        """Progress: [lo, hi] shared out between the parts, `frac` of the way through part k."""
        return int(lo + (hi - lo) * (k + frac) / n_parts)

    report(percent=2, step='Reading video' if n_parts == 1 else 'Reading the parts')
    parts, offset = [], 0.0
    for k, s in enumerate(srcs):
        _touch(s['path'])       # a long analysis must not be the reason its own source ages out
        info = sc.probe_source(pipeline.FFPROBE, s['path'])
        if not info['fps'] or info['frames'] <= 0 or not info['width']:
            report(error=_label(n_parts, k, s['name']) + 'This file could not be read as video. If it plays '
                         'elsewhere, it may be in a codec this server cannot decode -- try an H.264/ProRes copy.')
            return
        parts.append({'path': s['path'], 'name': s['name'], 'info': info, 'fps': info['fps'],
                      'duration': float(info['duration']), 'offset': offset,
                      'origin': pipeline.origin_of_path(s['path'])})
        offset += float(info['duration'])
    duration = offset
    fps = parts[0]['fps']
    if duration < min_dur + 5:
        report(error=f'This video is only {duration:.0f}s long -- too short to cut '
                         f'{int(min_dur)}-{int(max_dur)}s shorts from. Lower the minimum length or use a longer source.'
                     if n_parts == 1 else
                     f'These {n_parts} parts add up to only {duration:.0f}s -- too short to cut '
                     f'{int(min_dur)}-{int(max_dur)}s shorts from. Lower the minimum length.')
        return

    prod = pipeline.load_production_defaults()
    shots, cuts = [], [0.0]
    for k, P in enumerate(parts):
        report(percent=at(5, 20, k), step='Detecting cuts' if n_parts == 1 else f'Detecting cuts (part {k + 1}/{n_parts})')
        scene_list = pipeline.detect_scenes(P['path'], threshold=float(prod['scene_threshold']),
                                            min_scene_len_sec=float(prod['min_scene_len']),
                                            detector=prod['detector'],
                                            adaptive_threshold=float(prod['adaptive_threshold']))
        o = P['offset']
        P['cut_frames'] = sorted({int(pipeline.tc_frames(s)) for s, _ in scene_list} | {0})
        local = [(pipeline.tc_seconds(s), pipeline.tc_seconds(e)) for s, e in scene_list] or [(0.0, P['duration'])]
        shots.extend((o + a, o + b) for a, b in local)
        cuts.extend(o + f / P['fps'] for f in P['cut_frames'])
        cuts.append(o + P['duration'])
    cuts = sorted(set(cuts))

    # ---- Layer 1: how dramatic does it look ----
    times = sc.vision_sample_times(shots, duration, budget=params['vision_frames'])
    report(percent=20, step=f'Rating {len(times)} frames (AI vision)')
    items = [(t, b) for t, b in _grab_by_part(parts, times) if b]
    visual, errors = [], []
    progress = {'done': 0}
    lock = threading.Lock()

    def _rate(item):
        t, b64 = item
        try:
            score, desc, kind = sc.ask_vision(pipeline.OLLAMA_URL, vision_model, b64, with_kind=True)
            if score is not None:
                with lock:
                    visual.append({'t': t, 'score': score, 'desc': desc, 'kind': kind})
        except Exception as e:
            with lock:
                errors.append(str(e))
        with lock:
            progress['done'] += 1
            report(percent=20 + int(25 * progress['done'] / max(len(items), 1)),
                             step=f"AI-rating frame {progress['done']}/{len(items)}")

    if items:
        with ThreadPoolExecutor(max_workers=max(1, min(pipeline.AI_SCORE_WORKERS, len(items)))) as ex:
            list(ex.map(_rate, items))
    visual.sort(key=lambda v: v['t'])
    if not visual:
        why = (errors[0] if errors else 'its replies could not be read as a rating')
        report(error=f'The AI Vision model "{vision_model}" did not rate a single frame ({why}). '
                         'Check that the model is pulled on the Ollama server and is a vision model, '
                         'or pick another one under Advanced.')
        return
    if prod.get('unload_vision_after_scoring', True):
        # Same hand-off as the promo pipeline: free the GPU before whisper
        # loads, since the two commonly share one card.
        pipeline.unload_ollama_model(vision_model)

    # ---- Layer 2: does it tell a story ----
    warnings = []
    aligned = 0
    words_aligned = 0
    heard_why = None
    words, segments = [], []
    # Only what was chosen is passed on: no language means Whisper decides.
    stt = {k: v for k, v in (('language', params.get('language')), ('prompt', params.get('stt_prompt'))) if v}
    for k, P in enumerate(parts):
        lab = _label(n_parts, k, P['name'])
        report(percent=at(46, 54, k), step='Transcribing dialogue' if n_parts == 1
               else f'Transcribing dialogue (part {k + 1}/{n_parts})')
        w, sg, heard = pipeline.transcribe_video_detailed(P['path'], **stt)
        if not heard.get('ok'):
            # A transcription that FAILED is not a programme without dialogue.
            # Carrying on would pick moments on picture alone and present them
            # like any others -- the outcome the up-front service check exists
            # to prevent, arrived at by a different road.
            report(error=f"{lab}Could not transcribe the dialogue: {heard.get('reason') or 'unknown error'}. "
                         'Moments are chosen from what is said as well as what is seen, so nothing was picked. '
                         'Fix the speech-to-text service (Config > Services) and try again.')
            return
        heard_why = heard_why or heard.get('reason')
        w, sg = sc.normalize_transcript(w, sg)
        if heard.get('take'):
            # A master with its sound on separate tracks: the shorts take their
            # audio from where the dialogue was found, not from the first track.
            P['info']['audio_take'] = heard['take']
        if heard.get('audio'):
            warnings.append(f"{lab}The dialogue was read from {heard['audio']} of this file's audio"
                            + ('; the shorts use the same audio.' if heard.get('take') else '.'))
        # The episode's sound level, read once: lines are matched to it, the
        # moments' in and out points are moved onto it, and how loud a moment
        # gets is part of its score.
        report(percent=at(54, 58, k), step='Reading the sound')
        P['db'] = sc.read_envelope(pipeline.FFMPEG, P['path'], P['info'], timeout=pipeline.FFMPEG_LONG_TIMEOUT)
        if w:
            # Captions come up with the first word of a phrase: put that word
            # where the speech begins, not where Whisper put it after a silence.
            w, moved_w = sc.align_word_starts(w, P['db'])
            if moved_w:
                words_aligned += moved_w
                sg = sc.sync_segment_starts(sg, w)
        if sg and not w and sc.coarse_times(sg):
            # Timed only to the line, and rounded to the whole second: every line
            # is moved onto where its sound really starts and stops, which is what
            # captions, in and out points and the cliffhanger all go by.
            report(percent=at(55, 58, k), step='Matching the dialogue to the sound')
            sg, moved = sc.align_segments(sg, P['db'])
            aligned += moved
            if moved:
                warnings.append(f'{lab}The speech-to-text service gave times for whole lines only, rounded to the second, so '
                                f'{moved} of {len(sg)} lines were moved onto where their sound starts and stops. '
                                'Captions change a line at a time; a service that returns word timings would be closer still.')
        P['words'], P['segments'] = w, sg
        o = P['offset']
        words.extend(dict(x, start=x['start'] + o, end=x['end'] + o) for x in w)
        segments.extend(dict(x, start=x['start'] + o, end=x['end'] + o) for x in sg)
    if errors:
        warnings.append(f'{len(errors)} of {len(items)} frames could not be rated by the vision model '
                        f'({errors[0][:120]}); the rest were used.')
    breaks = [(P['offset'], f'Part {k + 1}') for k, P in enumerate(parts) if k]
    walls = [P['offset'] for P in parts[1:]]
    wall_ranges = [(w - 0.001, w + 0.001, ['part break']) for w in walls]
    beats, chunks, skips = [], [], []
    if segments:
        chunks = sc.chunk_segments(segments, SHORTS_STORY_CHUNK_SEC, SHORTS_STORY_OVERLAP_SEC)
        # How many to ask each stretch for: enough, over them all, to fill
        # the count half as much again -- but never more than a stretch can
        # hold end to end at the shortest length (eight at the very most),
        # which is what "Auto" asks for. Asking for more than fit only
        # invites padding.
        room = max(2, min(8, int(SHORTS_STORY_CHUNK_SEC // max(min_dur, 1.0)) + 1))
        per_chunk = room if auto else max(2, min(room, int(math.ceil(count * 1.5 / len(chunks))) + 1))
        failed, first_err = 0, None
        for ci, (lo, hi) in enumerate(chunks):
            report(percent=60 + int(26 * ci / len(chunks)),
                             step=f'Finding story beats (part {ci + 1}/{len(chunks)})')
            prompt = sc.build_story_prompt(segments, lo, hi, visual, min_dur, max_dur, per_chunk,
                                           params.get('focus'), params.get('avoid'), breaks=breaks or None)
            try:
                # Room for the reply to list them all: a cut-off reply loses the last ones.
                reply = sc.ask_story(pipeline.OLLAMA_URL, story_model, prompt, num_ctx=SHORTS_STORY_NUM_CTX,
                                     num_predict=max(1100, 450 + 170 * per_chunk))
            except Exception as e:
                failed += 1
                first_err = first_err or str(e)
                print(f'Vertical Shorts: story analysis failed on part {ci + 1}/{len(chunks)}: {e}')
                continue
            beats.extend(sc.parse_story_reply(reply, lo, hi))
            skips.extend(sc.parse_story_skips(reply, lo, hi))
        if failed == len(chunks):
            report(error=f'The story model "{story_model}" failed on every part of the transcript '
                             f'({first_err}). Check that it is pulled on the Ollama server, or pick another '
                             'one under Advanced.')
            return
        if failed:
            warnings.append(f'{failed} of {len(chunks)} parts of the transcript could not be analysed for story '
                            f'({(first_err or "")[:120]}); moments there may have been missed.')
        if not beats:
            beats = sc.heuristic_beats(segments, min_dur, max_dur, count)
            warnings.append('The story model did not find any moment that stands on its own, so these are '
                            'dialogue-heavy stretches ranked by how they look and sound instead. Expect to '
                            're-time them.')
        if prod.get('unload_vision_after_scoring', True):
            pipeline.unload_ollama_model(story_model)
    else:
        beats = sc.visual_windows(visual, min_dur, max_dur, duration, count)
        warnings.append(f"No dialogue was transcribed ({heard_why or 'nothing was heard'}), so these "
                        'were picked on visual intensity alone -- they are not checked for making sense as a '
                        'story.')

    if (params.get('focus') or params.get('avoid')) and beats and beats[0].get('source') in ('heuristic', 'visual'):
        # The two fallbacks rank on sound and picture; neither reads meaning,
        # so what the editor asked to feature or avoid had no say in them.
        warnings.append('What to feature and what to avoid are judged by the story model, which found nothing '
                        'here, so neither was applied to these moments.')

    report(percent=88, step='Building candidates')
    limit = count
    if auto and beats and beats[0].get('source') in ('heuristic', 'visual'):
        # Neither fallback rates anything, so there is no "worth making" to
        # go by: as many as would fit the programme end to end, at most.
        limit = max(3, min(SHORTS_MAX_ITEMS, int(duration // ((min_dur + max_dur) / 2.0))))
    # What is not the drama -- billboards, credits, narration, a host,
    # recaps, teasers, black between acts, and what the editor left out --
    # is kept out of every moment.
    if params.get('leave_out_parts'):
        editor_out = [(parts[k]['offset'] + a, parts[k]['offset'] + min(b, parts[k]['duration']))
                      for k, a, b in params['leave_out_parts'] if a < parts[k]['duration']]
    else:
        editor_out = [(a, min(b, duration)) for a, b in params.get('leave_out') or [] if a < duration]
    blocked = sc.not_story_ranges(skips, segments, visual, shots, editor_out)
    lengths = {}
    # Half as many again as wanted are built, so that a closer look -- at
    # their own frames and their sound -- can change which make the list.
    pool = min(SHORTS_MAX_ITEMS * 2, int(math.ceil(limit * 1.5)) + 2)
    cands = sc.build_candidates(beats, segments, words, cuts, visual, duration,
                                min_dur=min_dur, max_dur=max_dur, limit=pool, fps=fps,
                                min_story=SHORTS_AUTO_MIN_STORY if auto else None, report=lengths,
                                blocked=blocked, walls=walls)
    # In and out points onto the sound: not in the middle of a word.
    edges_moved = _refine_by_part(cands, parts, blocked + wall_ranges, min_dur, max_dur)
    # A few frames inside each moment, rated like the first sample.
    inner = sc.inner_sample_times(cands, [v['t'] for v in visual], SHORTS_MOMENT_FRAMES,
                                  per_moment=SHORTS_MOMENT_FRAMES_EACH)
    more = sorted({t for ts in inner.values() for t in ts})
    inner_rated = 0
    if more:
        report(percent=89, step=f'Rating {len(more)} frames inside the moments (AI vision)')
        got = [(t, b) for t, b in _grab_by_part(parts, more) if b]
        before = len(visual)
        progress['done'] = 0

        def _rate_inner(item):
            t, b64 = item
            try:
                score, desc, kind = sc.ask_vision(pipeline.OLLAMA_URL, vision_model, b64, with_kind=True)
                if score is not None:
                    with lock:
                        visual.append({'t': t, 'score': score, 'desc': desc, 'kind': kind})
            except Exception as e:
                print(f'Vertical Shorts: rating a frame inside a moment failed: {e}')
            with lock:
                progress['done'] += 1
                report(percent=89 + int(6 * progress['done'] / max(len(got), 1)),
                       step=f"AI-rating frames inside the moments {progress['done']}/{len(got)}")

        if got:
            with ThreadPoolExecutor(max_workers=max(1, min(pipeline.AI_SCORE_WORKERS, len(got)))) as ex:
                list(ex.map(_rate_inner, got))
        inner_rated = len(visual) - before
        visual.sort(key=lambda v: v['t'])
        if prod.get('unload_vision_after_scoring', True):
            pipeline.unload_ollama_model(vision_model)
        # A frame inside a moment may show it reaching into credits or a billboard.
        blocked = sc.not_story_ranges(skips, segments, visual, shots, editor_out)
        cands, gone = sc.clear_of(cands, blocked + wall_ranges, min_dur, fps)
        lengths['not_story'] = lengths.get('not_story', 0) + gone
    if blocked:
        warnings.append('Left out of every moment as not part of the drama: ' + sc.describe_ranges(blocked) + '. '
                        'Moments are cut short of these, or dropped when too little is left'
                        + (f" ({lengths['not_story']} dropped)" if lengths.get('not_story') else '') + '. '
                        'If something was left out by mistake, add it back with "Add your own moment".')
    dbs = [P['db'] for P in parts if P.get('db') is not None and len(P['db'])]
    ref = sc.sound_reference(np.concatenate(dbs)) if dbs else None
    for c in cands:
        P = parts[_part_index(parts, c['start'])]
        o = P['offset']
        sc.rescore(c, visual, words, segments, sound=sc.sound_score(P.get('db'), c['start'] - o, c['end'] - o, ref))
    cands = sc.dedupe_windows(cands)[:max(1, int(limit))]
    for n, c in enumerate(cands, 1):
        c['id'] = f'c{n}'
    if lengths.get('too_short'):
        k = lengths['too_short']
        warnings.append(f"{k} moment{'' if k == 1 else 's'} the story model picked could not be brought up to the "
                        f"{int(min_dur)}-second minimum without running into another scene, and "
                        f"{'was' if k == 1 else 'were'} left out. Lower the minimum to see "
                        f"{'it' if k == 1 else 'them'}.")
    elif lengths.get('kept_short'):
        warnings.append(f'None of the moments found could be brought up to the {int(min_dur)}-second minimum, so '
                        'they are listed as they are, marked Short. Lower the minimum, or lengthen them by hand.')
    if auto and cands and all(c['story_score'] is not None and c['story_score'] < SHORTS_AUTO_MIN_STORY
                              for c in cands):
        warnings.append(f'Auto keeps the moments the story model rates {SHORTS_AUTO_MIN_STORY}/10 or better. '
                        f'None reached that here, so all {len(cands)} it found are listed instead.')
    if not cands:
        report(error='No usable moments were found in this video. Try a wider length range, '
                         'or add your own ranges by hand after re-running with a different model.')
        return
    aid = secrets.token_hex(8)
    for k, P in enumerate(parts):
        _candidate_thumbs(P['path'], [c for c in cands if _part_index(parts, c['start']) == k],
                          aid, P['fps'], offset=P['offset'])
    stored = {
        'excluded': [{'start': round(a, 2), 'end': round(b, 2), 'kinds': k} for a, b, k in blocked],
        'user_id': params.get('user_id'), 'username': params.get('username'),
        'project_id': params.get('project_id'),
        'candidates': cands, 'warnings': warnings,
        'options': {'min_dur': min_dur, 'max_dur': max_dur, 'count': 'auto' if auto else count,
                    'focus': params.get('focus'), 'avoid': params.get('avoid'),
                    'language': params.get('language'), 'stt_prompt': params.get('stt_prompt'),
                    'leave_out': editor_out},
        'stats': {'shots': len(shots), 'frames_rated': len(visual), 'transcript_lines': len(segments),
                  'lines_aligned': aligned, 'words_aligned': words_aligned, 'edges_moved': edges_moved, 'frames_inside': inner_rated,
                  'story_parts': len(chunks), 'vision_model': vision_model, 'story_model': story_model,
                  'parts': n_parts},
    }
    if n_parts == 1:
        P = parts[0]
        stored.update(origin=P['origin'], path=P['path'], orig_name=params['orig_name'], info=P['info'],
                      cut_frames=P['cut_frames'], words=P['words'], segments=P['segments'])
    else:
        # One timeline: the top level speaks for the whole episode (its
        # duration is the sum), and everything that needs a file, a frame
        # rate or a transcript in the file's own time lives in the part.
        stored.update(path=parts[0]['path'], orig_name=params['orig_name'],
                      info=dict(parts[0]['info'], duration=duration), cut_frames=[], words=[], segments=[],
                      parts=[{k: P[k] for k in ('path', 'name', 'info', 'duration', 'offset', 'cut_frames',
                                                 'words', 'segments', 'origin')} for P in parts])
    analysis_store(aid, stored)
    for P in parts:
        _touch(P['path'])
    report(percent=100, step='Done', done=True,
           result={'analysis_id': aid, 'candidates': len(cands)})
    return aid


def _refine_by_part(cands, parts, blocked, min_dur, max_dur):
    """sc.refine_edges, a part at a time: each moment's ends are moved onto
    the sound of the part it is in, in that part's own time, and put back on
    the joined timeline. How many moved."""
    moved = 0
    for P in parts:
        if P.get('db') is None:
            continue
        o, d = P['offset'], P['duration']
        mine = [c for c in cands if o - 1e-6 <= c['start'] < o + d - 1e-6]
        if not mine:
            continue
        for c in mine:
            c['start'] -= o
            c['end'] -= o
        local = [(a - o, b - o, k) for a, b, k in blocked if b > o and a < o + d]
        moved += sc.refine_edges(mine, P['db'], sc.speech_units(P['words'], P['segments']), d, min_dur, max_dur,
                                 P['fps'], blocked=local)
        for c in mine:
            c['start'] = round(c['start'] + o, 3)
            c['end'] = round(c['end'] + o, 3)
    return moved


# --------------------------------------------------------------------------
# Render job
# --------------------------------------------------------------------------

def _poster(video_path, thumb_path, at):
    try:
        pipeline.run_ffmpeg([pipeline.FFMPEG, '-y', '-hide_banner', '-loglevel', 'error',
                             '-ss', f'{max(0.0, at):.3f}', '-i', video_path, '-frames:v', '1',
                             '-vf', 'scale=270:-2', '-q:v', '4', thumb_path],
                            timeout=pipeline.FFMPEG_TIMEOUT, label='shorts poster')
    except pipeline.MediaToolTimeout:
        return False
    return os.path.exists(thumb_path) and os.path.getsize(thumb_path) > 0


def _framing_overrides(it, start_f, fps):
    """The editor's per-shot framing for a moment, as apply_framing wants
    it: [(frame counted from the clip's first, layout, x)]."""
    fr = it.get('framing') or {}
    return [(int(round(o['at'] * fps)) - start_f, o['layout'], o.get('x')) for o in fr.get('shots') or []]


def _plan_moment(a, src, info, start_f, end_f, reframe, speaker, detector, framing=None, mouth=None, splits_out=None):
    """How one moment is reframed: (segments, shot starts within it).

    The faces (and, where there are none, the head-and-shoulders) in it are
    sampled, the planner decides shot by shot, and the editor's corrections,
    if any, are laid over that. Shared by the render and the framing
    preview, so what the editor is shown is what is rendered."""
    fps = info['fps']
    n_frames = end_f - start_f
    crop_w, _ = sc.crop_geometry(info['disp_w'], info['disp_h'])
    bodies, focus = [], []
    samples = (sc.sample_faces(src, start_f, n_frames, fps, detector, sar=info['sar'],
                               mouth=speaker if mouth is None else mouth, bodies=bodies,
                               focus=focus, focus_w=crop_w)
               if detector else [])
    shot_starts = [c - start_f for c in a['cut_frames'] if start_f < c < end_f]
    t0, t1 = start_f / fps, end_f / fps
    speech = ([(s - t0, e - t0) for s, e in sc.speech_units(a['words'], a['segments']) if e > t0 and s < t1]
              if speaker else None)
    segs = sc.plan_reframe(samples, shot_starts, n_frames, info['disp_w'], info['disp_h'], crop_w,
                           mode=reframe, fps=fps, speaker=speaker, speech=speech, bodies=bodies,
                           focus=focus)
    # Where the screen could be split, for the editor to offer and to apply:
    # the same samples planned as 'split', so no second look at the faces.
    splits = []
    if (splits_out is not None or any(o[1] == 'split' for o in framing or [])) and samples and reframe != 'fit':
        splits = [x for x in sc.plan_reframe(samples, shot_starts, n_frames, info['disp_w'], info['disp_h'], crop_w,
                                             mode='split', fps=fps, speaker=False, speech=None, bodies=bodies,
                                             focus=focus) if x['layout'] == 'split']
    if splits_out is not None:
        splits_out.extend(splits)
    if framing:
        segs = sc.apply_framing(segs, shot_starts, n_frames, framing, max(0.0, float(info['disp_w'] - crop_w)),
                                splits=splits, disp=(info['disp_w'], info['disp_h']))
    return segs, shot_starts


def _parse_framing(raw, label, duration):
    """An item's per-shot framing from a request -> ({'shots': [{'at',
    'layout', 'x'}]} or None, error or None). `at` is a time in the source,
    in seconds, inside the shot meant; `x` the window's left edge in source
    pixels, for 'crop'."""
    if raw in (None, '', False):
        return None, None
    shots = raw.get('shots') if isinstance(raw, dict) else None
    if not isinstance(shots, list) or len(shots) > 500:
        return None, f'"{label}": its framing is not valid.'
    out = []
    for o in shots:
        try:
            at = float(o.get('at'))
            layout = str(o.get('layout'))
            x = float(o.get('x')) if layout == 'crop' else None
        except (AttributeError, TypeError, ValueError):
            return None, f'"{label}": its framing is not valid.'
        if layout not in ('crop', 'fit', 'split') or not (0.0 <= at <= duration) or (x is not None and not math.isfinite(x)):
            return None, f'"{label}": its framing is not valid.'
        out.append({'at': round(at, 3), 'layout': layout, 'x': None if x is None else round(max(0.0, x), 1)})
    return ({'shots': out} if out else None), None


def _auto_cues(a, max_chars):
    """`auto(t0, t1)` -> the automatic captions for that stretch of the
    analysed source, timed from t0."""
    return lambda t0, t1: sc.subtitle_cues(a['words'], a['segments'], t0, t1, max_chars=max_chars)


def _item_cues(a, it, t0, t1, max_chars):
    """(cues, edited?) for a moment running t0..t1 of the source: the
    automatic captions, or the editor's own where they sent some.

    Edited captions arrive with the range they were edited for. If the in
    or out point was moved afterwards they are carried over (sc.fit_cues),
    not thrown away."""
    auto = _auto_cues(a, max_chars)
    edit = it.get('captions')
    if not edit:
        return auto(t0, t1), False
    return sc.clean_cues(sc.fit_cues(edit['cues'], edit['start'], edit['end'], t0, t1, auto), t1 - t0), True


def _parse_captions(raw, label, duration):
    """An item's edited captions from a request -> ({'start', 'end', 'cues'}
    or None, error or None). `start`/`end` are the source range they were
    edited for; the cues are timed from that start."""
    if raw in (None, '', False):
        return None, None
    if not isinstance(raw, dict):
        return None, f'"{label}": its captions are not valid.'
    try:
        start, end = float(raw.get('start')), float(raw.get('end'))
    except (TypeError, ValueError):
        return None, f'"{label}": its captions are not valid.'
    if not (0.0 <= start < end <= duration + 1.0):
        return None, f'"{label}": its captions are not valid.'
    try:
        cues = sc.clean_cues(raw.get('cues'), end - start)
    except ValueError as e:
        return None, f'"{label}": {e}'
    return {'start': start, 'end': end, 'cues': cues}, None


def _kept_sfx(bdir, options):
    """A batch's ending sound effect, as kept beside its shorts, for rendering one of them again."""
    fx = (options or {}).get('ending_sfx') or {}
    path = os.path.join(bdir, os.path.basename(fx.get('file') or '')) if fx.get('file') else None
    return {'path': path, 'gain': fx.get('gain') or 0.0} if path and os.path.isfile(path) else None


def _encode_short(tag, src, info, plan, cues, opts, mp4_path, delivery_path, report=None):
    """Renders one short to `mp4_path` and, for a delivery format other than
    MP4, makes `delivery_path` from it. (ok, error).

    `plan` is where it is in the source and how it is reframed ({'start_f',
    'n_frames', 'segs', 'room', and for the cliffhanger ending 'after',
    'audio_out', 'audio_fade'}); `opts` how it is finished ({'burn',
    'subtitle_size', 'ending', 'format', 'loudness'}). Everything a render needs and
    nothing an analysis holds, so a saved short can be rendered again with
    different captions long after its analysis has gone."""
    work = app.config['UPLOAD_FOLDER']
    fmt = opts.get('format') or 'mp4_high'
    # Bare [A-Za-z0-9_] name in ffmpeg's working directory: see render_short.
    ass_name = re.sub(r'[^A-Za-z0-9_.]', '_', f'shsub_{tag}.ass') if opts.get('burn') and cues else None
    try:
        if ass_name:
            sc.write_ass(cues, os.path.join(work, ass_name), size=opts['subtitle_size'], font=SHORTS_SUB_FONT)
        ok, err = sc.render_short(pipeline.FFMPEG, src, mp4_path, plan['start_f'], plan['n_frames'], info,
                                  plan['segs'], ass_name=ass_name, work_dir=work,
                                  crf=SHORTS_CRF if fmt == 'mp4_high' else min(SHORTS_CRF, SHORTS_MASTER_CRF),
                                  preset=SHORTS_PRESET,
                                  loudness=pipeline.resolve_loudness(opts.get('loudness'), SHORTS_LOUDNESS),
                                  true_peak=SHORTS_TRUE_PEAK,
                                  timeout=pipeline.FFMPEG_LONG_TIMEOUT, ending=bool(opts.get('ending')),
                                  ending_room=plan.get('room'), ending_after=plan.get('after') or 0,
                                  audio_out=plan.get('audio_out'), audio_fade=plan.get('audio_fade'),
                                  ending_hold=opts.get('ending_hold'), ending_sfx=opts.get('ending_sfx'),
                                  ending_fade=opts.get('ending_fade'))
    except sc.ToolTimeout as e:
        ok, err = False, f'Encoding took too long and was stopped ({e}).'
    finally:
        if ass_name:
            try:
                os.remove(os.path.join(work, ass_name))
            except OSError:
                pass
    if not ok or fmt == 'mp4_high':
        return ok, err
    if report:
        report()
    label = pipeline.EXPORT_FORMATS[fmt]['label']
    try:
        r = pipeline.run_ffmpeg(pipeline.build_export_cmd(mp4_path, delivery_path, fmt),
                                timeout=pipeline.FFMPEG_LONG_TIMEOUT, label='shorts delivery file')
    except pipeline.MediaToolTimeout as e:
        return False, f'Making the {label} file took too long and was stopped ({e}).'
    if not (os.path.exists(delivery_path) and os.path.getsize(delivery_path) > 0):
        return False, (f'The {label} file could not be made: '
                       + (sc.ffmpeg_error(getattr(r, 'stderr', '')) or 'ffmpeg produced no output'))
    return True, None


def _run_render(jid, params):
    report = params.get('_report') or functools.partial(pipeline.job_set, jid)
    a = params['analysis']
    parts = _parts(a)
    multi = len(parts) > 1
    info = parts[0]['info']
    # In the order they happen in the episode, whatever order they were
    # ticked, ranked or listed in: the number in a short's filename is then
    # its place in the story, and a folder of them sorts into that order.
    items = sorted(params['items'], key=lambda it: (float(it['start']), float(it['end'])))
    items = [dict(it, title=it.get('title') or f'Short {n}') for n, it in enumerate(items, 1)]
    reframe = params['reframe']
    fmt = params.get('format') if params.get('format') in SHORTS_FORMATS else 'mp4_high'
    loudness = pipeline.resolve_loudness(params.get('loudness'), SHORTS_LOUDNESS)
    src, why = _ensure_source(a, params.get('analysis_id'), report)
    if not src:
        report(error=why)
        return
    for P in _parts(a):
        _touch(P['path'])
    report(percent=2, step='Preparing')

    # A render nobody reviewed first carries the analysis's own warnings
    # (a fallback was used, part of the transcript was skipped): the batch
    # card is then the only place the editor will ever see them.
    warnings = list(params.get('warnings') or [])
    want_captions = bool(params['subtitles'])
    burn = want_captions and _captions_available()
    if want_captions and not burn:
        warnings.append("This server's ffmpeg build has no libass, so captions could not be burned in. "
                        'The .srt files are still written.')
    detector = None
    if reframe != 'fit' and any(P['info']['disp_w'] > sc.crop_geometry(P['info']['disp_w'], P['info']['disp_h'])[0]
                                for P in parts):
        detector = sc.FaceDetector(_face_model_path())
        if not detector.available():
            detector = None
            warnings.append('No face detector is available in this OpenCV build, so shots were centre-cropped.')

    # Following the speaker needs faces to compare and words to time them
    # against; without either it is simply not applied, and says so.
    speaker = bool(params.get('speaker')) and detector is not None
    if speaker and not any(P['words'] or P['segments'] for P in parts):
        speaker = False
        warnings.append('No dialogue was transcribed for this video, so "Follow the speaker" had nothing to go on '
                        'and was not applied.')

    bid = f'{int(time.time())}_{secrets.token_hex(3)}'
    bdir = os.path.join(SHORTS_DIR, bid)
    os.makedirs(bdir, exist_ok=True)
    params['_batch_dir'] = bdir
    # The ending's sound effect is kept with the batch: a short can be rendered again (new
    # captions) long after the staged copy it was picked as has been cleared away.
    sfx_kept = None
    if params.get('ending') == 'cliffhanger' and params.get('ending_sfx'):
        sfx_kept = 'ending_sfx' + os.path.splitext(params['ending_sfx']['path'])[1].lower()
        try:
            shutil.copy(params['ending_sfx']['path'], os.path.join(bdir, sfx_kept))
        except OSError as e:
            sfx_kept = None
            warnings.append(f'The ending sound effect could not be used ({e}); the shorts have no sound on their ending.')
    stem = sc.slugify(os.path.splitext(a['orig_name'] or '')[0], 40) or 'video'
    manifest = {'batch_id': bid, 'created': time.time(), 'user_id': params.get('user_id'),
                'username': params.get('username'), 'orig_name': a['orig_name'], 'status': 'rendering',
                'project_id': a.get('project_id'),
                'source_duration': round(float(a['info'].get('duration') or 0), 3) or None,
                'parts': ([{'name': P.get('name'), 'offset': round(P['offset'], 3),
                            'duration': round(P['duration'], 3)} for P in parts] if multi else None),
                'options': {'reframe': reframe, 'subtitles': want_captions,
                            'subtitle_size': params['subtitle_size'],
                            'face_detector': detector.kind if detector else None,
                            'speaker': speaker, 'ending': params.get('ending') or 'none',
                            'ending_hold': params.get('ending_hold') if params.get('ending') == 'cliffhanger' else None,
                            'fade_last': bool(params.get('fade_last')),
                            'ending_fade': params.get('ending_fade') if params.get('fade_last') else None,
                            'ending_sfx': ({'file': sfx_kept, 'name': params['ending_sfx']['name'],
                                            'gain': params['ending_sfx']['gain']} if sfx_kept else None),
                            'format': fmt, 'format_label': pipeline.EXPORT_FORMATS[fmt]['label'],
                            'loudness': loudness},
                'numbering': 'episode',       # short N is the Nth of these moments in the episode
                # What rendering one of these again takes (new captions on a
                # saved short): the file it was cut from and how it reads.
                # Server-side only; never sent to the browser.
                'source': {'path': src, 'info': info, 'burn': burn},
                'shorts': [], 'errors': [], 'warnings': warnings}
    _write_manifest(bdir, manifest)

    ending = params.get('ending') == 'cliffhanger'
    max_chars = sc.SUBTITLE_SIZES[params['subtitle_size']][1]
    opts = {'burn': burn, 'subtitle_size': params['subtitle_size'], 'ending': ending, 'format': fmt,
            'loudness': loudness, 'ending_hold': params.get('ending_hold') if ending else None,
            'ending_sfx': ({'path': os.path.join(bdir, sfx_kept), 'gain': params['ending_sfx']['gain']}
                           if sfx_kept else None)}
    ext = pipeline.EXPORT_FORMATS[fmt]['ext']
    total = len(items)
    unit_cache = {}
    for n, it_global in enumerate(items, 1):
        base = 5 + 93.0 * (n - 1) / total
        span = 93.0 / total
        # The part this moment is in, and the moment in that part's own time.
        k, bad = _locate(a, it_global['start'], it_global['end'])
        if bad:
            manifest['errors'].append({'index': n, 'title': it_global['title'], 'error': bad})
            continue
        v = _part_view(a, k)
        src, info, off = v['path'], v['info'], v.get('_offset', 0.0)
        fps = info['fps']
        it = _localize_item(it_global, off)
        # The last short of the render may finish on a fade to black instead of the ending chosen.
        fades = bool(params.get('fade_last')) and n == total
        ending_k = ending and not fades
        if ending_k and k not in unit_cache:
            unit_cache[k] = sc.speech_units(v['words'], v['segments'])
        units = unit_cache.get(k, [])
        start_f = int(round(it['start'] * fps))
        end_f = min(info['frames'], int(round(it['end'] * fps)))
        n_frames = end_f - start_f
        if n_frames < int(SHORTS_MIN_CLIP * fps):
            manifest['errors'].append({'index': n, 'title': it['title'],
                                       'error': 'Range is past the end of the video.'})
            continue

        report(percent=int(base), step=f'Short {n}/{total}: finding faces')
        segs, shot_starts = _plan_moment(v, src, info, start_f, end_f, reframe, speaker, detector,
                                         framing=_framing_overrides(it, start_f, fps))
        # How long the clip's last shot has been on screen by its last frame:
        # the cliffhanger hold is made from the end of it, never across a cut.
        room = n_frames - max(shot_starts) if shot_starts else n_frames
        t0, t1 = start_f / fps, end_f / fps
        after, audio_out, audio_fade = 0, None, None
        if ending_k:
            # The hold is made from the frames just AFTER the out point where
            # that can be done -- the same shot carrying on, and still -- so
            # the moment plays to its last frame. One frame is left before
            # the next cut: a cut list can be a frame out.
            nxt = min((c for c in v['cut_frames'] if c >= end_f), default=None)
            ahead = min(info['frames'] - end_f, nxt - end_f - 1 if nxt is not None else info['frames'])
            after = sc.still_frames_after(src, end_f, fps, ahead) if ahead > 0 else 0
            if not after:
                # Out is on a cut, or at the end of the file: from the
                # moment's own last frames, and never ones in which the pose
                # is still changing.
                room = sc.still_frames(src, end_f - 1, fps, room)
            # The sound is not cut where the transcript says the last word
            # ends: it is listened to, and stops where the word does.
            nxt_word = next((s - t0 for s, _ in units if s >= t1 - 0.05), None)
            audio_out, audio_fade, heard = sc.measure_audio_out(pipeline.FFMPEG, src, info, start_f, n_frames,
                                                                nxt_word)
            if heard == 'pause' and audio_out - n_frames / fps > 0.05:
                print(f'Vertical Shorts: short {n}/{total}: last word runs '
                      f'{audio_out - n_frames / fps:.2f}s past the out point; the sound is let finish under the hold')
        cues, edited = _item_cues(v, it, start_f / fps, end_f / fps, max_chars)
        cues = sc.place_cues(cues, segs, fps)
        plan = {'start_f': start_f, 'n_frames': n_frames, 'segs': segs, 'room': room,
                'after': after, 'audio_out': audio_out, 'audio_fade': audio_fade}

        name = f"{stem}_short_{n:02d}_{sc.slugify(it['title'], 40) or 'clip'}"
        out_path = os.path.join(bdir, name + '.mp4')
        delivery = None if fmt == 'mp4_high' else f'{name}.{ext}'
        report(percent=int(base + span * 0.25), step=f'Short {n}/{total}: encoding')
        sopts = dict(opts, ending=ending_k, ending_hold=opts['ending_hold'] if ending_k else None,
                     ending_sfx=opts['ending_sfx'] if ending_k else None,
                     ending_fade=params.get('ending_fade') if fades else None)
        ok, err = _encode_short(f'{jid}_{n}', src, info, plan, cues, sopts, out_path,
                                os.path.join(bdir, delivery) if delivery else None,
                                report=lambda: report(percent=int(base + span * 0.8),
                                                      step=f'Short {n}/{total}: making the delivery file'))
        if not ok:
            print(f'Vertical Shorts: short {n}/{total} failed: {err}')
            for leftover in (out_path, os.path.join(bdir, delivery) if delivery else None):
                if leftover and os.path.exists(leftover):
                    try:
                        os.remove(leftover)
                    except OSError:
                        pass
            manifest['errors'].append({'index': n, 'title': it['title'], 'error': err})
            _write_manifest(bdir, manifest)
            continue

        entry = {'index': n, 'title': it['title'], 'file': name + '.mp4', 'srt': None, 'thumb': None,
                 'start': round(start_f / fps + off, 3), 'end': round(end_f / fps + off, 3),
                 'duration': round((n_frames + (sc.cliffhanger_extra(n_frames, fps, room, after, params.get('ending_hold'))
                                               if ending_k else 0)) / fps, 2),
                 'ending': 'fade' if fades else ('cliffhanger' if ending_k else 'none'),
                 'ending_fade': params.get('ending_fade') if fades else None,
                 'size': os.path.getsize(out_path),
                 'layouts': {'crop': sum(1 for s in segs if s['layout'] == 'crop'),
                             'fit': sum(1 for s in segs if s['layout'] == 'fit'),
                             'tracked': sum(1 for s in segs if s.get('keys')),
                             'speaker': sum(1 for s in segs if s.get('speaker')),
                             'split': sum(1 for s in segs if s['layout'] == 'split')},
                 'captions': bool(burn and cues),
                 'framing_edited': bool((it.get('framing') or {}).get('shots')),
                 # Kept so the captions can be changed afterwards: the lines
                 # themselves, and the plan a re-render needs.
                 'cues': cues, 'captions_edited': edited, 'plan': plan}
        if multi:
            # Which part it is from, where in it, and the file it was cut from
            # (what rendering it again with new captions goes back to).
            entry.update(part=k + 1, part_name=parts[k].get('name'), part_start=round(start_f / fps, 3),
                         part_end=round(end_f / fps, 3), source={'path': src, 'info': info})
        if delivery:
            entry['delivery'] = delivery
            entry['delivery_size'] = os.path.getsize(os.path.join(bdir, delivery))
        if cues:
            sc.write_srt(cues, os.path.join(bdir, name + '.srt'))
            entry['srt'] = name + '.srt'
        if _poster(out_path, os.path.join(bdir, name + '.jpg'), entry['duration'] * 0.3):
            entry['thumb'] = name + '.jpg'
        manifest['shorts'].append(entry)
        # Rewritten after every short, so a cancel, crash or restart part-way
        # through leaves a readable batch holding everything finished so far.
        _write_manifest(bdir, manifest)

    if not manifest['shorts']:
        first = manifest['errors'][0]['error'] if manifest['errors'] else 'unknown error'
        shutil.rmtree(bdir, ignore_errors=True)
        report(error=f'None of the {total} shorts could be rendered. First error: {first}')
        return
    manifest['status'] = 'complete' if not manifest['errors'] else 'partial'
    _write_manifest(bdir, manifest)
    report(percent=100, step='Done', done=True, result={'batch': _batch_public(manifest)})


class _Phase:
    """Progress reporting for one part of a job made of several.

    Passed to a job body as params['_report'] in place of job_set. It maps
    the body's own 0-100 onto [lo, hi] of the whole job's bar, and -- unless
    this is the last part -- keeps the body's "done" to itself (holding on
    to the result) so the job isn't marked finished while there is more to
    run. Errors and cancellation go straight through: either ends the job."""

    def __init__(self, jid, lo, hi, last, extra_result=None):
        self.jid, self.lo, self.hi, self.last = jid, lo, hi, last
        self.extra_result = extra_result or {}
        self.result = None

    def __call__(self, **kw):
        if kw.get('percent') is not None:
            kw['percent'] = int(round(self.lo + (self.hi - self.lo) * kw['percent'] / 100.0))
        if kw.get('done') and kw.get('error') is None:
            if not self.last:
                self.result = kw.pop('result', None) or {}
                kw.pop('done')
                kw.pop('step', None)
            elif kw.get('result') is not None:
                kw['result'] = dict(kw['result'], **self.extra_result)
        pipeline.job_set(self.jid, **kw)


def _run_auto(jid, params):
    """Analyse, then render every moment found, as one job with no review in
    between ("Generate without preview"). The analysis is stored exactly as
    a previewed one is, so the editor can still open the moments afterwards,
    re-time them and render again without analysing twice."""
    aid = _run_analysis(jid, dict(params, _report=_Phase(jid, 0, AUTO_SPLIT, last=False)))
    if not aid:
        return
    a = analysis_get(aid)
    if a is None:
        pipeline.job_set(jid, error='The analysis finished but was no longer available to render from. Try again.')
        return
    render = dict(params['render'], analysis=a, analysis_id=aid, warnings=list(a.get('warnings') or []),
                  items=[{'start': c['start'], 'end': c['end'], 'title': c['title']}
                         for c in a['candidates'][:SHORTS_MAX_ITEMS]],
                  user_id=params.get('user_id'), username=params.get('username'),
                  _report=_Phase(jid, AUTO_SPLIT, 100, last=True, extra_result={'analysis_id': aid}))
    params['_render'] = render          # where _settle_auto finds the batch folder
    _run_render(jid, render)


def _settle_auto(params):
    _settle_batch(params.get('_render') or {})


def _settle_batch(params):
    """Runs after a render job however it ended. A batch still marked
    'rendering' was interrupted: keep whatever finished, or remove the
    folder if nothing did."""
    bdir = params.get('_batch_dir')
    if not bdir or not os.path.isdir(bdir):
        return
    _settle_dir(bdir)


def _settle_dir(bdir):
    m = _read_manifest(bdir)
    if m is None:
        shutil.rmtree(bdir, ignore_errors=True)
        return
    if m.get('status') != 'rendering':
        return
    if not m.get('shorts'):
        shutil.rmtree(bdir, ignore_errors=True)
        return
    m['status'] = 'partial'
    _write_manifest(bdir, m)


def _guard(fn, after=None):
    """Wraps a job body with the same outcomes run_trailer_job gives a promo
    job: a cancel, a media-tool timeout and an unexpected crash each end the
    job with a message instead of leaving it 'running' forever."""
    def runner(jid, params):
        try:
            fn(jid, params)
        except pipeline.JobCancelled:
            print(f'Vertical Shorts job {jid} cancelled')
            pipeline.job_set(jid, error='Cancelled', status='cancelled')
        except (pipeline.MediaToolTimeout, sc.ToolTimeout) as e:
            pipeline.job_set(jid, error=f'A media processing step timed out and was stopped ({e}). '
                                        'The source may be corrupt, or the server is overloaded.')
        except Exception as e:
            traceback.print_exc()
            pipeline.job_set(jid, error=f'Unexpected error: {e}')
        finally:
            if after:
                try:
                    after(params)
                except Exception as e:
                    print(f'Vertical Shorts job {jid} cleanup error: {e}')
    return runner


def _spawn(fn, *args, **kwargs):
    """Runs a job body on its own thread. Its own function so the test suite
    can swap in a synchronous version without touching threading itself."""
    threading.Thread(target=fn, args=args, kwargs=kwargs, daemon=True).start()


def _start_job(kind, body, params, label, after=None):
    jid = pipeline.job_new(user_id=session.get('user_id'), username=session.get('username'), kind='shorts')
    pipeline.job_set_orig_name(jid, label)
    if len(_JOB_KINDS) > 500:
        for k in list(_JOB_KINDS)[:250]:
            _JOB_KINDS.pop(k, None)
    _JOB_KINDS[jid] = kind
    # Through the same gate as promo jobs: both are heavy ffmpeg + GPU work,
    # and MAX_CONCURRENT_JOBS is a limit on the server, not on one feature.
    _spawn(pipeline.run_trailer_job_gated, jid, params, runner=_guard(body, after))
    return jid


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

MAX_PARTS = 10


def _resolve_source(field='shorts_file'):
    """(path, display name, error). Same policy as the promo generator's
    load_video(): a direct upload only where the deployment allows it,
    otherwise a file already staged from a configured network share."""
    f = request.files.get(field)
    if f is not None and f.filename:
        if not pipeline.ALLOW_LOCAL_MEDIA_UPLOAD:
            return None, None, ('Direct file upload is disabled on this deployment. '
                                'Use Browse library to pick a file from a configured network folder.')
        if not pipeline.allowed_file(f.filename):
            return None, None, 'File type not allowed'
        fn = secure_filename(f.filename)
        if not fn:
            return None, None, 'Invalid filename'
        path = os.path.join(app.config['UPLOAD_FOLDER'],
                            f'shsrc_{int(time.time() * 1000)}_{threading.get_ident()}_{fn}')
        f.save(path)
        ok, err = pipeline.check_video_size(os.path.getsize(path), label=fn)
        if not ok:
            try:
                os.remove(path)
            except OSError:
                pass
            return None, None, err
        return path, fn, None
    staged = (request.form.get(field + '_network') or '').strip()
    if staged:
        safe = os.path.basename(staged)
        path = pipeline.staged_path(safe)          # the copy here, or the file where it was left on the share
        if safe.startswith('net_') and os.path.exists(path):
            return path, re.sub(r'^net_\d+_', '', safe), None
        return None, None, 'Selected network file is no longer available -- please re-select it'
    return None, None, 'No video provided'


def _resolve_sources():
    """([{path, name}], display name, error): the one source, or its parts
    (fields shorts_file, shorts_file2 ... shorts_file10, in that order).
    A gap -- part 3 given without part 2 -- is simply closed up."""
    srcs = []
    for field in ['shorts_file'] + [f'shorts_file{k}' for k in range(2, MAX_PARTS + 1)]:
        has = ((request.files.get(field) is not None and request.files.get(field).filename)
               or (request.form.get(field + '_network') or '').strip())
        if not has:
            continue
        path, name, err = _resolve_source(field)
        if not path:
            return None, None, (err if len(srcs) == 0 and field == 'shorts_file' else f'Part {len(srcs) + 1}: {err}')
        srcs.append({'path': path, 'name': name})
    if not srcs:
        return None, None, 'No video provided'
    if len(srcs) == 1:
        return srcs, srcs[0]['name'], None
    stem = os.path.splitext(srcs[0]['name'])[0]
    return srcs, f'{stem} ({len(srcs)} parts)', None


_PART_PREFIX = re.compile(r'^\s*p(?:art)?\s*(\d{1,2})\s*[:\s]\s*', re.I)


def _parse_leave_out(text, n_parts):
    """(plain ranges, per-part ranges, error). With one source this is the
    plain list. With several, each entry may start 'part 2' (or 'p2'); one
    that does not is read as part 1. Per-part ranges are [(k, start, end)]
    in that part's own time, 'end' meaning the part's end."""
    text = (text or '')[:1500]
    if n_parts <= 1:
        r, bad = sc.parse_ranges(text, 10 ** 7)
        return r, None, bad
    out = []
    for piece in re.split(r'[,;\n]+', text):
        if not piece.strip():
            continue
        m = _PART_PREFIX.match(piece)
        k = int(m.group(1)) if m else 1
        if not 1 <= k <= n_parts:
            return None, None, f'"{piece.strip()}": there is no part {k} -- this episode has {n_parts} parts.'
        r, bad = sc.parse_ranges(piece[m.end():] if m else piece, 10 ** 7)
        if bad:
            return None, None, bad
        out.extend((k - 1, a, b) for a, b in r)
    return [], out, None


def _form_num(name, default, lo, hi, cast=float):
    try:
        v = cast(float(request.form.get(name, default)))
    except (TypeError, ValueError):
        v = cast(default)
    return max(lo, min(hi, v))


def _num_or(v, default):
    try:
        v = float(v)
        return v if v == v else default
    except (TypeError, ValueError):
        return default


def _resolve_ending_sfx(opts):
    """The ending's sound effect as {'path', 'name', 'gain'} (or None), or an error to show.
    It is a file picked from the SFX folder with Browse library: only a name that this
    server staged itself is accepted, never a path."""
    name = opts.get('ending_sfx_name')
    if opts.get('ending') != 'cliffhanger' or not name:
        return None, None
    path = pipeline.staged_path(name)
    ext = os.path.splitext(name)[1].lower().lstrip('.')
    if not name.startswith('net_') or ext not in pipeline.AUDIO_EXTENSIONS or not os.path.isfile(path):
        return None, 'The ending sound effect is no longer available -- pick it again.'
    return {'path': path, 'name': re.sub(r'^net_\d+_', '', name), 'gain': opts.get('ending_sfx_gain') or 0.0}, None


def _render_options(data):
    """reframe / speaker / subtitles / subtitle_size / ending / format / loudness from a request body --
    JSON for /render, form fields for /analyze's one-button path -- with
    anything unrecognised falling back to the default rather than failing."""
    reframe = data.get('reframe') if data.get('reframe') in ('auto', 'split', 'crop', 'fit') else 'auto'
    # Absent means "whatever this server defaults to", so a client that
    # predates the option, or a script, gets the configured behaviour.
    speaker = data.get('speaker', SHORTS_SPEAKER_CROP) in (True, 1, '1', 'true', 'on', 'yes')
    return {'reframe': reframe,
            'speaker': speaker and reframe != 'fit',
            'subtitles': data.get('subtitles', True) not in (False, 0, '0', 'false', 'off', None),
            'subtitle_size': data.get('subtitle_size') if data.get('subtitle_size') in sc.SUBTITLE_SIZES else 'm',
            # How each short finishes: on its last frame ('none'), or stopped
            # dead on its last beat, held, and cut to black ('cliffhanger').
            'ending': data.get('ending') if data.get('ending') in ('none', 'cliffhanger') else 'none',
            # The LAST short of the render finishes on a fade to black instead of that ending
            # (for the final episode of a series), over this many seconds.
            'fade_last': data.get('fade_last') in (True, 1, '1', 'true', 'on', 'yes'),
            'ending_fade': sc.clamp_fade(data.get('ending_fade')) or SHORTS_FADE,
            # ...for this long (seconds), and with this sound effect (a file picked from the
            # SFX folder, by its staged name) coming in as the action stops, this many dB up or down.
            'ending_hold': sc.clamp_hold(data.get('ending_hold')) or SHORTS_CLIFF_HOLD,
            'ending_sfx_name': os.path.basename(str(data.get('ending_sfx') or '').strip()),
            'ending_sfx_gain': max(-30.0, min(6.0, _num_or(data.get('ending_sfx_gain'), 0.0))),
            # What is handed over: MP4 unless one of the other formats is named.
            'format': data.get('format') if data.get('format') in SHORTS_FORMATS else 'mp4_high',
            # How loud: this tab's usual level unless another is asked for.
            'loudness': pipeline.resolve_loudness(data.get('loudness'), SHORTS_LOUDNESS)}


@app.route('/api/shorts/options')
@require_permission('vertical_shorts')
def api_shorts_options():
    """What the tab needs to draw its controls: the Ollama models to choose
    from, and what this server can actually do (which face detector it has,
    whether its ffmpeg can burn captions)."""
    prod = pipeline.load_production_defaults()
    names, error = [], None
    try:
        r = requests.get(f'{pipeline.OLLAMA_URL}/api/tags', timeout=10)
        names = [m['name'] for m in r.json().get('models', [])]
    except Exception as e:
        error = f'Could not reach Ollama: {e}'
    vision = []
    if names:
        with ThreadPoolExecutor(max_workers=min(8, len(names))) as ex:
            flags = list(ex.map(pipeline._model_supports_vision, names))
        vision = [n for n, ok in zip(names, flags) if ok]
    det = sc.FaceDetector(_face_model_path())
    return jsonify(ok=True, vision_models=vision, text_models=names, error=error,
                   default_vision_model=prod.get('vision_model'),
                   face_detector=det.kind, captions_available=_captions_available(),
                   face_model_download=(det.kind != 'yunet' and session.get('role') == 'admin'
                                        and hasattr(cv2, 'FaceDetectorYN')),
                   speaker_default=SHORTS_SPEAKER_CROP,
                   vision_frames=SHORTS_VISION_FRAMES, max_items=SHORTS_MAX_ITEMS,
                   auto_min_story=SHORTS_AUTO_MIN_STORY,
                   formats=[{'key': k, 'label': pipeline.EXPORT_FORMATS[k]['label'],
                             'ext': pipeline.EXPORT_FORMATS[k]['ext']} for k in SHORTS_FORMATS],
                   loudness=SHORTS_LOUDNESS, levels=pipeline.loudness_choices(SHORTS_LOUDNESS),
                   min_clip=SHORTS_MIN_CLIP, max_clip=SHORTS_MAX_CLIP,
                   ending_seconds=round(SHORTS_CLIFF_HOLD + sc.CLIFFHANGER['black'], 1),
                   ending_hold=SHORTS_CLIFF_HOLD, ending_hold_range=list(sc.CLIFF_HOLD_RANGE),
                   ending_black=sc.CLIFFHANGER['black'], ending_fade=SHORTS_FADE,
                   languages=[{'code': c, 'name': n} for c, n in pipeline.WHISPER_LANGUAGES], language=SHORTS_LANGUAGE,
                   ending_fade_range=list(sc.FADE_RANGE))


@app.route('/api/shorts/analyze', methods=['POST'])
@require_permission('vertical_shorts')
def api_shorts_analyze():
    if not _job_submit_limiter.allow(_client_ip()):
        return jsonify(error='Too many requests. Wait a few minutes and try again.'), 429
    # Checked before the source is resolved: resolving a direct upload
    # writes the whole file to disk, which is not worth doing for a request
    # that is about to be refused anyway.
    min_dur = _form_num('min_dur', 60, 5, 240)
    max_dur = _form_num('max_dur', 120, 10, SHORTS_MAX_CLIP)
    if max_dur < min_dur + 5:
        return jsonify(error='The maximum length must be at least 5 seconds more than the minimum.'), 400
    # Before the source too, for the same reason: shorts are filed under a
    # project, and without one there is nowhere for them to go.
    project = sp.load(SHORTS_PROJECTS_DIR, (request.form.get('project_id') or '').strip())
    if not project:
        return jsonify(error='Choose the project these shorts are for first, or create one under Projects.'), 400
    srcs, orig_name, err = _resolve_sources()
    if not srcs:
        return jsonify(error=err), 400
    leave_out, leave_parts, bad = _parse_leave_out(request.form.get('leave_out'), len(srcs))
    if bad:
        return jsonify(error='Leave out: ' + bad), 400
    path = srcs[0]['path']
    prod = pipeline.load_production_defaults()
    vision_model = (request.form.get('vision_model') or '').strip() or prod.get('vision_model') or 'qwen3-vl:8b'
    params = {
        'path': path, 'orig_name': orig_name, 'project_id': project['project_id'],
        'user_id': session.get('user_id'), 'username': session.get('username'),
        'min_dur': min_dur, 'max_dur': max_dur,
        'count': ('auto' if (request.form.get('count') or '').strip().lower() == 'auto'
                  else _form_num('count', 8, 1, SHORTS_MAX_ITEMS, int)),
        'vision_frames': _form_num('vision_frames', SHORTS_VISION_FRAMES, 10, 300, int),
        'language': pipeline.normalize_language(request.form.get('language')) or SHORTS_LANGUAGE,
        'stt_prompt': pipeline.normalize_stt_prompt(request.form.get('stt_prompt')),
        'focus': ' '.join((request.form.get('focus') or '').split())[:300] or None,
        'avoid': ' '.join((request.form.get('avoid') or '').split())[:300] or None,
        'leave_out': leave_out, 'leave_out_parts': leave_parts,
        'parts': srcs if len(srcs) > 1 else None,
        'vision_model': vision_model,
        # One model for both layers unless told otherwise: a vision model
        # reads text perfectly well, and keeping a single model loaded
        # avoids swapping two in and out of the same GPU mid-job.
        'story_model': (request.form.get('story_model') or '').strip() or vision_model,
    }
    if (request.form.get('auto_render') or '').strip().lower() in ('1', 'true', 'on', 'yes'):
        # "Generate without preview": the render options travel with the
        # request, since there is no review step to choose them at.
        params['render'] = _render_options(request.form)
        params['render']['ending_sfx'], sfx_err = _resolve_ending_sfx(params['render'])
        if sfx_err:
            return jsonify(error=sfx_err), 400
        jid = _start_job('auto', _run_auto, params, f'{orig_name} (vertical shorts: analysis + render)',
                         after=_settle_auto)
    else:
        jid = _start_job('analyze', _run_analysis, params, f'{orig_name} (vertical shorts: analysis)')
    return jsonify(job_id=jid)


@app.route('/api/shorts/progress/<job_id>')
@require_permission('vertical_shorts')
def api_shorts_progress(job_id):
    j = pipeline.job_get(job_id)
    if not j:
        return jsonify(error='Unknown job id'), 404
    if not pipeline._owns_or_admin(j.get('user_id')):
        return jsonify(error='That job belongs to a different account.'), 403
    created = j.pop('created', None)
    if created:
        j['elapsed'] = round(time.time() - created, 1)
    stages = STAGES_BY_KIND.get(_JOB_KINDS.get(job_id), ANALYZE_STAGES)
    j['stages'] = [{'percent': p, 'label': lbl} for p, lbl in stages]
    return jsonify(**j)


@app.route('/api/shorts/cancel/<job_id>', methods=['POST'])
@require_permission('vertical_shorts')
def api_shorts_cancel(job_id):
    j = pipeline.job_get(job_id)
    if not j:
        return jsonify(error='Unknown job id'), 404
    if not pipeline._owns_or_admin(j.get('user_id')):
        return jsonify(error='That job belongs to a different account.'), 403
    if not pipeline.job_cancel(job_id):
        return jsonify(error='Job already finished'), 409
    return jsonify(cancelled=True, job_id=job_id)


def _analysis_or_error(aid):
    a = analysis_get(aid)
    if not a:
        return None, (jsonify(ok=False, error='That analysis has expired. Run it again.'), 404)
    if not pipeline._owns_or_admin(a.get('user_id')):
        return None, (jsonify(ok=False, error='That analysis belongs to a different account.'), 403)
    return a, None


@app.route('/api/shorts/analysis/<aid>')
@require_permission('vertical_shorts')
def api_shorts_analysis(aid):
    a, err = _analysis_or_error(aid)
    if err:
        return err
    info = a['info']
    return jsonify(ok=True, analysis_id=aid, orig_name=a['orig_name'], duration=round(info['duration'], 2),
                   width=info['width'], height=info['height'], fps=round(info['fps'], 3),
                   candidates=a['candidates'], warnings=a['warnings'], stats=a['stats'], options=a['options'],
                   project_id=a.get('project_id'), created=a.get('created'),
                   review=a.get('review'), reviewed_at=a.get('reviewed_at'),
                   source_available=_sources_available(a),
                   refetchable=all(P.get('origin') for P in _parts(a)),
                   parts=[{'name': P.get('name'), 'offset': round(P['offset'], 3), 'duration': round(P['duration'], 3)}
                          for P in _parts(a)])


def _source_gone(a):
    """The answer to a request that needs the episode when it is no longer
    on the server: 410, and whether it can be fetched again."""
    again = all(P.get('origin') for P in _parts(a))
    what = 'The episode is' if not _is_multi(a) else 'Part of the episode is'
    if again:
        msg = (f'{what} no longer on the server (staged files are cleared after a while, and on a '
               'restart). Use "Fetch the episode again" above the list to bring it back from the network folder.')
    else:
        msg = 'The source video is no longer staged -- pick it again and re-analyse.'
    return jsonify(ok=False, error=msg, refetchable=again), 410


@app.route('/api/shorts/analysis/<aid>/thumb/<name>')
@require_permission('vertical_shorts')
def api_shorts_analysis_thumb(aid, name):
    d = _analysis_dir(aid)
    if not d or not re.match(r'^[A-Za-z0-9_-]{1,32}\.jpg$', name or ''):
        return jsonify(ok=False, error='Not found.'), 404
    a, err = _analysis_or_error(aid)
    if err:
        return err
    resp = send_from_directory(d, name, max_age=86400)
    return resp


_REVIEW_KEEP = ('hook', 'why', 'source', 'flags', 'story_score', 'visual_score', 'sound_score', 'pace', 'score',
                'text', 'lines', 'thumb', 'edges')


@app.route('/api/shorts/analysis/<aid>/review', methods=['POST'])
@require_permission('vertical_shorts')
def api_shorts_review(aid):
    """Saves the editor's review of an analysis as it stands -- which
    moments are ticked, their titles, in and out points, corrected captions
    and framing, and moments added by hand -- so it can be picked up again
    later, after a restart, or by someone else on the project."""
    a, err = _analysis_or_error(aid)
    if err:
        return err
    data = request.get_json(silent=True) or {}
    raw = data.get('items')
    if not isinstance(raw, list) or len(raw) > SHORTS_MAX_ITEMS * 2:
        return jsonify(ok=False, error='The review could not be read.'), 400
    duration = float(a['info']['duration'])
    found = {c['id']: c for c in a.get('candidates') or []}
    out = []
    for n, it in enumerate(raw, 1):
        if not isinstance(it, dict):
            return jsonify(ok=False, error=f'Moment {n} is not valid.'), 400
        cid = str(it.get('id') or '')[:24]
        title = ' '.join(str(it.get('title') or '').split())[:80]
        label = title or f'Moment {n}'
        try:
            start, end = float(it.get('start')), float(it.get('end'))
        except (TypeError, ValueError):
            return jsonify(ok=False, error=f'"{label}": start and end must be numbers (seconds).'), 400
        start, end = max(0.0, start), min(duration, end)
        if not end > start:
            return jsonify(ok=False, error=f'"{label}": out must be after in.'), 400
        captions, bad = _parse_captions(it.get('captions'), label, duration)
        if bad:
            return jsonify(ok=False, error=bad), 400
        framing, bad = _parse_framing(it.get('framing'), label, duration)
        if bad:
            return jsonify(ok=False, error=bad), 400
        base = found.get(cid)
        row = {k: base[k] for k in _REVIEW_KEEP if base and k in base} if base else \
            {'source': 'manual', 'flags': [], 'score': None}
        row.update({'id': cid or f'm{n}', 'title': title, 'start': round(start, 3), 'end': round(end, 3),
                    'duration': round(end - start, 2), 'keep': bool(it.get('keep'))})
        if captions:
            row['captions'] = captions
        if framing:
            row['framing'] = framing
        out.append(row)
    a['review'] = out
    a['reviewed_at'] = time.time()
    a['reviewed_by'] = session.get('username')
    try:
        analysis_save(aid, a)
    except OSError as e:
        return jsonify(ok=False, error=f'The review could not be saved ({e}).'), 500
    return jsonify(ok=True, saved=len(out), reviewed_at=a['reviewed_at'])


@app.route('/api/shorts/analyses')
@require_permission('vertical_shorts')
def api_shorts_analyses():
    """The analyses kept for a project (or every one this account can
    see), newest first: earlier cuts that can be opened and reviewed again."""
    pid = str(request.args.get('project_id') or '').strip() or None
    items = [x for x in analysis_summaries(pid) if pipeline._owns_or_admin(x.get('user_id'))]
    return jsonify(ok=True, items=items, keep_days=SHORTS_ANALYSIS_DAYS)


@app.route('/api/shorts/analysis/<aid>', methods=['DELETE'])
@require_permission('vertical_shorts')
def api_shorts_analysis_delete(aid):
    a, err = _analysis_or_error(aid)
    if err:
        return err
    with ANALYSES_LOCK:
        ANALYSES.pop(aid, None)
    shutil.rmtree(_analysis_dir(aid), ignore_errors=True)
    audit_log('shorts_analysis_delete', target=f"{a.get('orig_name')} ({aid})", user_id=session.get('user_id'),
              username=session.get('username'), ip=_client_ip())
    return jsonify(ok=True)


@app.route('/api/shorts/analysis/<aid>/refetch', methods=['POST'])
@require_permission('vertical_shorts')
def api_shorts_refetch(aid):
    """Brings the analysed episode back onto the server from the network
    folder it was first fetched from, after it was cleared."""
    a, err = _analysis_or_error(aid)
    if err:
        return err
    path, why = _ensure_source(a, aid)
    if not path:
        return jsonify(ok=False, error=why), 409
    return jsonify(ok=True, source_available=True)


@app.route('/api/shorts/clip', methods=['POST'])
@require_permission('vertical_shorts')
def api_shorts_clip():
    """A small, browser-playable proxy of one range of the source, in its
    original framing -- for judging the moment and its in/out points before
    spending a render on it. Its own endpoint rather than /api/scene/clip
    because that one is capped at 30 seconds (it previews single shots) and
    a short runs to several minutes."""
    data = request.get_json(silent=True) or request.form
    a, err = _analysis_or_error((data.get('analysis_id') or '').strip())
    if err:
        return err
    try:
        start = max(0.0, float(data.get('start', 0)))
        end = min(float(a['info']['duration']), float(data.get('end', 0)))
    except (TypeError, ValueError):
        return jsonify(ok=False, error='Invalid start/end time.'), 400
    if end - start < 0.5:
        return jsonify(ok=False, error='End time must be after start time.'), 400
    k, bad = _locate(a, start, end)
    if bad:
        return jsonify(ok=False, error=bad), 400
    a = _part_view(a, k)
    if not os.path.exists(a['path']):
        return _source_gone(a)
    start, end = max(0.0, start - a.get('_offset', 0.0)), end - a.get('_offset', 0.0)
    dur = min(SHORTS_MAX_CLIP, end - start)
    _touch(a['path'])
    key = f"{os.path.basename(a['path'])}_{start:.2f}_{dur:.2f}"
    out_name = 'shclip_' + hashlib.sha1(key.encode()).hexdigest()[:16] + '.mp4'
    out_path = os.path.join(app.config['UPLOAD_FOLDER'], out_name)
    if not os.path.exists(out_path):
        # Encoded under a private name and renamed into place only once
        # complete. The cached name is derived from the range, so a second
        # request for the same preview while the first is still encoding
        # would otherwise find the file "already there" and be handed half
        # an MP4 -- which the browser then caches as broken.
        part = os.path.join(app.config['UPLOAD_FOLDER'], f'shpart_{secrets.token_hex(6)}.mp4')
        try:
            # The same sound the shorts will have (see sc.take_to_stereo).
            take = a['info'].get('audio_take')
            sound = (['-filter_complex', sc.take_to_stereo(take) + '[a]', '-map', '0:v:0', '-map', '[a]']
                     if take else [])
            r = pipeline.run_ffmpeg([pipeline.FFMPEG, '-y', '-ss', f'{start:.3f}', '-i', a['path'],
                                     '-t', f'{dur:.3f}'] + sound +
                                    ['-c:v', 'libx264', '-preset', 'veryfast', '-crf', '26',
                                     '-vf', "scale='min(640,iw)':-2", '-pix_fmt', 'yuv420p',
                                     '-c:a', 'aac', '-b:a', '96k', '-ac', '2', '-movflags', '+faststart', part],
                                    timeout=pipeline.FFMPEG_TIMEOUT, label='shorts preview clip')
        except pipeline.MediaToolTimeout as e:
            if os.path.exists(part):
                os.remove(part)
            return jsonify(ok=False, error=f'Preview took too long: {e}'), 504
        if not (os.path.exists(part) and os.path.getsize(part) > 0):
            if os.path.exists(part):
                os.remove(part)
            return jsonify(ok=False, error=f'Could not make that preview. ffmpeg error: {r.stderr[-300:]}'), 502
        try:
            os.replace(part, out_path)
        except OSError:
            # Another request finished the same clip first (and on Windows
            # may already be serving it); theirs is identical, use it.
            try:
                os.remove(part)
            except OSError:
                pass
            if not os.path.exists(out_path):
                return jsonify(ok=False, error='Could not store that preview.'), 502
    return jsonify(ok=True, url=f'/uploads/{out_name}')


@app.route('/api/shorts/captions', methods=['POST'])
@require_permission('vertical_shorts')
def api_shorts_captions():
    """The captions a moment would be rendered with: the automatic ones for
    its in and out points at the chosen size, or -- given `edited`, captions
    already changed for an earlier in/out -- those carried over to this one.
    What the review list's caption editor opens on."""
    data = request.get_json(silent=True) or {}
    a, err = _analysis_or_error(str(data.get('analysis_id') or '').strip())
    if err:
        return err
    duration = float(a['info']['duration'])
    try:
        start, end = max(0.0, float(data.get('start'))), min(duration, float(data.get('end')))
    except (TypeError, ValueError):
        return jsonify(ok=False, error='Invalid start/end time.'), 400
    if end - start < SHORTS_MIN_CLIP:
        return jsonify(ok=False, error=f'Must be at least {int(SHORTS_MIN_CLIP)} seconds long and inside the video.'), 400
    captions, bad = _parse_captions(data.get('edited'), 'This moment', duration)
    if bad:
        return jsonify(ok=False, error=bad), 400
    size = data.get('subtitle_size') if data.get('subtitle_size') in sc.SUBTITLE_SIZES else 'm'
    k, bad = _locate(a, start, end)
    if bad:
        return jsonify(ok=False, error=bad), 400
    a = _part_view(a, k)
    off = a.get('_offset', 0.0)
    if captions:
        captions = dict(captions, start=max(0.0, captions['start'] - off), end=max(0.0, captions['end'] - off))
    # On whole frames, as the render will cut it.
    fps = a['info']['fps']
    pdur = float(a['info']['duration'])
    t0, t1 = round(max(0.0, start - off) * fps) / fps, min(pdur, round((end - off) * fps) / fps)
    try:
        cues, edited = _item_cues(a, {'captions': captions}, t0, t1, sc.SUBTITLE_SIZES[size][1])
    except ValueError as e:
        return jsonify(ok=False, error=str(e)), 400
    return jsonify(ok=True, start=round(t0 + off, 3), end=round(t1 + off, 3), cues=cues, edited=edited,
                   spoken=bool(a['words'] or a['segments']))


@app.route('/api/shorts/frame', methods=['POST'])
@require_permission('vertical_shorts')
def api_shorts_frame():
    """One still from the episode at a time on its timeline, for the caption
    editor to show a line over. Small: it is a backdrop, not a preview of
    the framing."""
    data = request.get_json(silent=True) or {}
    a, err = _analysis_or_error(str(data.get('analysis_id') or '').strip())
    if err:
        return err
    try:
        t = max(0.0, min(float(a['info']['duration']), float(data.get('t'))))
    except (TypeError, ValueError):
        return jsonify(ok=False, error='Invalid time.'), 400
    parts = _parts(a)
    P = parts[_part_index(parts, t)]
    if not P.get('path') or not os.path.exists(P['path']):
        return _source_gone(a)
    _touch(P['path'])
    got = _grab_frames(P['path'], [max(0.0, t - P['offset'])], float(P['info']['fps']), max_w=360)
    b64 = got[0][1] if got else None
    if not b64:
        return jsonify(ok=False, error='No picture at that time.'), 404
    return jsonify(ok=True, image='data:image/jpeg;base64,' + b64, width=int(P['info']['width']),
                   height=int(P['info']['height']))


def _moment_request(data):
    """(analysis, start frame, end frame, error) for a request about one
    moment of an analysis: the framing and preview routes."""
    a, err = _analysis_or_error(str(data.get('analysis_id') or '').strip())
    if err:
        return None, None, None, err
    try:
        start, end = max(0.0, float(data.get('start'))), min(float(a['info']['duration']), float(data.get('end')))
    except (TypeError, ValueError):
        return None, None, None, (jsonify(ok=False, error='Invalid start/end time.'), 400)
    if end - start < SHORTS_MIN_CLIP or end - start > SHORTS_MAX_CLIP + 0.05:
        return None, None, None, (jsonify(ok=False, error=f'Must be {int(SHORTS_MIN_CLIP)} to {int(SHORTS_MAX_CLIP)} '
                                                          'seconds long and inside the video.'), 400)
    k, bad = _locate(a, start, end)
    if bad:
        return None, None, None, (jsonify(ok=False, error=bad), 400)
    a = _part_view(a, k)
    if not os.path.exists(a['path']):
        return None, None, None, _source_gone(a)
    info = a['info']
    start, end = max(0.0, start - a.get('_offset', 0.0)), end - a.get('_offset', 0.0)
    fps = info['fps']
    start_f = int(round(start * fps))
    end_f = min(info['frames'], int(round(end * fps)))
    _touch(a['path'])
    return a, start_f, end_f, None


def _cached_plan(a, start_f, end_f, reframe, speaker):
    """The automatic plan for a moment (no corrections), worked out once per
    analysis for each in/out and reframe setting: sampling the faces is the
    slow part, and the editor moves between the shot list and the preview."""
    key = (a.get('_part', 0), start_f, end_f, reframe, bool(speaker))
    plans = a.setdefault('_plans', {})
    if key not in plans:
        info = a['info']
        crop_w, _ = sc.crop_geometry(info['disp_w'], info['disp_h'])
        detector = None
        if reframe != 'fit' and info['disp_w'] > crop_w:
            detector = sc.FaceDetector(_face_model_path())
            detector = detector if detector.available() else None
        speaker = bool(speaker) and detector is not None and bool(a['words'] or a['segments'])
        if len(plans) > 40:
            plans.pop(next(iter(plans)))
        alt = []
        planned = _plan_moment(a, a['path'], info, start_f, end_f, reframe, speaker, detector, splits_out=alt)
        plans[key] = planned + (detector.kind if detector else None, alt)
    return plans[key]


def _shot_kind(segs):
    """How the plan frames one shot, in a word, and the window position to
    start a correction from."""
    layouts = {s['layout'] for s in segs}
    if 'split' in layouts:
        return 'split', None
    if 'fit' in layouts and len(layouts) == 1:
        return 'fit', None
    crops = [s for s in segs if s['layout'] == 'crop']
    if any(s.get('speaker') for s in crops):
        kind = 'speaker'
    elif any(s.get('keys') for s in crops):
        kind = 'follow'
    else:
        kind = 'crop'
    xs = [x for s in crops for x in ([k[1] for k in s['keys']] if s.get('keys') else [s['x']]) if x is not None]
    return kind, (float(sorted(xs)[len(xs) // 2]) if xs else None)


@app.route('/api/shorts/framing', methods=['POST'])
@require_permission('vertical_shorts')
def api_shorts_framing():
    """A moment's shots, each with how it will be framed and a still of it,
    for the editor to check and correct before rendering."""
    data = request.get_json(silent=True) or {}
    a, start_f, end_f, err = _moment_request(data)
    if err:
        return err
    opts = _render_options(data)
    info = a['info']
    fps = info['fps']
    segs, shot_starts, kind, alt = _cached_plan(a, start_f, end_f, opts['reframe'], opts['speaker'])
    crop_w, _ = sc.crop_geometry(info['disp_w'], info['disp_h'])
    shots = []
    cap = cv2.VideoCapture(a['path'])
    try:
        for k, (sa, sb) in enumerate(sc.shot_bounds(shot_starts, end_f - start_f)):
            how, x = _shot_kind([s for s in segs if s['a'] <= sb and s['b'] >= sa])     # joined shots span several
            mid = start_f + (sa + sb) // 2
            name = 'shf_' + hashlib.sha1(('%s_%d' % (a['path'], mid)).encode()).hexdigest()[:16] + '.jpg'
            path = os.path.join(app.config['UPLOAD_FOLDER'], name)
            if not os.path.exists(path):
                cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
                ok, frame = cap.read()
                if ok:
                    h, w = frame.shape[:2]
                    # Square pixels, as the planner measures: x in the picture is x here.
                    frame = cv2.resize(frame, (384, max(2, int(round(384 * info['disp_h'] / float(info['disp_w']))))),
                                       interpolation=cv2.INTER_AREA)
                    cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            off = a.get('_offset', 0.0)
            sp = [t for t in alt if t['a'] <= sb and t['b'] >= sa]
            split = ({'size': list(sp[0]['size']), 'panes': [list(p) for p in sp[0]['panes']]} if sp else None)
            shots.append({'index': k, 'can_split': bool(sp), 'split': split, 'start': round((start_f + sa) / fps + off, 3),
                          'end': round((start_f + sb + 1) / fps + off, 3),
                          'at': round(mid / fps + off, 3), 'auto': how, 'x': None if x is None else round(x, 1),
                          'thumb': f'/uploads/{name}' if os.path.exists(path) else None})
    finally:
        cap.release()
    return jsonify(ok=True, shots=shots, disp_w=info['disp_w'], disp_h=info['disp_h'], crop_w=crop_w,
                   max_x=max(0, info['disp_w'] - crop_w), detector=kind,
                   start=round(start_f / fps + a.get('_offset', 0.0), 3),
                   end=round(end_f / fps + a.get('_offset', 0.0), 3))


@app.route('/api/shorts/vpreview', methods=['POST'])
@require_permission('vertical_shorts')
def api_shorts_vpreview():
    """A small 9:16 render of one moment as it will come out -- the framing,
    with any corrections, and the captions -- to check before spending a
    full render on it. No ending, and quickly encoded: it is for looking at."""
    data = request.get_json(silent=True) or {}
    a, start_f, end_f, err = _moment_request(data)
    if err:
        return err
    opts = _render_options(data)
    info = a['info']
    fps = info['fps']
    off = a.get('_offset', 0.0)
    # What arrives is on the analysis's timeline; the part works in its own.
    duration = float(info['duration']) + off
    framing, bad = _parse_framing(data.get('framing'), 'This moment', duration)
    if bad:
        return jsonify(ok=False, error=bad), 400
    captions, bad = _parse_captions(data.get('captions'), 'This moment', duration)
    if bad:
        return jsonify(ok=False, error=bad), 400
    local = _localize_item({'start': 0.0, 'end': 0.0, 'framing': framing, 'captions': captions}, off)
    framing, captions = local.get('framing'), local.get('captions')
    segs, shot_starts, _, alt = _cached_plan(a, start_f, end_f, opts['reframe'], opts['speaker'])
    crop_w, _ = sc.crop_geometry(info['disp_w'], info['disp_h'])
    n_frames = end_f - start_f
    item = {'framing': framing, 'captions': captions}
    segs = sc.apply_framing(segs, shot_starts, n_frames, _framing_overrides(item, start_f, fps),
                            max(0.0, float(info['disp_w'] - crop_w)), splits=alt,
                            disp=(info['disp_w'], info['disp_h']))
    burn = opts['subtitles'] and _captions_available()
    cues = []
    if burn:
        try:
            cues, _ = _item_cues(a, item, start_f / fps, end_f / fps, sc.SUBTITLE_SIZES[opts['subtitle_size']][1])
        except ValueError as e:
            return jsonify(ok=False, error=str(e)), 400
        cues = sc.place_cues(cues, segs, fps)
    key = json.dumps([a['path'], start_f, end_f, segs, cues, opts['subtitle_size']], sort_keys=True, default=str)
    out_name = 'shv_' + hashlib.sha1(key.encode()).hexdigest()[:16] + '.mp4'
    out_path = os.path.join(app.config['UPLOAD_FOLDER'], out_name)
    if not os.path.exists(out_path):
        work = app.config['UPLOAD_FOLDER']
        tag = secrets.token_hex(6)
        ass_name = f'shsub_pv_{tag}.ass' if cues else None
        part = os.path.join(work, f'shpart_{tag}.mp4')
        try:
            if ass_name:
                sc.write_ass(cues, os.path.join(work, ass_name), size=opts['subtitle_size'], font=SHORTS_SUB_FONT)
            ok, err = sc.render_short(pipeline.FFMPEG, a['path'], part, start_f, n_frames, info, segs,
                                      ass_name=ass_name, work_dir=work, crf=30, preset='veryfast',
                                      loudness=SHORTS_LOUDNESS, true_peak=SHORTS_TRUE_PEAK,
                                      timeout=pipeline.FFMPEG_TIMEOUT, out_w=360, out_h=640)
        except sc.ToolTimeout as e:
            ok, err = False, f'The preview took too long ({e}).'
        finally:
            if ass_name:
                try:
                    os.remove(os.path.join(work, ass_name))
                except OSError:
                    pass
        if not ok:
            if os.path.exists(part):
                os.remove(part)
            return jsonify(ok=False, error=f'Could not make the preview: {err}'), 502
        try:
            os.replace(part, out_path)
        except OSError:
            if os.path.exists(part):
                os.remove(part)
    return jsonify(ok=True, url=f'/uploads/{out_name}')


@app.route('/api/shorts/face-model', methods=['POST'])
def api_shorts_face_model():
    """Fetches the YuNet face model from the OpenCV model zoo and puts it
    where Vertical Shorts looks for it (models/, next to the app). Admins
    only: it writes into the app's folder. What arrives is used only if it
    is exactly the published file -- its SHA-256 is checked -- and OpenCV
    can load it."""
    if session.get('role') != 'admin':
        return jsonify(ok=False, error='Admin access required.'), 403
    try:
        r = requests.get(YUNET_URL, timeout=60, stream=True)
        r.raise_for_status()
        data = r.raw.read(YUNET_SIZE * 2 + 1, decode_content=True) if hasattr(r, 'raw') and r.raw else r.content
    except Exception as e:
        return jsonify(ok=False, error=f'Could not download the face model from GitHub ({e}). If this server has no '
                                       f'internet access, download it elsewhere from {YUNET_URL} and copy it to '
                                       f'{YUNET_FILE}.'), 502
    if hashlib.sha256(data).hexdigest() != YUNET_SHA256:
        return jsonify(ok=False, error='What was downloaded is not the published face model (its checksum does not '
                                       'match), so it was not used.'), 502
    os.makedirs(os.path.dirname(YUNET_FILE), exist_ok=True)
    tmp = YUNET_FILE + '.part'
    with open(tmp, 'wb') as f:
        f.write(data)
    os.replace(tmp, YUNET_FILE)
    det = sc.FaceDetector(_face_model_path())
    if det.kind != 'yunet':
        return jsonify(ok=False, error="The face model was saved but this server's OpenCV could not load it, so the "
                                       'built-in Haar cascades are still used. OpenCV 4.8 or newer is needed.'), 500
    audit_log('shorts_face_model', target=os.path.basename(YUNET_FILE), user_id=session.get('user_id'),
              username=session.get('username'), ip=_client_ip())
    return jsonify(ok=True, face_detector=det.kind, path=YUNET_FILE)


@app.route('/api/shorts/render', methods=['POST'])
@require_permission('vertical_shorts')
def api_shorts_render():
    if not _job_submit_limiter.allow(_client_ip()):
        return jsonify(error='Too many render requests. Wait a few minutes and try again.'), 429
    data = request.get_json(silent=True) or {}
    a, err = _analysis_or_error(str(data.get('analysis_id') or '').strip())
    if err:
        return err
    raw = data.get('items')
    if not isinstance(raw, list) or not raw:
        return jsonify(error='Select at least one moment to render.'), 400
    if len(raw) > SHORTS_MAX_ITEMS:
        return jsonify(error=f'At most {SHORTS_MAX_ITEMS} shorts per render.'), 400
    duration = float(a['info']['duration'])
    items = []
    for n, it in enumerate(raw, 1):
        if not isinstance(it, dict):
            return jsonify(error=f'Moment {n} is not valid.'), 400
        title = ' '.join(str(it.get('title') or '').split())[:80]
        label = title or f'Moment {n}'
        try:
            start, end = float(it.get('start')), float(it.get('end'))
        except (TypeError, ValueError):
            return jsonify(error=f'"{label}": start and end must be numbers (seconds).'), 400
        start, end = max(0.0, start), min(duration, end)
        if end - start < SHORTS_MIN_CLIP:
            return jsonify(error=f'"{label}": must be at least {int(SHORTS_MIN_CLIP)} seconds long and '
                                 'inside the video.'), 400
        if end - start > SHORTS_MAX_CLIP + 0.05:
            return jsonify(error=f'"{label}": is {end - start:.0f}s long; the limit is '
                                 f'{int(SHORTS_MAX_CLIP)}s per short.'), 400
        _, bad = _locate(a, start, end)
        if bad:
            return jsonify(error=f'"{label}": {bad}'), 400
        captions, bad = _parse_captions(it.get('captions'), label, duration)
        if bad:
            return jsonify(error=bad), 400
        framing, bad = _parse_framing(it.get('framing'), label, duration)
        if bad:
            return jsonify(error=bad), 400
        items.append({'start': start, 'end': end, 'title': title, 'captions': captions, 'framing': framing})
    # Episode order (see _run_render), and an untitled one named for its place in it.
    items.sort(key=lambda it: (it['start'], it['end']))
    items = [dict(it, title=it['title'] or f'Short {k}') for k, it in enumerate(items, 1)]
    params = dict(_render_options(data), analysis=a, analysis_id=str(data.get('analysis_id')).strip(), items=items,
                  user_id=session.get('user_id'), username=session.get('username'))
    params['ending_sfx'], sfx_err = _resolve_ending_sfx(params)
    if sfx_err:
        return jsonify(error=sfx_err), 400
    jid = _start_job('render', _run_render, params, f"{a['orig_name']} (vertical shorts: {len(items)} to render)",
                     after=_settle_batch)
    return jsonify(job_id=jid)


@app.route('/api/shorts/batches')
@require_permission('vertical_shorts')
def api_shorts_batches():
    """Saved batches, newest first: ?project_id= for one project's (the whole
    team sees those), ?unfiled=1 for the ones not filed under any project
    (their maker and admins only), or neither for everything the caller may
    see."""
    want, unfiled = (request.args.get('project_id') or '').strip(), request.args.get('unfiled') in ('1', 'true')
    out = []
    for m in _all_manifests():
        filed = m.get('project_id') if m.get('project_id') and sp.load(SHORTS_PROJECTS_DIR, m['project_id']) else None
        if not m.get('shorts') or not _may_see(m):
            continue
        if (want and filed != want) or (unfiled and filed):
            continue
        out.append(_batch_public(m))
        if len(out) >= (200 if want else 50):
            break
    return jsonify(ok=True, items=out)


def _all_manifests():
    """Every readable batch manifest, newest first."""
    try:
        names = sorted((n for n in os.listdir(SHORTS_DIR) if _BATCH_ID.match(n)), reverse=True)
    except OSError:
        names = []
    for name in names:
        m = _load_manifest(name)
        if m:
            yield m


def _batch_or_error(bid, change=False):
    """(manifest, None), or (None, a 404) for a batch the caller may not
    see -- or, with change=True, may not delete or move."""
    m = _load_manifest(bid)
    if not m or not (_may_change(m) if change else _may_see(m)):
        return None, (jsonify(ok=False, error='Not found'), 404)
    return m, None


# --------------------------------------------------------------------------
# Projects
# --------------------------------------------------------------------------

def _project_public(proj, counts=None):
    pid = proj['project_id']
    c = (counts or {}).get(pid) or {'batches': 0, 'shorts': 0, 'last': None, 'poster': None}
    own = f"/api/shorts/projects/{pid}/thumb?v={int(proj.get('updated') or 0)}" if proj.get('thumb') else None
    return {'project_id': pid, 'title': proj.get('title'), 'episode': proj.get('episode') or '',
            'air_date': proj.get('air_date'), 'description': proj.get('description') or '',
            'name': sp.display_name(proj), 'created': proj.get('created'), 'updated': proj.get('updated'),
            'username': proj.get('username'), 'has_thumb': bool(proj.get('thumb')),
            # Its own picture, or until it has one, a frame of its newest short.
            'thumb_url': own or c['poster'],
            'batches': c['batches'], 'shorts': c['shorts'], 'last_activity': c['last'] or proj.get('created'),
            'can_delete': _may_change(proj)}


def _project_counts():
    """{project id: {'batches', 'shorts', 'last', 'poster'}} from the batches on disk."""
    counts = {}
    for m in _all_manifests():
        pid = m.get('project_id')
        if not pid or not m.get('shorts'):
            continue
        c = counts.setdefault(pid, {'batches': 0, 'shorts': 0, 'last': None, 'poster': None})
        c['batches'] += 1
        c['shorts'] += len(m['shorts'])
        if c['last'] is None:                       # newest first, so the first seen is the latest
            c['last'] = m.get('created')
            first = min(m['shorts'], key=lambda s: float(s.get('start') or 0))
            if first.get('thumb'):
                c['poster'] = f"/api/shorts/file/{m['batch_id']}/{first['thumb']}"
    return counts


def _project_thumb_upload():
    """JPEG bytes for the picture sent with this request, or None if none was.

    Accepted whatever ALLOW_LOCAL_MEDIA_UPLOAD says: that policy is about
    files handed to ffmpeg, and this never is. It is a small image, capped
    in size, decoded once by OpenCV and stored only as PRISM's own
    re-encoding of it -- the same footing as the script images the promo
    generator accepts."""
    f = request.files.get('thumbnail')
    if f is None or not f.filename:
        return None
    data = f.stream.read(sp.THUMB_MAX_BYTES + 1)
    return sp.make_thumbnail(data, f.filename)


@app.route('/api/shorts/projects', methods=['GET', 'POST'])
@require_permission('vertical_shorts')
def api_shorts_projects():
    """The projects, most recently changed first (GET), or a new one (POST:
    title, episode, air_date, description and an optional thumbnail file).
    Everyone with access to Vertical Shorts sees and can add to all of them."""
    if request.method == 'GET':
        counts = _project_counts()
        return jsonify(ok=True, items=[_project_public(p, counts) for p in sp.list_all(SHORTS_PROJECTS_DIR)])
    try:
        fields = sp.clean_fields(request.form)
        thumb = _project_thumb_upload()
    except sp.ProjectError as e:
        return jsonify(ok=False, error=str(e)), 400
    proj = sp.create(SHORTS_PROJECTS_DIR, fields, user_id=session.get('user_id'), username=session.get('username'),
                     thumb_jpeg=thumb)
    audit_log('shorts_project_create', target=sp.display_name(proj),
              user_id=session.get('user_id'), username=session.get('username'), ip=_client_ip())
    return jsonify(ok=True, project=_project_public(proj))


@app.route('/api/shorts/projects/<pid>', methods=['GET', 'POST', 'DELETE'])
@require_permission('vertical_shorts')
def api_shorts_project(pid):
    proj = sp.load(SHORTS_PROJECTS_DIR, pid)
    if not proj:
        return jsonify(ok=False, error='That project no longer exists.'), 404
    if request.method == 'GET':
        return jsonify(ok=True, project=_project_public(proj, _project_counts()))
    if request.method == 'DELETE':
        if not _may_change(proj):
            return jsonify(ok=False, error='Only whoever created a project, or an admin, can delete it.'), 403
        held = _project_counts().get(pid)
        if held:
            return jsonify(ok=False, error=f"This project still holds {held['shorts']} short"
                           f"{'' if held['shorts'] == 1 else 's'}. Delete them, or move them to another project, "
                           'first: deleting a project does not delete shorts.'), 409
        sp.delete(SHORTS_PROJECTS_DIR, pid)
        audit_log('shorts_project_delete', target=sp.display_name(proj),
                  user_id=session.get('user_id'), username=session.get('username'), ip=_client_ip())
        return jsonify(ok=True)
    try:
        fields = sp.clean_fields(request.form)
        thumb = _project_thumb_upload()
    except sp.ProjectError as e:
        return jsonify(ok=False, error=str(e)), 400
    proj = sp.update(SHORTS_PROJECTS_DIR, pid, fields, thumb_jpeg=thumb,
                     remove_thumb=request.form.get('remove_thumbnail') in ('1', 'true', 'on'))
    return jsonify(ok=True, project=_project_public(proj, _project_counts()))


@app.route('/api/shorts/projects/<pid>/thumb')
@require_permission('vertical_shorts')
def api_shorts_project_thumb(pid):
    proj = sp.load(SHORTS_PROJECTS_DIR, pid)
    if not proj or not proj.get('thumb'):
        return jsonify(ok=False, error='Not found'), 404
    resp = send_from_directory(sp.project_dir(SHORTS_PROJECTS_DIR, pid), 'thumb.jpg', conditional=True)
    resp.headers['Cache-Control'] = 'private, max-age=3600'
    return resp


@app.route('/api/shorts/batches/<bid>/project', methods=['POST'])
@require_permission('vertical_shorts')
def api_shorts_batch_move(bid):
    """Files a batch under a project -- for shorts made before there were
    projects, or put in the wrong one. Its maker or an admin."""
    m, err = _batch_or_error(bid, change=True)
    if err:
        return err
    proj = sp.load(SHORTS_PROJECTS_DIR, (request.get_json(silent=True) or {}).get('project_id'))
    if not proj:
        return jsonify(ok=False, error='That project no longer exists -- pick another.'), 400
    if m.get('status') == 'rendering':
        return jsonify(ok=False, error='That batch is still rendering.'), 409
    m['project_id'] = proj['project_id']
    _write_manifest(_batch_dir(bid), m)
    return jsonify(ok=True, batch=_batch_public(m))


@app.route('/api/shorts/batches/<bid>', methods=['DELETE'])
@require_permission('vertical_shorts')
def api_shorts_batch_delete(bid):
    m, err = _batch_or_error(bid, change=True)
    if err:
        return err
    if m.get('status') == 'rendering':
        return jsonify(ok=False, error='That batch is still rendering. Cancel the job first.'), 409
    # Renamed aside first, then removed. The rename is the all-or-nothing
    # step: on Windows it fails while any file inside is open (one of the
    # shorts is playing or still downloading), and in that case nothing has
    # been touched and the batch stays listed, intact -- instead of the
    # manifest going and the media it described being left behind unlisted.
    tomb = os.path.join(SHORTS_DIR, f'.deleting_{bid}_{secrets.token_hex(2)}')
    try:
        os.rename(_batch_dir(bid), tomb)
    except OSError:
        return jsonify(ok=False, error='One of these shorts is in use (playing or downloading). '
                                       'Close it and try again.'), 409
    shutil.rmtree(tomb, ignore_errors=True)
    audit_log('shorts_delete', target=f"{m.get('orig_name')} ({len(m.get('shorts') or [])} shorts)",
              user_id=session.get('user_id'), username=session.get('username'), ip=_client_ip())
    return jsonify(ok=True)


@app.route('/api/shorts/file/<bid>/<name>')
@require_permission('vertical_shorts')
def api_shorts_file(bid, name):
    m, err = _batch_or_error(bid)
    if err:
        return err
    # Only names the manifest itself lists are served, so nothing else that
    # might sit in the folder (the manifest, a half-written file) is reachable.
    if name not in _batch_file_names(m):
        return jsonify(ok=False, error='Not found'), 404
    resp = send_from_directory(_batch_dir(bid), name, conditional=True,
                               as_attachment=request.args.get('download') in ('1', 'true'))
    resp.headers['Cache-Control'] = 'private, max-age=3600'
    return resp


@app.route('/api/shorts/batches/<bid>/zip')
@require_permission('vertical_shorts')
def api_shorts_batch_zip(bid):
    m, err = _batch_or_error(bid)
    if err:
        return err
    bdir = _batch_dir(bid)
    fd, tmp = tempfile.mkstemp(suffix='.zip', dir=app.config['UPLOAD_FOLDER'])
    os.close(fd)
    # Stored, not deflated: finished video doesn't compress, so deflating
    # would only spend CPU time making the download start later.
    with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_STORED) as z:
        for s in m.get('shorts') or []:
            for name in (_delivery_name(s), s.get('srt')):
                p = os.path.join(bdir, name) if name else None
                if p and os.path.isfile(p):
                    z.write(p, name)
    stem = sc.slugify(os.path.splitext(m.get('orig_name') or '')[0], 40) or 'shorts'
    return pipeline.send_temp_download(tmp, f'{stem}_shorts.zip', 'application/zip',
                                       cleanup=lambda: os.path.exists(tmp) and os.remove(tmp))


@app.route('/api/shorts/batches/<bid>/send', methods=['POST'])
@require_permission('vertical_shorts')
def api_shorts_batch_send(bid):
    """Copies a batch's delivery files (and optionally their .srt files) to one of the
    network destinations configured under Config > Network -- the same ones
    the promo generator delivers to. `files` narrows it to some of the
    shorts; with exactly one, `filename` names the copy at the destination
    (its .srt, if sent, takes the same name). The saved short keeps its own
    name either way."""
    m, err = _batch_or_error(bid)
    if err:
        return err
    data = request.get_json(silent=True) or {}
    dest = network_destination_get(data.get('destination_id')) if data.get('destination_id') else None
    if not dest:
        return jsonify(ok=False, error='That destination no longer exists -- pick another.'), 400
    if dest.get('delivery_kind') != 'video':
        return jsonify(ok=False, error=f'"{dest["name"]}" is set up for scene lists or edit packages, not '
                       'finished video. Pick a video destination.'), 400
    wanted = data.get('files')
    shorts = [s for s in m.get('shorts') or [] if not isinstance(wanted, list) or s['file'] in wanted]
    if not shorts:
        return jsonify(ok=False, error='Nothing to send.'), 400
    rename = None
    if ' '.join(str(data.get('filename') or '').split()):
        if len(shorts) != 1:
            return jsonify(ok=False, error='A new name can be given to one short at a time. Send the shorts '
                           'one by one to rename them, or send them all under their own names.'), 400
        rename, bad = pipeline.destination_filename(data.get('filename'), _delivery_name(shorts[0]))
        if bad:
            return jsonify(ok=False, error=bad), 400
    bdir, sent = _batch_dir(bid), []
    for s in shorts:
        # The delivery file: the MP4, or the ProRes made from it.
        for local in (_delivery_name(s), s.get('srt')) if data.get('include_srt') else (_delivery_name(s),):
            if not local:
                continue
            # A renamed short's captions go with it under the same name.
            name = (os.path.splitext(rename)[0] + os.path.splitext(local)[1]) if rename else local
            try:
                pipeline.send_file_to_network_destination(os.path.join(bdir, local), name, dest)
            except ValueError as e:
                return jsonify(ok=False, error=str(e), sent=sent), 502
            sent.append(name)
    audit_log('shorts_send_to_destination', target=f'{dest["name"]}: {len(sent)} file(s) from {m.get("orig_name")}'
              + (f' ({_delivery_name(shorts[0])} as {rename})'
                 if rename and rename != _delivery_name(shorts[0]) else ''),
              user_id=session.get('user_id'), username=session.get('username'), ip=_client_ip())
    return jsonify(ok=True, sent=sent, destination=dest['name'])


# --------------------------------------------------------------------------
# Captions of a saved short
# --------------------------------------------------------------------------

_RECAPTIONING = set()
_RECAPTION_LOCK = threading.Lock()
# Its own allowance, not the one analyses and renders share (12 in five
# minutes): correcting a batch means one small job per short, and an editor
# working through twenty of them is not a flood. What protects the server is
# the job gate these wait at like any other job.
_recaption_limiter = _RateLimiter(limit=_env_num('SHORTS_RECAPTION_RATE_LIMIT', 60, int), window=300)


def _short_entry(m, index):
    return next((s for s in m.get('shorts') or [] if s.get('index') == index), None)


def _recaption_state(m, s):
    """How this short's captions can be changed: (what, why).

    'srt'    they are not in the picture, so only the .srt changes: instant.
    'render' they are burned in and the short can be rendered again.
    'frozen' they are burned in and it cannot -- `why` says which of the two
             things a re-render needs is missing."""
    src = dict(m.get('source') or {}, **(s.get('source') or {}))      # a multi-part short names its own part's file
    # In the picture: this short has captions burned in, or its batch was
    # rendered with burn-in on and this one simply had nothing said in it
    # (so lines added now belong in the picture too).
    if not (s.get('captions') or src.get('burn')):
        return 'srt', None
    if not (s.get('plan') and src.get('path') and src.get('info')):
        return 'frozen', ('This short was rendered before captions could be changed afterwards. Its .srt can '
                          'still be corrected; to change the captions in the picture, render it again from '
                          'Create shorts.')
    if not os.path.exists(src['path']):
        return 'frozen', ('The episode this short was cut from is no longer on the server (staged files are '
                          'cleared after a while), so the picture cannot be rendered again. Its .srt can still '
                          'be corrected; to change the captions in the picture, analyse the episode again.')
    return 'render', None


def _srt_cues(path):
    """Cues read back from an .srt this module wrote, for a short saved
    before its cues were kept in the manifest."""
    try:
        with open(path, encoding='utf-8') as f:
            blocks = re.split(r'\n\s*\n', f.read().strip())
    except OSError:
        return []
    cues = []
    for b in blocks:
        lines = [ln for ln in b.splitlines() if ln.strip()]
        at = next((k for k, ln in enumerate(lines) if '-->' in ln), None)
        if at is None:
            continue
        m = re.findall(r'(\d+):(\d+):(\d+)[,.](\d+)', lines[at])
        if len(m) != 2:
            continue
        t = [int(h) * 3600 + int(mi) * 60 + int(sec) + int(ms.ljust(3, '0')[:3]) / 1000.0 for h, mi, sec, ms in m]
        text = ' '.join(' '.join(lines[at + 1:]).split())
        if text and t[1] > t[0]:
            cues.append({'start': round(t[0], 3), 'end': round(t[1], 3), 'text': text})
    return cues


def _saved_cues(bid, s):
    if isinstance(s.get('cues'), list):
        return [{'start': c['start'], 'end': c['end'], 'text': c['text']} for c in s['cues']]
    return _srt_cues(os.path.join(_batch_dir(bid), s['srt'])) if s.get('srt') else []


def _clip_seconds(s):
    """Length of the part of a short that has dialogue under it: the moment
    itself, not the hold and black a cliffhanger ending adds after it."""
    return max(0.0, float(s.get('end') or 0) - float(s.get('start') or 0)) or float(s.get('duration') or 0)


@app.route('/api/shorts/batches/<bid>/captions/<int:index>', methods=['GET', 'POST'])
@require_permission('vertical_shorts')
def api_shorts_batch_captions(bid, index):
    """One saved short's captions (GET), or new ones for it (POST: `cues`).

    Reading them is for anyone who can see the batch; changing them is for
    whoever rendered it and admins, like deleting it. Captions that are not
    burned in are saved at once, to the .srt. Burned-in ones mean rendering
    that short again -- a job, whose id comes back -- which is possible
    while the episode it was cut from is still on the server."""
    m, err = _batch_or_error(bid, change=request.method == 'POST')
    if err:
        return err
    s = _short_entry(m, index)
    if not s:
        return jsonify(ok=False, error='Not found'), 404
    how, why = _recaption_state(m, s)
    if request.method == 'GET':
        return jsonify(ok=True, cues=_saved_cues(bid, s), duration=round(_clip_seconds(s), 3),
                       burned=how != 'srt', how=how, why=why, can_change=_may_change(m),
                       title=s.get('title'))
    if m.get('status') == 'rendering':
        return jsonify(ok=False, error='That batch is still rendering.'), 409
    try:
        cues = sc.clean_cues((request.get_json(silent=True) or {}).get('cues'), _clip_seconds(s))
    except ValueError as e:
        return jsonify(ok=False, error=str(e)), 400
    bdir = _batch_dir(bid)
    if how == 'render':
        if not _recaption_limiter.allow(_client_ip()):
            return jsonify(ok=False, error='Too many caption changes at once. Wait a few minutes and try again.'), 429
        key = (bid, index)
        with _RECAPTION_LOCK:
            if key in _RECAPTIONING:
                return jsonify(ok=False, error='This short is already being rendered with new captions.'), 409
            _RECAPTIONING.add(key)
        params = {'batch_id': bid, 'index': index, 'cues': cues,
                  'user_id': session.get('user_id'), 'username': session.get('username')}
        jid = _start_job('recaption', _run_recaption, params,
                         f"{s.get('title') or m.get('orig_name')} (vertical shorts: new captions)",
                         after=lambda p: _RECAPTIONING.discard((p['batch_id'], p['index'])))
        return jsonify(ok=True, job_id=jid)
    # Not in the picture (or the picture cannot be redone): the .srt is the captions.
    _write_short_srt(bdir, s, cues)
    s['cues'], s['captions_edited'], s['rev'] = cues, True, int(time.time())
    _write_manifest(bdir, m)
    audit_log('shorts_captions_edit', target=f"{m.get('orig_name')}: {s.get('title')} (.srt)",
              user_id=session.get('user_id'), username=session.get('username'), ip=_client_ip())
    return jsonify(ok=True, batch=_batch_public(m), srt_only=how == 'frozen')


def _write_short_srt(bdir, s, cues):
    """The .srt beside a short, replaced with `cues` -- or removed when
    there are none left."""
    name = s.get('srt') or (os.path.splitext(s['file'])[0] + '.srt')
    path = os.path.join(bdir, name)
    if cues:
        sc.write_srt(cues, path)
        s['srt'] = name
    else:
        try:
            os.remove(path)
        except OSError:
            pass
        s['srt'] = None


def _run_recaption(jid, params):
    """Renders one saved short again with different captions, in place.

    The new files are made under other names and only swapped in once all of
    them exist, so a failure -- or a cancel -- leaves the short exactly as it
    was. The swap is the one step that can fail on its own: on Windows a
    file that is being played or downloaded cannot be replaced."""
    bid, index = params['batch_id'], params['index']
    bdir = _batch_dir(bid)
    m = _read_manifest(bdir) if bdir else None
    s = _short_entry(m, index) if m else None
    if not s:
        pipeline.job_set(jid, error='That short no longer exists.')
        return
    how, why = _recaption_state(m, s)
    if how != 'render':
        pipeline.job_set(jid, error=why or 'These captions are not burned in; nothing to render.')
        return
    src, o = dict(m['source'], **(s.get('source') or {})), m.get('options') or {}
    info, plan = src['info'], s['plan']
    _touch(src['path'])
    pipeline.job_set(jid, percent=5, step='Preparing')
    cues = sc.place_cues([dict(c) for c in params['cues']], plan['segs'], info['fps'])
    fmt = o.get('format') if o.get('format') in SHORTS_FORMATS else 'mp4_high'
    # How THIS short ends: the last of a batch may have faded where the others held or cut.
    kind = s.get('ending') or o.get('ending') or 'none'
    opts = {'burn': True, 'subtitle_size': o.get('subtitle_size') if o.get('subtitle_size') in sc.SUBTITLE_SIZES else 'm',
            'ending': kind == 'cliffhanger', 'format': fmt,
            'ending_hold': o.get('ending_hold'),
            'ending_fade': (s.get('ending_fade') or o.get('ending_fade')) if kind == 'fade' else None,
            'ending_sfx': _kept_sfx(bdir, o),
            'loudness': pipeline.resolve_loudness(o.get('loudness'), SHORTS_LOUDNESS)}
    stem = os.path.splitext(s['file'])[0]
    new_mp4 = os.path.join(bdir, f'.new_{jid}_{stem}.mp4')
    delivery = s.get('delivery') if fmt != 'mp4_high' else None
    new_delivery = os.path.join(bdir, f'.new_{jid}_{delivery}') if delivery else None
    made = [p for p in (new_mp4, new_delivery) if p]
    try:
        pipeline.job_set(jid, percent=10, step='Rendering with the new captions')
        ok, err = _encode_short(f'{jid}_re', src['path'], info, plan, cues, opts, new_mp4, new_delivery,
                                report=lambda: pipeline.job_set(jid, percent=75, step='Making the delivery file'))
        if not ok:
            pipeline.job_set(jid, error=f'The short could not be rendered with the new captions: {err}')
            return
        pipeline.job_set(jid, percent=94, step='Saving')
        try:
            os.replace(new_mp4, os.path.join(bdir, s['file']))
            if new_delivery:
                os.replace(new_delivery, os.path.join(bdir, delivery))
        except OSError:
            pipeline.job_set(jid, error='This short is in use (playing or downloading), so it could not be '
                                        'replaced. Close it and save the captions again.')
            return
        # Read again: another short of this batch may have been changed meanwhile.
        m = _read_manifest(bdir)
        s = _short_entry(m, index) if m else None
        if not s:
            pipeline.job_set(jid, error='That short no longer exists.')
            return
        _write_short_srt(bdir, s, cues)
        s['cues'], s['captions_edited'], s['rev'] = cues, True, int(time.time())
        s['captions'] = bool(cues)
        s['size'] = os.path.getsize(os.path.join(bdir, s['file']))
        if delivery:
            s['delivery_size'] = os.path.getsize(os.path.join(bdir, delivery))
        if s.get('thumb'):
            _poster(os.path.join(bdir, s['file']), os.path.join(bdir, s['thumb']), float(s.get('duration') or 0) * 0.3)
        _write_manifest(bdir, m)
        pipeline.job_set(jid, percent=100, step='Done', done=True, result={'batch': _batch_public(m)})
    finally:
        for p in made:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass


def settle_interrupted_batches():
    """Run once at import. A batch still marked 'rendering' now belonged to
    a process that no longer exists -- the same reasoning jobs_db_init()
    applies to jobs. Left alone it would be undeletable (delete refuses a
    rendering batch) and, with nothing finished in it, invisible while
    still taking up disk. Also clears folders a failed delete left behind."""
    try:
        names = os.listdir(SHORTS_DIR)
    except OSError:
        return
    for name in names:
        p = os.path.join(SHORTS_DIR, name)
        if not os.path.isdir(p):
            continue
        if name.startswith('.deleting_'):
            shutil.rmtree(p, ignore_errors=True)
        elif _BATCH_ID.match(name):
            _settle_dir(p)
            # Half-made replacements from a re-caption the last process never finished.
            for left in (os.listdir(p) if os.path.isdir(p) else []):
                if left.startswith('.new_'):
                    try:
                        os.remove(os.path.join(p, left))
                    except OSError:
                        pass


os.makedirs(SHORTS_DIR, exist_ok=True)
os.makedirs(SHORTS_PROJECTS_DIR, exist_ok=True)
settle_interrupted_batches()

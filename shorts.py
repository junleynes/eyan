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
import requests
from flask import request, jsonify, session, send_from_directory
from werkzeug.utils import secure_filename

from core import app, _job_submit_limiter, _client_ip
from library_db import LIBRARY_DIR, audit_log, network_destination_get
from auth import require_permission
import pipeline
import shorts_core as sc


def _env_num(name, default, cast=float):
    try:
        return cast(os.environ.get(name, default))
    except (TypeError, ValueError):
        return cast(default)


# Finished shorts live next to the trailer library (and its branding/
# subfolder), not in UPLOAD_FOLDER: that one is a fresh temp dir every start
# and is swept by age, so anything left there is gone after a restart.
SHORTS_DIR = os.path.abspath(os.environ.get('SHORTS_DIR') or os.path.join(LIBRARY_DIR, 'shorts'))

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
SHORTS_TRUE_PEAK = _env_num('SHORTS_TRUE_PEAK', -1.5)
SHORTS_SUB_FONT = os.environ.get('SHORTS_SUB_FONT', 'Arial')
SHORTS_FACE_MODEL = os.environ.get('SHORTS_FACE_MODEL', '')
# Whether "Follow the speaker" starts ticked in the tab. It is a per-render
# choice either way; this only sets where the checkbox begins. Off unless
# asked for: it is a judgement from mouth movement, and when it is wrong it
# crops out the person talking, which showing the whole frame never does.
SHORTS_SPEAKER_CROP = os.environ.get('SHORTS_SPEAKER_CROP', '').strip().lower() in ('1', 'true', 'yes', 'on')

SHORTS_MAX_ITEMS = 20      # shorts per render job
SHORTS_MIN_CLIP = 3.0      # seconds
SHORTS_MAX_CLIP = 300.0

ANALYZE_STAGES = [(2, 'Reading video'), (5, 'Detecting cuts'), (20, 'Rating frames'),
                  (46, 'Transcribing dialogue'), (60, 'Finding story beats'),
                  (88, 'Building candidates'), (100, 'Done')]
RENDER_STAGES = [(2, 'Preparing'), (5, 'Rendering shorts'), (100, 'Done')]
# "Generate without preview" is the two jobs above run back to back as one:
# the analysis fills the first AUTO_SPLIT percent of the bar and the render
# the rest. Its stage list is derived from theirs so the three can't drift.
AUTO_SPLIT = 55
AUTO_STAGES = ([(max(1, p * AUTO_SPLIT // 100), lbl) for p, lbl in ANALYZE_STAGES[:-1]]
               + [(AUTO_SPLIT + p * (100 - AUTO_SPLIT) // 100, lbl) for p, lbl in RENDER_STAGES[1:]])
STAGES_BY_KIND = {'analyze': ANALYZE_STAGES, 'render': RENDER_STAGES, 'auto': AUTO_STAGES}

# ---- Analyses awaiting review ----
# Same shape and lifetime as the promo generator's PREVIEWS: held in memory
# between the analyse job and the render job, dropped after PREVIEW_TTL. An
# analysis holds the whole transcript and cut list, which the render needs
# for captions and shot boundaries and the browser has no use for.
ANALYSES = {}
ANALYSES_LOCK = threading.Lock()
_JOB_KINDS = {}


def analysis_store(aid, data):
    with ANALYSES_LOCK:
        now = time.time()
        for k in [k for k, v in ANALYSES.items() if now - v.get('created', 0) > pipeline.PREVIEW_TTL]:
            ANALYSES.pop(k, None)
        data['created'] = now
        ANALYSES[aid] = data


def analysis_get(aid):
    with ANALYSES_LOCK:
        a = ANALYSES.get(aid)
        if a and time.time() - a.get('created', 0) > pipeline.PREVIEW_TTL:
            ANALYSES.pop(aid, None)
            return None
        return a


def _touch(path):
    """Resets a staged source's age so the upload sweeper (which reclaims
    by mtime) doesn't delete it out from under a review that is still in
    progress. Best-effort."""
    try:
        os.utime(path, None)
    except OSError:
        pass


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
    for s in m.get('shorts') or []:
        d = {k: s.get(k) for k in ('index', 'title', 'file', 'srt', 'start', 'end', 'duration', 'size',
                                   'layouts', 'captions')}
        d['url'] = f"/api/shorts/file/{bid}/{s['file']}"
        d['thumb_url'] = f"/api/shorts/file/{bid}/{s['thumb']}" if s.get('thumb') else None
        d['srt_url'] = f"/api/shorts/file/{bid}/{s['srt']}" if s.get('srt') else None
        shorts.append(d)
    return {'batch_id': bid, 'orig_name': m.get('orig_name'), 'created': m.get('created'),
            'username': m.get('username'), 'status': m.get('status'), 'options': m.get('options') or {},
            'shorts': shorts, 'errors': m.get('errors') or [], 'warnings': m.get('warnings') or []}


def _batch_file_names(m):
    names = set()
    for s in m.get('shorts') or []:
        for k in ('file', 'srt', 'thumb'):
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


def _candidate_thumbs(path, cands, aid, fps):
    cap = cv2.VideoCapture(path)
    try:
        for c in cands:
            c['thumb'] = None
            at = c['start'] + 0.35 * (c['end'] - c['start'])
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(at * fps)))
            ok, frame = cap.read()
            if not ok:
                continue
            h, w = frame.shape[:2]
            small = cv2.resize(frame, (360, max(2, int(h * 360 / float(max(w, 1))))),
                               interpolation=cv2.INTER_AREA)
            name = f"shc_{aid}_{c['id']}.jpg"
            if cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], name), small,
                           [cv2.IMWRITE_JPEG_QUALITY, 82]):
                c['thumb'] = f'/uploads/{name}'
    finally:
        cap.release()


def _run_analysis(jid, params):
    """Finds the candidate moments. Returns the analysis id, or None when it
    ended with an error (already reported)."""
    report = params.get('_report') or functools.partial(pipeline.job_set, jid)
    path = params['path']
    vision_model, story_model = params['vision_model'], params['story_model']
    min_dur, max_dur, count = params['min_dur'], params['max_dur'], params['count']

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

    report(percent=2, step='Reading video')
    _touch(path)        # a long analysis must not be the reason its own source ages out
    info = sc.probe_source(pipeline.FFPROBE, path)
    if not info['fps'] or info['frames'] <= 0 or not info['width']:
        report(error='This file could not be read as video. If it plays elsewhere, it may be in a '
                         'codec this server cannot decode -- try an H.264/ProRes copy.')
        return
    fps, duration = info['fps'], info['duration']
    if duration < min_dur + 5:
        report(error=f'This video is only {duration:.0f}s long -- too short to cut '
                         f'{int(min_dur)}-{int(max_dur)}s shorts from. Lower the minimum length or use a longer source.')
        return

    report(percent=5, step='Detecting cuts')
    prod = pipeline.load_production_defaults()
    scene_list = pipeline.detect_scenes(path, threshold=float(prod['scene_threshold']),
                                        min_scene_len_sec=float(prod['min_scene_len']),
                                        detector=prod['detector'],
                                        adaptive_threshold=float(prod['adaptive_threshold']))
    cut_frames = sorted({int(pipeline.tc_frames(s)) for s, _ in scene_list} | {0})
    shots = [(pipeline.tc_seconds(s), pipeline.tc_seconds(e)) for s, e in scene_list] or [(0.0, duration)]
    cuts = [f / fps for f in cut_frames] + [duration]

    # ---- Layer 1: how dramatic does it look ----
    times = sc.vision_sample_times(shots, duration, budget=params['vision_frames'])
    report(percent=20, step=f'Rating {len(times)} frames (AI vision)')
    items = [(t, b) for t, b in _grab_frames(path, times, fps) if b]
    visual, errors = [], []
    progress = {'done': 0}
    lock = threading.Lock()

    def _rate(item):
        t, b64 = item
        try:
            score, desc = sc.ask_vision(pipeline.OLLAMA_URL, vision_model, b64)
            if score is not None:
                with lock:
                    visual.append({'t': t, 'score': score, 'desc': desc})
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
    report(percent=46, step='Transcribing dialogue')
    words, segments, heard = pipeline.transcribe_video_detailed(path)
    if not heard.get('ok'):
        # A transcription that FAILED is not a programme without dialogue.
        # Carrying on would pick moments on picture alone and present them
        # like any others -- the outcome the up-front service check exists
        # to prevent, arrived at by a different road.
        report(error=f"Could not transcribe the dialogue: {heard.get('reason') or 'unknown error'}. "
                     'Moments are chosen from what is said as well as what is seen, so nothing was picked. '
                     'Fix the speech-to-text service (Config > Services) and try again.')
        return
    words, segments = sc.normalize_transcript(words, segments)

    warnings = []
    if heard.get('take'):
        # A master with its sound on separate tracks: the shorts take their
        # audio from where the dialogue was found, not from the first track.
        info['audio_take'] = heard['take']
    if heard.get('audio'):
        warnings.append(f"The dialogue was read from {heard['audio']} of this file's audio"
                        + ('; the shorts use the same audio.' if heard.get('take') else '.'))
    if errors:
        warnings.append(f'{len(errors)} of {len(items)} frames could not be rated by the vision model '
                        f'({errors[0][:120]}); the rest were used.')
    beats, chunks = [], []
    if segments:
        chunks = sc.chunk_segments(segments, SHORTS_STORY_CHUNK_SEC, SHORTS_STORY_OVERLAP_SEC)
        per_chunk = max(2, min(4, int(math.ceil(count * 1.5 / len(chunks))) + 1))
        failed, first_err = 0, None
        for ci, (lo, hi) in enumerate(chunks):
            report(percent=60 + int(26 * ci / len(chunks)),
                             step=f'Finding story beats (part {ci + 1}/{len(chunks)})')
            prompt = sc.build_story_prompt(segments, lo, hi, visual, min_dur, max_dur, per_chunk,
                                           params.get('focus'), params.get('avoid'))
            try:
                reply = sc.ask_story(pipeline.OLLAMA_URL, story_model, prompt, num_ctx=SHORTS_STORY_NUM_CTX)
            except Exception as e:
                failed += 1
                first_err = first_err or str(e)
                print(f'Vertical Shorts: story analysis failed on part {ci + 1}/{len(chunks)}: {e}')
                continue
            beats.extend(sc.parse_story_reply(reply, lo, hi))
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
        warnings.append(f"No dialogue was transcribed ({heard.get('reason') or 'nothing was heard'}), so these "
                        'were picked on visual intensity alone -- they are not checked for making sense as a '
                        'story.')

    if (params.get('focus') or params.get('avoid')) and beats and beats[0].get('source') in ('heuristic', 'visual'):
        # The two fallbacks rank on sound and picture; neither reads meaning,
        # so what the editor asked to feature or avoid had no say in them.
        warnings.append('What to feature and what to avoid are judged by the story model, which found nothing '
                        'here, so neither was applied to these moments.')

    report(percent=88, step='Building candidates')
    cands = sc.build_candidates(beats, segments, words, cuts, visual, duration,
                                min_dur=min_dur, max_dur=max_dur, limit=count, fps=fps)
    if not cands:
        report(error='No usable moments were found in this video. Try a wider length range, '
                         'or add your own ranges by hand after re-running with a different model.')
        return
    aid = secrets.token_hex(8)
    _candidate_thumbs(path, cands, aid, fps)
    analysis_store(aid, {
        'user_id': params.get('user_id'), 'username': params.get('username'),
        'path': path, 'orig_name': params['orig_name'], 'info': info, 'cut_frames': cut_frames,
        'words': words, 'segments': segments, 'candidates': cands, 'warnings': warnings,
        'options': {'min_dur': min_dur, 'max_dur': max_dur, 'count': count,
                    'focus': params.get('focus'), 'avoid': params.get('avoid')},
        'stats': {'shots': len(shots), 'frames_rated': len(visual), 'transcript_lines': len(segments),
                  'story_parts': len(chunks), 'vision_model': vision_model, 'story_model': story_model},
    })
    _touch(path)
    report(percent=100, step='Done', done=True,
           result={'analysis_id': aid, 'candidates': len(cands)})
    return aid


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


def _run_render(jid, params):
    report = params.get('_report') or functools.partial(pipeline.job_set, jid)
    a = params['analysis']
    src, info = a['path'], a['info']
    items, reframe = params['items'], params['reframe']
    fps = info['fps']
    if not os.path.exists(src):
        report(error='The source video is no longer staged on the server (staged files are '
                         'cleared after a while). Pick it again and re-run the analysis.')
        return
    _touch(src)
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
    crop_w, _ = sc.crop_geometry(info['disp_w'], info['disp_h'])
    if reframe != 'fit' and info['disp_w'] > crop_w:
        detector = sc.FaceDetector(_face_model_path())
        if not detector.available():
            detector = None
            warnings.append('No face detector is available in this OpenCV build, so shots were centre-cropped.')

    # Following the speaker needs faces to compare and words to time them
    # against; without either it is simply not applied, and says so.
    speaker = bool(params.get('speaker')) and detector is not None
    if speaker and not (a['words'] or a['segments']):
        speaker = False
        warnings.append('No dialogue was transcribed for this video, so "Follow the speaker" had nothing to go on '
                        'and was not applied.')

    bid = f'{int(time.time())}_{secrets.token_hex(3)}'
    bdir = os.path.join(SHORTS_DIR, bid)
    os.makedirs(bdir, exist_ok=True)
    params['_batch_dir'] = bdir
    stem = sc.slugify(os.path.splitext(a['orig_name'] or '')[0], 40) or 'video'
    manifest = {'batch_id': bid, 'created': time.time(), 'user_id': params.get('user_id'),
                'username': params.get('username'), 'orig_name': a['orig_name'], 'status': 'rendering',
                'options': {'reframe': reframe, 'subtitles': want_captions,
                            'subtitle_size': params['subtitle_size'],
                            'face_detector': detector.kind if detector else None,
                            'speaker': speaker},
                'shorts': [], 'errors': [], 'warnings': warnings}
    _write_manifest(bdir, manifest)

    work = app.config['UPLOAD_FOLDER']
    max_chars = sc.SUBTITLE_SIZES[params['subtitle_size']][1]
    total = len(items)
    for n, it in enumerate(items, 1):
        base = 5 + 93.0 * (n - 1) / total
        span = 93.0 / total
        start_f = int(round(it['start'] * fps))
        end_f = min(info['frames'], int(round(it['end'] * fps)))
        n_frames = end_f - start_f
        if n_frames < int(SHORTS_MIN_CLIP * fps):
            manifest['errors'].append({'index': n, 'title': it['title'],
                                       'error': 'Range is past the end of the video.'})
            continue

        report(percent=int(base), step=f'Short {n}/{total}: finding faces')
        samples = (sc.sample_faces(src, start_f, n_frames, fps, detector, sar=info['sar'], mouth=speaker)
                   if detector else [])
        shot_starts = [c - start_f for c in a['cut_frames'] if start_f < c < end_f]
        t0, t1 = start_f / fps, end_f / fps
        speech = ([(s - t0, e - t0) for s, e in sc.speech_units(a['words'], a['segments']) if e > t0 and s < t1]
                  if speaker else None)
        segs = sc.plan_reframe(samples, shot_starts, n_frames, info['disp_w'], info['disp_h'], crop_w,
                               mode=reframe, fps=fps, speaker=speaker, speech=speech)
        cues = sc.place_cues(
            sc.subtitle_cues(a['words'], a['segments'], start_f / fps, end_f / fps, max_chars=max_chars), segs, fps)

        name = f"{stem}_short_{n:02d}_{sc.slugify(it['title'], 40) or 'clip'}"
        out_path = os.path.join(bdir, name + '.mp4')
        # Bare [A-Za-z0-9_] name in ffmpeg's working directory: see render_short.
        ass_name = re.sub(r'[^A-Za-z0-9_.]', '_', f'shsub_{jid}_{n}.ass') if burn and cues else None
        try:
            if ass_name:
                sc.write_ass(cues, os.path.join(work, ass_name), size=params['subtitle_size'],
                             font=SHORTS_SUB_FONT)
            report(percent=int(base + span * 0.25), step=f'Short {n}/{total}: encoding')
            ok, err = sc.render_short(pipeline.FFMPEG, src, out_path, start_f, n_frames, info, segs,
                                      ass_name=ass_name, work_dir=work, crf=SHORTS_CRF, preset=SHORTS_PRESET,
                                      loudness=SHORTS_LOUDNESS, true_peak=SHORTS_TRUE_PEAK,
                                      timeout=pipeline.FFMPEG_LONG_TIMEOUT)
        except sc.ToolTimeout as e:
            ok, err = False, f'Encoding took too long and was stopped ({e}).'
        finally:
            if ass_name:
                try:
                    os.remove(os.path.join(work, ass_name))
                except OSError:
                    pass
        if not ok:
            print(f'Vertical Shorts: short {n}/{total} failed: {err}')
            manifest['errors'].append({'index': n, 'title': it['title'], 'error': err})
            _write_manifest(bdir, manifest)
            continue

        entry = {'index': n, 'title': it['title'], 'file': name + '.mp4', 'srt': None, 'thumb': None,
                 'start': round(start_f / fps, 3), 'end': round(end_f / fps, 3),
                 'duration': round(n_frames / fps, 2), 'size': os.path.getsize(out_path),
                 'layouts': {'crop': sum(1 for s in segs if s['layout'] == 'crop'),
                             'fit': sum(1 for s in segs if s['layout'] == 'fit'),
                             'tracked': sum(1 for s in segs if s.get('keys')),
                             'speaker': sum(1 for s in segs if s.get('speaker')),
                             'split': sum(1 for s in segs if s['layout'] == 'split')},
                 'captions': bool(ass_name)}
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
    render = dict(params['render'], analysis=a, warnings=list(a.get('warnings') or []),
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
    jid = pipeline.job_new(user_id=session.get('user_id'), username=session.get('username'))
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

def _resolve_source():
    """(path, display name, error). Same policy as the promo generator's
    load_video(): a direct upload only where the deployment allows it,
    otherwise a file already staged from a configured network share."""
    f = request.files.get('shorts_file')
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
    staged = (request.form.get('shorts_file_network') or '').strip()
    if staged:
        safe = os.path.basename(staged)
        path = os.path.join(app.config['UPLOAD_FOLDER'], safe)
        if safe.startswith('net_') and os.path.exists(path):
            return path, re.sub(r'^net_\d+_', '', safe), None
        return None, None, 'Selected network file is no longer available -- please re-select it'
    return None, None, 'No video provided'


def _form_num(name, default, lo, hi, cast=float):
    try:
        v = cast(float(request.form.get(name, default)))
    except (TypeError, ValueError):
        v = cast(default)
    return max(lo, min(hi, v))


def _render_options(data):
    """reframe / speaker / subtitles / subtitle_size from a request body --
    JSON for /render, form fields for /analyze's one-button path -- with
    anything unrecognised falling back to the default rather than failing."""
    reframe = data.get('reframe') if data.get('reframe') in ('auto', 'split', 'crop', 'fit') else 'auto'
    # Absent means "whatever this server defaults to", so a client that
    # predates the option, or a script, gets the configured behaviour.
    speaker = data.get('speaker', SHORTS_SPEAKER_CROP) in (True, 1, '1', 'true', 'on', 'yes')
    return {'reframe': reframe,
            'speaker': speaker and reframe != 'fit',
            'subtitles': data.get('subtitles', True) not in (False, 0, '0', 'false', 'off', None),
            'subtitle_size': data.get('subtitle_size') if data.get('subtitle_size') in sc.SUBTITLE_SIZES else 'm'}


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
                   speaker_default=SHORTS_SPEAKER_CROP,
                   vision_frames=SHORTS_VISION_FRAMES, max_items=SHORTS_MAX_ITEMS,
                   min_clip=SHORTS_MIN_CLIP, max_clip=SHORTS_MAX_CLIP)


@app.route('/api/shorts/analyze', methods=['POST'])
@require_permission('vertical_shorts')
def api_shorts_analyze():
    if not _job_submit_limiter.allow(_client_ip()):
        return jsonify(error='Too many requests. Wait a few minutes and try again.'), 429
    # Checked before the source is resolved: resolving a direct upload
    # writes the whole file to disk, which is not worth doing for a request
    # that is about to be refused anyway.
    min_dur = _form_num('min_dur', 30, 5, 240)
    max_dur = _form_num('max_dur', 90, 10, SHORTS_MAX_CLIP)
    if max_dur < min_dur + 5:
        return jsonify(error='The maximum length must be at least 5 seconds more than the minimum.'), 400
    path, orig_name, err = _resolve_source()
    if not path:
        return jsonify(error=err), 400
    prod = pipeline.load_production_defaults()
    vision_model = (request.form.get('vision_model') or '').strip() or prod.get('vision_model') or 'qwen3-vl:8b'
    params = {
        'path': path, 'orig_name': orig_name,
        'user_id': session.get('user_id'), 'username': session.get('username'),
        'min_dur': min_dur, 'max_dur': max_dur,
        'count': _form_num('count', 8, 1, SHORTS_MAX_ITEMS, int),
        'vision_frames': _form_num('vision_frames', SHORTS_VISION_FRAMES, 10, 300, int),
        'focus': ' '.join((request.form.get('focus') or '').split())[:300] or None,
        'avoid': ' '.join((request.form.get('avoid') or '').split())[:300] or None,
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
                   source_available=os.path.exists(a['path']))


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
    if not os.path.exists(a['path']):
        return jsonify(ok=False, error='The source video is no longer staged -- pick it again and re-analyse.'), 410
    try:
        start = max(0.0, float(data.get('start', 0)))
        end = min(float(a['info']['duration']), float(data.get('end', 0)))
    except (TypeError, ValueError):
        return jsonify(ok=False, error='Invalid start/end time.'), 400
    if end - start < 0.5:
        return jsonify(ok=False, error='End time must be after start time.'), 400
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
        title = ' '.join(str(it.get('title') or '').split())[:80] or f'Short {n}'
        try:
            start, end = float(it.get('start')), float(it.get('end'))
        except (TypeError, ValueError):
            return jsonify(error=f'"{title}": start and end must be numbers (seconds).'), 400
        start, end = max(0.0, start), min(duration, end)
        if end - start < SHORTS_MIN_CLIP:
            return jsonify(error=f'"{title}": must be at least {int(SHORTS_MIN_CLIP)} seconds long and '
                                 'inside the video.'), 400
        if end - start > SHORTS_MAX_CLIP + 0.05:
            return jsonify(error=f'"{title}": is {end - start:.0f}s long; the limit is '
                                 f'{int(SHORTS_MAX_CLIP)}s per short.'), 400
        items.append({'start': start, 'end': end, 'title': title})
    params = dict(_render_options(data), analysis=a, items=items,
                  user_id=session.get('user_id'), username=session.get('username'))
    jid = _start_job('render', _run_render, params, f"{a['orig_name']} (vertical shorts: {len(items)} to render)",
                     after=_settle_batch)
    return jsonify(job_id=jid)


@app.route('/api/shorts/batches')
@require_permission('vertical_shorts')
def api_shorts_batches():
    """Saved batches, newest first. A regular account sees its own; an
    admin sees everyone's -- the same rule as the trailer library."""
    out = []
    try:
        names = sorted((n for n in os.listdir(SHORTS_DIR) if _BATCH_ID.match(n)), reverse=True)
    except OSError:
        names = []
    for name in names:
        m = _load_manifest(name)
        if not m or not m.get('shorts') or not pipeline._owns_or_admin(m.get('user_id')):
            continue
        out.append(_batch_public(m))
        if len(out) >= 50:
            break
    return jsonify(ok=True, items=out)


def _batch_or_error(bid):
    m = _load_manifest(bid)
    if not m or not pipeline._owns_or_admin(m.get('user_id')):
        return None, (jsonify(ok=False, error='Not found'), 404)
    return m, None


@app.route('/api/shorts/batches/<bid>', methods=['DELETE'])
@require_permission('vertical_shorts')
def api_shorts_batch_delete(bid):
    m, err = _batch_or_error(bid)
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
    # Stored, not deflated: H.264 doesn't compress, so deflating would only
    # spend CPU time making the download start later.
    with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_STORED) as z:
        for s in m.get('shorts') or []:
            for k in ('file', 'srt'):
                p = os.path.join(bdir, s[k]) if s.get(k) else None
                if p and os.path.isfile(p):
                    z.write(p, s[k])
    stem = sc.slugify(os.path.splitext(m.get('orig_name') or '')[0], 40) or 'shorts'
    return pipeline.send_temp_download(tmp, f'{stem}_shorts.zip', 'application/zip',
                                       cleanup=lambda: os.path.exists(tmp) and os.remove(tmp))


@app.route('/api/shorts/batches/<bid>/send', methods=['POST'])
@require_permission('vertical_shorts')
def api_shorts_batch_send(bid):
    """Copies a batch's MP4s (and optionally their .srt files) to one of the
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
        rename, bad = pipeline.destination_filename(data.get('filename'), shorts[0]['file'])
        if bad:
            return jsonify(ok=False, error=bad), 400
    bdir, sent = _batch_dir(bid), []
    for s in shorts:
        for k in ('file', 'srt') if data.get('include_srt') else ('file',):
            if not s.get(k):
                continue
            # A renamed short's captions go with it under the same name.
            name = (os.path.splitext(rename)[0] + os.path.splitext(s[k])[1]) if rename else s[k]
            try:
                pipeline.send_file_to_network_destination(os.path.join(bdir, s[k]), name, dest)
            except ValueError as e:
                return jsonify(ok=False, error=str(e), sent=sent), 502
            sent.append(name)
    audit_log('shorts_send_to_destination', target=f'{dest["name"]}: {len(sent)} file(s) from {m.get("orig_name")}'
              + (f' ({shorts[0]["file"]} as {rename})' if rename and rename != shorts[0]['file'] else ''),
              user_id=session.get('user_id'), username=session.get('username'), ip=_client_ip())
    return jsonify(ok=True, sent=sent, destination=dest['name'])


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


os.makedirs(SHORTS_DIR, exist_ok=True)
settle_interrupted_batches()

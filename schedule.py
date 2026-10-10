"""Schedule Plug tab -- routes and job plumbing.

The logic lives in schedule_core.py and knows nothing about Flask. This
file is the thin layer that connects it to PRISM: a request picks artwork,
a length, a music bed and a few words about how it should move; one job
draws and encodes it; the result is kept under SCHEDULE_DIR and can be
played, downloaded, sent to a network destination or deleted.

Kept as its own module, the way shorts.py is: a different product from the
episodic promo (a still brought to life, not footage cut down), sharing
only infrastructure with it -- job tracking and the concurrency gate, the
staged-file convention, the delivery formats, the network destinations.
"""
import json
import os
import re
import secrets
import shutil
import threading
import time
import traceback

import cv2
from flask import request, jsonify, session, send_from_directory

from core import app, _job_submit_limiter, _client_ip
from library_db import LIBRARY_DIR, audit_log, network_destination_get
from auth import require_permission
import pipeline
import schedule_core as sk
import shorts_core


# Next to the trailer library, for the reason finished shorts are: the
# upload folder is a fresh temp dir every start and is swept by age.
SCHEDULE_DIR = os.path.abspath(os.environ.get('SCHEDULE_DIR') or os.path.join(LIBRARY_DIR, 'schedule_plugs'))
# Reading the animation prompt with the language model is a refinement, not
# a requirement: set to 0 to always use the built-in reading of the words.
SCHEDULE_USE_LLM = os.environ.get('SCHEDULE_USE_LLM', '1').strip().lower() not in ('0', 'false', 'no', 'off')

STAGES = [(2, 'Reading artwork'), (10, 'Planning the animation'), (14, 'Drawing frames'),
          (82, 'Making the delivery file'), (96, 'Saving'), (100, 'Done')]

_PLUG_ID = re.compile(r'^\d{10,}_[0-9a-f]{6}$')
_JOBS = {}          # job id -> True, for jobs this module started (progress and cancel are scoped to them)


# --------------------------------------------------------------------------
# Saved plugs on disk: SCHEDULE_DIR/<id>/{plug.json, <name>.<ext>, preview.mp4, poster.jpg}
# --------------------------------------------------------------------------

def _plug_dir(pid):
    if not _PLUG_ID.match(str(pid or '')):
        return None
    return os.path.join(SCHEDULE_DIR, pid)


def _write_manifest(pdir, manifest):
    tmp = os.path.join(pdir, f'.plug.{secrets.token_hex(3)}.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=1)
    os.replace(tmp, os.path.join(pdir, 'plug.json'))


def _load_manifest(pid):
    pdir = _plug_dir(pid)
    if not pdir:
        return None
    try:
        with open(os.path.join(pdir, 'plug.json'), encoding='utf-8') as f:
            m = json.load(f)
        return m if isinstance(m, dict) and m.get('plug_id') == pid else None
    except (OSError, ValueError):
        return None


def _public(m):
    pid = m['plug_id']
    return {'plug_id': pid, 'title': m.get('title'), 'created': m.get('created'), 'username': m.get('username'),
            'duration': m.get('duration'), 'format': m.get('format'), 'format_label': m.get('format_label'),
            'loudness': m.get('loudness'),
            'size': m.get('size'), 'file': m.get('file'), 'layers': m.get('layers'), 'layered': m.get('layered'),
            'prompt': m.get('prompt'), 'animation': m.get('animation'), 'read_by': m.get('read_by'),
            'style': m.get('style'), 'style_label': m.get('style_label'),
            'roles': m.get('roles'), 'text_motion': m.get('text_motion'), 'layer_settings': m.get('layer_settings') or {},
            'page_of': m.get('page_of'), 'pages': m.get('pages') or [],
            'music': m.get('music'), 'notes': m.get('notes') or [],
            'url': f"/api/schedule/file/{pid}/{m['file']}",
            'preview_url': f"/api/schedule/file/{pid}/{m['preview']}" if m.get('preview') else None,
            'poster_url': f"/api/schedule/file/{pid}/{m['poster']}" if m.get('poster') else None}


# --------------------------------------------------------------------------
# The job
# --------------------------------------------------------------------------

def _read_prompt(prompt, layer_names, notes, style=None, content=None, use_llm=True):
    """(recipe, who read it). The chosen style is the starting point and the
    prompt changes what it names. The built-in reading always runs; the
    language model, when it is reachable, gets to refine it. Nothing here
    can fail the job: a plug in the plain style is still a plug."""
    base = sk.parse_prompt(prompt, layer_names, base=sk.style_recipe(style, content))
    if not (prompt and SCHEDULE_USE_LLM and use_llm):
        return base, 'built-in'
    prod = pipeline.load_production_defaults()
    model = prod.get('vision_model') or 'qwen3-vl:8b'
    try:
        if pipeline._check_service('ollama', pipeline.OLLAMA_URL, '/api/tags')['status'] != 'up':
            raise RuntimeError('Ollama is not reachable')
        reply, _ = shorts_core.ollama_generate(pipeline.OLLAMA_URL, {
            'model': model, 'prompt': sk.recipe_prompt(prompt, layer_names, base), 'stream': False, 'format': 'json',
            'options': {'temperature': 0.1, 'num_predict': 400}}, timeout=90)
        recipe = sk.parse_recipe_reply(reply, layer_names, base)
        if recipe is None:
            raise RuntimeError('its reply was not usable')
        if prod.get('unload_vision_after_scoring', True):
            pipeline.unload_ollama_model(model)
        return recipe, f'AI model ({model})'
    except Exception as e:
        print(f'Schedule Plug: animation prompt read without the language model ({e}).')
        notes.append('The animation prompt was read by keyword matching, because the AI model could not be used '
                     f'({str(e)[:120]}). Simple instructions (wipe, slide, fade, pop, stop motion, float, grow, rotate left, a direction) '
                     'work the same either way.')
        return base, 'built-in'


def _plan(art, params, duration, use_llm=True):
    """Everything decided before a frame is drawn: the recipe (style, then the
    words, then what was set on single layers), the timeline, and what to tell
    the editor. Shared by the real render and the quick preview."""
    names = [ly['name'] for ly in art['layers']]
    notes = list(art['notes'])
    roles = [ly['role'] for ly in art['layers']]
    recipe, read_by = _read_prompt(params.get('prompt'), names, notes, params.get('style'), params.get('content'),
                                   use_llm=use_llm)
    recipe = sk.apply_layer_animation(recipe, art['layers'])     # what was set on single layers wins
    recipe, cannot = sk.fit_to_artwork(recipe, names)
    if cannot:
        notes.append(cannot)
    notes.extend(sk.role_notes(roles, recipe))
    pages = [ly['page'] for ly in art['layers']]
    timeline = sk.build_timeline(recipe, names, duration, roles, pages)
    animation = sk.describe_recipe(recipe, names, roles)
    plan = sk.page_plan(timeline, pages, duration)
    if plan and plan['count'] > 1:
        animation += f"; {plan['count']} pages, about {plan['slot']:.0f} s each"
        notes.extend(sk.page_notes(plan, duration))
    return {'names': names, 'notes': notes, 'roles': roles, 'pages': pages, 'recipe': recipe, 'read_by': read_by,
            'timeline': timeline, 'animation': animation}


def _run(jid, params):
    report = lambda **kw: pipeline.job_set(jid, **kw)        # noqa: E731
    src, duration, fmt = params['image'], params['duration'], params['format']
    report(percent=2, step='Reading artwork')
    try:
        art = sk.load_artwork(src, overrides=params.get('overrides'))
    except sk.ArtworkError as e:
        report(error=str(e))
        return
    report(percent=10, step='Planning the animation')
    plan = _plan(art, params, duration)
    names, notes, roles, pages = plan['names'], plan['notes'], plan['roles'], plan['pages']
    recipe, read_by, timeline, animation = plan['recipe'], plan['read_by'], plan['timeline'], plan['animation']

    pid = f'{int(time.time())}_{secrets.token_hex(3)}'
    pdir = os.path.join(SCHEDULE_DIR, pid)
    os.makedirs(pdir, exist_ok=True)
    params['_plug_dir'] = pdir
    work = app.config['UPLOAD_FOLDER']
    master = os.path.join(work, f'schedmaster_{jid}.mov')
    prod = pipeline.load_production_defaults()
    loudness = pipeline.resolve_loudness(params.get('loudness'), prod.get('target_loudness', -14.0))
    fps = sk.fps_for(fmt)

    def drawing(done, total):
        report(percent=14 + int(68 * done / max(total, 1)), step=f'Drawing frames ({done}/{total})')

    try:
        animator = sk.Animator(art, timeline, recipe, duration)
        try:
            sk.encode(animator, master, fps, duration, ffmpeg=pipeline.FFMPEG, music_path=params.get('music'),
                      loudness=loudness, true_peak=float(prod.get('true_peak', -1.5)),
                      progress=drawing, timeout=pipeline.FFMPEG_LONG_TIMEOUT)
        except sk.EncodeError as e:
            report(error=f'The plug could not be encoded: {e}')
            return

        report(percent=82, step='Making the delivery file')
        ext = pipeline.EXPORT_FORMATS[fmt]['ext']
        stem = shorts_core.slugify(os.path.splitext(params['orig_name'] or '')[0], 40) or 'schedule'
        name = f'{stem}_schedule_{int(duration)}s.{ext}'
        out = os.path.join(pdir, name)
        r = pipeline.run_ffmpeg(pipeline.build_export_cmd(master, out, fmt), timeout=pipeline.FFMPEG_LONG_TIMEOUT,
                                label='schedule plug export')
        if not (os.path.exists(out) and os.path.getsize(out) > 0):
            report(error='The delivery file could not be made: '
                         + (shorts_core.ffmpeg_error(getattr(r, 'stderr', '')) or 'ffmpeg produced no output'))
            return

        report(percent=96, step='Saving')
        preview = None
        if ext != 'mp4':
            # ProRes and AVC-Intra don't play in a browser; a light H.264
            # copy stands in for them in the player. The delivery file is
            # what Download and Send hand over.
            preview = 'preview.mp4'
            pipeline.run_ffmpeg([pipeline.FFMPEG, '-y', '-i', master, '-vf', 'scale=1280:-2', '-c:v', 'libx264',
                                 '-preset', 'veryfast', '-crf', '23', '-pix_fmt', 'yuv420p', '-c:a', 'aac',
                                 '-b:a', '160k', '-movflags', '+faststart', os.path.join(pdir, preview)],
                                timeout=pipeline.FFMPEG_TIMEOUT, label='schedule plug preview')
            if not os.path.exists(os.path.join(pdir, preview)):
                preview = None
        poster = 'poster.jpg'
        # The artwork as designed, not a frame: the moving layers are never all
        # at rest at once. With pages, the first: all of them at once is no page.
        cover = [ly for ly in art['layers'] if ly['page'] in (None, 0)]
        if not cv2.imwrite(os.path.join(pdir, poster), cv2.resize(sk._flatten(cover, sk.CANVAS), (640, 360),
                                                                  interpolation=cv2.INTER_AREA)):
            poster = None
        manifest = {'plug_id': pid, 'created': time.time(), 'user_id': params.get('user_id'),
                    'username': params.get('username'), 'title': params['orig_name'], 'duration': duration,
                    'format': fmt, 'format_label': pipeline.EXPORT_FORMATS[fmt]['label'], 'file': name,
                    'size': os.path.getsize(out), 'preview': preview, 'poster': poster,
                    'layers': names, 'layered': art['layered'], 'prompt': params.get('prompt') or '',
                    'style': params.get('style'), 'style_label': sk.style_label(params.get('style')),
                    'roles': roles, 'text_motion': recipe.get('content'),
                    'page_of': pages, 'pages': art['pages'],
                    'animation': animation, 'read_by': read_by, 'music': params.get('music_name'),
                    'layer_settings': {ly['name']: {k: ly[k] for k in ('arrive', 'motion', 'tune') if ly.get(k)}
                                       for ly in art['layers'] if any(ly.get(k) for k in ('arrive', 'motion', 'tune'))},
                    'loudness': loudness if params.get('music') else None,
                    'fps': f'{fps[0]}/{fps[1]}', 'notes': notes}
        _write_manifest(pdir, manifest)
        report(percent=100, step='Done', done=True, result={'plug': _public(manifest)})
    finally:
        if os.path.exists(master):
            try:
                os.remove(master)
            except OSError:
                pass


def _settle(params):
    """After the job, however it ended: a folder without a manifest is a
    render that did not finish, and is removed."""
    pdir = params.get('_plug_dir')
    if pdir and os.path.isdir(pdir) and not os.path.exists(os.path.join(pdir, 'plug.json')):
        shutil.rmtree(pdir, ignore_errors=True)


def _guard(fn, after=None):
    """Same outcomes as every other PRISM job: a cancel, a media-tool
    timeout and an unexpected crash each end it with a message."""
    def runner(jid, params):
        try:
            fn(jid, params)
        except pipeline.JobCancelled:
            print(f'Schedule Plug job {jid} cancelled')
            pipeline.job_set(jid, error='Cancelled', status='cancelled')
        except pipeline.MediaToolTimeout as e:
            pipeline.job_set(jid, error=f'A media processing step timed out and was stopped ({e}).')
        except Exception as e:
            traceback.print_exc()
            pipeline.job_set(jid, error=f'Unexpected error: {e}')
        finally:
            if after:
                try:
                    after(params)
                except Exception as e:
                    print(f'Schedule Plug job {jid} cleanup error: {e}')
    return runner


def _spawn(fn, *args, **kwargs):
    """Its own function so the tests can run jobs inline."""
    threading.Thread(target=fn, args=args, kwargs=kwargs, daemon=True).start()


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

def _house_loudness():
    """Config > Production's loudness target: what a plug's music is
    levelled to unless another level is chosen for that plug."""
    return float(pipeline.load_production_defaults().get('target_loudness', -14.0))


@app.route('/api/schedule/options')
@require_permission('schedule_plug')
def api_schedule_options():
    psd = True
    try:
        import psd_tools  # noqa: F401
    except ImportError:
        psd = False
    return jsonify(ok=True, durations=list(sk.DURATIONS),
                   formats=[{'key': k, 'label': v['label']} for k, v in pipeline.EXPORT_FORMATS.items()],
                   loudness=_house_loudness(), levels=pipeline.loudness_choices(_house_loudness()),
                   styles=[{'key': k, 'label': label, 'arrives': v['effect'], 'then': v['ambient'],
                             'mood': 'Calm' if v['ambient'] in ('none', 'shine', 'float', 'breathe') else 'Lively'}
                            for k, label, v in sk.STYLES], default_style=sk.DEFAULT_STYLE,
                   motions=[{'key': k, 'label': sk.AMBIENT_LABELS[k]} for k in sk.MOTIONS], eases=list(sk.EASES),
                   tuning={k: list(v) for k, v in sk.TUNING.items()},
                   text_motions=[{'key': k, 'label': label} for k, label in sk.CONTENT_MODES],
                   default_text_motion=sk.DEFAULT_CONTENT,
                   effects=list(sk.EFFECTS), psd_supported=psd, max_layers=sk.MAX_LAYERS,
                   extensions=sorted(sk.IMAGE_EXTENSIONS if psd else sk.IMAGE_EXTENSIONS - {'psd', 'psb'}))


_TOKEN = re.compile(r'^schedule_image_\d+_\d+\.[a-z0-9]{2,4}$')


def _artwork():
    """(path, name to show, token) for the artwork a request names: a file
    sent with it, a network file already staged, or the copy an earlier
    inspect left behind (`schedule_image_token`, a name this module's own
    uploads are given and nothing else)."""
    tok = (request.form.get('schedule_image_token') or '').strip()
    if tok and _TOKEN.match(tok) and os.path.splitext(tok)[1].lstrip('.').lower() in sk.IMAGE_EXTENSIONS:
        path = pipeline.staged_path(tok)
        if os.path.exists(path):
            name = os.path.basename((request.form.get('schedule_image_name') or '').strip()) or tok
            return path, name, tok
    path = pipeline._resolve_upload('schedule_image', sk.IMAGE_EXTENSIONS)
    if not path:
        return None, None, None
    base = os.path.basename(path)
    if base.startswith('net_'):
        return path, re.sub(r'^net_\d+_', '', base), None
    f = request.files.get('schedule_image')
    return path, (os.path.basename(f.filename) if f is not None and f.filename else base), base


def _read_overrides():
    try:
        raw = json.loads(request.form.get('layer_overrides') or 'null')
    except ValueError:
        raw = None
    clean = sk.clean_overrides(raw) if raw else None
    return clean if clean and (clean['layers'] or clean['expand'] or clean['background_upto'] is not None) else None


def _read_names():
    """{path: layer name} as the editor saw them: what lets settings be kept by name."""
    try:
        raw = json.loads(request.form.get('layer_names') or 'null')
    except ValueError:
        return {}
    if not isinstance(raw, dict):
        return {}
    return {str(k)[:20]: str(v)[:120] for k, v in list(raw.items())[:200] if isinstance(v, (str, int))}


# --------------------------------------------------------------------------
# Looks: settings kept by layer name, shared by the team, so next week's file is ready sorted
# --------------------------------------------------------------------------

_PRESET_LOCK = threading.Lock()
MAX_PRESETS = 60


def _presets_path():
    return os.path.join(SCHEDULE_DIR, 'presets.json')


def _presets_read():
    try:
        with open(_presets_path(), encoding='utf-8') as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _presets_write(d):
    tmp = f'{_presets_path()}.{secrets.token_hex(3)}.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(d, f, indent=1)
    os.replace(tmp, _presets_path())


def _preset_public(p):
    return {'id': p['id'], 'name': p['name'], 'settings': p.get('settings') or {}, 'background': p.get('background'),
            'expand': p.get('expand') or [], 'layers': p.get('layers') or {}, 'username': p.get('username'),
            'created': p.get('created'), 'mine': p.get('user_id') == session.get('user_id')}


def _preset_visible(pid):
    p = _presets_read().get(pid)
    if not p:
        return None
    if str(pid).startswith('last_') and p.get('user_id') != session.get('user_id'):
        return None
    return p


def _remember_last(preset):
    """Keeps this person's latest settings as 'Same as last time'."""
    uid = session.get('user_id')
    if not preset or uid is None:
        return
    with _PRESET_LOCK:
        d = _presets_read()
        d[f'last_{uid}'] = dict(preset, id=f'last_{uid}', name='Same as last time', user_id=uid,
                               username=session.get('username'), created=time.time())
        _presets_write(d)


@app.route('/api/schedule/presets')
@require_permission('schedule_plug')
def api_schedule_presets():
    d = _presets_read()
    mine = d.get(f"last_{session.get('user_id')}")
    shared = sorted((p for k, p in d.items() if not str(k).startswith('last_')), key=lambda p: p['name'].lower())
    return jsonify(ok=True, presets=[_preset_public(p) for p in shared], last=_preset_public(mine) if mine else None)


@app.route('/api/schedule/presets', methods=['POST'])
@require_permission('schedule_plug')
def api_schedule_preset_save():
    """Saves a look. Either in its portable form (what an exported file holds), or as the
    editor's overrides with the layer names they refer to."""
    data = request.get_json(silent=True) or {}
    if isinstance(data.get('overrides'), dict):
        pre = sk.preset_from_overrides(data.get('name'), data.get('settings'), data['overrides'],
                                       {str(k): str(v) for k, v in (data.get('names') or {}).items()}
                                       if isinstance(data.get('names'), dict) else {})
    else:
        pre = sk.clean_preset(data)
    if not pre:
        return jsonify(ok=False, error='Give the look a name.'), 400
    with _PRESET_LOCK:
        d = _presets_read()
        same = next((k for k, p in d.items() if not k.startswith('last_') and p['name'].lower() == pre['name'].lower()), None)
        if same and not (d[same].get('user_id') == session.get('user_id') or session.get('role') == 'admin'):
            return jsonify(ok=False, error=f"A look called \"{d[same]['name']}\" already exists, made by {d[same].get('username')}. Use another name."), 409
        if not same and sum(1 for k in d if not k.startswith('last_')) >= MAX_PRESETS:
            return jsonify(ok=False, error='There are too many saved looks. Delete one you no longer use.'), 409
        pid = same or f'p{int(time.time())}{secrets.token_hex(3)}'
        d[pid] = dict(pre, id=pid, user_id=session.get('user_id'), username=session.get('username'), created=time.time())
        _presets_write(d)
    return jsonify(ok=True, preset=_preset_public(d[pid]))


@app.route('/api/schedule/presets/<pid>', methods=['DELETE'])
@require_permission('schedule_plug')
def api_schedule_preset_delete(pid):
    with _PRESET_LOCK:
        d = _presets_read()
        p = d.get(pid)
        if not p or pid.startswith('last_'):
            return jsonify(ok=False, error='Not found'), 404
        if not (p.get('user_id') == session.get('user_id') or session.get('role') == 'admin'):
            return jsonify(ok=False, error='That look was made by someone else.'), 403
        d.pop(pid)
        _presets_write(d)
    return jsonify(ok=True)


# --------------------------------------------------------------------------
# A quick look before the real render
# --------------------------------------------------------------------------

PREVIEW_SECONDS = 6
_PREVIEW_NAME = re.compile(r'^pv_\d{10,}_[0-9a-f]{6}\.mp4$')
_PREVIEW_LOCK = threading.Lock()


def _preview_dir():
    return os.path.join(SCHEDULE_DIR, '.previews')


def _sweep_previews(keep_seconds=3600):
    try:
        for n in os.listdir(_preview_dir()):
            p = os.path.join(_preview_dir(), n)
            if time.time() - os.path.getmtime(p) > keep_seconds:
                os.remove(p)
    except OSError:
        pass


@app.route('/api/schedule/preview', methods=['POST'])
@require_permission('schedule_plug')
def api_schedule_preview():
    """The first seconds of the plug, small and silent, in a few seconds' work: how it arrives and
    moves with these choices, before the full render. The animation words are read by keyword only,
    so a preview never waits for the language model."""
    if not _job_submit_limiter.allow(_client_ip()):
        return jsonify(error='Too many requests. Wait a few minutes and try again.'), 429
    path, name, token = _artwork()
    if not path:
        return jsonify(error='Pick the schedule artwork first, using Browse library.'), 400
    try:
        duration = int(float(request.form.get('duration') or sk.DURATIONS[0]))
    except ValueError:
        duration = sk.DURATIONS[0]
    if duration not in sk.DURATIONS:
        duration = sk.DURATIONS[0]
    style = (request.form.get('style') or '').strip() or sk.DEFAULT_STYLE
    content = (request.form.get('text_motion') or '').strip() or sk.DEFAULT_CONTENT
    if not sk.style_label(style) or content not in dict(sk.CONTENT_MODES):
        return jsonify(error='Choose one of the listed animation styles.'), 400
    params = {'style': style, 'content': content, 'prompt': ' '.join((request.form.get('prompt') or '').split())[:600]}
    if not _PREVIEW_LOCK.acquire(blocking=False):
        return jsonify(error='Another preview is being drawn. Try again in a moment.'), 429
    try:
        art = sk.load_artwork(path, overrides=_read_overrides())
        plan = _plan(art, params, duration, use_llm=False)
        os.makedirs(_preview_dir(), exist_ok=True)
        _sweep_previews()
        fname = f'pv_{int(time.time())}_{secrets.token_hex(3)}.mp4'
        animator = sk.Animator(art, plan['timeline'], plan['recipe'], duration)
        seconds = min(duration, max(PREVIEW_SECONDS, sk.settle_time(plan['timeline']) + 1.5))
        sk.encode_preview(animator, os.path.join(_preview_dir(), fname), seconds, ffmpeg=pipeline.FFMPEG)
    except sk.ArtworkError as e:
        return jsonify(error=str(e)), 422
    except sk.EncodeError as e:
        return jsonify(error=f'The preview could not be made: {e}'), 500
    except Exception as e:
        traceback.print_exc()
        return jsonify(error=f'The preview could not be made: {e}'), 500
    finally:
        _PREVIEW_LOCK.release()
    return jsonify(ok=True, url=f'/api/schedule/preview/{fname}', seconds=round(seconds, 1), duration=duration,
                   animation=plan['animation'], notes=plan['notes'])


@app.route('/api/schedule/preview/<fname>')
@require_permission('schedule_plug')
def api_schedule_preview_file(fname):
    if not _PREVIEW_NAME.match(fname):
        return jsonify(ok=False, error='Not found'), 404
    resp = send_from_directory(_preview_dir(), fname, conditional=True)
    resp.headers['Cache-Control'] = 'private, max-age=600'
    return resp


@app.route('/api/schedule/inspect', methods=['POST'])
@require_permission('schedule_plug')
def api_schedule_inspect():
    """Reads the artwork and says what the animation will do with each layer,
    so it can be checked, and corrected, before anything is drawn."""
    if not _job_submit_limiter.allow(_client_ip()):
        return jsonify(error='Too many requests. Wait a few minutes and try again.'), 429
    path, name, token = _artwork()
    if not path:
        return jsonify(error='Pick the schedule artwork first, using Browse library.'), 400
    preset, missing, applied = None, [], None
    try:
        pid = (request.form.get('preset_id') or '').strip()
        if pid:
            preset = _preset_visible(pid)
            if not preset:
                return jsonify(error='That saved look no longer exists.'), 404
            ov, missing = sk.overrides_from_preset(path, preset)
            applied = ov
        else:
            ov = _read_overrides()
        info = sk.inspect_artwork(path, overrides=ov)
    except sk.ArtworkError as e:
        return jsonify(error=str(e)), 422
    except Exception as e:
        traceback.print_exc()
        return jsonify(error=f'The artwork could not be read: {e}'), 500
    extra = {}
    if preset:
        extra = {'overrides': applied, 'settings': preset.get('settings') or {}, 'preset_missing': missing,
                 'preset_name': preset.get('name')}
    return jsonify(ok=True, name=name, token=token, **info, **extra)


@app.route('/api/schedule/render', methods=['POST'])
@require_permission('schedule_plug')
def api_schedule_render():
    if not _job_submit_limiter.allow(_client_ip()):
        return jsonify(error='Too many render requests. Wait a few minutes and try again.'), 429
    try:
        duration = int(float(request.form.get('duration') or 0))
    except ValueError:
        duration = 0
    if duration not in sk.DURATIONS:
        return jsonify(error='Choose a length of ' + ', '.join(str(d) for d in sk.DURATIONS) + ' seconds.'), 400
    fmt = pipeline.resolve_delivery_format(request.form.get('delivery_format'))
    style = (request.form.get('style') or '').strip() or sk.DEFAULT_STYLE
    if not sk.style_label(style):
        return jsonify(error='Choose one of the listed animation styles.'), 400
    content = (request.form.get('text_motion') or '').strip() or sk.DEFAULT_CONTENT
    if content not in dict(sk.CONTENT_MODES):
        return jsonify(error='Choose whether the logo and schedule text hold still or breathe.'), 400
    image, orig, _tok = _artwork()
    if not image:
        return jsonify(error='Pick the schedule artwork first, using Browse library.'), 400
    music = pipeline._resolve_upload('schedule_music', pipeline.AUDIO_EXTENSIONS)
    if not music and ((request.form.get('schedule_music_network') or '').strip()
                      or ('schedule_music' in request.files and request.files['schedule_music'].filename)):
        return jsonify(error='The selected music is no longer available -- please re-select it.'), 400

    def shown(path):
        return re.sub(r'^net_\d+_', '', os.path.basename(path))
    def original(field, path):
        # A staged network file keeps its own name after the net_<ts>_ prefix;
        # a direct upload is saved under a generated one, so ask the request.
        if os.path.basename(path).startswith('net_'):
            return shown(path)
        f = request.files.get(field)
        return os.path.basename(f.filename) if f is not None and f.filename else shown(path)
    params = {'image': image, 'orig_name': orig, 'music': music, 'overrides': _read_overrides(),
              'music_name': (original('schedule_music', music) if music else None),
              'duration': duration, 'format': fmt, 'style': style, 'content': content,
              'loudness': pipeline.resolve_loudness(request.form.get('loudness'), _house_loudness()),
              'prompt': ' '.join((request.form.get('prompt') or '').split())[:600],
              'user_id': session.get('user_id'), 'username': session.get('username')}
    look = (request.form.get('preset_id') or '').strip()
    if look == 'last':
        look = f"last_{session.get('user_id')}"
    if look:
        # A saved look is matched to this file by layer name, so the same one serves next week's file.
        pre = _preset_visible(look)
        if not pre:
            return jsonify(error='That saved look no longer exists.'), 404
        try:
            params['overrides'], _missing = sk.overrides_from_preset(image, pre)
        except sk.ArtworkError as e:
            return jsonify(error=str(e)), 422
    try:
        if not look:
            _remember_last(sk.preset_from_overrides('Same as last time', {
                'style': style, 'text_motion': content, 'prompt': params['prompt'], 'duration': duration, 'format': fmt},
                params['overrides'], _read_names()))
    except Exception as e:                                  # a convenience: never in the way of the render
        print(f'Schedule Plug: could not keep the last settings ({e})')
    jid = pipeline.job_new(user_id=session.get('user_id'), username=session.get('username'), kind='schedule')
    pipeline.job_set_orig_name(jid, f'{orig} (schedule plug, {duration}s)')
    if len(_JOBS) > 500:
        for k in list(_JOBS)[:250]:
            _JOBS.pop(k, None)
    _JOBS[jid] = True
    _spawn(pipeline.run_trailer_job_gated, jid, params, runner=_guard(_run, _settle))
    return jsonify(job_id=jid)


@app.route('/api/schedule/progress/<job_id>')
@require_permission('schedule_plug')
def api_schedule_progress(job_id):
    j = pipeline.job_get(job_id)
    if not j:
        return jsonify(error='Unknown job id'), 404
    if not pipeline._owns_or_admin(j.get('user_id')):
        return jsonify(error='That job belongs to a different account.'), 403
    created = j.pop('created', None)
    if created:
        j['elapsed'] = round(time.time() - created, 1)
    j['stages'] = [{'percent': p, 'label': lbl} for p, lbl in STAGES]
    return jsonify(**j)


@app.route('/api/schedule/cancel/<job_id>', methods=['POST'])
@require_permission('schedule_plug')
def api_schedule_cancel(job_id):
    j = pipeline.job_get(job_id)
    if not j:
        return jsonify(ok=False, error='Unknown job id'), 404
    if not pipeline._owns_or_admin(j.get('user_id')):
        return jsonify(ok=False, error='That job belongs to a different account.'), 403
    if not pipeline.job_cancel(job_id):
        return jsonify(ok=False, error='That job has already finished.'), 409
    return jsonify(ok=True)


@app.route('/api/schedule/items')
@require_permission('schedule_plug')
def api_schedule_items():
    """Saved plugs, newest first. A regular account sees its own; an admin
    sees everyone's -- the same rule as the trailer library."""
    out = []
    try:
        names = sorted((n for n in os.listdir(SCHEDULE_DIR) if _PLUG_ID.match(n)), reverse=True)
    except OSError:
        names = []
    for name in names:
        m = _load_manifest(name)
        if not m or not pipeline._owns_or_admin(m.get('user_id')):
            continue
        out.append(_public(m))
        if len(out) >= 60:
            break
    return jsonify(ok=True, items=out)


def _plug_or_error(pid):
    m = _load_manifest(pid)
    if not m or not pipeline._owns_or_admin(m.get('user_id')):
        return None, (jsonify(ok=False, error='Not found'), 404)
    return m, None


@app.route('/api/schedule/file/<pid>/<name>')
@require_permission('schedule_plug')
def api_schedule_file(pid, name):
    m, err = _plug_or_error(pid)
    if err:
        return err
    # Only the names the manifest lists are served.
    if name not in {m.get('file'), m.get('preview'), m.get('poster')} - {None}:
        return jsonify(ok=False, error='Not found'), 404
    resp = send_from_directory(_plug_dir(pid), name, conditional=True,
                               as_attachment=request.args.get('download') in ('1', 'true'))
    resp.headers['Cache-Control'] = 'private, max-age=3600'
    return resp


@app.route('/api/schedule/items/<pid>', methods=['DELETE'])
@require_permission('schedule_plug')
def api_schedule_delete(pid):
    m, err = _plug_or_error(pid)
    if err:
        return err
    # Renamed aside first, then removed: on Windows the rename fails while
    # the file is open (playing, downloading), and then nothing was touched.
    tomb = os.path.join(SCHEDULE_DIR, f'.deleting_{pid}_{secrets.token_hex(2)}')
    try:
        os.rename(_plug_dir(pid), tomb)
    except OSError:
        return jsonify(ok=False, error='This plug is in use (playing or downloading). Close it and try again.'), 409
    shutil.rmtree(tomb, ignore_errors=True)
    audit_log('schedule_plug_delete', target=str(m.get('file')), user_id=session.get('user_id'),
              username=session.get('username'), ip=_client_ip())
    return jsonify(ok=True)


@app.route('/api/schedule/items/<pid>/send', methods=['POST'])
@require_permission('schedule_plug')
def api_schedule_send(pid):
    """Copies the delivery file to one of the video destinations configured
    under Config > Network -- the same ones the promo generator delivers to
    -- under its own name, or under `filename` if one is given. The saved
    plug keeps its name either way: this names the copy, not the original."""
    m, err = _plug_or_error(pid)
    if err:
        return err
    data = request.get_json(silent=True) or {}
    dest = network_destination_get(data.get('destination_id')) if data.get('destination_id') else None
    if not dest:
        return jsonify(ok=False, error='That destination no longer exists -- pick another.'), 400
    if dest.get('delivery_kind') != 'video':
        return jsonify(ok=False, error=f'"{dest["name"]}" is set up for scene lists or edit packages, not '
                       'finished video. Pick a video destination.'), 400
    name, bad = pipeline.destination_filename(data.get('filename'), m['file'])
    if bad:
        return jsonify(ok=False, error=bad), 400
    try:
        pipeline.send_file_to_network_destination(os.path.join(_plug_dir(pid), m['file']), name, dest)
    except ValueError as e:
        return jsonify(ok=False, error=str(e)), 502
    audit_log('schedule_plug_send_to_destination',
              target=f'{dest["name"]}: {name}' + ('' if name == m['file'] else f' (saved as {m["file"]})'),
              user_id=session.get('user_id'), username=session.get('username'), ip=_client_ip())
    return jsonify(ok=True, sent=[name], destination=dest['name'])


def settle_interrupted():
    """Run once at import: folders left by a process that no longer exists
    (a render cut off before its manifest, a delete that did not finish)."""
    try:
        names = os.listdir(SCHEDULE_DIR)
    except OSError:
        return
    for name in names:
        p = os.path.join(SCHEDULE_DIR, name)
        if not os.path.isdir(p):
            continue
        if name.startswith('.deleting_') or (_PLUG_ID.match(name) and not os.path.exists(os.path.join(p, 'plug.json'))):
            shutil.rmtree(p, ignore_errors=True)


os.makedirs(SCHEDULE_DIR, exist_ok=True)
settle_interrupted()

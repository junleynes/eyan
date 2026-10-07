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
            'size': m.get('size'), 'file': m.get('file'), 'layers': m.get('layers'), 'layered': m.get('layered'),
            'prompt': m.get('prompt'), 'animation': m.get('animation'), 'read_by': m.get('read_by'),
            'style': m.get('style'), 'style_label': m.get('style_label'),
            'music': m.get('music'), 'notes': m.get('notes') or [],
            'url': f"/api/schedule/file/{pid}/{m['file']}",
            'preview_url': f"/api/schedule/file/{pid}/{m['preview']}" if m.get('preview') else None,
            'poster_url': f"/api/schedule/file/{pid}/{m['poster']}" if m.get('poster') else None}


# --------------------------------------------------------------------------
# The job
# --------------------------------------------------------------------------

def _read_prompt(prompt, layer_names, notes, style=None):
    """(recipe, who read it). The chosen style is the starting point and the
    prompt changes what it names. The built-in reading always runs; the
    language model, when it is reachable, gets to refine it. Nothing here
    can fail the job: a plug in the plain style is still a plug."""
    base = sk.parse_prompt(prompt, layer_names, base=sk.style_recipe(style))
    if not (prompt and SCHEDULE_USE_LLM):
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
                     f'({str(e)[:120]}). Simple instructions (wipe, slide, fade, pop, stop motion, float, a direction) '
                     'work the same either way.')
        return base, 'built-in'


def _run(jid, params):
    report = lambda **kw: pipeline.job_set(jid, **kw)        # noqa: E731
    src, duration, fmt = params['image'], params['duration'], params['format']
    report(percent=2, step='Reading artwork')
    try:
        art = sk.load_artwork(src)
    except sk.ArtworkError as e:
        report(error=str(e))
        return
    names = [ly['name'] for ly in art['layers']]
    notes = list(art['notes'])

    report(percent=10, step='Planning the animation')
    recipe, read_by = _read_prompt(params.get('prompt'), names, notes, params.get('style'))
    recipe, cannot = sk.fit_to_artwork(recipe, names)
    if cannot:
        notes.append(cannot)
    timeline = sk.build_timeline(recipe, names, duration)
    animation = sk.describe_recipe(recipe, names)

    pid = f'{int(time.time())}_{secrets.token_hex(3)}'
    pdir = os.path.join(SCHEDULE_DIR, pid)
    os.makedirs(pdir, exist_ok=True)
    params['_plug_dir'] = pdir
    work = app.config['UPLOAD_FOLDER']
    master = os.path.join(work, f'schedmaster_{jid}.mov')
    prod = pipeline.load_production_defaults()
    fps = sk.fps_for(fmt)

    def drawing(done, total):
        report(percent=14 + int(68 * done / max(total, 1)), step=f'Drawing frames ({done}/{total})')

    try:
        animator = sk.Animator(art, timeline, recipe, duration)
        try:
            sk.encode(animator, master, fps, duration, ffmpeg=pipeline.FFMPEG, music_path=params.get('music'),
                      loudness=float(prod.get('target_loudness', -14.0)), true_peak=float(prod.get('true_peak', -1.5)),
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
        if not cv2.imwrite(os.path.join(pdir, poster), cv2.resize(animator.frame(duration), (640, 360),
                                                                  interpolation=cv2.INTER_AREA)):
            poster = None
        manifest = {'plug_id': pid, 'created': time.time(), 'user_id': params.get('user_id'),
                    'username': params.get('username'), 'title': params['orig_name'], 'duration': duration,
                    'format': fmt, 'format_label': pipeline.EXPORT_FORMATS[fmt]['label'], 'file': name,
                    'size': os.path.getsize(out), 'preview': preview, 'poster': poster,
                    'layers': names, 'layered': art['layered'], 'prompt': params.get('prompt') or '',
                    'style': params.get('style'), 'style_label': sk.style_label(params.get('style')),
                    'animation': animation, 'read_by': read_by, 'music': params.get('music_name'),
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
                   styles=[{'key': k, 'label': label} for k, label, _ in sk.STYLES], default_style=sk.DEFAULT_STYLE,
                   effects=list(sk.EFFECTS), psd_supported=psd, max_layers=sk.MAX_LAYERS,
                   extensions=sorted(sk.IMAGE_EXTENSIONS if psd else sk.IMAGE_EXTENSIONS - {'psd', 'psb'}))


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
    image = pipeline._resolve_upload('schedule_image', sk.IMAGE_EXTENSIONS)
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
    orig = original('schedule_image', image)
    params = {'image': image, 'orig_name': orig, 'music': music,
              'music_name': (original('schedule_music', music) if music else None),
              'duration': duration, 'format': fmt, 'style': style,
              'prompt': ' '.join((request.form.get('prompt') or '').split())[:600],
              'user_id': session.get('user_id'), 'username': session.get('username')}
    jid = pipeline.job_new(user_id=session.get('user_id'), username=session.get('username'))
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
    under Config > Network -- the same ones the promo generator delivers to."""
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
    try:
        pipeline.send_file_to_network_destination(os.path.join(_plug_dir(pid), m['file']), m['file'], dest)
    except ValueError as e:
        return jsonify(ok=False, error=str(e)), 502
    audit_log('schedule_plug_send_to_destination', target=f'{dest["name"]}: {m["file"]}',
              user_id=session.get('user_id'), username=session.get('username'), ip=_client_ip())
    return jsonify(ok=True, sent=[m['file']], destination=dest['name'])


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

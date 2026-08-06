"""Entrypoint: pulls in every route module (for their @app.route
registration side effects), defines the last few top-level routes (the UI
shell, upload/download passthroughs), and starts the server.

Run this file instead of any of the individual modules: `python3 main.py`.
"""
import os, subprocess, threading
from flask import request, jsonify, session, send_from_directory, render_template

from core import app
import library_db      # noqa: F401 -- imported for its module-level init side effect
import auth             # noqa: F401 -- registers /login, /logout, /admin/users
from auth import require_permission, user_permissions
from pipeline import (   # noqa: F401 -- registers every /api/, trailer-generation,
    GENRE_DOCS_ROWS, EXPORT_FORMATS, build_export_cmd,
    _sweeper_loop, sweep_upload_folder, free_disk_mb, load_config_overrides, load_branding,
)
import pipeline          # noqa: F401 -- import the module itself too, for its
                          # own @app.route registrations beyond the names above


@app.route('/')
def index():
    perms = user_permissions(session.get('user_id'), session.get('role'))
    role = session.get('role')
    # Whichever nav tab is marked active server-side determines which panel
    # shows on load. If that were hardcoded to Generate Promo Plug (as it was
    # before groups existed) and a restricted account can't reach it, they'd
    # land on a hidden tab's panel -- visible content with no way to submit
    # it, since the routes underneath are gated too. Picks the first tab (in
    # the same order they appear in the sidebar) this session can actually
    # use; Docs is the final fallback since it's never gated.
    tab_order = [
        ('p-trailer', 'promo_generation'), ('p-music', 'music_generation'),
        ('p-sfx', 'text_to_sfx'), ('p-fish', 'text_to_speech'),
        ('p-stt', 'speech_to_text'), ('p-vision', 'scene_detection'),
        ('p-chat', 'ai_chat'), ('p-tools', 'player'), ('p-docs', None),
    ]
    default_tab = 'p-docs'
    for tab_id, perm in tab_order:
        if perm is None or role == 'admin' or perm in perms:
            default_tab = tab_id
            break
    brand = load_branding()
    return render_template('index.html', genre_rows=GENRE_DOCS_ROWS,
                                   current_username=session.get('username'),
                                   current_role=role,
                                   current_permissions=perms,
                                   default_tab=default_tab,
                                   brand_name=brand['name'],
                                   brand_tagline=brand['tagline'],
                                   brand_accent=brand['accent_color'])

@app.route('/uploads/<filename>')
def uploaded(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/download/<filename>')
@require_permission('promo_generation')
def download_file(filename):
    orig = request.args.get('name', filename)
    fmt_key = request.args.get('format', 'mp4_high')
    base_name, _ = os.path.splitext(orig)

    if fmt_key not in EXPORT_FORMATS:
        return jsonify(error=f'Unknown export format: {fmt_key}'), 400

    src_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(src_path):
        return jsonify(error='File not found'), 404

    ext = EXPORT_FORMATS[fmt_key]['ext']
    cache_name = f'{os.path.splitext(filename)[0]}_{fmt_key}.{ext}'
    cache_path = os.path.join(app.config['UPLOAD_FOLDER'], cache_name)
    if not (os.path.exists(cache_path) and os.path.getsize(cache_path) > 0):
        cmd = build_export_cmd(src_path, cache_path, fmt_key)
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if not (os.path.exists(cache_path) and os.path.getsize(cache_path) > 0):
            return jsonify(error=f'Export to {fmt_key} failed: {r.stderr[-800:]}'), 500

    resp = send_from_directory(app.config['UPLOAD_FOLDER'], cache_name)
    resp.headers['Content-Disposition'] = f'attachment; filename="{base_name}.{ext}"'
    return resp

load_config_overrides()  # apply any saved Config-tab overrides on top of the env-var defaults

if __name__ == '__main__':
    print(' * Server starting...')
    threading.Thread(target=_sweeper_loop, daemon=True).start()
    sweep_upload_folder()  # reclaim anything left over from a previous run
    _free = free_disk_mb()
    if _free is not None:
        print(f' * Disk free on work volume: {_free:,.0f} MB'
              + ('   ** LOW — renders may fail **' if _free < 2048 else ''))
    print(' * HTTP:  http://0.0.0.0:5000/')
    print(' * Access from local machine: http://localhost:5000/')
    print(' * Access from other devices: http://YOUR_IP:5000/')

    # Werkzeug's dev server (what app.run() below starts) prints its own
    # warning not to use it in production, and it's right -- it isn't
    # hardened for that. waitress is a real production WSGI server, pure
    # Python (no C build step, so it installs the same way on Windows as
    # Linux) and, importantly for this app specifically: single *process*,
    # multiple *threads*. That distinction matters here because job/preview
    # state (JOBS, PREVIEWS in pipeline.py) lives in plain in-memory dicts,
    # not a shared store like Redis -- a multi-process server (e.g. gunicorn
    # with -w 4) would give each worker its own separate copy, so a job
    # started on one worker could silently vanish from progress-polling
    # requests that happen to land on a different one. waitress's threading
    # model doesn't have that failure mode, so it's the default here rather
    # than something to opt into.
    #
    # DEV_SERVER=1 falls back to Werkzeug's dev server -- e.g. if you want
    # its interactive debugger for troubleshooting (which needs debug=True
    # to actually engage; this app runs with debug=False either way).
    if os.environ.get('DEV_SERVER', '').strip().lower() in ('1', 'true', 'yes'):
        print(' * DEV_SERVER=1 -- using Werkzeug\'s development server, not waitress.')
        print(' * Do not use this for anything but local troubleshooting.')
        app.run(debug=False, host='0.0.0.0', port=5000, threaded=True, use_reloader=False)
    else:
        try:
            from waitress import serve
        except ImportError:
            print(' * waitress is not installed (pip install waitress) -- falling back to')
            print(' * the development server. Fine for local use; install waitress before')
            print(' * running this anywhere production traffic can reach it.')
            app.run(debug=False, host='0.0.0.0', port=5000, threaded=True, use_reloader=False)
        else:
            threads = int(os.environ.get('WAITRESS_THREADS', 8))
            print(f' * Serving with waitress ({threads} threads, single process)')
            serve(app, host='0.0.0.0', port=5000, threads=threads)

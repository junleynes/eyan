"""Entrypoint: pulls in every route module (for their @app.route
registration side effects), defines the last few top-level routes (the UI
shell, upload/download passthroughs), and starts the server.

Run this file instead of any of the individual modules: `python3 main.py`.
"""
import os, subprocess, threading
from flask import request, jsonify, session, send_from_directory, render_template, redirect

from core import app, ensure_csrf_token
import library_db      # noqa: F401
import auth             # noqa: F401 -- registers /login, /logout, /admin/users
from auth import require_permission, user_permissions
from pipeline import (
    GENRE_DOCS_ROWS, EXPORT_FORMATS, build_export_cmd,
    _sweeper_loop, sweep_upload_folder, free_disk_mb, load_config_overrides, load_branding,
    ALLOW_LOCAL_MEDIA_UPLOAD,
)
import pipeline


@app.route('/')
def index():
    # Public visitors see the branded landing page. Existing authenticated
    # sessions go straight into the production workspace.
    if not session.get('authed'):
        brand = load_branding()
        return render_template('landing.html',
                               brand_name=brand['name'],
                               brand_tagline=brand['tagline'],
                               brand_theme=brand['theme_colors'])

    perms = user_permissions(session.get('user_id'), session.get('role'))
    role = session.get('role')
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
                           brand_accent=brand['theme_colors']['accent'],
                           brand_theme=brand['theme_colors'],
                           brand_footer=brand['footer'],
                           allow_local_upload=ALLOW_LOCAL_MEDIA_UPLOAD,
                           csrf_token=ensure_csrf_token())

@app.route('/uploads/<filename>')
def uploaded(filename):
    resp = send_from_directory(app.config['UPLOAD_FOLDER'], filename, conditional=True)
    resp.headers['Cache-Control'] = 'private, max-age=86400, immutable'
    return resp

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

load_config_overrides()

if __name__ == '__main__':
    print(' * Server starting...')
    threading.Thread(target=_sweeper_loop, daemon=True).start()
    sweep_upload_folder()
    _free = free_disk_mb()
    if _free is not None:
        print(f' * Disk free on work volume: {_free:,.0f} MB' + ('   ** LOW — renders may fail **' if _free < 2048 else ''))
    print(' * HTTP:  http://0.0.0.0:5000/')
    print(' * Access from local machine: http://localhost:5000/')
    print(' * Access from other devices: http://YOUR_IP:5000/')
    if os.environ.get('DEV_SERVER', '').strip().lower() in ('1', 'true', 'yes'):
        print(" * DEV_SERVER=1 -- using Werkzeug's development server, not waitress.")
        app.run(debug=False, host='0.0.0.0', port=5000, threaded=True, use_reloader=False)
    else:
        try:
            from waitress import serve
        except ImportError:
            print(' * waitress is not installed (pip install waitress) -- falling back to the development server.')
            app.run(debug=False, host='0.0.0.0', port=5000, threaded=True, use_reloader=False)
        else:
            threads = int(os.environ.get('WAITRESS_THREADS', 8))
            print(f' * Serving with waitress ({threads} threads, single process)')
            serve(app, host='0.0.0.0', port=5000, threads=threads)

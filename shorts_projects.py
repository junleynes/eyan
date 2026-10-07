"""
Vertical Shorts -- projects: where a programme's shorts are filed.

Before this, every render landed in one list. A project is the episode the
shorts were cut from, written down once -- programme, episode, air date, a
description and a picture -- and every batch rendered for it is filed under
it. A project has to be chosen before shorts are generated, so nothing is
saved loose.

This module is the store and the rules for a project's fields. It imports
nothing from the rest of PRISM (shorts.py owns the HTTP routes and who may
do what), so it can be tested without the app.

On disk: <root>/<project id>/project.json, and thumb.jpg beside it when a
picture was given.
"""
import datetime
import json
import os
import re
import secrets
import shutil
import time

import cv2
import numpy as np

PROJECT_ID = re.compile(r'^p[0-9a-f]{10}$')
TITLE_MAX, EPISODE_MAX, DESCRIPTION_MAX = 120, 40, 1000
THUMB_MAX_BYTES = 8 * 1024 * 1024
THUMB_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp'}
THUMB_SIDE = 640            # stored no larger than this on its longer side


class ProjectError(ValueError):
    """Something about the project as entered is not usable; the message is
    written for the person who entered it."""


def new_id():
    return 'p' + secrets.token_hex(5)


def project_dir(root, pid):
    """The folder for `pid`, or None for anything that is not a project id
    (which is also what keeps a crafted id from pointing somewhere else)."""
    if not isinstance(pid, str) or not PROJECT_ID.match(pid):
        return None
    return os.path.join(root, pid)


def clean_fields(raw):
    """The fields of a project from what was typed: {'title', 'episode',
    'air_date', 'description'}. Raises ProjectError for what cannot be
    accepted (no title, a date that is not a date)."""
    def text(key, limit):
        return ' '.join(str(raw.get(key) or '').split())[:limit]
    title = text('title', TITLE_MAX)
    if not title:
        raise ProjectError('Give the project a programme title.')
    air = str(raw.get('air_date') or '').strip()
    if air:
        try:
            air = datetime.date.fromisoformat(air).isoformat()
        except ValueError:
            raise ProjectError('The air date must be a date (YYYY-MM-DD).') from None
    description = '\n'.join(' '.join(ln.split()) for ln in str(raw.get('description') or '').splitlines())
    return {'title': title, 'episode': text('episode', EPISODE_MAX), 'air_date': air or None,
            'description': description.strip()[:DESCRIPTION_MAX]}


def make_thumbnail(data, filename=''):
    """JPEG bytes for a project picture, from an uploaded image file's bytes.

    The upload is decoded and what is stored is our own re-encoding of the
    pixels, scaled down: the file as sent is never kept, served or passed to
    anything else. Raises ProjectError if it is not an image PRISM accepts."""
    ext = os.path.splitext(str(filename or ''))[1].lower().lstrip('.')
    if filename and ext not in THUMB_EXTENSIONS:
        raise ProjectError('The thumbnail must be a JPEG, PNG or WebP image.')
    if not data:
        raise ProjectError('The thumbnail file is empty.')
    if len(data) > THUMB_MAX_BYTES:
        raise ProjectError(f'The thumbnail is larger than {THUMB_MAX_BYTES // (1024 * 1024)} MB. Use a smaller image.')
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        raise ProjectError('The thumbnail could not be read as an image. Use a JPEG, PNG or WebP file.')
    h, w = img.shape[:2]
    if max(h, w) > 12000:
        raise ProjectError('The thumbnail is too large in pixels. Use an image under 12000 pixels on a side.')
    k = min(1.0, THUMB_SIDE / float(max(h, w)))
    if k < 1.0:
        img = cv2.resize(img, (max(1, int(round(w * k))), max(1, int(round(h * k)))), interpolation=cv2.INTER_AREA)
    ok, out = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise ProjectError('The thumbnail could not be saved.')
    return out.tobytes()


def _write(pdir, project):
    """Atomically, so a reader never sees half a file."""
    tmp = os.path.join(pdir, f'.project_{secrets.token_hex(4)}.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(project, f, indent=1)
    os.replace(tmp, os.path.join(pdir, 'project.json'))


def load(root, pid):
    """The project, or None if there is no such project."""
    pdir = project_dir(root, pid)
    if not pdir:
        return None
    try:
        with open(os.path.join(pdir, 'project.json'), encoding='utf-8') as f:
            p = json.load(f)
    except (OSError, ValueError):
        return None
    return p if isinstance(p, dict) and p.get('project_id') == pid else None


def list_all(root):
    """Every project, most recently changed first."""
    try:
        names = [n for n in os.listdir(root) if PROJECT_ID.match(n)]
    except OSError:
        return []
    out = [p for p in (load(root, n) for n in names) if p]
    out.sort(key=lambda p: p.get('updated') or p.get('created') or 0, reverse=True)
    return out


def create(root, fields, user_id=None, username=None, thumb_jpeg=None):
    pid = new_id()
    pdir = os.path.join(root, pid)
    os.makedirs(pdir)
    now = time.time()
    project = dict(fields, project_id=pid, created=now, updated=now, user_id=user_id, username=username, thumb=None)
    if thumb_jpeg:
        with open(os.path.join(pdir, 'thumb.jpg'), 'wb') as f:
            f.write(thumb_jpeg)
        project['thumb'] = 'thumb.jpg'
    _write(pdir, project)
    return project


def update(root, pid, fields=None, thumb_jpeg=None, remove_thumb=False):
    """Changes a project's details and/or its picture. Returns the project,
    or None if there is no such project."""
    project = load(root, pid)
    if project is None:
        return None
    pdir = project_dir(root, pid)
    if fields:
        project.update(fields)
    if thumb_jpeg:
        with open(os.path.join(pdir, 'thumb.jpg'), 'wb') as f:
            f.write(thumb_jpeg)
        project['thumb'] = 'thumb.jpg'
    elif remove_thumb:
        try:
            os.remove(os.path.join(pdir, 'thumb.jpg'))
        except OSError:
            pass
        project['thumb'] = None
    project['updated'] = time.time()
    _write(pdir, project)
    return project


def delete(root, pid):
    pdir = project_dir(root, pid)
    if not pdir or not os.path.isdir(pdir):
        return False
    shutil.rmtree(pdir, ignore_errors=True)
    return not os.path.exists(pdir)


def display_name(project):
    """'Tadhana -- Ep. 101' : how a project is referred to in a sentence."""
    ep = (project.get('episode') or '').strip()
    return project.get('title') or 'Untitled project' if not ep else f"{project.get('title')} — {ep}"

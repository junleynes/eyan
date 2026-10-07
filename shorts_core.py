"""Vertical Shorts -- core logic.

Turns one long-form (usually 16:9) programme into stand-alone 9:16 shorts:
which stretches of the episode are worth cutting out, where exactly each one
should start and end, how to reframe a landscape picture into a portrait one,
and how to burn the dialogue in as captions.

Deliberately has NO Flask and NO imports from the rest of PRISM. Everything
here takes plain values (a transcript, a list of cut times, a file path, an
ffmpeg binary) and returns plain values, so the same code can be driven by
the PRISM tab (shorts.py) today and by a watch-folder script or a standalone
service later without dragging the web app along with it.

The selection is two-layer on purpose. Visual drama on its own finds
intensity peaks -- a shout, a slap -- with no idea whether the surrounding
minute makes sense to someone who never saw the episode; the transcript on
its own finds coherent story beats with no idea whether anything is
happening on screen. So:

  1. a vision model rates sampled frames for how dramatic they LOOK,
  2. a language model reads the transcript (with those visual notes inlined)
     and proposes moments that stand alone as a story: setup, turn, payoff,
  3. each proposal is fitted to the requested length, snapped so it neither
     cuts a sentence nor starts mid-shot when a cut is nearby, scored on
     story + visual + pace, and de-duplicated.
"""
import bisect
import json
import math
import os
import re
import subprocess

import cv2
import numpy as np
import requests


OUT_W, OUT_H = 1080, 1920


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def fmt_ts(sec):
    """Seconds -> 'MM:SS' (or 'H:MM:SS' past the hour), for prompts and UI."""
    sec = max(0, int(sec))
    h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
    return f'{h}:{m:02d}:{s:02d}' if h else f'{m:02d}:{s:02d}'


def even(n):
    """Nearest even integer >= 2. yuv420p needs even dimensions everywhere."""
    return max(2, 2 * int(round(n / 2.0)))


def slugify(text, maxlen=48):
    """ASCII, filesystem-safe slug for output filenames (Windows-safe too:
    no reserved characters, no trailing dot/space). Empty in, empty out --
    the caller supplies its own fallback."""
    t = re.sub(r'[^A-Za-z0-9]+', '_', str(text or '')).strip('_')
    return t[:maxlen].strip('_')


# --------------------------------------------------------------------------
# Transcript shaping
# --------------------------------------------------------------------------

_SENTENCE_END = re.compile(r'[.!?…]["\')\]]*$')


def segments_from_words(words, max_len=10.0, gap=0.6):
    """Builds sentence-ish lines from word timings alone, for a speech
    service that returned words but no segments."""
    segs, cur = [], []
    for w in words:
        if cur and (w['start'] - cur[-1]['end'] > gap
                    or cur[-1]['end'] - cur[0]['start'] >= max_len
                    or (_SENTENCE_END.search(cur[-1]['word'])
                        and cur[-1]['end'] - cur[0]['start'] >= 1.5)):
            segs.append(cur)
            cur = []
        cur.append(w)
    if cur:
        segs.append(cur)
    return [{'start': g[0]['start'], 'end': g[-1]['end'],
             'text': ' '.join(x['word'] for x in g)} for g in segs]


def normalize_transcript(words, segments, max_line=12.0):
    """Cleans a (words, segments) pair from the speech service into sorted,
    well-formed lists, and splits over-long segments.

    The lines matter more here than in the promo pipeline: they are the
    units the story model points at ("lines 41-58") and the units a window
    grows or shrinks by, so one 30-second run-on segment would make every
    boundary 30 seconds coarse. Whisper produces those on fast unpunctuated
    speech; when word timings are available such a segment is re-split at
    sentence punctuation or pauses."""
    w = []
    for x in words or []:
        try:
            s, e = float(x['start']), float(x['end'])
        except (KeyError, TypeError, ValueError):
            continue
        t = str(x.get('word') or '').strip()
        if t and e >= s:
            w.append({'start': s, 'end': e, 'word': t})
    w.sort(key=lambda d: d['start'])

    segs = []
    for x in segments or []:
        try:
            s, e = float(x['start']), float(x['end'])
        except (KeyError, TypeError, ValueError):
            continue
        t = ' '.join(str(x.get('text') or '').split())
        if t and e > s:
            segs.append({'start': s, 'end': e, 'text': t})
    segs.sort(key=lambda d: d['start'])

    if not segs and w:
        return w, segments_from_words(w)
    if not w:
        return w, segs

    starts = [x['start'] for x in w]
    out = []
    for sg in segs:
        if sg['end'] - sg['start'] <= max_line:
            out.append(sg)
            continue
        lo = bisect.bisect_left(starts, sg['start'] - 1.0)
        hi = bisect.bisect_right(starts, sg['end'])
        # By midpoint, so a word of the neighbouring line that merely starts
        # close to this one's edge isn't pulled into it.
        inner = [x for x in w[lo:hi] if sg['start'] - 0.02 <= (x['start'] + x['end']) / 2.0 <= sg['end'] + 0.02]
        parts = segments_from_words(inner, max_len=max_line * 0.8) if inner else []
        out.extend(parts if len(parts) > 1 else [sg])
    return w, out


def speech_units(words, segments):
    """(start, end) of every spoken unit, as finely as we know it: words
    when the service gave word timings, otherwise whole lines. Used to keep
    a cut from landing inside speech."""
    src = words if words else segments
    return sorted((float(x['start']), float(x['end'])) for x in src)


def chunk_segments(segments, chunk_sec=300.0, overlap_sec=60.0):
    """Splits the transcript into overlapping [lo, hi) index ranges of
    roughly `chunk_sec` each, for the story model.

    A whole episode's transcript does not fit a local model's context, and
    even where it does the model's attention thins out badly past a few
    thousand tokens. The overlap exists so a moment straddling a chunk
    boundary is seen whole by at least one chunk; duplicates it produces
    are removed later by the overlap de-duplication."""
    out, n, i = [], len(segments), 0
    while i < n:
        t0 = segments[i]['start']
        j = i
        while j < n and segments[j]['end'] - t0 <= chunk_sec:
            j += 1
        j = max(j, i + 1)
        out.append((i, j))
        if j >= n:
            break
        back = segments[j - 1]['end'] - overlap_sec
        k = j
        while k > i + 1 and segments[k - 1]['start'] >= back:
            k -= 1
        i = max(k, i + 1)
    return out


# --------------------------------------------------------------------------
# Layer 1: visual drama (prompt + reply parsing; the HTTP call is below)
# --------------------------------------------------------------------------

VISION_PROMPT = (
    'This is one frame from a TV drama. Rate how dramatic the moment LOOKS for a short '
    'vertical video, from 1 to 5.\n'
    '5 = intense emotion or action is clearly visible: crying, shouting, a confrontation, '
    'a slap or fight, shock, an embrace or kiss, danger.\n'
    '3 = people talking with visible emotion or tension.\n'
    '1 = nothing happening: an empty or establishing shot, a static wide shot, titles, '
    'credits, a logo or graphic.\n'
    'Also describe the frame in at most 12 words: who is visible and what they are doing.\n'
    'Reply with JSON only: {"score": <1-5>, "desc": "<description>"}'
)

VISION_FORMAT = {
    'type': 'object',
    'properties': {'score': {'type': 'integer', 'minimum': 1, 'maximum': 5},
                   'desc': {'type': 'string'}},
    'required': ['score', 'desc'],
}


def parse_vision_reply(text):
    """(score 1-5 or None, description) from a vision reply.

    Tries strict JSON first, then pulls the fields out by hand -- a reply cut
    off by the token budget is truncated mid-string and no longer parses as
    JSON, but usually still contains a usable score."""
    text = (text or '').strip()
    desc = ''
    if text.startswith('{'):
        try:
            obj = json.loads(text)
            desc = ' '.join(str(obj.get('desc') or '').split())[:160]
            sc = int(float(obj.get('score')))
            if 1 <= sc <= 5:
                return sc, desc
        except (ValueError, TypeError, AttributeError):
            pass
    m_desc = re.search(r'"desc"\s*:\s*"([^"]*)', text)
    if m_desc and not desc:
        desc = ' '.join(m_desc.group(1).split())[:160]
    m = (re.search(r'"?score"?\s*[:=]\s*"?([1-5])\b', text, re.I)
         or re.search(r'\b([1-5])\s*/\s*5\b', text))
    return (int(m.group(1)) if m else None), desc


def vision_sample_times(shots, duration, budget=90, min_gap=6.0):
    """Where in the programme to grab frames for the vision model.

    Evenly spread, `budget` frames at most, never closer than `min_gap`.
    Each sample is nudged to sit inside a shot rather than on its edge, so
    the frame is not half of a dissolve or the first soft frame after a cut.
    A long programme therefore gets a coarser sampling, not more calls: the
    cost of a job stays bounded whatever the source length."""
    if duration <= 0 or budget <= 0:
        return []
    step = max(float(min_gap), duration / float(budget))
    starts = [s for s, _ in shots]
    times, t = [], step / 2.0
    while t < duration and len(times) < budget:
        at = t
        k = bisect.bisect_right(starts, t) - 1
        if 0 <= k < len(shots):
            s, e = shots[k]
            if e - s <= 0.8:
                at = (s + e) / 2.0
            else:
                at = min(max(t, s + 0.3), e - 0.3)
        at = round(min(max(at, 0.0), max(0.0, duration - 0.05)), 3)
        if not times or at - times[-1] >= 0.5:
            times.append(at)
        t += step
    return times


# --------------------------------------------------------------------------
# Layer 2: narrative beats (prompt + reply parsing)
# --------------------------------------------------------------------------

STORY_FORMAT = {
    'type': 'object',
    'properties': {
        'moments': {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'start_id': {'type': 'integer'},
                    'end_id': {'type': 'integer'},
                    'title': {'type': 'string'},
                    'hook': {'type': 'string'},
                    'why': {'type': 'string'},
                    'score': {'type': 'integer', 'minimum': 1, 'maximum': 10},
                },
                'required': ['start_id', 'end_id', 'title', 'score'],
            },
        },
    },
    'required': ['moments'],
}


def build_story_prompt(segments, lo, hi, visual, min_dur, max_dur, max_moments=3, focus=None, avoid=None):
    """The story-model prompt for transcript lines [lo, hi).

    Line IDs are the GLOBAL segment indices, not renumbered per chunk, so
    answers from overlapping chunks point at the same lines and can be
    compared directly. Visual notes from layer 1 are interleaved in time
    order: the model picks beats knowing what is on screen, which is what
    lets a quiet-on-paper stretch (a long look, a slap with no line) still
    register, and a busy-on-paper one over a static wide shot rank lower.

    `focus` is what the editor wants more of and `avoid` what they want left
    out. They are worded differently on purpose: focus is a preference (a
    weak moment is not picked just for matching it), avoid is a rule (a
    strong moment is still left out for matching it)."""
    t_lo = segments[lo]['start'] - 5.0
    t_hi = segments[hi - 1]['end'] + 5.0
    rows = [(segments[i]['start'], 0,
             f"[{i}] ({fmt_ts(segments[i]['start'])}-{fmt_ts(segments[i]['end'])}) {segments[i]['text']}")
            for i in range(lo, hi)]
    for v in visual or []:
        if t_lo <= v['t'] <= t_hi and v.get('score'):
            desc = (v.get('desc') or '').strip() or 'no description'
            rows.append((v['t'], 1, f"[SCREEN {fmt_ts(v['t'])}] {desc} (visual drama {v['score']}/5)"))
    rows.sort(key=lambda r: (r[0], r[1]))
    body = '\n'.join(r[2] for r in rows)
    extra = ''
    if focus:
        extra = ('\nThe editor especially wants moments about: ' + ' '.join(str(focus).split())[:300]
                 + '\nPrefer moments that match this, but never pick a weak one just because it matches.\n')
    if avoid:
        extra += ('\nThe editor does NOT want: ' + ' '.join(str(avoid).split())[:300]
                  + '\nLeave out every moment that is mainly about this or shows it, however strong it is. '
                    'A moment that only mentions it in passing is fine.\n')
    return (
        'You are a short-form video editor cutting a TV drama episode into vertical shorts '
        '(TikTok / Reels / YouTube Shorts).\n\n'
        'Below is one stretch of the episode. Each numbered line is one spoken line: '
        '[ID] (start-end) text. Lines marked [SCREEN ...] are not dialogue: they describe what is '
        'visible at that moment and how dramatic it looks.\n\n'
        f'Find up to {int(max_moments)} moments that would each work as a stand-alone short of '
        f'{int(min_dur)} to {int(max_dur)} seconds. A good moment:\n'
        '- is one continuous run of consecutive lines,\n'
        '- makes sense to someone who has not seen the episode: it sets up a situation, has a '
        'conflict, confrontation, revelation or emotional turn, and ends on a payoff, a reaction '
        'or a cliffhanger,\n'
        '- opens on a line that grabs attention within the first three seconds,\n'
        '- does not start or end in the middle of a sentence or a thought.\n'
        'Do not pick recaps, greetings, small talk, credits or sponsor reads. If nothing in this '
        'stretch qualifies, return an empty list -- an empty list is a good answer.\n'
        + extra +
        '\nFor each moment give: start_id and end_id (the first and last line IDs, inclusive), '
        'title (at most 8 words, in the language of the dialogue), hook (the line or idea that '
        'grabs attention), why (one sentence), and score (1-10: how strong it is as a stand-alone '
        'dramatic short).\n'
        'Reply with JSON only: {"moments":[{"start_id":0,"end_id":0,"title":"","hook":"","why":"","score":0}]}\n\n'
        'EPISODE STRETCH\n' + body
    )


def _extract_moments(text):
    text = re.sub(r'^```[a-zA-Z]*\s*|\s*```$', '', (text or '').strip())
    cands = [text]
    a, b = text.find('{'), text.rfind('}')
    if 0 <= a < b:
        cands.append(text[a:b + 1])
    a, b = text.find('['), text.rfind(']')
    if 0 <= a < b:
        cands.append(text[a:b + 1])
    for c in cands:
        try:
            data = json.loads(c)
        except ValueError:
            continue
        if isinstance(data, dict):
            for key in ('moments', 'shorts', 'clips', 'results', 'items'):
                if isinstance(data.get(key), list):
                    return [d for d in data[key] if isinstance(d, dict)]
            if 'start_id' in data:
                return [data]
        elif isinstance(data, list):
            return [d for d in data if isinstance(d, dict)]
    # Truncated or chatty reply: salvage whichever flat objects are intact.
    out = []
    for m in re.finditer(r'\{[^{}]*\}', text):
        try:
            d = json.loads(m.group(0))
        except ValueError:
            continue
        if isinstance(d, dict) and 'start_id' in d:
            out.append(d)
    return out


def parse_story_reply(text, lo, hi):
    """Story-model reply -> beats, validated against the chunk it was asked
    about. Anything pointing outside [lo, hi) is the model inventing line
    numbers and is dropped rather than clamped into something it never
    actually chose; a range that merely overhangs the chunk is trimmed."""
    beats = []
    for o in _extract_moments(text):
        try:
            a, b = int(float(o.get('start_id'))), int(float(o.get('end_id')))
        except (TypeError, ValueError):
            continue
        if a > b:
            a, b = b, a
        if b < lo or a >= hi:
            continue
        a, b = max(a, lo), min(b, hi - 1)
        try:
            score = max(1, min(10, int(float(o.get('score')))))
        except (TypeError, ValueError):
            score = 5
        beats.append({
            'start_id': a, 'end_id': b, 'score': score, 'source': 'story',
            'title': ' '.join(str(o.get('title') or '').split())[:80],
            'hook': ' '.join(str(o.get('hook') or '').split())[:200],
            'why': ' '.join(str(o.get('why') or '').split())[:240],
        })
    return beats


def heuristic_beats(segments, min_dur, max_dur, limit=12):
    """Fallback proposals from the transcript alone, for when the story
    model is reachable but produced nothing usable.

    Not a substitute for it -- this only knows that a stretch is
    dialogue-dense and punctuated like an argument (questions,
    exclamations), not that it tells a story -- which is why every beat it
    returns is tagged source='heuristic' and carries no story score, and
    why the caller surfaces a warning instead of presenting these as the
    real thing."""
    n = len(segments)
    if not n:
        return []
    target = (float(min_dur) + float(max_dur)) / 2.0
    beats, i = [], 0
    while i < n:
        j = i
        while (j + 1 < n and segments[j + 1]['start'] - segments[j]['end'] <= 5.0
               and segments[j + 1]['end'] - segments[i]['start'] <= target):
            j += 1
        dur = segments[j]['end'] - segments[i]['start']
        if dur >= min(float(min_dur), target) * 0.6:
            text = ' '.join(s['text'] for s in segments[i:j + 1])
            heat = (2.0 * text.count('!') + 1.5 * text.count('?') + len(text.split()) / 20.0) / max(dur, 1.0)
            best = max(segments[i:j + 1], key=lambda s: s['text'].count('!') * 2 + s['text'].count('?'))
            title = ' '.join(best['text'].split()[:7]).strip(' .,;:')
            beats.append({'start_id': i, 'end_id': j, 'score': None, 'source': 'heuristic', 'heat': heat,
                          'title': title, 'hook': segments[i]['text'][:200],
                          'why': 'Dialogue-dense stretch (picked without story analysis).'})
        # Step about a third of a window so neighbouring proposals overlap
        # and the best-placed one survives de-duplication.
        t_next = segments[i]['start'] + target / 3.0
        k = i + 1
        while k < n and segments[k]['start'] < t_next:
            k += 1
        i = k
    beats.sort(key=lambda b: b['heat'], reverse=True)
    return beats[:limit * 3]


def visual_windows(visual, min_dur, max_dur, duration, limit=12):
    """Fallback for a source with no usable dialogue at all: raw time
    windows around the frames the vision model rated highest. This IS the
    'intensity peaks, no story' selection the two-layer design exists to
    avoid, used only when there is no second layer to be had."""
    target = (float(min_dur) + float(max_dur)) / 2.0
    out = []
    for v in sorted((v for v in visual or [] if v.get('score')), key=lambda v: v['score'], reverse=True):
        if v['score'] < 3:
            break
        t0 = max(0.0, v['t'] - target * 0.4)
        t1 = min(float(duration), t0 + target)
        t0 = max(0.0, t1 - target)
        if t1 - t0 >= float(min_dur) * 0.6:
            out.append({'t0': t0, 't1': t1, 'score': None, 'source': 'visual',
                        'title': (v.get('desc') or 'Visual moment')[:80], 'hook': '',
                        'why': 'Visually intense stretch (no dialogue to analyse).'})
        if len(out) >= limit * 3:
            break
    return out


# --------------------------------------------------------------------------
# Ollama client (thin: URL and model are always passed in)
# --------------------------------------------------------------------------

# Flipped off the first time a server rejects the structured-output `format`
# field, so an older Ollama pays for that discovery once per process.
_STRUCTURED_OK = True


def ollama_generate(base_url, payload, timeout=180):
    r = requests.post(str(base_url).rstrip('/') + '/api/generate', json=payload, timeout=timeout)
    data = r.json()
    if not isinstance(data, dict):
        raise RuntimeError('unexpected reply from Ollama')
    if data.get('error'):
        raise RuntimeError(str(data['error']))
    return (data.get('response') or '').strip(), data


def _generate(base_url, payload, schema, timeout):
    """One generation, constrained to `schema` when the server supports it.
    Constrained decoding is what stops a chatty model writing a paragraph
    and getting cut off before the answer -- prompt wording alone doesn't."""
    global _STRUCTURED_OK
    if schema is not None and _STRUCTURED_OK:
        try:
            return ollama_generate(base_url, dict(payload, format=schema), timeout)
        except RuntimeError as e:
            if 'format' not in str(e).lower():
                raise
            _STRUCTURED_OK = False
    return ollama_generate(base_url, payload, timeout)


def ask_vision(base_url, model, jpeg_b64, num_predict=200, timeout=180):
    """Rates one frame. Returns (score or None, description)."""
    payload = {'model': model, 'prompt': VISION_PROMPT, 'stream': False, 'images': [jpeg_b64],
               'think': False, 'options': {'temperature': 0.1, 'num_predict': num_predict}}
    text, data = _generate(base_url, payload, VISION_FORMAT, timeout)
    score, desc = parse_vision_reply(text or (data.get('thinking') or ''))
    if score is None:
        # A reasoning model that spent its budget thinking leaves `response`
        # empty. One retry, unconstrained and with room to finish.
        payload['options'] = {'temperature': 0.1, 'num_predict': num_predict * 3}
        payload.pop('think', None)
        text, data = ollama_generate(base_url, payload, timeout)
        score, desc2 = parse_vision_reply(text or (data.get('thinking') or ''))
        desc = desc or desc2
    return score, desc


def ask_story(base_url, model, prompt, num_ctx=8192, num_predict=900, timeout=300):
    """One story-analysis call. Returns the raw reply text.

    num_ctx is set explicitly: Ollama's default context is far smaller than
    a five-minute transcript chunk, and an over-long prompt is truncated
    from the FRONT without any error -- the model would silently lose the
    instructions and the first lines and answer about the rest."""
    payload = {'model': model, 'prompt': prompt, 'stream': False, 'think': False,
               'options': {'temperature': 0.2, 'num_predict': num_predict, 'num_ctx': int(num_ctx)}}
    text, data = _generate(base_url, payload, STORY_FORMAT, timeout)
    return text or (data.get('thinking') or '')


# --------------------------------------------------------------------------
# From a proposed beat to an exact, safe window
# --------------------------------------------------------------------------

def fit_indices(segments, i, j, min_dur, max_dur, gap_limit=6.0):
    """Grows or shrinks the line range [i, j] to respect the length limits.

    Too long: lines are dropped from the START. The model picked this
    stretch for where it lands -- the turn, the reaction, the cliffhanger --
    and a short that opens in the middle of an argument still works where
    one that stops before the payoff does not.

    Too short: neighbouring lines are added, earlier context first, then
    alternating, but never across a pause longer than `gap_limit` -- a gap
    that long is usually a scene change, and pulling in the tail of the
    previous scene is worse than running a little short."""
    n = len(segments)
    i, j = max(0, min(i, n - 1)), max(0, min(j, n - 1))
    if i > j:
        i, j = j, i

    def dur(a, b):
        return segments[b]['end'] - segments[a]['start']

    flags = []
    while i < j and dur(i, j) > max_dur:
        i += 1
        if 'trimmed' not in flags:
            flags.append('trimmed')
    back_first = True
    while dur(i, j) < min_dur:
        can_back = (i > 0 and segments[i]['start'] - segments[i - 1]['end'] <= gap_limit
                    and dur(i - 1, j) <= max_dur)
        can_fwd = (j < n - 1 and segments[j + 1]['start'] - segments[j]['end'] <= gap_limit
                   and dur(i, j + 1) <= max_dur)
        if can_back and (back_first or not can_fwd):
            i -= 1
        elif can_fwd:
            j += 1
        else:
            flags.append('short')
            break
        if 'extended' not in flags:
            flags.append('extended')
        back_first = not back_first
    return i, j, flags


def speech_bounds(t0, t1, words):
    """Tightens a line-level range to the first and last WORD in it. Whisper
    segment boundaries are padded and can overlap their neighbours; word
    timings are what a cut actually has to clear.

    A word belongs to the range if its MIDPOINT is inside it. Anything looser
    (an earlier version allowed 0.2 s either side) adopts the first word of
    the next line whenever speech runs on without a pause -- which is then
    both heard and captioned at the end of the short."""
    if not words:
        return t0, t1
    inner = [w for w in words if t0 - 0.02 <= (w['start'] + w['end']) / 2.0 <= t1 + 0.02]
    if not inner:
        return t0, t1
    return inner[0]['start'], inner[-1]['end']


def snap_window(t0, t1, cuts, units, duration, lead=0.25, tail=0.5, max_in=1.2, max_out=2.5):
    """Final in/out points for speech running from t0 to t1.

    Two rules, in priority order:
      1. Never inside speech. The in point stays after the previous spoken
         word, the out point before the next one.
      2. On a shot change when one is close enough. Starting exactly on a
         cut (up to `max_in` before the first word) and ending exactly on
         one (up to `max_out` after the last) looks edited rather than
         clipped; the longer allowance at the end leaves room for a
         reaction shot, which is usually where a drama beat really lands.
    With no usable cut nearby it falls back to a short lead-in and tail."""
    duration = float(duration)
    starts = [u[0] for u in units]
    ends = sorted(u[1] for u in units)

    # The 40 ms guard keeps clear of neighbouring SPEECH; with no speech on
    # that side (the very start or end of the programme) there is nothing to
    # keep clear of, and the first or last frame is a perfectly good cut.
    k = bisect.bisect_right(ends, t0 + 1e-3) - 1
    prev_end = ends[k] + 0.04 if k >= 0 else 0.0
    k = bisect.bisect_left(starts, t1 - 1e-3)
    next_start = starts[k] - 0.04 if k < len(starts) else duration

    lo = max(0.0, prev_end, t0 - max_in)
    if lo >= t0 - 0.02:
        start = max(0.0, min(t0, (prev_end - 0.04 + t0) / 2.0))
    else:
        a = bisect.bisect_left(cuts, lo)
        b = bisect.bisect_right(cuts, t0 - 0.03)
        start = cuts[b - 1] if b > a else max(lo, t0 - lead)

    hi = min(duration, next_start, t1 + max_out)
    if hi <= t1 + 0.05:
        end = min(duration, max(t1, (t1 + next_start + 0.04) / 2.0))
    else:
        a = bisect.bisect_left(cuts, t1 + 0.12)
        b = bisect.bisect_right(cuts, hi)
        end = cuts[a] if b > a else min(hi, t1 + tail)
    return max(0.0, start), min(duration, max(end, start + 0.5))


def _nearest_cut(t, cuts, max_shift):
    k = bisect.bisect_left(cuts, t)
    near = [c for c in cuts[max(0, k - 1):k + 1] if abs(c - t) <= max_shift]
    return min(near, key=lambda c: abs(c - t)) if near else t


def window_scores(start, end, story_score, visual, words, segments):
    """Component scores for a finished window, plus the 0-100 total.

    Story carries the most weight because it is the only component that
    knows whether the clip means anything; visual drama is next; pace
    (words per second) is a small tie-breaker that penalises both dead air
    and wall-to-wall talking. Missing components are left out of the
    weighted mean rather than counted as zero, so a fallback candidate
    with no story score is ranked on what IS known instead of sinking
    below every scored one by default."""
    dur = max(end - start, 1e-6)
    vis = sorted((v['score'] for v in visual or [] if start <= v['t'] <= end and v.get('score')),
                 reverse=True)
    visual_score = None
    if vis:
        top = vis[:max(1, math.ceil(len(vis) / 2.0))]
        visual_score = sum(top) / len(top)
    if words:
        wc = sum(1 for w in words if start <= (w['start'] + w['end']) / 2.0 <= end)
    else:
        wc = sum(len(s['text'].split()) for s in segments or []
                 if s['start'] < end and s['end'] > start)
    pace = wc / dur
    if 1.6 <= pace <= 3.6:
        pace_fit = 1.0
    elif pace < 1.6:
        pace_fit = max(0.0, (pace - 0.3) / 1.3)
    else:
        pace_fit = max(0.0, (5.5 - pace) / 1.9)
    parts = []
    if story_score is not None:
        parts.append((0.55, story_score / 10.0))
    if visual_score is not None:
        parts.append((0.30, (visual_score - 1.0) / 4.0))
    parts.append((0.15, pace_fit))
    total = sum(w * p for w, p in parts) / sum(w for w, _ in parts)
    return {'story': story_score,
            'visual': round(visual_score, 1) if visual_score is not None else None,
            'pace': round(pace, 2), 'score': int(round(total * 100))}


def dedupe_windows(cands, max_overlap=0.5):
    """Greedy overlap suppression, best score first: a candidate covering
    more than `max_overlap` of a better one's span (measured against the
    shorter of the two) is the same moment found twice -- by two
    overlapping transcript chunks, typically -- and is dropped."""
    kept = []
    for c in sorted(cands, key=lambda c: c['score'], reverse=True):
        ok = True
        for k in kept:
            inter = min(c['end'], k['end']) - max(c['start'], k['start'])
            if inter > max_overlap * min(c['end'] - c['start'], k['end'] - k['start']):
                ok = False
                break
        if ok:
            kept.append(c)
    return kept


def build_candidates(beats, segments, words, cuts, visual, duration,
                     min_dur=30.0, max_dur=90.0, limit=8, fps=25.0):
    """Beats (line ranges, or raw time windows from visual_windows) ->
    ranked, de-duplicated candidates with exact frame-aligned in/out points."""
    units = speech_units(words, segments)
    cuts = sorted(set([0.0, float(duration)] + [float(c) for c in cuts or []]))
    fps = float(fps) if fps and fps > 0 else 25.0
    out = []
    for b in beats:
        flags = []
        if 't0' in b:
            # A raw time window (no dialogue to protect): each end simply
            # moves to the nearest shot change, if one is within 3 seconds.
            start, end = _nearest_cut(b['t0'], cuts, 3.0), _nearest_cut(b['t1'], cuts, 3.0)
            if end - start < 3.0:
                start, end = b['t0'], b['t1']
            lines = [s for s in segments if s['start'] < end and s['end'] > start]
        else:
            i, j, flags = fit_indices(segments, b['start_id'], b['end_id'], min_dur, max_dur)
            t0, t1 = speech_bounds(segments[i]['start'], segments[j]['end'], words)
            if t1 - t0 > max_dur:
                # A single run-on line longer than the limit: nothing left
                # to drop, so the tail is cut by time.
                t1 = t0 + max_dur
                if 'trimmed' not in flags:
                    flags.append('trimmed')
            start, end = snap_window(t0, t1, cuts, units, duration)
            # The lead-in and the reaction tail are extras. Where they would
            # push the short past the maximum they are given back -- tail
            # first, then lead -- rather than handing the editor a candidate
            # the render step would refuse as too long.
            over = (end - start) - max_dur
            if over > 0:
                give = min(over, max(0.0, end - t1))
                end -= give
                over -= give
                if over > 0:
                    start += min(over, max(0.0, t0 - start))
            lines = segments[i:j + 1]
        # Whole frames, so the render's frame maths starts from exact values.
        start = round(start * fps) / fps
        end = min(float(duration), round(end * fps) / fps)
        if 't0' not in b and end - start > max_dur + 1e-6:
            end -= 1.0 / fps          # rounding to whole frames tipped it over by one
        if end - start < 3.0:
            continue
        sc = window_scores(start, end, b.get('score'), visual, words, segments)
        text = ' '.join(s['text'] for s in lines)
        out.append({
            'start': round(start, 3), 'end': round(end, 3), 'duration': round(end - start, 2),
            'title': b.get('title') or ' '.join(text.split()[:7]) or 'Untitled moment',
            'hook': b.get('hook') or '', 'why': b.get('why') or '',
            'source': b.get('source', 'story'), 'flags': flags,
            'story_score': sc['story'], 'visual_score': sc['visual'], 'pace': sc['pace'],
            'score': sc['score'], 'text': text[:2400], 'lines': len(lines),
        })
    out = dedupe_windows(out)[:max(1, int(limit))]
    for n, c in enumerate(out, 1):
        c['id'] = f'c{n}'
    return out


# --------------------------------------------------------------------------
# Reframing 16:9 -> 9:16
# --------------------------------------------------------------------------

def crop_geometry(disp_w, disp_h, out_w=OUT_W, out_h=OUT_H):
    """Largest source window with the output's aspect ratio: (crop_w, crop_h).
    For a landscape source that is the full height and a narrow slice of the
    width; for a source already narrower than 9:16 it is the full width."""
    if disp_w * out_h >= disp_h * out_w:
        crop_h = even(disp_h)
        crop_w = min(even(disp_w), even(crop_h * out_w / float(out_h)))
    else:
        crop_w = even(disp_w)
        crop_h = min(even(disp_h), even(crop_w * out_h / float(out_w)))
    return crop_w, crop_h


class FaceDetector:
    """Face boxes for crop planning.

    Prefers OpenCV's YuNet DNN when its model file is supplied -- it handles
    profiles, tilted heads and small faces far better -- and otherwise falls
    back to the Haar cascades that ship inside the opencv-python wheel, so
    reframing works on a stock install with nothing extra to download. One
    instance per render: neither backend is safe to share across threads."""

    def __init__(self, yunet_model=None, max_dim=640, score_threshold=0.7):
        self.kind = None
        self.max_dim = max_dim
        self._yn = self._front = self._profile = None
        if yunet_model and os.path.exists(yunet_model) and hasattr(cv2, 'FaceDetectorYN'):
            try:
                self._yn = cv2.FaceDetectorYN.create(yunet_model, '', (320, 320), score_threshold, 0.3, 50)
                self.kind = 'yunet'
            except Exception as e:
                print(f'Vertical Shorts: could not load the YuNet face model ({e}); using Haar cascades.')
                self._yn = None
        if self._yn is None:
            base = getattr(getattr(cv2, 'data', None), 'haarcascades', None)
            if base:
                front = cv2.CascadeClassifier(os.path.join(base, 'haarcascade_frontalface_default.xml'))
                if not front.empty():
                    self._front = front
                    self.kind = 'haar'
                    prof = cv2.CascadeClassifier(os.path.join(base, 'haarcascade_profileface.xml'))
                    self._profile = None if prof.empty() else prof

    def available(self):
        return self.kind is not None

    def detect(self, frame):
        """[(x, y, w, h, score)] in the frame's own pixel coordinates."""
        h, w = frame.shape[:2]
        k = min(1.0, self.max_dim / float(max(w, h)))
        small = cv2.resize(frame, (max(2, int(w * k)), max(2, int(h * k))),
                           interpolation=cv2.INTER_AREA) if k < 1.0 else frame
        out = []
        if self._yn is not None:
            sh, sw = small.shape[:2]
            self._yn.setInputSize((sw, sh))
            _, faces = self._yn.detect(small)
            for f in (faces if faces is not None else []):
                out.append((float(f[0]) / k, float(f[1]) / k, float(f[2]) / k, float(f[3]) / k, float(f[-1])))
        elif self._front is not None:
            # Plain greyscale, no global histogram equalisation: the cascade
            # already normalises each window it tests, and equalising the whole
            # frame was measured to wipe out a clearly visible face against a
            # dark, flat background (it stretches the face into the highlights).
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            m = max(24, int(min(small.shape[:2]) * 0.07))
            found = [tuple(b) for b in self._front.detectMultiScale(gray, 1.1, 6, minSize=(m, m))]
            if not found and self._profile is not None:
                # The profile cascade only knows faces turned one way, so the
                # mirror image is searched for the other.
                found = [tuple(b) for b in self._profile.detectMultiScale(gray, 1.1, 6, minSize=(m, m))]
                sw = gray.shape[1]
                found += [(sw - x - fw, y, fw, fh)
                          for (x, y, fw, fh) in self._profile.detectMultiScale(cv2.flip(gray, 1), 1.1, 6,
                                                                              minSize=(m, m))]
            for (x, y, fw, fh) in found:
                out.append((x / k, y / k, fw / k, fh / k, 1.0))
        return out


# Where, top to bottom of a face box, the mouth is and where the rigid upper
# face (eyes, nose bridge) used to steady it is; and how wide either is.
# Deliberately generous: both detectors box a face a little differently, and
# a band that clips the lower lip measures nothing.
MOUTH_BAND = (0.62, 0.98)
MOUTH_ANCHOR_BAND = (0.15, 0.58)
MOUTH_MIN_FACE = 28         # px; below this the mouth is a handful of pixels of noise
MOUTH_LAG_SEC = 0.08        # the two frames compared are this far apart
_MOUTH_CANVAS = 96          # every face is measured at this size, whatever its size on screen
_MOUTH_SEARCH = 8           # how far (canvas px) the head may have moved between the two frames


def mouth_activity(gray_a, gray_b, box):
    """How much a face's MOUTH moved between two greyscale frames a few
    hundredths of a second apart, as a fraction of that face's own contrast.
    None when it can't be measured (face too small, half out of frame, or
    featureless).

    A plain frame difference over the mouth mostly measures the HEAD moving:
    people nod and sway as they listen, and a listener's nod is a bigger
    change in those pixels than a speaker's syllable. So the head is taken
    out first. The upper face is rigid, so it is located again in the second
    frame, the mouth is read at that same offset, and what is left is the
    part of the change the head's own motion doesn't explain. Whatever the
    upper face itself failed to line up by (noise, blur, a blink) is then
    subtracted as the floor."""
    x, y, w, h = (float(v) for v in box[:4])
    if w < MOUTH_MIN_FACE or h < MOUTH_MIN_FACE:
        return None
    fh, fw = gray_a.shape[:2]
    c, sr, mg = _MOUTH_CANVAS, _MOUTH_SEARCH, 0.15
    rx0, ry0 = int(max(0, math.floor(x - mg * w))), int(max(0, math.floor(y - mg * h)))
    rx1, ry1 = int(min(fw, math.ceil(x + (1 + mg) * w))), int(min(fh, math.ceil(y + (1 + mg) * h)))
    if rx1 - rx0 < 24 or ry1 - ry0 < 24:
        return None
    pa = cv2.resize(gray_a[ry0:ry1, rx0:rx1], (c, c), interpolation=cv2.INTER_AREA).astype(np.float32)
    pb = cv2.resize(gray_b[ry0:ry1, rx0:rx1], (c, c), interpolation=cv2.INTER_AREA).astype(np.float32)
    kx, ky = c / float(rx1 - rx0), c / float(ry1 - ry0)

    def rect(band, left, right):
        return (int(round((x + left * w - rx0) * kx)), int(round((y + band[0] * h - ry0) * ky)),
                int(round((x + right * w - rx0) * kx)), int(round((y + band[1] * h - ry0) * ky)))

    ax0, ay0, ax1, ay1 = rect(MOUTH_ANCHOR_BAND, 0.15, 0.85)
    ax0, ay0, ax1, ay1 = max(sr, ax0), max(sr, ay0), min(c - sr, ax1), min(c - sr, ay1)
    if ax1 - ax0 < 16 or ay1 - ay0 < 10:
        return None
    anchor = pa[ay0:ay1, ax0:ax1]
    if float(anchor.std()) < 3.0:
        return None
    found = cv2.matchTemplate(pb[ay0 - sr:ay1 + sr, ax0 - sr:ax1 + sr], anchor, cv2.TM_CCOEFF_NORMED)
    _, _, _, loc = cv2.minMaxLoc(found)
    dx, dy = loc[0] - sr, loc[1] - sr

    def unexplained(r):
        x0, y0, x1, y1 = max(0, r[0]), max(0, r[1]), min(c, r[2]), min(c, r[3])
        if x1 - x0 < 10 or y1 - y0 < 6 or x0 + dx < 0 or y0 + dy < 0 or x1 + dx > c or y1 + dy > c:
            return None
        qa = cv2.GaussianBlur(pa[y0:y1, x0:x1], (3, 3), 0)
        qb = cv2.GaussianBlur(pb[y0 + dy:y1 + dy, x0 + dx:x1 + dx], (3, 3), 0)
        # Means removed, so a flash or a fade isn't read as motion either.
        return float(np.abs((qa - qa.mean()) - (qb - qb.mean())).mean())

    mouth, floor = unexplained(rect(MOUTH_BAND, 0.2, 0.8)), unexplained((ax0, ay0, ax1, ay1))
    if mouth is None or floor is None:
        return None
    return max(0.0, mouth - floor) / max(12.0, float(pa.std()))


def sample_faces(path, start_f, n_frames, fps, detector, step_sec=0.2, sar=1.0, mouth=False):
    """Runs the detector over a clip at ~1/step_sec samples a second.

    Returns [(frame index relative to the clip start, [(cx, cy, w, h), ...])]
    with x already in DISPLAY pixels (multiplied by the pixel aspect ratio),
    so the planner never has to think about anamorphic sources. Frames
    between samples are grab()bed, not decoded to an image, which is what
    keeps this several times faster than real time.

    With mouth=True each face gains a fifth value: mouth_activity() between
    the sampled frame and one MOUTH_LAG_SEC later (None where it couldn't be
    measured). That costs one more decoded frame per sample and no extra
    detection; it is what plan_reframe's speaker option reads."""
    out = []
    cap = cv2.VideoCapture(path)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_f))
        step = max(1, int(round(step_sec * fps)))
        lag = min(step - 1, max(1, int(round(MOUTH_LAG_SEC * fps)))) if mouth else 0
        pending = None
        for i in range(int(n_frames)):
            if i % step == 0:
                ok, frame = cap.read()
                if not ok:
                    break
                dets = detector.detect(frame)
                out.append((i, [((x + w / 2.0) * sar, y + h / 2.0, w * sar, h) for (x, y, w, h, _) in dets]))
                pending = (cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), dets) if lag and dets else None
            elif pending is not None and i % step == lag:
                ok, frame = cap.read()
                if not ok:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                idx, faces = out[-1]
                out[-1] = (idx, [f + (mouth_activity(pending[0], gray, d),) for f, d in zip(faces, pending[1])])
                pending = None
            elif not cap.grab():
                break
    finally:
        cap.release()
    return out


def _frame_target(faces, crop_w, disp_h, min_face_frac):
    fs = [f for f in faces if f[3] >= min_face_frac * disp_h]
    if not fs:
        return None
    fs.sort(key=lambda f: f[2] * f[3], reverse=True)
    big_area = fs[0][2] * fs[0][3]
    sig = [f for f in fs[:4] if f[2] * f[3] >= 0.4 * big_area]
    left = min(f[0] - f[2] / 2.0 for f in sig)
    right = max(f[0] + f[2] / 2.0 for f in sig)
    return {'big_cx': fs[0][0], 'span_cx': (left + right) / 2.0, 'span_w': right - left, 'sig': sig}


# ---- Following the speaker (optional) ----
# A face counts as speaking at a moment when its mouth is moving by at
# least SPEAKER_FLOOR and by SPEAKER_RATIO times as much as anyone else's.
# Below either, nobody is named and the planner does what it did before.
SPEAKER_FLOOR = 0.02
SPEAKER_RATIO = 1.5
SPEAKER_MIN_HOLD = 1.2      # seconds a framing is held before it may change again
SPEAKER_MIN_DECIDED = 0.4   # share of a shot's spoken samples that must name someone


def _face_tracks(hits):
    """Groups one shot's detections into people, by where they are: a face
    belongs to the person last seen nearest to it, within about a face
    width. Enough for the only case this is used for (faces too far apart
    to share a vertical frame), with no model and nothing to drift."""
    tracks = []
    for i, t in hits:
        taken = set()
        for f in sorted(t['sig'], key=lambda f: f[2] * f[3], reverse=True):
            best, best_d = None, None
            for k, tr in enumerate(tracks):
                if k in taken:
                    continue
                d = abs(f[0] - tr['cx'])
                if (d <= 0.75 * max(f[2], tr['w']) and abs(f[1] - tr['cy']) <= max(f[3], tr['h'])
                        and (best_d is None or d < best_d)):
                    best, best_d = k, d
            if best is None:
                tracks.append({'obs': []})
                best = len(tracks) - 1
            taken.add(best)
            tr = tracks[best]
            tr['cx'], tr['cy'], tr['w'], tr['h'] = f[0], f[1], f[2], f[3]
            tr['obs'].append((i, f))
    return tracks


def _in_speech(t, speech, starts, pad=0.12):
    k = bisect.bisect_right(starts, t + pad) - 1
    while k >= 0 and speech[k][0] > t - 12.0:
        if speech[k][0] - pad <= t <= speech[k][1] + pad:
            return True
        k -= 1
    return False


def _hold_runs(labels, min_len):
    """Absorbs every run shorter than min_len samples into its neighbour
    (the one before it; the one after for the first), shortest first, so a
    half-second interjection doesn't buy a cut there and a cut back."""
    labels = list(labels)
    while True:
        runs, k = [], 0
        while k < len(labels):
            j = k
            while j < len(labels) and labels[j] == labels[k]:
                j += 1
            runs.append((k, j))
            k = j
        short = [r for r in range(len(runs)) if runs[r][1] - runs[r][0] < min_len]
        if len(runs) <= 1 or not short:
            return labels
        r = min(short, key=lambda r: runs[r][1] - runs[r][0])
        src = runs[r - 1] if r > 0 else runs[r + 1]
        for k in range(*runs[r]):
            labels[k] = labels[src[0]]


def speaker_turns(grid, acts, a, b, fps, speech, min_hold=SPEAKER_MIN_HOLD, floor=SPEAKER_FLOOR,
                  ratio=SPEAKER_RATIO, min_decided=SPEAKER_MIN_DECIDED):
    """Who to frame, and from which frame to which, across one shot.

    grid is the shot's sampled frame numbers; acts holds, per person, the
    mouth activity at each of those samples (None where unknown); speech is
    [(start, end)] of the spoken words in clip seconds. Returns
    [(first frame, last frame, person index)] covering a..b, or None when
    the evidence doesn't name a speaker for enough of the shot's dialogue
    -- nobody is talking, the mouths can't be measured, or two people are
    moving theirs about equally.

    Only moments inside a spoken word vote: a mouth moving in silence is
    someone chewing or laughing, not a reason to cut to them. Activity is
    averaged over about a second first, because a talking mouth is caught
    closed on plenty of individual samples. Between lines the framing
    stays on whoever spoke last, and a change of speaker is placed just
    ahead of the next word after a pause, where an editor would cut."""
    if not speech or len(grid) < 3 or len(acts) < 2:
        return None
    speech = sorted((float(s), float(e)) for s, e in speech)
    starts = [s for s, _ in speech]
    n = len(grid)
    dt = float(np.median(np.diff(grid))) / fps
    half = max(1, int(round(0.4 / max(dt, 1e-3))))
    talking = [_in_speech(i / fps, speech, starts) for i in grid]
    raw = np.array([[np.nan if v is None else min(float(v), 0.5) for v in row] for row in acts], dtype=float)
    # Samples outside speech are blanked BEFORE averaging, not just denied a
    # vote: otherwise a laugh in the pause after a line leaks into the
    # average for the line's last word and takes the frame from there.
    raw[:, [k for k in range(n) if not talking[k]]] = np.nan
    smooth = np.full_like(raw, np.nan)
    for k in range(n):
        win = raw[:, max(0, k - half):k + half + 1]
        seen = (~np.isnan(win)).sum(axis=1)
        vals = np.nansum(win, axis=1) / np.maximum(seen, 1)
        smooth[:, k] = np.where(seen >= min(2, win.shape[1]), vals, np.nan)

    labels, spoken, decided = [None] * n, 0, 0
    for k in range(n):
        if not talking[k]:
            continue
        spoken += 1
        col = smooth[:, k]
        if np.isnan(col).any():
            continue
        order = np.argsort(col)[::-1]
        if col[order[0]] >= floor and col[order[0]] >= ratio * col[order[1]]:
            labels[k] = int(order[0])
            decided += 1
    if spoken < max(3, int(round(0.6 / max(dt, 1e-3)))) or decided < min_decided * spoken:
        return None

    last = next(v for v in labels if v is not None)
    for k in range(n):
        if labels[k] is None:
            labels[k] = last
        last = labels[k]
    labels = _hold_runs(labels, max(2, int(round(min_hold / max(dt, 1e-3)))))

    lead = max(1, int(round(0.15 * fps)))
    min_gap = max(2, int(round(0.5 * fps)))
    runs, start = [], a
    for k in range(1, n):
        if labels[k] == labels[k - 1]:
            continue
        t = grid[k] / fps
        cut = grid[k]
        # The next word that follows a pause, nearest to where the mouths
        # say the speaker changed: that is the new line starting.
        best = None
        j = bisect.bisect_left(starts, t - 0.9)
        while j < len(speech) and speech[j][0] <= t + 0.5:
            if j == 0 or speech[j][0] - speech[j - 1][1] >= 0.2:
                if best is None or abs(speech[j][0] - t) < abs(speech[best][0] - t):
                    best = j
            j += 1
        if best is not None:
            floor_f = int(math.ceil(speech[best - 1][1] * fps)) if best > 0 else a
            cut = max(floor_f, int(round(speech[best][0] * fps)) - lead)
        if not (start + min_gap <= cut <= b - min_gap):
            cut = grid[k]
        if not (start < cut <= b):
            continue
        runs.append((start, cut - 1, labels[k - 1]))
        start = cut
    runs.append((start, b, labels[-1]))
    return runs


def _speaker_segments(hits, a, b, fps, speech, crop_w, max_x, static_px):
    """Reframe segments for a shot whose faces don't fit one vertical frame,
    cut between them by who is speaking -- or None to leave the shot to the
    planner's ordinary handling."""
    if not speech or len(hits) < 3:
        return None
    need = max(2, int(math.ceil(0.3 * len(hits))))
    tracks = [t for t in _face_tracks(hits) if len(t['obs']) >= need]
    if len(tracks) < 2:
        return None
    grid = [i for i, _ in hits]
    # A sample whose second frame falls past the cut compared two different
    # shots; what it measured is the cut, not a mouth.
    guard = b - int(round(MOUTH_LAG_SEC * fps)) - 1
    acts = []
    for t in tracks:
        seen = {i: (f[4] if len(f) > 4 and i <= guard else None) for i, f in t['obs']}
        acts.append([seen.get(i) for i in grid])
    runs = speaker_turns(grid, acts, a, b, fps, speech)
    if not runs:
        return None
    out = []
    for fa, fb, who in runs:
        obs = tracks[who]['obs']
        allx = [f[0] for _, f in obs]
        # A person who stays put gets one framing for the whole shot, so
        # cutting back to them lands exactly where it was; one who drifts
        # is framed where they are during this turn.
        near = [f[0] for i, f in obs if fa <= i <= fb] or allx
        cx = np.median(allx) if max(allx) - min(allx) <= static_px else np.median(near)
        out.append({'a': fa, 'b': fb, 'layout': 'crop', 'x': float(np.clip(cx - crop_w / 2.0, 0.0, max_x)),
                    'keys': None, 'speaker': True})
    return out


def plan_reframe(samples, shot_starts, n_frames, disp_w, disp_h, crop_w, mode='auto', fps=25.0,
                 min_face_frac=0.05, speaker=False, speech=None):
    """Decides, shot by shot, how a landscape clip becomes a portrait one.

    Returns segments [{'a', 'b', 'layout', 'x', 'keys'}] covering frames
    0..n_frames-1 (a and b inclusive). layout is 'crop' (a 9:16 window cut
    out of the picture, at a fixed x or panning through `keys`) or 'fit'
    (the whole picture, small, over a blurred copy of itself).

    The decision is per SHOT because that is the unit a viewer reads: a
    crop that drifts or jumps inside a shot looks like a mistake, one that
    changes on a cut is invisible.

      * faces all fit inside the 9:16 window -> crop centred on them;
      * two or more similar-sized faces too far apart to fit -> 'auto'
        shows the whole frame (cutting to one of them would silently drop
        whoever is speaking half the time), 'crop' commits to the side
        with more face on it for the whole shot;
      * no faces -> centre crop.

    Within a shot the window is locked off unless the subject really moves
    (more than ~12% of the window's width); then it follows on a heavily
    smoothed path, never frame-by-frame detections, which jitter.

    speaker=True changes the second case only, and only when the evidence
    is there (samples carrying mouth activity from sample_faces(mouth=True),
    and `speech`, the [(start, end)] of the spoken words in clip seconds):
    the shot is then cut between tight framings of whoever is speaking,
    see speaker_turns(). A shot it can't call falls through to what `mode`
    would have done anyway."""
    n_frames = int(n_frames)
    max_x = max(0.0, float(disp_w - crop_w))
    center = max_x / 2.0
    bounds = sorted(set([0] + [int(s) for s in shot_starts or [] if 0 < int(s) < n_frames]))
    static_px = 0.12 * crop_w
    segs = []
    for bi, a in enumerate(bounds):
        b = (bounds[bi + 1] - 1) if bi + 1 < len(bounds) else n_frames - 1
        if mode == 'fit':
            segs.append({'a': a, 'b': b, 'layout': 'fit', 'x': None, 'keys': None})
            continue
        if max_x <= 0:
            segs.append({'a': a, 'b': b, 'layout': 'crop', 'x': 0.0, 'keys': None})
            continue
        ss = [(i, fs) for i, fs in samples if a <= i <= b]
        if len(ss) > 3:
            # The frame on either side of a cut can belong to the neighbour
            # when the cut list is off by one; don't let it vote.
            ss = [(i, fs) for i, fs in ss if a + 1 < i < b - 1] or ss
        info = [(i, _frame_target(fs, crop_w, disp_h, min_face_frac)) for i, fs in ss]
        hits = [(i, t) for i, t in info if t]
        if not ss or len(hits) < max(1, math.ceil(0.25 * len(ss))):
            segs.append({'a': a, 'b': b, 'layout': 'crop', 'x': center, 'keys': None})
            continue

        fits = [t['span_w'] <= 0.9 * crop_w for _, t in hits]
        if sum(fits) >= 0.6 * len(hits):
            targets = [(i, t['span_cx'] if ok else t['big_cx']) for (i, t), ok in zip(hits, fits)]
        elif speaker and (turns := _speaker_segments(hits, a, b, fps, speech, crop_w, max_x, static_px)):
            segs.extend(turns)
            continue
        elif mode == 'auto':
            segs.append({'a': a, 'b': b, 'layout': 'fit', 'x': None, 'keys': None})
            continue
        else:
            mid = float(np.median([t['span_cx'] for _, t in hits]))
            left_w = sum(f[2] * f[3] for _, t in hits for f in t['sig'] if f[0] < mid)
            right_w = sum(f[2] * f[3] for _, t in hits for f in t['sig'] if f[0] >= mid)
            want_left = left_w >= right_w
            targets = []
            for i, t in hits:
                side = [f for f in t['sig'] if (f[0] < mid) == want_left]
                if side:
                    wsum = sum(f[2] * f[3] for f in side)
                    targets.append((i, sum(f[0] * f[2] * f[3] for f in side) / wsum))
            if not targets:
                segs.append({'a': a, 'b': b, 'layout': 'crop', 'x': center, 'keys': None})
                continue

        frames = np.array([i for i, _ in targets], dtype=float)
        xs = np.clip(np.array([cx for _, cx in targets], dtype=float) - crop_w / 2.0, 0.0, max_x)
        spread = float(np.percentile(xs, 95) - np.percentile(xs, 5)) if len(xs) > 1 else 0.0
        if len(xs) < 4 or spread <= static_px:
            segs.append({'a': a, 'b': b, 'layout': 'crop', 'x': float(np.median(xs)), 'keys': None})
            continue

        grid = np.array([i for i, _ in ss], dtype=float)
        gx = np.interp(grid, frames, xs)
        if len(gx) >= 3:       # 3-tap median: one stray detection can't yank the window
            med = gx.copy()
            med[1:-1] = np.median(np.stack([gx[:-2], gx[1:-1], gx[2:]]), axis=0)
            gx = med
        dt = float(grid[1] - grid[0]) / fps if len(grid) > 1 else 0.2
        alpha = 1.0 - math.exp(-dt / 0.6)
        # Odd-reflect a few samples past each end before smoothing. Without
        # it a subject that is already moving at the first or last frame of
        # the shot drags the window toward the middle of its path there,
        # leaving them off-centre exactly where the shot begins and ends.
        pad = min(len(gx) - 1, max(1, int(round(1.8 / max(dt, 1e-3)))))
        n_real = len(gx)
        gx = np.concatenate([2 * gx[0] - gx[pad:0:-1], gx, 2 * gx[-1] - gx[-2:-pad - 2:-1]])
        # Smoothed forward, then backward over the result: the two passes'
        # lags cancel, so the window neither trails the subject nor leads it.
        prev = gx[0]
        for k in range(1, len(gx)):
            prev = prev + alpha * (gx[k] - prev)
            gx[k] = prev
        prev = gx[-1]
        for k in range(len(gx) - 2, -1, -1):
            prev = prev + alpha * (gx[k] - prev)
            gx[k] = prev
        gx = np.clip(gx[pad:pad + n_real], 0.0, max_x)
        if float(gx.max() - gx.min()) <= static_px * 0.6:
            segs.append({'a': a, 'b': b, 'layout': 'crop', 'x': float(np.median(gx)), 'keys': None})
            continue
        # The samples stop a few frames short of each cut, so the path is
        # carried out to the shot's real first and last frame at the slope it
        # had there rather than frozen at the nearest sample.
        gstep = max(1.0, float(grid[1] - grid[0]))
        x_a = float(np.clip(gx[0] - (gx[1] - gx[0]) / gstep * (grid[0] - a), 0.0, max_x))
        x_b = float(np.clip(gx[-1] + (gx[-1] - gx[-2]) / gstep * (b - grid[-1]), 0.0, max_x))
        kstep = max(1, int(round(0.4 / max(dt, 1e-3))))
        keys = [(a, x_a)]
        for k in range(0, len(gx), kstep):
            f = int(grid[k])
            if keys[-1][0] < f < b:
                keys.append((f, float(gx[k])))
        if b > keys[-1][0]:
            keys.append((b, x_b))
        segs.append({'a': a, 'b': b, 'layout': 'crop', 'x': None, 'keys': keys})

    merged = []
    for s in segs:
        p = merged[-1] if merged else None
        if p and p['layout'] == s['layout'] and not p['keys'] and not s['keys'] and (
                s['layout'] == 'fit' or abs((p['x'] or 0) - (s['x'] or 0)) < 1.0):
            p['b'] = s['b']
        else:
            merged.append(dict(s))
    return merged


def crop_x_expr(segs):
    """ffmpeg expression for the crop window's x, as a function of the frame
    number `n`.

    A flat sum of between(n,a,b)*value terms, one active per frame, rather
    than nested if()s: it stays shallow however many shots there are, and
    keying on the integer frame number instead of the timestamp makes every
    boundary frame-exact with no float comparison to get wrong at 29.97."""
    terms = []
    for s in segs:
        if s['layout'] != 'crop':
            continue
        if s.get('keys'):
            ks = s['keys']
            for (f0, x0), (f1, x1) in zip(ks, ks[1:]):
                if f1 <= f0:
                    continue
                hi = s['b'] if f1 >= s['b'] else f1 - 1
                terms.append(f'between(n,{f0},{hi})*({x0:.1f}+({x1 - x0:.1f})*(n-{f0})/{f1 - f0})')
        else:
            terms.append(f"between(n,{s['a']},{s['b']})*{int(round(s['x'] or 0))}")
    return '+'.join(terms) or '0'


def build_filtergraph(info, segs, out_w=OUT_W, out_h=OUT_H, ass_name=None):
    """The -filter_complex string for one short. Video in on [0:v], out on
    [vout].

    Everything stays inside ffmpeg (no frames round-tripped through
    OpenCV/numpy): that keeps the source's colour matrix and range intact
    end to end, which a Python pixel pipeline would quietly re-interpret.

    When a clip mixes layouts, both versions are produced for the whole
    clip and the 'fit' one is overlaid only during its shots, switched by
    frame number. That costs some redundant filtering on the shots that
    don't use it, in exchange for a graph whose size doesn't grow with the
    number of shots and whose switches are frame-exact."""
    disp_w, disp_h = info['disp_w'], info['disp_h']
    crop_w, crop_h = crop_geometry(disp_w, disp_h, out_w, out_h)
    pre = ['yadif=mode=send_frame:parity=auto:deint=interlaced']
    if info.get('sd_matrix'):
        # SD sources are BT.601; the output is HD-sized and will be read as
        # BT.709. Converting here is the difference between correct colour
        # and visibly shifted greens and reds.
        pre.append(f'scale={disp_w}:{disp_h}:in_color_matrix=bt601:out_color_matrix=bt709')
    else:
        # Unconditional, not only when the pixels aren't square. Every size
        # below (the crop window, the fit layout) is worked out from the
        # size the PLANNER measured, and ffmpeg does not always decode to
        # that: a newer ffmpeg applies a container's clean-aperture crop
        # (a 1920x1080 .mov arriving as 1888x1062), a Windows capture
        # backend reports the coded 1088 lines of a 1080 picture. A crop
        # even one line taller than what actually arrives stops the whole
        # render with "Invalid argument". Pinning the size here makes every
        # later number true by construction; when the picture already is
        # this size the filter passes frames straight through.
        pre.append(f'scale={disp_w}:{disp_h}')
    pre.append('setsar=1')
    pre = ','.join(pre)

    has_crop = any(s['layout'] == 'crop' for s in segs)
    has_fit = any(s['layout'] == 'fit' for s in segs)
    y = even((disp_h - crop_h) / 2.0) if disp_h > crop_h else 0
    crop_chain = (f"crop={crop_w}:{crop_h}:x='{crop_x_expr(segs)}':y={y},"
                  f"scale={out_w}:{out_h}:flags=lanczos")
    k = min(out_w / float(disp_w), out_h / float(disp_h))
    fg_w, fg_h = min(out_w, even(disp_w * k)), min(out_h, even(disp_h * k))
    bw, bh = even(out_w / 4.0), even(out_h / 4.0)
    fit_chain = (f"split=2[fa][fb];"
                 f"[fa]scale={bw}:{bh}:force_original_aspect_ratio=increase,crop={bw}:{bh},"
                 f"boxblur=12:2,scale={out_w}:{out_h}:flags=bilinear,lutyuv=y='16+(val-16)*0.72'[bg];"
                 f"[fb]scale={fg_w}:{fg_h}:flags=lanczos[fg];"
                 f"[bg][fg]overlay=(W-w)/2:(H-h)/2")
    if has_crop and has_fit:
        enable = '+'.join(f"between(n,{s['a']},{s['b']})" for s in segs if s['layout'] == 'fit')
        g = (f"[0:v]{pre},split=2[c0][f0];[c0]{crop_chain}[cv];[f0]{fit_chain}[fv];"
             f"[cv][fv]overlay=0:0:enable='{enable}'[v1]")
    elif has_fit:
        g = f"[0:v]{pre},{fit_chain}[v1]"
    else:
        g = f"[0:v]{pre},{crop_chain}[v1]"
    # setpts first: frame 0 at time 0 exactly, so captions (timed from the
    # clip start) and the encoder's frame slots both line up with `n`.
    tail = 'setpts=PTS-STARTPTS,' + (f'ass={ass_name},' if ass_name else '') + 'format=yuv420p,setsar=1'
    return f'{g};[v1]{tail}[vout]'


# --------------------------------------------------------------------------
# Captions
# --------------------------------------------------------------------------

SUBTITLE_SIZES = {'s': (52, 32), 'm': (64, 26), 'l': (78, 21)}   # font px on a 1920-high canvas, chars per cue


def subtitle_cues(words, segments, start, end, max_chars=26, max_words=6, max_dur=2.8, gap_break=0.6):
    """Caption cues for the clip [start, end], times relative to its start.

    Short cues (a few words, one line where possible) because that is what
    reads on a phone: a full broadcast-style two-line subtitle is too small
    at this width and sits on screen long enough to give the line away
    before it is spoken. Built from word timings when available so each cue
    appears with the words it shows; from line timings otherwise, split
    evenly by length."""
    dur = float(end - start)
    cues = []
    inner = [w for w in words or [] if start - 0.05 <= (w['start'] + w['end']) / 2.0 <= end + 0.05]
    if inner:
        cur = []

        def flush():
            if cur:
                cues.append({'start': cur[0]['start'] - start, 'end': cur[-1]['end'] - start,
                             'text': ' '.join(x['word'] for x in cur)})
                del cur[:]

        for w in inner:
            if cur:
                text_len = sum(len(x['word']) + 1 for x in cur) + len(w['word'])
                if (text_len > max_chars or len(cur) >= max_words
                        or w['start'] - cur[-1]['end'] > gap_break
                        or w['end'] - cur[0]['start'] > max_dur
                        or _SENTENCE_END.search(cur[-1]['word'])):
                    flush()
            cur.append(w)
        flush()
    else:
        for sg in segments or []:
            a, b = max(sg['start'], start), min(sg['end'], end)
            if b - a < 0.2:
                continue
            chunks, line = [], ''
            for tok in sg['text'].split():
                if line and len(line) + 1 + len(tok) > max_chars:
                    chunks.append(line)
                    line = tok
                else:
                    line = (line + ' ' + tok).strip()
            if line:
                chunks.append(line)
            total = float(sum(len(c) for c in chunks)) or 1.0
            t = a
            for c in chunks:
                d = (b - a) * len(c) / total
                cues.append({'start': t - start, 'end': t + d - start, 'text': c})
                t += d
    out = []
    for i, c in enumerate(cues):
        s, e = max(0.0, c['start']), min(dur, c['end'])
        nxt = max(0.0, cues[i + 1]['start']) if i + 1 < len(cues) else dur
        if nxt - e < 0.25:
            e = nxt                       # butt up to the next cue: no one-frame blink between them
        elif e - s < 0.6:
            e = min(nxt - 0.02, s + 0.6)  # hold a very short word long enough to read
        if e - s >= 0.08 and c['text'].strip():
            out.append({'start': round(s, 3), 'end': round(e, 3), 'text': c['text'].strip()})
    return out


def _ass_time(t):
    cs = int(round(max(0.0, t) * 100))
    return f'{cs // 360000}:{(cs // 6000) % 60:02d}:{(cs // 100) % 60:02d}.{cs % 100:02d}'


def _srt_time(t):
    ms = int(round(max(0.0, t) * 1000))
    return f'{ms // 3600000:02d}:{(ms // 60000) % 60:02d}:{(ms // 1000) % 60:02d},{ms % 1000:03d}'


def write_ass(cues, path, size='m', font='Arial', out_w=OUT_W, out_h=OUT_H):
    """Writes the burn-in caption file. Bold white with a heavy black
    outline, bottom-centre but lifted well clear of the bottom edge, where
    every short-video app draws its own caption and buttons."""
    font_px = SUBTITLE_SIZES.get(size, SUBTITLE_SIZES['m'])[0]
    font = re.sub(r'[,\r\n]', ' ', str(font or 'Arial')).strip() or 'Arial'
    lines = [
        '[Script Info]', 'ScriptType: v4.00+', f'PlayResX: {out_w}', f'PlayResY: {out_h}',
        'WrapStyle: 0', 'ScaledBorderAndShadow: yes', '',
        '[V4+ Styles]',
        'Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, '
        'Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, '
        'Alignment, MarginL, MarginR, MarginV, Encoding',
        f'Style: Cap,{font},{font_px},&H00FFFFFF,&H000000FF,&H00000000,&H96000000,-1,0,0,0,100,100,0,0,1,'
        f'{max(3, font_px // 15)},2,2,80,80,{int(out_h * 0.23)},1', '',
        '[Events]',
        'Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text',
    ]
    for c in cues:
        text = c['text'].replace('\\', '/').replace('{', '(').replace('}', ')')
        text = ' '.join(text.split())
        lines.append(f"Dialogue: 0,{_ass_time(c['start'])},{_ass_time(c['end'])},Cap,,0,0,0,,{text}")
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def write_srt(cues, path):
    """Sidecar captions, for platforms that take an uploaded caption file."""
    with open(path, 'w', encoding='utf-8') as f:
        for i, c in enumerate(cues, 1):
            f.write(f"{i}\n{_srt_time(c['start'])} --> {_srt_time(c['end'])}\n{c['text']}\n\n")


# --------------------------------------------------------------------------
# Probing and rendering
# --------------------------------------------------------------------------

class ToolTimeout(RuntimeError):
    """ffmpeg/ffprobe ran past its timeout and was killed."""


def run_tool(cmd, timeout, cwd=None, label='ffmpeg'):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, errors='replace',
                              timeout=timeout, cwd=cwd)
    except subprocess.TimeoutExpired as e:
        raise ToolTimeout(f'{label} exceeded {timeout}s') from e


def _parse_ratio(v):
    try:
        a, b = str(v).replace('/', ':').split(':')
        a, b = float(a), float(b)
        return a / b if a > 0 and b > 0 else None
    except (ValueError, AttributeError):
        return None


def probe_source(ffprobe, path, timeout=30):
    """Everything the planner and renderer need to know about the source.

    Frame rate, size and frame count come from OpenCV rather than ffprobe on
    purpose: the cut list was produced by PySceneDetect on OpenCV's frame
    numbering, and the render addresses frames by that same numbering, so
    the two must agree on what 'frame 1500' is."""
    cap = cv2.VideoCapture(path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    info = {'width': w, 'height': h, 'fps': fps, 'frames': frames,
            'duration': (frames / fps) if fps > 0 else 0.0,
            'sar': 1.0, 'audio_index': None, 'color_space': None, 'v_offset': 0.0}
    try:
        r = run_tool([ffprobe, '-v', 'error', '-show_entries',
                      'stream=codec_type,channels,color_space,sample_aspect_ratio,start_time:format=start_time',
                      '-of', 'json', path], timeout, label='ffprobe')
        probed = json.loads(r.stdout or '{}')
        streams = probed.get('streams', [])
    except (ToolTimeout, ValueError, OSError):
        probed, streams = {}, []
    a_idx, best = -1, 0
    seen_video = False
    for s in streams:
        if s.get('codec_type') == 'video' and not seen_video:
            seen_video = True
            info['color_space'] = (s.get('color_space') or '').lower() or None
            # How far into the file the first video frame sits. ffmpeg's -ss
            # counts from the start of the FILE (its earliest stream), frame
            # numbers count from the first VIDEO frame, and the two differ
            # whenever audio starts first -- commonly by 20-80 ms, i.e. a
            # frame or two. Ignoring it starts every render that many frames
            # early, which shows as a one-frame flash of the wrong crop at
            # each cut (seen on a concat-joined source, where it was 21 ms).
            try:
                off = float(s.get('start_time')) - float((probed.get('format') or {}).get('start_time'))
                if 0 < off < 10:
                    info['v_offset'] = off
            except (TypeError, ValueError):
                pass
            sar = _parse_ratio(s.get('sample_aspect_ratio'))
            if sar and 0.5 < sar < 2.5:
                info['sar'] = sar
        elif s.get('codec_type') == 'audio':
            a_idx += 1
            ch = int(s.get('channels') or 0)
            if ch > best:      # same pick ffmpeg makes by default: the stream with the most channels
                best, info['audio_index'] = ch, a_idx
    info['disp_w'] = even(w * info['sar']) if w else 0
    info['disp_h'] = even(h) if h else 0
    cs = info['color_space']
    info['sd_matrix'] = cs in ('smpte170m', 'bt470bg') or (cs in (None, 'unknown') and 0 < h < 720)
    info['tag_709'] = cs in (None, 'unknown', 'bt709') or info['sd_matrix']
    return info


def build_render_cmd(ffmpeg, src, out_path, start_f, n_frames, info, segs, ass_name=None,
                     crf=18, preset='medium', loudness=-14.0, true_peak=-1.5):
    fps = info['fps']
    dur = n_frames / fps
    # A quarter of a frame early, so accurate seek lands on exactly start_f
    # whatever way the float rounds: the first frame ffmpeg keeps is the
    # first whose timestamp is >= -ss. Not half a frame: that leaves the
    # first frame's timestamp exactly between two output slots, and the
    # encoder then sometimes rounds it up and pads slot 0 with a duplicate
    # (measured -- which also pushes the last frame off the end).
    ss = max(0.0, info.get('v_offset', 0.0) + (start_f - 0.25) / fps)
    cmd = [ffmpeg, '-y', '-hide_banner', '-nostats', '-loglevel', 'error',
           '-ss', f'{ss:.6f}', '-i', src, '-t', f'{dur:.6f}', '-frames:v', str(int(n_frames)),
           '-filter_complex', build_filtergraph(info, segs, ass_name=ass_name), '-map', '[vout]']
    if info.get('audio_index') is not None:
        fade_out = max(0.0, dur - 0.12)
        cmd += ['-map', f"0:a:{info['audio_index']}",
                '-af', f'afade=t=in:st=0:d=0.04,afade=t=out:st={fade_out:.3f}:d=0.12,'
                       f'loudnorm=I={loudness}:TP={true_peak}:LRA=11',
                '-c:a', 'aac', '-b:a', '192k', '-ar', '48000', '-ac', '2']
    else:
        cmd += ['-an']
    cmd += ['-c:v', 'libx264', '-preset', str(preset), '-crf', str(crf), '-pix_fmt', 'yuv420p',
            '-profile:v', 'high']
    if info.get('tag_709'):
        cmd += ['-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709']
    cmd += ['-movflags', '+faststart', out_path]
    return cmd


# What ffmpeg prints AFTER the line that says what went wrong: every thread
# reporting that it stopped, and the muxer noting it wrote nothing.
_FFMPEG_AFTERMATH = re.compile(r'Task finished with error code|Terminating thread with return code|'
                               r'Could not open encoder before EOF|Nothing was written into output file|'
                               r'Error (re)?initializing filters|Error while filtering|Conversion failed')


def ffmpeg_error(stderr, limit=600):
    """The part of a failed ffmpeg run's output that names the cause.

    The cause is the FIRST thing ffmpeg prints; half a dozen lines of
    aftermath follow. Keeping the last 600 characters, as this used to,
    showed an editor only the aftermath ("Task finished with error code
    -22") with the reason cut off the front. So the aftermath is dropped
    and what is left is kept from the start, with the pointer values
    ("@ 000001b7a24b4f40") that pad every line taken out."""
    lines = [re.sub(r'\s*@ (0x)?[0-9a-fA-F]{6,}', '', ln).strip() for ln in (stderr or '').splitlines()]
    lines = [ln for ln in lines if ln]
    cause = [ln for ln in lines if not _FFMPEG_AFTERMATH.search(ln)]
    text = ' | '.join(cause or lines)
    return text if len(text) <= limit else text[:limit - 1].rstrip() + '…'


def render_short(ffmpeg, src, out_path, start_f, n_frames, info, segs, ass_name=None, work_dir=None,
                 crf=18, preset='medium', loudness=-14.0, true_peak=-1.5, timeout=900):
    """Renders one short. Returns (ok, error_text).

    Runs ffmpeg with `work_dir` as its working directory and refers to the
    caption file by bare name. That sidesteps the filtergraph's path
    escaping entirely -- a Windows path's drive colon and backslashes each
    need a different number of escapes inside -filter_complex, and getting
    it wrong fails only on Windows, i.e. only in production."""
    # Absolute, because ffmpeg is about to run from a different directory.
    src, out_path = os.path.abspath(src), os.path.abspath(out_path)
    cmd = build_render_cmd(ffmpeg, src, out_path, start_f, n_frames, info, segs, ass_name=ass_name,
                           crf=crf, preset=preset, loudness=loudness, true_peak=true_peak)
    try:
        r = run_tool(cmd, timeout, cwd=work_dir, label='shorts render')
    except ToolTimeout:
        try:                       # don't leave a half-written MP4 behind in the batch folder
            if os.path.exists(out_path):
                os.remove(out_path)
        except OSError:
            pass
        raise
    if r.returncode != 0 or not (os.path.exists(out_path) and os.path.getsize(out_path) > 0):
        try:
            if os.path.exists(out_path):
                os.remove(out_path)
        except OSError:
            pass
        return False, ffmpeg_error(r.stderr) or f'ffmpeg exited with code {r.returncode}'
    return True, None


def has_filter(ffmpeg, name, timeout=20):
    """Whether this ffmpeg build has filter `name`. Caption burn-in needs
    libass, which not every build includes -- checked up front so a build
    without it renders clean, caption-less shorts with a warning instead of
    failing every render."""
    try:
        r = run_tool([ffmpeg, '-hide_banner', '-filters'], timeout, label='ffmpeg -filters')
    except (ToolTimeout, OSError):
        return False
    return bool(re.search(r'^\s*\S+\s+' + re.escape(name) + r'\s', r.stdout or '', re.M))

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
import time

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


# How long a line of so many words can plausibly take to say, generously:
# slow, with pauses. A line that claims far more than that did not take it.
def speaking_time(n_words):
    return 1.5 + 0.8 * max(1, int(n_words))


LONG_WORD = 2.0             # seconds; no single word takes this long to say


def trim_absorbed_silence(start, end, n_words):
    """The start of a line of `n_words` words timed start..end, moved later
    if the line cannot really have begun there.

    Reported: a caption on screen for thirty seconds before anyone spoke.
    Whisper places a line's END well. Its START, after a stretch with no
    speech -- music, a look, a walk across a room -- is often the start of
    that stretch instead: "Oh, my God." timed 0:00 to 0:30. Everything
    downstream then believes the line began there: its caption comes up 30
    seconds early, and a short that opens on it opens on half a minute of
    nothing.

    So a line that lasts far longer than its words could take is taken to
    end where it says and to have started as late as they allow. Only when
    it is far out -- more than double, and by more than three seconds -- so
    a line that really is delivered slowly is left exactly as timed."""
    span = speaking_time(n_words)
    if end - start > max(2.0 * span, span + 3.0):
        return end - span
    return start


def normalize_transcript(words, segments, max_line=12.0):
    """Cleans a (words, segments) pair from the speech service into sorted,
    well-formed lists, splits over-long segments, and corrects starts that
    cannot be right (see trim_absorbed_silence).

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
            # The same fault at the level of a word: the first one after a
            # silence given the whole silence as its length.
            if e - s > LONG_WORD:
                s = e - 1.0
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
        # Lines only. Their own word count is all there is to judge them by.
        for sg in segs:
            sg['start'] = trim_absorbed_silence(sg['start'], sg['end'], len(sg['text'].split()))
        return w, segs

    starts = [x['start'] for x in w]
    out = []
    for sg in segs:
        lo = bisect.bisect_left(starts, sg['start'] - 1.0)
        hi = bisect.bisect_right(starts, sg['end'])
        # By midpoint, so a word of the neighbouring line that merely starts
        # close to this one's edge isn't pulled into it.
        inner = [x for x in w[lo:hi] if sg['start'] - 0.02 <= (x['start'] + x['end']) / 2.0 <= sg['end'] + 0.02]
        if inner:
            # The words say when it was spoken; a line that starts well
            # before its first word has taken in the silence ahead of it.
            if inner[0]['start'] - sg['start'] > 1.0:
                sg = dict(sg, start=inner[0]['start'])
        else:
            sg = dict(sg, start=trim_absorbed_silence(sg['start'], sg['end'], len(sg['text'].split())))
        if sg['end'] - sg['start'] <= max_line:
            out.append(sg)
            continue
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
    'And say what kind of picture it is:\n'
    '"story" = a scene of the drama itself (on-screen text over a scene, such as a place name, is still story);\n'
    '"credits" = credits, a title card, or text or a graphic filling the screen;\n'
    '"logo" = a sponsor billboard, a product, a station or programme logo, an advertisement;\n'
    '"host" = a host, presenter or narrator speaking to the camera or in a studio, not part of the drama;\n'
    '"black" = a black or blank screen.\n'
    'Reply with JSON only: {"score": <1-5>, "desc": "<description>", "kind": "<kind>"}'
)

# What the vision model can say a frame is. Anything but "story" is material
# a short must not contain: billboards, credits, a host, black between acts.
VISION_KINDS = ('story', 'credits', 'logo', 'host', 'black')

VISION_FORMAT = {
    'type': 'object',
    'properties': {'score': {'type': 'integer', 'minimum': 1, 'maximum': 5},
                   'desc': {'type': 'string'},
                   'kind': {'type': 'string', 'enum': list(VISION_KINDS)}},
    'required': ['score', 'desc', 'kind'],
}


_KIND_WORDS = {'story': 'story', 'scene': 'story', 'drama': 'story',
               'credits': 'credits', 'credit': 'credits', 'title': 'credits', 'text': 'credits', 'graphic': 'credits',
               'logo': 'logo', 'billboard': 'logo', 'sponsor': 'logo', 'ad': 'logo', 'advertisement': 'logo',
               'host': 'host', 'presenter': 'host', 'narrator': 'host', 'anchor': 'host',
               'black': 'black', 'blank': 'black'}


def parse_vision_kind(text):
    """The kind of picture a vision reply says a frame is (VISION_KINDS), or
    None when it does not say -- which is taken as story, never as a reason
    to leave anything out."""
    text = (text or '').strip()
    raw = None
    if text.startswith('{'):
        try:
            raw = json.loads(text).get('kind')
        except (ValueError, TypeError, AttributeError):
            raw = None
    if raw is None:
        m = re.search(r'"kind"\s*:\s*"([^"]*)', text)
        raw = m.group(1) if m else None
    return _KIND_WORDS.get(' '.join(str(raw or '').lower().split()).strip(' ."'))


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
        'not_story': {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'first_id': {'type': 'integer'},
                    'last_id': {'type': 'integer'},
                    'kind': {'type': 'string', 'enum': ['billboard', 'credits', 'narration', 'host',
                                                        'recap', 'teaser', 'sponsor']},
                },
                'required': ['first_id', 'last_id', 'kind'],
            },
        },
    },
    'required': ['moments', 'not_story'],
}


def build_story_prompt(segments, lo, hi, visual, min_dur, max_dur, max_moments=3, focus=None, avoid=None,
                       breaks=None):
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
            seen = f", looks like {v['kind']}" if v.get('kind') and v['kind'] != 'story' else ''
            rows.append((v['t'], 1, f"[SCREEN {fmt_ts(v['t'])}] {desc} (visual drama {v['score']}/5{seen})"))
    # Where one part of a multi-part episode ends and the next begins.
    near = [(t, label) for t, label in breaks or [] if t_lo <= t <= t_hi]
    for t, label in near:
        rows.append((t, -1, f'[PART BREAK] {label} begins here; time has passed and nothing runs across this line'))
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
        '- does not start or end in the middle of a sentence or a thought,\n'
        + ('- never includes a [PART BREAK] line: this is a multi-part episode and a short stays inside one part,\n'
           if near else '') +
        '- is the drama itself and nothing else (see below).\n'
        'Do not pick greetings or small talk. If nothing in this stretch qualifies, return an empty '
        'list -- an empty list is a good answer.\n'
        '\nNOT STORY. A broadcast episode also carries material that is not the drama, and a short must '
        'never contain any of it. List every such run of lines under "not_story" with its first and '
        'last line ID and its kind:\n'
        '- billboard: an opening or closing billboard (OBB/CBB), "this program is brought to you by", '
        '"hatid sa inyo ng", sponsor and product mentions;\n'
        '- sponsor: an advertisement or a sponsor read inside the programme;\n'
        '- credits: opening titles, theme song over titles, end credits;\n'
        '- narration: a narrator or voice-over telling the story rather than a character in a scene;\n'
        '- host: a host or presenter speaking to the audience (introducing the episode, a studio '
        'segment, a closing message);\n'
        '- recap: "previously" / "sa nakaraang" summaries of earlier episodes;\n'
        '- teaser: "next time" / "abangan" previews and plugs for other shows.\n'
        'A moment may sit between such runs but must not include any line of them. If there is none, '
        'return "not_story": [].\n'
        + extra +
        '\nFor each moment give: start_id and end_id (the first and last line IDs, inclusive), '
        'title (at most 8 words, in the language of the dialogue), hook (the line or idea that '
        'grabs attention), why (one sentence), and score (1-10: how strong it is as a stand-alone '
        'dramatic short).\n'
        'Reply with JSON only: {"moments":[{"start_id":0,"end_id":0,"title":"","hook":"","why":"","score":0}],'
        '"not_story":[{"first_id":0,"last_id":0,"kind":"billboard"}]}\n\n'
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


NOT_STORY_LABELS = {'billboard': 'billboard', 'sponsor': 'sponsor read', 'credits': 'credits',
                    'narration': 'narration', 'host': 'host', 'recap': 'recap', 'teaser': 'teaser',
                    'logo': 'billboard / logo', 'black': 'black', 'editor': 'left out by you'}


def parse_story_skips(text, lo, hi):
    """The runs of lines a story reply says are not the drama (billboards,
    credits, narration, a host, recaps, teasers): [{'start_id', 'end_id',
    'kind'}], kept to the chunk asked about."""
    text = re.sub(r'^```[a-zA-Z]*\s*|\s*```$', '', (text or '').strip())
    items = None
    a, b = text.find('{'), text.rfind('}')
    for c in (text, text[a:b + 1] if 0 <= a < b else ''):
        try:
            data = json.loads(c)
        except ValueError:
            continue
        if isinstance(data, dict) and isinstance(data.get('not_story'), list):
            items = data['not_story']
            break
    if items is None:
        # Truncated: whichever of its objects are intact.
        items = []
        for m in re.finditer(r'\{[^{}]*"first_id"[^{}]*\}', text):
            try:
                items.append(json.loads(m.group(0)))
            except ValueError:
                pass
    out = []
    for o in items:
        if not isinstance(o, dict):
            continue
        kind = str(o.get('kind') or '').strip().lower()
        if kind not in NOT_STORY_LABELS or kind in ('logo', 'black', 'editor'):
            continue
        try:
            x, y = int(float(o.get('first_id'))), int(float(o.get('last_id')))
        except (TypeError, ValueError):
            continue
        if x > y:
            x, y = y, x
        if y < lo or x >= hi:
            continue
        out.append({'start_id': max(x, lo), 'end_id': min(y, hi - 1), 'kind': kind})
    return out


def parse_ranges(text, duration):
    """An editor's "leave out" list -- '0:00-1:30, 44:10-end' -- as
    [(start, end)] seconds. (ranges, None) or (None, why)."""
    out = []
    for part in re.split(r'[,;\n]+', text or ''):
        part = part.strip()
        if not part:
            continue
        m = re.match(r'^(\S+)\s*(?:-|–|—|to)\s*(\S+)$', part)
        if not m:
            return None, f'"{part}" is not a range: write it as start-end, e.g. 0:00-1:30 or 44:10-end.'
        try:
            a = _parse_clock(m.group(1))
            b = float(duration) if m.group(2).lower() in ('end', 'dulo') else _parse_clock(m.group(2))
        except ValueError:
            return None, f'"{part}" is not a range: write times as m:ss or h:mm:ss.'
        if not 0 <= a < b:
            return None, f'"{part}": the end must be after the start.'
        out.append((a, min(b, float(duration))))
    return out, None


def _parse_clock(t):
    t = t.strip()
    if not re.match(r'^\d+(:\d{1,2}){0,2}(\.\d+)?$', t):
        raise ValueError(t)
    secs = 0.0
    for p in t.split(':'):
        secs = secs * 60 + float(p)
    return secs


def describe_ranges(ranges, most=6):
    """'billboard 0:00-0:42; credits 44:10-45:30; ...' for a warning."""
    bits = [f"{' / '.join(NOT_STORY_LABELS.get(k, k) for k in kinds)} {fmt_ts(a)}-{fmt_ts(b)}"
            for a, b, kinds in ranges[:most]]
    more = len(ranges) - most
    return '; '.join(bits) + (f'; and {more} more' if more > 0 else '')


def merge_ranges(ranges):
    """Overlapping or touching (start, end, kind) ranges joined; the kinds
    of a joined range are kept in order, without repeats."""
    out = []
    for a, b, kind in sorted(ranges, key=lambda r: (r[0], r[1])):
        if out and a <= out[-1][1] + 0.05:
            if b > out[-1][1]:
                out[-1][1] = b
            if kind not in out[-1][2]:
                out[-1][2].append(kind)
        else:
            out.append([a, b, [kind]])
    return [(a, b, kinds) for a, b, kinds in out]


NOT_STORY_FRAME_REACH = 8.0     # a non-story frame in a long shot stands for this much either side of it
NOT_STORY_BRIDGE = 20.0         # two non-story frames this close, with no story frame between, are one run


def not_story_ranges(skips, segments, visual, shots, editor=None):
    """Everything a short must not contain, as merged (start, end, kinds):
    the lines the story model said are not the drama, the shots the vision
    model saw as credits, a billboard, a host or black, and the ranges the
    editor left out.

    A frame stands for its shot, but no more than NOT_STORY_FRAME_REACH
    either side of it (a credits roll with no cut in it would otherwise be
    one frame wide, and one misread frame in a long scene would take the
    whole scene). Two such frames close together with no story frame
    between them are taken as one run: credits do not stop for a few
    seconds in the middle."""
    rng = []
    n = len(segments or [])
    for k in skips or []:
        i, j = max(0, k['start_id']), min(n - 1, k['end_id'])
        if i <= j:
            # The whole run, the pauses inside it included.
            rng.append((float(segments[i]['start']), float(segments[j]['end']), k['kind']))
    shots = sorted(shots or [])
    frames = sorted(visual or [], key=lambda v: v['t'])
    prev = None
    for v in frames:
        kind = v.get('kind')
        if not kind or kind == 'story':
            prev = None
            continue
        t = float(v['t'])
        a, b = t - NOT_STORY_FRAME_REACH, t + NOT_STORY_FRAME_REACH
        for s0, s1 in shots:
            if s0 <= t < s1:
                a, b = max(a, s0), min(b, s1)
                break
        if prev is not None and t - prev[0] <= NOT_STORY_BRIDGE:
            a = min(a, prev[1])
        rng.append((a, b, kind))
        prev = (t, b)
    for a, b in editor or []:
        rng.append((float(a), float(b), 'editor'))
    return merge_ranges(rng)


def _overlap(a, b, ranges):
    return sum(max(0.0, min(b, r[1]) - max(a, r[0])) for r in ranges or [])


def allowed_part(start, end, blocked, prefer=None):
    """The part of [start, end] clear of every blocked range that keeps the
    most of `prefer` (start, end -- the speech chosen), or the longest part
    if there is none to keep. None when nothing is clear."""
    parts, a = [], float(start)
    for r0, r1, *_ in sorted(blocked or []):
        if r1 <= a or r0 >= end:
            continue
        if r0 > a:
            parts.append((a, r0))
        a = max(a, r1)
    if a < end:
        parts.append((a, float(end)))
    parts = [p for p in parts if p[1] - p[0] > 0.05]
    if not parts:
        return None
    if prefer:
        p0, p1 = prefer
        return max(parts, key=lambda p: (max(0.0, min(p[1], p1) - max(p[0], p0)), p[1] - p[0]))
    return max(parts, key=lambda p: p[1] - p[0])


def blocked_lines(segments, blocked):
    """Indices of the lines mostly (half or more) inside a blocked range."""
    out = set()
    for k, sg in enumerate(segments or []):
        d = max(float(sg['end']) - float(sg['start']), 1e-3)
        if _overlap(float(sg['start']), float(sg['end']), blocked) >= 0.5 * d:
            out.add(k)
    return out


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


def ask_vision(base_url, model, jpeg_b64, num_predict=200, timeout=180, with_kind=False):
    """Rates one frame. Returns (score or None, description), and with
    `with_kind` also the kind of picture (VISION_KINDS, or None)."""
    payload = {'model': model, 'prompt': VISION_PROMPT, 'stream': False, 'images': [jpeg_b64],
               'think': False, 'options': {'temperature': 0.1, 'num_predict': num_predict}}
    text, data = _generate(base_url, payload, VISION_FORMAT, timeout)
    score, desc = parse_vision_reply(text or (data.get('thinking') or ''))
    kind = parse_vision_kind(text or (data.get('thinking') or ''))
    if score is None:
        # A reasoning model that spent its budget thinking leaves `response`
        # empty. One retry, unconstrained and with room to finish.
        payload['options'] = {'temperature': 0.1, 'num_predict': num_predict * 3}
        payload.pop('think', None)
        text, data = ollama_generate(base_url, payload, timeout)
        score, desc2 = parse_vision_reply(text or (data.get('thinking') or ''))
        desc = desc or desc2
        kind = kind or parse_vision_kind(text or (data.get('thinking') or ''))
    return (score, desc, kind) if with_kind else (score, desc)


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

def fit_indices(segments, i, j, min_dur, max_dur, gap_limit=6.0, far_gap=10.0, blocked=None, walls=None):
    """Grows or shrinks the line range [i, j] to respect the length limits.

    Too long: lines are dropped from the START. The model picked this
    stretch for where it lands -- the turn, the reaction, the cliffhanger --
    and a short that opens in the middle of an argument still works where
    one that stops before the payoff does not.

    Too short: neighbouring lines are added, earlier context first, then
    alternating, without crossing a pause longer than `gap_limit` -- a gap
    that long is often a scene change. If that is not enough the minimum
    still has to be met, so longer pauses (up to `far_gap`) are crossed too,
    the shorter one first: a look held for ten seconds is a pause in a
    scene more often than the end of one. What still cannot reach the
    minimum is flagged 'short', and the caller leaves it out. Lines in
    `blocked` (indices: billboards, credits, narration...) are never added."""
    blocked = blocked or ()
    walls = walls or ()

    def wall(p, q):
        # A part break between line p and the later line q.
        return any(segments[p]['end'] <= w + 1e-6 and w <= segments[q]['start'] + 1e-6 for w in walls)

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
        can_back = (i > 0 and i - 1 not in blocked and not wall(i - 1, i)
                    and segments[i]['start'] - segments[i - 1]['end'] <= gap_limit and dur(i - 1, j) <= max_dur)
        can_fwd = (j < n - 1 and j + 1 not in blocked and not wall(j, j + 1)
                   and segments[j + 1]['start'] - segments[j]['end'] <= gap_limit and dur(i, j + 1) <= max_dur)
        if can_back and (back_first or not can_fwd):
            i -= 1
        elif can_fwd:
            j += 1
        else:
            break
        if 'extended' not in flags:
            flags.append('extended')
        back_first = not back_first
    while dur(i, j) < min_dur:
        # Across longer pauses, whichever side has the shorter one.
        back = (segments[i]['start'] - segments[i - 1]['end']
                if i > 0 and i - 1 not in blocked and not wall(i - 1, i) and dur(i - 1, j) <= max_dur else None)
        fwd = (segments[j + 1]['start'] - segments[j]['end']
               if j < n - 1 and j + 1 not in blocked and not wall(j, j + 1) and dur(i, j + 1) <= max_dur else None)
        sides = [(g, side) for g, side in ((back, 'back'), (fwd, 'fwd')) if g is not None and g <= far_gap]
        if not sides:
            flags.append('short')
            break
        if min(sides)[1] == 'back':
            i -= 1
        else:
            j += 1
        for f in ('extended', 'bridged'):
            if f not in flags:
                flags.append(f)
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


def window_scores(start, end, story_score, visual, words, segments, sound=None):
    """Component scores for a finished window, plus the 0-100 total.

    Story carries the most weight because it is the only component that
    knows whether the clip means anything; visual drama is next; pace
    (words per second) is a small tie-breaker that penalises both dead air
    and wall-to-wall talking. Missing components are left out of the
    weighted mean rather than counted as zero, so a fallback candidate
    with no story score is ranked on what IS known instead of sinking
    below every scored one by default. `sound` (0-1, see sound_score),
    when known, is how loud and lively it is against the rest of the
    episode -- a raised voice, a laugh, a crash -- and counts for a little.
    """
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
    if sound is not None:
        parts.append((SOUND_WEIGHT, float(sound)))
    total = sum(w * p for w, p in parts) / sum(w for w, _ in parts)
    return {'story': story_score,
            'visual': round(visual_score, 1) if visual_score is not None else None,
            'pace': round(pace, 2), 'score': int(round(total * 100))}


SOUND_WEIGHT = 0.10


def rescore(c, visual, words, segments, sound=None):
    """A built candidate's scores again, from (more) frame ratings and its
    sound. Its story score stands: it is the story model's."""
    r = window_scores(c['start'], c['end'], c.get('story_score'), visual, words, segments, sound=sound)
    c['visual_score'], c['pace'], c['score'] = r['visual'], r['pace'], r['score']
    c['sound_score'] = round(sound * 10.0, 1) if sound is not None else None
    return c


def inner_sample_times(cands, rated, budget, per_moment=4, min_gap=1.5):
    """Where to look inside each moment, so a moment is judged on its own
    frames and not the few the whole-episode sample happened to land in:
    {candidate index: [times]}, evenly through each, skipping times within
    `min_gap` of a frame already rated, `per_moment` at most each and
    `budget` in all (best-placed moments first, as they are listed)."""
    if not cands or budget <= 0:
        return {}
    k = max(1, min(int(per_moment), int(budget) // max(len(cands), 1)))
    have = sorted(float(t) for t in rated)
    out, used = {}, 0
    for n, c in enumerate(cands):
        if used >= budget:
            break
        a, b = float(c['start']), float(c['end'])
        picked = []
        for q in range(k):
            t = a + (b - a) * (q + 0.5) / k
            i = bisect.bisect_left(have, t)
            if any(abs(have[j] - t) < min_gap for j in (i - 1, i) if 0 <= j < len(have)):
                continue
            picked.append(round(t, 2))
        picked = picked[:max(0, int(budget) - used)]
        if picked:
            out[n] = picked
            used += len(picked)
    return out


SOUND_SPAN = 12.0           # dB above (below) the episode's usual loud that counts as 1 (0)


def sound_reference(db):
    """The episode's usual loud: the 90th percentile of its level, read
    over the whole file. None without sound."""
    if db is None or len(db) < 100:
        return None
    return float(np.percentile(np.asarray(db, dtype=np.float64), 90))


def sound_score(db, start, end, ref, hop=None):
    """0-1: how loud a stretch gets against the episode as a whole -- the
    90th percentile of its level, and how much of it rises clearly above
    the episode's usual loud (shouting, laughter, a slam). 0.5 is an
    ordinary stretch of this programme. None without sound to go by."""
    hop = hop or ENVELOPE_HOP
    if ref is None or db is None:
        return None
    a, b = int(max(0.0, start) / hop), int(max(0.0, end) / hop)
    win = np.asarray(db[a:b], dtype=np.float64)
    if len(win) < 50:
        return None
    level = float(np.percentile(win, 90))
    burst = float((win > ref + 3.0).mean())
    return round(min(1.0, max(0.0, 0.5 + (level - ref) / SOUND_SPAN + burst)), 3)


EDGE = {'window': 3.0}      # seconds of sound read either side of an in or out point


def _word_start(db, at, prev_speech, hop=None):
    """word_end run backwards: where the sound under way at `at` began --
    (seconds, kind) or None. The onset of the word an in point falls in,
    not the line before it (`prev_speech`, where that ended)."""
    hop = hop or ENVELOPE_HOP
    n = len(db)
    rev = np.asarray(db[::-1])
    last = (n - 1) * hop
    nxt = (last - prev_speech) if prev_speech is not None else None
    got = word_end(rev, last - at, nxt, hop=hop)
    if not got:
        return None
    t, kind = got
    return round(max(0.0, last - t), 3), kind


def refine_edges(cands, db, units, duration, min_dur, max_dur, fps, hop=None, blocked=None):
    """Moves each moment's in and out points onto the sound: an out point
    that falls inside a word is moved to where the word ends (its pause,
    or -- where the speech runs straight on -- the gap between two words
    nearest it); an in point inside a word goes back to where that word
    starts. The transcript's times are the transcript service's guess at
    the words; the sound is where they are. Moments are never taken past
    the maximum length, under the minimum, or into the next (previous)
    line's speech. Sets c['edges'] = {'in': kind, 'out': kind} for what
    was found ('quiet' -- nothing to do; 'pause', 'dip'; None -- the sound
    said nothing usable), and counts how many moved."""
    hop = hop or ENVELOPE_HOP
    moved = 0
    if db is None or not len(db):
        return moved
    starts = sorted(u[0] for u in units)
    ends = sorted(u[1] for u in units)
    span = EDGE['window']
    fps = float(fps) if fps and fps > 0 else 25.0
    for c in cands:
        start, end = float(c['start']), float(c['end'])
        edges = {'in': None, 'out': None}
        # Out point.
        lo = max(0.0, end - span)
        a, b = int(lo / hop), min(len(db), int((end + span) / hop))
        if b - a > 20:
            k = bisect.bisect_right(starts, end + 1e-3)
            nxt = starts[k] if k < len(starts) else None
            got = word_end(db[a:b], end - a * hop, (nxt - a * hop) if nxt is not None else None, hop=hop)
            if got:
                t, kind = got
                edges['out'] = kind
                t += a * hop
                if kind in ('pause', 'dip'):
                    t = min(t, start + max_dur, float(duration))
                    if (t - start >= min(min_dur, end - start) - 1e-6 and abs(t - end) > 0.5 / fps
                            and not _overlap(min(t, end), max(t, end), blocked)):
                        end = t
        # In point.
        a, b = int(max(0.0, start - span) / hop), min(len(db), int((start + span) / hop))
        if b - a > 20:
            k = bisect.bisect_left(ends, start - 1e-3) - 1
            prev = ends[k] if k >= 0 else None
            if prev is not None and prev < a * hop:
                prev = None
            got = _word_start(db[a:b], start - a * hop, (prev - a * hop) if prev is not None else None, hop=hop)
            if got:
                t, kind = got
                edges['in'] = kind
                t += a * hop
                if kind in ('pause', 'dip'):
                    t = max(t, end - max_dur, 0.0)
                    if (end - t >= min(min_dur, end - start) - 1e-6 and abs(t - start) > 0.5 / fps
                            and not _overlap(min(t, start), max(t, start), blocked)):
                        start = t
        start = round(start * fps) / fps
        end = min(float(duration), round(end * fps) / fps)
        if end - start > max_dur + 1e-6:
            end -= 1.0 / fps
        if abs(start - c['start']) > 1e-3 or abs(end - c['end']) > 1e-3:
            moved += 1
            c['start'], c['end'] = round(start, 3), round(end, 3)
            c['duration'] = round(end - start, 2)
        c['edges'] = edges
    return moved


def clear_of(cands, blocked, min_dur, fps):
    """Moments built before something not-story was found inside them (by
    the frames rated inside the moments): each cut to its clean part, and
    left out when that is shorter than the minimum. Returns (kept, how
    many were left out)."""
    if not blocked:
        return cands, 0
    fps = float(fps) if fps and fps > 0 else 25.0
    kept, gone = [], 0
    for c in cands:
        part = allowed_part(c['start'], c['end'], blocked)
        if part is None:
            gone += 1
            continue
        a, b = math.ceil(part[0] * fps - 1e-6) / fps, math.floor(part[1] * fps + 1e-6) / fps
        if (a, b) != (c['start'], c['end']) and abs(a - c['start']) + abs(b - c['end']) > 1e-3:
            if b - a < min_dur - 0.5:
                gone += 1
                continue
            c['start'], c['end'], c['duration'] = round(a, 3), round(b, 3), round(b - a, 2)
            if 'cleaned' not in c['flags']:
                c['flags'].append('cleaned')
        kept.append(c)
    return kept, gone


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


PAD_QUIET = 6.0             # seconds of non-speech a moment may be given, each side, to reach the minimum


def build_candidates(beats, segments, words, cuts, visual, duration,
                     min_dur=30.0, max_dur=90.0, limit=8, fps=25.0, min_story=None, report=None, blocked=None,
                     walls=None):
    """Beats (line ranges, or raw time windows from visual_windows) ->
    ranked, de-duplicated candidates with exact frame-aligned in/out points.

    `min_story` is for "as many as are worth making" rather than "up to N":
    of the moments the story model scored, those below it are left out.
    Moments it did not score (a fallback was used) are not judged by it, and
    if nothing reaches it everything found is kept -- a list of weak moments
    to review is more use than an empty one.

    The minimum length is a minimum. A moment whose lines fall short of it
    is first given the quiet around it, up to the next speech on either
    side and PAD_QUIET seconds at most; one that still falls short is left out. (Again unless that leaves
    nothing: then the short ones are listed, flagged.) `report`, a dict if
    given, is told how many were left out ('too_short') and whether short
    ones had to be kept ('kept_short').

    `blocked` is what is not the drama -- billboards, credits, narration, a
    host, recaps, black (see not_story_ranges): (start, end, kinds) ranges
    no moment may include any of. A proposal running into one keeps the
    longest clean run of its lines; lines are never added from one; and the
    lead-in, tail and padding stop at its edge. ('cleaned' flags a moment
    that lost something to this; report['not_story'] counts the proposals
    that had nothing clean left.)"""
    units = speech_units(words, segments)
    blocked = list(blocked or [])
    bad_lines = blocked_lines(segments, blocked) if blocked else set()
    # Where one part of a multi-part source ends and the next begins, on the
    # joined timeline: nothing runs across it. Held as hairline blocked
    # ranges as well, so lead-ins, tails and padding stop at it.
    walls = sorted(float(w) for w in walls or [])
    blocked = blocked + [(w - 0.001, w + 0.001, ['part break']) for w in walls]

    def wall_between(p, q):
        return any(segments[p]['end'] <= w + 1e-6 and w <= segments[q]['start'] + 1e-6 for w in walls)
    gone = 0
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
            i0, j0 = max(0, min(b['start_id'], b['end_id'])), min(len(segments) - 1, max(b['start_id'], b['end_id']))
            cleaned = False
            if (bad_lines and any(k in bad_lines for k in range(i0, j0 + 1))) or \
                    (walls and any(wall_between(k - 1, k) for k in range(i0 + 1, j0 + 1))):
                # The longest clean run of the lines chosen (later on a tie:
                # the end is what the moment was picked for).
                runs, run = [], []
                for k in range(i0, j0 + 1):
                    if k in bad_lines:
                        if run:
                            runs.append(run)
                        run = []
                    else:
                        if run and wall_between(k - 1, k):
                            runs.append(run)
                            run = []
                        run.append(k)
                if run:
                    runs.append(run)
                if not runs:
                    gone += 1
                    continue
                best = max(runs, key=lambda r: (segments[r[-1]]['end'] - segments[r[0]]['start'], r[0]))
                i0, j0, cleaned = best[0], best[-1], True
            i, j, flags = fit_indices(segments, i0, j0, min_dur, max_dur, blocked=bad_lines, walls=walls)
            if cleaned:
                flags.append('cleaned')
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
            if end - start < min_dur:
                # Short of the minimum by a little: the quiet after it, then
                # before it -- as far as the next speech either way, and no
                # more than PAD_QUIET of it on a side. A few seconds of a
                # reaction is part of the moment; half a minute of nothing
                # is not a way to make sixteen seconds of talk a minute long.
                k = bisect.bisect_left([u[0] for u in units], t1 - 1e-3)
                room_after = (units[k][0] - 0.04 if k < len(units) else float(duration)) - end
                end += max(0.0, min(room_after, PAD_QUIET - (end - t1), min_dur - (end - start)))
                k = bisect.bisect_right(sorted(u[1] for u in units), t0 + 1e-3) - 1
                room_before = start - (sorted(u[1] for u in units)[k] + 0.04 if k >= 0 else 0.0)
                start -= max(0.0, min(room_before, PAD_QUIET - (t0 - start), min_dur - (end - start)))
            lines = segments[i:j + 1]
        if blocked:
            keep = (t0, t1) if 't0' not in b else (start, end)
            part = allowed_part(start, end, blocked, prefer=keep)
            if part is None:
                gone += 1
                continue
            if part != (start, end):
                lost = (part[0] - start) + (end - part[1])
                start, end = part
                if lost > 0.1 and 'cleaned' not in flags:
                    flags.append('cleaned')
        # Whole frames, so the render's frame maths starts from exact values.
        start = round(start * fps) / fps
        end = min(float(duration), round(end * fps) / fps)
        if 't0' not in b and end - start > max_dur + 1e-6:
            end -= 1.0 / fps          # rounding to whole frames tipped it over by one
        if end - start < 3.0:
            continue
        # Short is what it is once everything has been tried, not what its
        # lines alone came to.
        flags = [f for f in flags if f != 'short']
        if end - start < min_dur - 0.5:
            flags.append('short')
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
    long_enough = [c for c in out if 'short' not in c['flags']]
    if report is not None:
        report['not_story'] = gone
        report['too_short'] = len(out) - len(long_enough) if long_enough else 0
        report['kept_short'] = bool(out) and not long_enough
    out = dedupe_windows(long_enough or out)
    if min_story is not None:
        strong = [c for c in out if c['story_score'] is None or c['story_score'] >= min_story]
        out = strong or out
    out = out[:max(1, int(limit))]
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

    def detect_bodies(self, frame):
        """[(x, y, w, h)] of head-and-shoulders, in the frame's own pixels:
        for the shots where no face is found -- someone seen from behind,
        in profile, looking down. OpenCV's upper-body cascade, which ships
        with it. It sees backs as well as fronts, and it is not precise:
        what it finds is only used when it is found in the same place
        across a shot (see plan_reframe)."""
        if getattr(self, '_upper', None) is None:
            base = getattr(getattr(cv2, 'data', None), 'haarcascades', None)
            casc = cv2.CascadeClassifier(os.path.join(base, 'haarcascade_upperbody.xml')) if base else None
            self._upper = casc if casc is not None and not casc.empty() else False
        if self._upper is False:
            return []
        h, w = frame.shape[:2]
        k = min(1.0, self.max_dim / float(max(w, h)))
        small = cv2.resize(frame, (max(2, int(w * k)), max(2, int(h * k))),
                           interpolation=cv2.INTER_AREA) if k < 1.0 else frame
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        m = max(24, int(small.shape[0] * BODY_MIN_FRAC))
        return [(x / k, y / k, bw / k, bh / k) for (x, y, bw, bh) in self._upper.detectMultiScale(gray, 1.05, 3, minSize=(m, m))]

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


# A head-and-shoulders smaller than this share of the picture's height is
# someone in the background, not who the shot is of.
BODY_MIN_FRAC = 0.2


# ---- Where the picture is in focus ----
# Drama is shot with the subject sharp and the background soft. In a shot
# with no face to go by, where the picture is sharpest is where the person
# is: someone walking away, a back, a hand on a table.
FOCUS_STRENGTH = 1.3        # how much sharper than average the sharpest window must be to mean anything


def focus_point(frame, win_w, sar=1.0):
    """(centre x in DISPLAY pixels of the sharpest `win_w`-wide window of the
    picture, how much sharper it is than the average window), or None.

    Sharpness is the Laplacian -- fine detail -- summed down each column of
    a small grey copy, over the top 85% of the picture (the bottom is where
    graphics and captions go). Grain and noise are fine detail everywhere,
    which is why the answer comes with how much the best window stands out:
    a picture sharp all over has no answer worth taking."""
    h, w = frame.shape[:2]
    k = 320.0 / float(w)
    small = cv2.resize(frame, (320, max(8, int(round(h * k)))), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    col = lap[:int(lap.shape[0] * 0.85)].sum(axis=0)
    col = np.convolve(col, np.ones(5) / 5.0, mode='same')
    win = max(4, min(len(col) - 1, int(round(win_w / float(sar) * k))))
    sums = np.convolve(col, np.ones(win), mode='valid')
    if not len(sums) or float(sums.mean()) <= 1e-6:
        return None
    best = int(np.argmax(sums))
    return ((best + win / 2.0) / k) * sar, float(sums[best] / sums.mean())


def _focus_targets(focus, a, b, max_x, crop_w):
    """[(frame, x)] of where a shot with no face in it is in focus, if the
    readings agree: sharply enough in at least 40% of the frames sampled.
    None otherwise -- and a picture in focus all over, or a shot that cannot
    make up its mind, is better centre-cropped than put somewhere wrong."""
    ss = [(i, f) for i, f in focus or [] if a <= i <= b]
    if len(ss) < 3:
        return None
    good = [(i, f[0]) for i, f in ss if f and f[1] >= FOCUS_STRENGTH]
    if len(good) < max(2, int(math.ceil(0.4 * len(ss)))):
        return None
    xs = sorted(x for _, x in good)
    lo, hi = xs[len(xs) // 10], xs[-1 - len(xs) // 10]
    if hi - lo > 1.5 * crop_w:
        return None                     # all over the place: no one thing is in focus
    return good


def sample_faces(path, start_f, n_frames, fps, detector, step_sec=0.2, sar=1.0, mouth=False, bodies=None,
                 focus=None, focus_w=None):
    """Runs the detector over a clip at ~1/step_sec samples a second.

    Returns [(frame index relative to the clip start, [(cx, cy, w, h), ...])]
    with x already in DISPLAY pixels (multiplied by the pixel aspect ratio),
    so the planner never has to think about anamorphic sources. Frames
    between samples are grab()bed, not decoded to an image, which is what
    keeps this several times faster than real time.

    With mouth=True each face gains a fifth value: mouth_activity() between
    the sampled frame and one MOUTH_LAG_SEC later (None where it couldn't be
    measured). That costs one more decoded frame per sample and no extra
    detection; it is what plan_reframe's speaker option reads.

    `bodies`, a list if given, collects (frame index, [(cx, cy, w, h)]) of
    head-and-shoulders for every sampled frame in which no face was found
    -- from the frame already decoded, so it costs no extra reading.
    plan_reframe falls back on them for a shot with no faces in it.

    `focus`, likewise, collects (frame index, focus_point()) for those
    frames, for a window `focus_w` display pixels wide (the crop's): the
    next thing plan_reframe falls back on."""
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
                if bodies is not None and not dets and hasattr(detector, 'detect_bodies'):
                    bodies.append((i, [((x + w / 2.0) * sar, y + h / 2.0, w * sar, h)
                                       for (x, y, w, h) in detector.detect_bodies(frame)]))
                if focus is not None and not dets and focus_w:
                    focus.append((i, focus_point(frame, focus_w, sar)))
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
                    'keys': None, 'speaker': True, 'person': who})
    return out


# ---- Split screen (optional) ----
# Two people too far apart for one vertical frame, each given half of it:
# the one on the left of the picture on top, the one on the right below.
SPLIT_PANE_ASPECT = (OUT_W, OUT_H // 2)     # each pane is 1080x960, i.e. 9:8
SPLIT_FACE_AT = 0.42                         # how far down its pane a face's centre sits
SPLIT_MIN_ZOOM_OUT = 0.45                    # a pane may shrink to this share of the largest window, no further


def split_pane_size(disp_w, disp_h):
    """The largest source window with a pane's shape: (w, h). For a
    landscape source that is the full height."""
    pw, ph = SPLIT_PANE_ASPECT
    h = even(disp_h)
    w = even(h * pw / float(ph))
    if w > disp_w:
        w = even(disp_w)
        h = even(w * ph / float(pw))
    return min(w, even(disp_w)), min(h, even(disp_h))


def _split_people(hits):
    """The two people of a wide two-shot as (left, right), each the median
    (cx, cy, w, h) of where that face was seen -- or None when the shot is
    not two people (one, or three and more: a third person has no pane)."""
    if len(hits) < 3:
        return None
    need = max(2, int(math.ceil(0.3 * len(hits))))
    tracks = [t for t in _face_tracks(hits) if len(t['obs']) >= need]
    if len(tracks) != 2:
        return None
    people = [tuple(float(np.median([f[k] for _, f in t['obs']])) for k in range(4)) for t in tracks]
    return tuple(sorted(people, key=lambda f: f[0]))


def _pane_origin(face, w, h, disp_w, disp_h):
    x = float(np.clip(face[0] - w / 2.0, 0.0, max(0.0, disp_w - w)))
    y = float(np.clip(face[1] - SPLIT_FACE_AT * h, 0.0, max(0.0, disp_h - h)))
    return x, y


def split_window(people, disp_w, disp_h):
    """The widest pane window (w, h) that shows each person WITHOUT the
    other one's face in it, or None if no acceptable window does.

    Starts from the largest window and tightens. Tightening is what keeps a
    slice of the other person's cheek out of the edge of the pane when the
    two sit close, or when one is near the edge of the picture and their
    window cannot be centred on them; it is also a zoom, so it stops at
    SPLIT_MIN_ZOOM_OUT rather than blow a face up into mush."""
    full_w, full_h = split_pane_size(disp_w, disp_h)
    left, right = people
    w = float(full_w)
    while w >= SPLIT_MIN_ZOOM_OUT * full_w:
        h = w * full_h / float(full_w)
        lx, _ = _pane_origin(left, w, h, disp_w, disp_h)
        rx, _ = _pane_origin(right, w, h, disp_w, disp_h)
        gap_l = (right[0] - right[2] / 2.0) - (lx + w)     # right face's near edge, past the left pane
        gap_r = rx - (left[0] + left[2] / 2.0)             # and the mirror of it
        own = (left[0] - left[2] / 2.0 >= lx and left[0] + left[2] / 2.0 <= lx + w
               and right[0] - right[2] / 2.0 >= rx and right[0] + right[2] / 2.0 <= rx + w)
        margin = 0.1 * max(left[2], right[2])
        if own and gap_l >= margin and gap_r >= margin:
            return even(w), even(h)
        w *= 0.96
    return None


def _split_segment(hits, a, b, disp_w, disp_h):
    people = _split_people(hits)
    if not people:
        return None
    size = split_window(people, disp_w, disp_h)
    if not size:
        return None
    return {'a': a, 'b': b, 'layout': 'split', 'x': None, 'keys': None, 'people': people, 'size': size}


def _finish_split(segs, disp_w, disp_h):
    """Gives every split segment of a clip the SAME window size (the
    tightest any of them needs) and works out where each pane's window sits.

    One size for the whole clip because ffmpeg's crop can move its window
    frame by frame but cannot resize it; a tighter window than a shot
    strictly needs only frames its two people a little closer."""
    splits = [s for s in segs if s['layout'] == 'split']
    if not splits:
        return
    w = min(s['size'][0] for s in splits)
    full_w, full_h = split_pane_size(disp_w, disp_h)
    h = min(even(w * full_h / float(full_w)), even(disp_h))
    for s in splits:
        s['size'] = (w, h)
        s['panes'] = [tuple(int(round(v)) for v in _pane_origin(f, w, h, disp_w, disp_h)) for f in s['people']]


EXPR_SUM_GROUP = 8


def expr_sum(terms):
    """`terms` added together as one ffmpeg expression, however many there
    are ('0' for none).

    Not simply '+'.join(terms). ffmpeg reads a+b+c+d as ((a+b)+c)+d, one
    level deeper per term, and since FFmpeg 8 refuses an expression more
    than 100 levels deep -- without saying so: the filter just reports
    "Failed to configure input pad", and the render is lost. A short with a
    hundred pieces to its crop path (a minute of following someone around)
    is ordinary. So the terms are added in groups of eight, the groups in
    groups of eight, and so on: thousands of terms stay a few dozen levels
    deep. Eight or fewer come out exactly as a plain sum."""
    terms = list(terms)
    if not terms:
        return '0'
    while len(terms) > EXPR_SUM_GROUP:
        terms = ['(' + '+'.join(terms[i:i + EXPR_SUM_GROUP]) + ')' for i in range(0, len(terms), EXPR_SUM_GROUP)]
    return '+'.join(terms)


def split_exprs(segs):
    """ffmpeg expressions for the two panes' crop origins, as functions of
    the frame number: ((top x, top y), (bottom x, bottom y))."""
    def expr(pane, axis):
        return expr_sum(f"between(n,{s['a']},{s['b']})*{s['panes'][pane][axis]}"
                        for s in segs if s['layout'] == 'split')
    return (expr(0, 0), expr(0, 1)), (expr(1, 0), expr(1, 1))


# The quieter of two people must hold the floor for this share of a shot
# before the shot counts as a conversation worth splitting the screen for.
SPLIT_MIN_SHARE = 0.2


def _both_talk(turns, a, b):
    """Whether a shot's speaker turns amount to two people talking, not one
    person talking and the other getting a word (or a laugh) in. Measured on
    a real clip: one speaker for 19 of 20 seconds, the other credited with
    the last second, and the whole shot was being split on the strength of
    that second."""
    held = {}
    for t in turns:
        held[t['person']] = held.get(t['person'], 0) + (t['b'] - t['a'] + 1)
    shares = sorted(held.values(), reverse=True)
    return len(shares) > 1 and shares[1] >= SPLIT_MIN_SHARE * (b - a + 1)


# ---- Who is in the shot ----
# One face clearly nearer the camera than the others is who the shot is of:
# this many times their area (about a quarter larger across).
DOMINANT_FACE = 1.6


def _people(hits):
    """The people a shot is of, largest first, each {'frames', 'xs', 'cx',
    'w', 'area'} -- or None when the detections do not say.

    Reported: crops that were not centred on anyone. The window used to be
    placed from each sampled frame on its own -- on the one face found in
    it, or midway between two -- and then averaged. But a face is not found
    in every frame: it turns, hair falls across it, a hand passes. A second
    person found in a third of the frames pulled the window part of the way
    toward them and no further, centred on nobody; found on and off, they
    were taken for movement, and the window drifted back and forth.

    So the shot is read as a whole first. A person is a face seen in the
    same place in at least 30% of the frames that show any face. Two are
    two people only if they are seen TOGETHER at least twice: seen one
    after the other, they may be one person who moved fast, or a pan from
    one to the next, and the answer is None -- the frame-by-frame reading
    that follows such a shot well is used instead."""
    if len(hits) < 3:
        return None
    need = max(2, int(math.ceil(0.3 * len(hits))))
    tracks = [t for t in _face_tracks(hits) if len(t['obs']) >= need]
    if not tracks:
        return None
    seen = [set(i for i, _ in t['obs']) for t in tracks]
    for k in range(len(tracks)):
        for m in range(k + 1, len(tracks)):
            if len(seen[k] & seen[m]) < 2:
                return None
    people = []
    for t in tracks:
        faces = [f for _, f in t['obs']]
        people.append({'frames': [i for i, _ in t['obs']], 'xs': [f[0] for f in faces],
                       'cx': float(np.median([f[0] for f in faces])),
                       'w': float(np.median([f[2] for f in faces])),
                       'area': float(np.median([f[2] * f[3] for f in faces]))})
    people.sort(key=lambda p: p['area'], reverse=True)
    return [p for p in people if p['area'] >= 0.4 * people[0]['area']]


def _group_targets(people, frames):
    """[(frame, x of the middle of the group)] at each of `frames`. A person
    not found in a frame is taken to be where they were last seen, not to
    have left: that is what keeps the window from lurching."""
    out = []
    for i in frames:
        at = [float(np.interp(i, p['frames'], p['xs'])) for p in people]
        left = min(x - p['w'] / 2.0 for x, p in zip(at, people))
        right = max(x + p['w'] / 2.0 for x, p in zip(at, people))
        out.append((i, (left + right) / 2.0))
    return out


def _wide_shot(hits, a, b, fps, mode, speaker, speech, disp_w, disp_h, crop_w, max_x, static_px, lead=False):
    """Segments for a shot whose faces don't fit one vertical frame, or None
    for the caller to crop it: 'always crop' committing to a side, or any
    mode when one of them is clearly who the shot is of (`lead`).

    In order: split the screen if asked and both people are (or may be)
    talking; follow the speaker if asked and it can be called; the lead, if
    there is one; otherwise the whole frame."""
    turns = _speaker_segments(hits, a, b, fps, speech, crop_w, max_x, static_px) if speaker else None
    if mode == 'split' and (turns is None or _both_talk(turns, a, b)):
        seg = _split_segment(hits, a, b, disp_w, disp_h)
        if seg:
            return [seg]
    if turns:
        return turns
    if mode in ('auto', 'split') and not lead:
        return [{'a': a, 'b': b, 'layout': 'fit', 'x': None, 'keys': None}]
    return None


def _body_targets(bodies, a, b, disp_h):
    """[(frame, x)] to follow the person in a shot with no faces in it, from
    head-and-shoulders detections -- or None if they do not show one.

    The upper-body cascade is noisy, so it has to agree with itself: the
    same figure, large enough to be who the shot is of, in at least 40% of
    the frames sampled, and if several, the largest of those."""
    ss = [(i, bs) for i, bs in bodies or [] if a <= i <= b]
    if len(ss) < 3:
        return None
    hits = [(i, {'sig': [f for f in bs if f[3] >= BODY_MIN_FRAC * disp_h]}) for i, bs in ss]
    hits = [(i, t) for i, t in hits if t['sig']]
    if len(hits) < max(2, int(math.ceil(0.4 * len(ss)))):
        return None
    need = max(2, int(math.ceil(0.4 * len(ss))))
    tracks = [t for t in _face_tracks(hits) if len(t['obs']) >= need]
    if not tracks:
        return None
    best = max(tracks, key=lambda t: float(np.median([f[2] * f[3] for _, f in t['obs']])))
    person = {'frames': [i for i, _ in best['obs']], 'xs': [f[0] for _, f in best['obs']],
              'w': float(np.median([f[2] for _, f in best['obs']]))}
    return _group_targets([person], [i for i, _ in hits])


def plan_reframe(samples, shot_starts, n_frames, disp_w, disp_h, crop_w, mode='auto', fps=25.0,
                 min_face_frac=0.05, speaker=False, speech=None, bodies=None, focus=None):
    """Decides, shot by shot, how a landscape clip becomes a portrait one.

    Returns segments [{'a', 'b', 'layout', 'x', 'keys'}] covering frames
    0..n_frames-1 (a and b inclusive). layout is 'crop' (a 9:16 window cut
    out of the picture, at a fixed x or panning through `keys`) or 'fit'
    (the whole picture, small, over a blurred copy of itself).

    The decision is per SHOT because that is the unit a viewer reads: a
    crop that drifts or jumps inside a shot looks like a mistake, one that
    changes on a cut is invisible.

      * one face, or faces that all fit inside the 9:16 window -> crop
        centred on them (one face is cropped to however large it is);
      * faces too far apart to fit, one of them clearly nearer the camera
        (DOMINANT_FACE) -> crop centred on that one: the shot is of them;
      * two or more similar-sized faces too far apart to fit -> 'auto'
        shows the whole frame (cutting to one of them would silently drop
        whoever is speaking half the time), 'crop' commits to the side
        with more face on it for the whole shot;
      * no faces -> the person seen from behind or side-on, if a body
        detector (see sample_faces' `bodies`) agrees with itself about one
        across the shot; failing that, where the picture is in focus, if
        that is clear (see focus_point); otherwise centre crop.

    Who is in a shot is decided from the shot as a whole, not frame by
    frame (see _people), so a face the detector loses now and then neither
    drags the window off the person it is on nor sets it wandering.

    Within a shot the window is locked off unless the subject really moves
    (more than ~12% of the window's width); then it follows on a heavily
    smoothed path, never frame-by-frame detections, which jitter.

    speaker=True changes the second case only, and only when the evidence
    is there (samples carrying mouth activity from sample_faces(mouth=True),
    and `speech`, the [(start, end)] of the spoken words in clip seconds):
    the shot is then cut between tight framings of whoever is speaking,
    see speaker_turns(). A shot it can't call falls through to what `mode`
    would have done anyway.

    mode='split' is 'auto' with a different answer to that same second case:
    when the shot is exactly two people, each gets half the frame, stacked
    (layout 'split'; _finish_split adds its 'size' and 'panes'). With
    speaker=True as well it is split only while BOTH of them talk: a shot
    where one person does all the talking is framed on that person, since
    half the screen is a lot to give to someone listening."""
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
        faceless = not ss or len(hits) < max(1, math.ceil(0.25 * len(ss)))
        targets = (_body_targets(bodies, a, b, disp_h) or _focus_targets(focus, a, b, max_x, crop_w)
                   if faceless else None)
        if faceless and not targets:
            # No face, and no figure the body detector agrees on: the middle.
            segs.append({'a': a, 'b': b, 'layout': 'crop', 'x': center, 'keys': None})
            continue
        if faceless:
            ss = [(i, []) for i, _ in targets]        # the frames the path is drawn through

        # "Too wide for one frame" is a question about TWO people. One face
        # always fits, in the sense that matters: there is nobody to lose by
        # cropping to it. A close-up boxed wider than the window (the Haar
        # cascades box a face generously) used to fail this test and be
        # handled like a wide two-shot -- a single person shown small over a
        # blurred background, the opposite of what a close-up should get.
        # Who the shot is of, read from the shot as a whole (see _people).
        people = None if faceless else _people(hits)
        group = lead = None
        if people:
            left = min(p['cx'] - p['w'] / 2.0 for p in people)
            right = max(p['cx'] + p['w'] / 2.0 for p in people)
            if len(people) == 1 or right - left <= 0.9 * crop_w:
                group = people                  # everyone fits: frame them all, steadily
            elif people[0]['area'] >= DOMINANT_FACE * people[1]['area']:
                lead = people[0]                # they do not, but one of them is the subject
        fits = [len(t['sig']) < 2 or t['span_w'] <= 0.9 * crop_w for _, t in hits]
        if faceless:
            pass                                    # following a figure: targets already set
        elif group:
            targets = _group_targets(group, [i for i, _ in hits])
        elif people is None and sum(fits) >= 0.6 * len(hits):
            targets = [(i, t['span_cx'] if ok else t['big_cx']) for (i, t), ok in zip(hits, fits)]
        elif (placed := _wide_shot(hits, a, b, fps, mode, speaker, speech, disp_w, disp_h,
                                   crop_w, max_x, static_px, lead=bool(lead))) is not None:
            segs.extend(placed)
            continue
        elif lead:
            targets = _group_targets([lead], [i for i, _ in hits])
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

    _finish_split(segs, disp_w, disp_h)
    merged = []
    for s in segs:
        p = merged[-1] if merged else None
        if p and p['layout'] == s['layout'] and not p['keys'] and not s['keys'] and (
                s['layout'] == 'fit' or (s['layout'] == 'split' and p['panes'] == s['panes'])
                or (s['layout'] == 'crop' and abs((p['x'] or 0) - (s['x'] or 0)) < 1.0)):
            p['b'] = s['b']
        else:
            merged.append(dict(s))
    return merged


def shot_bounds(shot_starts, n_frames):
    """[(a, b)] of each shot of a clip, inclusive, from where its cuts are."""
    bounds = sorted(set([0] + [int(c) for c in shot_starts or [] if 0 < int(c) < int(n_frames)]))
    return [(a, (bounds[k + 1] - 1) if k + 1 < len(bounds) else int(n_frames) - 1) for k, a in enumerate(bounds)]


def apply_framing(segs, shot_starts, n_frames, overrides, max_x):
    """The planned framing with the editor's corrections laid over it.

    `overrides` are [(frame, layout, x)]: for the shot containing `frame`
    (counted from the clip's first frame), 'crop' at window position `x`,
    or 'fit' (the whole picture). A shot named twice takes the first; a
    shot not named keeps its plan. A frame rather than a shot number names
    the shot, so a correction stays on the shot it was made for when the
    in point is moved and the shots are counted differently.

    A corrected shot is one segment, still: whatever the plan had done
    within it (followed someone, cut between speakers, split the screen)
    gives way to what the editor chose."""
    if not overrides:
        return segs
    chosen = {}
    for a, b in shot_bounds(shot_starts, n_frames):
        for f, layout, x in overrides:
            if a <= f <= b and layout in ('crop', 'fit'):
                chosen[(a, b)] = (layout, x)
                break
    if not chosen:
        return segs
    # The planner joins neighbouring shots framed alike into one segment.
    # Cut those back into shots first, so one of them can be changed alone.
    # (Only ever ones held still: a moving window is never joined.)
    pieces = []
    for s in segs:
        if s.get('keys'):
            pieces.append(s)
            continue
        for a, b in shot_bounds(shot_starts, n_frames):
            if a <= s['b'] and b >= s['a']:
                pieces.append(dict(s, a=max(a, s['a']), b=min(b, s['b'])))
    segs = pieces
    out = []
    for s in segs:
        own = next(((a, b) for (a, b) in chosen if a <= s['a'] <= b), None)
        if own is None:
            out.append(s)
            continue
        if out and out[-1].get('_editor') == own:
            continue                                # the rest of a shot already replaced
        layout, x = chosen[own]
        out.append({'a': own[0], 'b': own[1], 'layout': layout, 'keys': None, '_editor': own,
                    'x': float(min(max(0.0, float(x or 0.0)), max_x)) if layout == 'crop' else None})
    for s in out:
        s.pop('_editor', None)
    return out


def thin_keys(keys, tol=1.0):
    """The same path through fewer keys: every key that straight-line
    interpolation between its kept neighbours already reproduces to within
    `tol` pixels is dropped (Ramer-Douglas-Peucker on x against frame).

    The planner keys a moving window every 0.4 s whether or not it is
    turning. A smoothed path is mostly straight runs, so most of those keys
    say nothing, and each one costs a term in the crop expression."""
    if len(keys) <= 2 or tol <= 0:
        return list(keys)
    keep = [False] * len(keys)
    keep[0] = keep[-1] = True
    todo = [(0, len(keys) - 1)]
    while todo:
        i, j = todo.pop()
        (f0, x0), (f1, x1) = keys[i], keys[j]
        worst, at = 0.0, None
        for k in range(i + 1, j):
            f, x = keys[k]
            err = abs(x - (x0 + (x1 - x0) * (f - f0) / float(f1 - f0 or 1)))
            if err > worst:
                worst, at = err, k
        if at is not None and worst > tol:
            keep[at] = True
            todo += [(i, at), (at, j)]
    return [k for k, kept in zip(keys, keep) if kept]


def crop_x_expr(segs, tol=1.0):
    """ffmpeg expression for the crop window's x, as a function of the frame
    number `n`.

    A sum of between(n,a,b)*value terms, one active per frame, rather than
    nested if()s, and keyed on the integer frame number instead of the
    timestamp, which makes every boundary frame-exact with no float
    comparison to get wrong at 29.97. See expr_sum() for how the sum is
    written, and thin_keys() for `tol`: how far, in source pixels, the
    window may stray from the planned path for the sake of fewer terms."""
    terms = []
    for s in segs:
        if s['layout'] != 'crop':
            continue
        if s.get('keys'):
            ks = thin_keys(s['keys'], tol)
            for (f0, x0), (f1, x1) in zip(ks, ks[1:]):
                if f1 <= f0:
                    continue
                hi = s['b'] if f1 >= s['b'] else f1 - 1
                terms.append(f'between(n,{f0},{hi})*({x0:.1f}+({x1 - x0:.1f})*(n-{f0})/{f1 - f0})')
        else:
            terms.append(f"between(n,{s['a']},{s['b']})*{int(round(s['x'] or 0))}")
    return expr_sum(terms)


# The whole ffmpeg command has to fit on a Windows command line (32,767
# characters). The crop path is the one part of it that grows with the clip.
MAX_CROP_EXPR = 16000


def fitted_crop_x_expr(segs):
    """crop_x_expr() at the finest tolerance that keeps it a sane length:
    1 px for anything ordinary, coarser only for a clip of several minutes
    spent following movement throughout."""
    for tol in (1.0, 2.0, 4.0, 8.0, 16.0, 32.0):
        expr = crop_x_expr(segs, tol)
        if len(expr) <= MAX_CROP_EXPR:
            break
    return expr


# The optional "cliffhanger" ending. On the last beat of the moment the
# action stops dead: the pose and the expression stay where they are, the
# picture takes on a graded, vignetted look and creeps in by a few percent,
# it holds for about two seconds, and it cuts to black.
#
# A frozen frame is dead still, and reads as a fault rather than a held
# breath. So the hold is not one frame: it is the last few frames of the
# moment -- a sixth of a second of real time -- played forward and back
# again, stretched across the whole hold with each frame blended into the
# next. Nothing is invented: what moves is what was moving in the footage
# (a breath, hair, cloth), slowed about twelve times and returned to where
# it started, so the pose never goes anywhere.
#   loop    seconds of real footage the hold is made from
#   hold    seconds the hold lasts
#   zoom    how far the picture has crept in by the end of the hold
#   black   seconds of black after the cut, so the cut is seen and the short
#           does not simply end on the held picture
#   sound   seconds over which the sound is taken out when the action stops
#   still   how different (mean grey level, on a small copy of the picture) a
#           frame may be from the last one and still be part of the loop
CLIFFHANGER = {'loop': 0.16, 'hold': 2.0, 'zoom': 0.03, 'black': 0.4, 'sound': 0.2, 'still': 4.0}
# The look of the hold: contrast up, colour pulled back, shadows toward teal
# and highlights toward amber, the corners a little darker.
# How long the hold may be asked to last, in seconds.
CLIFF_HOLD_RANGE = (0.5, 8.0)


# The fade-to-black ending: how long the picture and sound take to go, in seconds.
FADE_RANGE = (0.3, 3.0)


def clamp_fade(seconds):
    """A fade length asked for, kept within FADE_RANGE; None (or nonsense) means the default."""
    try:
        v = float(seconds)
    except (TypeError, ValueError):
        return None
    if v != v:
        return None
    return max(FADE_RANGE[0], min(FADE_RANGE[1], v))


def clamp_hold(seconds):
    """A hold length asked for, kept within CLIFF_HOLD_RANGE; None (or nonsense) means the default."""
    try:
        v = float(seconds)
    except (TypeError, ValueError):
        return None
    if v != v:
        return None
    return max(CLIFF_HOLD_RANGE[0], min(CLIFF_HOLD_RANGE[1], v))


CLIFFHANGER_GRADE = ('eq=contrast=1.12:saturation=0.82:brightness=0.02,'
                     'colorbalance=rs=-0.07:bs=0.09:rm=0.02:bm=-0.02:rh=0.08:bh=-0.09,'
                     'vignette=angle=PI/10')


def still_frames(src, last_frame, fps, room=None):
    """How many of the frames ending at `last_frame` are near enough the
    same picture to make the cliffhanger hold from: at least 1.

    The hold loops its frames back and forth, slowed about twelve times.
    That is a held breath when little is moving. When a lot is -- someone
    turning, a hand coming up, a camera that will not keep still -- the same
    loop is the pose swaying slowly out and back, which is the opposite of
    locked. So each earlier frame is compared with the last, on a small
    grey copy where a breath or a strand of hair barely registers and a
    change of pose does, and the loop reaches back only as far as they
    still agree. Where even the frame before is too different, it is one
    frame: a true freeze, alive only in the creep in.

    The comparison is by region, and it is the regions that changed most
    that count. Averaged over the whole picture, a hand coming up in one
    corner of a still frame is a small number -- smaller than the grain of
    a noisy but motionless one -- and it is exactly what must not be looped
    (measured: a clip with one moving thing in it averaged 1.6 grey levels
    a frame, a motionless close-up 0.4, while their most-changed regions
    differed by 25 and 0.6)."""
    want = max(1, int(round(CLIFFHANGER['loop'] * float(fps))))
    want = max(1, min(want, int(room) if room else want, int(last_frame) + 1))
    if want < 2:
        return 1
    small = _small_frames(src, int(last_frame) - want + 1, want)
    if len(small) < 2:
        return 1
    return _still_run(small[-1], reversed(small[:-1]))       # walking back from the last frame


def still_frames_after(src, first_frame, fps, room):
    """The same question asked forwards: how many frames starting AT
    `first_frame` are near enough the same picture as it. 0 when there are
    none to be had (`room` is how many exist before the next cut or the end
    of the file), 1 when the picture is moving on from there.

    These are the frames just after a short's out point. Making the hold
    from them, where they will do, means the short gives up nothing of its
    own last moments to its ending."""
    want = min(max(1, int(round(CLIFFHANGER['loop'] * float(fps)))), max(0, int(room or 0)))
    if want < 1:
        return 0
    small = _small_frames(src, int(first_frame), want)
    if not small:
        return 0
    return _still_run(small[0], small[1:])


def _small_frames(src, first, count):
    """`count` frames from `first`, each as a 160x90 grey picture: small
    enough that grain and a breath all but vanish, large enough that a
    change of pose does not."""
    out = []
    cap = cv2.VideoCapture(src)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(first)))
        for _ in range(int(count)):
            ok, frame = cap.read()
            if not ok:
                break
            out.append(cv2.cvtColor(cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA),
                                    cv2.COLOR_BGR2GRAY).astype(np.float32))
    finally:
        cap.release()
    return out


def _still_run(anchor, others):
    """1 for `anchor` itself, plus how many of `others` -- taken in order,
    stopping at the first that is not -- are still the same picture."""
    keep = 1
    for frame in others:
        # 144 regions of 10 x 10; the three that changed most.
        regions = np.abs(frame - anchor).reshape(9, 10, 16, 10).mean(axis=(1, 3))
        if float(np.sort(regions, axis=None)[-3:].mean()) > CLIFFHANGER['still']:
            break
        keep += 1
    return keep


# ---- Where the sound really ends ----
# The out point of a moment is placed from the transcript: after its last
# word. But a transcript's idea of where a word ends is early as a rule --
# the tail of a vowel, a final consonant, the breath after it are all past
# the time it gives -- and where the timings are per line rather than per
# word it is only roughly right. Cut there and the last word is clipped. So
# for the cliffhanger ending, which goes from sound to dead silence with
# nothing to hide a clipped word behind, the sound itself is measured.
ENVELOPE_HOP = 0.01         # seconds between level readings
WORD_END = {'reach': 0.8,   # how far past the out point a word may run on and still be let finish
            'pause': 0.08,  # seconds of quiet that count as the word being over
            'near': 0.6,    # the next word starting within this of the out point: speech is running on
            'back': 0.12,   # running on: how far before the out point the gap between words may be
            'ahead': 0.3,   # ...and how far after it
            'contrast': 9.0,    # dB between loud and quiet there must be for any of this to mean anything
            'dip': 6.0}     # dB below the loud a gap between words must be to be taken for one


def audio_envelope(pcm, rate, hop=ENVELOPE_HOP):
    """Level in dB, one reading every `hop` seconds, of mono samples in
    -1..1: RMS over a window two hops long, lightly smoothed, so a single
    glottal pulse or the zero crossing of a low note is not a pause."""
    x = np.asarray(pcm, dtype=np.float64)
    step = max(1, int(round(rate * hop)))
    n = len(x) // step
    if n < 3:
        return np.zeros(0)
    power = (x[:n * step] ** 2).reshape(n, step).mean(axis=1)
    power = np.convolve(power, np.ones(2) / 2.0, mode='same')
    db = 10.0 * np.log10(np.maximum(power, 1e-10))
    return np.convolve(np.pad(db, 1, mode='edge'), np.ones(3) / 3.0, mode='valid')


def word_end(db, at, next_speech=None, hop=ENVELOPE_HOP):
    """Where the sound under way at `at` seconds into the envelope `db`
    comes to an end: (seconds into the envelope, kind), or None when the
    level says nothing useful and the transcript's time has to stand.

    'quiet'  nothing is sounding at `at`: it is already in a pause.
    'pause'  a word is, and this is where it gives way to quiet. Up to
             WORD_END['reach'] later -- the sound is let run on that far.
    'dip'    speech is running straight on into the next line
             (`next_speech`, seconds into the envelope, is close): there is
             no pause to end in, so this is the gap between two words that
             is nearest `at` -- which may be a little before it as well as
             after. Nearest, not deepest: `at` was put where the transcript
             has one word ending and the next beginning, and the gap wanted
             is that one, not a deeper one a word away. (Gaps that reach
             quiet are preferred to ones that only dip.)

    Loud and quiet are judged against this stretch of sound itself, not a
    fixed level: dialogue over a music bed never goes silent, it only drops
    to the music. Where the two are not far enough apart to tell a word
    from a gap (a wall of music, a crowd, nothing at all) the answer is
    None."""
    db = np.asarray(db, dtype=np.float64)
    if len(db) < 20:
        return None
    floor, top = float(np.percentile(db, 10)), float(np.percentile(db, 95))
    if top - floor < WORD_END['contrast']:
        return None
    quiet = db < floor + max(4.0, 0.3 * (top - floor))
    i = int(round(at / hop))
    if not 2 <= i < len(db) - 2:
        return None
    if quiet[i - 2:i + 3].mean() >= 0.6:
        return at, 'quiet'
    running = next_speech is not None and next_speech - at < WORD_END['near']
    need = max(2, int(round(WORD_END['pause'] / hop)))
    reach = WORD_END['reach'] if not running else min(WORD_END['reach'], max(0.15, next_speech - at + 0.1))
    last = min(len(db) - need, i + int(round(reach / hop)))
    for j in range(i, last + 1):
        if quiet[j:j + need].all():
            return round(j * hop + 0.02, 3), 'pause'
    if not running:
        return None                 # whatever is sounding, it is not a word that is about to finish
    lo = max(1, i - int(round(WORD_END['back'] / hop)))
    hi = min(len(db) - 1, i + int(round(WORD_END['ahead'] / hop)) + 1)
    gaps = [j for j in range(lo, hi)
            if db[j] <= db[j - 1] and db[j] <= db[j + 1] and db[j] <= top - WORD_END['dip']]
    if not gaps:
        return None
    # A gap that drops right down to quiet, however briefly, is a gap
    # between words; one that only dips may be a syllable inside one. So the
    # quiet ones are chosen from first. Then the nearest; of two as near,
    # the later (it keeps the whole of the word).
    gaps = [j for j in gaps if quiet[j]] or gaps
    j = min(gaps, key=lambda j: (abs(j - i), -j))
    return round(j * hop, 3), 'dip'


def read_envelope(ffmpeg, src, info, rate=8000, hop=ENVELOPE_HOP, timeout=1800):
    """The level (dB, one reading every `hop` seconds) of a whole file's
    sound -- the same sound a short gets (see read_pcm) -- read as it is
    decoded, so a 45-minute episode is never held in memory as samples:
    about a quarter of a million readings. None if there is no sound."""
    take = info.get('audio_take')
    if take:
        pick = ['-filter_complex', take_to_stereo(take) + '[a]', '-map', '[a]']
    elif info.get('audio_index') is not None:
        pick = ['-map', f"0:a:{info['audio_index']}"]
    else:
        return None
    step = max(1, int(round(rate * hop)))
    cmd = [ffmpeg, '-hide_banner', '-nostats', '-loglevel', 'error', '-ss', f"{max(0.0, info.get('v_offset', 0.0)):.6f}",
           '-i', src, '-vn'] + pick + ['-ac', '1', '-ar', str(int(rate)), '-f', 's16le', '-']
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except OSError:
        return None
    power, carry = [], np.zeros(0, dtype=np.float64)
    started = time.time()
    try:
        while True:
            chunk = proc.stdout.read(step * 2 * 2000)            # 20 s at a time
            if not chunk:
                break
            if time.time() - started > timeout:
                proc.kill()
                return None
            x = np.concatenate([carry, np.frombuffer(chunk[:len(chunk) // 2 * 2], dtype='<i2') / 32768.0])
            n = len(x) // step
            if n:
                power.append((x[:n * step] ** 2).reshape(n, step).mean(axis=1))
            carry = x[n * step:]
    finally:
        try:
            proc.stdout.close()
        except OSError:
            pass
        proc.wait()
    if not power:
        return None
    p = np.concatenate(power)
    p = np.convolve(p, np.ones(2) / 2.0, mode='same')
    db = 10.0 * np.log10(np.maximum(p, 1e-10))
    return np.convolve(np.pad(db, 1, mode='edge'), np.ones(3) / 3.0, mode='valid').astype(np.float32)


# Lines timed only to the line, by a service that rounds -- to the whole
# second, in the case that was reported -- are moved onto the sound.
ALIGN = {'early': 0.6,      # a line may really start this much before the time it was given
         'late': 1.5,       # ...or this much after (more often: it took in the silence before it)
         'end_early': 1.0,  # and end this much before
         'end_late': 0.7,   # or after
         'window': 3.0,     # seconds either side judged together for what is loud and quiet here
         'sound': 0.06,     # a start is sound lasting at least this long after quiet
         'quiet': 0.08}     # an end is quiet lasting at least this long after sound


def coarse_times(segments, min_lines=6, share=0.7):
    """True when a transcript's line times are rounded -- to the whole second,
    as reported, whatever offset they carry (0:30.29, 0:32.29, 0:33.29...):
    most of them share one fraction of a second. Times given to the
    hundredth are left as they are: moving those onto the sound gains
    little, and over a busy soundtrack can lose."""
    times = [float(t) for sg in segments or [] for t in (sg['start'], sg['end'])]
    if len(segments or []) < min_lines:
        return False
    buckets = {}
    for t in times:
        b = int(round((t % 1.0) / 0.05)) % 20
        buckets[b] = buckets.get(b, 0) + 1
    return max(buckets.values()) >= share * len(times)


def align_segments(segments, db, hop=ENVELOPE_HOP):
    """Line timings moved onto the sound: (segments, how many were moved).

    For a transcript timed only to the line. Each line's start goes to
    where its sound begins after quiet, and its end to where it gives way
    to quiet, looked for on whichever side of the time it was given the
    sound says: a line given a start in silence begins later, one given a
    start in the middle of sound began earlier.
    Where the sound cannot tell speech from a gap -- a wall of music, a
    crowd -- the line keeps its times. A line never starts before the one
    ahead of it has ended, so two lines are never moved over each other."""
    if db is None or not len(db) or not segments:
        return segments, 0
    db = np.asarray(db, dtype=np.float64)
    n = len(db)
    sound_run = max(2, int(round(ALIGN['sound'] / hop)))
    quiet_run = max(2, int(round(ALIGN['quiet'] / hop)))
    out, moved = [], 0
    for k, sg in enumerate(segments):
        s, e = float(sg['start']), float(sg['end'])
        lo = max(0, int((s - ALIGN['window']) / hop))
        hi = min(n, int((e + ALIGN['window']) / hop) + 1)
        here = db[lo:hi]
        if len(here) < 30:
            out.append(dict(sg))
            continue
        floor, top = float(np.percentile(here, 10)), float(np.percentile(here, 95))
        if top - floor < WORD_END['contrast']:
            out.append(dict(sg))
            continue
        quiet = here < floor + max(4.0, 0.3 * (top - floor))
        starts = [j for j in range(1, len(here) - sound_run)
                  if quiet[j - 1] and not quiet[j:j + sound_run].any()]
        ends = [j for j in range(1, len(here) - quiet_run)
                if not quiet[j - 1] and quiet[j:j + quiet_run].all()]
        prev_end = out[-1]['end'] if out else 0.0
        nxt = float(segments[k + 1]['start']) if k + 1 < len(segments) else float('inf')

        def sounding(t):
            j = int(round(t / hop)) - lo
            return 0 <= j < len(here) and quiet[max(0, j - 3):j + 4].mean() < 0.5

        def times(cands):
            return [(lo + j) * hop for j in cands]
        # The start. Quiet at the time given: the line had not begun -- it
        # begins where the sound first starts after it (it took in the
        # silence ahead of it, often by more than a second). Sound at that
        # time: it had already begun, where that sound started.
        if sounding(s):
            before = [t for t in times(starts) if max(s - ALIGN['early'], prev_end - 0.02) <= t <= s]
            ns = before[-1] if before else None
        else:
            after = [t for t in times(starts) if s <= t <= min(e - 0.2, s + max(ALIGN['late'], 3.0))
                     and t >= prev_end - 0.02]
            ns = after[0] if after else None
        # The end, the other way round: sound at the time given -- it went on
        # to where the sound stops; quiet -- it had stopped already.
        if sounding(e):
            later = [t for t in times(ends) if e <= t <= min(e + ALIGN['end_late'], nxt + 0.5)]
            ne = later[0] if later else None
        else:
            earlier = [t for t in times(ends) if e - ALIGN['end_early'] <= t <= e]
            ne = earlier[-1] if earlier else None
        new_s = round(ns - 0.03, 3) if ns is not None else s
        new_e = round(ne + 0.03, 3) if ne is not None else e
        if new_e - new_s < 0.2:
            new_s, new_e = s, e
        new_s = max(new_s, prev_end)
        if new_e <= new_s:
            new_s, new_e = max(s, prev_end), max(e, prev_end + 0.2)
        if abs(new_s - s) > 0.02 or abs(new_e - e) > 0.02:
            moved += 1
        out.append(dict(sg, start=new_s, end=new_e))
    return out, moved


def read_pcm(ffmpeg, src, info, start, dur, rate=16000, timeout=60):
    """`dur` seconds of the source's sound from `start` (seconds on the
    clock build_render_cmd seeks by, so the two agree), as mono float
    samples at `rate` -- the same sound a short gets: the channels the
    dialogue was found on where that was worked out (info['audio_take']),
    the chosen audio stream otherwise. None when there is none to read."""
    take = info.get('audio_take')
    if take:
        pick = ['-filter_complex', take_to_stereo(take) + '[a]', '-map', '[a]']
    elif info.get('audio_index') is not None:
        pick = ['-map', f"0:a:{info['audio_index']}"]
    else:
        return None
    cmd = [ffmpeg, '-hide_banner', '-nostats', '-loglevel', 'error', '-ss', f'{max(0.0, start):.6f}', '-i', src,
           '-t', f'{dur:.3f}', '-vn'] + pick + ['-ac', '1', '-ar', str(int(rate)), '-f', 's16le', '-']
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=timeout)
    except (subprocess.TimeoutExpired, OSError):
        return None
    if r.returncode != 0 or len(r.stdout) < 2:
        return None
    return np.frombuffer(r.stdout[:len(r.stdout) // 2 * 2], dtype='<i2').astype(np.float32) / 32768.0


# How the sound goes once it is known where: let ring out after a word that
# finished, taken out fast where the next word is already on its way.
AUDIO_FADE = {'pause': 0.2, 'quiet': 0.12, 'dip': 0.05, None: 0.12}


def measure_audio_out(ffmpeg, src, info, start_f, n_frames, next_speech=None, before=2.5):
    """(where the sound of a short should stop, how long its fade is, how it
    was decided) for a short of `n_frames` from `start_f`: seconds from the
    start of the short.

    Listens to the source from `before` seconds ahead of the out point to
    past the furthest a word may run on, and asks word_end. `next_speech`
    is when the transcript has the next word starting, in seconds from the
    start of the short, if it has one. With nothing to go on -- no sound, a
    read that failed, a level that says nothing -- the answer is the out
    point itself."""
    fps = info['fps']
    out = n_frames / fps
    plain = (round(out, 3), AUDIO_FADE[None], None)
    lead = min(float(before), out)
    clock = max(0.0, info.get('v_offset', 0.0) + (start_f - 0.25) / fps)     # as build_render_cmd seeks
    rate = 16000
    pcm = read_pcm(ffmpeg, src, info, clock + out - lead, lead + WORD_END['reach'] + 0.5, rate)
    if pcm is None or len(pcm) < rate // 2:
        return plain
    found = word_end(audio_envelope(pcm, rate), lead,
                     None if next_speech is None else lead + (float(next_speech) - out))
    if not found:
        return plain
    at, kind = found
    return round(out + (at - lead), 3), AUDIO_FADE[kind], kind


def cliffhanger_plan(n_frames, fps, room=None, after=0, hold_s=None):
    """(loop, hold, black) in whole frames for a clip of `n_frames`.

    The hold is made from `loop` frames of footage. Where they come from is
    the caller's choice, by what it found:

    `after` > 0: that many frames just past the clip's out point are usable
    (same shot, still enough: see still_frames_after). The hold is made from
    them and the clip plays to its last frame first, giving up nothing.

    Otherwise they are the clip's own last frames, and `room` is how many of
    those may be used: no more than the clip's last shot has run for (frames
    from before a cut are another picture, and blending across one would
    flash it through the hold), and no more than are still enough (see
    still_frames). With a single frame of room the hold is a true freeze.

    `hold_s` is how long the hold lasts in seconds, where the editor chose
    one; otherwise CLIFFHANGER['hold']."""
    hold_s = clamp_hold(hold_s)
    if hold_s is None:
        hold_s = CLIFFHANGER['hold']
    loop = max(1, int(round(CLIFFHANGER['loop'] * float(fps))))
    if after and int(after) > 0:
        loop = max(1, min(loop, int(after)))
    else:
        loop = max(1, min(loop, int(n_frames) - 1, int(room) if room else loop))
    return loop, max(2, int(round(hold_s * float(fps)))), max(1, int(round(CLIFFHANGER['black'] * float(fps))))


def cliffhanger_stop(n_frames, fps, room=None, after=0):
    """The frame the picture stops on: the first frame of the hold, counted
    from the clip's first. The clip's own length when the hold is made from
    what follows it, `loop` frames short of that when it is made from the
    clip's own end."""
    if after and int(after) > 0:
        return int(n_frames)
    return int(n_frames) - cliffhanger_plan(n_frames, fps, room)[0]


def cliffhanger_extra(n_frames, fps, room=None, after=0, hold_s=None):
    """How many frames longer the short is for its ending."""
    _, hold, black = cliffhanger_plan(n_frames, fps, room, after, hold_s)
    return cliffhanger_stop(n_frames, fps, room, after) + hold + black - int(n_frames)


def run_on(segs, extra):
    """The reframing plan carried `extra` frames past its end: the last
    shot's framing held where it finished. For the frames just after the
    out point that the cliffhanger hold is made from, which the plan --
    made for the clip -- says nothing about. Without it they would be
    cropped from the left edge of the picture."""
    if not segs or extra <= 0:
        return segs
    last = dict(segs[-1])
    if last.get('keys'):
        keys = [tuple(k) for k in last['keys']]
        last['keys'] = keys + [(last['b'] + int(extra), keys[-1][1])]
    last['b'] = last['b'] + int(extra)
    return list(segs[:-1]) + [last]


def cliffhanger_graph(n_frames, fps, room=None, out_w=OUT_W, out_h=OUT_H, ass_name=None, after=0, hold_s=None):
    """The filters that put the ending on the finished picture of a clip of
    `n_frames`: fed the clip, they give the clip up to where the action
    stops, then the hold, then black.

    Counted in frames throughout. The clip is split in two at the frame the
    action stops on (cliffhanger_stop). The `loop` frames from there -- the
    clip's own last ones, or with `after` the ones that follow it in the
    source, which the graph is then fed as well -- become the hold: reversed and
    joined to themselves (forward, then back), re-timed so they span it,
    and brought back to the clip's frame rate by `framerate`, which blends
    neighbouring frames in proportion (not motion estimation, which bends
    faces). `tpad` and `trim` then make it exactly `hold` + `black` frames,
    and the last `black` of them are painted out at the very end: appending
    black frames after `zoompan` instead gives them timestamps far in the
    future, and they are then never written. The creep in is `zoompan` on a
    picture enlarged first: it positions its window in whole pixels, and at
    this speed that shows as a shimmer unless a pixel is made smaller than
    the eye can follow.

    Captions (`ass_name`) are burnt into the first part only. They belong to
    the action: one still up when it stops goes with it, on the same frame
    as the sound. Burnt in before the split, it would be held, pushed in on
    and graded with the picture; after the join, it would sit ungraded over
    the first frames of the hold and then drop off a moment into it."""
    n = int(n_frames)
    loop, hold, black = cliffhanger_plan(n, fps, room, after, hold_s)
    stop = cliffhanger_stop(n, fps, room, after)
    rate = frame_rate_arg(fps)
    spread = hold / float(2 * loop)                    # output frames per frame of the loop
    zoom = CLIFFHANGER['zoom']
    caption = f'ass={ass_name},' if ass_name else ''
    return (f"split=2[em][et];"
            f"[em]trim=end_frame={stop},setpts=PTS-STARTPTS,{caption}format=yuv420p,setsar=1[ea];"
            f"[et]trim=start_frame={stop}:end_frame={stop + loop},setpts=PTS-STARTPTS,split=2[ef][eq];"
            f"[eq]reverse[er];[ef][er]concat=n=2:v=1:a=0,"
            f"setpts=N*{spread:.6f}/({rate})/TB,"
            f"framerate=fps={rate}:interp_start=0:interp_end=255:scene=100,"
            f"tpad=stop_mode=clone:stop={hold + black},trim=end_frame={hold + black},setpts=PTS-STARTPTS,"
            f"scale={2 * out_w}:{2 * out_h}:flags=bicubic,"
            f"zoompan=z='1+{zoom}*on/{max(1, hold - 1)}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
            f":d=1:s={out_w}x{out_h}:fps={rate},"
            f"{CLIFFHANGER_GRADE},"
            f"drawbox=x=0:y=0:w=iw:h=ih:color=black:t=fill:enable='gte(n,{hold})',format=yuv420p,setsar=1[eb];"
            f"[ea][eb]concat=n=2:v=1:a=0,")


def build_filtergraph(info, segs, out_w=OUT_W, out_h=OUT_H, ass_name=None, ending=None, fade=None):
    """The -filter_complex string for one short. Video in on [0:v], out on
    [vout].

    Everything stays inside ffmpeg (no frames round-tripped through
    OpenCV/numpy): that keeps the source's colour matrix and range intact
    end to end, which a Python pixel pipeline would quietly re-interpret.

    When a clip mixes layouts, both versions are produced for the whole
    clip and the 'fit' one is overlaid only during its shots, switched by
    frame number. That costs some redundant filtering on the shots that
    don't use it, in exchange for a graph whose size doesn't grow with the
    number of shots and whose switches are frame-exact.

    `ending` is (the clip's length in frames, how many frames its last shot
    has run for by then, how many usable frames follow it) when it is to
    finish on the cliffhanger hold (see cliffhanger_plan), or None. The
    third may be left off: none follow."""
    if ending:
        ending = tuple(ending) + (0,) * (3 - len(ending)) + (None,) * max(0, 4 - max(3, len(ending)))
        if ending[2]:
            segs = run_on(segs, cliffhanger_plan(ending[0], info['fps'], ending[1], ending[2])[0])
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

    y = even((disp_h - crop_h) / 2.0) if disp_h > crop_h else 0
    crop_chain = (f"crop={crop_w}:{crop_h}:x='{fitted_crop_x_expr(segs)}':y={y},"
                  f"scale={out_w}:{out_h}:flags=lanczos")
    k = min(out_w / float(disp_w), out_h / float(disp_h))
    fg_w, fg_h = min(out_w, even(disp_w * k)), min(out_h, even(disp_h * k))
    bw, bh = even(out_w / 4.0), even(out_h / 4.0)
    fit_chain = (f"split=2[fa][fb];"
                 f"[fa]scale={bw}:{bh}:force_original_aspect_ratio=increase,crop={bw}:{bh},"
                 f"boxblur=12:2,scale={out_w}:{out_h}:flags=bilinear,lutyuv=y='16+(val-16)*0.72'[bg];"
                 f"[fb]scale={fg_w}:{fg_h}:flags=lanczos[fg];"
                 f"[bg][fg]overlay=(W-w)/2:(H-h)/2")
    chains = {'crop': crop_chain, 'fit': fit_chain}
    splits = [s for s in segs if s['layout'] == 'split']
    if splits:
        # Two windows of one size, each moved to its person shot by shot and
        # stacked, with a thin dark line where they meet so the join reads
        # as a deliberate split rather than a torn frame.
        sw, sh = splits[0]['size']
        (tx, ty), (bx, by) = split_exprs(segs)
        half = out_h // 2
        chains['split'] = (f"split=2[sa][sb];"
                           f"[sa]crop={sw}:{sh}:x='{tx}':y='{ty}',scale={out_w}:{half}:flags=lanczos[st];"
                           f"[sb]crop={sw}:{sh}:x='{bx}':y='{by}',scale={out_w}:{half}:flags=lanczos[sm];"
                           f"[st][sm]vstack,drawbox=x=0:y={half - 3}:w={out_w}:h=6:color=black@0.85:t=fill")
    # One version of the clip per layout in use, the first as the base and
    # each other laid over it only during its own shots, switched by frame
    # number. (crop, fit, split) is also the order the labels are built in.
    used = [name for name in ('crop', 'fit', 'split') if any(s['layout'] == name for s in segs)] or ['crop']
    if len(used) == 1:
        g = f"[0:v]{pre},{chains[used[0]]}[v1]"
    else:
        tag = {'crop': 'c', 'fit': 'f', 'split': 's'}
        g = f"[0:v]{pre},split={len(used)}" + ''.join(f'[{tag[u]}0]' for u in used) + ';'
        g += ';'.join(f'[{tag[u]}0]{chains[u]}[{tag[u]}v]' for u in used) + ';'
        base = f'[{tag[used[0]]}v]'
        for i, u in enumerate(used[1:]):
            enable = expr_sum(f"between(n,{s['a']},{s['b']})" for s in segs if s['layout'] == u)
            out = '[v1]' if i == len(used) - 2 else f'[m{i}]'
            g += f"{base}[{tag[u]}v]overlay=0:0:enable='{enable}'{out}" + ('' if out == '[v1]' else ';')
            base = out
    # setpts first: frame 0 at time 0 exactly, so captions (timed from the
    # clip start) and the encoder's frame slots both line up with `n`.
    # With the ending, the captions go on inside it: see cliffhanger_graph.
    if ending:
        tail = (cliffhanger_graph(ending[0], info['fps'], ending[1], out_w, out_h, ass_name, ending[2], ending[3])
                + 'setpts=PTS-STARTPTS,format=yuv420p,setsar=1')
    else:
        # `fade` is (the clip's length in frames, how many of its last frames
        # go to black): captions fade with the picture, being part of it.
        goes = (f'fade=t=out:s={max(0, int(fade[0]) - int(fade[1]))}:n={int(fade[1])}:color=black,' if fade else '')
        tail = 'setpts=PTS-STARTPTS,' + (f'ass={ass_name},' if ass_name else '') + goes + 'format=yuv420p,setsar=1'
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


MAX_CUES = 800              # per short: far more than five minutes of speech makes
MAX_CUE_CHARS = 200


def clean_cues(raw, dur):
    """Captions as an editor sent them -> cues fit to burn in, or ValueError
    saying what is wrong.

    [{'start', 'end', 'text'}], seconds from the start of the short. A line
    with no text is a line taken out. They are put in time order, kept
    inside the short, and where two overlap the earlier one gives way: one
    caption on screen at a time is the whole point of them."""
    if not isinstance(raw, list):
        raise ValueError('Captions must be a list of lines.')
    if len(raw) > MAX_CUES:
        raise ValueError(f'At most {MAX_CUES} caption lines per short.')
    dur = float(dur)
    cues = []
    for n, c in enumerate(raw, 1):
        if not isinstance(c, dict):
            raise ValueError(f'Caption {n} is not valid.')
        text = ' '.join(str(c.get('text') or '').split())
        if not text:
            continue
        if len(text) > MAX_CUE_CHARS:
            raise ValueError(f'Caption {n} is {len(text)} characters long; the limit is {MAX_CUE_CHARS}. '
                             'Split it into two lines.')
        try:
            start, end = float(c.get('start')), float(c.get('end'))
        except (TypeError, ValueError):
            raise ValueError(f'Caption {n} ("{text[:30]}"): start and end must be numbers (seconds).')
        if not (math.isfinite(start) and math.isfinite(end)):
            raise ValueError(f'Caption {n} ("{text[:30]}"): start and end must be numbers (seconds).')
        if end <= start:
            raise ValueError(f'Caption {n} ("{text[:30]}"): it must end after it starts.')
        start, end = max(0.0, start), min(dur, end)
        if end - start < 0.08:
            continue                    # outside the short, or all but
        cues.append({'start': round(start, 3), 'end': round(end, 3), 'text': text})
    cues.sort(key=lambda c: (c['start'], c['end']))
    out = []
    for c in cues:
        if out and c['start'] < out[-1]['end']:
            out[-1]['end'] = c['start']
            if out[-1]['end'] - out[-1]['start'] < 0.08:
                out.pop()
        out.append(c)
    return out


def fit_cues(edited, old_start, old_end, start, end, auto):
    """Captions edited for the range [old_start, old_end] of the source,
    carried over to [start, end] -- the editor nudged the in or out point
    after correcting them, and the corrections should survive that.

    Every edited line still inside the new range is kept, moved to where it
    now falls. What the new range adds at either end was never edited, so it
    gets the automatic captions: `auto(a, b)` returns those for a stretch of
    the source, timed from `a`."""
    shift = float(old_start) - float(start)
    dur = float(end) - float(start)
    lo, hi = max(0.0, shift), min(dur, float(old_end) - float(start))
    if hi <= lo:
        # Moved clear of everything that was edited: there is nothing to carry.
        return [dict(c) for c in auto(float(start), float(end))]
    kept = []
    for c in edited:
        s, e = max(lo, c['start'] + shift), min(hi, c['end'] + shift)
        if e - s >= 0.08:
            kept.append({'start': round(s, 3), 'end': round(e, 3), 'text': c['text']})
    head = [dict(c) for c in auto(float(start), float(start) + lo)] if lo > 0.3 else []
    tail = ([{'start': round(c['start'] + hi, 3), 'end': round(c['end'] + hi, 3), 'text': c['text']}
             for c in auto(float(start) + hi, float(end))] if dur - hi > 0.3 else [])
    return head + kept + tail


def place_cues(cues, segs, fps):
    """Marks the cues that play over a split-screen shot (cue['seam'] = True)
    so write_ass centres them on the join between the two panes.

    Their usual place, low in the frame, is the middle of the lower pane --
    across the chin of whoever is in it. The join is the one strip of a
    split screen with nobody's face on it. A cue belongs to the layout its
    midpoint falls in, so one that straddles a cut isn't moved for the sake
    of a few frames."""
    spans = [(s['a'], s['b']) for s in segs if s['layout'] == 'split']
    for c in cues:
        mid = (c['start'] + c['end']) / 2.0 * fps
        c['seam'] = any(a <= mid <= b for a, b in spans)
    return cues


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
        if c.get('seam'):       # see place_cues: centred on the join of a split screen
            text = f'{{\\an5\\pos({out_w // 2},{out_h // 2})}}' + text
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


def take_to_stereo(take):
    """ffmpeg filter chain (inputs named, no output label) that makes a
    stereo programme out of chosen channels of the source:
    [[audio stream, channel], ...] counted from 0, one channel or two.

    For a master that keeps its sound as separate tracks -- eight mono
    tracks, say, with the full mix on 3 and 4. The default there, the first
    track or all of them folded together, is the wrong sound: silence, or
    music and effects without the dialogue. The take is whatever the
    transcription found the dialogue on."""
    (s0, c0) = take[0]
    if len(take) == 1:
        return f'[0:a:{s0}]pan=stereo|c0=c{c0}|c1=c{c0}'
    (s1, c1) = take[1]
    if s0 == s1:
        return f'[0:a:{s0}]pan=stereo|c0=c{c0}|c1=c{c1}'
    return (f'[0:a:{s0}]pan=mono|c0=c{c0}[tl];[0:a:{s1}]pan=mono|c0=c{c1}[tr];'
            '[tl][tr]amerge=inputs=2,pan=stereo|c0=c0|c1=c1')


def frame_rate_arg(fps):
    """`fps` as ffmpeg wants a frame rate written: '30000/1001', '25', '24000/1001'."""
    from fractions import Fraction
    r = Fraction(float(fps)).limit_denominator(1001)
    if abs(float(r) - float(fps)) > 1e-3:           # not a rate with a small denominator: say it as measured
        r = Fraction(float(fps)).limit_denominator(100000)
    return f'{r.numerator}/{r.denominator}' if r.denominator != 1 else str(r.numerator)


def build_render_cmd(ffmpeg, src, out_path, start_f, n_frames, info, segs, ass_name=None,
                     crf=18, preset='medium', loudness=-14.0, true_peak=-1.5, ending=False, ending_room=None,
                     ending_after=0, audio_out=None, audio_fade=None, out_w=OUT_W, out_h=OUT_H,
                     ending_hold=None, ending_sfx=None, ending_fade=None):
    """`ending_fade` (seconds) is the fade-to-black ending instead: the last
    seconds of the clip itself go to black, picture and sound together, and
    the short is no longer for it.
    `ending` adds the cliffhanger ending (see cliffhanger_graph).
    `ending_hold` is how many seconds the hold lasts (default CLIFFHANGER['hold']).
    `ending_sfx` is {'path', 'gain'} -- a sound effect that comes in at the
    moment the action stops, `gain` dB up or down, and goes out with the
    black; it is mixed in after the dialogue has been levelled, then limited."
    `out_w` x `out_h` is the picture size: the full 1080x1920 unless a
    small preview of the framing is wanted.
    `ending_room` is how many frames the clip's last shot has run for by
    its last frame, when the caller knows where the cuts are, and
    `ending_after` how many usable frames follow the clip (see
    cliffhanger_plan).

    `audio_out` is where, in seconds from the start of the clip, the sound
    of that ending stops, and `audio_fade` how long it takes to go. The
    sound is not tied to the picture: it may run on under the hold until
    the word being spoken is finished (see word_end). Left as None it stops
    where the picture does, which is how shorts were made before the sound
    was measured, and how one of those is made again."""
    fps = info['fps']
    dur = n_frames / fps
    # A quarter of a frame early, so accurate seek lands on exactly start_f
    # whatever way the float rounds: the first frame ffmpeg keeps is the
    # first whose timestamp is >= -ss. Not half a frame: that leaves the
    # first frame's timestamp exactly between two output slots, and the
    # encoder then sometimes rounds it up and pads slot 0 with a duplicate
    # (measured -- which also pushes the last frame off the end).
    ss = max(0.0, info.get('v_offset', 0.0) + (start_f - 0.25) / fps)
    fade_s = None
    if not ending and ending_fade:
        fade_s = min(clamp_fade(ending_fade) or FADE_RANGE[0], max(0.1, dur - 0.1))
    graph = build_filtergraph(info, segs, out_w, out_h, ass_name=ass_name,
                              ending=(n_frames, ending_room, ending_after, ending_hold) if ending else None,
                              fade=(n_frames, max(1, int(round(fade_s * fps)))) if fade_s else None)
    extra = cliffhanger_extra(n_frames, fps, ending_room, ending_after, ending_hold) if ending else 0
    sfx_chain = None
    if ending:
        # The sound is taken out with a cubic fade (all but gone in half its
        # length, and no click), and there is nothing after it. The nothing
        # is padding added AFTER the loudness stage, so it is digital silence
        # and not a levelled-up noise floor. Wherever it was asked to stop,
        # it is silent for the last half second of the hold at least: the
        # cut to black is a cut in the picture, not the end of a sentence.
        _, hold, _ = cliffhanger_plan(n_frames, fps, ending_room, ending_after, ending_hold)
        stop = cliffhanger_stop(n_frames, fps, ending_room, ending_after)
        out = CLIFFHANGER['sound'] if audio_fade is None else max(0.02, float(audio_fade))
        stops = stop / fps if audio_out is None else max(0.1, float(audio_out))
        stops = min(stops, (stop + hold) / fps - 0.5 - out)
        polish = (f'afade=t=in:st=0:d=0.04,atrim=end={stops + out:.3f},'
                  f'afade=t=out:st={stops:.3f}:d={out:.3f}:curve=cub,'
                  f'loudnorm=I={loudness}:TP={true_peak}:LRA=11,apad=whole_dur={dur + extra / fps:.3f}')
        if ending_sfx and ending_sfx.get('path'):
            total = dur + extra / fps
            begins = stop / fps                         # the instant the action stops
            room_s = max(0.2, total - begins)
            ramp = min(0.25, room_s / 2)
            ms = int(round(begins * 1000))
            gain = max(-40.0, min(12.0, float(ending_sfx.get('gain') or 0.0)))
            sfx_chain = (f'[1:a]aresample=48000,aformat=channel_layouts=stereo,volume={gain:.1f}dB,'
                         f'atrim=end={room_s:.3f},afade=t=out:st={max(0.0, room_s - ramp):.3f}:d={ramp:.3f},'
                         f'adelay={ms}|{ms},apad=whole_dur={total:.3f}')
    else:
        fade_len = fade_s if fade_s else 0.12
        fade_out = max(0.0, dur - fade_len)
        if fade_s:
            # A long fade goes on AFTER the levelling: measured with it in, the quiet tail
            # would be counted and the whole short turned up to make up for it.
            polish = (f'afade=t=in:st=0:d=0.04,loudnorm=I={loudness}:TP={true_peak}:LRA=11,'
                      f'afade=t=out:st={fade_out:.3f}:d={fade_len:.3f}')
        else:
            polish = (f'afade=t=in:st=0:d=0.04,afade=t=out:st={fade_out:.3f}:d={fade_len:.3f},'
                      f'loudnorm=I={loudness}:TP={true_peak}:LRA=11')
    take = info.get('audio_take')
    mix = ('[am][sfx]amix=inputs=2:normalize=0:duration=first:dropout_transition=0,alimiter=limit=0.95[aout]')
    if take:
        # The channels the dialogue was found on (see take_to_stereo), not
        # whichever stream ffmpeg would pick.
        if sfx_chain:
            graph += f';{take_to_stereo(take)},{polish}[am];{sfx_chain}[sfx];{mix}'
        else:
            graph += f';{take_to_stereo(take)},{polish}[aout]'
    elif sfx_chain:
        if info.get('audio_index') is not None:
            graph += f";[0:a:{info['audio_index']}]{polish}[am];{sfx_chain}[sfx];{mix}"
        else:
            graph += f';{sfx_chain}[aout]'            # a silent source: the effect alone
    cmd = [ffmpeg, '-y', '-hide_banner', '-nostats', '-loglevel', 'error',
           '-ss', f'{ss:.6f}', '-i', src]
    if sfx_chain:
        cmd += ['-i', ending_sfx['path']]
    cmd += ['-t', f'{dur + extra / fps:.6f}', '-frames:v', str(int(n_frames) + extra),
            '-filter_complex', graph, '-map', '[vout]']
    if take or sfx_chain:
        cmd += ['-map', '[aout]', '-c:a', 'aac', '-b:a', '192k', '-ar', '48000', '-ac', '2']
    elif info.get('audio_index') is not None:
        cmd += ['-map', f"0:a:{info['audio_index']}", '-af', polish,
                '-c:a', 'aac', '-b:a', '192k', '-ar', '48000', '-ac', '2']
    else:
        cmd += ['-an']
    # The frame rate is stated, not left for ffmpeg to work out. The graph
    # resets the timestamps (setpts), which newer ffmpeg takes to mean the
    # stream has no fixed rate; with none stated it then assumes 25 and
    # drops frames to fit -- a 29.97 source came out as 25 fps with one frame
    # in six missing. Every frame already sits exactly on this rate's grid,
    # so stating it changes nothing where it was being worked out correctly.
    cmd += ['-r', frame_rate_arg(fps)]
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
                 crf=18, preset='medium', loudness=-14.0, true_peak=-1.5, timeout=900, ending=False,
                 ending_room=None, ending_after=0, audio_out=None, audio_fade=None, out_w=OUT_W, out_h=OUT_H,
                 ending_hold=None, ending_sfx=None, ending_fade=None):
    """Renders one short. Returns (ok, error_text).

    Runs ffmpeg with `work_dir` as its working directory and refers to the
    caption file by bare name. That sidesteps the filtergraph's path
    escaping entirely -- a Windows path's drive colon and backslashes each
    need a different number of escapes inside -filter_complex, and getting
    it wrong fails only on Windows, i.e. only in production."""
    # Absolute, because ffmpeg is about to run from a different directory.
    src, out_path = os.path.abspath(src), os.path.abspath(out_path)
    cmd = build_render_cmd(ffmpeg, src, out_path, start_f, n_frames, info, segs, ass_name=ass_name,
                           crf=crf, preset=preset, loudness=loudness, true_peak=true_peak, ending=ending,
                           ending_room=ending_room, ending_after=ending_after, audio_out=audio_out,
                           audio_fade=audio_fade, out_w=out_w, out_h=out_h, ending_hold=ending_hold,
                           ending_sfx=ending_sfx, ending_fade=ending_fade)
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

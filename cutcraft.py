"""
Pure helpers for clean cuts, card fitting and script-aware narration.

Nothing here touches ffmpeg, Flask or the network, so every rule can be tested
with plain numbers. pipeline.py does the measuring (Whisper words, silences,
card durations) and the encoding; this module decides where to cut.

Three jobs:

1. Cuts that never land inside a word.
   settle_in / settle_out move a clip's in and out so the speech on either side
   keeps room for the transition, prefer the start of a sentence for the in
   point, and classify_cut reports how clean each edge ended up.

2. Cards that fit the plug length.
   card_floor says how short a title or end card can safely get (its own voice
   and its animation must finish), plan_card_fit shrinks cards before scenes
   are chosen when they would crowd the scenes out, and card_adjust_limit scales
   the last-resort adjustment with the plug length.

3. Narration cut at the script's own points.
   parse_vo_script reads timed lines ("00:03 VO: ..."), align_script finds each
   line in the Whisper words of the recorded VO, and plan_vo cuts the VO at
   word edges, places every line at its in point, and when the result does not
   fit tightens pauses, speeds up slightly, drops optional lines and as a last
   step trims at the end of a word. Never mid-word.
"""
import bisect
import difflib
import re
import unicodedata

# ---------------------------------------------------------------- cuts

LEAD_MAX = 0.25         # room kept before a word that opens a clip
TRAIL_MAX = 0.25        # room kept after a word that closes a clip
EDGE_TOL = 0.02         # a cut this close to a word edge counts as on the edge


def _pairs(starts, ends):
    """(start, end) word pairs, sorted, dropping malformed rows."""
    out = []
    for s, e in zip(starts or [], ends or []):
        try:
            s, e = float(s), float(e)
        except (TypeError, ValueError):
            continue
        if e >= s:
            out.append((s, e))
    out.sort()
    return out


def refine_words(words, quiet, tol=0.15):
    """Pull Whisper's word edges onto measured silence edges.

    Whisper times are usually right to within 50-150 ms, no better. A silence
    detector measures where the sound actually stops and starts, so when a
    silence edge is within `tol` of a word edge the measured edge wins:
      * a word that ends near the START of a silence ends where that silence starts
      * a word that starts near the END of a silence starts where that silence ends
    Order and length stay sane (a word never ends before it starts)."""
    words = [list(w) for w in (words or [])]
    quiet = sorted((float(a), float(b)) for a, b in (quiet or []) if b > a)
    if not words or not quiet:
        return [tuple(w) for w in words]
    q_starts = [q[0] for q in quiet]
    q_ends = [q[1] for q in quiet]
    for w in words:
        s, e = w
        i = bisect.bisect_left(q_starts, e - tol)
        if i < len(quiet) and abs(q_starts[i] - e) <= tol and q_starts[i] > s + 0.05:
            w[1] = q_starts[i]
        j = bisect.bisect_left(q_ends, s - tol)
        if j < len(quiet) and abs(q_ends[j] - s) <= tol and q_ends[j] < w[1] - 0.05:
            w[0] = q_ends[j]
    return [tuple(w) for w in words]


def _word_at(t, words, tol=EDGE_TOL):
    """Index of the word that t falls inside, else None."""
    for i, (s, e) in enumerate(words):
        if s + tol < t < e - tol:
            return i
        if s > t:
            break
    return None


def _room(gap, limit, floor=0.03):
    """How much of a gap between two words to keep as room: most of it, never
    more than `limit`, never so little the cut is flush to a word."""
    if gap is None:
        return limit
    return min(limit, max(floor, gap * 0.6))


BREATH_MIN = 0.08       # a gap shorter than this is not a place to cut
BREATH_OK = 0.12        # a gap at least this long is
BREATH_WIN = 1.0        # how far to look for one


def settle_in(t, words, lo=0.0, hi=None, sentences=(), lead=LEAD_MAX,
              sent_back=0.6, sent_fwd=0.9, breath_win=BREATH_WIN):
    """A clean in point near `t`.

    `words` is [(start, end), ...]; `lo` is the earliest the picture may start
    (the scene's own start); `hi` the latest. `sentences` is [(start, end)] of
    spoken phrases.

    1. Starting mid-sentence is the common "cut in the middle of a thought"
       fault, so if t falls inside a sentence the in point moves to that
       sentence's start (up to `sent_back` earlier) or the next one's (up to
       `sent_fwd` later), whichever is nearer.
    2. If it then falls inside a word, back up to just before the word, leaving
       room for the transition fade; if the scene does not start early enough,
       go forward to the next word instead.
    3. If a word starts right after t with less than the transition's room,
       pull back to leave it.
    4. If the word boundary is only a hairline gap (two words run together),
       look for the nearest real pause within `breath_win` and cut there."""
    if hi is None:
        hi = float('inf')
    words = sorted(words or [])
    t = max(lo, min(t, hi))

    for s, e in sorted(sentences or []):
        if s + 0.05 < t < e - 0.05:
            options = []
            if t - s <= sent_back and s >= lo:
                options.append(s)
            nxt = [s2 for s2, _e2 in sorted(sentences) if s2 > t and s2 - t <= sent_fwd and s2 <= hi]
            if nxt:
                options.append(nxt[0])
            if options:
                t = min(options, key=lambda x: abs(x - t))
            break

    def gap_before(j):
        return None if j == 0 else words[j][0] - words[j - 1][1]

    def cand_for(j):
        prev_end = words[j - 1][1] if j > 0 else None
        c = words[j][0] - _room(gap_before(j), lead)
        return max(c, prev_end + 0.01) if prev_end is not None else c

    k = _word_at(t, words)
    if k is None:
        nxt = next((n for n, (s, _e) in enumerate(words) if s > t), None)
        if nxt is not None and words[nxt][0] - t < _room(gap_before(nxt), lead):
            k = nxt
    if k is None:
        return t

    g = gap_before(k)
    if g is not None and g < BREATH_MIN:
        best = None
        for j in range(len(words)):
            gj = gap_before(j)
            if gj is not None and gj < BREATH_OK:
                continue
            c = cand_for(j)
            d = abs(words[j][0] - t)
            if lo <= c <= hi and d <= breath_win and (best is None or d < best[0]):
                best = (d, c)
        if best:
            return best[1]
    c = cand_for(k)
    if c >= lo:
        return c
    for j in range(k + 1, len(words)):         # the scene starts inside this word
        c = cand_for(j)
        if lo <= c <= hi:
            return c
    return t


def settle_out(t, words, lo=0.0, hi=None, trail=TRAIL_MAX, breath_win=BREATH_WIN):
    """A clean out point near `t`: after the last word that starts before it,
    with room for the transition fade. If that room does not fit inside the
    scene (`hi`), cut before the word instead, in the gap that precedes it.
    If the boundary is only a hairline gap between two words, the nearest real
    pause within `breath_win` is used instead."""
    if hi is None:
        hi = float('inf')
    words = sorted(words or [])
    t = min(max(t, lo), hi)
    idx = None
    for k, (s, _e) in enumerate(words):
        if s < t - 0.005:
            idx = k
        else:
            break
    if idx is None:
        return t

    def gap_after(j):
        return words[j + 1][0] - words[j][1] if j + 1 < len(words) else None

    def cand_for(j):
        c = words[j][1] + _room(gap_after(j), trail)
        nxt = words[j + 1][0] if j + 1 < len(words) else None
        return min(c, nxt - 0.01) if nxt is not None else c

    s, e = words[idx]
    g = gap_after(idx)
    if g is not None and g < BREATH_MIN:
        best = None
        for j in range(len(words)):
            gj = gap_after(j)
            if gj is not None and gj < BREATH_OK:
                continue
            c = cand_for(j)
            d = abs(words[j][1] - t)
            if lo <= c <= hi and d <= breath_win and (best is None or d < best[0]):
                best = (d, c)
        if best:
            return best[1]
    room = _room(g, trail)
    if t >= e + room - 0.005:
        return t                                    # already clear of the word
    cand = cand_for(idx)
    if lo <= cand <= hi:
        return cand
    # No room to finish the word inside the scene: end before it.
    prev_end = words[idx - 1][1] if idx > 0 else None
    back = prev_end + _room(s - prev_end, trail) if prev_end is not None else s - 0.03
    if lo <= back <= hi:
        return back
    return min(max(cand, lo), hi)


def classify_cut(kind, t, words, sentences=(), tight=0.04, near=1.0):
    """How clean is this edge? ('in' or 'out')

    clean    no speech to protect, or the edge sits at a sentence boundary
    word     on a word boundary, but mid-sentence
    tight    on a boundary, but the speech starts/ends within `tight` s of the
             edge, so the transition can bite into it
    clipped  inside a word
    silent   no speech nearby (b-roll)"""
    words = sorted(words or [])
    if not any(abs(s - t) <= near or abs(e - t) <= near or s <= t <= e for s, e in words):
        return 'silent'
    if _word_at(t, words) is not None:
        return 'clipped'
    if kind == 'in':
        nxt = next(((s, e) for s, e in words if s >= t), None)
        gap = (nxt[0] - t) if nxt else None
        if gap is not None and gap < tight:
            return 'tight'
        edge = nxt[0] if nxt else t
        at_sentence = any(abs(s - edge) <= 0.35 for s, _e in (sentences or []))
    else:
        prv = None
        for s, e in words:
            if e <= t + EDGE_TOL:
                prv = (s, e)
        gap = (t - prv[1]) if prv else None
        if gap is not None and gap < tight:
            return 'tight'
        edge = prv[1] if prv else t
        at_sentence = any(abs(e - edge) <= 0.35 for _s, e in (sentences or []))
    if at_sentence:
        return 'clean'
    inside_sentence = any(s + 0.05 < t < e - 0.05 for s, e in (sentences or []))
    if inside_sentence:
        return 'word'
    return 'clean'


CUT_NOTES = {
    'clean': 'clean',
    'word': 'on a word edge, mid-sentence',
    'tight': 'tight to the words',
    'clipped': 'inside a word',
    'silent': 'no speech',
}


def refine_clip(trim_start, dur, bounds, words, sentences=(), xfade=0.3, min_len=0.8):
    """Settle one clip's in and out. Returns (new_start, new_dur, report).

    `bounds` = (scene_start, scene_end): the footage the clip may use. The
    transition's fade is why the room before and after speech scales with
    `xfade` (capped), and the clip is left exactly as it was if settling would
    make it shorter than `min_len`."""
    lo, hi = bounds
    lead = min(LEAD_MAX, max(0.08, float(xfade)))
    end = trim_start + dur
    new_start = settle_in(trim_start, words, lo=lo, hi=max(lo, end - min_len), sentences=sentences, lead=lead)
    new_end = settle_out(end, words, lo=new_start + min_len, hi=hi, trail=lead)
    new_dur = new_end - new_start
    if new_dur < min_len or new_start < lo - 1e-6 or new_end > hi + 1e-6:
        new_start, new_dur = trim_start, dur
    else:
        new_dur = new_end - new_start
    a = classify_cut('in', new_start, words, sentences)
    b = classify_cut('out', new_start + new_dur, words, sentences)
    report = {'in': a, 'out': b, 'moved_in': round(new_start - trim_start, 3),
              'moved_out': round((new_start + new_dur) - end, 3)}
    return new_start, new_dur, report


def cut_summary(reports):
    """Counts for the review header: how many edges are clean, risky, silent."""
    out = {'clean': 0, 'word': 0, 'tight': 0, 'clipped': 0, 'silent': 0}
    for r in reports:
        for k in ('in', 'out'):
            v = (r or {}).get(k)
            if v in out:
                out[v] += 1
    out['edges'] = sum(out[k] for k in ('clean', 'word', 'tight', 'clipped', 'silent'))
    out['risky'] = out['tight'] + out['clipped']
    return out


# ---------------------------------------------------------------- cards

CARD_MIN = 1.0              # a card still has to read as a card
CARD_VOICE_PAD = 0.25       # keep a breath after a card's own voice
CARD_STILL_PAD = 0.4        # keep a beat after the animation settles


def parse_freeze_start(stderr):
    """First freeze_start from ffmpeg's freezedetect log, else None. The
    picture stops moving there: everything after it is a hold that can be
    shortened without cutting the animation."""
    m = re.search(r'freeze_start:\s*([0-9.]+)', stderr or '')
    return float(m.group(1)) if m else None


def parse_freeze_tail(stderr, duration, tail_tol=0.15):
    """When the picture stops moving for good: the start of the last freeze
    that runs to the end of the card, else None. A freeze in the middle of a
    card that then moves again is not a hold."""
    starts = [float(x) for x in re.findall(r'freeze_start:\s*([0-9.]+)', stderr or '')]
    ends = [float(x) for x in re.findall(r'freeze_end:\s*([0-9.]+)', stderr or '')]
    if not starts:
        return None
    if len(ends) >= len(starts) and ends[-1] < float(duration) - tail_tol:
        return None
    return starts[-1]


def voice_end_from_silences(silences, duration, tail_tol=0.1):
    """Where a card's sound last stops, from its silent intervals. A card whose
    sound runs to the end (music bed) returns its full duration, which makes it
    unshortenable: cutting a bed mid-phrase is worse than a long card."""
    tail = None
    for s, e in sorted(silences or []):
        if e >= float(duration) - tail_tol:
            tail = s
    return float(tail) if tail is not None else float(duration)


def card_floor(duration, voice_end=None, still_from=None, minimum=CARD_MIN):
    """The shortest this card can safely be: past its own voice and past the
    point the picture settles, never under `minimum`, never over its length."""
    floor = minimum
    if voice_end:
        floor = max(floor, voice_end + CARD_VOICE_PAD)
    if still_from is not None:
        floor = max(floor, still_from + CARD_STILL_PAD)
    return min(floor, duration)


def card_adjust_limit(length, policy='fit'):
    """How much the cards may stretch or shrink in total, to close the last
    gap. 'off' keeps the old fixed 1.0 s; 'fit' scales with the plug (12 % of
    it, never under 1 s): a 30 s plug can give 3.6 s, a 10 s one 1.2 s."""
    if policy == 'off':
        return 1.0
    return max(1.0, round(0.12 * float(length), 2))


def min_scene_time(length):
    """The least scene time a plug should have. Cards that would push scenes
    under this are shortened first."""
    return max(5.0, 0.45 * float(length))


def plan_card_fit(length, durations, floors, min_scenes=None):
    """Targets for each card before scenes are chosen.

    If cards leave the scenes at least `min_scenes` seconds, nothing changes.
    Otherwise the shortfall is taken from the cards in proportion to how far
    each is above its floor. Returns {'targets': [...], 'scene_budget': x,
    'short_by': y, 'shrunk': bool}; short_by > 0 means even at their floors the
    cards leave the scenes too little."""
    durations = [float(d) for d in durations]
    floors = [min(float(f), d) for f, d in zip(floors, durations)]
    if min_scenes is None:
        min_scenes = min_scene_time(length)
    total = sum(durations)
    budget = float(length) - total
    targets = list(durations)
    need = min_scenes - budget
    shrunk = False
    if need > 0.01 and durations:
        room = [max(0.0, d - f) for d, f in zip(durations, floors)]
        avail = sum(room)
        take = min(need, avail)
        if avail > 0.01:
            targets = [d - take * (r / avail) for d, r in zip(durations, room)]
            shrunk = take > 0.01
        need -= take
    scene_budget = float(length) - sum(targets)
    return {'targets': targets, 'scene_budget': scene_budget,
            'short_by': max(0.0, need), 'shrunk': shrunk, 'min_scenes': min_scenes}


def card_budget_line(length, card_secs, labels=None):
    """'15 s = title 3.0 + scenes 8.4 + end 3.6 - transitions 0.6' style text
    pieces for the Ready? panel (the browser builds its own; this keeps the
    server's wording and the tests in one place)."""
    labels = labels or ['title', 'end card']
    parts = [f'{l} {c:.1f}' for l, c in zip(labels, card_secs)]
    scenes = float(length) - sum(card_secs)
    return f'{length:g} s = ' + ' + '.join(parts) + f' + scenes {scenes:.1f}'


# ---------------------------------------------------------------- script

_TIME_RE = re.compile(
    r'^\s*[\[\(]?\s*(?:@\s*)?'
    r'(?:(\d{1,2}):)?(\d{1,2}):(\d{2})(?:[.,:](\d{1,3}))?'
    r'\s*(?:s|sec)?\s*[\]\)]?\s*[-–—:]*\s*', re.I)
_AT_RE = re.compile(r'^\s*@\s*(\d+(?:[.,]\d+)?)\s*s?\b\s*[-–—:]*\s*', re.I)
_VO_TAG_RE = re.compile(r'^\s*(?:V\.?O\.?|VOICE\s*OVER|NARRATOR|NARRATION|SOT)\s*[:\-–—]\s*', re.I)
_OPT_RE = re.compile(r'[\[\(]\s*(?:optional|opt)\s*[\]\)]|\bOPTIONAL\s*:', re.I)


def tc_to_seconds(h, m, s, frac):
    """Clock fields to seconds. Two fields are MM:SS; three are HH:MM:SS. The
    fraction is milliseconds if it has three digits, else a decimal."""
    total = (int(h) if h else 0) * 3600 + int(m) * 60 + int(s)
    if frac:
        total += float('0.' + frac)
    return float(total)


def parse_vo_script(text):
    """Read narration text into lines: [{'at': seconds|None, 'text', 'optional'}].

    A leading time marks where the line goes on the plug's timeline:
        00:03 VO: Abangan ngayong gabi
        [0:07.5] ...        @12.5s ...
    Lines without a time follow the previous line. A line marked (optional) is
    the first thing dropped if the narration will not fit. Delivery cues in
    [brackets] are removed."""
    lines = []
    for raw in str(text or '').replace('\r\n', '\n').replace('\r', '\n').split('\n'):
        line = raw.strip()
        if not line:
            continue
        at = None
        m = _AT_RE.match(line)
        if m:
            at = float(m.group(1).replace(',', '.'))
            line = line[m.end():]
        else:
            m = _TIME_RE.match(line)
            if m and (m.group(1) is not None or int(m.group(3)) < 60):
                try:
                    at = tc_to_seconds(m.group(1), m.group(2), m.group(3), m.group(4))
                    line = line[m.end():]
                except ValueError:
                    at = None
        line = _VO_TAG_RE.sub('', line)
        optional = bool(_OPT_RE.search(line))
        line = _OPT_RE.sub(' ', line)
        line = re.sub(r'\[[^\]]*\]', ' ', line)
        line = re.sub(r'\s+', ' ', line).strip()
        if line:
            lines.append({'at': at, 'text': line, 'optional': optional})
    return lines


def strip_vo_markers(text):
    """The spoken words only: what selection and matching should see."""
    return ' '.join(l['text'] for l in parse_vo_script(text))


def has_vo_times(text):
    return any(l['at'] is not None for l in parse_vo_script(text))


def norm_tokens(text):
    text = unicodedata.normalize('NFKD', str(text or '').lower())
    text = ''.join(c for c in text if not unicodedata.combining(c))
    return re.findall(r"[a-z0-9ñ']+", text)


def words_from_segments(segments):
    """Spread each segment's words evenly across its time: coarse stand-ins for
    word times when the speech service gave line times only."""
    words = []
    for seg in segments or []:
        toks = (seg.get('text') or '').split()
        if not toks:
            continue
        s, e = float(seg.get('start') or 0), float(seg.get('end') or 0)
        if e <= s:
            continue
        step = (e - s) / len(toks)
        for k, tok in enumerate(toks):
            words.append({'word': tok, 'start': s + k * step, 'end': s + (k + 1) * step - 0.01})
    return words


def align_script(lines, words, min_score=0.4):
    """Find each script line in the VO's words, in order.

    Returns one entry per line: {'w0', 'w1', 'score'} (word indexes, inclusive)
    or {'w0': None, ...} when the line was not found. Matching is by shared
    words in sequence, so a small mishearing does not lose the line."""
    toks = [norm_tokens(w.get('word', '')) for w in words]
    flat, owner = [], []
    for i, tk in enumerate(toks):
        for t in tk:
            flat.append(t)
            owner.append(i)
    out = []
    cursor = 0
    for ln in lines:
        want = norm_tokens(ln['text'])
        if not want or cursor >= len(flat):
            out.append({'w0': None, 'w1': None, 'score': 0.0})
            continue
        sm = difflib.SequenceMatcher(None, flat[cursor:], want, autojunk=False)
        blocks = [b for b in sm.get_matching_blocks() if b.size]
        matched = sum(b.size for b in blocks)
        score = matched / len(want)
        if not blocks or score < min_score:
            out.append({'w0': None, 'w1': None, 'score': round(score, 2)})
            continue
        first, last = blocks[0], blocks[-1]
        f0 = max(cursor, cursor + first.a - first.b)
        f1 = min(len(flat) - 1, cursor + last.a + last.size - 1 + (len(want) - (last.b + last.size)))
        out.append({'w0': owner[f0], 'w1': owner[max(f0, f1)], 'score': round(score, 2)})
        cursor = f1 + 1
    return out


# ---------------------------------------------------------------- narration

PAUSE_KEEP = 0.15       # shortest pause kept when tightening
SPLIT_GAP = 0.6         # a pause this long inside a line can be tightened
LINE_GAP = 0.25         # default pause between lines with no time of their own
MAX_TEMPO = 1.08        # the most the narration is sped up (pitch is kept)
VO_FADE = 0.03          # fade on every cut edge
VO_TAIL = 0.15          # keep this much of the plug's end free of narration


def _tidy_words(words):
    out = []
    for w in words or []:
        try:
            s, e = float(w['start']), float(w['end'])
        except (KeyError, TypeError, ValueError):
            continue
        out.append({'word': w.get('word', ''), 'start': s, 'end': max(e, s)})
    out.sort(key=lambda x: x['start'])
    return out


def _line_elements(words, w0, w1, lo, hi):
    """Cut ranges for one line: [{'a','b','gap_before'}], split at long pauses.
    The line gets gap-aware room before its first word and after its last, and
    nothing reaches past `lo`/`hi` (the neighbours' speech)."""
    pairs = [(w['start'], w['end']) for w in words]
    first, last = pairs[w0], pairs[w1]
    gap_prev = (first[0] - pairs[w0 - 1][1]) if w0 > 0 else None
    gap_next = (pairs[w1 + 1][0] - last[1]) if w1 + 1 < len(pairs) else None
    a = max(lo, first[0] - _room(gap_prev, 0.12))
    b = min(hi, last[1] + _room(gap_next, 0.15))
    a = min(a, first[0])
    b = max(b, last[1])
    cuts = []
    for i in range(w0, w1):
        if pairs[i + 1][0] - pairs[i][1] >= SPLIT_GAP:
            cuts.append((pairs[i][1] + 0.1, pairs[i + 1][0] - 0.1))
    elems, seg_a = [], a
    for end_i, start_next in cuts:
        elems.append({'a': seg_a, 'b': end_i})
        seg_a = start_next
    elems.append({'a': seg_a, 'b': b})
    prev_b = None
    for el in elems:
        el['gap_before'] = None if prev_b is None else max(0.0, el['a'] - prev_b)
        prev_b = el['b']
    return elems


def plan_vo(lines, words, *, start_at=0.0, window_end=None, max_tempo=MAX_TEMPO, aligned=None):
    """Cut the recorded narration at word edges and place it.

    lines    parse_vo_script() output (with no script, one line per spoken
             sentence, only the first with a time).
    words    Whisper words of the recording [{'word','start','end'}], times
             relative to the file (any trim already applied).
    start_at where the first line starts when it has no time of its own.
    window_end  the narration must end by this plug time (None = no limit).

    Returns {'pieces', 'notes', 'fits', 'end', 'lines'}. A piece is one cut of
    the recording: {'a','b','at','tempo','fade_in','fade_out','line'}.

    When the narration is too long for the room it has, in order: pauses are
    tightened to 0.15 s, the speech is sped up by at most 8 %, lines marked
    optional are dropped from the end, and as the last step the end is trimmed
    at a word edge. The cut never falls inside a word."""
    words = _tidy_words(words)
    notes = []
    if not words or not lines:
        return {'pieces': [], 'notes': ['No speech found in the recording.'],
                'fits': True, 'end': start_at, 'lines': []}
    aligned = aligned or align_script(lines, words)

    bounds = []
    for k, al in enumerate(aligned):
        if al['w0'] is None:
            bounds.append(None)
            continue
        prev_hi = next((aligned[j]['w1'] for j in range(k - 1, -1, -1) if aligned[j]['w1'] is not None), None)
        next_lo = next((aligned[j]['w0'] for j in range(k + 1, len(aligned)) if aligned[j]['w0'] is not None), None)
        lo = words[prev_hi]['end'] + 0.01 if prev_hi is not None else max(0.0, words[al['w0']]['start'] - 0.3)
        hi = words[next_lo]['start'] - 0.01 if next_lo is not None else words[al['w1']]['end'] + 0.3
        bounds.append(_line_elements(words, al['w0'], al['w1'], lo, hi))

    info, runs = [], []
    for k, ln in enumerate(lines):
        row = {'text': ln['text'], 'status': 'ok', 'at': None, 'end': None,
               'score': aligned[k]['score'], 'optional': ln['optional']}
        info.append(row)
        if bounds[k] is None:
            row['status'] = 'missing'
            notes.append(f'Line {k + 1} was not found in the recording: "{ln["text"][:50]}".')
            continue
        explicit = ln['at'] is not None
        if explicit or not runs:
            runs.append({'at': ln['at'] if explicit else float(start_at), 'items': []})
        elif runs[-1]['items']:
            # Keep the pause the speaker actually left between these lines.
            prev_last = runs[-1]['items'][-1][1][-1]
            bounds[k][0]['gap_before'] = min(1.5, max(PAUSE_KEEP, bounds[k][0]['a'] - prev_last['b']))
        runs[-1]['items'].append((k, bounds[k]))

    pieces = []
    for ri, run in enumerate(runs):
        deadline = window_end
        if ri + 1 < len(runs):
            nxt = runs[ri + 1]['at'] - 0.1
            deadline = nxt if deadline is None else min(deadline, nxt)
        pieces.extend(_fit_run(run, deadline, lines, info, max_tempo, notes, words))

    end = max((p['at'] + (p['b'] - p['a']) / p['tempo'] for p in pieces), default=start_at)
    fits = window_end is None or end <= window_end + 0.02
    return {'pieces': pieces, 'notes': notes, 'fits': fits, 'end': round(end, 3), 'lines': info}


def _fit_run(run, deadline, lines, info, max_tempo, notes, words):
    """Lay out one run from its anchor and fit it before `deadline`.
    Returns that run's pieces."""
    anchor = run['at']
    items = [(k, [dict(e) for e in els]) for k, els in run['items']]

    def flat(items_):
        seq = []
        for k, els in items_:
            for n, e in enumerate(els):
                natural = e['gap_before'] if e['gap_before'] is not None else (LINE_GAP if n == 0 else PAUSE_KEEP)
                seq.append({'line': k, 'a': e['a'], 'b': e['b'], 'first': n == 0,
                            'natural': max(PAUSE_KEEP, natural)})
        return seq

    def layout(seq, squeeze, tempo):
        """squeeze 0 keeps the natural pauses, 1 tightens them to PAUSE_KEEP."""
        t, out = anchor, []
        for idx, e in enumerate(seq):
            if idx > 0:
                t += PAUSE_KEEP + (e['natural'] - PAUSE_KEEP) * (1.0 - squeeze)
            dur = (e['b'] - e['a']) / tempo
            out.append({'line': e['line'], 'a': e['a'], 'b': e['b'], 'at': t, 'tempo': tempo, 'dur': dur})
            t += dur
        return out, t

    seq = flat(items)
    laid, end = layout(seq, 0.0, 1.0)
    how = None
    if deadline is not None and end > deadline + 0.02:
        laid, end = layout(seq, 1.0, 1.0)
        how = 'tightened'
        if end > deadline + 0.02:
            need = (end - anchor) / max(deadline - anchor, 0.01)
            laid, end = layout(seq, 1.0, min(max_tempo, need))
            how = 'sped up'
        while end > deadline + 0.02:
            opt = [k for k, _els in items if lines[k]['optional']]
            if not opt:
                break
            drop = opt[-1]
            items = [(k, els) for k, els in items if k != drop]
            info[drop]['status'] = 'dropped'
            notes.append(f'Dropped optional line {drop + 1} to fit: "{lines[drop]["text"][:50]}".')
            seq = flat(items)
            if not seq:
                laid, end = [], anchor
                break
            laid, end = layout(seq, 1.0, max_tempo)
            how = 'sped up'
        if seq and end > deadline + 0.02:
            laid, end = layout(seq, 1.0, max_tempo)
            laid, end, cut_line = _trim_to(laid, deadline, words)
            how = 'trimmed'
            if cut_line is not None:
                notes.append(f'Narration was cut at a word edge to end by {deadline:.1f}s.')
        if how == 'tightened':
            notes.append('Pauses were tightened to fit the narration.')
        elif how == 'sped up':
            notes.append(f'Narration sped up slightly (at most {int(round((max_tempo - 1) * 100))} %) to fit.')

    have = {}
    for p in laid:
        have.setdefault(p['line'], []).append(p)
    for k, _els in items:
        row = info[k]
        if k not in have:
            if row['status'] != 'dropped':
                row['status'] = 'cut off'
            continue
        row['at'] = round(have[k][0]['at'], 2)
        row['end'] = round(have[k][-1]['at'] + have[k][-1]['dur'], 2)
        if how == 'trimmed' and (k == laid[-1]['line']) and row['status'] == 'ok':
            row['status'] = 'trimmed'
        elif how in ('tightened', 'sped up') and row['status'] == 'ok':
            row['status'] = how
    return [{'line': p['line'], 'a': round(p['a'], 3), 'b': round(p['b'], 3), 'at': round(p['at'], 3),
             'tempo': round(p['tempo'], 4), 'fade_in': VO_FADE, 'fade_out': VO_FADE} for p in laid]


def _trim_to(laid, deadline, words):
    """Cut a laid-out run so it ends by `deadline`, on a word edge.
    Returns (kept pieces, end time, line that was cut or None)."""
    kept, cut_line = [], None
    end = laid[0]['at'] if laid else 0.0
    for p in laid:
        p_end = p['at'] + p['dur']
        if p_end <= deadline + 0.02:
            kept.append(p)
            end = p_end
            continue
        cut_line = p['line']
        room = (deadline - p['at']) * p['tempo']         # source seconds that still fit
        if room > 0.1:
            limit = p['a'] + room
            fit = [w for w in words if w['start'] >= p['a'] - 0.01 and w['end'] + 0.05 <= limit]
            if fit:
                nxt = next((w['start'] for w in words if w['start'] > fit[-1]['start']), None)
                b = min(fit[-1]['end'] + 0.08, limit, (nxt - 0.01) if nxt is not None else 1e9)
                if b - p['a'] > 0.1:
                    q = dict(p)
                    q['b'] = b
                    q['dur'] = (b - p['a']) / p['tempo']
                    kept.append(q)
                    end = q['at'] + q['dur']
        break
    return kept, end, cut_line


def vo_filter(pieces, sample_rate=44100):
    """ffmpeg filter_complex that builds the edited narration from input 0:
    each piece trimmed at its word edges, sped up if needed, faded, and laid
    end to end with silence between. Returns (graph, out_label) or (None,
    None) when there is nothing to build."""
    if not pieces:
        return None, None
    pieces = sorted(pieces, key=lambda p: p['at'])
    n = len(pieces)
    parts = []
    if n == 1:
        parts.append('[0:a]acopy[src0]')
    else:
        parts.append('[0:a]asplit=%d' % n + ''.join(f'[src{i}]' for i in range(n)))
    labels = []
    t = 0.0
    seg = 0
    for i, p in enumerate(pieces):
        gap = p['at'] - t
        if gap > 0.002:
            parts.append(f'anullsrc=r={sample_rate}:cl=stereo,atrim=duration={gap:.4f}[gap{seg}]')
            labels.append(f'[gap{seg}]')
            seg += 1
        dur_src = p['b'] - p['a']
        dur = dur_src / p['tempo']
        fi = min(p.get('fade_in', VO_FADE), dur / 3)
        fo = min(p.get('fade_out', VO_FADE), dur / 3)
        chain = [f'atrim=start={p["a"]:.4f}:end={p["b"]:.4f}', 'asetpts=PTS-STARTPTS']
        if abs(p['tempo'] - 1.0) > 0.001:
            chain.append(f'atempo={p["tempo"]:.4f}')
        chain.append(f'aformat=sample_rates={sample_rate}:channel_layouts=stereo')
        chain.append(f'afade=t=in:st=0:d={fi:.4f}')
        chain.append(f'afade=t=out:st={max(0.0, dur - fo):.4f}:d={fo:.4f}')
        parts.append(f'[src{i}]' + ','.join(chain) + f'[pc{i}]')
        labels.append(f'[pc{i}]')
        t = p['at'] + dur
    parts.append(''.join(labels) + f'concat=n={len(labels)}:v=0:a=1[vo]')
    return ';'.join(parts), '[vo]'

"""
Vertical Shorts -- the decisions in shorts_core.py, with no services, no
Flask and (in this file) no ffmpeg: what gets proposed, where a window is
allowed to start and end, how it is scored, and how a shot is reframed.

Everything here is pure logic over plain values, so these run in
milliseconds and pin the behaviour the tab's docs promise an editor:

  * a window never starts or ends inside a spoken word;
  * a too-long moment loses its opening, never its ending;
  * a moment is not extended across a scene-length pause;
  * the same moment found by two overlapping transcript chunks appears once;
  * a wide two-shot is shown whole in Auto rather than cropping a person out;
  * the crop only moves when the subject really does, and every frame of a
    clip is covered by exactly one reframing instruction;
  * "Follow the speaker" only ever changes a wide two-shot, only cuts to a
    mouth that is moving while words are being spoken, and leaves a shot it
    cannot call exactly as it would have been with the option off.
"""
import json
import re

import cv2
import numpy as np

import shorts_core as sc


def _segs(spec):
    """[(start, end, text)] -> segment dicts."""
    return [{'start': a, 'end': b, 'text': t} for a, b, t in spec]


def _dialogue(n=30, line=3.0, gap=0.5, start=1.0):
    """n evenly spaced lines with word timings inside each."""
    segs, words, t = [], [], start
    for i in range(n):
        text = f'Line number {i} here.'
        ws = text.split()
        step = line / len(ws)
        for k, w in enumerate(ws):
            words.append({'start': round(t + k * step, 3), 'end': round(t + k * step + step * 0.8, 3), 'word': w})
        segs.append({'start': t, 'end': round(t + line - step * 0.2, 3), 'text': text})
        t += line + gap
    return words, segs


# ---- transcript shaping ----

def test_normalize_drops_malformed_rows_and_sorts():
    words = [{'start': 2.0, 'end': 2.3, 'word': 'b'}, {'start': 1.0, 'end': 1.2, 'word': ' a '},
             {'start': 'x', 'end': 1, 'word': 'bad'}, {'start': 3, 'end': 3.2, 'word': '  '}]
    segs = [{'start': 5, 'end': 6, 'text': ' second  line '}, {'start': 0, 'end': 0, 'text': 'zero length'},
            {'start': 1, 'end': 2.4, 'text': 'first'}, {'end': 9, 'text': 'no start'}]
    w, s = sc.normalize_transcript(words, segs)
    assert [x['word'] for x in w] == ['a', 'b']
    assert [x['text'] for x in s] == ['first', 'second line']


def test_normalize_splits_a_run_on_segment_using_word_timings():
    # One 30 s segment of unpunctuated-then-punctuated speech: left whole it
    # would make every window boundary 30 s coarse.
    words, t = [], 0.0
    for i in range(60):
        words.append({'start': t, 'end': t + 0.4, 'word': f'w{i}' + ('.' if i % 10 == 9 else '')})
        t += 0.5
    _, segs = sc.normalize_transcript(words, [{'start': 0.0, 'end': 30.0, 'text': 'x ' * 60}])
    assert len(segs) > 2
    assert max(s['end'] - s['start'] for s in segs) <= 12.0
    assert segs[0]['start'] == 0.0 and abs(segs[-1]['end'] - 29.9) < 1e-6


def test_segments_are_built_from_words_when_the_service_returns_none():
    words, _ = _dialogue(4)
    w, segs = sc.normalize_transcript(words, [])
    assert len(segs) == 4 and segs[0]['text'] == 'Line number 0 here.'


def test_chunks_cover_every_line_and_overlap():
    _, segs = _dialogue(200)            # ~700 s of dialogue
    chunks = sc.chunk_segments(segs, chunk_sec=300, overlap_sec=60)
    assert chunks[0][0] == 0 and chunks[-1][1] == len(segs)
    covered = set()
    for lo, hi in chunks:
        assert hi > lo
        assert segs[hi - 1]['end'] - segs[lo]['start'] <= 300 + 1e-6
        covered.update(range(lo, hi))
    assert covered == set(range(len(segs)))
    for (_, hi_a), (lo_b, _) in zip(chunks, chunks[1:]):
        assert lo_b < hi_a, 'consecutive chunks must share lines'
        assert segs[hi_a - 1]['end'] - segs[lo_b]['start'] >= 50


def test_chunking_terminates_on_a_single_line_longer_than_a_chunk():
    segs = _segs([(0, 500, 'one enormous line'), (501, 503, 'next')])
    assert sc.chunk_segments(segs, 300, 60) == [(0, 1), (1, 2)]


# ---- story layer ----

def test_story_prompt_uses_global_ids_and_inlines_visual_notes_in_time_order():
    _, segs = _dialogue(20)
    visual = [{'t': segs[11]['start'] + 0.1, 'score': 5, 'desc': 'woman crying'},
              {'t': 9999.0, 'score': 4, 'desc': 'outside this chunk'},
              {'t': segs[10]['start'] + 0.1, 'score': None, 'desc': 'unrated frame'}]
    p = sc.build_story_prompt(segs, 10, 14, visual, 30, 90, max_moments=3, focus='the  missing\nwill')
    ids = [int(x) for x in re.findall(r'^\[(\d+)\]', p, re.M)]
    assert ids == [10, 11, 12, 13]
    assert 'woman crying (visual drama 5/5)' in p
    assert 'outside this chunk' not in p and 'unrated frame' not in p
    assert p.index('[11]') < p.index('woman crying') < p.index('[12]')
    assert '30 to 90 seconds' in p and 'the missing will' in p


def test_story_reply_parsing_accepts_what_models_actually_send():
    good = {'moments': [{'start_id': 12, 'end_id': 15, 'title': 'T', 'hook': 'h', 'why': 'w', 'score': 8}]}
    for text in (json.dumps(good),
                 '```json\n' + json.dumps(good) + '\n```',
                 'Here are the moments:\n' + json.dumps(good) + '\nHope that helps.',
                 json.dumps(good['moments'])):
        beats = sc.parse_story_reply(text, 10, 20)
        assert [(b['start_id'], b['end_id'], b['score']) for b in beats] == [(12, 15, 8)], text


def test_story_reply_salvages_complete_objects_from_a_truncated_reply():
    text = ('{"moments":[{"start_id":11,"end_id":13,"title":"A","score":7},'
            '{"start_id":15,"end_id":18,"title":"B","sco')
    assert [(b['start_id'], b['end_id']) for b in sc.parse_story_reply(text, 10, 20)] == [(11, 13)]


def test_story_reply_validates_ids_against_the_chunk_it_was_asked_about():
    text = json.dumps({'moments': [
        {'start_id': 400, 'end_id': 410, 'title': 'invented ids', 'score': 9},
        {'start_id': 18, 'end_id': 25, 'title': 'overhangs', 'score': 99},
        {'start_id': 14, 'end_id': 12, 'title': 'reversed', 'score': 'high'},
        {'start_id': 'x', 'end_id': 3, 'title': 'garbage', 'score': 5},
    ]})
    beats = sc.parse_story_reply(text, 10, 20)
    assert [(b['title'], b['start_id'], b['end_id'], b['score']) for b in beats] == [
        ('overhangs', 18, 19, 10), ('reversed', 12, 14, 5)]


def test_story_reply_with_nothing_usable_is_an_empty_list():
    for text in ('', 'I could not find any.', '{"moments": []}', '{"moments": "none"}', '[1, 2, 3]'):
        assert sc.parse_story_reply(text, 0, 10) == []


def test_heuristic_and_visual_fallbacks_are_labelled_as_such():
    _, segs = _dialogue(60)
    segs[20]['text'] = 'Bakit?! Hindi totoo yan!'
    beats = sc.heuristic_beats(segs, 20, 40, limit=5)
    assert beats and all(b['source'] == 'heuristic' and b['score'] is None for b in beats)
    assert any(b['start_id'] <= 20 <= b['end_id'] for b in beats[:3]), 'the heated stretch should rank near the top'

    vis = [{'t': 100.0, 'score': 5, 'desc': 'a fight'}, {'t': 300.0, 'score': 2, 'desc': 'empty room'}]
    wins = sc.visual_windows(vis, 20, 40, duration=600, limit=5)
    assert len(wins) == 1 and wins[0]['source'] == 'visual' and wins[0]['t0'] < 100 < wins[0]['t1']
    assert sc.heuristic_beats([], 20, 40) == [] and sc.visual_windows([], 20, 40, 600) == []


# ---- vision layer ----

def test_vision_reply_parsing():
    assert sc.parse_vision_reply('{"score": 4, "desc": "two people  arguing"}') == (4, 'two people arguing')
    assert sc.parse_vision_reply('{"score": 5, "desc": "a woman cry') == (5, 'a woman cry')   # cut off mid-string
    assert sc.parse_vision_reply('SCORE: 3 - people talking')[0] == 3
    assert sc.parse_vision_reply('I would rate this 2/5.')[0] == 2
    assert sc.parse_vision_reply('{"score": 9, "desc": "out of range"}') == (None, 'out of range')
    assert sc.parse_vision_reply('') == (None, '')


def test_vision_sampling_is_bounded_and_stays_inside_shots():
    shots = [(i * 4.0, i * 4.0 + 4.0) for i in range(675)]        # a 45-minute episode
    times = sc.vision_sample_times(shots, 2700.0, budget=90)
    assert 80 <= len(times) <= 90
    assert times == sorted(times) and times[0] > 0 and times[-1] < 2700
    for t in times:
        s = (t // 4.0) * 4.0
        assert s + 0.29 <= t <= s + 3.71, 'a sample must not sit on a cut'
    assert len(sc.vision_sample_times([(0, 60)], 60.0, budget=90)) == 10      # min gap wins on a short source
    assert sc.vision_sample_times([], 0, 90) == []


# ---- fitting a proposal to the length limits ----

def test_a_too_long_moment_loses_its_opening_not_its_ending():
    _, segs = _dialogue(40)                         # 3.5 s per line
    i, j, flags = sc.fit_indices(segs, 5, 35, min_dur=20, max_dur=40)
    assert j == 35, 'the ending the model chose must survive'
    assert i > 5 and segs[j]['end'] - segs[i]['start'] <= 40
    assert segs[j]['end'] - segs[i - 1]['start'] > 40, 'only as much as needed is dropped'
    assert flags == ['trimmed']


def test_a_too_short_moment_grows_backwards_first_then_alternates():
    _, segs = _dialogue(40)
    i, j, flags = sc.fit_indices(segs, 20, 21, min_dur=20, max_dur=40)
    assert segs[j]['end'] - segs[i]['start'] >= 20
    assert i < 20 and j > 21 and (20 - i) >= (j - 21)
    assert flags == ['extended']


def test_extension_does_not_cross_a_scene_length_pause():
    segs = _segs([(0, 3, 'previous scene'), (20, 23, 'a'), (23.5, 26, 'b'), (40, 43, 'next scene')])
    i, j, flags = sc.fit_indices(segs, 1, 2, min_dur=30, max_dur=60)
    assert (i, j) == (1, 2)
    assert 'short' in flags


# ---- placing the cut ----

def test_snap_starts_and_ends_on_a_nearby_cut():
    units = [(10.0, 10.5), (10.6, 12.0), (30.0, 31.0), (31.2, 33.0)]
    cuts = [0.0, 9.4, 20.0, 34.5, 60.0]
    start, end = sc.snap_window(10.0, 33.0, cuts, units, 60.0)
    assert (start, end) == (9.4, 34.5)


def test_snap_falls_back_to_a_short_lead_and_tail_without_a_usable_cut():
    units = [(10.0, 12.0), (30.0, 33.0)]
    start, end = sc.snap_window(10.0, 33.0, [0.0, 5.0, 50.0, 60.0], units, 60.0)
    assert abs(start - 9.75) < 1e-6 and abs(end - 33.5) < 1e-6


def test_snap_never_lands_inside_neighbouring_speech():
    # Speech butts right up against both ends, and there are cuts INSIDE the
    # neighbouring lines that a cut-first rule would happily jump to.
    units = [(5.0, 9.9), (10.0, 12.0), (30.0, 33.0), (33.1, 36.0)]
    cuts = [0.0, 9.0, 34.0, 60.0]
    start, end = sc.snap_window(10.0, 33.0, cuts, units, 60.0)
    assert 9.9 <= start <= 10.0, 'must not include the tail of the previous line'
    assert 33.0 <= end <= 33.1, 'must not include the head of the next line'


def test_snap_at_the_very_edges_of_the_programme():
    start, end = sc.snap_window(0.1, 59.8, [0.0, 60.0], [(0.1, 2.0), (58.0, 59.8)], 60.0)
    assert start == 0.0 and end == 60.0


# ---- scoring and de-duplication ----

def test_score_weights_story_over_visual_over_pace():
    words = [{'start': t / 2.0, 'end': t / 2.0 + 0.3, 'word': 'w'} for t in range(200)]   # 2 words/s: ideal pace
    vis = [{'t': 10.0, 'score': 5}, {'t': 20.0, 'score': 1}]
    full = sc.window_scores(0, 60, 10, [{'t': 10.0, 'score': 5}], words, [])
    assert full['score'] == 100 and full['visual'] == 5.0 and full['pace'] == 2.0
    strong_story = sc.window_scores(0, 60, 9, [{'t': 10.0, 'score': 2}], words, [])
    strong_look = sc.window_scores(0, 60, 3, [{'t': 10.0, 'score': 5}], words, [])
    assert strong_story['score'] > strong_look['score']
    assert sc.window_scores(0, 60, 5, vis, words, [])['visual'] == 5.0, 'the stronger half of the frames counts'


def test_missing_components_are_left_out_not_counted_as_zero():
    words = [{'start': t / 2.0, 'end': t / 2.0 + 0.3, 'word': 'w'} for t in range(200)]
    no_story = sc.window_scores(0, 60, None, [{'t': 10.0, 'score': 5}], words, [])
    assert no_story['story'] is None and no_story['score'] == 100
    nothing = sc.window_scores(0, 60, None, [], [], [])
    assert nothing['visual'] is None and nothing['score'] == 0      # silent and unrated: pace alone, and it is zero


def test_dedupe_keeps_the_better_of_two_overlapping_windows():
    cands = [{'start': 100, 'end': 160, 'score': 70}, {'start': 110, 'end': 165, 'score': 85},
             {'start': 155, 'end': 215, 'score': 60}, {'start': 400, 'end': 430, 'score': 10}]
    kept = sc.dedupe_windows(cands)
    assert [(c['start'], c['score']) for c in kept] == [(110, 85), (155, 60), (400, 10)]


def test_build_candidates_end_to_end():
    words, segs = _dialogue(60)                     # 3 s lines, 0.5 s gaps, from 1.0 s
    cuts = [0.0] + [s['start'] - 0.2 for s in segs[::2]] + [215.0]
    visual = [{'t': t, 'score': 4, 'desc': ''} for t in range(5, 210, 10)]
    beats = [
        {'start_id': 10, 'end_id': 19, 'score': 9, 'title': 'Reveal', 'hook': 'h', 'why': 'w', 'source': 'story'},
        {'start_id': 11, 'end_id': 20, 'score': 6, 'title': 'Same moment again', 'source': 'story'},
        {'start_id': 40, 'end_id': 41, 'score': 7, 'title': 'Too short', 'source': 'story'},
    ]
    out = sc.build_candidates(beats, segs, words, cuts, visual, 215.0, min_dur=20, max_dur=40, limit=5, fps=25.0)
    assert [c['title'] for c in out] == ['Reveal', 'Too short']
    assert [c['id'] for c in out] == ['c1', 'c2']
    first = out[0]
    # Starts on the cut 0.2 s before line 10, ends on the cut before line 20.
    assert abs(first['start'] - (segs[10]['start'] - 0.2)) < 0.021
    assert abs(first['end'] - (segs[20]['start'] - 0.2)) < 0.021
    for c in out:
        assert 20 <= c['duration'] <= 40.5
        assert abs(c['start'] * 25 - round(c['start'] * 25)) < 1e-6, 'in point must be a whole frame'
        assert abs(c['end'] * 25 - round(c['end'] * 25)) < 1e-6
        inside = [w for w in words if c['start'] < w['end'] and w['start'] < c['end']]
        assert all(c['start'] <= w['start'] and w['end'] <= c['end'] for w in inside), 'no word is cut in half'
    assert 'extended' in out[1]['flags'] and out[1]['story_score'] == 7
    assert out[0]['score'] > out[1]['score']


def test_build_candidates_respects_the_limit_and_drops_slivers():
    words, segs = _dialogue(60)
    beats = [{'start_id': i, 'end_id': i + 8, 'score': 5, 'title': f'm{i}'} for i in range(0, 50, 10)]
    assert len(sc.build_candidates(beats, segs, words, [], [], 215.0, 20, 40, limit=3)) == 3
    assert sc.build_candidates([{'t0': 10.0, 't1': 11.0, 'source': 'visual'}], [], [], [], [], 60.0) == []


# ---- reframing ----

def test_crop_geometry():
    assert sc.crop_geometry(1920, 1080) == (608, 1080)
    assert sc.crop_geometry(1280, 720) == (404, 720)
    assert sc.crop_geometry(1080, 1920) == (1080, 1920)        # already vertical: nothing to crop
    assert sc.crop_geometry(854, 480) == (270, 480)
    assert sc.crop_geometry(608, 1920) == (608, 1080)          # narrower than 9:16: full width, trimmed height
    w, h = sc.crop_geometry(1920, 1080)
    assert w % 2 == 0 and h % 2 == 0 and abs(w / h - 9 / 16) < 0.002


def _face(cx, w=160, cy=400):
    return (float(cx), float(cy), float(w), float(w))


def _samples(n_frames, face_fn, step=5):
    return [(i, face_fn(i)) for i in range(0, n_frames, step)]


def test_a_single_face_gets_a_locked_off_crop_centred_on_it():
    samples = _samples(100, lambda i: [_face(500 + (i % 3))])       # detector jitter, not movement
    segs = sc.plan_reframe(samples, [], 100, 1920, 1080, 608)
    assert len(segs) == 1 and segs[0]['layout'] == 'crop' and segs[0]['keys'] is None
    assert abs(segs[0]['x'] - (501 - 304)) <= 2
    assert (segs[0]['a'], segs[0]['b']) == (0, 99)


def test_the_crop_changes_only_on_the_cut():
    samples = _samples(200, lambda i: [_face(400)] if i < 100 else [_face(1500)])
    segs = sc.plan_reframe(samples, [100], 200, 1920, 1080, 608)
    assert [(s['a'], s['b'], s['layout'], round(s['x'])) for s in segs] == [
        (0, 99, 'crop', 96), (100, 199, 'crop', 1196)]


def test_a_wide_two_shot_is_shown_whole_in_auto_and_commits_to_a_side_in_crop():
    two = lambda i: [_face(300, 170), _face(1600, 150)]
    auto = sc.plan_reframe(_samples(100, two), [], 100, 1920, 1080, 608, mode='auto')
    assert [s['layout'] for s in auto] == ['fit']
    crop = sc.plan_reframe(_samples(100, two), [], 100, 1920, 1080, 608, mode='crop')
    assert crop[0]['layout'] == 'crop' and crop[0]['keys'] is None
    assert crop[0]['x'] == 0.0, 'the larger (left) face wins, for the whole shot'


def test_two_faces_that_fit_are_framed_together_and_a_small_background_face_is_ignored():
    close = sc.plan_reframe(_samples(100, lambda i: [_face(800, 160), _face(1050, 150)]), [], 100, 1920, 1080, 608)
    assert close[0]['layout'] == 'crop' and abs(close[0]['x'] + 304 - 922.5) <= 2      # centred on the pair
    extra = sc.plan_reframe(_samples(100, lambda i: [_face(600, 300), _face(1700, 70)]), [], 100, 1920, 1080, 608)
    assert extra[0]['layout'] == 'crop' and abs(extra[0]['x'] - (600 - 304)) <= 2


def test_no_faces_means_a_centre_crop_and_fit_mode_never_crops():
    none = sc.plan_reframe(_samples(100, lambda i: []), [], 100, 1920, 1080, 608)
    assert none[0]['layout'] == 'crop' and none[0]['x'] == (1920 - 608) / 2
    one_stray = sc.plan_reframe(_samples(100, lambda i: [_face(100)] if i == 50 else []), [], 100, 1920, 1080, 608)
    assert one_stray[0]['x'] == (1920 - 608) / 2, 'one detection in twenty frames is noise, not a subject'
    fit = sc.plan_reframe(_samples(100, lambda i: [_face(400)]), [30], 100, 1920, 1080, 608, mode='fit')
    assert [(s['a'], s['b'], s['layout']) for s in fit] == [(0, 99, 'fit')], 'identical neighbours are merged'


def test_a_moving_subject_is_followed_on_a_smooth_path():
    samples = _samples(100, lambda i: [_face(400 + 11 * i)])        # walks right across the frame
    seg = sc.plan_reframe(samples, [], 100, 1920, 1080, 608)[0]
    assert seg['layout'] == 'crop' and seg['x'] is None
    keys = seg['keys']
    assert keys[0][0] == 0 and keys[-1][0] == 99
    frames = [f for f, _ in keys]
    xs = [x for _, x in keys]
    assert frames == sorted(set(frames)) and xs == sorted(xs), 'never reverses on a steady move'
    for f, x in keys:
        assert abs((x + 304) - (400 + 11 * f)) < 60, 'the subject stays near the middle of the window'
    assert all(0 <= x <= 1920 - 608 for x in xs)


def test_a_source_no_wider_than_the_output_is_never_panned():
    segs = sc.plan_reframe(_samples(100, lambda i: [_face(300)]), [50], 100, 1080, 1920, 1080)
    assert [(s['layout'], s['x'], s['keys']) for s in segs] == [('crop', 0.0, None)]


def _eval_expr(expr, n):
    """Evaluates a crop_x_expr() result for frame n the way ffmpeg would."""
    def between(v, a, b):
        return 1.0 if a <= v <= b else 0.0
    return eval(expr, {'__builtins__': {}}, {'between': between, 'n': n})


def test_crop_expression_covers_every_frame_exactly_once():
    samples = (_samples(60, lambda i: [_face(400)]) +
               [(i, [_face(300, 170), _face(1600, 150)]) for i in range(60, 120, 5)] +
               [(i, [_face(400 + 12 * (i - 120))]) for i in range(120, 200, 5)])
    segs = sc.plan_reframe(samples, [60, 120], 200, 1920, 1080, 608)
    assert [s['layout'] for s in segs] == ['crop', 'fit', 'crop']
    expr = sc.crop_x_expr(segs)
    for n in range(200):
        active = len(re.findall(r'between\(n,(\d+),(\d+)\)', expr))
        hits = sum(1 for a, b in re.findall(r'between\(n,(\d+),(\d+)\)', expr) if int(a) <= n <= int(b))
        seg = next(s for s in segs if s['a'] <= n <= s['b'])
        assert hits == (1 if seg['layout'] == 'crop' else 0), (n, hits, active)
    assert abs(_eval_expr(expr, 30) - 96) <= 2
    assert _eval_expr(expr, 119) == 0                                   # fit shot: crop value is unused
    x_start, x_end = _eval_expr(expr, 120), _eval_expr(expr, 199)
    assert x_end > x_start + 300, 'the pan actually moves'
    path = [_eval_expr(expr, n) for n in range(120, 200)]
    assert max(abs(b - a) for a, b in zip(path, path[1:])) < 20, 'no jump between keyframes'


def test_filtergraph_shape_for_each_layout_mix():
    info = {'width': 1920, 'height': 1080, 'disp_w': 1920, 'disp_h': 1080, 'sar': 1.0, 'sd_matrix': False}
    crop = [{'a': 0, 'b': 49, 'layout': 'crop', 'x': 96.0, 'keys': None}]
    fit = [{'a': 0, 'b': 49, 'layout': 'fit', 'x': None, 'keys': None}]
    mixed = crop + [{'a': 50, 'b': 99, 'layout': 'fit', 'x': None, 'keys': None}]

    g = sc.build_filtergraph(info, crop)
    assert g.startswith('[0:v]') and g.endswith('[vout]')
    assert ',scale=1920:1080,setsar=1,crop=' in g, \
        'the picture is pinned to the size the planner measured before anything is cut out of it'
    assert "crop=608:1080:x='between(n,0,49)*96':y=0" in g and 'scale=1080:1920' in g
    assert 'boxblur' not in g and 'overlay' not in g and 'ass=' not in g

    g = sc.build_filtergraph(info, fit)
    assert 'boxblur' in g and 'crop=608' not in g and 'scale=1080:608' in g

    g = sc.build_filtergraph(info, mixed, ass_name='shsub_1.ass')
    assert "overlay=0:0:enable='between(n,50,99)'" in g
    assert 'ass=shsub_1.ass' in g and g.index('ass=') > g.index('enable='), 'captions go on after the layouts are combined'
    assert 'in_color_matrix' not in g


def test_a_failed_render_reports_the_cause_not_the_aftermath():
    """What a Windows server actually printed (ffmpeg 7+), shortened: the one
    line that says why comes first and six lines of every thread reporting
    that it stopped come after. The last 600 characters -- which is what an
    editor used to be shown -- held none of the reason."""
    stderr = (
        "[Parsed_crop_3 @ 000001b7a24b1c80] Invalid too big or non positive size for width '608' or height '1080'\n"
        "[Parsed_crop_3 @ 000001b7a24b1c80] Failed to configure input pad on Parsed_crop_3\n"
        "[fc#0 @ 000001b7a24b4f40] Error reinitializing filters!\n"
        "[fc#0 @ 000001b7a24b4f40] Task finished with error code: -22 (Invalid argument)\n"
        "[fc#0 @ 000001b7a24b4f40] Terminating thread with return code -22 (Invalid argument)\n"
        "[vost#0:0/libx264 @ 000001b7a31f8300] [enc:libx264 @ 000001b7a32baf00] Could not open encoder before EOF\n"
        "[vost#0:0/libx264 @ 000001b7a31f8300] Task finished with error code: -22 (Invalid argument)\n"
        "[vost#0:0/libx264 @ 000001b7a31f8300] Terminating thread with return code -22 (Invalid argument)\n"
        "[out#0/mp4 @ 000001b7a24fefc0] Nothing was written into output file, because at least one of its "
        "streams received no packets.\n")
    assert 'Invalid too big' not in stderr.strip()[-600:], 'the old tail really did lose it'
    msg = sc.ffmpeg_error(stderr)
    assert msg == ("[Parsed_crop_3] Invalid too big or non positive size for width '608' or height '1080' | "
                   "[Parsed_crop_3] Failed to configure input pad on Parsed_crop_3")
    # Nothing but aftermath to show: show that rather than nothing.
    assert 'Could not open encoder' in sc.ffmpeg_error('[vost#0:0/libx264 @ 0x55d0c0ffee00] Could not open encoder before EOF')
    assert sc.ffmpeg_error('') == '' and sc.ffmpeg_error(None) == ''
    long = sc.ffmpeg_error('x' * 5000)
    assert len(long) == 600 and long.endswith('\u2026')


def test_filtergraph_converts_sd_colour_and_squares_anamorphic_pixels():
    info = {'width': 720, 'height': 480, 'disp_w': 854, 'disp_h': 480, 'sar': 32 / 27, 'sd_matrix': True}
    g = sc.build_filtergraph(info, [{'a': 0, 'b': 9, 'layout': 'crop', 'x': 292.0, 'keys': None}])
    assert 'scale=854:480:in_color_matrix=bt601:out_color_matrix=bt709' in g
    assert 'crop=270:480' in g
    hd_anamorphic = {'width': 1440, 'height': 1080, 'disp_w': 1920, 'disp_h': 1080, 'sar': 4 / 3, 'sd_matrix': False}
    g = sc.build_filtergraph(hd_anamorphic, [{'a': 0, 'b': 9, 'layout': 'crop', 'x': 0.0, 'keys': None}])
    assert 'scale=1920:1080,setsar=1' in g and 'in_color_matrix' not in g


# ---- captions ----

def test_captions_are_short_cues_timed_to_the_words_they_show():
    text = 'Bakit mo ginawa sa akin ito? Wala kang karapatan! Alam ko na ang lahat, Ramon.'
    words = [{'start': 100.0 + i * 0.4, 'end': 100.0 + i * 0.4 + 0.3, 'word': w} for i, w in enumerate(text.split())]
    cues = sc.subtitle_cues(words, [], 99.0, 108.0)
    assert [c['text'] for c in cues] == ['Bakit mo ginawa sa akin', 'ito?', 'Wala kang karapatan!',
                                         'Alam ko na ang lahat,', 'Ramon.']
    assert all(len(c['text']) <= 26 for c in cues)
    assert cues[0]['start'] == 1.0, 'relative to the clip, appearing with its first word'
    for a, b in zip(cues, cues[1:]):
        assert a['end'] <= b['start'] + 1e-6, 'cues never overlap'
    assert all(0 <= c['start'] < c['end'] <= 9.0 for c in cues)


def test_captions_only_include_words_inside_the_clip_and_break_on_pauses():
    words = [{'start': s, 'end': s + 0.3, 'word': w} for s, w in
             [(1.0, 'before'), (5.0, 'one'), (5.4, 'two'), (8.0, 'three'), (20.0, 'after')]]
    cues = sc.subtitle_cues(words, [], 4.5, 9.0)
    assert [c['text'] for c in cues] == ['one two', 'three']
    assert cues[1]['end'] - cues[1]['start'] >= 0.59, 'a lone 0.3 s word is held long enough to read'


def test_captions_fall_back_to_line_timings_without_word_timestamps():
    segs = _segs([(10.0, 16.0, 'Hindi mo ako maloloko, alam ko na ang lahat ng ginawa mo.')])
    cues = sc.subtitle_cues([], segs, 9.0, 20.0)
    assert len(cues) >= 2 and ' '.join(c['text'] for c in cues) == segs[0]['text']
    assert abs(cues[0]['start'] - 1.0) < 1e-6 and abs(cues[-1]['end'] - 7.0) < 1e-6
    assert sc.subtitle_cues([], [], 0, 10) == []


def test_caption_files(tmp_path):
    cues = [{'start': 0.5, 'end': 2.25, 'text': 'Sabi ko {\\an8}sa iyo'},
            {'start': 61.0, 'end': 3723.456, 'text': 'Huli na'}]
    ass, srt = tmp_path / 'c.ass', tmp_path / 'c.srt'
    sc.write_ass(cues, str(ass), size='l', font='Arial, Bold')
    body = ass.read_text(encoding='utf-8')
    assert 'PlayResX: 1080' in body and 'PlayResY: 1920' in body
    assert 'Style: Cap,Arial  Bold,78,' in body, 'a comma in the font name would shift every style field'
    assert 'Dialogue: 0,0:00:00.50,0:00:02.25,Cap,,0,0,0,,Sabi ko (/an8)sa iyo' in body, \
        'override tags in transcript text must not be executed'
    assert 'Dialogue: 0,0:01:01.00,1:02:03.46,Cap' in body
    sc.write_srt(cues, str(srt))
    assert srt.read_text(encoding='utf-8').startswith('1\n00:00:00,500 --> 00:00:02,250\nSabi ko')
    assert '2\n00:01:01,000 --> 01:02:03,456\nHuli na' in srt.read_text(encoding='utf-8')


# ---- odds and ends ----

def test_slugify_is_filesystem_safe():
    assert sc.slugify('Anak mo ang batang iyon!') == 'Anak_mo_ang_batang_iyon'
    assert sc.slugify('../../etc/passwd') == 'etc_passwd'
    assert sc.slugify('CON: <a|b>? "x".') == 'CON_a_b_x'
    assert sc.slugify('日本語') == '' and sc.slugify(None) == ''
    assert len(sc.slugify('word ' * 40, 20)) <= 20


def test_timestamp_formatting():
    assert sc.fmt_ts(0) == '00:00' and sc.fmt_ts(754.9) == '12:34' and sc.fmt_ts(3725) == '1:02:05'


def test_render_command_seeks_to_the_exact_frame_including_the_video_start_offset():
    info = {'width': 1920, 'height': 1080, 'disp_w': 1920, 'disp_h': 1080, 'sar': 1.0, 'sd_matrix': False,
            'fps': 25.0, 'audio_index': 1, 'tag_709': True, 'v_offset': 0.021}
    segs = [{'a': 0, 'b': 249, 'layout': 'crop', 'x': 656.0, 'keys': None}]
    cmd = sc.build_render_cmd('ffmpeg', '/in.mp4', '/out.mp4', 1000, 250, info, segs)
    ss = float(cmd[cmd.index('-ss') + 1])
    # A quarter-frame before frame 1000's real timestamp: after frame 999, before 1000.
    assert 0.021 + 999 / 25.0 < ss < 0.021 + 1000 / 25.0
    assert abs(ss - (0.021 + 999.75 / 25.0)) < 1e-6
    assert cmd.index('-ss') < cmd.index('-i'), 'input seeking, or every short would decode from the start of the episode'
    assert cmd[cmd.index('-frames:v') + 1] == '250' and float(cmd[cmd.index('-t') + 1]) == 10.0
    assert cmd[cmd.index('-map', cmd.index('[vout]')) + 1] == '0:a:1'
    assert 'loudnorm=I=-14.0:TP=-1.5' in cmd[cmd.index('-af') + 1]
    assert cmd[cmd.index('-colorspace') + 1] == 'bt709'

    info.update(audio_index=None, tag_709=False)
    cmd = sc.build_render_cmd('ffmpeg', '/in.mp4', '/out.mp4', 0, 250, info, segs)
    assert '-an' in cmd and '-af' not in cmd and '-colorspace' not in cmd
    assert float(cmd[cmd.index('-ss') + 1]) >= 0.0


# ---- found in review ----

def test_the_next_lines_first_word_is_not_adopted_when_speech_runs_on():
    """Back-to-back dialogue: line 21 starts 50 ms after line 20 ends. A
    looser 'belongs to this range' rule took that first word as the end of
    the moment, so it was heard and captioned at the end of the short."""
    words, segs, t = [], [], 0.0
    for i in range(30):
        s0 = t
        for k in range(5):
            words.append({'start': round(t, 3), 'end': round(t + 0.3, 3), 'word': f'w{i}_{k}'})
            t += 0.35
        segs.append({'start': s0, 'end': round(t - 0.05, 3), 'text': ' '.join(f'w{i}_{k}' for k in range(5))})
    t0, t1 = sc.speech_bounds(segs[10]['start'], segs[20]['end'], words)
    assert abs(t0 - segs[10]['start']) < 1e-6 and abs(t1 - segs[20]['end']) < 1e-6
    out = sc.build_candidates([{'start_id': 10, 'end_id': 20, 'score': 8, 'title': 'T'}], segs, words, [], [],
                              t, min_dur=5, max_dur=60)
    c = out[0]
    assert segs[20]['end'] <= c['end'] < segs[21]['start'] + 0.3, 'ends before the next line can be heard'
    cues = sc.subtitle_cues(words, segs, c['start'], c['end'])
    shown = ' '.join(x['text'] for x in cues).split()
    assert shown[0] == 'w10_0' and shown[-1] == 'w20_4', (shown[0], shown[-1])


def test_resplitting_a_long_line_does_not_borrow_the_previous_lines_last_word():
    words = [{'start': 9.8, 'end': 10.1, 'word': 'previous.'}]
    t = 10.15
    for i in range(40):
        words.append({'start': round(t, 2), 'end': round(t + 0.4, 2), 'word': f'w{i}' + ('.' if i % 8 == 7 else '')})
        t += 0.5
    segs = [{'start': 5.0, 'end': 10.1, 'text': 'something previous.'},
            {'start': 10.15, 'end': round(t - 0.1, 2), 'text': ' '.join(f'w{i}' for i in range(40))}]
    _, out = sc.normalize_transcript(words, segs)
    assert len(out) > 2 and out[1]['text'].startswith('w0 ') and out[1]['start'] == 10.15


def test_lead_in_and_tail_never_push_a_candidate_past_the_maximum():
    """A beat already at the length limit, with a cut just before it and one
    just after: taking both would overshoot by seconds, and at the tab's own
    300 s ceiling the render request would then be refused outright."""
    segs = [{'start': 5.0 + 3.0 * i, 'end': 8.0 + 3.0 * i, 'text': f'line {i}'} for i in range(100)]   # contiguous
    cuts = [0.0, 4.0, 307.0, 400.0]
    out = sc.build_candidates([{'start_id': 0, 'end_id': 99, 'score': 9, 'title': 'Long'}], segs, [], cuts, [],
                              400.0, min_dur=30, max_dur=300, fps=25.0)
    assert len(out) == 1 and out[0]['duration'] <= 300.0 + 1e-6, out[0]['duration']
    assert out[0]['start'] <= 5.0 and out[0]['end'] >= 305.0, 'the speech itself is still whole'

    # With room to spare, the same cuts ARE used.
    out = sc.build_candidates([{'start_id': 0, 'end_id': 9, 'score': 9, 'title': 'Short'}], segs[:10], [],
                              [0.0, 4.0, 36.0, 400.0], [], 400.0, min_dur=10, max_dur=300, fps=25.0)
    assert (out[0]['start'], out[0]['end']) == (4.0, 36.0)

    # One run-on line longer than the limit is cut to it by time.
    out = sc.build_candidates([{'start_id': 0, 'end_id': 0, 'score': 5, 'title': 'Run-on'}],
                              [{'start': 10.0, 'end': 80.0, 'text': 'x ' * 100}], [], [], [], 100.0,
                              min_dur=10, max_dur=30, fps=25.0)
    assert out[0]['duration'] <= 30.0 + 1e-6 and 'trimmed' in out[0]['flags']


# ---- following the speaker ----

def _face_frame(shift=(0, 0), mouth_open=0, size=400, box=(100, 100, 200, 200), seed=7):
    """A textured stand-in for a face inside a frame: fixed 'features' in the
    upper half of `box`, a mouth that can be opened, the whole lot movable."""
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur(rng.integers(40, 215, (size, size)).astype(np.float32), (0, 0), 6)
    img = (img - img.min()) / (img.max() - img.min()) * 200 + 25
    x, y, w, h = box
    cv2.circle(img, (x + w // 3, y + h // 3), 14, 30, -1)                 # eyes
    cv2.circle(img, (x + 2 * w // 3, y + h // 3), 14, 30, -1)
    cv2.ellipse(img, (x + w // 2, y + int(0.8 * h)), (36, 6 + mouth_open), 0, 0, 360, 20, -1)
    m = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
    return cv2.warpAffine(img, m, (size, size), borderMode=cv2.BORDER_REFLECT).astype(np.uint8)


def test_mouth_activity_sees_the_mouth_and_not_the_head():
    box = (100, 100, 200, 200)
    still = _face_frame()
    assert sc.mouth_activity(still, still, box) == 0.0
    nod = sc.mouth_activity(still, _face_frame(shift=(4, 6)), box)
    talk = sc.mouth_activity(still, _face_frame(mouth_open=14), box)
    both = sc.mouth_activity(still, _face_frame(shift=(4, 6), mouth_open=14), box)
    assert nod < sc.SPEAKER_FLOOR, 'a head that only moved is not a mouth moving'
    assert talk > 3 * sc.SPEAKER_FLOOR and both > 3 * sc.SPEAKER_FLOOR
    assert both > 5 * max(nod, 1e-4), 'the mouth still reads through a moving head'
    brighter = np.clip(_face_frame().astype(np.int16) + 30, 0, 255).astype(np.uint8)
    assert sc.mouth_activity(still, brighter, box) < sc.SPEAKER_FLOOR, 'a flash is not movement'


def test_mouth_activity_declines_what_it_cannot_measure():
    still = _face_frame()
    assert sc.mouth_activity(still, still, (100, 100, 20, 20)) is None, 'too small'
    flat = np.full((400, 400), 128, np.uint8)
    assert sc.mouth_activity(flat, flat, (100, 100, 200, 200)) is None, 'nothing to lock on to'
    assert sc.mouth_activity(still, still, (390, 390, 200, 200)) is None, 'almost wholly out of frame'


def _words(spans, step=0.3, length=0.25):
    out = []
    for a, b in spans:
        t = a
        while t < b - 1e-6:
            out.append((round(t, 3), round(t + length, 3)))
            t += step
    return out


def _acts(grid, fps, *spans_per_person, level=0.1):
    return [[level if any(a <= i / fps < b for a, b in spans) else 0.0 for i in grid] for spans in spans_per_person]


def test_speaker_turns_cut_to_whoever_is_talking_just_ahead_of_their_line():
    grid, fps = list(range(0, 200, 5)), 25.0
    speech = _words([(0, 4), (4.6, 8)])
    runs = sc.speaker_turns(grid, _acts(grid, fps, [(0, 4)], [(4.6, 8)]), 0, 199, fps, speech)
    assert runs == [(0, 110, 0), (111, 199, 1)], 'the change lands 0.15 s before the word at 4.6 s (frame 115)'


def test_a_mouth_moving_in_silence_does_not_take_the_frame():
    grid, fps = list(range(0, 200, 5)), 25.0
    speech = _words([(0, 4), (5.5, 8)])
    # Person 1 laughs through the pause; person 0 has every spoken word.
    acts = _acts(grid, fps, [(0, 4), (5.5, 8)], [(4.2, 5.3)])
    assert sc.speaker_turns(grid, acts, 0, 199, fps, speech) == [(0, 199, 0)]


def test_a_brief_interjection_does_not_buy_two_cuts():
    grid, fps = list(range(0, 250, 5)), 25.0
    speech = _words([(0, 10)])
    acts = _acts(grid, fps, [(0, 4.6), (5.2, 10)], [(4.6, 5.2)])
    assert sc.speaker_turns(grid, acts, 0, 249, fps, speech) == [(0, 249, 0)]


def test_speaker_turns_decline_when_the_evidence_is_not_there():
    grid, fps = list(range(0, 200, 5)), 25.0
    speech = _words([(0, 8)])
    one = _acts(grid, fps, [(0, 8)], [])
    assert sc.speaker_turns(grid, one, 0, 199, fps, speech) == [(0, 199, 0)]
    assert sc.speaker_turns(grid, one, 0, 199, fps, []) is None, 'no transcript, nothing to time mouths against'
    assert sc.speaker_turns(grid, _acts(grid, fps, [(0, 8)], [(0, 8)]), 0, 199, fps, speech) is None, 'two mouths alike'
    assert sc.speaker_turns(grid, _acts(grid, fps, [(0, 8)], [], level=0.005), 0, 199, fps, speech) is None, 'too faint'
    unknown = [[None] * len(grid), [None] * len(grid)]
    assert sc.speaker_turns(grid, unknown, 0, 199, fps, speech) is None, 'mouths never measured'
    assert sc.speaker_turns(grid, one, 0, 199, fps, _words([(20, 30)])) is None, 'nobody speaks during this shot'


def _talking(n_frames, left, right, fps=25.0, step=5, lx=500, rx=1500):
    """A wide two-shot: a face at lx and one at rx, each with mouth activity
    during its own spans (seconds)."""
    def act(i, spans):
        return 0.1 if any(a <= i / fps < b for a, b in spans) else 0.0
    return [(i, [_face(lx) + (act(i, left),), _face(rx) + (act(i, right),)]) for i in range(0, n_frames, step)]


def test_follow_the_speaker_cuts_a_wide_two_shot_between_the_two_people():
    samples = _talking(200, [(0, 4)], [(4.6, 8)])
    speech = _words([(0, 4), (4.6, 8)])
    segs = sc.plan_reframe(samples, [], 200, 1920, 1080, 608, speaker=True, speech=speech)
    assert [(s['a'], s['b'], s['layout'], round(s['x'])) for s in segs] == [
        (0, 110, 'crop', 500 - 304), (111, 199, 'crop', 1500 - 304)]
    assert all(s['speaker'] and s['keys'] is None for s in segs)
    expr = sc.crop_x_expr(segs)
    for n in range(200):
        assert sum(1 for a, b in re.findall(r'between\(n,(\d+),(\d+)\)', expr) if int(a) <= n <= int(b)) == 1, n
    assert _eval_expr(expr, 110) == 196 and _eval_expr(expr, 111) == 1196
    # The same option in "always crop" mode: the speaker, not the side with more face on it.
    crop = sc.plan_reframe(samples, [], 200, 1920, 1080, 608, mode='crop', speaker=True, speech=speech)
    assert [(s['a'], round(s['x'])) for s in crop] == [(0, 196), (111, 1196)]


def test_follow_the_speaker_changes_nothing_when_it_is_off_or_cannot_tell():
    samples = _talking(200, [(0, 4)], [(4.6, 8)])
    speech = _words([(0, 4), (4.6, 8)])

    def plan(s, **kw):
        return [(x['a'], x['b'], x['layout'], x['x'], x['keys']) for x in sc.plan_reframe(s, [], 200, 1920, 1080, 608, **kw)]
    whole = [(0, 199, 'fit', None, None)]
    assert plan(samples) == whole, 'off: mouth data in the samples is ignored'
    assert plan(samples, speaker=True) == whole and plan(samples, speaker=True, speech=[]) == whole, 'no transcript'
    alike = _talking(200, [(0, 8)], [(0, 8)])
    assert plan(alike, speaker=True, speech=speech) == whole, 'both mouths moving: show both people'
    unmeasured = [(i, [f[:4] for f in fs]) for i, fs in samples]
    assert plan(unmeasured, speaker=True, speech=speech) == whole, 'no mouth data'
    for mode in ('auto', 'crop', 'fit'):
        assert plan(alike, mode=mode, speaker=True, speech=speech) == plan(alike, mode=mode)


def test_follow_the_speaker_leaves_every_other_kind_of_shot_alone():
    speech = _words([(0, 16)])
    # Two people close enough to share the frame; one person; nobody.
    together = _talking(200, [(0, 4)], [(4, 8)], lx=800, rx=1050)
    alone = [(i, [_face(500) + (0.1,)]) for i in range(0, 200, 5)]
    empty = _samples(200, lambda i: [])
    for samples in (together, alone, empty):
        on = sc.plan_reframe(samples, [], 200, 1920, 1080, 608, speaker=True, speech=speech)
        off = sc.plan_reframe(samples, [], 200, 1920, 1080, 608)
        assert on == off and not any(s.get('speaker') for s in on)
    # And it works shot by shot: a wide two-shot followed by a single.
    mixed = _talking(200, [(0, 4)], [(4.6, 8)]) + [(i, [_face(900)]) for i in range(200, 300, 5)]
    segs = sc.plan_reframe(mixed, [200], 300, 1920, 1080, 608, speaker=True,
                           speech=_words([(0, 4), (4.6, 8), (8.2, 12)]))
    assert [(s['a'], s['b'], bool(s.get('speaker'))) for s in segs] == [
        (0, 110, True), (111, 199, True), (200, 299, False)]

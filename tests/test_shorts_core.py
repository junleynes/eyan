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
import pytest
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


def test_story_prompt_states_what_to_avoid_as_a_rule_and_what_to_feature_as_a_preference():
    _, segs = _dialogue(20)
    plain = sc.build_story_prompt(segs, 0, 4, [], 30, 90)
    assert 'does NOT want' not in plain and 'especially wants' not in plain
    p = sc.build_story_prompt(segs, 0, 4, [], 30, 90, focus='the wedding', avoid='the  hospital\nscenes ' + 'x' * 400)
    assert 'The editor especially wants moments about: the wedding' in p
    assert 'never pick a weak one just because it matches' in p
    assert 'The editor does NOT want: the hospital scenes xxx' in p, 'whitespace collapsed'
    assert 'Leave out every moment that is mainly about this or shows it, however strong it is.' in p
    assert 'x' * 301 not in p, 'capped: the note must not crowd the transcript out of the context window'
    assert p.index('especially wants') < p.index('does NOT want') < p.index('EPISODE STRETCH'), \
        'both notes sit with the instructions, ahead of the transcript'
    only = sc.build_story_prompt(segs, 0, 4, [], 30, 90, avoid='spoilers')
    assert 'does NOT want: spoilers' in only and 'especially wants' not in only


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


def test_a_close_up_wider_than_the_window_is_cropped_not_shown_whole():
    """Real, reported case: a solo close-up rendered small over a blurred
    background. The Haar cascades had boxed the face 606 px wide on a
    1920-wide picture; the window is 608 and "fits" meant 90% of that, so one
    person failed a test meant for two people standing far apart."""
    big = _samples(100, lambda i: [_face(975, 606)])
    for mode in ('auto', 'split', 'crop'):
        segs = sc.plan_reframe(big, [], 100, 1920, 1080, 608, mode=mode)
        assert [(s['a'], s['b'], s['layout'], s['keys']) for s in segs] == [(0, 99, 'crop', None)], mode
        assert abs(segs[0]['x'] - (975 - 304)) <= 2, 'centred on the face'
    # Even one far larger than the window: there is still nobody else to lose.
    huge = sc.plan_reframe(_samples(100, lambda i: [_face(960, 900)]), [], 100, 1920, 1080, 608)
    assert [s['layout'] for s in huge] == ['crop']
    # A small face in the background does not turn it into a two-shot...
    extra = sc.plan_reframe(_samples(100, lambda i: [_face(975, 606), _face(1800, 90)]), [], 100, 1920, 1080, 608)
    assert [s['layout'] for s in extra] == ['crop']
    # ...but two real faces too far apart still are one.
    two = sc.plan_reframe(_samples(100, lambda i: [_face(400, 300), _face(1500, 300)]), [], 100, 1920, 1080, 608)
    assert [s['layout'] for s in two] == ['fit']


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


def test_the_frame_rate_is_stated_so_newer_ffmpeg_does_not_assume_25():
    """The graph resets the timestamps, which newer ffmpeg takes to mean
    the stream has no fixed rate; left to guess, it assumed 25 and dropped
    one frame in six of a 29.97 source (84 frames written for 100)."""
    assert [sc.frame_rate_arg(f) for f in (25.0, 29.97002997, 23.976023976, 59.94005994, 30.0, 50.0, 24.0)] == [
        '25', '30000/1001', '24000/1001', '60000/1001', '30', '50', '24']
    assert sc.frame_rate_arg(29.97) == '2997/100' and sc.frame_rate_arg(12.5) == '25/2', 'an odd rate, as measured'
    info = {'width': 1920, 'height': 1080, 'disp_w': 1920, 'disp_h': 1080, 'sar': 1.0, 'sd_matrix': False,
            'fps': 30000 / 1001.0, 'audio_index': 0, 'tag_709': True, 'v_offset': 0.0}
    segs = [{'a': 0, 'b': 299, 'layout': 'crop', 'x': 656.0, 'keys': None}]
    cmd = sc.build_render_cmd('ffmpeg', '/in.mp4', '/out.mp4', 0, 300, info, segs)
    assert cmd[cmd.index('-r') + 1] == '30000/1001' and cmd.index('-r') > cmd.index('-i'), 'an output option'


def test_the_cliffhanger_ending_is_counted_in_frames():
    """(loop, hold, black): a 2-second hold made from at most the last 0.16 s
    of the clip, then 0.4 s of black. The frames the hold is made from are
    taken OFF the end of the clip, so the short grows by hold + black - loop."""
    assert sc.cliffhanger_plan(250, 25.0) == (4, 50, 10)
    assert sc.cliffhanger_plan(250, 30000 / 1001.0) == (5, 60, 12)
    assert sc.cliffhanger_plan(250, 60000 / 1001.0) == (10, 120, 24)
    assert sc.cliffhanger_extra(250, 25.0) == 56 and sc.cliffhanger_extra(250, 30000 / 1001.0) == 67
    # No further back than the last cut, or than the picture is still: one frame is a true freeze.
    assert sc.cliffhanger_plan(250, 25.0, room=2) == (2, 50, 10) and sc.cliffhanger_plan(250, 25.0, room=1) == (1, 50, 10)
    assert sc.cliffhanger_plan(250, 25.0, room=400) == (4, 50, 10), 'room to spare changes nothing'
    assert sc.cliffhanger_extra(250, 25.0, room=1) == 59
    assert sc.cliffhanger_plan(3, 25.0) == (2, 50, 10), 'a clip always keeps a frame of its own'
    assert 0.02 <= sc.CLIFFHANGER['zoom'] <= 0.04 and abs(sc.CLIFFHANGER['hold'] - 2.0) < 0.26, 'as asked for'


def test_the_cliffhanger_ending_is_built_in_the_right_order():
    g = sc.cliffhanger_graph(250, 25.0, None, ass_name='c.ass')
    assert g == (
        "split=2[em][et];"
        "[em]trim=end_frame=246,setpts=PTS-STARTPTS,ass=c.ass,format=yuv420p,setsar=1[ea];"
        "[et]trim=start_frame=246:end_frame=250,setpts=PTS-STARTPTS,split=2[ef][eq];"
        "[eq]reverse[er];[ef][er]concat=n=2:v=1:a=0,"
        "setpts=N*6.250000/(25)/TB,"
        "framerate=fps=25:interp_start=0:interp_end=255:scene=100,"
        "tpad=stop_mode=clone:stop=60,trim=end_frame=60,setpts=PTS-STARTPTS,"
        "scale=2160:3840:flags=bicubic,"
        "zoompan=z='1+0.03*on/49':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s=1080x1920:fps=25,"
        f"{sc.CLIFFHANGER_GRADE},"
        "drawbox=x=0:y=0:w=iw:h=ih:color=black:t=fill:enable='gte(n,50)',format=yuv420p,setsar=1[eb];"
        "[ea][eb]concat=n=2:v=1:a=0,")
    # The look: more contrast, less colour, a split tone, darker corners -- and nothing else.
    assert [f.split('=')[0] for f in sc.CLIFFHANGER_GRADE.split(',')] == ['eq', 'colorbalance', 'vignette']
    # Captions go on the action only: before the join, never on the hold's branch.
    assert g.count('ass=') == 1 and g.index('ass=c.ass') < g.index('[ea];')
    assert 'ass=' not in sc.cliffhanger_graph(250, 25.0, None)
    # The push-in and the grade are on the hold's branch only; black is painted on last.
    hold = g[g.index('[et]'):g.index('[eb];')]
    assert hold.index('framerate=') < hold.index('tpad=') < hold.index('zoompan=') < hold.index('eq=') \
        < hold.index('vignette=') < hold.index('drawbox=')
    assert 'minterpolate' not in g, 'frames are blended, not invented: motion estimation bends faces'
    # A true freeze: one frame, held.
    one = sc.cliffhanger_graph(250, 30000 / 1001.0, 1)
    assert 'trim=end_frame=249,' in one and 'trim=start_frame=249:end_frame=250,' in one
    assert 'setpts=N*30.000000/(30000/1001)/TB' in one and "enable='gte(n,60)'" in one and 'stop=72,' in one

    info = {'width': 1920, 'height': 1080, 'disp_w': 1920, 'disp_h': 1080, 'sar': 1.0, 'sd_matrix': False,
            'fps': 25.0, 'audio_index': 1, 'tag_709': True, 'v_offset': 0.0}
    segs = [{'a': 0, 'b': 249, 'layout': 'crop', 'x': 656.0, 'keys': None}]
    plain = sc.build_filtergraph(info, segs, ass_name='c.ass')
    full = sc.build_filtergraph(info, segs, ass_name='c.ass', ending=(250, None))
    assert plain.endswith('[v1]setpts=PTS-STARTPTS,ass=c.ass,format=yuv420p,setsar=1[vout]'), 'unchanged without it'
    assert full.endswith(f'[v1]{g}setpts=PTS-STARTPTS,format=yuv420p,setsar=1[vout]')
    assert full[:full.index('[v1]split=2')] == plain[:plain.index('[v1]setpts')], 'the reframing itself is the same'
    assert sc.build_filtergraph(info, segs, ending=(250, 1)).count('trim=end_frame=249,') == 1


def test_the_cliffhanger_ending_stops_the_sound_with_the_picture():
    info = {'width': 1920, 'height': 1080, 'disp_w': 1920, 'disp_h': 1080, 'sar': 1.0, 'sd_matrix': False,
            'fps': 25.0, 'audio_index': 1, 'tag_709': True, 'v_offset': 0.0}
    segs = [{'a': 0, 'b': 249, 'layout': 'crop', 'x': 656.0, 'keys': None}]
    cmd = sc.build_render_cmd('ffmpeg', '/in.mp4', '/out.mp4', 0, 250, info, segs, ending=True)
    assert cmd[cmd.index('-frames:v') + 1] == '306' and abs(float(cmd[cmd.index('-t') + 1]) - 12.24) < 1e-6
    af = cmd[cmd.index('-af') + 1]
    assert af == ('afade=t=in:st=0:d=0.04,atrim=end=10.040,afade=t=out:st=9.840:d=0.200:curve=cub,'
                  'loudnorm=I=-14.0:TP=-1.5:LRA=11,apad=whole_dur=12.240')
    assert af.index('loudnorm') < af.index('apad'), 'silence added after levelling stays silence'
    # With one frame to hold, the action runs a few frames longer, and so does the sound.
    cmd = sc.build_render_cmd('ffmpeg', '/in.mp4', '/out.mp4', 0, 250, info, segs, ending=True, ending_room=1)
    assert cmd[cmd.index('-frames:v') + 1] == '309' and 'afade=t=out:st=9.960:d=0.200' in cmd[cmd.index('-af') + 1]
    assert 'trim=end_frame=249,' in cmd[cmd.index('-filter_complex') + 1]
    # Audio taken from chosen channels gets the same ending, inside the graph.
    cmd = sc.build_render_cmd('ffmpeg', '/in.mp4', '/out.mp4', 0, 250, dict(info, audio_take=[[2, 0], [3, 0]]), segs,
                              ending=True)
    graph = cmd[cmd.index('-filter_complex') + 1]
    assert 'curve=cub' in graph and 'apad=whole_dur=12.240[aout]' in graph and '-af' not in cmd
    # Without it, nothing about the command changes.
    cmd = sc.build_render_cmd('ffmpeg', '/in.mp4', '/out.mp4', 0, 250, info, segs)
    assert cmd[cmd.index('-frames:v') + 1] == '250' and 'apad' not in cmd[cmd.index('-af') + 1]
    assert 'afade=t=out:st=9.880:d=0.12' in cmd[cmd.index('-af') + 1]
    assert 'zoompan' not in cmd[cmd.index('-filter_complex') + 1]


def _clip(path, frames, fps=25.0):
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), fps, (w, h))
    assert out.isOpened()
    for f in frames:
        out.write(f)
    out.release()
    return str(path)


def test_the_hold_is_made_only_from_frames_in_which_nothing_is_really_moving(tmp_path):
    """Looping the last few frames slowly back and forth is a held breath
    when they are nearly the same picture, and a slow sway when they are
    not. So how far back the loop reaches is measured, per short."""
    x = np.arange(640)
    scene = np.repeat((110 + 70 * np.sin(x / 23.0) + 30 * np.sin(x / 5.0))[None, :, None], 360, axis=0)
    scene = np.repeat(scene, 3, axis=2).astype(np.uint8)                          # a room with things in it

    def frame(shift=0, breath=0.0, hand=None):
        f = np.roll(scene, shift, axis=1).copy()
        cv2.circle(f, (320, 180), 70, (150, 170, 210), -1)                       # a face that stays put
        cv2.ellipse(f, (320, 300), (110, int(round(40 + breath))), 0, 0, 360, (90, 60, 50), -1)    # shoulders that breathe
        if hand is not None:
            cv2.rectangle(f, (560, 300 - hand), (610, 350 - hand), (200, 215, 235), -1)
        return f

    calm = _clip(tmp_path / 'calm.avi', [frame(breath=1.5 * np.sin(i / 6.0)) for i in range(40)])
    assert sc.still_frames(calm, 39, 25.0) == 4, 'a breath is not movement: the whole 0.16 s'
    assert sc.still_frames(calm, 39, 25.0, room=2) == 2, 'never further back than the last cut'
    assert sc.still_frames(calm, 39, 25.0, room=1) == 1 and sc.still_frames(calm, 0, 25.0) == 1
    assert sc.still_frames(calm, 39, 60000 / 1001.0) == 10, 'the same 0.16 s at 59.94'

    busy = _clip(tmp_path / 'busy.avi', [frame(shift=12 * i) for i in range(40)])
    assert sc.still_frames(busy, 39, 25.0) == 1, 'a moving camera: one frame, a true freeze'

    # One hand coming up at the edge of an otherwise motionless frame: under
    # 1% of the picture, and the very thing that would be seen swaying.
    hand = _clip(tmp_path / 'hand.avi', [frame(hand=6 * i) for i in range(40)])
    assert sc.still_frames(hand, 39, 25.0) == 1

    # Movement that stops just before the end: only the frames since it stopped.
    settle = _clip(tmp_path / 'settle.avi', [frame(hand=6 * min(i, 37)) for i in range(40)])
    assert sc.still_frames(settle, 39, 25.0) == 3
    assert sc.still_frames(str(tmp_path / 'missing.avi'), 39, 25.0) == 1, 'unreadable is not an error here'


# ---- line timings moved onto the sound; people found without their faces; the editor's framing ----

def _bursts(spans, seconds=14.0, rate=16000, bed=-52.0, seed=2):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * rate)) / rate
    x = rng.normal(0, 10 ** (bed / 20.0), len(t))
    for a, b in spans:
        m = (t >= a) & (t < b)
        x[m] += 0.12 * np.sin(2 * np.pi * 170 * t[m]) * (0.6 + 0.4 * np.sin(2 * np.pi * 6 * t[m]))
    return sc.audio_envelope(x, rate)


def test_lines_timed_to_the_whole_second_are_moved_onto_their_sound():
    """Reported, from the caption editor: every line starting and ending on
    a whole second (30.29, 32.29, 33.29...). The service gives times for
    lines only, rounded; captions and cut points were up to a second out."""
    true = [(1.3, 2.6), (3.1, 4.9), (5.4, 6.1), (8.7, 10.2), (10.5, 12.8)]
    db = _bursts(true)
    rounded = [{'start': float(round(a + 0.29) - 0.29), 'end': float(round(b + 0.29) - 0.29), 'text': f'line {k}'}
               for k, (a, b) in enumerate(true)]
    rounded[2]['end'] = 6.71                                 # (the rounding made this one zero-length)
    rounded[3]['start'] = 6.71                               # and this one took in two seconds of silence
    out, moved = sc.align_segments(rounded, db)
    assert moved == 5 and [s['text'] for s in out] == [s['text'] for s in rounded]
    for (a, b), s in zip(true, out):
        assert abs(s['start'] - a) < 0.08 and abs(s['end'] - b) < 0.08, (a, b, s)
    assert all(out[k]['start'] >= out[k - 1]['end'] for k in range(1, len(out))), 'never over each other'
    # Over a wall of sound (no quiet to find), or with nothing to go on, a line keeps its times.
    flat = sc.audio_envelope(np.full(16000 * 14, 0.1) * np.sin(np.arange(16000 * 14) / 7.0), 16000)
    assert sc.align_segments(rounded, flat) == (rounded, 0)
    assert sc.align_segments(rounded, None) == (rounded, 0) and sc.align_segments([], db) == ([], 0)
    # Already right: left alone.
    exact = [{'start': a - 0.03, 'end': b + 0.03, 'text': 'x'} for a, b in true]
    again, moved = sc.align_segments(exact, db)
    assert all(abs(p['start'] - q['start']) < 0.05 and abs(p['end'] - q['end']) < 0.05 for p, q in zip(exact, again))


def test_a_shot_with_no_face_follows_the_figure_the_body_detector_agrees_on():
    """Reported: a boy walking away from the camera, centre-cropped wherever
    he was. No face to find; his head and shoulders can be."""
    samples = [(i, []) for i in range(0, 200, 5)]

    def figure(cx, h=400, w=380):
        return (float(cx), 500.0, float(w), float(h))
    seg = sc.plan_reframe(samples, [], 200, 1920, 1080, 608,
                          bodies=[(i, [figure(1300)] if (i // 5) % 10 < 7 else []) for i in range(0, 200, 5)])
    assert [(s['layout'], s['keys']) for s in seg] == [('crop', None)] and abs(seg[0]['x'] - (1300 - 304)) <= 2
    walk = sc.plan_reframe(samples, [], 200, 1920, 1080, 608, bodies=[(i, [figure(700 + 4 * i)]) for i in range(0, 200, 5)])
    xs = [x for _, x in walk[0]['keys']]
    assert xs == sorted(xs) and xs[-1] > xs[0] + 500, 'followed as he walks'
    centre = (1920 - 608) / 2.0
    for name, bodies in (('seen in 20% of frames', [(i, [figure(1300)] if (i // 5) % 5 == 0 else []) for i in range(0, 200, 5)]),
                         ('too small: someone in the background', [(i, [figure(1300, 150, 140)]) for i in range(0, 200, 5)]),
                         ('all over the place: noise', [(i, [figure(200 + (i * 37) % 1500)]) for i in range(0, 200, 5)]),
                         ('none', None)):
        assert sc.plan_reframe(samples, [], 200, 1920, 1080, 608, bodies=bodies)[0]['x'] == centre, name
    # A shot WITH faces is framed on them; what the body detector says is not asked.
    faces = [(i, [(500.0, 400.0, 160.0, 160.0)]) for i in range(0, 200, 5)]
    assert abs(sc.plan_reframe(faces, [], 200, 1920, 1080, 608, bodies=[(i, [figure(1300)]) for i in range(0, 200, 5)])[0]['x'] - 196) <= 2


def test_head_and_shoulders_are_looked_for_only_where_no_face_was_found(tmp_path):
    frames = [np.full((180, 320, 3), 90, np.uint8) for _ in range(30)]
    clip = _clip(tmp_path / 'b.avi', frames)

    class Detector:
        def detect(self, frame):
            self.n = getattr(self, 'n', 0) + 1
            return [(100, 50, 40, 40, 1.0)] if self.n % 2 else []

        def detect_bodies(self, frame):
            return [(150, 60, 80, 100)]
    bodies = []
    faces = sc.sample_faces(clip, 0, 30, 25.0, Detector(), sar=1.0, bodies=bodies)
    assert [i for i, f in faces] == [0, 5, 10, 15, 20, 25]
    assert [i for i, _ in bodies] == [5, 15, 25], 'only the frames with no face'
    assert bodies[0][1] == [(190.0, 110.0, 80.0, 100.0)], 'centres, like faces'
    assert sc.sample_faces(clip, 0, 30, 25.0, Detector(), sar=1.0) == faces, 'not asked: nothing changes'
    # The real one runs on a plain frame without complaint and finds nothing there.
    assert sc.FaceDetector().detect_bodies(np.full((360, 640, 3), 90, np.uint8)) == []


def test_the_editor_can_reframe_a_shot_or_show_it_whole():
    segs = [{'a': 0, 'b': 49, 'layout': 'crop', 'x': 100.0, 'keys': None},
            {'a': 50, 'b': 69, 'layout': 'crop', 'x': 300.0, 'keys': None, 'speaker': True},
            {'a': 70, 'b': 99, 'layout': 'crop', 'x': 900.0, 'keys': None, 'speaker': True},
            {'a': 100, 'b': 149, 'layout': 'fit', 'x': None, 'keys': None}]
    assert sc.shot_bounds([50, 100], 150) == [(0, 49), (50, 99), (100, 149)] and sc.shot_bounds([], 10) == [(0, 9)]

    def laid(overrides):
        return [(s['a'], s['b'], s['layout'], s['x']) for s in sc.apply_framing(segs, [50, 100], 150, overrides, 1312.0)]
    # The middle shot cut between two speakers; set to one crop for the whole shot.
    assert laid([(60, 'crop', 500)]) == [(0, 49, 'crop', 100.0), (50, 99, 'crop', 500.0), (100, 149, 'fit', None)]
    # A shot shown whole made a crop, kept inside the picture; one cropped made whole.
    assert laid([(120, 'crop', 9999), (10, 'fit', None)]) == [
        (0, 49, 'fit', None), (50, 69, 'crop', 300.0), (70, 99, 'crop', 900.0), (100, 149, 'crop', 1312.0)]
    # Two for the same shot: the first. One for no shot, or 'auto': nothing changes.
    assert laid([(10, 'crop', 7), (20, 'crop', 9)])[0] == (0, 49, 'crop', 7.0)
    assert laid([(500, 'crop', 7), (10, 'auto', None)]) == laid([]) == [(s['a'], s['b'], s['layout'], s['x']) for s in segs]
    # Shots the planner joined because they were framed alike: one of them changed on its own.
    joined = [{'a': 0, 'b': 149, 'layout': 'crop', 'x': 656.0, 'keys': None}]
    assert [(g['a'], g['b'], g['x']) for g in sc.apply_framing(joined, [50, 100], 150, [(70, 'crop', 10)], 1312.0)] == [
        (0, 49, 656.0), (50, 99, 10.0), (100, 149, 656.0)]
    # A split shot replaced keeps the rest of the clip renderable.
    split = [{'a': 0, 'b': 49, 'layout': 'split', 'x': None, 'keys': None, 'size': (1214, 1080), 'panes': [(0, 0), (700, 0)]},
             {'a': 50, 'b': 99, 'layout': 'crop', 'x': 600.0, 'keys': None}]
    out = sc.apply_framing(split, [50], 100, [(5, 'crop', 10)], 1312.0)
    assert [s['layout'] for s in out] == ['crop', 'crop']
    info = {'width': 1920, 'height': 1080, 'disp_w': 1920, 'disp_h': 1080, 'sar': 1.0, 'sd_matrix': False,
            'fps': 25.0, 'audio_index': 1, 'tag_709': True, 'v_offset': 0.0}
    assert 'split' not in sc.build_filtergraph(info, out) and 'between(n,0,49)*10' in sc.build_filtergraph(info, out)
    # And at the small preview size.
    small = sc.build_filtergraph(info, out, 360, 640)
    assert 'scale=360:640' in small and 'scale=1080:1920' not in small


# ---- a shot with no face in it: where the picture is in focus ----

def _dof_scene(subject_x, soft=True, seed=2):
    """A 1920x1080 picture: a sharply detailed figure 300 px wide at
    `subject_x`, over a background that is soft (shallow focus) or as
    sharp as the figure (deep focus)."""
    rng = np.random.default_rng(seed)
    bg = rng.integers(0, 255, (1080, 1920, 3)).astype(np.uint8)
    bg = cv2.GaussianBlur(bg, (0, 0), 14) if soft else bg
    figure = cv2.GaussianBlur(rng.integers(0, 255, (700, 300, 3)).astype(np.uint8), (0, 0), 1.0)
    bg[300:1000, subject_x - 150:subject_x + 150] = figure
    return bg


def test_the_sharpest_part_of_the_picture_is_where_the_person_is():
    """Drama is shot with the subject sharp and the background soft: someone
    walking away from the camera has no face to find, but is the one thing
    in focus."""
    for x in (500, 1300, 1700):
        at, strength = sc.focus_point(_dof_scene(x), 608)
        assert strength >= sc.FOCUS_STRENGTH and abs(at - x) <= 304 - 150, (x, at, strength)
    at, strength = sc.focus_point(_dof_scene(900, soft=False), 608)
    assert strength < sc.FOCUS_STRENGTH, 'sharp all over: no answer'
    # Anamorphic: the window is measured in display pixels, the picture stored narrower.
    squeezed = cv2.resize(_dof_scene(1300), (1440, 1080), interpolation=cv2.INTER_AREA)
    at, strength = sc.focus_point(squeezed, 608, sar=4 / 3.0)
    assert abs(at - 1300) <= 160 and strength >= sc.FOCUS_STRENGTH
    assert sc.focus_point(np.zeros((360, 640, 3), np.uint8), 608) is None, 'a black frame has nothing in focus'


def test_a_faceless_shot_is_framed_on_what_is_in_focus_when_the_readings_agree():
    no_faces = [(i, []) for i in range(0, 100, 5)]
    steady = [(i, (1300.0 + (i % 3), 2.2)) for i in range(0, 100, 5)]
    seg = sc.plan_reframe(no_faces, [], 100, 1920, 1080, 608, focus=steady)
    assert [(s['layout'], s['keys']) for s in seg] == [('crop', None)] and abs(seg[0]['x'] - (1301 - 304)) <= 2
    # Weak readings, or too few of them: centre crop, as without it.
    weak = [(i, (1300.0, 1.1)) for i in range(0, 100, 5)]
    few = [(i, (1300.0, 2.2) if i < 20 else (900.0, 1.0)) for i in range(0, 100, 5)]
    scattered = [(i, (200.0 if (i // 5) % 2 else 1750.0, 2.2)) for i in range(0, 100, 5)]
    for focus in (weak, few, scattered, None):
        seg = sc.plan_reframe(no_faces, [], 100, 1920, 1080, 608, focus=focus)
        assert seg[0]['x'] == (1920 - 608) / 2.0, focus and focus[:2]
    # A face, when there is one, still decides.
    faces = [(i, [_face(500)]) for i in range(0, 100, 5)]
    assert abs(sc.plan_reframe(faces, [], 100, 1920, 1080, 608, focus=steady)[0]['x'] - (500 - 304)) <= 2
    # And it is a shot-by-shot matter: one shot in focus on the right, the next with a face on the left.
    mixed = [(i, [] if i < 50 else [_face(400)]) for i in range(0, 100, 5)]
    segs = sc.plan_reframe(mixed, [50], 100, 1920, 1080, 608, focus=[(i, (1500.0, 2.0)) for i in range(0, 50, 5)])
    assert [(s['a'], round(s['x'])) for s in segs] == [(0, 1196), (50, 96)]


def test_focus_is_read_only_from_frames_with_no_face(tmp_path):
    frames = [cv2.resize(_dof_scene(1300), (640, 360), interpolation=cv2.INTER_AREA) for _ in range(20)]
    face = frames[0].copy()
    cv2.circle(face, (200, 150), 50, (150, 170, 210), -1)

    class Detector:
        kind = 'test'

        def detect(self, frame):
            return [(150.0, 100.0, 100.0, 100.0, 1.0)] if frame[150, 200, 2] > 200 and frame[150, 200, 0] > 140 else []
    clip = _clip(tmp_path / 'f.avi', frames[:10] + [face] * 10)
    focus = []
    samples = sc.sample_faces(clip, 0, 20, 25.0, Detector(), focus=focus, focus_w=608 / 3.0)
    assert [i for i, _ in focus] == [0, 5] and all(i >= 10 for i, fs in samples if fs)
    assert all(f is not None and abs(f[0] - 1300 / 3.0) < 80 for _, f in focus)
    assert sc.sample_faces(clip, 0, 20, 25.0, Detector()) == samples, 'nothing changes unless it is asked for'


# ---- reported together: a 16-second "short" at a 60-second minimum, crops on nobody, a caption 30 s early ----

def test_a_line_timed_as_lasting_far_longer_than_its_words_is_given_a_start_they_allow():
    """Reported: "Oh, my God." on screen from 0:00 to 0:30 of a short, with
    nobody speaking until the end of it. The service had given the line the
    whole silent stretch before it."""
    lines = [{'start': 100.0, 'end': 130.0, 'text': 'Oh, my God.'},          # 3 words, "30 seconds"
             {'start': 130.0, 'end': 132.0, 'text': "Let's go."},
             {'start': 132.0, 'end': 133.0, 'text': 'Bored na-bored na ako dito.'},
             {'start': 140.0, 'end': 146.5, 'text': 'Hindi... ko... alam.'},  # slow, with pauses: left alone
             {'start': 150.0, 'end': 158.0, 'text': 'Mami?'}]                 # one word, "8 seconds"
    _, segs = sc.normalize_transcript([], lines)
    assert [(round(s['start'], 3), s['end']) for s in segs] == [
        (126.1, 130.0), (130.0, 132.0), (132.0, 133.0), (140.0, 146.5), (155.7, 158.0)]
    assert abs(sc.speaking_time(3) - 3.9) < 1e-9 and sc.trim_absorbed_silence(0.0, 7.7, 3) == 0.0, 'not unless it is far out'
    assert abs(sc.trim_absorbed_silence(0.0, 7.9, 3) - 4.0) < 1e-9

    # Its caption comes up when it is spoken, not when the silence began...
    cues = sc.subtitle_cues([], segs, 99.71, 165.0)
    assert cues[0]['text'] == 'Oh, my God.' and abs(cues[0]['start'] - 26.39) < 0.01 and cues[0]['end'] <= 30.3
    old = sc.subtitle_cues([], lines, 99.71, 165.0)
    assert old[0]['start'] < 0.3 and old[0]['end'] > 30.0, '(as it was: up for half a minute)'
    # ...and a moment that opens on it opens on it, not on thirty seconds of nothing.
    out = sc.build_candidates([{'start_id': 0, 'end_id': 2, 'score': 8, 'title': 'T'}], segs, [], [], [], 200.0,
                              min_dur=5, max_dur=60)
    assert 124.5 <= out[0]['start'] <= 126.1 and out[0]['duration'] < 10

    # With word timings, the words say when: a line that starts long before its first word is pulled in to it,
    words = [{'start': 128.6, 'end': 128.9, 'word': 'Oh,'}, {'start': 129.0, 'end': 129.3, 'word': 'my'},
             {'start': 129.4, 'end': 130.0, 'word': 'God.'}]
    _, segs = sc.normalize_transcript(words, lines[:1])
    assert segs[0]['start'] == 128.6
    # and a first word given the silence as its own length is cut down to a word.
    w, segs = sc.normalize_transcript([dict(words[0], start=100.0)] + words[1:], lines[:1])
    assert w[0]['start'] == 127.9 and w[0]['end'] == 128.9 and segs[0]['start'] == 127.9


def test_the_minimum_length_is_a_minimum():
    """Reported: a 16.5-second moment listed, flagged Short, at a 60-second
    minimum. Its lines stood alone between two long pauses, and falling
    short was allowed."""
    def lines(times):
        return [{'start': float(a), 'end': float(b), 'text': f'line {k}'} for k, (a, b) in enumerate(times)]
    # A 16-second exchange with a 9-second pause before it and an 8-second one after.
    talk = lines([(0, 20), (22, 40), (49, 57), (58, 65), (73, 90), (91, 110), (111, 130)])
    i, j, flags = sc.fit_indices(talk, 2, 3, min_dur=60, max_dur=120)
    assert sc.fit_indices(talk, 2, 3, min_dur=60, max_dur=120, far_gap=6.0) == (2, 3, ['short']), 'as it was'
    assert talk[j]['end'] - talk[i]['start'] >= 60 and 'short' not in flags and 'bridged' in flags
    assert (i, j) == (2, 5), 'across the shorter pause first, and no further than it has to go'
    # Nothing within reach: flagged, for the caller to leave out.
    alone = lines([(0, 5), (30, 46), (80, 90)])
    assert sc.fit_indices(alone, 1, 1, min_dur=60, max_dur=120) == (1, 1, ['short'])

    told = {}
    beats = [{'start_id': 1, 'end_id': 1, 'score': 9, 'title': 'Alone'}]
    plenty = lines([(k * 10, k * 10 + 9) for k in range(40)])
    both = sc.build_candidates(beats + [{'start_id': 10, 'end_id': 12, 'score': 4, 'title': 'Fine'}],
                               lines([(0, 5), (30, 46), (80, 90)]) + [dict(l, start=l['start'] + 200, end=l['end'] + 200) for l in plenty],
                               [], [], [], 700.0, min_dur=60, max_dur=120, report=told)
    assert [c['title'] for c in both] == ['Fine'] and told == {'too_short': 1, 'kept_short': False}
    assert all(c['duration'] >= 60 for c in both), 'the weaker one that is long enough, not the stronger one that is not'
    # Every one of them short: listed as they are, and the caller is told so.
    only = sc.build_candidates(beats, alone, [], [], [], 100.0, min_dur=60, max_dur=120, report=told)
    assert [c['title'] for c in only] == ['Alone'] and 'short' in only[0]['flags']
    assert told == {'too_short': 0, 'kept_short': True}
    # Short by a little is not short: the quiet around it is used first.
    near = lines([(100, 130), (131, 157)])
    c = sc.build_candidates([{'start_id': 0, 'end_id': 1, 'score': 7, 'title': 'Near'}], near, [], [], [], 400.0,
                            min_dur=60, max_dur=120, report=told)[0]
    assert c['duration'] >= 60 and 'short' not in c['flags'] and c['start'] <= 100.0 and told['too_short'] == 0


def _faces(n, fn, step=5):
    return [(i, fn(i)) for i in range(0, n, step)]


def test_the_window_is_placed_on_the_people_in_the_shot_not_frame_by_frame():
    """Reported: crops that were not centred on anyone. A second face found
    in only some of the frames pulled the window part-way toward it, and
    being found on and off was taken for movement."""
    a, b = _face(800, 180), _face(1100, 170)                     # two people who fit one window together
    whole = sc.plan_reframe(_faces(200, lambda i: [a, b]), [], 200, 1920, 1080, 608)
    centre = (800 - 90 + 1100 + 85) / 2.0 - 304
    assert whole[0]['keys'] is None and abs(whole[0]['x'] - centre) <= 2
    # The second one lost by the detector in 60% of the frames: the same framing, and still.
    flicker = sc.plan_reframe(_faces(200, lambda i: [a, b] if (i // 5) % 5 < 2 else [a]), [], 200, 1920, 1080, 608)
    assert flicker[0]['layout'] == 'crop' and flicker[0]['keys'] is None, 'no wandering'
    assert abs(flicker[0]['x'] - centre) <= 2, 'on the two of them'
    # A face that turns up in a few frames only is not someone the shot is of.
    stray = sc.plan_reframe(_faces(200, lambda i: [a, b] if (i // 5) % 7 == 0 else [a]), [], 200, 1920, 1080, 608)
    assert stray[0]['keys'] is None and abs(stray[0]['x'] - (800 - 304)) <= 2, 'centred on the one who is'
    # The two of them walking: followed together, one path.
    walk = sc.plan_reframe(_faces(200, lambda i: [_face(600 + 3 * i, 180), _face(880 + 3 * i, 170)] if i % 15 else
                                  [_face(600 + 3 * i, 180)]), [], 200, 1920, 1080, 608)[0]
    xs = [x for _, x in walk['keys']]
    assert walk['layout'] == 'crop' and xs == sorted(xs), 'never back toward one of them when the other is lost'
    # Seen one after the other and never together -- a pan from one to the
    # next, or one person crossing fast: followed as before, not called a two-shot.
    pan = sc.plan_reframe(_faces(200, lambda i: [_face(500, 200)] if i < 100 else [_face(1400, 200)]), [], 200, 1920, 1080, 608)
    assert [s['layout'] for s in pan] == ['crop'] and pan[0]['keys']
    assert sc._people([(i, {'sig': [_face(500, 200)] if i < 50 else [_face(1400, 200)]}) for i in range(0, 100, 5)]) is None
    assert sc._people([(0, {'sig': [a]}), (5, {'sig': [a]})]) is None, 'too few frames to say'


def test_a_shot_of_one_person_with_another_behind_is_cropped_to_that_person():
    """Reported earlier, with a frame: a woman in the middle, a man behind
    her at the edge, shown as a small wide shot. They do not fit one window,
    and nothing said she was the subject."""
    woman, man = _face(960, 296), _face(1700, 226)               # her face 1.7 times the area of his
    for mode in ('auto', 'crop'):
        seg = sc.plan_reframe(_faces(200, lambda i: [woman, man]), [], 200, 1920, 1080, 608, mode=mode)
        assert [(s['layout'], s['keys']) for s in seg] == [('crop', None)] and abs(seg[0]['x'] - (960 - 304)) <= 2, mode
    # Split screen still gives them a pane each: that is what was asked for.
    assert [s['layout'] for s in sc.plan_reframe(_faces(200, lambda i: [woman, man]), [], 200, 1920, 1080, 608,
                                                 mode='split')] == ['split']
    # Two people much the same size are still two people: nobody is cut out on a guess.
    alike = sc.plan_reframe(_faces(200, lambda i: [_face(400, 260), _face(1500, 250)]), [], 200, 1920, 1080, 608)
    assert [s['layout'] for s in alike] == ['fit']
    nearly = sc.plan_reframe(_faces(200, lambda i: [_face(400, 300), _face(1500, 245)]), [], 200, 1920, 1080, 608)
    assert [s['layout'] for s in nearly] == ['fit'], '1.5 times the area is not clearly nearer'
    assert sc.DOMINANT_FACE == 1.6


# ---- the cliffhanger: where its hold comes from, and where its sound stops ----

def test_the_hold_is_made_from_the_frames_after_the_out_point_when_there_are_some():
    # With frames to use after the clip, it gives up none of its own: it stops ON its last frame.
    assert sc.cliffhanger_plan(250, 25.0, after=4) == (4, 50, 10) and sc.cliffhanger_stop(250, 25.0, after=4) == 250
    assert sc.cliffhanger_extra(250, 25.0, after=4) == 60
    assert sc.cliffhanger_plan(250, 25.0, room=1, after=2) == (2, 50, 10), 'what is behind the clip no longer matters'
    assert sc.cliffhanger_plan(250, 25.0, after=1) == (1, 50, 10) and sc.cliffhanger_plan(250, 30000 / 1001.0, after=9) == (5, 60, 12)
    # With none, as before: from its own end.
    assert sc.cliffhanger_stop(250, 25.0) == 246 and sc.cliffhanger_stop(250, 25.0, room=1, after=0) == 249
    assert sc.cliffhanger_extra(250, 25.0) == 56

    g = sc.cliffhanger_graph(250, 25.0, None, ass_name='c.ass', after=4)
    assert '[em]trim=end_frame=250,setpts=PTS-STARTPTS,ass=c.ass,' in g and '[et]trim=start_frame=250:end_frame=254,' in g
    assert 'setpts=N*6.250000/(25)/TB' in g and "enable='gte(n,50)'" in g and 'stop=60,' in g
    assert g.replace('end_frame=250,', 'end_frame=246,').replace('start_frame=250:end_frame=254', 'start_frame=246:end_frame=250') \
        == sc.cliffhanger_graph(250, 25.0, None, ass_name='c.ass'), 'the same ending, four frames later'

    # The reframing plan is carried over those frames: the last shot's framing, held.
    still = [{'a': 0, 'b': 99, 'layout': 'fit', 'x': None, 'keys': None},
             {'a': 100, 'b': 249, 'layout': 'crop', 'x': 656.0, 'keys': None}]
    on = sc.run_on(still, 4)
    assert (on[1]['b'], on[0]['b'], still[1]['b']) == (253, 99, 249), 'a copy; the plan itself is not changed'
    assert 'between(n,100,253)*656' in sc.crop_x_expr(on)
    pan = [{'a': 0, 'b': 249, 'layout': 'crop', 'x': None, 'keys': [(0, 100.0), (249, 600.0)]}]
    expr = sc.crop_x_expr(sc.run_on(pan, 4))
    assert 'between(n,0,248)*(100.0+(500.0)*(n-0)/249)' in expr and 'between(n,249,253)*(600.0+(0.0)*(n-249)/4)' in expr, expr
    assert sc.run_on(pan, 0) is pan and sc.run_on([], 4) == []
    split = [{'a': 0, 'b': 249, 'layout': 'split', 'x': None, 'keys': None, 'size': (1214, 1080), 'panes': [(0, 0), (700, 0)]}]
    assert sc.split_exprs(sc.run_on(split, 4))[1][0] == 'between(n,0,253)*700'

    info = {'width': 1920, 'height': 1080, 'disp_w': 1920, 'disp_h': 1080, 'sar': 1.0, 'sd_matrix': False,
            'fps': 25.0, 'audio_index': 1, 'tag_709': True, 'v_offset': 0.0}
    full = sc.build_filtergraph(info, still, ending=(250, None, 4))
    assert "between(n,100,253)*656" in full and "overlay=0:0:enable='between(n,0,99)'" in full
    assert 'trim=start_frame=250:end_frame=254' in full
    assert sc.build_filtergraph(info, still, ending=(250, 3)) == sc.build_filtergraph(info, still, ending=(250, 3, 0)), \
        'a plan from before this says nothing about what follows: nothing does'

    cmd = sc.build_render_cmd('ffmpeg', '/in.mp4', '/out.mp4', 0, 250, info, still, ending=True, ending_after=4)
    assert cmd[cmd.index('-frames:v') + 1] == '310' and abs(float(cmd[cmd.index('-t') + 1]) - 12.4) < 1e-6


def test_the_sound_of_the_ending_stops_where_it_is_told_not_where_the_picture_does():
    info = {'width': 1920, 'height': 1080, 'disp_w': 1920, 'disp_h': 1080, 'sar': 1.0, 'sd_matrix': False,
            'fps': 25.0, 'audio_index': 1, 'tag_709': True, 'v_offset': 0.0}
    segs = [{'a': 0, 'b': 249, 'layout': 'crop', 'x': 656.0, 'keys': None}]

    def af(**kw):
        cmd = sc.build_render_cmd('ffmpeg', '/in.mp4', '/out.mp4', 0, 250, info, segs, ending=True, **kw)
        return cmd[cmd.index('-af') + 1]
    # A word that finishes 0.45 s after the out point: let run on under the hold, then rung out.
    assert af(ending_after=4, audio_out=10.45, audio_fade=0.2) == (
        'afade=t=in:st=0:d=0.04,atrim=end=10.650,afade=t=out:st=10.450:d=0.200:curve=cub,'
        'loudnorm=I=-14.0:TP=-1.5:LRA=11,apad=whole_dur=12.400')
    # Speech running on: stopped a little BEFORE the out point, fast.
    assert 'atrim=end=9.990,afade=t=out:st=9.940:d=0.050:curve=cub' in af(ending_after=4, audio_out=9.94, audio_fade=0.05)
    # Told nothing (a short made before the sound was measured): where the picture stops, as then.
    assert 'afade=t=out:st=9.840:d=0.200:curve=cub' in af() and 'afade=t=out:st=10.000:d=0.200' in af(ending_after=4)
    # The picture's own frames, but the sound still to the out point and beyond.
    assert 'afade=t=out:st=10.300:d=0.200' in af(ending_room=4, audio_out=10.3, audio_fade=0.2)
    # Never to the end of the hold: the last half second of it is silent whatever is asked.
    late = af(ending_after=4, audio_out=30.0, audio_fade=0.2)
    assert 'afade=t=out:st=11.300:d=0.200' in late and 'atrim=end=11.500' in late
    assert sc.AUDIO_FADE == {'pause': 0.2, 'quiet': 0.12, 'dip': 0.05, None: 0.12}
    assert sc.WORD_END['reach'] < sc.CLIFFHANGER['hold'] - 0.5 - sc.AUDIO_FADE['pause'], 'a word let finish always fits'


def _speech(spans, seconds=4.0, bed=-55.0, rate=16000, seed=3):
    """Mono samples: bursts ('words') at the given times over a quiet bed."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * rate)) / rate
    x = rng.normal(0, 10 ** (bed / 20.0), len(t))
    for a, b in spans:
        m = (t >= a) & (t < b)
        env = np.minimum(1.0, np.minimum((t[m] - a) / 0.02, (b - t[m]) / 0.06))
        x[m] += env * 0.12 * np.sin(2 * np.pi * 180 * t[m]) * (0.6 + 0.4 * np.sin(2 * np.pi * 7 * t[m]))
    return x


def test_where_a_word_really_ends_is_read_off_the_sound():
    words = [(0.2, 0.6), (0.7, 1.1), (1.2, 1.7), (1.8, 2.75)]

    def end(spans, at, nxt=None, **kw):
        return sc.word_end(sc.audio_envelope(_speech(spans, **kw), 16000), at, nxt)
    env = sc.audio_envelope(_speech(words), 16000)
    assert len(env) == 400 and env[40] > env[65] + 20, 'one reading every 10 ms; a word is far above a gap'
    # In the pause after the word: it stops there.
    assert end(words, 2.9) == (2.9, 'quiet')
    # Inside the last word -- the transcript ended it early: on to where it gives way to quiet.
    at, kind = end(words, 2.5)
    assert kind == 'pause' and 2.72 <= at <= 2.80
    # The same, whatever the transcript believes comes next, when a pause is what is there.
    assert end(words + [(3.5, 3.9)], 2.5, 2.7)[1] == 'pause'
    # Speech running on with a 50 ms gap. Out just inside it, or 60 ms into the next word: the gap.
    run = words + [(2.80, 3.6)]
    for out in (2.78, 2.86):
        at, kind = end(run, out, 2.80)
        assert kind in ('quiet', 'dip') and 2.74 <= at <= 2.81, (out, at, kind)
    # ...and not a deeper gap a word away.
    at, kind = end(words[:3] + [(1.75, 2.30), (2.34, 3.6)], 2.36, 2.34)
    assert kind == 'dip' and 2.28 <= at <= 2.36, (at, kind)
    # Over a music bed it never goes silent; it drops to the bed, and that is the pause.
    at, kind = end(words, 2.5, bed=-34.0)
    assert kind == 'pause' and 2.70 <= at <= 2.80
    # Nothing to go on: one level throughout, silence, a sound that is not about to stop, the edge of what was read.
    assert end([(0.0, 4.0)], 2.5) is None and end([], 2.5) is None
    assert end(words[:3] + [(1.8, 4.0)], 2.5) is None, 'not chased past the reach'
    assert end(words, 0.0) is None and end(words, 3.99) is None and sc.word_end(np.zeros(5), 0.02) is None
    assert len(sc.audio_envelope(np.zeros(100), 16000)) == 0


def test_stillness_is_judged_forwards_as_well_as_back(tmp_path):
    x = np.arange(640)
    scene = np.repeat((110 + 70 * np.sin(x / 23.0) + 30 * np.sin(x / 5.0))[None, :, None], 360, axis=0)
    scene = np.repeat(scene, 3, axis=2).astype(np.uint8)

    def frame(shift=0, breath=0.0):
        f = np.roll(scene, shift, axis=1).copy()
        cv2.ellipse(f, (320, 300), (110, int(round(40 + breath))), 0, 0, 360, (90, 60, 50), -1)
        return f
    # Still to frame 23, then the camera moves; a different picture from frame 30.
    frames = [frame(breath=1.5 * np.sin(i / 6.0)) for i in range(24)] + [frame(shift=12 * (i - 23)) for i in range(24, 30)] \
        + [np.full((360, 640, 3), 200, np.uint8)] * 10
    clip = _clip(tmp_path / 'c.avi', frames)
    assert sc.still_frames_after(clip, 10, 25.0, 30) == 4, 'the whole 0.16 s'
    assert sc.still_frames_after(clip, 10, 25.0, 2) == 2 and sc.still_frames_after(clip, 10, 25.0, 1) == 1
    assert sc.still_frames_after(clip, 10, 25.0, 0) == 0 == sc.still_frames_after(clip, 10, 25.0, None)
    assert sc.still_frames_after(clip, 22, 25.0, 30) == 2, 'only as far as it stays still'
    assert sc.still_frames_after(clip, 25, 25.0, 30) == 1, 'moving on from there: one frame, held'
    assert sc.still_frames_after(clip, 29, 25.0, 30) == 1, 'a cut the cut list missed is not blended across'
    assert sc.still_frames_after(clip, 400, 25.0, 30) == 0 and sc.still_frames_after(str(tmp_path / 'no.avi'), 0, 25.0, 9) == 0
    assert sc.still_frames(clip, 23, 25.0) == 4, 'and backwards as it was'


# ---- captions an editor has corrected ----

def test_corrected_captions_are_tidied_into_something_that_can_be_burned_in():
    raw = [{'start': 2, 'end': 4, 'text': '  second   line '}, {'start': 0, 'end': 2.5, 'text': 'first'},
           {'start': 5, 'end': 6, 'text': '   '}, {'start': 9, 'end': 12, 'text': 'runs past the end'},
           {'start': 20, 'end': 21, 'text': 'after it'}, {'start': '7.5', 'end': '8', 'text': 'typed as text'}]
    assert sc.clean_cues(raw, 10) == [
        {'start': 0.0, 'end': 2.0, 'text': 'first'},            # gives way to the one that follows it
        {'start': 2.0, 'end': 4.0, 'text': 'second line'},      # in time order, spaces tidied
        {'start': 7.5, 'end': 8.0, 'text': 'typed as text'},
        {'start': 9.0, 'end': 10.0, 'text': 'runs past the end'}]   # kept inside; the emptied and the outside ones gone
    assert sc.clean_cues([], 10) == []
    # One swallowed whole by the next is dropped, not left as a single-frame flash.
    assert [c['text'] for c in sc.clean_cues([{'start': 1.0, 'end': 3.0, 'text': 'a'},
                                              {'start': 1.02, 'end': 3.0, 'text': 'b'}], 10)] == ['b']
    for bad, says in ((None, 'must be a list'), ([3], 'not valid'),
                      ([{'start': 'x', 'end': 2, 'text': 'Hi there'}], '"Hi there"'),
                      ([{'start': 3, 'end': 3, 'text': 'Zero'}], 'must end after it starts'),
                      ([{'start': float('nan'), 'end': 3, 'text': 'NaN'}], 'must be numbers'),
                      ([{'start': 0, 'end': 1, 'text': 'x' * 201}], 'Split it into two'),
                      ([{'start': 0, 'end': 1, 'text': 'x'}] * 801, 'At most 800')):
        with pytest.raises(ValueError) as e:
            sc.clean_cues(bad, 10)
        assert says in str(e.value), (bad if not isinstance(bad, list) else len(bad), str(e.value))


def test_corrected_captions_follow_the_moment_when_its_in_and_out_are_moved():
    edited = [{'start': 0.0, 'end': 2.0, 'text': 'one'}, {'start': 4.0, 'end': 5.0, 'text': 'two'},
              {'start': 8.0, 'end': 10.0, 'text': 'three'}]
    asked = []

    def auto(a, b):
        asked.append((a, b))
        return [{'start': 0.1, 'end': round(min(1.0, b - a), 3), 'text': f'auto {a:g}-{b:g}'}]
    # Unmoved: exactly what was edited, and nothing automatic.
    assert sc.fit_cues(edited, 100, 110, 100, 110, auto) == edited and asked == []
    # Both ends moved out: the edits where they now fall, automatic captions for what was added.
    out = sc.fit_cues(edited, 100, 110, 98, 113, auto)
    assert [(c['start'], c['end'], c['text']) for c in out] == [
        (0.1, 1.0, 'auto 98-100'), (2.0, 4.0, 'one'), (6.0, 7.0, 'two'), (10.0, 12.0, 'three'), (12.1, 13.0, 'auto 110-113')]
    assert asked == [(98.0, 100.0), (110.0, 113.0)]
    # Both ends moved in: what is cut off goes, what straddles the new edge is clipped to it.
    out = sc.fit_cues(edited, 100, 110, 101, 109, auto)
    assert [(c['start'], c['end'], c['text']) for c in out] == [(0.0, 1.0, 'one'), (3.0, 4.0, 'two'), (7.0, 8.0, 'three')]
    # Moved clear of every edit: all automatic.
    asked.clear()
    assert [c['text'] for c in sc.fit_cues(edited, 100, 110, 200, 210, auto)] == ['auto 200-210']
    # A nudge of a few frames is not a stretch worth captioning on its own.
    asked.clear()
    assert sc.fit_cues(edited, 100, 110, 99.8, 110.2, auto)[0]['text'] == 'one' and asked == []


def test_a_story_floor_keeps_the_moments_worth_making():
    segs = [{'start': 10.0 * i, 'end': 10.0 * i + 8.0, 'text': f'line {i}'} for i in range(30)]
    beats = [{'start_id': 0, 'end_id': 3, 'score': 9, 'title': 'A'}, {'start_id': 8, 'end_id': 11, 'score': 6, 'title': 'B'},
             {'start_id': 16, 'end_id': 19, 'score': 5, 'title': 'C'}, {'start_id': 24, 'end_id': 27, 'score': 2, 'title': 'D'}]

    def titles(**kw):
        return sorted(c['title'] for c in sc.build_candidates(beats, segs, [], [], [], 300.0, min_dur=20, max_dur=60,
                                                              limit=100, **kw))
    assert titles() == ['A', 'B', 'C', 'D'] and titles(min_story=6) == ['A', 'B'] and titles(min_story=9) == ['A']
    assert titles(min_story=10) == ['A', 'B', 'C', 'D'], 'none reach it: all of them, not none'
    # Moments nobody scored (a fallback was used) are not judged by it.
    unscored = [dict(b, score=None, source='heuristic') for b in beats]
    assert len(sc.build_candidates(unscored, segs, [], [], [], 300.0, min_dur=20, max_dur=60, limit=100, min_story=6)) == 4
    ids = [c['id'] for c in sc.build_candidates(beats, segs, [], [], [], 300.0, min_dur=20, max_dur=60, limit=100, min_story=6)]
    assert ids == ['c1', 'c2'], 'numbered after the weak ones are taken out'


def test_a_saved_reframing_plan_renders_the_same_after_a_trip_through_json():
    """A short's plan is kept in its batch's manifest so it can be rendered
    again with new captions. JSON turns its tuples into lists."""
    info = {'width': 1920, 'height': 1080, 'disp_w': 1920, 'disp_h': 1080, 'sar': 1.0, 'sd_matrix': False,
            'fps': 25.0, 'audio_index': 1, 'tag_709': True, 'v_offset': 0.0, 'audio_take': [[2, 0], [3, 0]]}
    segs = [{'a': 0, 'b': 99, 'layout': 'crop', 'x': None, 'keys': [(0, 100.0), (40, 400.5), (99, 655.25)]},
            {'a': 100, 'b': 149, 'layout': 'fit', 'x': None, 'keys': None},
            {'a': 150, 'b': 199, 'layout': 'split', 'x': None, 'keys': None,
             'people': ((400.0, 500.0, 200.0, 220.0), (1500.0, 480.0, 190.0, 210.0))},
            {'a': 200, 'b': 249, 'layout': 'crop', 'x': 656.0, 'keys': None, 'speaker': True, 'person': 1}]
    for s in segs:
        if s['layout'] == 'split':
            s['size'] = sc.split_window(s['people'], 1920, 1080)
    sc._finish_split(segs, 1920, 1080)
    back = json.loads(json.dumps({'segs': segs, 'info': info}))
    for kw in ({}, {'ass_name': 'c.ass'}, {'ending': (250, 4)}):
        assert sc.build_filtergraph(back['info'], back['segs'], **kw) == sc.build_filtergraph(info, segs, **kw)
    assert sc.build_render_cmd('ffmpeg', '/in.mp4', '/out.mp4', 10, 250, back['info'], back['segs'], ending=True) == \
        sc.build_render_cmd('ffmpeg', '/in.mp4', '/out.mp4', 10, 250, info, segs, ending=True)
    cues = [{'start': 6.5, 'end': 7.5, 'text': 'over the split'}, {'start': 1.0, 'end': 2.0, 'text': 'not'}]
    assert [c['seam'] for c in sc.place_cues(cues, back['segs'], 25.0)] == [True, False]


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


# ---- split screen ----

def _two(n_frames, lx=500, rx=1500, w=160, step=5, extra=None):
    """A wide two-shot with plain 4-value faces (no mouth data)."""
    return [(i, [_face(lx, w), _face(rx, w)] + (extra or [])) for i in range(0, n_frames, step)]


def test_split_pane_window_is_a_full_height_9_by_8_slice():
    w, h = sc.split_pane_size(1920, 1080)
    assert (w, h) == (1216, 1080) and abs(w / h - 1080 / 960) < 0.002
    assert sc.split_pane_size(640, 360) == (404, 360)
    assert sc.split_pane_size(1080, 1920) == (1080, 960), 'a source narrower than a pane: full width instead'


def test_split_gives_each_of_two_people_half_the_frame_left_on_top():
    segs = sc.plan_reframe(_two(100), [], 100, 1920, 1080, 608, mode='split')
    assert len(segs) == 1
    s = segs[0]
    assert (s['a'], s['b'], s['layout'], s['x'], s['keys']) == (0, 99, 'split', None, None)
    w, h = s['size']
    assert (w, h) == (1216, 1080), 'no tightening needed: the two are far enough apart'
    (tx, ty), (bx, by) = s['panes']
    assert tx == 0 and bx == 1920 - 1216, 'each window pushed to its own side of the picture'
    assert tx + w < 1500 - 80 and bx > 500 + 80, "and neither holds the other person's face"
    assert ty == 0 and by == 0
    # The same shot in Auto is shown whole; in "always crop" one side is chosen.
    assert [x['layout'] for x in sc.plan_reframe(_two(100), [], 100, 1920, 1080, 608)] == ['fit']
    assert [x['layout'] for x in sc.plan_reframe(_two(100), [], 100, 1920, 1080, 608, mode='crop')] == ['crop']


def test_split_tightens_the_window_until_the_other_face_is_out_of_it():
    # Close together, and the left one near the edge so their window cannot be centred on them.
    people = ((200.0, 400.0, 160.0, 160.0), (1000.0, 400.0, 160.0, 160.0))
    w, h = sc.split_window(people, 1920, 1080)
    assert w < 1216 and abs(w / h - 1216 / 1080) < 0.01, 'tightened, same shape'
    lx, _ = sc._pane_origin(people[0], w, h, 1920, 1080)
    rx, _ = sc._pane_origin(people[1], w, h, 1920, 1080)
    assert lx + w <= 1000 - 80 and rx >= 200 + 80
    # Faces lifted towards the top of their pane, never past the picture's edge.
    _, y = sc._pane_origin((960.0, 500.0, 160.0, 160.0), w, h, 1920, 1080)
    assert abs((500 - y) / h - sc.SPLIT_FACE_AT) < 0.02
    assert sc._pane_origin((960.0, 1000.0, 160.0, 160.0), w, h, 1920, 1080)[1] == 1080 - h
    assert sc._pane_origin((960.0, 100.0, 160.0, 160.0), w, h, 1920, 1080)[1] == 0
    # Too close to separate without blowing a face up: no split.
    assert sc.split_window(((700.0, 400.0, 260.0, 260.0), (1000.0, 400.0, 260.0, 260.0)), 1920, 1080) is None


def test_split_only_applies_to_exactly_two_people_too_far_apart():
    def plan(samples, **kw):
        return [(x['a'], x['b'], x['layout'], x['x'], x['keys']) for x in
                sc.plan_reframe(samples, [], 100, 1920, 1080, 608, **kw)]
    third = _two(100, extra=[_face(1000, 150)])
    assert plan(third, mode='split') == [(0, 99, 'fit', None, None)], 'a third person has no pane: show everyone'
    close = _two(100, lx=800, rx=1150, w=260)
    assert plan(close, mode='split') == plan(close) == [(0, 99, 'fit', None, None)], 'cannot be separated cleanly'
    for samples in (_two(100, lx=800, rx=1050), _samples(100, lambda i: [_face(500)]), _samples(100, lambda i: [])):
        assert plan(samples, mode='split') == plan(samples), 'two who fit, one person, nobody: exactly as Auto'


def test_every_split_shot_in_a_clip_shares_one_window_size():
    samples = _two(100) + [(i, [_face(200), _face(1000)]) for i in range(100, 200, 5)]
    segs = sc.plan_reframe(samples, [100], 200, 1920, 1080, 608, mode='split')
    assert [(s['a'], s['b'], s['layout']) for s in segs] == [(0, 99, 'split'), (100, 199, 'split')]
    assert segs[0]['size'] == segs[1]['size'] and segs[0]['size'][0] < 1216, 'the tighter of the two, for both'
    assert segs[0]['panes'] != segs[1]['panes']
    w = segs[0]['size'][0]
    for s in segs:
        (lcx, _, lw, _), (rcx, _, rw, _) = s['people']
        assert s['panes'][0][0] + w <= rcx - rw / 2 and s['panes'][1][0] >= lcx + lw / 2
    # Two consecutive shots framed identically are one instruction.
    same = sc.plan_reframe(_two(200), [100], 200, 1920, 1080, 608, mode='split')
    assert [(s['a'], s['b']) for s in same] == [(0, 199)]


def test_split_with_follow_the_speaker_splits_a_conversation_and_frames_a_monologue():
    speech = _words([(0, 8)])
    conversation = _talking(200, [(0, 4)], [(4, 8)])
    segs = sc.plan_reframe(conversation, [], 200, 1920, 1080, 608, mode='split', speaker=True, speech=speech)
    assert [s['layout'] for s in segs] == ['split'], 'both talk: both stay on screen'
    monologue = _talking(200, [(0, 8)], [])
    segs = sc.plan_reframe(monologue, [], 200, 1920, 1080, 608, mode='split', speaker=True, speech=speech)
    assert [(s['layout'], round(s['x']), s['speaker']) for s in segs] == [('crop', 196, True)], \
        'one person talking: half the screen is not spent on the listener'
    # One line from the other person in a long speech is not a conversation.
    aside = _talking(500, [(0, 16.6), (18.2, 20)], [(16.6, 18.2)])
    segs = sc.plan_reframe(aside, [], 500, 1920, 1080, 608, mode='split', speaker=True, speech=_words([(0, 20)]))
    assert all(s['layout'] == 'crop' and s['speaker'] for s in segs) and len(segs) == 3
    # Nothing to go on (no transcript): split, which is what the mode is for.
    segs = sc.plan_reframe(conversation, [], 200, 1920, 1080, 608, mode='split', speaker=True, speech=[])
    assert [s['layout'] for s in segs] == ['split']


def test_filtergraph_with_a_split_layout_alone_and_mixed_with_the_others():
    info = {'width': 1920, 'height': 1080, 'disp_w': 1920, 'disp_h': 1080, 'sar': 1.0, 'sd_matrix': False}
    samples = (_samples(50, lambda i: [_face(400)]) + [(i, f) for i, f in _two(100) if i >= 50] +
               [(i, [_face(500), _face(1000, 150), _face(1500)]) for i in range(100, 150, 5)])
    segs = sc.plan_reframe(samples, [50, 100], 150, 1920, 1080, 608, mode='split')
    assert [(s['a'], s['b'], s['layout']) for s in segs] == [(0, 49, 'crop'), (50, 99, 'split'), (100, 149, 'fit')]
    g = sc.build_filtergraph(info, segs, ass_name='c.ass')
    assert ',split=3[c0][f0][s0];' in g
    assert "[sa]crop=1216:1080:x='between(n,50,99)*0':y='between(n,50,99)*0',scale=1080:960:flags=lanczos[st]" in g
    assert "[sb]crop=1216:1080:x='between(n,50,99)*704':y='between(n,50,99)*0',scale=1080:960:flags=lanczos[sm]" in g
    assert '[st][sm]vstack,drawbox=x=0:y=957:w=1080:h=6' in g
    assert "[cv][fv]overlay=0:0:enable='between(n,100,149)'[m0];[m0][sv]overlay=0:0:enable='between(n,50,99)'[v1]" in g
    assert g.index('ass=c.ass') > g.index('[v1]'), 'captions go on after the layouts are combined'
    # Every frame belongs to exactly one layout.
    for n in range(150):
        on = [name for name, pat in (('fit', r"\[cv\]\[fv\]overlay=0:0:enable='([^']*)'"),
                                     ('split', r"\[sv\]overlay=0:0:enable='([^']*)'"))
              if _eval_expr(re.search(pat, g).group(1), n)]
        assert on == ([] if n < 50 else ['split'] if n < 100 else ['fit']), (n, on)

    only = sc.build_filtergraph(info, [s for s in sc.plan_reframe(_two(100), [], 100, 1920, 1080, 608, mode='split')])
    assert 'overlay' not in only and 'boxblur' not in only and '[st][sm]vstack' in only
    two = sc.build_filtergraph(info, segs[:2])
    assert ',split=2[c0][s0];' in two and "[cv][sv]overlay=0:0:enable='between(n,50,99)'[v1]" in two
    # The two-layout graph that already existed is unchanged.
    old = sc.build_filtergraph(info, [segs[0], segs[2]])
    assert ",split=2[c0][f0];" in old and "[cv][fv]overlay=0:0:enable='between(n,100,149)'[v1]" in old


def test_captions_over_a_split_shot_sit_on_the_join(tmp_path):
    segs = [{'a': 0, 'b': 49, 'layout': 'crop', 'x': 0.0, 'keys': None},
            {'a': 50, 'b': 99, 'layout': 'split', 'x': None, 'keys': None}]
    cues = [{'start': 0.2, 'end': 1.4, 'text': 'in the crop'}, {'start': 1.8, 'end': 2.4, 'text': 'straddles the cut'},
            {'start': 2.5, 'end': 3.6, 'text': 'in the split'}]
    placed = sc.place_cues(cues, segs, 25.0)
    assert [c['seam'] for c in placed] == [False, True, True], 'by midpoint: 2.1 s is frame 52'
    path = tmp_path / 'c.ass'
    sc.write_ass(placed, str(path))
    lines = [ln for ln in path.read_text(encoding='utf-8').splitlines() if ln.startswith('Dialogue:')]
    assert lines[0].endswith(',,in the crop')
    assert lines[2].endswith(',,{\\an5\\pos(540,960)}in the split')
    assert all(c['seam'] is False for c in sc.place_cues(list(cues), segs[:1], 25.0))


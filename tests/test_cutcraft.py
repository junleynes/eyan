"""Clean cuts, card fit and script-aware narration (cutcraft.py)."""
import shutil
import subprocess

import pytest

import cutcraft as cc

# "Abangan ngayong gabi, only on GMA."  One sentence 10.0-12.6 s, next 14.0-15.0
WORDS = [(10.0, 10.5), (10.55, 11.2), (11.25, 11.6), (11.9, 12.2), (12.25, 12.6),
         (14.0, 14.4), (14.45, 15.0)]
SENT = [(10.0, 12.6), (14.0, 15.0)]


# ------------------------------------------------------------------ cuts

def test_in_point_inside_a_word_backs_up_before_the_word():
    words = [(1.0, 1.5), (2.0, 2.6)]
    t = cc.settle_in(2.3, words, lo=0.0, sentences=[])
    assert 1.5 < t < 2.0                   # in the pause before the word, after the previous one


def test_in_point_prefers_sentence_start_when_mid_sentence():
    t = cc.settle_in(11.3, WORDS, lo=9.0, sentences=SENT, sent_back=1.5)
    assert 9.7 <= t <= 10.0                # the sentence starts at 10.0, room kept before it


def test_in_point_goes_to_next_sentence_when_closer():
    t = cc.settle_in(12.4, WORDS, lo=9.0, hi=20, sentences=SENT, sent_fwd=2.0)
    assert 13.7 <= t <= 14.0


def test_in_point_leaves_silence_alone():
    assert cc.settle_in(13.0, WORDS, lo=9.0, sentences=SENT) == 13.0


def test_in_point_keeps_room_before_a_word_that_starts_right_after():
    t = cc.settle_in(13.95, WORDS, lo=9.0)
    assert 13.7 <= t < 13.9


def test_in_point_inside_first_word_of_scene_moves_to_next_word():
    # the scene starts at 10.2, inside the first word
    t = cc.settle_in(10.3, WORDS, lo=10.2, sentences=[])
    assert t >= 10.2
    assert cc.classify_cut('in', t, WORDS) != 'clipped'


def test_out_point_inside_a_word_extends_past_it_with_room():
    t = cc.settle_out(11.5, WORDS, lo=10.0, hi=20.0)
    assert t >= 11.6 + 0.03


def test_out_point_pulls_back_when_the_scene_ends_inside_the_word():
    t = cc.settle_out(11.5, WORDS, lo=10.0, hi=11.55)
    assert t <= 11.25                      # ends before word 3 starts
    assert cc.classify_cut('out', t, WORDS) != 'clipped'


def test_out_point_in_silence_is_unchanged():
    assert cc.settle_out(13.0, WORDS, lo=10.0, hi=20.0) == 13.0


def test_out_point_room_after_last_word_is_capped_by_the_gap():
    t = cc.settle_out(12.61, WORDS, lo=10.0, hi=20.0)
    assert t >= 12.61


def test_refine_words_moves_edges_to_measured_silence():
    words = [(1.0, 1.5), (1.6, 2.0)]
    quiet = [(1.42, 1.62)]                 # sound really stops at 1.42, resumes 1.62
    out = cc.refine_words(words, quiet)
    assert out[0][1] == pytest.approx(1.42)
    assert out[1][0] == pytest.approx(1.62)


def test_refine_words_ignores_far_silences():
    words = [(1.0, 1.5)]
    assert cc.refine_words(words, [(3.0, 3.5)]) == [(1.0, 1.5)]


def test_classify_cut_states():
    assert cc.classify_cut('in', 10.7, WORDS) == 'clipped'
    assert cc.classify_cut('in', 5.0, WORDS) == 'silent'
    assert cc.classify_cut('in', 9.9, WORDS, SENT) == 'clean'          # at a sentence start
    assert cc.classify_cut('in', 11.8, WORDS, SENT) == 'word'           # a pause, but mid-sentence
    assert cc.classify_cut('in', 10.54, WORDS, SENT) == 'tight'
    assert cc.classify_cut('out', 12.7, WORDS, SENT) == 'clean'         # after the sentence's end
    assert cc.classify_cut('out', 12.61, WORDS, SENT) == 'tight'


def test_refine_clip_cleans_both_edges_and_reports():
    start, dur, rep = cc.refine_clip(10.7, 1.0, (9.0, 20.0), WORDS, SENT, xfade=0.3)
    assert cc.classify_cut('in', start, WORDS, SENT) in ('clean', 'word')
    assert cc.classify_cut('out', start + dur, WORDS, SENT) in ('clean', 'word')
    assert rep['in'] != 'clipped' and rep['out'] != 'clipped'
    assert rep['moved_in'] < 0


def test_refine_clip_never_leaves_the_scene():
    start, dur, _ = cc.refine_clip(10.7, 0.9, (10.6, 11.6), WORDS, [], xfade=0.3)
    assert start >= 10.6 - 1e-6 and start + dur <= 11.6 + 1e-6


def test_refine_clip_keeps_original_if_result_would_be_too_short():
    start, dur, _ = cc.refine_clip(10.7, 0.9, (10.7, 11.6), [(10.6, 12.0)], [], xfade=0.3, min_len=0.8)
    assert start == 10.7 and dur == pytest.approx(0.9)


def test_cut_summary_counts_risky_edges():
    s = cc.cut_summary([{'in': 'clean', 'out': 'clipped'}, {'in': 'tight', 'out': 'silent'}])
    assert s['edges'] == 4 and s['risky'] == 2 and s['clean'] == 1


# ------------------------------------------------------------------ cards

def test_card_floor_respects_voice_and_picture():
    assert cc.card_floor(5.0) == 1.0
    assert cc.card_floor(5.0, voice_end=2.0) == pytest.approx(2.25)
    assert cc.card_floor(5.0, voice_end=2.0, still_from=2.5) == pytest.approx(2.9)
    assert cc.card_floor(2.0, voice_end=4.0) == 2.0           # never over its own length


def test_parse_freeze_start():
    assert cc.parse_freeze_start('[freezedetect @ 0x1] lavfi.freezedetect.freeze_start: 1.52\n') == 1.52
    assert cc.parse_freeze_start('nothing') is None


def test_card_adjust_limit_scales_with_length():
    assert cc.card_adjust_limit(15) == 1.8
    assert cc.card_adjust_limit(30) == 3.6
    assert cc.card_adjust_limit(5) == 1.0
    assert cc.card_adjust_limit(30, 'off') == 1.0


def test_plan_card_fit_leaves_alone_when_scenes_have_room():
    p = cc.plan_card_fit(30, [3.0, 3.0], [1.0, 1.0])
    assert p['targets'] == [3.0, 3.0] and not p['shrunk']


def test_plan_card_fit_shrinks_to_leave_scenes_room():
    # 15 s plug, min scene time 6.75: cards 5+5 leave 5, so 1.75 comes off
    p = cc.plan_card_fit(15, [5.0, 5.0], [2.0, 2.0])
    assert sum(p['targets']) == pytest.approx(8.25)
    assert p['scene_budget'] == pytest.approx(6.75) and p['shrunk']
    assert all(t >= 2.0 for t in p['targets'])


def test_plan_card_fit_never_goes_under_the_floor_and_says_so():
    p = cc.plan_card_fit(10, [4.0, 4.0], [3.5, 3.5])
    assert p['targets'] == [3.5, 3.5]
    assert p['short_by'] > 0


def test_plan_card_fit_shrinks_longer_card_more():
    p = cc.plan_card_fit(15, [6.0, 3.0], [1.0, 1.0])
    assert (6.0 - p['targets'][0]) > (3.0 - p['targets'][1])


# ------------------------------------------------------------------ script

def test_parse_vo_script_reads_times_tags_and_optional():
    lines = cc.parse_vo_script(
        "00:03 VO: Abangan ngayong gabi\n"
        "[0:07.5] Only on GMA (optional)\n"
        "@12.5s Huwag palampasin\n"
        "plain line [soft]\n")
    assert [l['at'] for l in lines] == [3.0, 7.5, 12.5, None]
    assert lines[0]['text'] == 'Abangan ngayong gabi'
    assert lines[1]['optional'] and lines[1]['text'] == 'Only on GMA'
    assert lines[3]['text'] == 'plain line'


def test_parse_vo_script_hours_minutes_seconds():
    assert cc.parse_vo_script('01:02:03 hello')[0]['at'] == 3723.0


def test_strip_vo_markers_leaves_spoken_words():
    assert cc.strip_vo_markers('00:03 VO: Hello there\n@5 More') == 'Hello there More'


def _w(word, s, e):
    return {'word': word, 'start': s, 'end': e}


# "abangan ngayong gabi" .. "only on gma" .. "huwag palampasin"
VO_WORDS = [_w('Abangan', 0.5, 1.0), _w('ngayong', 1.05, 1.6), _w('gabi.', 1.65, 2.1),
            _w('Only', 3.0, 3.3), _w('on', 3.35, 3.5), _w('GMA.', 3.55, 4.1),
            _w('Huwag', 5.5, 5.9), _w('palampasin!', 5.95, 6.8)]


def test_align_script_finds_lines_in_order():
    lines = cc.parse_vo_script('Abangan ngayong gabi\nOnly on GMA\nHuwag palampasin')
    al = cc.align_script(lines, VO_WORDS)
    assert [(a['w0'], a['w1']) for a in al] == [(0, 2), (3, 5), (6, 7)]


def test_align_script_tolerates_a_misheard_word():
    words = list(VO_WORDS)
    words[1] = _w('ngayon', 1.05, 1.6)
    lines = cc.parse_vo_script('Abangan ngayong gabi')
    al = cc.align_script(lines, words)
    assert (al[0]['w0'], al[0]['w1']) == (0, 2)


def test_align_script_flags_a_line_that_was_not_recorded():
    lines = cc.parse_vo_script('Abangan ngayong gabi\nSomething never said\nHuwag palampasin')
    al = cc.align_script(lines, VO_WORDS)
    assert al[1]['w0'] is None and al[2]['w0'] == 6


def test_plan_vo_places_each_line_at_its_in_point_cut_on_word_edges():
    lines = cc.parse_vo_script('00:02 Abangan ngayong gabi\n00:06 Only on GMA\n00:10 Huwag palampasin')
    plan = cc.plan_vo(lines, VO_WORDS, window_end=15)
    pcs = plan['pieces']
    assert [round(p['at']) for p in pcs] == [2, 6, 10] and plan['fits']
    for p in pcs:
        assert cc.classify_cut('in', p['a'], [(w['start'], w['end']) for w in VO_WORDS]) != 'clipped'
        assert cc.classify_cut('out', p['b'], [(w['start'], w['end']) for w in VO_WORDS]) != 'clipped'
    # line 2 is cut out of the middle of the recording, not from its start
    assert 2.5 < pcs[1]['a'] < 3.0 and 4.1 <= pcs[1]['b'] < 5.5


def test_plan_vo_untimed_lines_follow_each_other():
    lines = cc.parse_vo_script('Abangan ngayong gabi\nOnly on GMA')
    plan = cc.plan_vo(lines, VO_WORDS, start_at=1.0, window_end=30)
    a, b = plan['pieces']
    assert a['at'] == pytest.approx(1.0)
    # the pause the speaker left between the lines is kept
    assert b['at'] == pytest.approx(a['at'] + (a['b'] - a['a']) + (b['a'] - a['b']), abs=0.01)


def test_plan_vo_tightens_pauses_before_anything_else():
    lines = cc.parse_vo_script('Abangan ngayong gabi\nOnly on GMA\nHuwag palampasin')
    natural = cc.plan_vo(lines, VO_WORDS, window_end=60)['end']
    plan = cc.plan_vo(lines, VO_WORDS, window_end=natural - 0.2)
    assert plan['fits']
    assert all(p['tempo'] == 1.0 for p in plan['pieces'])
    assert any('Pauses' in n for n in plan['notes'])


def test_plan_vo_speeds_up_only_a_little():
    lines = cc.parse_vo_script('Abangan ngayong gabi\nOnly on GMA\nHuwag palampasin')
    tight = cc.plan_vo(lines, VO_WORDS, window_end=60)
    # find the length with pauses squeezed, then ask for 5 % less
    squeezed = cc.plan_vo(lines, VO_WORDS, window_end=0.1)  # forces everything
    assert all(p['tempo'] <= cc.MAX_TEMPO + 1e-6 for p in squeezed['pieces'])
    assert tight['fits']


def test_plan_vo_drops_optional_lines_before_cutting_words():
    lines = cc.parse_vo_script('Abangan ngayong gabi\nOnly on GMA (optional)\nHuwag palampasin')
    plan = cc.plan_vo(lines, VO_WORDS, start_at=0.0, window_end=4.2)
    statuses = [l['status'] for l in plan['lines']]
    assert statuses[1] == 'dropped'
    assert plan['fits']
    assert statuses[0] != 'trimmed' and statuses[2] != 'trimmed'


def test_plan_vo_last_resort_trims_on_a_word_edge():
    lines = cc.parse_vo_script('Abangan ngayong gabi')
    plan = cc.plan_vo(lines, VO_WORDS, start_at=0.0, window_end=1.3)
    p = plan['pieces'][-1]
    ends = [(w['start'], w['end']) for w in VO_WORDS]
    assert cc.classify_cut('out', p['b'], ends) != 'clipped'
    assert plan['lines'][0]['status'] == 'trimmed'
    assert plan['end'] <= 1.3 + 0.02


def test_plan_vo_overlapping_line_is_fitted_before_the_next_in_point():
    # line 1 would run 2.0 -> 4.2 but line 2 must start at 3.0
    lines = cc.parse_vo_script('00:02 Abangan ngayong gabi\n00:03.2 Only on GMA')
    plan = cc.plan_vo(lines, VO_WORDS, window_end=30)
    a, b = plan['pieces'][0], plan['pieces'][-1]
    a_end = a['at'] + (a['b'] - a['a']) / a['tempo']
    assert a_end <= b['at'] + 0.001


def test_plan_vo_reports_a_missing_line():
    lines = cc.parse_vo_script('Abangan ngayong gabi\nNever recorded sentence here')
    plan = cc.plan_vo(lines, VO_WORDS, window_end=30)
    assert plan['lines'][1]['status'] == 'missing'
    assert any('not found' in n for n in plan['notes'])


def test_plan_vo_splits_long_pauses_inside_a_line():
    words = [_w('one', 0.0, 0.4), _w('two', 0.45, 0.9), _w('three', 3.0, 3.4), _w('four', 3.45, 3.9)]
    lines = cc.parse_vo_script('one two three four')
    long = cc.plan_vo(lines, words, window_end=60)
    short = cc.plan_vo(lines, words, window_end=3.2)
    assert len(long['pieces']) == 2                       # the 2 s pause is its own gap
    assert short['fits'] and short['end'] <= 3.22
    assert all(p['tempo'] <= cc.MAX_TEMPO + 1e-6 for p in short['pieces'])


def test_words_from_segments_gives_coarse_word_times():
    w = cc.words_from_segments([{'start': 1.0, 'end': 3.0, 'text': 'one two three four'}])
    assert len(w) == 4 and w[0]['start'] == 1.0 and w[-1]['end'] <= 3.0


# ------------------------------------------------------------------ audio

@pytest.mark.skipif(shutil.which('ffmpeg') is None, reason='ffmpeg not available')
def test_vo_filter_builds_audio_of_the_planned_length(tmp_path):
    src = tmp_path / 'vo.wav'
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'lavfi', '-i', 'sine=frequency=440:duration=8',
                    '-ac', '1', '-ar', '44100', str(src)], check=True)
    pieces = [{'a': 0.5, 'b': 2.0, 'at': 1.0, 'tempo': 1.0, 'fade_in': 0.03, 'fade_out': 0.03, 'line': 0},
              {'a': 3.0, 'b': 4.0, 'at': 4.0, 'tempo': 1.08, 'fade_in': 0.03, 'fade_out': 0.03, 'line': 1}]
    graph, label = cc.vo_filter(pieces)
    out = tmp_path / 'out.wav'
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(src), '-filter_complex', graph,
                    '-map', label, str(out)], check=True)
    r = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of',
                        'default=nw=1:nk=1', str(out)], capture_output=True, text=True, check=True)
    expected = 4.0 + 1.0 / 1.08
    assert float(r.stdout) == pytest.approx(expected, abs=0.08)


def test_vo_filter_single_piece():
    g, label = cc.vo_filter([{'a': 0, 'b': 1, 'at': 0.0, 'tempo': 1.0, 'line': 0}])
    assert '[0:a]acopy[src0]' in g and label == '[vo]'
    assert cc.vo_filter([]) == (None, None)


def test_hairline_gap_between_words_moves_to_the_nearest_real_pause():
    # words 10.5 -> 10.55 are only 50 ms apart; the real pause is before 11.9
    t = cc.settle_in(10.7, WORDS, lo=9.0, sentences=[])
    assert t < 10.55
    words = [(0.0, 0.5), (0.52, 1.0), (1.02, 1.5), (1.9, 2.4)]
    # in-point inside word 2, whose gap before is 20 ms: use the pause before word 4? too far; first word instead
    t = cc.settle_in(0.7, words, lo=-1.0, sentences=[])
    assert t < 0.0 + 0.001
    o = cc.settle_out(1.0, words, lo=0.5, hi=3.0)
    assert o > 1.5                                   # the next real pause is after word 3

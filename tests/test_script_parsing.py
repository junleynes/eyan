"""
Tests for script/rundown parsing (parse_script_cues) and the scoring boost it
feeds into scene selection (apply_script_priority).
"""
import pipeline


class TestParseScriptCues:
    def test_hh_mm_ss_format(self):
        cues = pipeline.parse_script_cues("00:01:30 The confrontation scene")
        assert len(cues) == 1
        assert cues[0]['time'] == 90.0
        assert 'confrontation' in cues[0]['desc']

    def test_mm_ss_format(self):
        cues = pipeline.parse_script_cues("1:30 Short form timecode")
        assert len(cues) == 1
        assert cues[0]['time'] == 90.0

    def test_smpte_frames_format(self):
        # 00:01:30:12 at 25fps -> 90s + 12/25 = 90.48s
        cues = pipeline.parse_script_cues("00:01:30:12 Frame-accurate cue", fps=25.0)
        assert len(cues) == 1
        assert abs(cues[0]['time'] - 90.48) < 1e-9

    def test_dotted_milliseconds_not_confused_with_frames(self):
        # 00:01:30.500 is 500ms, NOT frame 500 -- the whole reason this
        # distinction exists in the parser (mixing them up puts a cue up to
        # a second off).
        cues = pipeline.parse_script_cues("00:01:30.500 Milliseconds not frames")
        assert len(cues) == 1
        assert abs(cues[0]['time'] - 90.5) < 1e-9

    def test_lines_without_timecodes_are_ignored(self):
        text = "RUNDOWN\nSome header text\nJust prose, no timecode here\n"
        assert pipeline.parse_script_cues(text) == []

    def test_multiple_cues_sorted_by_time(self):
        text = "00:02:00 Second cue\n00:00:30 First cue\n00:01:00 Middle cue\n"
        cues = pipeline.parse_script_cues(text)
        assert [c['time'] for c in cues] == [30.0, 60.0, 120.0]

    def test_empty_text_returns_empty_list(self):
        assert pipeline.parse_script_cues("") == []
        assert pipeline.parse_script_cues(None) == []

    def test_description_excludes_the_timecode_itself(self):
        cues = pipeline.parse_script_cues("00:01:30 - Kitchen confrontation")
        assert '00:01:30' not in cues[0]['desc']
        assert 'Kitchen confrontation' in cues[0]['desc']


class TestClockTimeVsTimecode:
    """A real, user-reported bug: a rundown's own schedule/promo text (e.g.
    "GMA 8:50 PM", "SECOND TELECAST GTV 10:30 PM") sat in the same part of
    the document as real M1/M2 cues -- not a column-mixing problem, this
    text is genuinely positioned alongside real cues in the source PDF --
    and _TC_PATTERNS' own MM:SS pattern had no way to tell "8:50" (a
    wall-clock time slot) apart from a real MM:SS video timecode, so it
    was accepted as a spurious scene-selection cue. A real video timecode
    is never followed by AM/PM; a wall-clock time slot always is when
    written this way, which is what _CLOCK_TIME_SUFFIX_RE checks for."""

    def test_pm_suffixed_time_is_not_a_timecode(self):
        assert pipeline._parse_timecode_line("GMA 8:50 PM") is None
        assert pipeline._parse_timecode_line("SECOND TELECAST GTV 10:30 PM") is None

    def test_am_suffixed_time_is_not_a_timecode(self):
        assert pipeline._parse_timecode_line("Airs weekdays 6:00 AM") is None

    def test_lowercase_and_dotted_am_pm_are_also_rejected(self):
        assert pipeline._parse_timecode_line("Rerun at 9:15 pm") is None
        assert pipeline._parse_timecode_line("Morning block 7:00 a.m.") is None
        assert pipeline._parse_timecode_line("Evening slot 7:00 P.M.") is None

    def test_real_timecode_on_the_same_line_as_a_clock_time_mention_is_still_kept(self):
        # A real cue and an incidental wall-clock mention can share one
        # line -- only the clock-time match should be rejected, not the
        # whole line.
        result = pipeline._parse_timecode_line("M1 3:33 airs at 8:50 PM tonight")
        assert result is not None
        in_secs, desc, meta = result
        assert in_secs == 213.0  # 3:33 == 3*60+33

    def test_real_rundown_lines_from_the_actual_reported_document(self):
        # The exact lines from the user's own script: 5 genuine M1/M2 cues
        # plus the two schedule-text lines that were being misread as cues.
        text = "\n".join([
            "M1 3:33\u201437 HABOL TACKLE",
            "M2 00:35\u201442 BUNOT",
            "GMA 8:50 PM",
            "SECOND TELECAST GTV 10:30 PM",
            "M1 3:38\u201443",
            "M2 00:12\u201420",
            "M2 00:43\u201446",
        ])
        cues = pipeline.parse_script_cues(text, available_materials={1, 2})
        assert len(cues) == 5
        times = sorted(c['time'] for c in cues)
        # M1 3:33 (213s), M2 0:35 (35s), M1 3:38 (218s), M2 0:12 (12s), M2 0:43 (43s)
        assert times == [12.0, 35.0, 43.0, 213.0, 218.0]
        # The other, equally real assertion: neither spurious schedule-text
        # line produced a cue at all.
        descs = [c['desc'] for c in cues]
        assert not any('GMA' in d or 'TELECAST' in d or 'PRIME' in d for d in descs)


class TestApplyScriptPriority:
    def test_boosts_scene_matching_a_cue_timecode(self):
        scenes = [
            {'start': 0.0, 'duration': 4.0, 'total_score': 5.0},
            {'start': 10.0, 'duration': 4.0, 'total_score': 3.0},  # this one gets the cue
            {'start': 30.0, 'duration': 4.0, 'total_score': 5.0},
        ]
        cues = [{'time': 12.0, 'desc': 'the reveal'}]
        matched = pipeline.apply_script_priority(scenes, cues)
        assert matched == 1
        # The cue-matched scene's score should now exceed the higher-scoring
        # untouched scenes -- that's the entire point of the boost (an
        # explicit script call should generally win over automatic scoring).
        assert scenes[1]['total_score'] > scenes[0]['total_score']
        assert scenes[1]['total_score'] > scenes[2]['total_score']

    def test_scenes_outside_window_are_not_boosted(self):
        scenes = [{'start': 0.0, 'duration': 4.0, 'total_score': 5.0}]
        cues = [{'time': 100.0, 'desc': 'far away cue'}]
        matched = pipeline.apply_script_priority(scenes, cues)
        assert matched == 0
        assert scenes[0]['total_score'] == 5.0

    def test_no_cues_boosts_nothing(self):
        scenes = [{'start': 0.0, 'duration': 4.0, 'total_score': 5.0}]
        matched = pipeline.apply_script_priority(scenes, [])
        assert matched == 0
        assert scenes[0]['total_score'] == 5.0


class TestSegmentOffsets:
    """Real-world scripts for multi-part episode footage commonly label
    each raw plug file "M1", "M2", etc., each timecoded from its own zero
    -- but combining those files into one source video (Browse Library's
    multi-select combine) means the render only ever sees a single
    timeline. These cover the offset logic that makes a script written
    against separate files still line up correctly once combined."""

    def test_m1_defaults_to_offset_zero_with_no_segment_info_at_all(self):
        # The common "only one material this week" case -- a script still
        # labelled M1 needs no adjustment when nothing was combined.
        cues = pipeline.parse_script_cues("M1 0:05 only one file this week")
        assert len(cues) == 1
        assert cues[0]['time'] == 5.0

    def test_m2_with_no_known_offset_is_dropped_not_guessed(self):
        # M2 named but segment_offsets has no entry for it (no combine
        # happened, or fewer than 2 files were combined) -- must be
        # skipped rather than treated as offset 0, which would silently
        # pin the wrong scene.
        cues = pipeline.parse_script_cues("M2 0:05 second file that doesn't exist here")
        assert cues == []

    def test_m2_offset_by_the_correct_cumulative_duration(self):
        cues = pipeline.parse_script_cues("M2 0:12 twelve seconds into the second file",
                                           segment_offsets={2: 240.0})
        assert len(cues) == 1
        assert cues[0]['time'] == 252.0

    def test_m1_and_m2_cues_together_real_script_shape(self):
        # Mirrors the actual reported script's shape: M1 cues unaffected,
        # M2 cues correctly offset, in the same document.
        text = "M1 3:33 first file cue\nM2 0:12 second file cue"
        cues = pipeline.parse_script_cues(text, segment_offsets={2: 240.0})
        assert len(cues) == 2
        times = sorted(c['time'] for c in cues)
        assert times == [213.0, 252.0]

    def test_m3_offset_when_three_segments_combined(self):
        # Not just a two-file special case -- offsets stack correctly for
        # a third (or later) segment too.
        cues = pipeline.parse_script_cues("M3 0:05 third file",
                                           segment_offsets={2: 100.0, 3: 180.0})
        assert len(cues) == 1
        assert cues[0]['time'] == 185.0

    def test_line_without_any_segment_prefix_is_unaffected_by_offsets(self):
        # A script that never uses the M1/M2 convention at all must behave
        # exactly as before, even when segment_offsets is supplied (e.g. a
        # combine happened, but this particular script wasn't written
        # against separate files).
        cues = pipeline.parse_script_cues("00:01:30 no segment prefix here",
                                           segment_offsets={2: 240.0})
        assert len(cues) == 1
        assert cues[0]['time'] == 90.0

    def test_segment_prefix_does_not_get_confused_with_the_timecode_itself(self):
        # M1/M2 must not be mistaken for part of the timecode pattern, and
        # the timecode must still be found correctly alongside it.
        cues = pipeline.parse_script_cues("M1 01:02:03 fully qualified timecode")
        assert len(cues) == 1
        assert cues[0]['time'] == 3723.0

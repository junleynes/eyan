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

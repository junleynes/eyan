# Tests

```
pip install pytest --break-system-packages
pytest
```

Covers the pure/near-pure logic that's easy to silently get wrong and
expensive to catch by hand:

- `test_speech_safety.py` — the cut-point snapping (`nearest_word_boundary`,
  `nearest_speech_out`) and the exact-duration corrector's speech guard
  (`speech_free_slack`). Two real bugs in this area were only caught by
  manual, ad-hoc testing during development; these are the regression tests
  that should have caught them the first time.
- `test_script_parsing.py` — script/rundown timecode parsing (all four
  supported formats: `HH:MM:SS`, `MM:SS`, SMPTE frames, dotted milliseconds)
  and the scoring boost it feeds into scene selection.
- `test_production_defaults.py` — the Config > Production settings
  load/save round-trip, including that malformed values are rejected
  rather than silently corrupting the deployment-wide delivery spec.
- `test_auth.py` — password policy, the TOTP enrollment lifecycle (not
  enabled until confirmed, backup codes genuinely single-use), and
  per-account lockout. Uses a real throwaway SQLite DB per test (see the
  `users_db` fixture in `conftest.py`), not mocks, since this logic
  actually depends on the database round-trip.

- `test_shorts_core.py` — the Vertical Shorts decisions with no services
  and no ffmpeg: story/vision reply parsing, fitting a moment to the length
  limits (a too-long one loses its opening, never its ending), in/out
  placement that never lands inside speech, scoring, de-duplication,
  shot-by-shot reframing, and captions.
- `test_shorts.py` — the Vertical Shorts tab end to end against real
  PySceneDetect/OpenCV/ffmpeg on generated video, with only Ollama and
  faster-whisper stood in for. Decodes the rendered shorts to check that
  each shot's crop switches on exactly the frame the picture does, that SD
  sources are converted to BT.709, and that captions are burned in.

Not covered yet: the full render pipeline (scene detection through ffmpeg
assembly) isn't unit-testable as-is — it's one large function mixing I/O,
subprocess calls, and business logic. Extracting the pure decision-making
parts of it (which scenes get selected, how duration correction picks what
to trim) into testable functions the way `speech_free_slack` already is
would be the natural next step, not a rewrite of what's here.

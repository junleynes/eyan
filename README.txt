PRISM update - scene cutting / gap-filling fix
================================================

Apply on top of the previous update package.

  pipeline.py                                        -> repo root (overwrite)
  tests/test_shortfall_topup_from_unused_scenes.py   -> add (new file)

What changed: the fully-automatic "Generate promo plug" path now has a
last-resort step to pull in extra, closely-spaced scenes when the normal
selection can't otherwise reach the target duration (mirrors what the
manual Preview-the-cut drop/add flow already did). No UI changes.

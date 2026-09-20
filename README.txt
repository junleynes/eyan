PRISM update package
=====================

Your GitHub main already has the destination-config dropdown + checkbox
removal work (looks like you applied the earlier zip already - thank you).

This package adds everything from this session on top of that. Paths in
this zip match the repo layout - unzip at the repo root and these files
land in the right place directly:

  pipeline.py                          -> repo root (overwrite)
  templates/index.html                 -> overwrite
  tests/test_script_parsing.py         -> overwrite
  tests/test_scene_search_endpoint.py  -> add (new file)

What changed:
1. Timecode Priority now recognises period-separated timecodes
   ("2.11", "00.00.03.53", etc.) in uploaded scripts, not just colon
   ones - this is what was silently failing on 4 of your 6 sample scripts.
2. The trailer preview's "Show more clips" grid is back to how it was
   (the "Scene search" box that briefly replaced it there has been
   reverted, per your feedback).
3. The standalone "Scene Detection & Analysis" tool (left rail) is now
   "Scene Search": pick a hires video, type what scene you want (plus an
   optional negative prompt to exclude some), and it detects every cut,
   describes each with AI Vision, aligns Whisper dialogue, and shows
   matching scenes as a thumbnail grid you can preview. No rating/scoring
   happens here - just detection + vision + whisper, as you asked.

DELETE this file from your repo if it exists (the old scene-search-in-
preview feature it tested no longer exists after the revert):
  tests/test_scene_search_pool.py

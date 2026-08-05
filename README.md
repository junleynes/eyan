<p align="center"><img src="static/logo.svg" alt="EYAN — Engine For Your AI Needs" width="480"></p>

# EYAN — Engine For Your AI Needs

Formerly AIMP. Split out of the original single `main_app_57_1_.py`
(~9,500 lines). Same behavior, same env vars (`LIBRARY_DIR`, `SECRET_KEY_FILE`,
`ADMIN_USERNAME`, `ADMIN_PASSWORD`, `RESET_ADMIN`, `TEMPLATES_DIR`,
`FISH_AUDIO_URL`, etc. — see `.env.example` for the full list), same port
(5000).

## Install

```
# Linux
./install.sh              # add --with-ffmpeg to also install ffmpeg via apt/dnf/yum/pacman/zypper

# Windows (PowerShell)
.\install.ps1              # add -WithFFmpeg to also install ffmpeg via winget
```

Either script: checks for Python 3.10+ (numpy 2.x, pinned in
`requirements.txt`, requires it), checks for `ffmpeg`/`ffprobe` on PATH,
creates a `venv/` and installs `requirements.txt` into it, and copies
`.env.example` to `.env` if you don't already have one. Safe to re-run —
it reuses an existing `venv/` and never touches an existing `.env`.

Then:

```
# Linux
source venv/bin/activate
python3 main.py

# Windows
venv\Scripts\Activate.ps1
python main.py
```

**`.env` is real now** — as of this pass, `core.py` loads it via
`python-dotenv` before anything else runs, so variables in `.env` actually
take effect (this used to be silently decorative). One thing worth knowing
if you hand-edit it: every line in `.env.example` is commented out on
purpose — uncommenting a line with nothing after the `=` sets an actual
empty string, which is *not* the same as leaving the variable unset, and
some of the app's own defaults only apply when a variable is completely
absent. Uncomment a line only when you're also giving it a real value.

## Layout

| File | What's in it |
|---|---|
| `core.py` | Flask app instance, session cookie config, the request-level access gate (login required, admin-only `/admin/*`), rate limiters. Everything else imports `app` from here. |
| `auth.py` | User accounts (SQLite `users.db`), `/login`, `/logout`, `/admin/users` management page and its routes. |
| `library_db.py` | The persistent trailer library (SQLite `library.db`) — saved trailers, metadata, cleanup. |
| `pipeline.py` | Everything else: scene detection, AI vision scoring, TTS/STT, music/SFX generation, the ffmpeg render pipeline, per-show asset templates, network (SMB) browsing, the Config tab, and all `/api/*` routes. ~5,000 lines — see below for why this one wasn't split further. |
| `templates/index.html` | The UI shell (was a 3,900-line Python string assigned to `UI` and passed through `render_template_string`; now a real template file loaded via `render_template`). |
| `main.py` | Entrypoint — imports the other modules (registering their routes), defines the last few top-level routes (`/`, `/uploads/<filename>`, `/download/<filename>`), and starts the server. |
| `static/logo.svg`, `static/logo-mark.svg` | The EYAN logo (full lockup and icon-only mark). Served automatically by Flask's default `/static/` handling since this folder sits next to `core.py`. |
| `install.sh`, `install.ps1` | One-time setup: Python/ffmpeg checks, venv + `requirements.txt`, `.env` from `.env.example`. See Install above. |
| `requirements.txt`, `.env.example` | Every third-party dependency, and every environment variable the app reads with sensible defaults documented (all commented out — see the note above about why). |

## Scene detection fix (this pass)

You reported wrong/inconsistent cuts. Found and fixed two real bugs in
`detect_scenes()` and its callers in `pipeline.py`:

1. **Downscale was silently never applying.** The code called
   `video.set_downscale_factor(downscale)`, which doesn't exist on the
   installed PySceneDetect/backend combo — it was wrapped in a bare
   `try/except: pass`, so it failed silently every time. `SceneManager` also
   defaults to auto-picking its own downscale factor (`auto_downscale=True`),
   which overrides an explicit one unless turned off first. Fixed to set
   `sm.auto_downscale = False; sm.downscale = downscale` instead. Verified
   against a synthetic test clip: before the fix, PySceneDetect printed
   `Downscale factor will be ignored because auto_downscale=True!`; after,
   it doesn't, and detection still finds the right cuts.
2. **The "Scene Detection & Analysis" tab hardcoded `threshold=30.0`**,
   ignoring whatever you'd tuned — so it silently stopped matching what
   generation would actually do the moment you changed the threshold
   anywhere else. It now reads `scene_threshold`, `min_scene_len`,
   `detector`, and `adaptive_threshold` from the request like every other
   caller, via a new shared `_scene_detector_params()` parser used by the
   preview endpoint, this tab, and the real render job — so all three are
   guaranteed to agree instead of three separate hand-copied defaults
   drifting apart over time.

**New: an Adaptive detector option**, alongside the existing Content
detector, in both the generator form and the Scene Detection & Analysis
tab. Content (PySceneDetect's `ContentDetector`) compares each frame to a
fixed threshold, so fast pans/handheld/zooms can look like cuts. Adaptive
(`AdaptiveDetector`) compares each frame to a local rolling average
instead, so sustained camera movement doesn't trip it the way a real cut
does. Neither reliably catches slow cross-dissolves between two scenes —
that's a fundamentally different detection problem (no single frame has a
large jump to key on); Adaptive's lower effective noise floor tends to
catch more of a dissolve's ramp than Content does, but it's not a
dedicated fix. Existing templates/saved settings are unaffected — Content
stays the default, so nothing changes unless you switch it.

Tested with a synthetic 3-shot hard-cut clip (ffmpeg-generated, no real
footage needed for this part): both detectors correctly find all 3 shots
through the actual HTTP endpoint, no errors in the server log. What I
couldn't test: real footage with actual pans/handheld motion or dissolves,
since there's none in this sandbox — the Adaptive option is a real,
verified-functional PySceneDetect algorithm, but how well it performs on
*your* footage compared to Content is something to judge from the Scene
Detection & Analysis tab before trusting it on an air date.

## What I tested

Booted the split app fresh, confirmed: first-run admin bootstrap, login,
session cookie, the index page rendering from the extracted template
(matches original title/size), `/admin/users`, unauthenticated `/api/*`
correctly getting a 401, and `/api/health` returning valid JSON. No errors
in the server log across several runs.

What I *couldn't* exercise here: an actual trailer render (needs real
footage, ffmpeg, and reachable Ollama/Fish Audio/Whisper/SMB endpoints —
none of which exist in this sandbox). The render pipeline's control flow
wasn't touched — it's the same code, just relocated into `pipeline.py` —
but you'll want to run one real render on your server before trusting this
in production, the same way you would after any refactor.

## Why `pipeline.py` is still one big file

A few pieces of state in there are deliberately shared via Python's
`global` keyword so a nested closure deep in one route (e.g. the AI
scoring loop, or the Config tab saving a new Fish Audio URL) can update a
value that a *different* route reads on its next call — that's how the
Config tab's live URL overrides and the "disable structured output after
one failure" fallback work. Splitting the code that reads those values
into a different file than the code that sets them would require rewriting
every read site to a qualified `module.attr` lookup instead of a bare name,
which is a real code change, not just a move — and I can't fully verify
that kind of change without a real render to test against.

If you want to keep going, the natural next slices out of `pipeline.py`
(each testable independently against a real render) are:
- **Asset templates** (per-show bundles + their `/api/` routes) — fairly
  self-contained already.
- **AI services** (Fish Audio TTS, Whisper, Ollama Vision, Woosh, ACE-Step,
  and the Config tab that live-reloads their URLs) as one module, kept
  together specifically because of the `global`-shared URL state above.
- **Trailer generator routes** (the ffmpeg render pipeline itself) —
  the biggest remaining chunk, best split last once the others are out.

I'd do each of those as its own pass with a real render in between, rather
than all at once.

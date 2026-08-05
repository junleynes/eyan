<p align="center"><img src="static/logo.svg" alt="AIMP — AI Media Provider" width="480"></p>

# AIMP — AI Media Provider

A self-hosted toolkit for producing broadcast episodic promos ("plugs")
end to end — from raw footage to a finished, mixed trailer — plus the
individual AI tools behind it, usable on their own. Formerly AIMP.

## What it does

- **Promo plug generator** — the core workflow. Drop in an episode (or
  pick one off a network SMB share), and it detects every scene cut
  (PySceneDetect, with a choice of Content or motion-resistant Adaptive
  detection), scores each scene for quality (sharpness/brightness/faces)
  and AI vision content, then assembles a trailer at your target length
  with music, SFX, narration, and title/end cards mixed in via ffmpeg.
  **Preview the cut first** before committing to a full render — see
  and play back the exact scenes it picked, swap in alternates, then
  reuse that same analysis for the final render instead of repeating it.
- **Music generation** (ACE-Step) — prompt-driven original music, sung
  or instrumental, with control over duration, tempo, key, time
  signature, negative styles, an LM "thinking" planning pass, and
  audio2audio restyling from a reference track.
- **Text to SFX** (Woosh) and **Text to speech** (Fish Audio, voice
  cloning from a reference sample) — generate one-off sound effects and
  narration outside the main generator, for building a reusable library.
- **Speech to text** (Whisper) — transcribe audio/video, used internally
  for beat-syncing narration and available as a standalone tool.
- **Scene detection & analysis** — the same detector the generator uses,
  as a standalone preview/diagnostic tool with AI vision commentary per
  scene and playback of any detected cut.
- **AI chat** — an LLM assistant tab with file attachment support, for
  drafting promo copy or asking questions about a script/document.
- **Shared player** — plays anything on the configured network share or
  already in the trailer library, with automatic format conversion for
  browser playback (ProRes/DNxHD masters and the like).
- **Per-show templates** — save a full generator configuration (genre,
  transition, lengths, audio targets, voice, plus the music/SFX/VO/cards
  themselves) under a show name; each new episode is one dropdown away.
- **Saved trailer library** — every render is kept, browsable and
  re-downloadable, independent of the source footage's lifecycle.
- **Per-user accounts** — real login (not a shared passphrase), with an
  admin role for account management (`/admin/users`) and audit-lite
  tracking (last login per account). Optional groups can restrict a
  regular account to only specific tabs (e.g. a group that can only
  reach Music Generation); an account with no group assigned is
  unrestricted, so this is opt-in per account, not a default lockdown.
- **Configurable branding** — name, tagline, logo, and favicon are all
  editable at runtime from Config &gt; Branding (admin-only), not
  hardcoded — rebranding the whole app (like this project's own
  EYAN&nbsp;→&nbsp;AIMP rename) doesn't require touching a single file.

Same behavior as the original as of this split, same env vars
(`LIBRARY_DIR`, `SECRET_KEY_FILE`, `ADMIN_USERNAME`, `ADMIN_PASSWORD`,
`RESET_ADMIN`, `TEMPLATES_DIR`, `FISH_AUDIO_URL`, etc. — see
`.env.example` for the full list), same port (5000).

## Install

### Quick install (one line)

Fetches the repo and installs everything in one shot — nothing else to
download first:

```bash
# Linux / macOS
curl -fsSL https://raw.githubusercontent.com/junleynes/eyan/main/install.sh | bash

# with the ffmpeg auto-install flag:
curl -fsSL https://raw.githubusercontent.com/junleynes/eyan/main/install.sh | bash -s -- --with-ffmpeg
```

```powershell
# Windows (PowerShell)
irm https://raw.githubusercontent.com/junleynes/eyan/main/install.ps1 | iex

# with the ffmpeg auto-install flag (switches can't be passed through
# a piped iex, so this uses an env var instead):
$env:EYAN_WITH_FFMPEG = '1'; irm https://raw.githubusercontent.com/junleynes/eyan/main/install.ps1 | iex
```

This clones the repo into a new `eyan/` folder in whatever directory
you run it from, then runs the full install described below from
inside it. `cd eyan` afterward to run the app (see "3. Run it").

If you'd rather review a script before running it (reasonable instinct
for anything piped into a shell), skip to "Manual install, step by step"
below and run `./install.sh` / `.\install.ps1` locally instead — same
script, same result, just no piping.

### Prerequisites

- **Python 3.10 or newer** — `numpy` 2.x (pinned in `requirements.txt`) requires it.
  - Windows: [python.org/downloads](https://www.python.org/downloads/) or `winget install Python.Python.3.12`. During setup, tick **"Add python.exe to PATH."**
  - Linux: usually already installed (`python3 --version` to check); if not, your distro's package manager (`sudo apt install python3`, etc.)
- **ffmpeg and ffprobe** — the render pipeline shells out to both directly; they're not something `pip` can install. The install scripts below check for these and can install them for you (`--with-ffmpeg` / `-WithFFmpeg`), or tell you how if you'd rather do it yourself.
- **Git**, or just a downloaded copy of this repo — either works, you only need the files on disk. (The quick install above clones for you if `git` is available, or falls back to downloading a ZIP/tarball if it isn't.)

### Manual install, step by step

### 1. Get the code

```
git clone https://github.com/junleynes/eyan.git
cd eyan
```

(Or download the repo as a ZIP from GitHub and extract it — the install scripts don't care how the files got there, only that they're run from inside this folder.)

### 2. Run the install script

**Linux / macOS:**

```
chmod +x install.sh        # only needed once, if it isn't already executable
./install.sh
# or, to also install ffmpeg automatically via apt/dnf/yum/pacman/zypper:
./install.sh --with-ffmpeg
```

**Windows (PowerShell):**

```powershell
.\install.ps1
# or, to also install ffmpeg automatically via winget:
.\install.ps1 -WithFFmpeg
```

If Windows refuses to run the script with a message about execution
policy, that's PowerShell's default safety setting blocking unsigned
scripts — allow it for your own account with:

```powershell
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```

then run `.\install.ps1` again. This only needs doing once per machine.

Either script does the same four things, and is safe to re-run any time
(it reuses an existing `venv/` and never touches an existing `.env`):

1. Checks for Python 3.10+ and for `ffmpeg`/`ffprobe` on PATH.
2. Creates a `venv/` virtual environment.
3. Installs everything in `requirements.txt` into it.
4. Copies `.env.example` to `.env`, if you don't already have one.

You'll see `ok` next to each check that passes, and clear instructions
printed at the end regardless of outcome — including what to do if
`ffmpeg` wasn't found and you didn't pass the auto-install flag.

### 3. Run it

```
# Linux
source venv/bin/activate
python3 main.py

# Windows
venv\Scripts\Activate.ps1
python main.py
```

Then open **http://localhost:5000**. First run with no existing
`users.db` creates a default admin account and prints its username and
password once to the console — watch for it, or set
`ADMIN_USERNAME`/`ADMIN_PASSWORD` in `.env` beforehand to choose your own
(see "Lost the admin password?" below for recovery if you ever need it).

**This serves production traffic by default now** — `python3 main.py`
runs under [waitress](https://github.com/Pylons/waitress), not Flask's
built-in dev server. Specifically single-process, multi-threaded
(`WAITRESS_THREADS`, default 8) rather than multi-process: job/preview
state lives in plain in-memory dicts, not a shared store, so a
multi-process server would give each worker its own separate copy and a
job could silently vanish from progress polling if it landed on a
different worker than the one that started it. Set `DEV_SERVER=1` to
fall back to the Flask dev server for local troubleshooting — don't use
that for anything real traffic can reach.

### Lost the admin password?

Set `RESET_ADMIN=1` in `.env` (optionally alongside `ADMIN_USERNAME`/
`ADMIN_PASSWORD` to choose the new credentials) and restart. This resets
that account to an active admin with a new password on every boot —
regardless of what's already in `users.db` — and creates it if it
doesn't exist. **Remove `RESET_ADMIN` again afterward**, or the account
keeps getting reset back to that password on every future restart.

### `.env` — real now, one gotcha to know about

As of this pass, `core.py` loads `.env` via `python-dotenv` before
anything else runs, so variables in it actually take effect (this used
to be silently decorative — nothing read the file at all). One thing
worth knowing if you hand-edit it: every line in `.env.example` is
commented out on purpose — uncommenting a line with nothing after the
`=` sets an actual empty string, which is *not* the same as leaving the
variable unset, and some of the app's own defaults only apply when a
variable is completely absent. Uncomment a line only when you're also
giving it a real value.

### If something goes wrong

- **`./install.sh: Permission denied`** — run `chmod +x install.sh` first, or invoke it as `bash install.sh` instead.
- **`No Python 3.10+ found on PATH`** — install a newer Python (see Prerequisites above) and make sure it's on PATH; open a new terminal afterward so the updated PATH takes effect.
- **ffmpeg warnings during install** — the app will still start without it, but any render will fail until `ffmpeg`/`ffprobe` are on PATH. Re-run with `--with-ffmpeg` / `-WithFFmpeg`, or install manually from [ffmpeg.org](https://ffmpeg.org/download.html).
- **pip install fails partway through** — usually a flaky network blip on a large package (`opencv-python`, `numpy`). Just re-run the install script; it reuses the existing `venv/` and only needs to finish installing what's missing.
- **Windows: `venv\Scripts\Activate.ps1` still blocked after `Set-ExecutionPolicy`** — you can skip activation entirely and run `venv\Scripts\python.exe main.py` directly; same effect, no execution policy involved.

## Layout

Split out of the original single `main_app_57_1_.py` (~9,500 lines) into
the modules below — see "Why `pipeline.py` is still one big file" further
down for why that split stopped where it did.

| File | What's in it |
|---|---|
| `core.py` | Flask app instance, session cookie config, the request-level access gate (login required, admin-only `/admin/*`), rate limiters. Everything else imports `app` from here. |
| `auth.py` | User accounts (SQLite `users.db`), `/login`, `/logout`, `/admin/users` management page and its routes. |
| `library_db.py` | The persistent trailer library (SQLite `library.db`) — saved trailers, metadata, cleanup. |
| `pipeline.py` | Everything else: scene detection, AI vision scoring, TTS/STT, music/SFX generation, the ffmpeg render pipeline, per-show asset templates, network (SMB) browsing, the Config tab, and all `/api/*` routes. ~5,000 lines — see below for why this one wasn't split further. |
| `templates/index.html` | The UI shell (was a 3,900-line Python string assigned to `UI` and passed through `render_template_string`; now a real template file loaded via `render_template`). |
| `main.py` | Entrypoint — imports the other modules (registering their routes), defines the last few top-level routes (`/`, `/uploads/<filename>`, `/download/<filename>`), and starts the server. |
| `static/logo.svg`, `static/logo-mark.svg` | The built-in default logo (full lockup and icon-only mark) — the actual displayed name/tagline/logo/favicon are configurable at runtime from Config &gt; Branding (admin-only), stored separately from these files. Served automatically by Flask's default `/static/` handling since this folder sits next to `core.py`. |
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

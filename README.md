<p align="center">
  <img src="static/prism-logo.svg" alt="PRISM" width="360">
</p>

# PRISM

Self-hosted toolkit for producing broadcast episodic promos end to end — from raw footage to a finished, mixed trailer — plus the individual AI tools behind it, usable on their own.

## Features

- **Promo generator** — Drop in an episode (or pick one from a network SMB share). Scene cuts are detected (PySceneDetect Content or Adaptive), scored for quality and AI vision content, then assembled into a trailer at your target length with music, SFX, narration, and title/end cards mixed via ffmpeg. Preview the cut first, swap scenes, then render without re-running analysis.
- **Music generation** (ACE-Step) — Prompt-driven original music, sung or instrumental. Control duration, tempo, key, time signature, negative styles, an LM planning pass, and audio2audio restyling from a reference track.
- **Text to SFX** (Woosh) and **Text to speech** (Fish Audio, voice cloning from a reference sample) — Build a reusable library of sound effects and narration outside the main generator.
- **Speech to text** (Whisper) — Transcribe audio/video. Used internally for beat-syncing narration and available as a standalone tool.
- **Scene detection & analysis** — Same detector as the generator, as a standalone preview with AI vision commentary per scene and playback of any cut.
- **AI chat** — LLM assistant with file attachment support for drafting promo copy or working from a script/document.
- **Shared player** — Play anything on the configured network share or in the trailer library, with automatic format conversion for browser playback.
- **Per-show templates** — Save a full generator configuration (genre, transitions, lengths, audio targets, voice, music/SFX/VO/cards) under a show name. New episodes are one dropdown away.
- **Trailer library** — Every render is kept, browsable and re-downloadable, independent of the source footage.
- **Per-user accounts** — Real login with admin role for account management and last-login tracking. Optional groups can restrict accounts to specific tabs.
- **Configurable branding** — Name, tagline, logo, and favicon editable at runtime from Config → Branding (admin-only).
- **Favorite folders** — Pin network-browser locations for quick access.
- **Narration preview & output meter** — Preview VO before render; live peak/channel meter on output.
- **Delivery formats** — Choose the delivery format up front.

## Install

**Quick install**

```bash
# Linux / macOS
curl -fsSL https://raw.githubusercontent.com/junleynes/eyan/main/install.sh | bash
# optional: also install ffmpeg
curl -fsSL https://raw.githubusercontent.com/junleynes/eyan/main/install.sh | bash -s -- --with-ffmpeg
```

```powershell
# Windows (PowerShell)
irm https://raw.githubusercontent.com/junleynes/eyan/main/install.ps1 | iex
# optional: also install ffmpeg
$env:EYAN_WITH_FFMPEG = '1'; irm https://raw.githubusercontent.com/junleynes/eyan/main/install.ps1 | iex
```

**Manual**

```bash
git clone https://github.com/junleynes/eyan.git
cd eyan
./install.sh          # or install.ps1 on Windows
# optional: ./install.sh --with-ffmpeg
```

**Requirements:** Python 3.10+, ffmpeg/ffprobe, Git (or a downloaded copy of the repo).

**Run**

```bash
# Linux / macOS
source venv/bin/activate
python3 main.py

# Windows
venv\Scripts\Activate.ps1
python main.py
```

Open **http://localhost:5000**. On first run (no `users.db`), a default admin account is created and its credentials are printed once to the console. Set `ADMIN_USERNAME` / `ADMIN_PASSWORD` in `.env` beforehand to choose them yourself.

Production traffic is served by waitress by default (single-process, multi-threaded). Set `DEV_SERVER=1` only for local troubleshooting.

**Reset admin password:** set `RESET_ADMIN=1` in `.env` (optionally with new `ADMIN_USERNAME` / `ADMIN_PASSWORD`), restart, then remove `RESET_ADMIN`.

Copy `.env.example` → `.env` and uncomment only the variables you want to override with real values. Leaving a line blank after `=` is not the same as leaving it commented out.

## Security

Treat the host like the storage it can reach: the app holds network-share credentials and hands media to ffmpeg.

- Prefer an auth layer in front of the app (e.g. Cloudflare Access, or nginx basic auth / client certs). The built-in login is a second line, not the only one.
- Behind a reverse proxy, set:

  ```bash
  TRUST_PROXY_HEADERS=1
  TRUST_PROXY_HOPS=1      # increase if you chain proxies
  FORCE_HTTPS=1           # Secure session cookie when TLS terminates at the proxy
  ```

- Sessions default to 1 day. Configure `SESSION_LIFETIME_DAYS` as needed.
- Keep `SECRET_KEY_FILE` and share/API credentials on the host private; they are stored as local config, not in the repo.

## License

See the repository for license terms.

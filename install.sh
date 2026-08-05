#!/usr/bin/env bash
# EYAN install script (Linux). Two ways to run it:
#
#   Already have the repo:
#     ./install.sh
#
#   Don't have it yet -- fetches the repo itself, then installs:
#     curl -fsSL https://raw.githubusercontent.com/junleynes/eyan/main/install.sh | bash
#     curl -fsSL https://raw.githubusercontent.com/junleynes/eyan/main/install.sh | bash -s -- --with-ffmpeg
#
# What this does:
#   0. If core.py isn't found in the current directory, clones the repo
#      into ./eyan first (via git if available, else a downloaded tarball)
#      and continues from inside it. Skipped entirely if you already have
#      the repo and ran this from inside it.
#   1. Checks for Python 3.10+ (numpy 2.x, which requirements.txt pins,
#      requires it).
#   2. Checks for ffmpeg/ffprobe on PATH -- these are external programs the
#      render pipeline shells out to, not something pip can install. Pass
#      --with-ffmpeg to have this script install them via your distro's
#      package manager (needs sudo); otherwise it just tells you how.
#   3. Creates a venv/ virtual environment and installs requirements.txt
#      into it.
#   4. Copies .env.example to .env if .env doesn't already exist yet (never
#      overwrites one you already have).
#   5. Prints exactly how to run the app.
#
# What this does NOT do (the app already handles these itself on first
# run, so there's nothing for an installer to set up):
#   - Generate a session secret key (core.py creates trailer_library/.secret_key
#     itself, once, the first time it doesn't find one)
#   - Create the trailer_library/ or show_templates/ storage directories
#     (library_db.py / pipeline.py os.makedirs() these on import)
#   - Create a default admin account (auth.py bootstraps one on first boot
#     if users.db is empty -- see README.md's "Lost the admin password?"
#     section for recovery if you ever need it)
set -euo pipefail

WITH_FFMPEG=0
for arg in "$@"; do
  case "$arg" in
    --with-ffmpeg) WITH_FFMPEG=1 ;;
    -h|--help)
      echo "Usage: ./install.sh [--with-ffmpeg]"
      echo "  --with-ffmpeg   Also install ffmpeg via the distro package manager (needs sudo)."
      exit 0
      ;;
  esac
done

# ---- Colors (skipped automatically if not a real terminal) ----
if [ -t 1 ]; then
  BOLD=$'\033[1m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; RED=$'\033[31m'; RESET=$'\033[0m'
else
  BOLD=''; GREEN=''; YELLOW=''; RED=''; RESET=''
fi
info()  { echo "${BOLD}==>${RESET} $1"; }
ok()    { echo "${GREEN}  ok${RESET} $1"; }
warn()  { echo "${YELLOW}  warning${RESET} $1"; }
fail()  { echo "${RED}  error${RESET} $1"; exit 1; }

# ---- 0. Locate the repo, or fetch it ----
# Deliberately NOT using $0/BASH_SOURCE to find "where this script lives" --
# that only points to a real file when run as ./install.sh. Piped in via
# `curl | bash`, there is no script file on disk at all, so everything below
# works off the current directory instead, and self-bootstraps by cloning
# the repo into ./eyan when core.py isn't already here.
if [ ! -f "core.py" ] || [ ! -f "requirements.txt" ]; then
  info "core.py not found here -- fetching the EYAN repo first"
  REPO_URL="https://github.com/junleynes/eyan.git"
  REPO_DIR="eyan"
  if [ -d "$REPO_DIR" ]; then
    ok "$REPO_DIR/ already exists, using it"
  elif command -v git >/dev/null 2>&1; then
    git clone --depth 1 "$REPO_URL" "$REPO_DIR"
    ok "cloned into $REPO_DIR/"
  else
    warn "git not found -- downloading a tarball instead"
    curl -fsSL "https://github.com/junleynes/eyan/archive/refs/heads/main.tar.gz" | tar -xz
    mv "eyan-main" "$REPO_DIR"
    ok "downloaded into $REPO_DIR/"
  fi
  cd "$REPO_DIR"
fi
if [ ! -f "core.py" ] || [ ! -f "requirements.txt" ]; then
  fail "core.py / requirements.txt still not found in $(pwd) after fetching -- something's wrong with the repo layout."
fi

# ---- 1. Python version ----
info "Checking for Python 3.10+"
PYTHON_BIN=""
for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
  if command -v "$candidate" >/dev/null 2>&1; then
    ver="$("$candidate" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
    major="${ver%%.*}"; minor="${ver##*.}"
    if [ "$major" -eq 3 ] && [ "$minor" -ge 10 ]; then
      PYTHON_BIN="$candidate"
      ok "found $candidate (Python $ver)"
      break
    fi
  fi
done
if [ -z "$PYTHON_BIN" ]; then
  fail "No Python 3.10+ found on PATH. numpy 2.x (pinned in requirements.txt) needs 3.10+. Install a newer Python and re-run."
fi

# ---- 2. ffmpeg / ffprobe ----
info "Checking for ffmpeg and ffprobe"
MISSING_FFMPEG=0
for bin in ffmpeg ffprobe; do
  if command -v "$bin" >/dev/null 2>&1; then
    ok "found $bin"
  else
    warn "$bin not found on PATH"
    MISSING_FFMPEG=1
  fi
done
if [ "$MISSING_FFMPEG" -eq 1 ]; then
  if [ "$WITH_FFMPEG" -eq 1 ]; then
    info "Installing ffmpeg (--with-ffmpeg was passed)"
    if command -v apt-get >/dev/null 2>&1; then
      sudo apt-get update -qq && sudo apt-get install -y ffmpeg
    elif command -v dnf >/dev/null 2>&1; then
      sudo dnf install -y ffmpeg
    elif command -v yum >/dev/null 2>&1; then
      sudo yum install -y ffmpeg
    elif command -v pacman >/dev/null 2>&1; then
      sudo pacman -Sy --noconfirm ffmpeg
    elif command -v zypper >/dev/null 2>&1; then
      sudo zypper install -y ffmpeg
    else
      fail "Don't recognize this system's package manager. Install ffmpeg manually, then re-run without --with-ffmpeg."
    fi
    command -v ffmpeg >/dev/null 2>&1 && command -v ffprobe >/dev/null 2>&1 \
      && ok "ffmpeg installed" \
      || fail "ffmpeg install appears to have failed -- check the output above."
  else
    warn "ffmpeg/ffprobe are required at runtime (the render pipeline shells out to both)."
    warn "Install them yourself, or re-run this script with --with-ffmpeg to have it try for you:"
    warn "  Debian/Ubuntu:  sudo apt-get install ffmpeg"
    warn "  Fedora:         sudo dnf install ffmpeg"
    warn "  RHEL/CentOS:    sudo yum install ffmpeg"
    warn "  Arch:           sudo pacman -S ffmpeg"
    warn "Continuing install without it -- the app will start, but any render will fail until it's on PATH."
  fi
fi

# ---- 3. Virtual environment + dependencies ----
info "Setting up venv/ and installing requirements.txt (this can take a few minutes)"
if [ ! -d "venv" ]; then
  "$PYTHON_BIN" -m venv venv
  ok "created venv/"
else
  ok "venv/ already exists, reusing it"
fi
# shellcheck disable=SC1091
source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q
ok "dependencies installed"
deactivate

# ---- 4. .env ----
info "Setting up .env"
if [ -f ".env" ]; then
  ok ".env already exists, leaving it alone"
elif [ -f ".env.example" ]; then
  cp .env.example .env
  ok "created .env from .env.example -- edit it to set ADMIN_PASSWORD, service URLs, etc."
else
  warn ".env.example not found, skipping -- the app runs fine on real env vars alone if you'd rather set it up that way."
fi

# ---- 5. Done ----
echo
echo "${BOLD}${GREEN}Install complete.${RESET}"
echo
echo "To run EYAN:"
echo "  source venv/bin/activate"
echo "  python3 main.py"
echo
echo "First run with no existing users.db creates a default admin account and"
echo "prints its username/password once to the console -- watch for it, or set"
echo "ADMIN_USERNAME/ADMIN_PASSWORD in .env first to choose your own. See"
echo "README.md for the full rundown (including how to recover a lost admin"
echo "password with RESET_ADMIN)."

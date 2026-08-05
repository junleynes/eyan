#Requires -Version 5.1
<#
.SYNOPSIS
  EYAN install script (Windows). Run from the repo root:
    .\install.ps1

.DESCRIPTION
  1. Checks for Python 3.10+ (numpy 2.x, pinned in requirements.txt, needs it).
  2. Checks for ffmpeg/ffprobe on PATH -- external programs the render
     pipeline shells out to, not something pip can install. Pass
     -WithFFmpeg to have this script install them via winget; otherwise
     it just tells you how.
  3. Creates a venv\ virtual environment and installs requirements.txt
     into it.
  4. Copies .env.example to .env if .env doesn't already exist (never
     overwrites one you already have).
  5. Prints exactly how to run the app.

  What this does NOT do (the app already handles these itself on first
  run, so there's nothing for an installer to set up):
    - Generate a session secret key (core.py creates .secret_key itself,
      once, the first time it doesn't find one)
    - Create the trailer_library\ or show_templates\ storage directories
      (library_db.py / pipeline.py create these on import)
    - Create a default admin account (auth.py bootstraps one on first
      boot if users.db is empty -- see README.md's "Lost the admin
      password?" section for recovery if you ever need it)

.PARAMETER WithFFmpeg
  Also install ffmpeg via winget if it isn't already on PATH.
#>
param(
    [switch]$WithFFmpeg
)

$ErrorActionPreference = 'Stop'

function Write-Info  { param($msg) Write-Host "==> $msg" -ForegroundColor Cyan }
function Write-Ok    { param($msg) Write-Host "  ok $msg" -ForegroundColor Green }
function Write-Warn2 { param($msg) Write-Host "  warning $msg" -ForegroundColor Yellow }
function Write-Fail  { param($msg) Write-Host "  error $msg" -ForegroundColor Red; exit 1 }

# ---- Must be run from the repo root ----
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir
if (-not (Test-Path "core.py") -or -not (Test-Path "requirements.txt")) {
    Write-Fail "core.py / requirements.txt not found here ($ScriptDir). Run this from the repo root."
}

# ---- 1. Python version ----
Write-Info "Checking for Python 3.10+"
$PythonCmd = $null
foreach ($candidate in @('py', 'python', 'python3')) {
    $cmdInfo = Get-Command $candidate -ErrorAction SilentlyContinue
    if (-not $cmdInfo) { continue }
    try {
        if ($candidate -eq 'py') {
            $verOutput = & py -3 -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
            $launchArgs = @('-3')
        } else {
            $verOutput = & $candidate -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
            $launchArgs = @()
        }
    } catch {
        continue
    }
    if (-not $verOutput) { continue }
    $parts = $verOutput.Trim().Split('.')
    $major = [int]$parts[0]; $minor = [int]$parts[1]
    if ($major -eq 3 -and $minor -ge 10) {
        $PythonCmd = $candidate
        $PythonArgs = $launchArgs
        Write-Ok "found $candidate $launchArgs (Python $verOutput)"
        break
    }
}
if (-not $PythonCmd) {
    Write-Fail "No Python 3.10+ found on PATH. numpy 2.x (pinned in requirements.txt) needs 3.10+. Install a newer Python (python.org or 'winget install Python.Python.3.12') and re-run."
}

# ---- 2. ffmpeg / ffprobe ----
Write-Info "Checking for ffmpeg and ffprobe"
$missingFFmpeg = $false
foreach ($bin in @('ffmpeg', 'ffprobe')) {
    if (Get-Command $bin -ErrorAction SilentlyContinue) {
        Write-Ok "found $bin"
    } else {
        Write-Warn2 "$bin not found on PATH"
        $missingFFmpeg = $true
    }
}
if ($missingFFmpeg) {
    if ($WithFFmpeg) {
        Write-Info "Installing ffmpeg (-WithFFmpeg was passed)"
        if (Get-Command winget -ErrorAction SilentlyContinue) {
            winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements
            Write-Warn2 "winget finished -- you may need to open a NEW terminal for the updated PATH to take effect before running EYAN."
        } else {
            Write-Fail "winget isn't available. Install ffmpeg manually from https://ffmpeg.org/download.html and add it to PATH, then re-run."
        }
    } else {
        Write-Warn2 "ffmpeg/ffprobe are required at runtime (the render pipeline shells out to both)."
        Write-Warn2 "Install with winget, or re-run this script with -WithFFmpeg to have it try for you:"
        Write-Warn2 "  winget install --id Gyan.FFmpeg -e"
        Write-Warn2 "Or download from https://ffmpeg.org/download.html and add the bin\ folder to PATH."
        Write-Warn2 "Continuing install without it -- the app will start, but any render will fail until it's on PATH."
    }
}

# ---- 3. Virtual environment + dependencies ----
Write-Info "Setting up venv\ and installing requirements.txt (this can take a few minutes)"
if (-not (Test-Path "venv")) {
    & $PythonCmd @PythonArgs -m venv venv
    Write-Ok "created venv\"
} else {
    Write-Ok "venv\ already exists, reusing it"
}
$venvPython = Join-Path $ScriptDir "venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Fail "venv\Scripts\python.exe not found after venv creation -- something went wrong above."
}
& $venvPython -m pip install --upgrade pip -q
& $venvPython -m pip install -r requirements.txt -q
Write-Ok "dependencies installed"

# ---- 4. .env ----
Write-Info "Setting up .env"
if (Test-Path ".env") {
    Write-Ok ".env already exists, leaving it alone"
} elseif (Test-Path ".env.example") {
    Copy-Item ".env.example" ".env"
    Write-Ok "created .env from .env.example -- edit it to set ADMIN_PASSWORD, service URLs, etc."
} else {
    Write-Warn2 ".env.example not found, skipping -- the app runs fine on real env vars alone if you'd rather set it up that way."
}

# ---- 5. Done ----
Write-Host ""
Write-Host "Install complete." -ForegroundColor Green
Write-Host ""
Write-Host "To run EYAN:"
Write-Host "  venv\Scripts\Activate.ps1"
Write-Host "  python main.py"
Write-Host ""
Write-Host "(If Activate.ps1 is blocked by execution policy, either run PowerShell"
Write-Host " as Administrator once and 'Set-ExecutionPolicy RemoteSigned -Scope"
Write-Host " CurrentUser', or just skip activation and run venv\Scripts\python.exe"
Write-Host " main.py directly -- same effect.)"
Write-Host ""
Write-Host "First run with no existing users.db creates a default admin account and"
Write-Host "prints its username/password once to the console -- watch for it, or set"
Write-Host "ADMIN_USERNAME/ADMIN_PASSWORD in .env first to choose your own. See"
Write-Host "README.md for the full rundown (including how to recover a lost admin"
Write-Host "password with RESET_ADMIN)."

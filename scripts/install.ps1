<#
.SYNOPSIS
    Windows 10/11 PowerShell installer for AI-Video-Editor (OpenCode + Kdenlive workflow).
.DESCRIPTION
    Creates an isolated Python virtual environment (.venv) inside the repository,
    installs free open-source dependencies, checks for FFmpeg and Kdenlive,
    optionally caches the faster-whisper speech recognition model ('base' or 'small'),
    and runs scripts/diagnose.py to verify installation readiness.
#>
[CmdletBinding()]
param(
    [ValidateSet("none", "base", "small")]
    [string]$PreDownloadWhisperModel = "none",

    [switch]$SkipModelDownload
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " AI-Video-Editor Setup for Windows 10/11 (OpenCode + Kdenlive)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# 1. Locate Python 3.10+
$PythonCmd = $null
if (Get-Command "py" -ErrorAction SilentlyContinue) {
    try {
        $ver = & py -3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
        if ($LASTEXITCODE -eq 0) {
            $PythonCmd = "py -3"
        }
    } catch {}
}
if (-not $PythonCmd -and (Get-Command "python" -ErrorAction SilentlyContinue)) {
    $PythonCmd = "python"
}
if (-not $PythonCmd) {
    Write-Error "Python 3.10 or newer was not found on PATH. Install Python 3.11+ from https://www.python.org/downloads/windows/ and re-run this script."
    exit 1
}

Write-Host "[1/5] Using Python launcher: $PythonCmd" -ForegroundColor Green

# 2. Create .venv if it does not exist
$VenvDir = Join-Path $RepoRoot ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Host "[2/5] Creating virtual environment at $VenvDir ..." -ForegroundColor Green
    Invoke-Expression "$PythonCmd -m venv `"$VenvDir`""
} else {
    Write-Host "[2/5] Virtual environment already exists at $VenvDir" -ForegroundColor Green
}

# 3. Upgrade pip and install package + dev test suite
Write-Host "[3/5] Installing AI-Video-Editor and open-source dependencies ..." -ForegroundColor Green
& $VenvPython -m pip install --upgrade pip setuptools wheel
& $VenvPython -m pip install -e ".[dev]"

# 4. Optional Whisper model pre-download
if (-not $SkipModelDownload -and $PreDownloadWhisperModel -ne "none") {
    Write-Host "[4/5] Pre-downloading faster-whisper '$PreDownloadWhisperModel' model weights (CPU int8) ..." -ForegroundColor Green
    & $VenvPython -c "from faster_whisper import WhisperModel; WhisperModel('$PreDownloadWhisperModel', device='cpu', compute_type='int8'); print('Cached faster-whisper model:', '$PreDownloadWhisperModel')"
} else {
    Write-Host "[4/5] Skipping upfront Whisper model download (models download automatically on first use or pass -PreDownloadWhisperModel base)." -ForegroundColor Yellow
}

# 5. Run diagnostics
Write-Host "[5/5] Running environment diagnostics ..." -ForegroundColor Green
& $VenvPython (Join-Path $RepoRoot "scripts\diagnose.py")

Write-Host ""
Write-Host "Setup complete! To edit a gaming video in OpenCode or PowerShell, run:" -ForegroundColor Cyan
Write-Host "  .\.venv\Scripts\python.exe -m ai_video_editor.cli edit --input `"C:\Path\To\Video.mp4`" --title `"My Gaming Video`"" -ForegroundColor White
Write-Host "Note: If Kdenlive is not yet installed on Windows, install it free via:" -ForegroundColor Yellow
Write-Host "  winget install -e --id KDE.Kdenlive" -ForegroundColor White

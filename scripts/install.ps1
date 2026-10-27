<#
.SYNOPSIS
    Windows 10/11 PowerShell installer for AI-Video-Editor (OpenCode + Kdenlive workflow).
.DESCRIPTION
    Creates an isolated Python virtual environment (.venv) inside the repository,
    verifies Python >= 3.10, installs free open-source dependencies, checks for
    FFmpeg and Kdenlive, optionally caches a faster-whisper speech recognition model
    ('base', 'small', 'base.en', 'small.en'), and runs scripts/diagnose.py.
    Safe for repeated execution (idempotent).
#>
[CmdletBinding()]
param(
    [ValidateSet("none", "base", "small", "base.en", "small.en")]
    [string]$PreDownloadWhisperModel = "none",

    [switch]$SkipModelDownload
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $RepoRoot

try {
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host " AI-Video-Editor Setup for Windows 10/11 (OpenCode + Kdenlive)" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan

    # 1. Locate and verify Python >= 3.10 (ignoring WindowsApps Microsoft Store stub)
    $PythonExe = $null
    $PythonArgs = @()
    $PythonVersionStr = $null

    if (Get-Command "py" -ErrorAction SilentlyContinue) {
        try {
            $verCheck = & py -3 -c "import sys; assert sys.version_info >= (3, 10); print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')" 2>$null
            if ($LASTEXITCODE -eq 0 -and $verCheck) {
                $PythonExe = "py"
                $PythonArgs = @("-3")
                $PythonVersionStr = ($verCheck | Select-Object -First 1).Trim()
            }
        } catch {}
    }

    if (-not $PythonExe -and (Get-Command "python" -ErrorAction SilentlyContinue)) {
        try {
            $verCheck = & python -c "import sys; assert sys.version_info >= (3, 10); print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')" 2>$null
            if ($LASTEXITCODE -eq 0 -and $verCheck) {
                $PythonExe = "python"
                $PythonArgs = @()
                $PythonVersionStr = ($verCheck | Select-Object -First 1).Trim()
            }
        } catch {}
    }

    if (-not $PythonExe) {
        throw "Python 3.10 or newer was not found on PATH (or only the Microsoft Store stub is present). Install Python 3.11+ from https://www.python.org/downloads/windows/ and re-run this script."
    }

    Write-Host "[1/5] Verified Python $PythonVersionStr via '$PythonExe $($PythonArgs -join ' ')'" -ForegroundColor Green

    # 2. Create .venv if it does not exist (safe for repeated runs and paths with spaces/special chars)
    $VenvDir = Join-Path $RepoRoot ".venv"
    $VenvPython = Join-Path $VenvDir "Scripts\python.exe"

    if (-not (Test-Path -LiteralPath $VenvPython)) {
        Write-Host "[2/5] Creating virtual environment at $VenvDir ..." -ForegroundColor Green
        & $PythonExe @PythonArgs -m venv $VenvDir
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $VenvPython)) {
            throw "Failed to create Python virtual environment at $VenvDir (exit code $LASTEXITCODE)."
        }
    } else {
        Write-Host "[2/5] Virtual environment already exists at $VenvDir" -ForegroundColor Green
    }

    # 3. Upgrade pip and install package + dev test suite
    Write-Host "[3/5] Installing AI-Video-Editor and open-source dependencies ..." -ForegroundColor Green
    & $VenvPython -m pip install --upgrade pip setuptools wheel
    if ($LASTEXITCODE -ne 0) {
        throw "pip upgrade failed with exit code $LASTEXITCODE."
    }

    & $VenvPython -m pip install -e ".[dev]"
    if ($LASTEXITCODE -ne 0) {
        throw "Package installation failed with exit code $LASTEXITCODE."
    }

    # 4. Optional Whisper model pre-download (always uses CPU int8 for safe caching)
    if (-not $SkipModelDownload -and $PreDownloadWhisperModel -ne "none") {
        Write-Host "[4/5] Pre-downloading faster-whisper '$PreDownloadWhisperModel' model weights (CPU int8) ..." -ForegroundColor Green
        $dlScript = "from faster_whisper import WhisperModel; WhisperModel('$PreDownloadWhisperModel', device='cpu', compute_type='int8'); print('Cached faster-whisper model: $PreDownloadWhisperModel')"
        & $VenvPython -c $dlScript
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "Whisper model '$PreDownloadWhisperModel' could not be pre-downloaded right now (exit code $LASTEXITCODE). Check your internet connection or retry later."
        }
    } else {
        Write-Host "[4/5] Skipping upfront Whisper model download (pass -PreDownloadWhisperModel base to cache offline)." -ForegroundColor Yellow
    }

    # 5. Run diagnostics
    Write-Host "[5/5] Running environment diagnostics ..." -ForegroundColor Green
    $DiagScript = Join-Path $RepoRoot "scripts\diagnose.py"
    & $VenvPython $DiagScript --human
    if ($LASTEXITCODE -ne 0) {
        throw "Environment diagnostics reported missing required components (exit code $LASTEXITCODE)."
    }

    Write-Host ""
    Write-Host "Setup complete! To edit a gaming video in OpenCode or PowerShell, run:" -ForegroundColor Cyan
    Write-Host "  .\.venv\Scripts\python.exe -m ai_video_editor.cli edit --input `"C:\Path\To\Video.mp4`" --title `"My Gaming Video`"" -ForegroundColor White
    Write-Host "Note: If Kdenlive is not yet installed on Windows, install it free via:" -ForegroundColor Yellow
    Write-Host "  winget install -e --id KDE.Kdenlive" -ForegroundColor White
} finally {
    Pop-Location
}

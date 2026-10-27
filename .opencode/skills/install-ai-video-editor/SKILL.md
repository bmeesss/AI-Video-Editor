---
name: install-ai-video-editor
description: Set up the local Python virtual environment, free open-source dependencies (PyAV, faster-whisper, ONNX Runtime, PySceneDetect, OpenCV), FFmpeg, and Kdenlive checks on Windows 10/11 or Linux.
license: MIT
compatibility: opencode
metadata:
  audience: gaming-creators
  workflow: phase-4-installation
---

# Install AI Video Editor (`install-ai-video-editor`)

## When to Use This Skill
Load this skill when the user wants to install, set up, or repair the AI-Video-Editor environment on Windows 10/11 (or Linux) for use with OpenCode and Kdenlive.

## Safety & Privacy Principles
- **100% Free & Local:** All dependencies (`faster-whisper`, `onnxruntime`, `av`, `imageio-ffmpeg`, `scenedetect`, `opencv-python`, `numpy`, `lxml`) are open-source and run locally. No API keys, cloud accounts, or paid services are used.
- **Isolated Virtual Environment:** All Python packages are installed inside the repository's `.venv` folder so global system Python packages are never modified.
- **No Unapproved System Changes:** Ask the user before running `install.ps1` or installing Kdenlive via `winget`.

## Command Usage

### Windows 10/11 PowerShell Installer
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install.ps1
```
Optional flags for `scripts\install.ps1`:
- `-PreDownloadWhisperModel base` (or `small`): Pre-caches the `faster-whisper` model weights locally so subsequent edits work 100% offline.
- `-SkipModelDownload`: Skips pre-downloading Whisper weights during install.

### Linux / Headless Container Setup
```bash
python3 -m venv .venv
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install -e ".[dev]"
# On headless Linux systems without libGL.so.1, ensure opencv-python-headless is active:
./.venv/bin/pip uninstall -y opencv-python opencv-python-headless
./.venv/bin/pip install opencv-python-headless
./.venv/bin/python scripts/diagnose.py
```

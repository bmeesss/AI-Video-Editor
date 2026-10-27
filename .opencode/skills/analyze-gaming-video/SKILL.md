---
name: analyze-gaming-video
description: Analyze a raw gameplay recording locally in read-only mode to extract multi-track audio roles, faster-whisper speech transcripts, Silero VAD voice intervals, audio transients, and visual scene/motion windows into a canonical JSON report.
license: MIT
compatibility: opencode
metadata:
  audience: gaming-creators
  workflow: phase-2-analysis
---

# Analyze Gaming Video (`analyze-gaming-video`)

## When to Use This Skill
Load this skill when the user provides a gameplay recording file (`.mp4`, `.mkv`, `.mov`) and wants to inspect its streams, transcribe commentary, detect dead air or verbal stumbles, or prepare an `analysis_report.json` before generating a Kdenlive timeline project.

## Non-Negotiable Safety & Methodological Rules
1. **Strictly Read-Only Source Media:** Never move, rename, modify, transcode in place, or delete the user's original recording. The analyzer captures SHA-256 and MD5 checksums before and after probing to enforce zero mutation.
2. **Unique Output Paths:** Always write `analysis_report.json` into a new, collision-free job folder (never overwrite an existing report or project).
3. **Signals, Not Ground Truth:** Do not claim to the user that visual motion scores or audio loudness spikes reliably detect "funny" or "exciting" moments. Explain that they are used conservatively as protective signals so active silent gameplay and reactions are not accidentally cut as dead air.
4. **Hardware-Aware Whisper Selection:** By default (`--whisper-model auto`), the analyzer selects between `base` and `small` based on available RAM/CPU/GPU and automatically falls back to CPU (`int8`) if CUDA is unavailable.

## Command Usage

### Windows PowerShell
```powershell
.\.venv\Scripts\python.exe -m ai_video_editor.cli analyze `
  --input "C:\Path\To\Recording.mp4" `
  --output-report ".\output" `
  --whisper-model auto `
  --whisper-device auto
```

### Linux / macOS
```bash
./.venv/bin/python -m ai_video_editor.cli analyze \
  --input "/path/to/recording.mp4" \
  --output-report "./output" \
  --whisper-model auto \
  --whisper-device auto
```

## Interpreting the Output
The command prints a JSON summary to stdout containing:
- `analysis_report_path`: Path to the generated `analysis_report.json`.
- `audio_streams`: Detected audio tracks and their inferred roles (`microphone`, `gameplay`, `mixed`, `secondary`).
- `speech_intervals_count` & `transcript_segments_count`: Detected commentary segments.
- `overall_analysis_confidence`: Confidence score in `[0.0, 1.0]`.
- `warnings`: Important notices (e.g., Variable Frame Rate `VFR` warnings or offline model fallback notices). Always surface any `warnings` to the user.

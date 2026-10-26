# AI-Video-Editor

A free, open-source, locally executed AI-assisted gaming video editor designed for **OpenCode on Windows 10/11** (and cross-compatible with Linux/macOS). Given a raw gameplay recording and a video title, `AI-Video-Editor` analyzes the recording in strictly **read-only mode** and generates a genuinely editable **Kdenlive** project (`.kdenlive` + `.srt` subtitles) for manual review and export.

---

## Core Design & Safety Principles

1. **Never Auto-Renders Final Video:** The tool **never** automatically renders or exports a finished video file. It creates an editable `.kdenlive` timeline so you retain full creative control before exporting manually in Kdenlive (**Project -> Render / `Ctrl+Return`**).
2. **Non-Destructive Source Media Protection:** Original recordings are opened strictly in read-only mode. SHA-256 and MD5 checksums of the source recording are captured before and verified after every operation.
3. **Collision-Free Output Directories:** Every editing job creates a brand-new, timestamped directory (`<title_slug>_<YYYYMMDD_HHMMSS>`). Existing projects and files are never overwritten.
4. **Conservative Gaming Edits:**
   - Full-length YouTube gaming videos only (no Shorts generation in v1).
   - Preserves chronological event order and always keeps the opening intro (`--preserve-intro 8.0` seconds by default).
   - Removes long dead air (`>= 2.5s` by default) and high-confidence isolated verbal stumbles conservatively with pre-roll (`0.35s`) and post-roll (`0.45s`) safety margins.
   - Protects silent/tense gameplay moments whenever visual gameplay motion or game audio activity is present.
   - Never adds background music or random/unjustified zooms, memes, or sound effects.
   - Stops for human review (`exit code 2`) if overall edit confidence falls below the configured threshold (`--min-confidence 0.55`).
5. **100% Free & Local:** Powered by open-source local libraries (`PyAV`, `imageio-ffmpeg`, `faster-whisper`, `onnxruntime` with bundled Silero VAD v6, `PySceneDetect`, `OpenCV`, `NumPy`, and `lxml`). No paid APIs, cloud uploads, or subscriptions.

---

## Timeline Modes: `review-layers` (Default) vs. `ripple`

| Feature | `review-layers` (Default) | `ripple` (Clean Ripple) |
| :--- | :--- | :--- |
| **Main Video & Audio (`V1`, `A1`, `A2`)** | Kept clips placed contiguously (ripple-edited) with frame-accurate A/V sync. | Kept clips placed contiguously (ripple-edited) with frame-accurate A/V sync. |
| **Removed Cuts Track (`V2_Review`, `A_Review`)** | Included on dedicated hidden/muted tracks (`hide="both"` / `hide=3`) with each clip in Kdenlive's **Disabled** state (`set.test_audio=1, set.test_image=1`) so removed audio/video never bleeds into playback. | Omitted from timeline tracks. |
| **Timeline Guides (Markers)** | Placed at every cut point with exact removed source timecodes, reason, and confidence. | Placed at every cut point with exact removed source timecodes, reason, and confidence. |
| **Project Bin Clip Zones** | Every removed segment is saved as a named Clip Zone on the master bin clip for instant drag-and-drop recovery. | Every removed segment is saved as a named Clip Zone on the master bin clip for instant drag-and-drop recovery. |
| **Subtitles (`.kdenlive.srt` & `.srt`)** | Synchronized to the edited timeline. | Synchronized to the edited timeline. |

---

## Quick Start on Windows 10/11

### 1. Prerequisites
- **Windows 10 or Windows 11** (64-bit)
- **Python 3.10 – 3.12** ([python.org](https://www.python.org/downloads/windows/))
- **Kdenlive 23.04 or newer** (24.x / 25.x / 26.x recommended). Install for free via Winget or [kdenlive.org](https://kdenlive.org/en/download/):
  ```powershell
  winget install -e --id KDE.Kdenlive
  ```

### 2. Run the PowerShell Installer
From the repository root in PowerShell:
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install.ps1 -PreDownloadWhisperModel base
```
This creates an isolated `.venv` virtual environment, installs all Python dependencies (including a bundled FFmpeg binary via `imageio-ffmpeg`), optionally caches the `faster-whisper` model weights, and runs `scripts\diagnose.py`.

### 3. Edit a Gaming Recording
#### Via OpenCode Agent
Open the repository in [OpenCode](https://opencode.ai) and prompt:
> *"Edit `C:\Videos\EldenRing_Run01.mp4` with the title `Elden Ring Boss Run - No Summons` using the default review-layers timeline."*

OpenCode automatically discovers the repository skills in `.opencode/skills/`:
- `analyze-gaming-video`
- `generate-timeline-project`
- `verify-and-diagnose`
- `install-ai-video-editor`

#### Via PowerShell CLI Directly
```powershell
.\.venv\Scripts\python.exe -m ai_video_editor.cli edit `
  --input "C:\Videos\EldenRing_Run01.mp4" `
  --title "Elden Ring Boss Run - No Summons" `
  --output-dir ".\output" `
  --timeline-mode review-layers
```

---

## Whisper Model Sizes & Hardware Requirements

`AI-Video-Editor` automatically selects between `base` and `small` (`--whisper-model auto`) based on your system hardware, with automatic fallback to CPU (`int8` quantization) when an NVIDIA CUDA GPU is not present:

| Whisper Model | Disk Download Size | Approx. RAM / VRAM (`int8` / `fp16`) | Auto-Selection Rule (`--whisper-model auto`) |
| :--- | :--- | :--- | :--- |
| **`base`** | ~145 MB | ~0.8 – 1.2 GB RAM | Selected on CPU-only machines with `< 12 GB` RAM or `< 8` CPU cores. |
| **`small`** | ~480 MB | ~1.8 – 2.4 GB RAM | Selected when a CUDA GPU is available OR when system RAM `>= 12 GB` and CPU cores `>= 8`. |
| **Silero VAD v6** | ~2 MB (bundled in `faster-whisper`) | < 100 MB RAM (`onnxruntime`) | Always enabled for millisecond voice activity detection. |

You can override model selection explicitly at any time with `--whisper-model base`, `--whisper-model small`, or `--whisper-device cpu`.

---

## Reviewing, Recovering Cuts & Exporting Manually in Kdenlive

1. **Open the Project:** Launch Kdenlive and open `<output_dir>\<job_folder>\<title_slug>.kdenlive`.
2. **Play & Inspect Cuts:**
   - Press `Space` to preview the edited timeline.
   - Colored **Timeline Guides** along the top ruler mark every cut point (`Cut #1: Removed 4.20s (dead_air_silence) [00:00:05.00 -> 00:00:09.20]`).
3. **Recover a Removed Cut (Two Easy Ways):**
   - **From the Review Track (`review-layers` mode):** Locate the disabled clip under the cut guide on `V2 - Review Removed Cuts` (or `A_Review`), right-click the clip -> **Enable Clip**, and drag it onto `V1` / `A1` (or unhide the track to preview it in place).
   - **From the Project Bin (`review-layers` and `ripple` modes):** Expand the source video clip in the **Project Bin** to view its saved **Clip Zones** (`Removed #1`, `Removed #2`, ...) and drag any zone directly into the timeline.
4. **Review Subtitles:**
   - Inspect and edit captions on the Kdenlive **Subtitles** track (loaded from `<project>.kdenlive.srt`; Kdenlive 24.08+ automatically upgrades this sidecar to `.ass` while keeping the `.srt` file intact).
5. **Manual Final Export:**
   - Press `Ctrl+Return` (or **Project -> Render**), choose your preferred preset (e.g., **MP4-H264/AAC**), and click **Render to File**.

---

## Supported vs. Unsupported Scenarios (v1)

### Supported
- Full-length gameplay recordings in `.mp4`, `.mkv`, or `.mov` containers.
- Single-track mixed audio **and** multi-track OBS recordings (e.g., Track 1 Microphone Commentary + Track 2 Desktop/Game Audio).
- Constant Frame Rate (CFR) recordings at standard rates (`23.976`, `24`, `25`, `29.97`, `30`, `50`, `59.94`, `60`, `120` fps), plus automatic detection and warnings for Variable Frame Rate (VFR) recordings.
- Paths containing spaces, punctuation, and Unicode characters (`Jürgen`, `太郎`, etc.).

### Intentionally Unsupported in v1
- Automatic final video rendering/exporting (must be done manually in Kdenlive).
- Vertical Shorts / TikTok / Reels generation.
- Shotcut (`.mlt`) or DaVinci Resolve (`.fcpxml`) exporters (Kdenlive is the sole v1 target).
- Automatic background music or random AI zoom/meme/SFX injection.
- Claiming that visual motion or loudness spikes alone prove a moment is "funny" or "exciting" (they are used strictly as contextual signals to avoid cutting active gameplay or reactions).

---

## Contributor Guide: Running Tests & Verification

```bash
# Run the full unit, integration, and end-to-end test suite
./.venv/bin/pytest

# Run environment diagnostics
./.venv/bin/python scripts/diagnose.py --human

# Generate and verify the Phase 0 minimal validation sample project
./.venv/bin/python scripts/verify_project.py --generate-minimal-sample ./output/validation_sample

# Verify any existing .kdenlive project against the MLT DTD and Kdenlive 1.1 C++ invariants
./.venv/bin/python scripts/verify_project.py ./output/validation_sample/phase0_minimal_validation_sample_*/minimal_test_project.kdenlive
```

See also:
- [`docs/phase0-validation.md`](docs/phase0-validation.md) — Phase 0 technical validation report, C++ source verification against `KDE/kdenlive`, and corrections to earlier research claims.
- [`docs/user-guide-and-troubleshooting.md`](docs/user-guide-and-troubleshooting.md) — Detailed troubleshooting, VFR handling, and privacy architecture.
- [`LICENSE`](LICENSE) & [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) — MIT license and third-party open-source notices.

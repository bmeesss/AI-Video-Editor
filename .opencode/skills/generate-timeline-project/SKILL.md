---
name: generate-timeline-project
description: Generate a frame-accurate, editable Kdenlive project (.kdenlive) and synchronized .srt subtitles from a gameplay recording and title using either review-layers (default) or clean ripple timeline mode.
license: MIT
compatibility: opencode
metadata:
  audience: gaming-creators
  workflow: phase-3-kdenlive-generation
---

# Generate Timeline Project (`generate-timeline-project`)

## When to Use This Skill
Load this skill when the user wants to turn a gameplay recording and a video title (or an existing `analysis_report.json` + title) into an editable Kdenlive project (`.kdenlive` + `.srt`).

## Non-Negotiable Editing & Safety Rules
1. **Never Auto-Render:** Never invoke `melt` or `ffmpeg` to automatically render a final exported video. The user must review the generated `.kdenlive` timeline in Kdenlive and export manually via **Project -> Render (`Ctrl+Return`)**.
2. **Preserve Original Recording & Existing Projects:** Each editing job creates a brand-new timestamped directory (`<slug>_<YYYYMMDD_HHMMSS>`). Existing files and directories are never overwritten.
3. **Chronological Order & Preserved Intro:** Clips are always kept in their original chronological order, and the opening intro (`--preserve-intro 8.0` seconds by default) is always preserved at the start of the video.
4. **Conservative Cutting:** Only dead air (`>= --min-silence`, default `2.5s`) without active gameplay motion or audio transients, plus high-confidence isolated verbal stumbles, are cut. No background music or random zooms/effects are added.
5. **Timeline Modes (`review-layers` vs `ripple`):**
   - `review-layers` (default): Places kept edits contiguously on `V1` and `A1`/`A2`, while placing removed cuts as disabled clips (`set.test_audio=1, set.test_image=1`) on hidden/muted review tracks (`V2 - Review Removed Cuts`, `A_Review`) with timeline guides at every cut point and bin clip zones for one-click recovery.
   - `ripple`: Places only the kept clips on `V1` and `A1`/`A2`, with timeline guides marking where cuts occurred and bin clip zones for recovery.
6. **Low-Confidence Halt:** If the command exits with code `2` (`status: "halted_low_confidence"`), **stop immediately** and explain to the user which warnings or low-confidence segments require human inspection. Do not loop or retry with lower thresholds unless the user explicitly asks.

## Command Usage

### One-Step End-to-End Edit (Windows PowerShell)
```powershell
.\.venv\Scripts\python.exe -m ai_video_editor.cli edit `
  --input "C:\Path\To\Recording.mp4" `
  --title "My Gaming Video Title" `
  --output-dir ".\output" `
  --timeline-mode review-layers `
  --preserve-intro 8.0 `
  --min-silence 2.5
```

### Generate from Existing Analysis Report (Windows PowerShell)
```powershell
.\.venv\Scripts\python.exe -m ai_video_editor.cli generate `
  --report ".\output\job_folder\analysis_report.json" `
  --title "My Gaming Video Title" `
  --output-dir ".\output" `
  --timeline-mode review-layers
```

### Linux / macOS
```bash
./.venv/bin/python -m ai_video_editor.cli edit \
  --input "/path/to/recording.mp4" \
  --title "My Gaming Video Title" \
  --output-dir "./output" \
  --timeline-mode review-layers
```

## Post-Generation Hand-Off to User
After running `edit` or `generate`, always report:
1. The path to the generated `.kdenlive` file and `.srt` subtitle files.
2. How many segments were kept vs. isolated for review.
3. Whether verification passed (`verification_is_valid: true`).
4. Brief instructions on opening the `.kdenlive` file in Kdenlive, reviewing cuts via timeline guides, recovering any cut from the review track or bin clip zones, and exporting manually.

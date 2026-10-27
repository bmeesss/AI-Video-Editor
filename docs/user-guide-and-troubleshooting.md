# User Guide, Troubleshooting & Privacy Reference

## 1. Detailed Workflow Walkthrough

### Step 1: Environment Check (`diagnose`)
Before processing your first recording, verify your installation:
```powershell
.\.venv\Scripts\python.exe scripts\diagnose.py --human
```
This checks:
- Python `>= 3.10`
- `PyAV`, `imageio-ffmpeg`, `faster-whisper`, `onnxruntime`, `PySceneDetect`, `OpenCV`, `NumPy`, and `lxml`
- Physical RAM, CPU core count, and CUDA availability (to recommend Whisper `base` vs. `small`)
- Whether Kdenlive is installed on your machine

### Step 2: Analyze & Generate (`edit`)
Run the one-step `edit` command (or ask OpenCode to run `generate-timeline-project`):
```powershell
.\.venv\Scripts\python.exe -m ai_video_editor.cli edit `
  --input "D:\OBS\Apex_Ranked_Session.mkv" `
  --title "Apex Legends Ranked Climb" `
  --output-dir "D:\Edits" `
  --timeline-mode review-layers `
  --whisper-model auto `
  --preserve-intro 8.0 `
  --min-silence 2.5
```

Every run creates a new, collision-free folder under `--output-dir`, for example:
`D:\Edits\Apex_Legends_Ranked_Climb_20261026_153000\`
Inside that folder you will find:
1. `<slug>.kdenlive` — The native Kdenlive 23.04+ (`version="1.1"`) project file.
2. `<slug>.kdenlive.srt` & `<slug>.srt` — Frame-synchronized SubRip subtitle files.
3. `analysis_report.json` — Complete read-only media, stream, VAD, transcript, audio energy, and scene/motion report.
4. `edit_decisions.json` — Exact frame boundaries, reasons, and confidence scores for every kept and removed segment.
5. `verification_report.json` — Structural MLT DTD and Kdenlive C++ semantic verification report.

---

## 2. Troubleshooting Common Issues

### Variable Frame Rate (VFR) Warning
**Symptom:** `warnings` contains `"Video stream 0 appears to use Variable Frame Rate (VFR jitter=...)"`.
**Cause:** ShadowPlay, Xbox Game Bar, or mobile screen recorders sometimes record variable frame rates where frame intervals fluctuate during GPU load spikes.
**Recommendation:**
- Open the generated `.kdenlive` project and check audio/video sync near the end of the timeline.
- When Kdenlive opens a VFR clip, it may offer its built-in **Transcode to Edit-Friendly Format** dialog. If you choose to transcode inside Kdenlive, Kdenlive creates a separate Constant Frame Rate copy without deleting your original recording.
- For future OBS Studio recordings, set **Settings -> Output -> Recording -> Rate Control** with a fixed **Common FPS Value** (e.g., `60 FPS`).

### Whisper Model Not Cached When Offline
**Symptom:** `warnings` contains `"Whisper model 'base' could not be loaded locally ... Continuing with Silero VAD speech intervals only"`.
**Cause:** `faster-whisper` downloads model weights from Hugging Face Hub on first use, and your machine was offline before the model was cached.
**Solution:** Connect to the internet once and run:
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install.ps1 -PreDownloadWhisperModel base
```
Or point `--whisper-model` to a local folder containing a CTranslate2 Whisper model. Even without Whisper weights cached, the bundled Silero VAD v6 ONNX model (`silero_vad_v6.onnx`) runs 100% offline to detect voice activity intervals.

### Low-Confidence Halt (`exit code 2`)
**Symptom:** CLI outputs `"status": "halted_low_confidence"` and exits with code `2`.
**Cause:** The overall edit confidence score fell below `--min-confidence` (default `0.55`), for example due to severely corrupted audio or extreme VFR jitter.
**Solution:** Inspect `analysis_report.json` and `edit_decisions.json` in the job directory to see which stream or segment lowered confidence. After reviewing, you can re-run `generate` with an adjusted `--min-confidence` if appropriate.

### Headless Linux `libGL.so.1` ImportError with OpenCV
**Symptom:** `ImportError: libGL.so.1: cannot open shared object file` on a headless Linux server or container.
**Cause:** `scenedetect` installs the GUI build of `opencv-python` by default.
**Solution:** Replace `opencv-python` with `opencv-python-headless` inside `.venv`:
```bash
./.venv/bin/pip uninstall -y opencv-python opencv-python-headless
./.venv/bin/pip install opencv-python-headless
```

---

## 3. Privacy & Data Handling

- **Zero Telemetry or Cloud Uploads:** All video decoding, audio resampling, VAD, speech transcription, and timeline generation occur locally on your CPU/GPU.
- **Read-Only Media Guarantee:** Your original gameplay recordings are never modified in place, moved, or deleted.
- **Plain-Text Auditable Outputs:** All outputs (`.kdenlive` XML, `.srt` subtitles, and `.json` reports) are human-readable UTF-8 text files stored in your chosen output directory.

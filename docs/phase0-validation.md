# Phase 0: Technical Validation & Architecture Verification Report

This document records the findings of **Phase 0** conducted before implementing the full AI-Video-Editor pipeline. Every assumption was tested against upstream source code, authoritative specifications, or executable checks in the test environment.

---

## 1. Repository & Environment Inspection

- **Repository:** `https://github.com/bmeesss/AI-Video-Editor.git`
- **Branch:** `arena/9dd2a259-ai-video-editor` (branched from initial commit `ae86eaf6be8a3c9176bad01b54f39cd04b523839` on `main`).
- **Initial State:** Clean working tree containing only `README.md` (`# AI-Video-Editor`).
- **Sandbox Execution Environment:**
  - OS: Debian GNU/Linux 12 (bookworm) `x86_64` (kernel `6.1.158+`), Python `3.11.2`.
  - Display / GUI: Headless container (no `X11`/`Wayland` display server, no `kdenlive` or `melt` binary pre-installed).
  - Outbound network policy: Restricted to `github.com`, `api.github.com`, `codeload.github.com`, `pypi.org`, `files.pythonhosted.org`, and `registry.npmjs.org` (`huggingface.co` is not reachable from inside this sandbox container).

---

## 2. Kdenlive Project Format Verification (Upstream Source & Dataset Analysis)

Instead of guessing Kdenlive's XML schema or relying solely on high-level documentation, we inspected Kdenlive's official developer documentation (`dev-docs/fileformat.md`), real project files in `tests/dataset/*.kdenlive` (`av.kdenlive`, `clip-ids.kdenlive`, `test-nesting.kdenlive`), and Kdenlive's C++ timeline loader and validator in `KDE/kdenlive`:
- `src/timeline2/model/builders/meltBuilder.cpp`
- `src/bin/projectitemmodel.cpp`
- `src/bin/projectclip.cpp`
- `src/timeline2/model/trackmodel.cpp`
- `src/timeline2/model/clipmodel.cpp`
- `src/timeline2/model/groupsmodel.cpp`
- `src/bin/model/markerlistmodel.cpp`
- `src/bin/model/subtitlemodel.cpp`
- `src/doc/documentvalidator.cpp`

### Critical Technical Findings from Kdenlive's C++ Implementation

1. **Generation 5 (`kdenlive:docproperties.version = "1.1"`) Structure:**
   - Root element must be `<mlt LC_NUMERIC="C" producer="main_bin" version="7.23.0" root="...">`.
   - `<profile>` defines `frame_rate_num`, `frame_rate_den`, `width`, `height`, `progressive`, `sample_aspect_num`, `sample_aspect_den`, `display_aspect_num`, `display_aspect_den`, and `colorspace="709"`.
   - A built-in `<producer id="producer0">` (with `<property name="kdenlive:playlistid">black_track</property>`, `<property name="resource">black</property>`, `<property name="mlt_service">color</property>`) acts as track 0 (`black_track`) in the sequence tractor.
   - Each timeline track is a double-playlist `<tractor id="tractorN">` wrapping two `<playlist>` elements (`playlist2N` and `playlist2N+1`) to support same-track mixes. All normal clips are placed on sub-playlist `0` (`playlist2N`).
   - All timeline tracks are assembled inside a sequence `<tractor id="{sequence-uuid}">` with `<property name="kdenlive:uuid">{sequence-uuid}</property>` and `<property name="kdenlive:producer_type">17</property>`.
   - `<playlist id="main_bin">` lists all project bin `<chain>` clips and the sequence `<tractor id="{sequence-uuid}">`.
   - The final `<tractor>` in the document has `<property name="kdenlive:projectTractor">1</property>` and wraps `<track producer="{sequence-uuid}" in="..." out="..."/>`.

2. **Bin-to-Timeline Clip ID & Control UUID Mapping (`meltBuilder.cpp` & `projectitemmodel.cpp`):**
   - In `ProjectItemModel::loadBinPlaylist`, Kdenlive loads each bin clip and populates `binIdCorresp[control_uuid] = newId`.
   - In `constructTrackFromMelt` (`meltBuilder.cpp`), when a timeline `<entry>` references a track-specific `<chain>` (which does not have `_kdenlive_processed=1`), Kdenlive reads `<property name="kdenlive:control_uuid">` from the `<chain>`. If `kdenlive:control_uuid` is missing or not a valid UUID (`QUuid(clipId).isNull()`), Kdenlive logs `"Project corrupted. Clip ... with invalid Id."` and removes the clip!
   - **Requirement:** Every bin `<chain>` and every timeline track `<chain>` for the same source media file **must** share both the same integer `kdenlive:id` (e.g., `"4"`) and the same brace-formatted UUID `kdenlive:control_uuid` (e.g., `"{...}"`), and every `<entry>` in a `<playlist>` must also include `<property name="kdenlive:id">4</property>`.

3. **One Master `<chain>` per Timeline Track (`projectclip.cpp`):**
   - In `ProjectClip::giveMasterAndGetTimelineProducer`, Kdenlive registers one `<chain>` per `(trackId, audioStream)` for audio tracks and one `<chain>` per `trackId` for video tracks (`m_audioProducers`, `m_videoProducers`).
   - Reusing the exact same `<chain>` across different tracks causes MLT seek contention; creating a distinct `<chain>` per clip on the same track triggers Kdenlive's `"FOUND INCORRECT PRODUCER ON VIDEO/AUDIO TRACK; FIXING"` recovery path.
   - **Requirement:** Emit 1 master `<chain>` for `main_bin`, 1 `<chain>` per active video track (`set.test_audio=1`, `set.test_image=0`), 1 `<chain>` per active audio track/stream (`set.test_audio=0`, `set.test_image=1`, `audio_index=<stream>`), and 1 disabled `<chain>` (`set.test_audio=1`, `set.test_image=1`) for disabled review clips.

4. **Outdated Upstream `mlt-xml.dtd` Caveats:**
   - Testing Kdenlive's own `tests/dataset/clip-ids.kdenlive` and `test-nesting.kdenlive` against MLT's upstream `mlt-xml.dtd` revealed three DTD discrepancies:
     1. `mlt-xml.dtd` line 29 has invalid XML DTD syntax (`<!ELEMENT properties property >` instead of `<!ELEMENT properties (property)* >`).
     2. Kdenlive Generation 5 uses `{uuid}` as the `id` attribute of the sequence `<tractor>`, which is rejected if `id` is typed as a strict XML 1.0 `ID` token.
     3. Kdenlive Generation 5 writes `in` and `out` attributes on `<track>` inside the final `projectTractor`.
   - Therefore, valid XML or raw DTD validation alone does **not** prove a `.kdenlive` file is usable. Our verifier (`project_verifier.py`) validates both the corrected MLT XML structure and Kdenlive's exact `meltBuilder.cpp` / `DocumentValidator` invariants.

---

## 3. Disabled Tracks, Hidden Clips & Review-Layer Audio Behavior in Kdenlive

We verified how Kdenlive handles muted tracks, hidden tracks, and disabled clips in `meltBuilder.cpp`, `trackmodel.cpp`, and `projectclip.cpp`:

1. **Track `hide` Bitmask (`trackmodel.cpp` & `meltBuilder.cpp`):**
   - Kdenlive uses MLT's `hide` bitmask on tracks:
     - `1` (`hide="video"`): hides video output (standard on audio tracks).
     - `2` (`hide="audio"`): mutes audio output (standard on video tracks).
     - `3` (`hide="both"`): hides video **and** mutes audio (`TrackModel::isHidden()` checks `hide & 1`; `TrackModel::isMute()` checks `hide & 2`).
   - Importantly, in `constructTrackFromMelt` (`meltBuilder.cpp` lines 422–432), Kdenlive initializes `timeline->setTrackProperty(tid, "hide", "1")` for audio tracks and `"2"` for video tracks, and then checks `int muteState = playlist.get_int("hide");`. In MLT's XML parser, `hide="both"` on `<track hide="both" producer="playlistX"/>` sets `hide=3` on the sub-playlist, **provided** `<property name="hide">3</property>` is also present or propagated on the playlist.
   - To guarantee that Kdenlive always marks a review track as hidden/muted on load across all Kdenlive 23.x–26.x releases, we set **both** `hide="both"` on the `<track>` elements inside the review track's `<tractor>` **and** `<property name="hide">3</property>` on the review track's `<playlist>` and `<tractor>`.

2. **Individual Disabled Clips (`PlaylistState::Disabled` in `meltBuilder.cpp` & `projectclip.cpp`):**
   - In `meltBuilder.cpp` (`inferState`), a clip's state is inferred from its parent producer's `set.test_image` and `set.test_audio` properties:
     - `set.test_image=0, set.test_audio=1` -> `PlaylistState::VideoOnly`
     - `set.test_image=1, set.test_audio=0` -> `PlaylistState::AudioOnly`
     - `set.test_image=1, set.test_audio=1` -> `PlaylistState::Disabled`
   - When a clip on the timeline references a producer with `set.test_image=1` and `set.test_audio=1` (and preserves `<property name="kdenlive:audio_index">` on the `<entry>`), Kdenlive loads the clip in **Disabled** state (grayed out on the timeline, producing neither video frames nor audio samples even if its track is unhidden/unmuted, yet re-enableable with a single right-click -> *Enable Clip* in Kdenlive).

3. **Review-Layer Timeline Architecture Decision:**
   - By combining **both** mechanisms on the Review Layer:
     1. Main kept edits are placed contiguously (ripple-edited) on `V1` and `A1` (plus `A2` when separate game audio is present) so pressing **Play** in Kdenlive plays the clean edited video without dead-air gaps.
     2. Removed segments are placed on a dedicated Review Track (`V2 - Review (Removed Cuts)`) whose track is hidden (`hide="both"` / `hide=3`) **and** whose clips are individually in `PlaylistState::Disabled` (`set.test_image=1`, `set.test_audio=1`), non-overlapping and clamped within the sequence duration so they never extend project length or bleed audio/video into playback.
     3. Every cut point on the main timeline is marked with a Timeline Guide (`kdenlive:sequenceproperties.guides`) indicating the exact removed source timecode range and reason, and each removed segment is also registered as a named clip zone (`kdenlive:clipzones`) on the Project Bin clip for instant drag-and-drop recovery.

---

## 4. Subtitles & Timeline Guides Verification

- **Timeline Guides (`markerlistmodel.cpp`):**
  - Stored in `<property name="kdenlive:sequenceproperties.guides">` on the sequence `<tractor>` as a JSON array of objects:
    `[{"pos": <frame_int>, "comment": "<label>", "type": <category_index_0_to_8>}]`.
  - Categories are defined in `<property name="kdenlive:docproperties.guidesCategories">` on `main_bin`.
- **Subtitles (`subtitlemodel.cpp` & `documentvalidator.cpp`):**
  - A standard UTF-8 `.srt` sidecar named `<project_file>.kdenlive.srt` (plus a standalone `<project_file>.srt`) placed in the same directory as `<project_file>.kdenlive`, paired with `<property name="kdenlive:sequenceproperties.subtitlesList">[{"file": "<project_file>.kdenlive.srt", "id": 0, "name": "Subtitles"}]</property>` and an `avfilter.subtitles` `<filter>` on the sequence tractor, is natively loaded by Kdenlive 23.x–24.05 and automatically upgraded into Kdenlive's `.ass` subtitle layer by `DocumentValidator::convertSubtitles()` in Kdenlive 24.08–26.x while preserving the `.srt` file.

---

## 5. OpenCode Skill Discovery Rules Verification

Verified against the official OpenCode documentation (`https://opencode.ai/docs/skills/`):
- **Project discovery path:** `.opencode/skills/<name>/SKILL.md` (and cross-compatible `.agents/skills/<name>/SKILL.md` / `.claude/skills/<name>/SKILL.md`).
- **Filename:** Must be `SKILL.md` in all caps.
- **YAML Frontmatter:**
  - `name` (required): 1–64 chars, lowercase alphanumeric with single hyphen separators (`^[a-z0-9]+(-[a-z0-9]+)*$`), must not start or end with `-` or contain `--`, and **must exactly match the parent directory name**.
  - `description` (required): 1–1024 characters describing what the skill does and when to load it.
  - `license` (optional), `compatibility` (optional), `metadata` (optional string-to-string map).

---

## 6. Dependency Licenses & Packaging Corrections

| Dependency | Verified License | Corrections / Notes Discovered in Phase 0 |
| :--- | :--- | :--- |
| `av` (PyAV 18.x) | BSD-3-Clause (bundles LGPL FFmpeg libs) | Decodes video/audio and probes streams directly in Python without requiring a system-wide `ffmpeg` install. |
| `imageio-ffmpeg` | BSD-2-Clause (bundles static `ffmpeg` binary) | Provides a portable fallback `ffmpeg` executable inside the Python `.venv` when system `ffmpeg` is absent. |
| `faster-whisper` | MIT License (code & Whisper weights) | **Correction:** `faster_whisper.vad` already bundles the MIT-licensed Silero VAD v6 ONNX model (`silero_vad_v6.onnx`) and runs via `onnxruntime`. Using `faster_whisper.vad` eliminates the need for the standalone `silero-vad` PyPI package and its multi-gigabyte `torch`/`torchaudio` dependency. |
| `onnxruntime` | MIT License | Executes the bundled Silero VAD ONNX model on CPU in milliseconds. |
| `scenedetect` | BSD-3-Clause | **Correction:** `scenedetect` on PyPI depends on `opencv-python` (GUI build), which fails on headless Linux (`libGL.so.1` missing) unless `opencv-python-headless` is installed over it, and `scenedetect` also supports `backend="pyav"`. Our installer and scene analyzer handle both backends cleanly. |
| `opencv-python` / `opencv-python-headless` | Apache-2.0 (OpenCV) / MIT (wrapper) | Used for fast downscaled frame-motion analysis. |
| `numpy` | BSD-3-Clause | Used for RMS audio level, dBFS, and frame-delta calculations. |
| `pytest` | MIT License | Used for unit, integration, and end-to-end tests. |

---

## 7. Corrections to Earlier Research Claims

1. **CapCut / JianYing Claims:**
   - Earlier research cited third-party blog posts and community gists claiming specific CapCut Terms of Service update dates (April 15, 2026), specific Windows CapCut build numbers rejecting drafts (8.4, 8.7, 9.3), and JianYing 6.0+ AES encryption internals. Those came from secondary community sources (`wireflow.ai`, `renezander030/capcut-cli`), **not** primary official documentation.
   - **Corrected Position:** We do not rely on unverified claims about CapCut's internal encryption or TOS dates. Instead, Kdenlive is chosen because it is GPL-3.0 open-source software with an openly documented, plain-text MLT XML project format and publicly auditable C++ source code.
2. **MLT XML DTD Sufficiency:**
   - Earlier research suggested MLT DTD validation alone proves `.kdenlive` validity. Phase 0 proved that upstream `mlt-xml.dtd` is out of date with Kdenlive Generation 5 (`version="1.1"`) and does not check Kdenlive's `kdenlive:control_uuid`, per-track `<chain>` separation, or `AVSplit` group rules.
3. **Standalone `silero-vad` Package:**
   - Removed the standalone `silero-vad` PyPI dependency (which pulls in `torch`) in favor of `faster_whisper.vad` + `onnxruntime`.

---

## 8. Reproducible Local Kdenlive Validation Procedure (Windows 10/11)

Because the Arena cloud sandbox is a headless Linux container without a GUI or Kdenlive binary, actual GUI opening cannot be executed inside the sandbox. To validate any generated `.kdenlive` project in the Kdenlive desktop application on Windows:

1. Generate the Phase 0 minimal test project (or run the full workflow):
   ```powershell
   .\.venv\Scripts\python.exe scripts\verify_project.py --generate-minimal-sample .\validation_sample
   ```
2. Open Kdenlive (version 23.04 or newer, e.g., Kdenlive 24.x / 25.x / 26.x on Windows).
3. Click **File -> Open** and select `validation_sample\minimal_test_project.kdenlive`.
4. Confirm in Kdenlive:
   - **Project Bin** contains the source video clip (with saved clip zones for any removed segments) and `Sequence 1`.
   - **Timeline Tracks** contain separate video (`V1`, and `V2 - Review` in review-layers mode) and audio (`A1`, and `A2` for multi-track recordings) tracks.
   - **Review Track (`V2 - Review`)** is hidden (eye icon crossed out) and its clips are disabled (dimmed), so pressing `Space` plays only the active cut without review-layer bleed.
   - **Timeline Guides** appear along the top ruler with color-coded labels.
   - **Subtitle Track** loads the `.kdenlive.srt` sidecar captions synchronized with the timeline.
   - Exporting is performed manually by the user via **Project -> Render (Ctrl+Return)**.

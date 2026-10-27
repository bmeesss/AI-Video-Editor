---
name: verify-and-diagnose
description: Run environment diagnostics and validate generated .kdenlive projects against the MLT XML DTD and Kdenlive 1.1 C++ loader invariants (track chains, control UUIDs, A/V sync, review track muting, and SRT syntax).
license: MIT
compatibility: opencode
metadata:
  audience: gaming-creators
  workflow: phase-4-verification
---

# Verify and Diagnose (`verify-and-diagnose`)

## When to Use This Skill
Load this skill when:
- Verifying a generated `.kdenlive` project file to ensure it meets both MLT XML DTD structure and Kdenlive Generation 5 (`version="1.1"`, `meltBuilder.cpp`) semantic rules.
- Generating the Phase 0 minimal validation sample project for manual testing in Kdenlive.
- Diagnosing missing dependencies, hardware capabilities (RAM, CPU cores, CUDA availability), FFmpeg availability, or Kdenlive desktop installation paths.

## Important Verification Principle
Valid XML syntax alone does **not** prove a `.kdenlive` project is usable in Kdenlive. `verify_project.py` checks:
1. MLT XML DTD element ordering (`<property>*` before `<track>+` before `<filter|transition>*`).
2. Kdenlive `version="1.1"` `kdenlive:control_uuid` and `kdenlive:id` consistency between `main_bin` and per-track `<chain>` producers.
3. Strict per-track `<chain>` separation (`giveMasterAndGetTimelineProducer`).
4. Frame-exact A/V synchronization and chronological ordering across active tracks.
5. Review-layer protection (`hide="both"` + `<property name="hide">3</property>` + disabled clip state `set.test_audio=1, set.test_image=1`).
6. Source media file existence and MD5 checksum match.
7. `.srt` subtitle sidecar formatting and monotonic timestamps.

## Command Usage

### Run Environment Diagnostics
```powershell
# Windows PowerShell
.\.venv\Scripts\python.exe scripts\diagnose.py

# Linux / macOS
./.venv/bin/python scripts/diagnose.py
```

### Verify an Existing `.kdenlive` Project
```powershell
# Windows PowerShell
.\.venv\Scripts\python.exe scripts\verify_project.py ".\output\job_folder\project.kdenlive"

# Linux / macOS
./.venv/bin/python scripts/verify_project.py "./output/job_folder/project.kdenlive"
```

### Generate & Verify the Minimal Validation Sample Project
```powershell
# Windows PowerShell
.\.venv\Scripts\python.exe scripts\verify_project.py --generate-minimal-sample ".\output\validation_sample"

# Linux / macOS
./.venv/bin/python scripts/verify_project.py --generate-minimal-sample "./output/validation_sample"
```

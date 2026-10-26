"""Phase 3 tests: deterministic planner, review-layers vs ripple modes, Kdenlive 1.1 XML + SRT generation, and verifier."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List

import numpy as np
import pytest

from ai_video_editor.analyzer_pipeline import analyze_recording
from ai_video_editor.config import EditingConfig
from ai_video_editor.kdenlive_export import (
    export_kdenlive_project,
    frame_to_mlt_timecode,
    mlt_timecode_to_frame,
    normalize_kdenlive_path,
)
from ai_video_editor.models import TranscriptSegment, TranscriptWord
from ai_video_editor.planner import build_edit_decision_model
from ai_video_editor.project_verifier import verify_kdenlive_project
from ai_video_editor.safety import (
    capture_source_snapshot,
    create_unique_job_directory,
    verify_source_untouched,
)
from ai_video_editor.speech_transcriber import WhisperRuntimeSelection

from .synthetic_media import create_synthetic_gaming_video


def _mock_gaming_transcriber(
    _samples: np.ndarray,
    _cfg: EditingConfig,
    _sel: WhisperRuntimeSelection,
) -> List[TranscriptSegment]:
    return [
        TranscriptSegment(
            segment_id=0,
            start=0.5,
            end=4.5,
            text="Welcome back to the channel, let's jump into the boss arena!",
            confidence=0.95,
            words=[TranscriptWord("Welcome", 0.5, 0.9, 0.95)],
        ),
        TranscriptSegment(
            segment_id=1,
            start=10.2,
            end=14.5,
            text="Phase two just started, dodge the laser sweep!",
            confidence=0.93,
            words=[TranscriptWord("Phase", 10.2, 10.6, 0.93)],
        ),
        TranscriptSegment(
            segment_id=2,
            start=18.2,
            end=19.8,
            text="We got the victory, thanks for watching!",
            confidence=0.96,
            words=[TranscriptWord("We", 18.2, 18.5, 0.96)],
        ),
    ]


@pytest.fixture(scope="module")
def unicode_gaming_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    d = tmp_path_factory.mktemp("phase3_media")
    vid = d / "Jürgen & 太郎 — Boss Fight #1 (60fps).mp4"
    return create_synthetic_gaming_video(vid, duration_seconds=20, fps=30, dual_audio_tracks=True)


def test_mlt_timecode_roundtrip() -> None:
    for fps_num, fps_den in [(30, 1), (60, 1), (25, 1), (30000, 1001)]:
        for f in [0, 1, 15, 29, 30, 150, 599, 1800]:
            tc = frame_to_mlt_timecode(f, fps_num, fps_den)
            back = mlt_timecode_to_frame(tc, fps_num, fps_den)
            assert back == f


def test_windows_and_unicode_path_normalization() -> None:
    win_path = r"C:\Users\Jürgen 太郎\Videos\OBS Recordings\match 01.mp4"
    norm = normalize_kdenlive_path(win_path)
    assert "\\" not in norm
    assert norm == "C:/Users/Jürgen 太郎/Videos/OBS Recordings/match 01.mp4"


def test_review_layers_timeline_generation_and_verification(
    unicode_gaming_video: Path,
    tmp_path: Path,
) -> None:
    snap = capture_source_snapshot(unicode_gaming_video)
    cfg = EditingConfig(
        timeline_mode="review-layers",
        preserve_intro_seconds=4.5,
        min_silence_duration_seconds=2.5,
        whisper_model="base",
        whisper_device="cpu",
    )
    report = analyze_recording(unicode_gaming_video, config=cfg, custom_transcriber=_mock_gaming_transcriber)
    edm = build_edit_decision_model(report, title="Jürgen's Epic Boss Run!", config=cfg)

    # 1. Must preserve intro at frame 0
    v1_track = next(t for t in edm.tracks if t.track_id == "V1")
    assert v1_track.clips[0].source_in_frame == 0
    assert v1_track.clips[0].timeline_in_frame == 0

    # 2. Must remove the dead-air section between ~5s and ~10s
    assert len(edm.removed_clips_summary) >= 1
    dead_air_cut = edm.removed_clips_summary[0]
    assert 4.5 <= dead_air_cut.source_in_sec <= 6.0
    assert 9.0 <= dead_air_cut.source_out_sec <= 10.5

    # 3. Must preserve the 15s-18s silent commentary section because visual gameplay motion is active!
    kept_covering_16s = [
        c for c in v1_track.clips if c.source_in_sec <= 16.0 <= c.source_out_sec
    ]
    assert len(kept_covering_16s) == 1

    # 4. Clips must be in strict chronological order
    for prev_c, next_c in zip(v1_track.clips, v1_track.clips[1:]):
        assert prev_c.source_out_frame < next_c.source_in_frame
        assert prev_c.timeline_out_frame + 1 == next_c.timeline_in_frame

    # 5. Export .kdenlive and .srt into a unique job directory
    job_dir = create_unique_job_directory(tmp_path / "jobs", edm.title, source_path=unicode_gaming_video)
    kdenlive_path, kdenlive_srt, standalone_srt = export_kdenlive_project(edm, job_dir, "edit_review")

    assert kdenlive_path.exists()
    assert kdenlive_srt is not None and kdenlive_srt.exists()
    assert standalone_srt is not None and standalone_srt.exists()
    assert verify_source_untouched(snap) is True

    # 6. Verify Kdenlive project with MLT DTD + C++ meltBuilder semantic verifier
    verification = verify_kdenlive_project(kdenlive_path, check_source_media_exists=True)
    assert verification.is_valid is True, f"Verification errors: {verification.errors}"
    assert verification.dtd_valid is True
    assert verification.kdenlive_semantics_valid is True
    assert verification.source_media_verified is True
    assert verification.subtitles_verified is True

    # 7. Confirm clipzones exist on chain_bin_0 for instant drag-and-drop recovery in Kdenlive
    xml_root = ET.fromstring(kdenlive_path.read_text(encoding="utf-8"))
    clipzones_prop = xml_root.find("./chain[@id='chain_bin_0']/property[@name='kdenlive:clipzones']")
    assert clipzones_prop is not None and clipzones_prop.text
    zones = json.loads(clipzones_prop.text)
    assert len(zones) == len(edm.removed_clips_summary)


def test_clean_ripple_timeline_mode_excludes_review_tracks(
    unicode_gaming_video: Path,
    tmp_path: Path,
) -> None:
    cfg = EditingConfig(
        timeline_mode="ripple",
        preserve_intro_seconds=4.5,
        min_silence_duration_seconds=2.5,
        whisper_model="base",
        whisper_device="cpu",
    )
    report = analyze_recording(unicode_gaming_video, config=cfg, custom_transcriber=_mock_gaming_transcriber)
    edm = build_edit_decision_model(report, title="Clean Ripple Run", config=cfg)

    review_tracks = [t for t in edm.tracks if t.is_review_track]
    assert review_tracks == []
    assert len(edm.tracks) == 3  # V1, A1, A2

    job_dir = create_unique_job_directory(tmp_path / "jobs_ripple", edm.title, source_path=unicode_gaming_video)
    kdenlive_path, _, _ = export_kdenlive_project(edm, job_dir, "edit_ripple")
    verification = verify_kdenlive_project(kdenlive_path, check_source_media_exists=True)
    assert verification.is_valid is True, f"Verification errors: {verification.errors}"


def test_low_confidence_halt_stops_export(
    unicode_gaming_video: Path,
    tmp_path: Path,
) -> None:
    cfg = EditingConfig(
        timeline_mode="review-layers",
        min_overall_edit_confidence=0.99,  # Unreachably high threshold to trigger halt
    )
    report = analyze_recording(unicode_gaming_video, config=cfg, custom_transcriber=_mock_gaming_transcriber)
    edm = build_edit_decision_model(report, title="Low Confidence Test", config=cfg)

    assert edm.low_confidence_halt is True
    assert any("below configured minimum" in w for w in edm.warnings)

    with pytest.raises(RuntimeError, match="Refusing to generate .kdenlive project"):
        export_kdenlive_project(edm, tmp_path / "should_not_export", "halted")

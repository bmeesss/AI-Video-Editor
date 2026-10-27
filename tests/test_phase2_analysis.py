"""Phase 2 tests: media probing, multi-track audio role detection, VAD, visual scene/motion, and JSON report."""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pytest

from ai_video_editor.analyzer_pipeline import analyze_recording
from ai_video_editor.config import EditingConfig
from ai_video_editor.media_probe import probe_media
from ai_video_editor.models import AnalysisReport, TranscriptSegment, TranscriptWord
from ai_video_editor.safety import capture_source_snapshot, verify_source_untouched
from ai_video_editor.speech_transcriber import (
    WhisperRuntimeSelection,
    detect_stumble_in_segment,
    select_whisper_model_and_device,
)

from .synthetic_media import create_synthetic_gaming_video


@pytest.fixture(scope="module")
def sample_dual_track_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    media_dir = tmp_path_factory.mktemp("synthetic_media")
    video_path = media_dir / "Jürgen's Boss Fight (Dual Track).mp4"
    return create_synthetic_gaming_video(video_path, duration_seconds=20, fps=30, dual_audio_tracks=True)


def test_probe_media_detects_streams_and_multi_track_roles(sample_dual_track_video: Path) -> None:
    snap_before = capture_source_snapshot(sample_dual_track_video)
    info = probe_media(sample_dual_track_video)
    assert verify_source_untouched(snap_before) is True

    assert info.duration_seconds == pytest.approx(20.0, abs=0.25)
    assert len(info.video_streams) == 1
    v = info.video_streams[0]
    assert v.width == 640
    assert v.height == 360
    assert v.frame_rate_num == 30
    assert v.frame_rate_den == 1
    assert v.is_vfr is False

    assert len(info.audio_streams) == 2
    roles = [a.inferred_role for a in info.audio_streams]
    assert "microphone" in roles
    assert "gameplay" in roles
    assert info.primary_mic_stream is not None
    assert info.primary_mic_stream.inferred_role == "microphone"


def test_whisper_model_and_device_selection_honors_config_and_fallback() -> None:
    cfg_auto = EditingConfig(whisper_model="auto", whisper_device="cpu")
    sel_auto = select_whisper_model_and_device(cfg_auto)
    assert sel_auto.model_name in ("base", "small")
    assert sel_auto.device == "cpu"
    assert sel_auto.compute_type == "int8"

    cfg_base = EditingConfig(whisper_model="base", whisper_device="cuda")
    sel_base = select_whisper_model_and_device(cfg_base)
    assert sel_base.model_name == "base"
    # In this headless CPU container without CUDA, it must fall back cleanly to cpu/int8
    if not sel_base.cuda_available:
        assert sel_base.device == "cpu"
        assert sel_base.compute_type == "int8"


def test_conservative_stumble_detection() -> None:
    # Pure hesitation filler segment -> flagged as stumble candidate
    is_stumble, conf = detect_stumble_in_segment(
        "uh um er",
        [
            TranscriptWord("uh", 5.0, 5.2, 0.9),
            TranscriptWord("um", 5.2, 5.4, 0.9),
            TranscriptWord("er", 5.4, 5.6, 0.9),
        ],
        segment_confidence=0.9,
    )
    assert is_stumble is True
    assert conf >= 0.78

    # Expressive gaming repetition ("go go go!") -> MUST NOT be flagged as a stumble
    is_stumble_expr, conf_expr = detect_stumble_in_segment(
        "go go go!",
        [
            TranscriptWord("go", 10.0, 10.2, 0.95),
            TranscriptWord("go", 10.2, 10.4, 0.95),
            TranscriptWord("go!", 10.4, 10.7, 0.95),
        ],
        segment_confidence=0.95,
    )
    assert is_stumble_expr is False
    assert conf_expr == 0.0


def test_full_analysis_pipeline_produces_inspectable_report(
    sample_dual_track_video: Path,
    tmp_path: Path,
) -> None:
    def mock_transcriber(
        _samples: np.ndarray,
        _cfg: EditingConfig,
        _sel: WhisperRuntimeSelection,
    ) -> List[TranscriptSegment]:
        return [
            TranscriptSegment(
                segment_id=0,
                start=0.5,
                end=4.5,
                text="Welcome back everyone, today we are taking on the boss!",
                confidence=0.94,
                words=[
                    TranscriptWord("Welcome", 0.5, 1.0, 0.95),
                    TranscriptWord("back", 1.0, 1.4, 0.95),
                ],
            ),
            TranscriptSegment(
                segment_id=1,
                start=10.2,
                end=14.5,
                text="Watch out for that second phase attack right there!",
                confidence=0.92,
            ),
            TranscriptSegment(
                segment_id=2,
                start=18.2,
                end=19.8,
                text="GG, that was a clean run!",
                confidence=0.96,
            ),
        ]

    snap_before = capture_source_snapshot(sample_dual_track_video)
    report = analyze_recording(
        sample_dual_track_video,
        config=EditingConfig(whisper_model="base", whisper_device="cpu"),
        custom_transcriber=mock_transcriber,
    )
    assert verify_source_untouched(snap_before) is True

    # Speech intervals must detect the commentary bursts and NOT the 5.0-10.0s dead air
    assert len(report.speech_intervals) >= 3
    dead_air_speech = [
        s for s in report.speech_intervals if s.start >= 5.2 and s.end <= 9.8
    ]
    assert dead_air_speech == []

    # Visual analysis must detect multiple scene transitions and active gameplay motion between 15-18s
    assert len(report.scene_boundaries) >= 2
    active_15_18 = [
        m for m in report.motion_windows if 15.0 <= m.start < 18.0 and m.is_active_gameplay
    ]
    assert len(active_15_18) >= 1

    # Round-trip JSON persistence test
    report_path = tmp_path / "analysis_report.json"
    report.save_json(report_path)
    loaded = AnalysisReport.load_json(report_path)
    assert loaded.media_info.file_hash_sha256 == report.media_info.file_hash_sha256
    assert len(loaded.transcript_segments) == 3
    assert loaded.overall_analysis_confidence >= 0.75

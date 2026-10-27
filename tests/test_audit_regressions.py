"""Independent code-audit regression tests covering file safety, planner edge cases, hardware selection, and Kdenlive verification."""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
from typing import List

import numpy as np
import pytest

from ai_video_editor.cli import main as cli_main
from ai_video_editor.config import ConfigValidationError, EditingConfig
from ai_video_editor.kdenlive_export import export_kdenlive_project
from ai_video_editor.models import (
    AnalysisReport,
    AudioEnergyWindow,
    AudioStreamInfo,
    MediaInfo,
    MotionWindow,
    SpeechInterval,
    TranscriptSegment,
    TranscriptWord,
    VideoStreamInfo,
)
from ai_video_editor.planner import build_edit_decision_model
from ai_video_editor.project_verifier import verify_kdenlive_project, verify_srt_file
from ai_video_editor.safety import (
    SafetyViolationError,
    capture_source_snapshot,
    create_unique_job_directory,
    sanitize_filename_slug,
    verify_source_untouched,
)
from ai_video_editor.speech_transcriber import (
     select_whisper_model_and_device,
    transcribe_audio_stream,
)

from .synthetic_media import create_synthetic_gaming_video


def _make_dummy_media_info(source_path: Path, duration_sec: float = 20.0, fps: int = 30) -> MediaInfo:
    snap = capture_source_snapshot(source_path)
    return MediaInfo(
        path=str(source_path.resolve()),
        filename=source_path.name,
        container_format="mp4",
        duration_seconds=duration_sec,
        file_size_bytes=snap.size_bytes,
        file_hash_md5=snap.md5_hex,
        file_hash_sha256=snap.sha256_hex,
        video_streams=[
            VideoStreamInfo(
                stream_index=0,
                codec_name="h264",
                width=1920,
                height=1080,
                frame_rate_num=fps,
                frame_rate_den=1,
                fps=float(fps),
                duration_seconds=duration_sec,
                total_frames=int(round(duration_sec * fps)),
            )
        ],
        audio_streams=[
            AudioStreamInfo(
                stream_index=1,
                audio_track_order=0,
                codec_name="aac",
                sample_rate=48000,
                channels=2,
                duration_seconds=duration_sec,
                title="Microphone",
                inferred_role="microphone",
            ),
            AudioStreamInfo(
                stream_index=2,
                audio_track_order=1,
                codec_name="aac",
                sample_rate=48000,
                channels=2,
                duration_seconds=duration_sec,
                title="Game Audio",
                inferred_role="gameplay",
            ),
        ],
    )


# ---------------------------------------------------------------------------
# 1. File Safety & Non-Overwrite Regressions
# ---------------------------------------------------------------------------


def test_save_json_never_overwrites_existing_files_or_source(tmp_path: Path) -> None:
    src = tmp_path / "source.mp4"
    src.write_bytes(b"source_video_bytes_12345")
    media = _make_dummy_media_info(src)

    report = AnalysisReport(
        schema_version="1.0.0",
        created_at_utc="2026-10-26T00:00:00Z",
        media_info=media,
        whisper_model_used="base",
        whisper_device_used="cpu",
        transcription_status="completed",
    )

    # Attempting to save report over the source file must raise SafetyViolationError
    with pytest.raises(SafetyViolationError, match="Refusing to write to source"):
        report.save_json(src)

    out_json = tmp_path / "report.json"
    report.save_json(out_json)
    assert out_json.is_file()

    # Second call to save_json on the same file must raise SafetyViolationError
    with pytest.raises(SafetyViolationError, match="Refusing to overwrite existing file"):
        report.save_json(out_json)

    edm = build_edit_decision_model(report, title="Test Save JSON")
    edm_json = tmp_path / "edm.json"
    edm.save_json(edm_json)
    with pytest.raises(SafetyViolationError, match="Refusing to overwrite existing file"):
        edm.save_json(edm_json)


def test_missing_empty_and_ancestor_source_file_guards(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.mp4"
    with pytest.raises(SafetyViolationError, match="does not exist"):
        capture_source_snapshot(missing)

    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    with pytest.raises(SafetyViolationError, match="empty"):
        capture_source_snapshot(empty)

    src = tmp_path / "real_video.mp4"
    src.write_bytes(b"valid_bytes")

    # Attempting to create an output directory inside a regular file path must fail
    with pytest.raises(SafetyViolationError):
        create_unique_job_directory(src / "subdir", "Title", source_path=src)


def test_windows_reserved_names_with_extensions_sanitized() -> None:
    for reserved in ("CON.txt", "nul.json", "COM1.mp4", "LPT9.kdenlive", "aux.srt"):
        slug = sanitize_filename_slug(reserved)
        assert slug.startswith("project_")


def test_cli_analyze_output_report_never_overwrites_file_or_nonexistent_dir(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    src = create_synthetic_gaming_video(
        tmp_path / "clip.mp4",
        duration_seconds=6,
        fps=30,
        dual_audio_tracks=False,
    )
    snap = capture_source_snapshot(src)

    # Case A: --output-report points to a non-existent directory path (no .json suffix)
    out_dir = tmp_path / "fresh_output_dir"
    rc1 = cli_main(["analyze", "--input", str(src), "--output-report", str(out_dir)])
    assert rc1 == 0
    res1 = json.loads(capsys.readouterr().out)

    # Calling a second time with the exact same directory path must create a second unique folder
    rc2 = cli_main(["analyze", "--input", str(src), "--output-report", str(out_dir)])
    assert rc2 == 0
    res2 = json.loads(capsys.readouterr().out)
    assert res1["analysis_report_path"] != res2["analysis_report_path"]

    # Case B: --output-report points to an existing .json file -> must fail with exit code 1 and not overwrite
    existing_json = tmp_path / "custom_report.json"
    existing_json.write_text('{"keep": "original"}', encoding="utf-8")
    rc3 = cli_main(["analyze", "--input", str(src), "--output-report", str(existing_json)])
    assert rc3 == 1
    assert existing_json.read_text(encoding="utf-8") == '{"keep": "original"}'
    assert verify_source_untouched(snap) is True


# ---------------------------------------------------------------------------
# 2. Montage Quality & Planner Regressions
# ---------------------------------------------------------------------------


def test_continuous_background_game_audio_does_not_block_dead_air_cuts(tmp_path: Path) -> None:
    src = tmp_path / "dual_track.mp4"
    src.write_bytes(b"dummy_video_content")
    media = _make_dummy_media_info(src, duration_sec=20.0, fps=30)

    # Track 2 (Game Audio, stream_index=2) has continuous -26 dBFS background music/ambiance across 0..20s
    game_windows = [
        AudioEnergyWindow(
            start=float(t),
            end=float(t + 1),
            rms_dbfs=-26.0,
            peak_dbfs=-20.0,
            is_quiet=False,
            is_transient_peak=False,
            stream_index=2,
        )
        for t in range(20)
    ]
    # Track 1 (Microphone, stream_index=1) has speech only at [0..4s] and [14..20s], with 10s dead air [4..14s]
    report = AnalysisReport(
        schema_version="1.0.0",
        created_at_utc="2026-10-26T00:00:00Z",
        media_info=media,
        whisper_model_used="base",
        whisper_device_used="cpu",
        transcription_status="completed",
        speech_intervals=[
            SpeechInterval(start=0.5, end=4.0, confidence=0.95, stream_index=1),
            SpeechInterval(start=14.0, end=19.5, confidence=0.95, stream_index=1),
        ],
        audio_energy_windows=game_windows,
    )

    cfg = EditingConfig(preserve_intro_seconds=4.0, min_silence_duration_seconds=2.5)
    edm = build_edit_decision_model(report, title="Background Game Audio Test", config=cfg)

    # The 10-second commentary dead air between 4s and 14s MUST be cut despite -26 dBFS ambient game audio!
    assert len(edm.removed_clips_summary) == 1
    cut = edm.removed_clips_summary[0]
    assert cut.source_in_sec == pytest.approx(4.45, abs=0.1)
    assert cut.source_out_sec == pytest.approx(13.65, abs=0.1)


def test_min_silence_threshold_is_strictly_respected(tmp_path: Path) -> None:
    src = tmp_path / "min_silence.mp4"
    src.write_bytes(b"dummy_video_content")
    media = _make_dummy_media_info(src, duration_sec=20.0, fps=30)

    # Pause 1: 4.0s -> 6.3s (2.3s silence, shorter than --min-silence 2.5s) -> MUST BE KEPT
    # Pause 2: 10.0s -> 12.8s (2.8s silence, longer than --min-silence 2.5s) -> MUST BE CUT
    report = AnalysisReport(
        schema_version="1.0.0",
        created_at_utc="2026-10-26T00:00:00Z",
        media_info=media,
        whisper_model_used="base",
        whisper_device_used="cpu",
        transcription_status="completed",
        speech_intervals=[
            SpeechInterval(start=0.0, end=4.0, confidence=0.95, stream_index=1),
            SpeechInterval(start=6.3, end=10.0, confidence=0.95, stream_index=1),
            SpeechInterval(start=12.8, end=20.0, confidence=0.95, stream_index=1),
        ],
    )

    cfg = EditingConfig(preserve_intro_seconds=2.0, min_silence_duration_seconds=2.5)
    edm = build_edit_decision_model(report, title="Min Silence Threshold Test", config=cfg)

    assert len(edm.removed_clips_summary) == 1
    cut = edm.removed_clips_summary[0]
    assert 10.0 <= cut.source_in_sec <= 10.6
    assert 12.3 <= cut.source_out_sec <= 12.8


def test_micro_clip_borrows_only_needed_frames_and_trailing_cut_appears_on_review_track(
    tmp_path: Path,
) -> None:
    src = tmp_path / "trailing_cut.mp4"
    src.write_bytes(b"dummy_video_content")
    media = _make_dummy_media_info(src, duration_sec=20.0, fps=30)

    # Speech from 0..10s, then a brief 0.5s speech burst at 10.0..10.5s, followed by 9.5s trailing silence [10.5..20.0s]
    report = AnalysisReport(
        schema_version="1.0.0",
        created_at_utc="2026-10-26T00:00:00Z",
        media_info=media,
        whisper_model_used="base",
        whisper_device_used="cpu",
        transcription_status="completed",
        speech_intervals=[
            SpeechInterval(start=0.0, end=4.0, confidence=0.95, stream_index=1),
            SpeechInterval(start=9.0, end=9.3, confidence=0.95, stream_index=1),
        ],
    )

    cfg = EditingConfig(
        timeline_mode="review-layers",
        preserve_intro_seconds=3.0,
        min_silence_duration_seconds=2.5,
        min_kept_clip_duration_seconds=1.2,
    )
    edm = build_edit_decision_model(report, title="Micro Clip & Trailing Cut", config=cfg)

    # Both the middle silence and the long trailing silence must be removed (not flipped to 100% kept!)
    assert len(edm.removed_clips_summary) == 2
    assert edm.edited_duration_frames < 240  # Less than 8 seconds kept out of 20s

    # And the review track V2_Review must contain both removed cuts, clamped within edited_duration_frames
    v2_rev = next(t for t in edm.tracks if t.track_id == "V2_Review")
    assert len(v2_rev.clips) >= 1
    assert all(c.timeline_out_frame < edm.edited_duration_frames for c in v2_rev.clips)


# ---------------------------------------------------------------------------
# 3. Whisper & Hardware Selection Regressions (Ryzen 5 7535HS + RTX 2050 4GB + 8GB RAM)
# ---------------------------------------------------------------------------


def test_whisper_selection_for_ryzen5_7535hs_rtx2050_4gb_vram_8gb_ram() -> None:
    cfg = EditingConfig(whisper_model="auto", whisper_device="auto")

    # Scenario 1: User's laptop with CUDA active (7.4 GB usable system RAM, 12 threads, 4.0 GB VRAM RTX 2050)
    sel_cuda = select_whisper_model_and_device(
        cfg,
        system_ram_gb=7.4,
        cpu_cores=12,
        cuda_available=True,
        gpu_vram_gb=4.0,
    )
    assert sel_cuda.model_name == "base"
    assert sel_cuda.device == "cuda"
    assert sel_cuda.compute_type == "int8"

    # Scenario 2: User's laptop when cuBLAS DLLs are missing (falls back to CPU int8 + 'base')
    sel_cpu = select_whisper_model_and_device(
        cfg,
        system_ram_gb=7.4,
        cpu_cores=12,
        cuda_available=False,
        gpu_vram_gb=4.0,
    )
    assert sel_cpu.model_name == "base"
    assert sel_cpu.device == "cpu"
    assert sel_cpu.compute_type == "int8"

    # Scenario 3: High-end desktop (32 GB RAM, 16 cores, 12 GB VRAM GPU) -> selects 'small' + 'float16'
    sel_desktop = select_whisper_model_and_device(
        cfg,
        system_ram_gb=32.0,
        cpu_cores=16,
        cuda_available=True,
        gpu_vram_gb=12.0,
    )
    assert sel_desktop.model_name == "small"
    assert sel_desktop.device == "cuda"
    assert sel_desktop.compute_type == "float16"


def test_whisper_lazy_cuda_error_during_transcribe_falls_back_to_cpu_int8(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: List[tuple[str, str, str]] = []

    class FakeWhisperModel:
        def __init__(self, model_name: str, device: str = "cpu", compute_type: str = "int8", **_kwargs: object) -> None:
            self.model_name = model_name
            self.device = device
            self.compute_type = compute_type
            calls.append((model_name, device, compute_type))

        def transcribe(self, *_args: object, **_kwargs: object) -> tuple[object, object]:
            if self.device == "cuda":
                # Simulate CTranslate2 lazy failure when cublas64_12.dll is missing or 4GB VRAM OOMs during iteration
                def _failing_gen() -> object:
                    raise RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")
                    yield  # pragma: no cover

                return _failing_gen(), SimpleNamespace()

            seg = SimpleNamespace(
                start=0.5,
                end=2.5,
                text="Fallback to CPU int8 succeeded!",
                avg_logprob=-0.15,
                no_speech_prob=0.01,
                words=[SimpleNamespace(word="Fallback", start=0.5, end=1.0, probability=0.95)],
            )
            return iter([seg]), SimpleNamespace()

    import faster_whisper

    monkeypatch.setattr(faster_whisper, "WhisperModel", FakeWhisperModel)
    monkeypatch.setattr(
        "ai_video_editor.speech_transcriber.probe_cuda_runtime",
        lambda: (True, "Simulated CUDA device"),
    )

    samples = np.zeros(16000 * 3, dtype=np.float32)
    cfg = EditingConfig(whisper_model="base", whisper_device="cuda")
    segments, final_sel, status, warnings = transcribe_audio_stream(samples, cfg)

    assert status == "completed_cpu_fallback"
    assert final_sel.device == "cpu"
    assert final_sel.compute_type == "int8"
    assert len(segments) == 1
    assert segments[0].text == "Fallback to CPU int8 succeeded!"
    assert len(calls) == 2
    assert calls[0][1] == "cuda"
    assert calls[1][1] == "cpu"
    assert any("falling back automatically to CPU (int8)" in w for w in warnings)


def test_local_whisper_model_directory_validation(tmp_path: Path) -> None:
    # Passing a regular file (like README.md) must be rejected
    regular_file = tmp_path / "not_a_dir.txt"
    regular_file.write_text("hello", encoding="utf-8")
    with pytest.raises(ConfigValidationError, match="model.bin"):
        EditingConfig(whisper_model=str(regular_file))

    # Passing an empty directory without model.bin must be rejected
    empty_dir = tmp_path / "empty_model_dir"
    empty_dir.mkdir()
    with pytest.raises(ConfigValidationError, match="model.bin"):
        EditingConfig(whisper_model=str(empty_dir))

    # Passing a directory with model.bin succeeds
    valid_dir = tmp_path / "valid_ct2_model"
    valid_dir.mkdir()
    (valid_dir / "model.bin").write_bytes(b"ct2_weights")
    cfg = EditingConfig(whisper_model=str(valid_dir))
    assert cfg.whisper_model == str(valid_dir)


# ---------------------------------------------------------------------------
# 4. Kdenlive Review-Layers vs Ripple Equivalence & Verifier Strictness
# ---------------------------------------------------------------------------


def test_review_layers_and_ripple_produce_identical_kept_content_and_verifier_catches_active_blanks(
    tmp_path: Path,
) -> None:
    src = tmp_path / "modes_compare.mp4"
    src.write_bytes(b"dummy_video_content")
    media = _make_dummy_media_info(src, duration_sec=20.0, fps=30)

    report = AnalysisReport(
        schema_version="1.0.0",
        created_at_utc="2026-10-26T00:00:00Z",
        media_info=media,
        whisper_model_used="base",
        whisper_device_used="cpu",
        transcription_status="completed",
        speech_intervals=[
            SpeechInterval(start=0.5, end=4.5, confidence=0.95, stream_index=1),
            SpeechInterval(start=10.5, end=19.5, confidence=0.95, stream_index=1),
        ],
    )

    edm_rev = build_edit_decision_model(
        report,
        title="Compare Modes",
        config=EditingConfig(timeline_mode="review-layers", preserve_intro_seconds=4.0),
    )
    edm_rip = build_edit_decision_model(
        report,
        title="Compare Modes",
        config=EditingConfig(timeline_mode="ripple", preserve_intro_seconds=4.0),
    )

    assert edm_rev.edited_duration_frames == edm_rip.edited_duration_frames
    for track_id in ("V1", "A1", "A2"):
        t_rev = next(t for t in edm_rev.tracks if t.track_id == track_id)
        t_rip = next(t for t in edm_rip.tracks if t.track_id == track_id)
        assert t_rev.clips == t_rip.clips

    # Export ripple project and verify that injecting a <blank> into an active track fails verification
    kdenlive_path, _, _ = export_kdenlive_project(edm_rip, tmp_path / "rip_out", "rip")
    assert verify_kdenlive_project(kdenlive_path).is_valid is True

    corrupted_xml = kdenlive_path.read_text(encoding="utf-8").replace(
        '<playlist id="playlist0">\n  <property name="kdenlive:audio_track">1</property>',
        '<playlist id="playlist0">\n  <property name="kdenlive:audio_track">1</property>\n  <blank length="00:00:01.000"/>',
    )
    corrupted_path = tmp_path / "rip_out" / "corrupted_blank.kdenlive"
    corrupted_path.write_text(corrupted_xml, encoding="utf-8")
    ver_bad = verify_kdenlive_project(corrupted_path)
    assert ver_bad.is_valid is False
    assert any("contains a <blank> gap" in e for e in ver_bad.errors)


def test_srt_verifier_catches_overlapping_cues(tmp_path: Path) -> None:
    bad_srt = tmp_path / "overlapping.srt"
    bad_srt.write_text(
        "1\n00:00:01,000 --> 00:00:04,000\nFirst line\n\n"
        "2\n00:00:03,500 --> 00:00:06,000\nOverlapping second line\n",
        encoding="utf-8",
    )
    ok, errs, _ = verify_srt_file(bad_srt)
    assert ok is False
    assert any("overlaps previous subtitle cue" in e for e in errs)


def test_stumble_removal_and_no_ghost_subtitle(tmp_path: Path) -> None:
    src = tmp_path / "stumble.mp4"
    src.write_bytes(b"dummy_video_content")
    media = _make_dummy_media_info(src, duration_sec=15.0, fps=30)

    report = AnalysisReport(
        schema_version="1.0.0",
        created_at_utc="2026-10-26T00:00:00Z",
        media_info=media,
        whisper_model_used="base",
        whisper_device_used="cpu",
        transcription_status="completed",
        speech_intervals=[
            SpeechInterval(start=0.0, end=15.0, confidence=0.95, stream_index=1),
        ],
        transcript_segments=[
            TranscriptSegment(
                segment_id=0,
                start=0.5,
                end=4.5,
                text="Intro commentary here.",
                confidence=0.95,
            ),
            TranscriptSegment(
                segment_id=1,
                start=6.0,
                end=7.5,
                text="uh um er",
                confidence=0.92,
                is_stumble_candidate=True,
                stumble_confidence=0.90,
            ),
            TranscriptSegment(
                segment_id=2,
                start=8.0,
                end=14.5,
                text="Clean follow-up sentence after stumble.",
                confidence=0.96,
            ),
        ],
    )

    cfg = EditingConfig(preserve_intro_seconds=4.0, remove_obvious_stumbles=True)
    edm = build_edit_decision_model(report, title="Stumble Cut Test", config=cfg)

    assert len(edm.removed_clips_summary) == 1
    assert edm.removed_clips_summary[0].decision_reason == "obvious_stumble"
    # Removed stumble must not appear in subtitles
    assert len(edm.subtitles) == 2
    assert all("uh um er" not in s.text for s in edm.subtitles)


def test_short_recording_and_missing_audio_or_video_streams(tmp_path: Path) -> None:
    src = tmp_path / "short_or_partial.mp4"
    src.write_bytes(b"dummy_video_content")
    snap = capture_source_snapshot(src)

    # 1. Recording shorter than preserve_intro_seconds (3s video vs 8s intro)
    short_media = _make_dummy_media_info(src, duration_sec=3.0, fps=30)
    short_report = AnalysisReport(
        schema_version="1.0.0",
        created_at_utc="2026-10-26T00:00:00Z",
        media_info=short_media,
        whisper_model_used="base",
        whisper_device_used="cpu",
        transcription_status="completed",
    )
    edm_short = build_edit_decision_model(short_report, title="Short Clip")
    assert edm_short.edited_duration_frames == 90
    assert edm_short.removed_clips_summary == []

    # 2. Video-only recording (no audio streams) -> still exports valid video-only Kdenlive project
    video_only_media = MediaInfo(
        path=str(src.resolve()),
        filename=src.name,
        container_format="mp4",
        duration_seconds=10.0,
        file_size_bytes=snap.size_bytes,
        file_hash_md5=snap.md5_hex,
        file_hash_sha256=snap.sha256_hex,
        video_streams=short_media.video_streams,
        audio_streams=[],
    )
    vo_report = AnalysisReport(
        schema_version="1.0.0",
        created_at_utc="2026-10-26T00:00:00Z",
        media_info=video_only_media,
        whisper_model_used="none",
        whisper_device_used="cpu",
        transcription_status="no_audio",
        motion_windows=[MotionWindow(start=0.0, end=10.0, motion_score=8.0, is_active_gameplay=True)],
    )
    edm_vo = build_edit_decision_model(vo_report, title="Video Only Clip")
    vo_kdenlive, _, _ = export_kdenlive_project(edm_vo, tmp_path / "vo_out", "vo")
    assert verify_kdenlive_project(vo_kdenlive).is_valid is True

    # 3. Audio-only file (missing video stream) -> triggers low_confidence_halt
    audio_only_media = MediaInfo(
        path=str(src.resolve()),
        filename=src.name,
        container_format="mp4",
        duration_seconds=10.0,
        file_size_bytes=snap.size_bytes,
        file_hash_md5=snap.md5_hex,
        file_hash_sha256=snap.sha256_hex,
        video_streams=[],
        audio_streams=short_media.audio_streams,
    )
    ao_report = AnalysisReport(
        schema_version="1.0.0",
        created_at_utc="2026-10-26T00:00:00Z",
        media_info=audio_only_media,
        whisper_model_used="base",
        whisper_device_used="cpu",
        transcription_status="completed",
        overall_analysis_confidence=0.40,
    )
    edm_ao = build_edit_decision_model(ao_report, title="Audio Only Clip")
    assert edm_ao.low_confidence_halt is True


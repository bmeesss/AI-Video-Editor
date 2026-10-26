"""Phase 1 unit tests: safety rules, non-destructive guards, config validation, and data models."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from ai_video_editor.config import ConfigValidationError, EditingConfig
from ai_video_editor.models import (
    AudioStreamInfo,
    MediaInfo,
    SpeechInterval,
    TranscriptSegment,
    TranscriptWord,
    VideoStreamInfo,
)
from ai_video_editor.safety import (
    SafetyViolationError,
    capture_source_snapshot,
    create_unique_job_directory,
    safe_write_text,
    sanitize_filename_slug,
    verify_source_untouched,
)


def test_source_snapshot_and_verification_detects_tampering(tmp_path: Path) -> None:
    source_file = tmp_path / "Jürgen's Gameplay #1 (1080p60).mp4"
    original_bytes = b"\x00\x00\x00\x18ftypmp42" + b"A" * 4096
    source_file.write_bytes(original_bytes)

    snap = capture_source_snapshot(source_file)
    assert snap.size_bytes == len(original_bytes)
    assert verify_source_untouched(snap) is True

    # Modify a single byte in the source file and verify tampering is caught
    tampered = bytearray(original_bytes)
    tampered[-1] = ord("B")
    source_file.write_bytes(bytes(tampered))

    with pytest.raises(SafetyViolationError, match="content hash changed"):
        verify_source_untouched(snap)


def test_source_snapshot_detects_deletion_or_size_change(tmp_path: Path) -> None:
    source_file = tmp_path / "recording.mkv"
    source_file.write_bytes(b"1234567890")
    snap = capture_source_snapshot(source_file)

    source_file.write_bytes(b"12345")
    with pytest.raises(SafetyViolationError, match="size changed"):
        verify_source_untouched(snap)

    source_file.unlink()
    with pytest.raises(SafetyViolationError, match="disappeared"):
        verify_source_untouched(snap)


def test_unique_job_directory_never_overwrites_existing(tmp_path: Path) -> None:
    fixed_time = dt.datetime(2026, 10, 26, 14, 30, 0, tzinfo=dt.timezone.utc)
    base_dir = tmp_path / "Outputs"

    dir1 = create_unique_job_directory(base_dir, "Elden Ring Boss Run!", timestamp=fixed_time)
    dir2 = create_unique_job_directory(base_dir, "Elden Ring Boss Run!", timestamp=fixed_time)
    dir3 = create_unique_job_directory(base_dir, "Elden Ring Boss Run!", timestamp=fixed_time)

    assert dir1 != dir2 != dir3
    assert dir1.exists() and dir2.exists() and dir3.exists()
    assert dir1.name.endswith("20261026_143000")
    assert dir2.name.endswith("20261026_143000_001")
    assert dir3.name.endswith("20261026_143000_002")


def test_safe_write_text_never_overwrites_existing_or_source(tmp_path: Path) -> None:
    source_file = tmp_path / "source.mp4"
    source_file.write_bytes(b"raw_video_data")

    # Attempting to overwrite source file directly must fail
    with pytest.raises(SafetyViolationError, match="Refusing to write to source"):
        safe_write_text(source_file, "corrupted", source_path=source_file)

    project_file = tmp_path / "job1" / "edit.kdenlive"
    written = safe_write_text(project_file, "<mlt/>", source_path=source_file)
    assert written.read_text(encoding="utf-8") == "<mlt/>"

    # Attempting to write to the same project file again must fail
    with pytest.raises(SafetyViolationError, match="Refusing to overwrite existing file"):
        safe_write_text(project_file, "<mlt>overwrite</mlt>", source_path=source_file)

    # Original source file remains untouched
    assert source_file.read_bytes() == b"raw_video_data"


def test_sanitize_filename_slug_handles_windows_reserved_and_unicode() -> None:
    assert sanitize_filename_slug("CON") == "project_CON"
    assert sanitize_filename_slug("NUL") == "project_NUL"
    assert sanitize_filename_slug('My: "Best" <Game> | Run? *1*') == "My_Best_Game_Run_1"
    slug_unicode = sanitize_filename_slug("Jürgen & 太郎 — Let's Play #42")
    assert "Jürgen" in slug_unicode
    assert "太郎" in slug_unicode


def test_editing_config_validation_enforces_rules() -> None:
    cfg = EditingConfig()
    assert cfg.timeline_mode == "review-layers"
    assert cfg.preserve_intro_seconds == 8.0
    assert cfg.enable_background_music is False
    assert cfg.enable_auto_zooms_or_effects is False

    with pytest.raises(ConfigValidationError, match="timeline_mode"):
        EditingConfig(timeline_mode="invalid")  # type: ignore[arg-type]

    with pytest.raises(ConfigValidationError, match="whisper_model"):
        EditingConfig(whisper_model="large-v3")

    with pytest.raises(ConfigValidationError, match="enable_background_music"):
        EditingConfig(enable_background_music=True)

    with pytest.raises(ConfigValidationError, match="enable_auto_zooms_or_effects"):
        EditingConfig(enable_auto_zooms_or_effects=True)

    with pytest.raises(ConfigValidationError, match="Unknown configuration keys"):
        EditingConfig.from_dict({"non_existent_option": 123})


def test_models_roundtrip_serialization() -> None:
    media = MediaInfo(
        path="C:\\Users\\Gamer\\Videos\\match_01.mp4",
        filename="match_01.mp4",
        container_format="mp4",
        duration_seconds=30.0,
        file_size_bytes=102400,
        file_hash_md5="d41d8cd98f00b204e9800998ecf8427e",
        file_hash_sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        video_streams=[
            VideoStreamInfo(
                stream_index=0,
                codec_name="h264",
                width=1920,
                height=1080,
                frame_rate_num=60,
                frame_rate_den=1,
                fps=60.0,
                duration_seconds=30.0,
                total_frames=1800,
            )
        ],
        audio_streams=[
            AudioStreamInfo(
                stream_index=1,
                audio_track_order=0,
                codec_name="aac",
                sample_rate=48000,
                channels=2,
                duration_seconds=30.0,
                title="Microphone",
                inferred_role="microphone",
            )
        ],
    )
    restored = MediaInfo.from_dict(media.to_dict())
    assert restored == media
    assert restored.primary_video is not None
    assert restored.primary_mic_stream is not None
    assert restored.primary_mic_stream.inferred_role == "microphone"

    seg = TranscriptSegment(
        segment_id=0,
        start=1.0,
        end=2.5,
        text="Hello everyone",
        confidence=0.95,
        words=[TranscriptWord(word="Hello", start=1.0, end=1.5, probability=0.96)],
    )
    assert TranscriptSegment.from_dict(seg.to_dict()) == seg
    interval = SpeechInterval(start=1.0, end=2.5, confidence=0.9)
    assert interval.duration == pytest.approx(1.5)

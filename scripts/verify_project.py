#!/usr/bin/env python3
"""Standalone project verifier and Phase 0 minimal validation sample generator."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# Ensure src/ is importable even if invoked directly
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import imageio_ffmpeg  # noqa: E402

from ai_video_editor.kdenlive_export import export_kdenlive_project  # noqa: E402
from ai_video_editor.media_probe import probe_media  # noqa: E402
from ai_video_editor.models import (  # noqa: E402
    EditDecisionModel,
    SubtitleEntry,
    TimelineClip,
    TimelineMarker,
    TimelineTrack,
)
from ai_video_editor.project_verifier import verify_kdenlive_project  # noqa: E402
from ai_video_editor.safety import create_unique_job_directory, safe_write_text  # noqa: E402


def _create_minimal_sample_source_video(video_path: Path) -> Path:
    """Create a 12-second 30fps dual-audio-track MP4 sample video using bundled ffmpeg."""
    video_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    vf_expr = (
        "color=c=0x182030:s=640x360:r=30:d=12,"
        "drawbox=enable='between(t,0,4)':x='mod(t*120,500)':y=80:w=120:h=120:c=dodgerblue@1.0:t=fill,"
        "drawbox=enable='between(t,4,8)':x=0:y=0:w=640:h=360:c=0x101015@1.0:t=fill,"
        "drawbox=enable='between(t,8,12)':x='mod(t*180,500)':y=100:w=140:h=140:c=limegreen@1.0:t=fill,"
        "format=yuv420p[vout]"
    )
    mic_expr = (
        "aevalsrc='0.3*(sin(2*PI*220*t)+0.5*sin(2*PI*440*t))*(between(t,0.4,3.6)+between(t,8.2,11.6))'"
        ":s=48000:d=12,pan=stereo|c0=c0|c1=c0[mic]"
    )
    game_expr = (
        "aevalsrc='0.15*sin(2*PI*165*t)*(between(t,0,4)+between(t,8,12))'"
        ":s=48000:d=12,pan=stereo|c0=c0|c1=c0[game]"
    )
    cmd = [
        ffmpeg_exe,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-filter_complex",
        f"{vf_expr};{mic_expr};{game_expr}",
        "-map",
        "[vout]",
        "-map",
        "[mic]",
        "-map",
        "[game]",
        "-metadata:s:a:0",
        "title=Microphone",
        "-metadata:s:a:1",
        "title=Game Audio",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-g",
        "30",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        str(video_path),
    ]
    subprocess.run(cmd, check=True)
    return video_path


def generate_and_verify_minimal_sample(base_output_dir: Path) -> int:
    """Generate the Phase 0 minimal validation test project and run full verification."""
    job_dir = create_unique_job_directory(base_output_dir, "phase0_minimal_validation_sample")
    source_video = _create_minimal_sample_source_video(job_dir / "sample_source.mp4")
    media_info = probe_media(source_video)

    # 12 seconds @ 30fps = 360 frames:
    # Clip 1 (kept): frames 0..119 (0.0s - 4.0s) -> timeline 0..119
    # Removed dead air: frames 120..239 (4.0s - 8.0s) -> placed on disabled review track at frame 120..239
    # Clip 2 (kept): frames 240..359 (8.0s - 12.0s) -> timeline 120..239
    def _make_kept_clips(prefix: str, ttype: str, a_idx: int | None) -> list[TimelineClip]:
        return [
            TimelineClip(
                clip_id=f"{prefix}_000",
                source_path=media_info.path,
                source_in_sec=0.0,
                source_out_sec=4.0,
                source_in_frame=0,
                source_out_frame=119,
                timeline_start_sec=0.0,
                timeline_end_sec=4.0,
                timeline_in_frame=0,
                timeline_out_frame=119,
                duration_frames=120,
                track_type=ttype,  # type: ignore[arg-type]
                audio_stream_index=a_idx,
                enabled=True,
                action="keep",
                decision_reason="preserved_intro",
                confidence=0.96,
            ),
            TimelineClip(
                clip_id=f"{prefix}_001",
                source_path=media_info.path,
                source_in_sec=8.0,
                source_out_sec=12.0,
                source_in_frame=240,
                source_out_frame=359,
                timeline_start_sec=4.0,
                timeline_end_sec=8.0,
                timeline_in_frame=120,
                timeline_out_frame=239,
                duration_frames=120,
                track_type=ttype,  # type: ignore[arg-type]
                audio_stream_index=a_idx,
                enabled=True,
                action="keep",
                decision_reason="speech_commentary",
                confidence=0.94,
            ),
        ]

    removed_v_clip = TimelineClip(
        clip_id="v_rev_000",
        source_path=media_info.path,
        source_in_sec=4.0,
        source_out_sec=8.0,
        source_in_frame=120,
        source_out_frame=239,
        timeline_start_sec=4.0,
        timeline_end_sec=8.0,
        timeline_in_frame=120,
        timeline_out_frame=239,
        duration_frames=120,
        track_type="video",
        audio_stream_index=None,
        enabled=False,
        action="remove",
        decision_reason="dead_air_silence",
        confidence=0.91,
    )
    removed_a_clip = TimelineClip(
        clip_id="a_rev_000",
        source_path=media_info.path,
        source_in_sec=4.0,
        source_out_sec=8.0,
        source_in_frame=120,
        source_out_frame=239,
        timeline_start_sec=4.0,
        timeline_end_sec=8.0,
        timeline_in_frame=120,
        timeline_out_frame=239,
        duration_frames=120,
        track_type="audio",
        audio_stream_index=1,
        enabled=False,
        action="remove",
        decision_reason="dead_air_silence",
        confidence=0.91,
    )

    tracks = [
        TimelineTrack(
            track_id="V1",
            track_name="V1 - Gameplay Video",
            track_type="video",
            is_review_track=False,
            is_hidden=False,
            is_muted=False,
            clips=_make_kept_clips("v1", "video", None),
        ),
        TimelineTrack(
            track_id="A1",
            track_name="A1 - Microphone",
            track_type="audio",
            is_review_track=False,
            is_hidden=False,
            is_muted=False,
            audio_stream_index=1,
            clips=_make_kept_clips("a1", "audio", 1),
        ),
        TimelineTrack(
            track_id="A2",
            track_name="A2 - Game Audio",
            track_type="audio",
            is_review_track=False,
            is_hidden=False,
            is_muted=False,
            audio_stream_index=2,
            clips=_make_kept_clips("a2", "audio", 2),
        ),
        TimelineTrack(
            track_id="V2_Review",
            track_name="V2 - Review Removed Cuts (Hidden/Disabled)",
            track_type="video",
            is_review_track=True,
            is_hidden=True,
            is_muted=True,
            clips=[removed_v_clip],
        ),
        TimelineTrack(
            track_id="A_Review",
            track_name="A_Review - Removed Audio Cuts (Muted/Disabled)",
            track_type="audio",
            is_review_track=True,
            is_hidden=True,
            is_muted=True,
            audio_stream_index=1,
            clips=[removed_a_clip],
        ),
    ]

    markers = [
        TimelineMarker(frame=0, position_sec=0.0, comment="Preserved Intro (0.0s - 4.0s)", category=3),
        TimelineMarker(
            frame=120,
            position_sec=4.0,
            comment="Cut #1: Removed 4.00s dead_air_silence [00:00:04.00 -> 00:00:08.00]",
            category=0,
        ),
    ]
    subtitles = [
        SubtitleEntry(
            index=1,
            start_sec=0.4,
            end_sec=3.6,
            text="Welcome back! This is clip one of the minimal validation project.",
            confidence=0.96,
        ),
        SubtitleEntry(
            index=2,
            start_sec=4.2,
            end_sec=7.6,
            text="And this is clip two after the 4-second dead-air cut!",
            confidence=0.94,
        ),
    ]

    edm = EditDecisionModel(
        title="Phase 0 Minimal Validation Project",
        timeline_mode="review-layers",
        source_media=media_info,
        fps_num=30,
        fps_den=1,
        width=640,
        height=360,
        source_duration_frames=360,
        edited_duration_frames=240,
        tracks=tracks,
        markers=markers,
        subtitles=subtitles,
        removed_clips_summary=[removed_v_clip],
        overall_confidence=0.94,
        requires_human_review=True,
        low_confidence_halt=False,
        review_notes=[
            "Phase 0 minimal validation test project containing 1 source video, multiple clips, separate video/audio tracks, timeline guides, and .srt subtitles."
        ],
    )

    kdenlive_path, kdenlive_srt, standalone_srt = export_kdenlive_project(
        edm, job_dir, "minimal_test_project"
    )
    verification = verify_kdenlive_project(kdenlive_path, check_source_media_exists=True)
    safe_write_text(
        job_dir / "verification_report.json",
        verification.to_json(indent=2),
        source_path=source_video,
    )

    output = {
        "status": "ok" if verification.is_valid else "failed",
        "sample_directory": str(job_dir),
        "source_video": str(source_video),
        "kdenlive_project": str(kdenlive_path),
        "kdenlive_srt": str(kdenlive_srt) if kdenlive_srt else None,
        "standalone_srt": str(standalone_srt) if standalone_srt else None,
        "verification": verification.to_dict(),
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0 if verification.is_valid else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify a .kdenlive project file or generate and verify the Phase 0 minimal validation sample."
    )
    parser.add_argument(
        "project",
        nargs="?",
        default=None,
        help="Path to an existing .kdenlive project file to verify.",
    )
    parser.add_argument(
        "--generate-minimal-sample",
        metavar="OUTPUT_DIR",
        default=None,
        help="Generate the Phase 0 minimal validation test project under OUTPUT_DIR and verify it.",
    )
    parser.add_argument(
        "--no-check-source-media",
        action="store_true",
        help="Skip verifying that the source media file exists on disk.",
    )
    args = parser.parse_args()

    if args.generate_minimal_sample:
        return generate_and_verify_minimal_sample(Path(args.generate_minimal_sample))

    if not args.project:
        parser.error("Provide either a <project.kdenlive> path or --generate-minimal-sample <output_dir>.")

    report = verify_kdenlive_project(
        args.project,
        check_source_media_exists=not args.no_check_source_media,
    )
    print(report.to_json(indent=2))
    return 0 if report.is_valid else 1


if __name__ == "__main__":
    sys.exit(main())

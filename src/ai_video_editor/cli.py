"""Command-line interface for AI-Video-Editor."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from .analyzer_pipeline import analyze_recording
from .config import EditingConfig
from .diagnostics import run_diagnostics
from .kdenlive_export import export_kdenlive_project
from .models import AnalysisReport
from .planner import build_edit_decision_model
from .project_verifier import verify_kdenlive_project
from .safety import (
    SafetyViolationError,
    assert_not_source_path,
    capture_source_snapshot,
    create_unique_job_directory,
    safe_write_text,
    sanitize_filename_slug,
    verify_source_untouched,
)


def _add_common_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--timeline-mode",
        choices=["review-layers", "ripple"],
        default="review-layers",
        help="Timeline layout mode: 'review-layers' (default, keeps removed cuts on disabled/hidden review track) or 'ripple' (clean kept clips only).",
    )
    parser.add_argument(
        "--whisper-model",
        default="auto",
        help="Whisper model selection: 'auto' (chooses 'base' or 'small' by hardware), 'base', 'small', 'base.en', 'small.en', or path to local model dir.",
    )
    parser.add_argument(
        "--whisper-device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Execution device for faster-whisper ('auto', 'cpu', or 'cuda'). Automatically falls back to CPU if CUDA is unavailable.",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Optional spoken language code (e.g. 'en', 'nl', 'de'). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--preserve-intro",
        type=float,
        default=8.0,
        help="Seconds at the very start of the recording to always preserve intact (default: 8.0).",
    )
    parser.add_argument(
        "--min-silence",
        type=float,
        default=2.5,
        help="Minimum silence/dead-air duration in seconds before a gap is eligible for trimming (default: 2.5).",
    )
    parser.add_argument(
        "--speech-pre-roll",
        type=float,
        default=0.35,
        help="Safety margin in seconds preserved before each speech segment (default: 0.35).",
    )
    parser.add_argument(
        "--speech-post-roll",
        type=float,
        default=0.45,
        help="Safety margin in seconds preserved after each speech segment (default: 0.45).",
    )
    parser.add_argument(
        "--min-kept-clip",
        type=float,
        default=1.2,
        help="Minimum duration in seconds of a kept clip to avoid micro-cuts (default: 1.2).",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.55,
        help="Minimum required overall edit confidence [0.0-1.0]; halts for human review if below threshold (default: 0.55).",
    )
    parser.add_argument(
        "--no-remove-stumbles",
        action="store_true",
        help="Disable automatic removal of obvious hesitation/filler stumbles.",
    )
    parser.add_argument(
        "--disable-visual-analysis",
        action="store_true",
        help="Disable scene cut and visual gameplay motion analysis.",
    )
    parser.add_argument(
        "--no-subtitles",
        action="store_true",
        help="Disable .srt subtitle sidecar generation.",
    )


def _config_from_args(args: argparse.Namespace) -> EditingConfig:
    return EditingConfig(
        timeline_mode=getattr(args, "timeline_mode", "review-layers"),
        whisper_model=getattr(args, "whisper_model", "auto"),
        whisper_device=getattr(args, "whisper_device", "auto"),
        language=getattr(args, "language", None),
        preserve_intro_seconds=getattr(args, "preserve_intro", 8.0),
        min_silence_duration_seconds=getattr(args, "min_silence", 2.5),
        speech_pre_roll_seconds=getattr(args, "speech_pre_roll", 0.35),
        speech_post_roll_seconds=getattr(args, "speech_post_roll", 0.45),
        min_kept_clip_duration_seconds=getattr(args, "min_kept_clip", 1.2),
        min_overall_edit_confidence=getattr(args, "min_confidence", 0.55),
        remove_obvious_stumbles=not getattr(args, "no_remove_stumbles", False),
        enable_visual_analysis=not getattr(args, "disable_visual_analysis", False),
        generate_subtitles=not getattr(args, "no_subtitles", False),
    )


def _resolve_analyze_report_path(
    output_report_arg: Optional[str],
    source_path: Path,
) -> Path:
    """Resolve and validate the target path for analysis_report.json BEFORE running analysis.

    - If output_report_arg is omitted or does not end with '.json' (or is an existing directory),
      allocates a collision-free job directory under that base path and returns <job_dir>/analysis_report.json.
    - If output_report_arg ends with '.json' (and is not an existing directory), verifies upfront that
      the file does not already exist and is not the source file.
    """
    if not output_report_arg:
        job_dir = create_unique_job_directory(
            Path.cwd() / "output",
            source_path.stem,
            source_path=source_path,
        )
        return job_dir / "analysis_report.json"

    out_target = Path(output_report_arg).expanduser().resolve()
    assert_not_source_path(out_target, source_path)

    if out_target.exists() and out_target.is_dir():
        job_dir = create_unique_job_directory(
            out_target,
            source_path.stem,
            source_path=source_path,
        )
        return job_dir / "analysis_report.json"

    if out_target.suffix.lower() == ".json":
        if out_target.exists():
            raise SafetyViolationError(
                f"Refusing to overwrite existing analysis report file: {out_target}"
            )
        for parent in out_target.parents:
            if parent.exists() and not parent.is_dir():
                raise SafetyViolationError(
                    f"Parent path exists and is not a directory: {parent}"
                )
        return out_target

    # Otherwise, treat out_target as a base output directory (even if it does not exist yet)
    job_dir = create_unique_job_directory(
        out_target,
        source_path.stem,
        source_path=source_path,
    )
    return job_dir / "analysis_report.json"


def _cmd_analyze(args: argparse.Namespace) -> int:
    source_path = Path(args.input).expanduser().resolve()
    cfg = _config_from_args(args)
    snap = capture_source_snapshot(source_path)

    try:
        report_path = _resolve_analyze_report_path(args.output_report, source_path)
        report = analyze_recording(source_path, config=cfg)
        safe_write_text(report_path, report.to_json(indent=2), source_path=source_path)
    finally:
        verify_source_untouched(snap)

    summary = {
        "status": "ok",
        "source_media": str(source_path),
        "analysis_report_path": str(report_path),
        "duration_seconds": report.media_info.duration_seconds,
        "video_streams": len(report.media_info.video_streams),
        "audio_streams": [
            {"index": a.stream_index, "role": a.inferred_role, "title": a.title}
            for a in report.media_info.audio_streams
        ],
        "whisper_model_used": report.whisper_model_used,
        "whisper_device_used": report.whisper_device_used,
        "transcription_status": report.transcription_status,
        "speech_intervals_count": len(report.speech_intervals),
        "transcript_segments_count": len(report.transcript_segments),
        "scene_boundaries_count": len(report.scene_boundaries),
        "overall_analysis_confidence": report.overall_analysis_confidence,
        "warnings": report.warnings,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    report_path = Path(args.report).expanduser().resolve()
    report = AnalysisReport.load_json(report_path)
    source_path = Path(report.media_info.path)
    snap = capture_source_snapshot(source_path)
    cfg = _config_from_args(args)

    try:
        job_dir = create_unique_job_directory(
            args.output_dir,
            args.title,
            source_path=source_path,
        )
        edm = build_edit_decision_model(report, title=args.title, config=cfg)
        edm_path = safe_write_text(
            job_dir / "edit_decisions.json",
            edm.to_json(indent=2),
            source_path=source_path,
        )

        if edm.low_confidence_halt:
            halt_payload = {
                "status": "halted_low_confidence",
                "reason": "Edit confidence is below configured minimum; human review required.",
                "overall_confidence": edm.overall_confidence,
                "min_required_confidence": cfg.min_overall_edit_confidence,
                "job_dir": str(job_dir),
                "edit_decisions_path": str(edm_path),
                "warnings": edm.warnings,
            }
            print(json.dumps(halt_payload, indent=2, ensure_ascii=False))
            return 2

        slug = sanitize_filename_slug(args.title)
        kdenlive_path, kdenlive_srt, standalone_srt = export_kdenlive_project(
            edm=edm,
            output_dir=job_dir,
            project_basename=slug,
        )
        verification = verify_kdenlive_project(kdenlive_path, check_source_media_exists=True)
        ver_path = safe_write_text(
            job_dir / "verification_report.json",
            verification.to_json(indent=2),
            source_path=source_path,
        )
    finally:
        verify_source_untouched(snap)

    result = {
        "status": "ok" if verification.is_valid else "verification_failed",
        "job_dir": str(job_dir),
        "kdenlive_project": str(kdenlive_path),
        "kdenlive_srt_sidecar": str(kdenlive_srt) if kdenlive_srt else None,
        "standalone_srt": str(standalone_srt) if standalone_srt else None,
        "edit_decisions_json": str(edm_path),
        "verification_report_json": str(ver_path),
        "timeline_mode": edm.timeline_mode,
        "source_duration_frames": edm.source_duration_frames,
        "edited_duration_frames": edm.edited_duration_frames,
        "kept_clips_count": len(edm.tracks[0].clips) if edm.tracks else 0,
        "removed_clips_count": len(edm.removed_clips_summary),
        "overall_confidence": edm.overall_confidence,
        "verification_is_valid": verification.is_valid,
        "manual_export_notice": (
            "Automatic video rendering is disabled by design. Open the .kdenlive file in Kdenlive, "
            "review the cuts and timeline guides, and export manually via Project -> Render (Ctrl+Return)."
        ),
        "warnings": edm.warnings + verification.warnings,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if verification.is_valid else 1


def _cmd_edit(args: argparse.Namespace) -> int:
    source_path = Path(args.input).expanduser().resolve()
    snap = capture_source_snapshot(source_path)
    cfg = _config_from_args(args)

    try:
        job_dir = create_unique_job_directory(
            args.output_dir,
            args.title,
            source_path=source_path,
        )

        report = analyze_recording(source_path, config=cfg)
        report_path = safe_write_text(
            job_dir / "analysis_report.json",
            report.to_json(indent=2),
            source_path=source_path,
        )

        edm = build_edit_decision_model(report, title=args.title, config=cfg)
        edm_path = safe_write_text(
            job_dir / "edit_decisions.json",
            edm.to_json(indent=2),
            source_path=source_path,
        )

        if edm.low_confidence_halt:
            halt_payload = {
                "status": "halted_low_confidence",
                "reason": "Overall edit confidence is below configured minimum; stopping for human review.",
                "overall_confidence": edm.overall_confidence,
                "min_required_confidence": cfg.min_overall_edit_confidence,
                "job_dir": str(job_dir),
                "analysis_report_json": str(report_path),
                "edit_decisions_json": str(edm_path),
                "warnings": edm.warnings,
            }
            print(json.dumps(halt_payload, indent=2, ensure_ascii=False))
            return 2

        slug = sanitize_filename_slug(args.title)
        kdenlive_path, kdenlive_srt, standalone_srt = export_kdenlive_project(
            edm=edm,
            output_dir=job_dir,
            project_basename=slug,
        )
        verification = verify_kdenlive_project(kdenlive_path, check_source_media_exists=True)
        ver_path = safe_write_text(
            job_dir / "verification_report.json",
            verification.to_json(indent=2),
            source_path=source_path,
        )
    finally:
        verify_source_untouched(snap)

    output = {
        "status": "ok" if verification.is_valid else "verification_failed",
        "job_dir": str(job_dir),
        "source_media": str(source_path),
        "source_sha256_verified": snap.sha256_hex,
        "kdenlive_project": str(kdenlive_path),
        "kdenlive_srt_sidecar": str(kdenlive_srt) if kdenlive_srt else None,
        "standalone_srt": str(standalone_srt) if standalone_srt else None,
        "analysis_report_json": str(report_path),
        "edit_decisions_json": str(edm_path),
        "verification_report_json": str(ver_path),
        "timeline_mode": edm.timeline_mode,
        "source_duration_frames": edm.source_duration_frames,
        "edited_duration_frames": edm.edited_duration_frames,
        "kept_clips_count": len(edm.tracks[0].clips) if edm.tracks else 0,
        "removed_clips_count": len(edm.removed_clips_summary),
        "overall_confidence": edm.overall_confidence,
        "verification_is_valid": verification.is_valid,
        "manual_export_notice": (
            "Automatic video rendering is disabled by design. Open the .kdenlive file in Kdenlive, "
            "review the cuts and timeline guides, and export manually via Project -> Render (Ctrl+Return)."
        ),
        "warnings": edm.warnings + verification.warnings,
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0 if verification.is_valid else 1


def _cmd_verify(args: argparse.Namespace) -> int:
    ver = verify_kdenlive_project(
        args.project,
        check_source_media_exists=not args.no_check_source_media,
    )
    print(ver.to_json(indent=2))
    return 0 if ver.is_valid else 1


def _cmd_diagnose(_args: argparse.Namespace) -> int:
    diag = run_diagnostics()
    print(diag.to_json(indent=2))
    return 0 if diag.all_required_ok else 1


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai-video-editor",
        description=(
            "Free, open-source AI-assisted gaming video editor for OpenCode & Kdenlive. "
            "Analyzes recordings locally in read-only mode and generates editable .kdenlive + .srt projects."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # analyze
    p_analyze = sub.add_parser("analyze", help="Analyze a gaming recording in read-only mode and output JSON report.")
    p_analyze.add_argument("--input", "-i", required=True, help="Path to source gaming video recording.")
    p_analyze.add_argument("--output-report", "-o", default=None, help="Output JSON file path or base directory.")
    _add_common_config_args(p_analyze)
    p_analyze.set_defaults(func=_cmd_analyze)

    # generate
    p_gen = sub.add_parser("generate", help="Generate a Kdenlive (.kdenlive + .srt) project from an analysis report.")
    p_gen.add_argument("--report", "-r", required=True, help="Path to analysis_report.json.")
    p_gen.add_argument("--title", "-t", required=True, help="Video project title.")
    p_gen.add_argument("--output-dir", "-o", default="./output", help="Base directory for unique project folder.")
    _add_common_config_args(p_gen)
    p_gen.set_defaults(func=_cmd_generate)

    # edit (end-to-end analyze + generate + verify)
    p_edit = sub.add_parser("edit", help="Analyze a gaming recording and generate a verified .kdenlive project in a unique directory.")
    p_edit.add_argument("--input", "-i", required=True, help="Path to source gaming video recording.")
    p_edit.add_argument("--title", "-t", required=True, help="Video project title.")
    p_edit.add_argument("--output-dir", "-o", default="./output", help="Base directory for unique project folder.")
    _add_common_config_args(p_edit)
    p_edit.set_defaults(func=_cmd_edit)

    # verify
    p_ver = sub.add_parser("verify", help="Verify a .kdenlive project against MLT DTD and Kdenlive 1.1 C++ invariants.")
    p_ver.add_argument("project", help="Path to .kdenlive project file.")
    p_ver.add_argument("--no-check-source-media", action="store_true", help="Skip checking if source media exists on disk.")
    p_ver.set_defaults(func=_cmd_verify)

    # diagnose
    p_diag = sub.add_parser("diagnose", help="Inspect Python environment, dependencies, hardware, and Kdenlive installation.")
    p_diag.set_defaults(func=_cmd_diagnose)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Standalone diagnostic script for AI-Video-Editor."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure src/ is importable even if invoked directly without editable install
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ai_video_editor.diagnostics import run_diagnostics  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect Python environment, required open-source libraries, hardware resources, and Kdenlive availability."
    )
    parser.add_argument(
        "--human",
        action="store_true",
        help="Print a formatted human-readable table in addition to JSON.",
    )
    args = parser.parse_args()

    report = run_diagnostics()
    if args.human:
        print(f"Platform: {report.platform_system} {report.platform_release} (Python {report.python_version})")
        print(f"Hardware: {report.cpu_cores} CPU cores, {report.system_ram_gb:.1f} GB RAM, CUDA={report.cuda_available}")
        print(f"Recommended Whisper model: {report.recommended_whisper_model}")
        print("-" * 72)
        for comp in report.components:
            status = "OK" if comp.available else ("MISSING (REQUIRED)" if comp.required else "OPTIONAL (NOT FOUND)")
            print(f"[{status:20s}] {comp.name}: {comp.version_or_path}")
        if report.warnings:
            print("-" * 72)
            for w in report.warnings:
                print(f"WARNING: {w}")
        print("-" * 72)

    print(report.to_json(indent=2))
    return 0 if report.all_required_ok else 1


if __name__ == "__main__":
    sys.exit(main())

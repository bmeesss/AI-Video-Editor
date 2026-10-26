"""SRT subtitle sidecar generation synchronized with the edited timeline."""

from __future__ import annotations

from pathlib import Path
from typing import List

from .models import SubtitleEntry
from .safety import safe_write_text


def format_srt_timestamp(seconds: float) -> str:
    """Format seconds as standard SubRip timestamp HH:MM:SS,mmm."""
    clamped = max(0.0, float(seconds))
    total_ms = int(round(clamped * 1000.0))
    hours = total_ms // 3_600_000
    rem = total_ms % 3_600_000
    minutes = rem // 60_000
    rem = rem % 60_000
    secs = rem // 1000
    millis = rem % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def render_srt_content(entries: List[SubtitleEntry]) -> str:
    """Render a list of SubtitleEntry items into valid UTF-8 .srt content."""
    blocks: List[str] = []
    for idx, entry in enumerate(entries, start=1):
        text = entry.text.strip()
        if not text:
            continue
        start_ts = format_srt_timestamp(entry.start_sec)
        end_sec = max(entry.start_sec + 0.25, entry.end_sec)
        end_ts = format_srt_timestamp(end_sec)
        blocks.append(f"{idx}\n{start_ts} --> {end_ts}\n{text}\n")
    return "\n".join(blocks)


def write_srt_sidecar(
    target_srt_path: Path | str,
    entries: List[SubtitleEntry],
    source_media_path: Path | str | None = None,
) -> Path:
    """Write a non-destructive .srt sidecar file using exclusive creation."""
    content = render_srt_content(entries)
    return safe_write_text(target_srt_path, content, source_path=source_media_path)

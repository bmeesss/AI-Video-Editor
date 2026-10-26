"""Non-destructive file safety guards, source integrity verification, and collision-free output paths."""

from __future__ import annotations

import datetime as dt
import hashlib
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict


WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    "COM1",
    "COM2",
    "COM3",
    "COM4",
    "COM5",
    "COM6",
    "COM7",
    "COM8",
    "COM9",
    "LPT1",
    "LPT2",
    "LPT3",
    "LPT4",
    "LPT5",
    "LPT6",
    "LPT7",
    "LPT8",
    "LPT9",
}


class SafetyViolationError(RuntimeError):
    """Raised when an operation threatens source media integrity or existing user files."""


@dataclass(frozen=True)
class SourceIntegritySnapshot:
    """Cryptographic and metadata snapshot of an original recording file."""

    resolved_path: str
    size_bytes: int
    mtime_ns: int
    md5_hex: str
    sha256_hex: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_file_hashes(path: Path, chunk_size: int = 1024 * 1024) -> tuple[str, str]:
    """Compute (md5_hex, sha256_hex) of a file in strictly read-only mode."""
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            md5.update(chunk)
            sha256.update(chunk)
    return md5.hexdigest(), sha256.hexdigest()


def capture_source_snapshot(source_path: Path | str) -> SourceIntegritySnapshot:
    """Capture a baseline integrity snapshot of the input recording before any processing."""
    p = Path(source_path).expanduser().resolve()
    if not p.exists():
        raise SafetyViolationError(f"Source media file does not exist: {p}")
    if not p.is_file():
        raise SafetyViolationError(f"Source media path is not a regular file: {p}")

    stat = p.stat()
    if stat.st_size == 0:
        raise SafetyViolationError(f"Source media file is empty (0 bytes): {p}")

    md5_hex, sha256_hex = compute_file_hashes(p)
    return SourceIntegritySnapshot(
        resolved_path=str(p),
        size_bytes=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        md5_hex=md5_hex,
        sha256_hex=sha256_hex,
    )


def verify_source_untouched(snapshot: SourceIntegritySnapshot) -> bool:
    """Verify that the original recording file has not been deleted, moved, resized, or modified."""
    p = Path(snapshot.resolved_path)
    if not p.exists():
        raise SafetyViolationError(
            f"CRITICAL SAFETY VIOLATION: Source file disappeared: {snapshot.resolved_path}"
        )
    stat = p.stat()
    if stat.st_size != snapshot.size_bytes:
        raise SafetyViolationError(
            f"CRITICAL SAFETY VIOLATION: Source file size changed from "
            f"{snapshot.size_bytes} to {stat.st_size}: {snapshot.resolved_path}"
        )
    md5_hex, sha256_hex = compute_file_hashes(p)
    if sha256_hex != snapshot.sha256_hex or md5_hex != snapshot.md5_hex:
        raise SafetyViolationError(
            f"CRITICAL SAFETY VIOLATION: Source file content hash changed for {snapshot.resolved_path} "
            f"(expected sha256={snapshot.sha256_hex}, actual={sha256_hex})"
        )
    return True


def sanitize_filename_slug(title: str, max_length: int = 48) -> str:
    """Convert an arbitrary user title (including Unicode, spaces, or symbols) into a safe cross-platform slug."""
    cleaned = title.strip()
    # Replace Windows/POSIX forbidden filename characters and whitespace runs with underscores
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", cleaned)
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("._ ")
    if not cleaned:
        cleaned = "gaming_project"
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length].rstrip("._ ")
    if not cleaned or cleaned.upper() in WINDOWS_RESERVED_NAMES:
        cleaned = f"project_{cleaned}" if cleaned else "gaming_project"
    return cleaned


def assert_not_source_path(target_path: Path, source_path: Path) -> None:
    """Ensure target_path never resolves to source_path."""
    resolved_target = target_path.expanduser().resolve()
    resolved_source = source_path.expanduser().resolve()
    if resolved_target == resolved_source:
        raise SafetyViolationError(
            f"Refusing to write to source media path: {resolved_target}"
        )


def create_unique_job_directory(
    base_output_dir: Path | str,
    title: str,
    source_path: Path | str | None = None,
    timestamp: dt.datetime | None = None,
) -> Path:
    """Create a guaranteed-new, collision-resistant directory for an editing job.

    Never overwrites or reuses an existing directory.
    """
    base = Path(base_output_dir).expanduser().resolve()
    if source_path is not None:
        src_resolved = Path(source_path).expanduser().resolve()
        if base == src_resolved:
            raise SafetyViolationError(
                f"Base output directory cannot be the source media file: {base}"
            )
    if base.exists() and not base.is_dir():
        raise SafetyViolationError(
            f"Base output path exists and is not a directory: {base}"
        )
    base.mkdir(parents=True, exist_ok=True)

    slug = sanitize_filename_slug(title)
    now = timestamp or dt.datetime.now(dt.timezone.utc)
    ts_str = now.strftime("%Y%m%d_%H%M%S")

    for attempt in range(0, 10000):
        suffix = f"_{attempt:03d}" if attempt > 0 else ""
        candidate = base / f"{slug}_{ts_str}{suffix}"
        try:
            os.makedirs(candidate, exist_ok=False)
            return candidate
        except FileExistsError:
            continue

    raise SafetyViolationError(
        f"Unable to allocate a unique job directory under {base} after 10,000 attempts."
    )


def safe_write_text(
    target_path: Path | str,
    content: str,
    source_path: Path | str | None = None,
    encoding: str = "utf-8",
) -> Path:
    """Write text to a brand-new file using exclusive creation ('x') so existing files are never overwritten."""
    target = Path(target_path).expanduser().resolve()
    if source_path is not None:
        assert_not_source_path(target, Path(source_path))
    if target.exists():
        raise SafetyViolationError(
            f"Refusing to overwrite existing file: {target}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("x", encoding=encoding, newline="\n") as fh:
            fh.write(content)
    except FileExistsError as exc:
        raise SafetyViolationError(
            f"Refusing to overwrite existing file: {target}"
        ) from exc
    return target

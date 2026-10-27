"""Environment and dependency diagnostics for AI-Video-Editor on Windows and Linux."""

from __future__ import annotations

import importlib
import json
import os
import platform
import shutil
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .speech_transcriber import get_system_ram_gb, is_cuda_available


@dataclass
class ComponentCheck:
    name: str
    available: bool
    version_or_path: str
    required: bool = True
    notes: str = ""


@dataclass
class DiagnosticReport:
    platform_system: str
    platform_release: str
    python_version: str
    python_executable: str
    cpu_cores: int
    system_ram_gb: float
    cuda_available: bool
    recommended_whisper_model: str
    kdenlive_path: Optional[str]
    ffmpeg_path: Optional[str]
    all_required_ok: bool
    components: List[ComponentCheck] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


def find_kdenlive_executable() -> Optional[str]:
    """Locate Kdenlive on PATH or in standard Windows installation directories."""
    on_path = shutil.which("kdenlive")
    if on_path:
        return on_path

    if os.name == "nt":
        candidates = [
            Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "kdenlive" / "bin" / "kdenlive.exe",
            Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "kdenlive" / "bin" / "kdenlive.exe",
            Path(r"C:\Program Files\kdenlive\bin\kdenlive.exe"),
        ]
        for cand in candidates:
            if cand.exists():
                return str(cand)
    return None


def find_ffmpeg_executable() -> Optional[str]:
    """Locate system ffmpeg or bundled imageio-ffmpeg binary."""
    sys_ff = shutil.which("ffmpeg")
    if sys_ff:
        return sys_ff
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def run_diagnostics() -> DiagnosticReport:
    """Inspect Python environment, dependencies, hardware resources, and Kdenlive installation."""
    components: List[ComponentCheck] = []
    warnings: List[str] = []

    py_ok = sys.version_info >= (3, 10)
    components.append(
        ComponentCheck(
            name="Python (>=3.10)",
            available=py_ok,
            version_or_path=sys.version.split()[0],
            required=True,
        )
    )

    pkg_specs = [
        ("av", "PyAV (container & multi-track audio/video probing)", True),
        ("imageio_ffmpeg", "imageio-ffmpeg (bundled FFmpeg binary)", True),
        ("numpy", "NumPy (audio/motion signal arrays)", True),
        ("faster_whisper", "faster-whisper (local Whisper + Silero VAD)", True),
        ("onnxruntime", "ONNX Runtime (Silero VAD CPU execution)", True),
        ("scenedetect", "PySceneDetect (scene cut detection)", True),
        ("cv2", "OpenCV (downscaled visual motion scoring)", True),
        ("lxml", "lxml (MLT XML DTD validation)", True),
    ]

    for mod_name, label, req in pkg_specs:
        try:
            mod = importlib.import_module(mod_name)
            ver = getattr(mod, "__version__", "installed")
            components.append(
                ComponentCheck(
                    name=label,
                    available=True,
                    version_or_path=str(ver),
                    required=req,
                )
            )
        except Exception as exc:
            components.append(
                ComponentCheck(
                    name=label,
                    available=False,
                    version_or_path="missing",
                    required=req,
                    notes=str(exc),
                )
            )

    ff_path = find_ffmpeg_executable()
    components.append(
        ComponentCheck(
            name="FFmpeg executable (system or imageio-ffmpeg)",
            available=ff_path is not None,
            version_or_path=ff_path or "not found",
            required=True,
        )
    )

    kdenlive_exe = find_kdenlive_executable()
    components.append(
        ComponentCheck(
            name="Kdenlive desktop application (for manual review & export)",
            available=kdenlive_exe is not None,
            version_or_path=kdenlive_exe or "not found on PATH",
            required=False,
            notes="Required on user's workstation to open generated .kdenlive projects and render manually.",
        )
    )
    if kdenlive_exe is None:
        warnings.append(
            "Kdenlive executable was not detected on PATH or standard install folders. "
            "Project generation and verification work without Kdenlive installed, "
            "but Kdenlive 23.04+ is required to open, review, and manually export the final video."
        )

    ram_gb = get_system_ram_gb()
    cores = os.cpu_count() or 4
    cuda_ok = is_cuda_available()
    rec_model = "small" if (ram_gb >= 16.0 and cores >= 8) else "base"

    all_req_ok = all(c.available for c in components if c.required)

    return DiagnosticReport(
        platform_system=platform.system(),
        platform_release=platform.release(),
        python_version=sys.version.split()[0],
        python_executable=sys.executable,
        cpu_cores=cores,
        system_ram_gb=ram_gb,
        cuda_available=cuda_ok,
        recommended_whisper_model=rec_model,
        kdenlive_path=kdenlive_exe,
        ffmpeg_path=ff_path,
        all_required_ok=all_req_ok,
        components=components,
        warnings=warnings,
    )

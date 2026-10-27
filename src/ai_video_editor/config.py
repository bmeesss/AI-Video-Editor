"""Configuration model and validation for AI-Video-Editor."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .models import TimelineMode


VALID_TIMELINE_MODES = ("review-layers", "ripple")
VALID_WHISPER_MODELS = ("auto", "base", "small", "base.en", "small.en")


class ConfigValidationError(ValueError):
    """Raised when user configuration violates editing or safety constraints."""


@dataclass(frozen=True)
class EditingConfig:
    """Validated configuration governing media analysis and timeline generation."""

    timeline_mode: TimelineMode = "review-layers"
    whisper_model: str = "auto"
    whisper_device: str = "auto"
    whisper_compute_type: str = "int8"
    language: Optional[str] = None

    # Non-negotiable conservative editing defaults
    preserve_intro_seconds: float = 8.0
    min_silence_duration_seconds: float = 2.5
    speech_pre_roll_seconds: float = 0.35
    speech_post_roll_seconds: float = 0.45
    min_kept_clip_duration_seconds: float = 1.2

    # Audio & VAD thresholds
    vad_threshold: float = 0.45
    silence_dbfs_threshold: float = -42.0
    transient_peak_dbfs: float = -12.0
    audio_window_seconds: float = 0.25

    # Visual motion / scene analysis thresholds
    # Crucial: do not cut quiet moments if visual gameplay motion is active
    enable_visual_analysis: bool = True
    motion_sample_fps: float = 4.0
    active_gameplay_motion_threshold: float = 4.5
    scene_detection_threshold: float = 27.0

    # Stumble detection (conservative)
    remove_obvious_stumbles: bool = True
    min_stumble_confidence: float = 0.78

    # Confidence gate
    min_overall_edit_confidence: float = 0.55

    # Policy flags (enforced by requirements)
    enable_background_music: bool = False
    enable_auto_zooms_or_effects: bool = False
    generate_subtitles: bool = True

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.timeline_mode not in VALID_TIMELINE_MODES:
            raise ConfigValidationError(
                f"Invalid timeline_mode '{self.timeline_mode}'. Must be one of {VALID_TIMELINE_MODES}."
            )
        if self.whisper_model not in VALID_WHISPER_MODELS:
            model_path = Path(self.whisper_model).expanduser()
            if not model_path.exists() or not model_path.is_dir() or not (model_path / "model.bin").is_file():
                raise ConfigValidationError(
                    f"Invalid whisper_model '{self.whisper_model}'. "
                    f"Allowed values in v1: {VALID_WHISPER_MODELS} or an existing local CTranslate2 Whisper model directory containing 'model.bin'."
                )
        if self.whisper_device not in ("auto", "cpu", "cuda"):
            raise ConfigValidationError(
                f"Invalid whisper_device '{self.whisper_device}'. Must be 'auto', 'cpu', or 'cuda'."
            )
        if self.preserve_intro_seconds < 0.0:
            raise ConfigValidationError("preserve_intro_seconds must be >= 0.0.")
        if self.min_silence_duration_seconds < 0.5:
            raise ConfigValidationError(
                "min_silence_duration_seconds must be >= 0.5s to prevent overly aggressive cuts."
            )
        if self.speech_pre_roll_seconds < 0.0 or self.speech_post_roll_seconds < 0.0:
            raise ConfigValidationError("speech pre-roll and post-roll margins must be >= 0.0.")
        if self.min_kept_clip_duration_seconds < 0.2:
            raise ConfigValidationError("min_kept_clip_duration_seconds must be >= 0.2s.")
        if not (0.05 <= self.vad_threshold <= 0.95):
            raise ConfigValidationError("vad_threshold must be between 0.05 and 0.95.")
        if not (-90.0 <= self.silence_dbfs_threshold <= -10.0):
            raise ConfigValidationError("silence_dbfs_threshold must be between -90.0 and -10.0 dBFS.")
        if not (0.0 <= self.min_overall_edit_confidence <= 1.0):
            raise ConfigValidationError("min_overall_edit_confidence must be between 0.0 and 1.0.")
        if self.enable_background_music:
            raise ConfigValidationError(
                "enable_background_music=True is not permitted in v1 (no background music by default)."
            )
        if self.enable_auto_zooms_or_effects:
            raise ConfigValidationError(
                "enable_auto_zooms_or_effects=True is disabled in v1 to prevent random/unjustified effects."
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EditingConfig":
        valid_keys = {f for f in cls.__dataclass_fields__}
        unknown = set(data.keys()) - valid_keys
        if unknown:
            raise ConfigValidationError(f"Unknown configuration keys: {sorted(unknown)}")
        return cls(**data)

    @classmethod
    def load_json(cls, path: Path) -> "EditingConfig":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ConfigValidationError("Configuration JSON root must be an object.")
        return cls.from_dict(raw)

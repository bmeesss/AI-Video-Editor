"""Local speech transcription with faster-whisper, hardware-aware base/small selection, and conservative stumble detection."""

from __future__ import annotations

import ctypes
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import ctranslate2
import numpy as np

from .config import EditingConfig
from .models import TranscriptSegment, TranscriptWord


FILLER_WORDS = {"uh", "um", "erm", "er", "ah", "hmm", "uhh", "umm"}


@dataclass(frozen=True)
class WhisperRuntimeSelection:
    """Resolved Whisper model, device, and compute_type with hardware rationale."""

    model_name: str
    device: str
    compute_type: str
    system_ram_gb: float
    cpu_cores: int
    cuda_available: bool
    rationale: str


def get_system_ram_gb() -> float:
    """Detect total physical RAM in GiB across Windows and POSIX platforms."""
    if os.name == "nt":
        try:
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))  # type: ignore[attr-defined]
            return round(stat.ullTotalPhys / float(1024**3), 2)
        except Exception:
            return 8.0
    else:
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            if pages > 0 and page_size > 0:
                return round((pages * page_size) / float(1024**3), 2)
        except Exception:
            pass
    return 8.0


def is_cuda_available() -> bool:
    """Safely check whether CTranslate2 detects a usable CUDA GPU."""
    try:
        return int(ctranslate2.get_cuda_device_count()) > 0
    except Exception:
        return False


def select_whisper_model_and_device(config: EditingConfig) -> WhisperRuntimeSelection:
    """Select between Whisper 'base' and 'small' and CPU/CUDA with safe CPU fallback."""
    ram_gb = get_system_ram_gb()
    cores = os.cpu_count() or 4
    cuda_ok = is_cuda_available()

    if config.whisper_device == "cuda":
        if cuda_ok:
            device = "cuda"
            compute_type = config.whisper_compute_type if config.whisper_compute_type != "int8" else "float16"
        else:
            device = "cpu"
            compute_type = "int8"
    elif config.whisper_device == "cpu":
        device = "cpu"
        compute_type = "int8"
    else:
        device = "cuda" if cuda_ok else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"

    if config.whisper_model != "auto":
        model_name = config.whisper_model
        rationale = (
            f"Explicitly configured model '{model_name}' on device='{device}' "
            f"(compute_type='{compute_type}', RAM={ram_gb:.1f} GB, cores={cores}, cuda={cuda_ok})."
        )
    else:
        # Automatic selection between 'base' and 'small' based on hardware resources
        if device == "cuda" or (ram_gb >= 12.0 and cores >= 8):
            model_name = "small"
            rationale = (
                f"Auto-selected Whisper 'small' because hardware meets higher-tier threshold "
                f"(device='{device}', RAM={ram_gb:.1f} GB, cores={cores})."
            )
        else:
            model_name = "base"
            rationale = (
                f"Auto-selected Whisper 'base' for responsive CPU execution on constrained hardware "
                f"(device='{device}', RAM={ram_gb:.1f} GB, cores={cores})."
            )

    return WhisperRuntimeSelection(
        model_name=model_name,
        device=device,
        compute_type=compute_type,
        system_ram_gb=ram_gb,
        cpu_cores=cores,
        cuda_available=cuda_ok,
        rationale=rationale,
    )


def detect_stumble_in_segment(
    text: str,
    words: List[TranscriptWord],
    segment_confidence: float,
) -> Tuple[bool, float]:
    """Conservatively detect obvious verbal stumbles or false-start filler segments.

    Returns (is_stumble_candidate, stumble_confidence).
    Never flags normal expressive repetition (e.g. 'go go go!', 'no no no!').
    """
    cleaned = re.sub(r"[^\w\s']", " ", text.lower()).strip()
    tokens = [t for t in cleaned.split() if t]
    if not tokens:
        return False, 0.0

    # Case 1: Segment consists entirely of hesitation filler words ("uh", "um", "er")
    filler_count = sum(1 for t in tokens if t in FILLER_WORDS)
    if len(tokens) <= 4 and filler_count == len(tokens):
        conf = min(0.95, max(0.75, segment_confidence * 0.92))
        return True, round(conf, 4)

    # Case 2: Short false-start phrase with heavy filler ratio (>=60% filler words and <=5 words)
    if len(tokens) <= 5 and (filler_count / float(len(tokens))) >= 0.60:
        conf = min(0.90, max(0.70, segment_confidence * 0.85))
        return True, round(conf, 4)

    # Case 3: Immediate 2-to-3 word stuttered false-start with a pause gap before restart,
    # e.g. "I think we... uh I think we should..." when the segment itself is just the aborted fragment.
    EXPRESSIVE_WORDS = {"go", "no", "yes", "wait", "run", "jump", "shoot", "nice", "gg", "wow"}
    if 2 <= len(tokens) <= 4 and tokens[0] == tokens[1] and tokens[0] not in EXPRESSIVE_WORDS and filler_count >= 1:
        conf = min(0.88, max(0.72, segment_confidence * 0.84))
        return True, round(conf, 4)

    return False, 0.0


TranscriberCallable = Callable[[np.ndarray, EditingConfig, WhisperRuntimeSelection], List[TranscriptSegment]]


def transcribe_audio_stream(
    samples_16k: np.ndarray,
    config: EditingConfig,
    custom_transcriber: Optional[TranscriberCallable] = None,
) -> Tuple[List[TranscriptSegment], WhisperRuntimeSelection, str, List[str]]:
    """Transcribe a 16 kHz mono float32 audio stream using faster-whisper with CPU fallback.

    Returns:
      (segments, runtime_selection, status_string, warnings_list)
    """
    selection = select_whisper_model_and_device(config)
    warnings: List[str] = []

    if samples_16k.size == 0:
        return [], selection, "empty_audio", warnings

    if custom_transcriber is not None:
        segments = custom_transcriber(samples_16k, config, selection)
        return segments, selection, "completed_custom", warnings

    from faster_whisper import WhisperModel

    model: Optional[WhisperModel] = None
    active_selection = selection

    try:
        model = WhisperModel(
            selection.model_name,
            device=selection.device,
            compute_type=selection.compute_type,
        )
    except Exception as first_err:
        if selection.device == "cuda":
            warnings.append(
                f"CUDA initialization failed for Whisper '{selection.model_name}' ({first_err}); "
                "falling back automatically to CPU (int8)."
            )
            active_selection = WhisperRuntimeSelection(
                model_name=selection.model_name,
                device="cpu",
                compute_type="int8",
                system_ram_gb=selection.system_ram_gb,
                cpu_cores=selection.cpu_cores,
                cuda_available=False,
                rationale=selection.rationale + " (CPU fallback after CUDA error)",
            )
            try:
                model = WhisperModel(
                    active_selection.model_name,
                    device="cpu",
                    compute_type="int8",
                )
            except Exception as cpu_err:
                warnings.append(
                    f"Whisper model '{active_selection.model_name}' could not be loaded locally ({cpu_err}). "
                    "Continuing with Silero VAD speech intervals only; run scripts/install.ps1 with internet "
                    "access to cache the faster-whisper model locally for full subtitle and stumble analysis."
                )
                return [], active_selection, "skipped_model_unavailable", warnings
        else:
            warnings.append(
                f"Whisper model '{selection.model_name}' could not be loaded locally ({first_err}). "
                "Continuing with Silero VAD speech intervals only; run scripts/install.ps1 with internet "
                "access to cache the faster-whisper model locally for full subtitle and stumble analysis."
            )
            return [], active_selection, "skipped_model_unavailable", warnings

    try:
        raw_segments, _info = model.transcribe(
            samples_16k,
            language=config.language,
            beam_size=3,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(
                threshold=config.vad_threshold,
                min_speech_duration_ms=200,
                min_silence_duration_ms=300,
                speech_pad_ms=150,
            ),
        )

        results: List[TranscriptSegment] = []
        for idx, seg in enumerate(raw_segments):
            text = (seg.text or "").strip()
            if not text:
                continue
            words_list: List[TranscriptWord] = []
            if seg.words:
                for w in seg.words:
                    words_list.append(
                        TranscriptWord(
                            word=str(w.word),
                            start=round(float(w.start), 4),
                            end=round(float(w.end), 4),
                            probability=round(float(getattr(w, "probability", 0.85)), 4),
                        )
                    )

            avg_logprob = float(getattr(seg, "avg_logprob", -0.35))
            no_speech_prob = float(getattr(seg, "no_speech_prob", 0.05))
            if words_list:
                word_prob = sum(w.probability for w in words_list) / float(len(words_list))
            else:
                word_prob = math.exp(max(-3.0, min(0.0, avg_logprob)))
            seg_conf = round(min(1.0, max(0.05, word_prob * (1.0 - 0.5 * no_speech_prob))), 4)

            is_stumble, stumble_conf = detect_stumble_in_segment(text, words_list, seg_conf)
            results.append(
                TranscriptSegment(
                    segment_id=idx,
                    start=round(float(seg.start), 4),
                    end=round(float(seg.end), 4),
                    text=text,
                    confidence=seg_conf,
                    words=words_list,
                    is_stumble_candidate=is_stumble,
                    stumble_confidence=stumble_conf,
                )
            )
        return results, active_selection, "completed", warnings
    except Exception as exc:
        warnings.append(f"Transcription runtime failed ({exc}); falling back to VAD intervals.")
        return [], active_selection, "failed_fallback_to_vad", warnings

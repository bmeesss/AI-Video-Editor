"""Local speech transcription with faster-whisper, hardware-aware base/small selection, and conservative stumble detection."""

from __future__ import annotations

import ctypes
import ctypes.util
import math
import os
import re
import subprocess
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
    gpu_vram_gb: Optional[float] = None


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


def get_gpu_vram_gb() -> Optional[float]:
    """Query total GPU VRAM in GiB via nvidia-smi if available, returning None on failure."""
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            first_line = proc.stdout.strip().splitlines()[0].strip()
            mib = float(first_line)
            if mib > 0:
                return round(mib / 1024.0, 2)
    except Exception:
        pass
    return None


def _can_load_cublas_runtime() -> bool:
    """Verify that the CUDA cuBLAS runtime library required by CTranslate2 can actually be loaded."""
    if os.name == "nt":
        candidates = ["cublas64_12.dll", "cublas64_11.dll"]
    else:
        candidates = ["libcublas.so.12", "libcublas.so.11", "libcublas.so"]

    for lib_name in candidates:
        try:
            ctypes.CDLL(lib_name)
            return True
        except Exception:
            continue

    found = ctypes.util.find_library("cublas")
    if found:
        try:
            ctypes.CDLL(found)
            return True
        except Exception:
            pass
    return False


def probe_cuda_runtime() -> Tuple[bool, str]:
    """Check whether both an NVIDIA GPU and the required CUDA cuBLAS runtime libraries are usable."""
    try:
        count = int(ctranslate2.get_cuda_device_count())
    except Exception as exc:
        return False, f"CTranslate2 CUDA device query failed ({exc}); using CPU int8."

    if count <= 0:
        return False, "No CUDA GPU reported by CTranslate2; using CPU int8."

    if not _can_load_cublas_runtime():
        return (
            False,
            "NVIDIA GPU detected by driver, but CUDA cuBLAS runtime libraries (cublas64_12.dll / libcublas.so) "
            "are not available; falling back safely to CPU int8.",
        )

    return True, f"CUDA GPU and cuBLAS runtime verified ({count} device(s))."


def is_cuda_available() -> bool:
    """Safely check whether both a CUDA GPU and its required cuBLAS runtime are usable."""
    usable, _reason = probe_cuda_runtime()
    return usable


def select_whisper_model_and_device(
    config: EditingConfig,
    *,
    system_ram_gb: Optional[float] = None,
    cpu_cores: Optional[int] = None,
    cuda_available: Optional[bool] = None,
    gpu_vram_gb: Optional[float] = None,
) -> WhisperRuntimeSelection:
    """Select between Whisper 'base' and 'small' and CPU/CUDA with safe CPU fallback.

    Specifically tuned so 8 GB RAM / 4 GB VRAM laptops (e.g. Ryzen 5 7535HS + RTX 2050)
    default to the lightweight 'base' model and quantized int8 compute type to avoid
    RAM or VRAM exhaustion.
    """
    ram_gb = system_ram_gb if system_ram_gb is not None else get_system_ram_gb()
    cores = cpu_cores if cpu_cores is not None else (os.cpu_count() or 4)
    if cuda_available is not None:
        cuda_ok = cuda_available
        cuda_note = "CUDA status supplied by caller."
    else:
        cuda_ok, cuda_note = probe_cuda_runtime()

    vram_gb = gpu_vram_gb if gpu_vram_gb is not None else (get_gpu_vram_gb() if cuda_ok else None)

    # Determine execution device and memory-safe compute_type
    # Note: On constrained laptops (< 12 GB system RAM or <= 4 GB VRAM like RTX 2050),
    # use int8 quantization even on CUDA to minimize VRAM and host RAM footprint.
    is_memory_constrained = (ram_gb < 12.0) or (vram_gb is not None and vram_gb < 6.0)

    if config.whisper_device == "cuda":
        if cuda_ok:
            device = "cuda"
            if config.whisper_compute_type != "int8":
                compute_type = config.whisper_compute_type
            else:
                compute_type = "int8" if is_memory_constrained else "float16"
        else:
            device = "cpu"
            compute_type = "int8"
    elif config.whisper_device == "cpu":
        device = "cpu"
        compute_type = "int8"
    else:
        device = "cuda" if cuda_ok else "cpu"
        if device == "cuda":
            compute_type = "int8" if is_memory_constrained else "float16"
        else:
            compute_type = "int8"

    if config.whisper_model != "auto":
        model_name = config.whisper_model
        vram_str = f", VRAM={vram_gb:.1f} GB" if vram_gb is not None else ""
        rationale = (
            f"Explicitly configured model '{model_name}' on device='{device}' "
            f"(compute_type='{compute_type}', RAM={ram_gb:.1f} GB{vram_str}, cores={cores}, cuda={cuda_ok}; {cuda_note})"
        )
    else:
        # Automatic selection between 'base' and 'small':
        # Only choose 'small' when system RAM >= 12 GB AND either:
        #   - CUDA is usable with >= 6 GB VRAM, OR
        #   - System RAM >= 16 GB and CPU cores >= 8.
        # On 8 GB RAM / 4 GB VRAM laptops (e.g., Ryzen 5 7535HS + RTX 2050), always choose 'base'.
        high_tier_cuda = (
            device == "cuda"
            and ram_gb >= 12.0
            and (vram_gb is None or vram_gb >= 6.0)
        )
        high_tier_cpu = ram_gb >= 16.0 and cores >= 8

        vram_str = f", VRAM={vram_gb:.1f} GB" if vram_gb is not None else ""
        if high_tier_cuda or high_tier_cpu:
            model_name = "small"
            rationale = (
                f"Auto-selected Whisper 'small' because hardware meets high-headroom threshold "
                f"(device='{device}', compute_type='{compute_type}', RAM={ram_gb:.1f} GB{vram_str}, cores={cores})."
            )
        else:
            model_name = "base"
            rationale = (
                f"Auto-selected Whisper 'base' ({compute_type}) for safe, low-memory execution "
                f"(device='{device}', RAM={ram_gb:.1f} GB{vram_str}, cores={cores}; {cuda_note})"
            )

    return WhisperRuntimeSelection(
        model_name=model_name,
        device=device,
        compute_type=compute_type,
        system_ram_gb=ram_gb,
        cpu_cores=cores,
        cuda_available=cuda_ok,
        rationale=rationale,
        gpu_vram_gb=vram_gb,
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


def _parse_faster_whisper_segments(raw_segments: object) -> List[TranscriptSegment]:
    """Consume faster-whisper generator and convert into typed TranscriptSegment list."""
    results: List[TranscriptSegment] = []
    for idx, seg in enumerate(raw_segments):  # type: ignore[arg-type]
        text = (seg.text or "").strip()
        if not text:
            continue
        words_list: List[TranscriptWord] = []
        if getattr(seg, "words", None):
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
    return results


def transcribe_audio_stream(
    samples_16k: np.ndarray,
    config: EditingConfig,
    custom_transcriber: Optional[TranscriberCallable] = None,
) -> Tuple[List[TranscriptSegment], WhisperRuntimeSelection, str, List[str]]:
    """Transcribe a 16 kHz mono float32 audio stream using faster-whisper with full runtime CPU fallback.

    Note: CTranslate2 loads cuBLAS and allocates VRAM lazily during model.transcribe() iteration.
    Therefore, both WhisperModel() initialization AND segment consumption are protected by the
    CUDA -> CPU int8 fallback handler.
    """
    selection = select_whisper_model_and_device(config)
    warnings: List[str] = []

    if samples_16k.size == 0:
        return [], selection, "empty_audio", warnings

    if custom_transcriber is not None:
        segments = custom_transcriber(samples_16k, config, selection)
        return segments, selection, "completed_custom", warnings

    from faster_whisper import WhisperModel

    def _run_attempt(sel: WhisperRuntimeSelection) -> List[TranscriptSegment]:
        model = WhisperModel(
            sel.model_name,
            device=sel.device,
            compute_type=sel.compute_type,
            cpu_threads=min(4, max(1, sel.cpu_cores)),
        )
        raw_segments, _info = model.transcribe(
            samples_16k,
            language=config.language,
            beam_size=2 if sel.system_ram_gb < 12.0 else 3,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(
                threshold=config.vad_threshold,
                min_speech_duration_ms=200,
                min_silence_duration_ms=300,
                speech_pad_ms=150,
            ),
        )
        return _parse_faster_whisper_segments(raw_segments)

    active_selection = selection
    try:
        segments = _run_attempt(active_selection)
        return segments, active_selection, "completed", warnings
    except MemoryError as mem_err:
        warnings.append(
            f"Insufficient memory while running Whisper '{active_selection.model_name}' on {active_selection.device} ({mem_err}). "
            "Falling back to Silero VAD speech intervals; close heavy background applications or use '--whisper-model base --whisper-device cpu'."
        )
        return [], active_selection, "failed_out_of_memory", warnings
    except Exception as first_err:
        err_text = str(first_err)
        if active_selection.device == "cuda":
            fallback_model = (
                "base"
                if (config.whisper_model == "auto" and "memory" in err_text.lower())
                else active_selection.model_name
            )
            warnings.append(
                f"CUDA execution failed for Whisper '{active_selection.model_name}' ({first_err}); "
                f"falling back automatically to CPU (int8) with model '{fallback_model}'."
            )
            active_selection = WhisperRuntimeSelection(
                model_name=fallback_model,
                device="cpu",
                compute_type="int8",
                system_ram_gb=selection.system_ram_gb,
                cpu_cores=selection.cpu_cores,
                cuda_available=False,
                rationale=selection.rationale + " (Automatic CPU int8 fallback after CUDA error)",
                gpu_vram_gb=selection.gpu_vram_gb,
            )
            try:
                segments = _run_attempt(active_selection)
                return segments, active_selection, "completed_cpu_fallback", warnings
            except Exception as cpu_err:
                first_err = cpu_err
                err_text = str(cpu_err)

        if any(k in err_text.lower() for k in ("memory", "alloc", "bad_alloc", "out of memory")):
            warnings.append(
                f"Insufficient RAM/VRAM while loading or running Whisper '{active_selection.model_name}' ({first_err}). "
                "Continuing with Silero VAD speech intervals only; use '--whisper-model base --whisper-device cpu' to reduce memory usage."
            )
            return [], active_selection, "failed_out_of_memory", warnings

        warnings.append(
            f"Whisper model '{active_selection.model_name}' could not be loaded or downloaded ({first_err}). "
            "Continuing with local Silero VAD speech intervals only; connect to the internet and run "
            "'powershell -ExecutionPolicy Bypass -File .\\scripts\\install.ps1 -PreDownloadWhisperModel base' "
            "to cache the model locally for offline subtitle and stumble analysis."
        )
        return [], active_selection, "skipped_model_unavailable", warnings

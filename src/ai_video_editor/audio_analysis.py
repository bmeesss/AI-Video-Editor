"""Voice Activity Detection (Silero VAD ONNX + vocal-energy guard) and audio level/transient analysis."""

from __future__ import annotations

import math
from pathlib import Path
from typing import List

import av
import numpy as np
from faster_whisper.vad import VadOptions, get_speech_timestamps, get_vad_model

from .config import EditingConfig
from .models import AudioEnergyWindow, SpeechInterval
from .safety import capture_source_snapshot, verify_source_untouched


SAMPLE_RATE_16K = 16000


def extract_audio_pcm_16k_mono(source_path: Path | str, stream_index: int) -> np.ndarray:
    """Decode an audio stream in read-only mode to a 16 kHz mono float32 numpy array."""
    snap = capture_source_snapshot(source_path)
    path = Path(snap.resolved_path)
    chunks: List[np.ndarray] = []

    try:
        with av.open(str(path), mode="r") as container:
            target_stream = next(
                (s for s in container.streams.audio if s.index == stream_index),
                None,
            )
            if target_stream is None:
                if container.streams.audio:
                    target_stream = container.streams.audio[0]
                else:
                    verify_source_untouched(snap)
                    return np.zeros(0, dtype=np.float32)

            resampler = av.AudioResampler(format="fltp", layout="mono", rate=SAMPLE_RATE_16K)
            for frame in container.decode(target_stream):
                for rframe in resampler.resample(frame):
                    arr = rframe.to_ndarray()[0].astype(np.float32, copy=False)
                    if arr.size > 0:
                        chunks.append(arr)
    finally:
        verify_source_untouched(snap)

    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks)


def _merge_overlapping_intervals(intervals: List[SpeechInterval], min_gap_sec: float = 0.25) -> List[SpeechInterval]:
    """Merge overlapping or closely adjacent speech intervals chronologically."""
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda x: (x.start, x.end))
    merged: List[SpeechInterval] = [ordered[0]]
    for current in ordered[1:]:
        prev = merged[-1]
        if current.start <= prev.end + min_gap_sec:
            new_end = max(prev.end, current.end)
            new_conf = round(max(prev.confidence, current.confidence), 4)
            new_src = prev.source if prev.source == current.source else f"{prev.source}+{current.source}"
            merged[-1] = SpeechInterval(
                start=round(prev.start, 4),
                end=round(new_end, 4),
                confidence=new_conf,
                stream_index=prev.stream_index,
                source=new_src,
            )
        else:
            merged.append(current)
    return merged


def detect_voice_activity(
    samples_16k: np.ndarray,
    stream_index: int,
    config: EditingConfig,
    include_energy_guard: bool = True,
) -> List[SpeechInterval]:
    """Run Silero VAD ONNX model and conservative vocal-energy activity protection."""
    if samples_16k.size == 0:
        return []

    intervals: List[SpeechInterval] = []
    window_size_samples = 512

    # 1. Neural Silero VAD via faster-whisper + onnxruntime
    try:
        vad_opts = VadOptions(
            threshold=config.vad_threshold,
            min_speech_duration_ms=200,
            min_silence_duration_ms=250,
            speech_pad_ms=150,
        )
        raw_speeches = get_speech_timestamps(
            samples_16k,
            vad_options=vad_opts,
            sampling_rate=SAMPLE_RATE_16K,
        )
        pad_len = (window_size_samples - (samples_16k.shape[0] % window_size_samples)) % window_size_samples
        padded = np.pad(samples_16k, (0, pad_len)) if pad_len else samples_16k
        probs = get_vad_model()(padded).reshape(-1)

        for item in raw_speeches:
            s_sample = int(item["start"])
            e_sample = int(item["end"])
            w_start = max(0, s_sample // window_size_samples)
            w_end = max(w_start + 1, min(len(probs), (e_sample + window_size_samples - 1) // window_size_samples))
            seg_probs = probs[w_start:w_end]
            conf = float(np.mean(seg_probs)) if seg_probs.size > 0 else config.vad_threshold
            intervals.append(
                SpeechInterval(
                    start=round(s_sample / float(SAMPLE_RATE_16K), 4),
                    end=round(e_sample / float(SAMPLE_RATE_16K), 4),
                    confidence=round(min(1.0, max(0.5, conf)), 4),
                    stream_index=stream_index,
                    source="silero-vad",
                )
            )
    except Exception:
        # Fallback if ONNX runtime fails on an exotic platform
        include_energy_guard = True

    # 2. Conservative vocal-energy activity guard so non-verbal reactions (laughs, gasps)
    #    and active commentary above silence_dbfs_threshold + 8 dB are never clipped.
    if include_energy_guard:
        step = int(0.10 * SAMPLE_RATE_16K)  # 100 ms frames
        active_thresh_db = config.silence_dbfs_threshold + 8.0
        in_burst = False
        burst_start = 0.0
        burst_dbs: List[float] = []

        for idx in range(0, len(samples_16k), step):
            chunk = samples_16k[idx : idx + step]
            if chunk.size == 0:
                continue
            rms = float(np.sqrt(np.mean(np.square(chunk))))
            db = 20.0 * math.log10(max(rms, 1e-9))
            t_sec = idx / float(SAMPLE_RATE_16K)
            if db >= active_thresh_db:
                if not in_burst:
                    in_burst = True
                    burst_start = t_sec
                    burst_dbs = [db]
                else:
                    burst_dbs.append(db)
            else:
                if in_burst:
                    burst_end = t_sec
                    if burst_end - burst_start >= 0.20:
                        mean_db = float(np.mean(burst_dbs))
                        conf = min(0.90, max(0.60, (mean_db - active_thresh_db) / 25.0 + 0.65))
                        intervals.append(
                            SpeechInterval(
                                start=round(burst_start, 4),
                                end=round(burst_end, 4),
                                confidence=round(conf, 4),
                                stream_index=stream_index,
                                source="vocal-energy-guard",
                            )
                        )
                    in_burst = False
                    burst_dbs = []

        if in_burst:
            burst_end = len(samples_16k) / float(SAMPLE_RATE_16K)
            if burst_end - burst_start >= 0.20:
                intervals.append(
                    SpeechInterval(
                        start=round(burst_start, 4),
                        end=round(burst_end, 4),
                        confidence=0.75,
                        stream_index=stream_index,
                        source="vocal-energy-guard",
                    )
                )

    return _merge_overlapping_intervals(intervals, min_gap_sec=0.25)


def analyze_audio_energy_windows(
    samples_16k: np.ndarray,
    stream_index: int,
    config: EditingConfig,
) -> List[AudioEnergyWindow]:
    """Compute RMS and peak dBFS windows plus transient spike markers.

    Note: Loudness peaks are treated strictly as auxiliary signals (e.g. to preserve
    reactions or loud gameplay events), never as ground-truth 'funny/exciting' labels.
    """
    if samples_16k.size == 0:
        return []

    win_samples = max(160, int(round(config.audio_window_seconds * SAMPLE_RATE_16K)))
    raw_windows: List[ tuple[float, float, float, float] ] = []

    for start_idx in range(0, len(samples_16k), win_samples):
        chunk = samples_16k[start_idx : start_idx + win_samples]
        if chunk.size == 0:
            continue
        rms = float(np.sqrt(np.mean(np.square(chunk))))
        peak = float(np.max(np.abs(chunk)))
        rms_db = 20.0 * math.log10(max(rms, 1e-9))
        peak_db = 20.0 * math.log10(max(peak, 1e-9))
        s_sec = start_idx / float(SAMPLE_RATE_16K)
        e_sec = (start_idx + chunk.size) / float(SAMPLE_RATE_16K)
        raw_windows.append((s_sec, e_sec, rms_db, peak_db))

    if not raw_windows:
        return []

    non_silent_rms = [w[2] for w in raw_windows if w[2] > config.silence_dbfs_threshold]
    baseline_rms = float(np.median(non_silent_rms)) if non_silent_rms else -30.0

    results: List[AudioEnergyWindow] = []
    for s_sec, e_sec, rms_db, peak_db in raw_windows:
        is_quiet = rms_db < config.silence_dbfs_threshold
        is_transient = (
            peak_db >= config.transient_peak_dbfs
            and rms_db >= baseline_rms + 5.0
            and not is_quiet
        )
        results.append(
            AudioEnergyWindow(
                start=round(s_sec, 4),
                end=round(e_sec, 4),
                rms_dbfs=round(rms_db, 2),
                peak_dbfs=round(peak_db, 2),
                is_quiet=is_quiet,
                is_transient_peak=is_transient,
                stream_index=stream_index,
            )
        )
    return results

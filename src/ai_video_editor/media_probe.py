"""Non-destructive media metadata and multi-track audio stream inspection using PyAV."""

from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path
from typing import List, Tuple

import av
import numpy as np

from .models import AudioRole, AudioStreamInfo, MediaInfo, VideoStreamInfo
from .safety import SafetyViolationError, capture_source_snapshot, verify_source_untouched


STANDARD_FPS_FRACTIONS: List[Tuple[float, int, int]] = [
    (23.976023976023978, 24000, 1001),
    (24.0, 24, 1),
    (25.0, 25, 1),
    (29.97002997002997, 30000, 1001),
    (30.0, 30, 1),
    (50.0, 50, 1),
    (59.94005994005994, 60000, 1001),
    (60.0, 60, 1),
    (119.88011988011988, 120000, 1001),
    (120.0, 120, 1),
]


def normalize_fps_fraction(fps_value: float, raw_rate: Fraction | None = None) -> Tuple[int, int, float]:
    """Normalize a floating or rational frame rate to standard NLE frame rate numerator/denominator."""
    if fps_value <= 0 or math.isnan(fps_value) or math.isinf(fps_value):
        return 30, 1, 30.0

    for target_fps, num, den in STANDARD_FPS_FRACTIONS:
        if abs(fps_value - target_fps) < 0.005:
            return num, den, float(num) / float(den)

    if raw_rate is not None and raw_rate.denominator > 0 and raw_rate.numerator > 0:
        f = Fraction(raw_rate.numerator, raw_rate.denominator).limit_denominator(1001)
        return f.numerator, f.denominator, float(f.numerator) / float(f.denominator)

    rounded = max(1, int(round(fps_value)))
    return rounded, 1, float(rounded)


def _sample_audio_dbfs(path: Path, stream_index: int, max_seconds: float = 30.0) -> Tuple[float, float, float]:
    """Read-only sample of an audio stream returning (mean_dbfs, peak_dbfs, silence_fraction)."""
    rms_values: List[float] = []
    peak_val = 1e-9
    silent_windows = 0
    total_windows = 0

    with av.open(str(path), mode="r") as container:
        audio_stream = next(
            (s for s in container.streams.audio if s.index == stream_index),
            None,
        )
        if audio_stream is None:
            return -90.0, -90.0, 1.0

        resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)
        collected = 0
        max_samples = int(max_seconds * 16000)
        buf: List[np.ndarray] = []

        for frame in container.decode(audio_stream):
            for rframe in resampler.resample(frame):
                arr = rframe.to_ndarray()[0].astype(np.float32, copy=False)
                if arr.size == 0:
                    continue
                buf.append(arr)
                collected += arr.size
            if collected >= max_samples:
                break

        if not buf:
            return -90.0, -90.0, 1.0

        samples = np.concatenate(buf)[:max_samples]
        peak_val = max(float(np.max(np.abs(samples))), 1e-9)
        win_size = 1600  # 100 ms windows
        for start in range(0, len(samples), win_size):
            chunk = samples[start : start + win_size]
            if chunk.size == 0:
                continue
            rms = float(np.sqrt(np.mean(np.square(chunk))))
            db = 20.0 * math.log10(max(rms, 1e-9))
            rms_values.append(db)
            total_windows += 1
            if db < -42.0:
                silent_windows += 1

    mean_dbfs = float(np.mean(rms_values)) if rms_values else -90.0
    peak_dbfs = 20.0 * math.log10(max(peak_val, 1e-9))
    silence_fraction = float(silent_windows) / float(max(1, total_windows))
    return round(mean_dbfs, 2), round(peak_dbfs, 2), silence_fraction


def _detect_vfr(path: Path, video_stream_index: int, max_packets: int = 600) -> Tuple[bool, float]:
    """Inspect packet PTS deltas in read-only mode to detect Variable Frame Rate (VFR) recordings."""
    pts_seconds: List[float] = []
    with av.open(str(path), mode="r") as container:
        vstream = next(
            (s for s in container.streams.video if s.index == video_stream_index),
            None,
        )
        if vstream is None or vstream.time_base is None:
            return False, 0.0
        tb = float(vstream.time_base)
        for packet in container.demux(vstream):
            if packet.pts is not None:
                pts_seconds.append(float(packet.pts) * tb)
            if len(pts_seconds) >= max_packets:
                break

    if len(pts_seconds) < 10:
        return False, 0.0

    pts_seconds.sort()
    deltas = np.diff(np.array(pts_seconds, dtype=np.float64))
    deltas = deltas[deltas > 1e-5]
    if deltas.size < 5:
        return False, 0.0

    mean_dt = float(np.mean(deltas))
    if mean_dt <= 0:
        return False, 0.0
    std_dt = float(np.std(deltas))
    jitter_ratio = round(std_dt / mean_dt, 4)
    # VFR recordings typically exhibit >5% frame interval standard deviation
    return jitter_ratio > 0.05, jitter_ratio


def _infer_audio_roles(
    raw_streams: List[Tuple[int, int, str, int, int, float, str, str, float, float, float]]
) -> List[AudioStreamInfo]:
    """Assign semantic roles (mixed, microphone, gameplay, secondary) to audio tracks."""
    if not raw_streams:
        return []

    if len(raw_streams) == 1:
        idx, order, codec, sr, ch, dur, title, lang, mean_db, peak_db, _ = raw_streams[0]
        title_lower = title.lower()
        if any(k in title_lower for k in ("mic", "voice", "commentary", "vocal")):
            role: AudioRole = "microphone"
        elif any(k in title_lower for k in ("game", "desktop", "system")):
            role = "gameplay"
        else:
            role = "mixed"
        return [
            AudioStreamInfo(
                stream_index=idx,
                audio_track_order=order,
                codec_name=codec,
                sample_rate=sr,
                channels=ch,
                duration_seconds=dur,
                title=title,
                language=lang,
                inferred_role=role,
                mean_dbfs=mean_db,
                peak_dbfs=peak_db,
            )
        ]

    # Multi-track container (common in OBS Studio: Track 1 Mic + Track 2 Game, or Track 1 Mix + Track 2 Mic + Track 3 Game)
    roles: List[AudioRole] = ["unknown"] * len(raw_streams)
    for i, item in enumerate(raw_streams):
        title_lower = item[6].lower()
        if any(k in title_lower for k in ("mic", "voice", "commentary", "vocal", "input")):
            roles[i] = "microphone"
        elif any(k in title_lower for k in ("game", "desktop", "system", "speaker")):
            roles[i] = "gameplay"
        elif any(k in title_lower for k in ("mix", "master", "combined", "full")):
            roles[i] = "mixed"

    # If metadata titles did not disambiguate, use silence fraction heuristic:
    # A microphone track has intermittent pauses (higher silence_fraction) compared to continuous game audio.
    if "microphone" not in roles:
        best_mic_idx = max(range(len(raw_streams)), key=lambda i: raw_streams[i][10])
        roles[best_mic_idx] = "microphone"

    if "gameplay" not in roles:
        for i in range(len(raw_streams)):
            if roles[i] == "unknown":
                roles[i] = "gameplay"
                break

    for i in range(len(raw_streams)):
        if roles[i] == "unknown":
            roles[i] = "secondary"

    result: List[AudioStreamInfo] = []
    for i, item in enumerate(raw_streams):
        idx, order, codec, sr, ch, dur, title, lang, mean_db, peak_db, _ = item
        result.append(
            AudioStreamInfo(
                stream_index=idx,
                audio_track_order=order,
                codec_name=codec,
                sample_rate=sr,
                channels=ch,
                duration_seconds=dur,
                title=title,
                language=lang,
                inferred_role=roles[i],
                mean_dbfs=mean_db,
                peak_dbfs=peak_db,
            )
        )
    return result


def probe_media(source_path: Path | str) -> MediaInfo:
    """Inspect a media file in strictly read-only mode and return a typed MediaInfo model."""
    snap = capture_source_snapshot(source_path)
    path = Path(snap.resolved_path)
    warnings: List[str] = []

    try:
        with av.open(str(path), mode="r") as container:
            container_format = container.format.name if container.format else path.suffix.lstrip(".")
            container_duration = (
                float(container.duration) / float(av.time_base)
                if container.duration is not None
                else 0.0
            )

            video_streams_info: List[VideoStreamInfo] = []
            for vstream in container.streams.video:
                avg_rate = vstream.average_rate or vstream.guessed_rate or vstream.base_rate
                raw_fps = float(avg_rate) if avg_rate is not None else 30.0
                raw_frac = (
                    Fraction(int(avg_rate.numerator), int(avg_rate.denominator))
                    if avg_rate is not None and avg_rate.denominator
                    else None
                )
                num, den, norm_fps = normalize_fps_fraction(raw_fps, raw_frac)

                if vstream.duration is not None and vstream.time_base is not None:
                    v_dur = float(vstream.duration * vstream.time_base)
                else:
                    v_dur = container_duration

                is_vfr, jitter_ratio = _detect_vfr(path, vstream.index)
                if is_vfr:
                    warnings.append(
                        f"Video stream {vstream.index} appears to use Variable Frame Rate (VFR jitter={jitter_ratio:.3f}). "
                        "Frame-accurate NLE timelines may experience slight audio/video sync drift on long recordings; "
                        "verify alignment in Kdenlive before final manual export."
                    )

                sar = vstream.sample_aspect_ratio
                sar_num = int(sar.numerator) if sar and sar.numerator else 1
                sar_den = int(sar.denominator) if sar and sar.denominator else 1
                width = int(vstream.codec_context.width or 1920)
                height = int(vstream.codec_context.height or 1080)
                dar_frac = Fraction(width * sar_num, max(1, height * sar_den)).limit_denominator(64)

                total_frames = max(1, int(round(v_dur * norm_fps))) if v_dur > 0 else int(vstream.frames or 1)
                if v_dur <= 0 and total_frames > 0:
                    v_dur = float(total_frames) / norm_fps

                video_streams_info.append(
                    VideoStreamInfo(
                        stream_index=int(vstream.index),
                        codec_name=str(vstream.codec_context.name or "h264"),
                        width=width,
                        height=height,
                        frame_rate_num=num,
                        frame_rate_den=den,
                        fps=round(norm_fps, 6),
                        duration_seconds=round(v_dur, 4),
                        total_frames=total_frames,
                        pix_fmt=str(vstream.codec_context.pix_fmt or "yuv420p"),
                        colorspace=709 if height >= 720 else 601,
                        progressive=1,
                        sample_aspect_num=sar_num,
                        sample_aspect_den=sar_den,
                        display_aspect_num=dar_frac.numerator,
                        display_aspect_den=dar_frac.denominator,
                        is_vfr=is_vfr,
                        vfr_jitter_ratio=jitter_ratio,
                    )
                )

            raw_audio_streams: List[Tuple[int, int, str, int, int, float, str, str, float, float, float]] = []
            for order, astream in enumerate(container.streams.audio):
                if astream.duration is not None and astream.time_base is not None:
                    a_dur = float(astream.duration * astream.time_base)
                else:
                    a_dur = container_duration or (
                        video_streams_info[0].duration_seconds if video_streams_info else 0.0
                    )
                meta = astream.metadata or {}
                title = str(meta.get("title") or meta.get("HANDLER_NAME") or meta.get("handler_name") or "")
                lang = str(meta.get("language") or "")
                mean_db, peak_db, silence_frac = _sample_audio_dbfs(path, astream.index)
                raw_audio_streams.append(
                    (
                        int(astream.index),
                        order,
                        str(astream.codec_context.name or "aac"),
                        int(astream.codec_context.sample_rate or 48000),
                        int(astream.codec_context.channels or 2),
                        round(a_dur, 4),
                        title,
                        lang,
                        mean_db,
                        peak_db,
                        silence_frac,
                    )
                )
    except Exception as exc:
        verify_source_untouched(snap)
        raise SafetyViolationError(f"Failed to probe media file '{path}': {exc}") from exc

    verify_source_untouched(snap)

    if not video_streams_info:
        warnings.append("No video stream detected in container.")
    if not raw_audio_streams:
        warnings.append("No audio streams detected in container; audio-based silence and speech analysis will be unavailable.")

    audio_streams_info = _infer_audio_roles(raw_audio_streams)
    overall_duration = container_duration
    if video_streams_info and video_streams_info[0].duration_seconds > 0:
        overall_duration = video_streams_info[0].duration_seconds
    elif audio_streams_info and audio_streams_info[0].duration_seconds > 0:
        overall_duration = audio_streams_info[0].duration_seconds

    return MediaInfo(
        path=str(path),
        filename=path.name,
        container_format=container_format,
        duration_seconds=round(overall_duration, 4),
        file_size_bytes=snap.size_bytes,
        file_hash_md5=snap.md5_hex,
        file_hash_sha256=snap.sha256_hex,
        video_streams=video_streams_info,
        audio_streams=audio_streams_info,
        warnings=warnings,
    )

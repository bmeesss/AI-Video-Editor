"""Full Phase 2 media analysis pipeline producing a canonical, inspectable AnalysisReport."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import List, Optional

from .audio_analysis import (
    analyze_audio_energy_windows,
    detect_voice_activity,
    extract_audio_pcm_16k_mono,
)
from .config import EditingConfig
from .media_probe import probe_media
from .models import (
    AnalysisReport,
    AudioEnergyWindow,
    SpeechInterval,
    TranscriptSegment,
)
from .safety import capture_source_snapshot, verify_source_untouched
from .speech_transcriber import TranscriberCallable, transcribe_audio_stream
from .visual_analysis import analyze_visual_scenes_and_motion


def _merge_transcript_into_speech_intervals(
    vad_intervals: List[SpeechInterval],
    transcript: List[TranscriptSegment],
    stream_index: int,
) -> List[SpeechInterval]:
    """Ensure every transcribed speech segment is also represented in speech_intervals."""
    combined = list(vad_intervals)
    for seg in transcript:
        if seg.text.strip():
            combined.append(
                SpeechInterval(
                    start=seg.start,
                    end=seg.end,
                    confidence=seg.confidence,
                    stream_index=stream_index,
                    source="whisper-transcript",
                )
            )
    if not combined:
        return []

    ordered = sorted(combined, key=lambda x: (x.start, x.end))
    merged: List[SpeechInterval] = [ordered[0]]
    for current in ordered[1:]:
        prev = merged[-1]
        if current.start <= prev.end + 0.20:
            merged[-1] = SpeechInterval(
                start=round(prev.start, 4),
                end=round(max(prev.end, current.end), 4),
                confidence=round(max(prev.confidence, current.confidence), 4),
                stream_index=prev.stream_index,
                source=prev.source if prev.source == current.source else f"{prev.source}+{current.source}",
            )
        else:
            merged.append(current)
    return merged


def analyze_recording(
    source_path: Path | str,
    config: Optional[EditingConfig] = None,
    custom_transcriber: Optional[TranscriberCallable] = None,
) -> AnalysisReport:
    """Analyze a gaming recording in strictly read-only mode and return a canonical AnalysisReport."""
    cfg = config or EditingConfig()
    initial_snap = capture_source_snapshot(source_path)

    try:
        media_info = probe_media(source_path)
        warnings: List[str] = list(media_info.warnings)
        notes: List[str] = [
            "Source media inspected in strictly read-only mode; SHA-256 and MD5 hashes verified before and after analysis.",
            "Visual motion and audio loudness transients are treated solely as contextual signals to avoid cutting active gameplay or reactions, never as ground-truth labels of 'funny' or 'exciting' moments.",
        ]

        speech_intervals: List[SpeechInterval] = []
        transcript_segments: List[TranscriptSegment] = []
        audio_energy_windows: List[AudioEnergyWindow] = []
        whisper_model_used = "none"
        whisper_device_used = "cpu"
        transcription_status = "no_audio"

        primary_mic = media_info.primary_mic_stream
        if primary_mic is not None:
            notes.append(
                f"Selected audio stream {primary_mic.stream_index} (role='{primary_mic.inferred_role}', "
                f"title='{primary_mic.title or 'untitled'}') as primary commentary/speech analysis stream."
            )
            mic_samples = extract_audio_pcm_16k_mono(source_path, primary_mic.stream_index)
            vad_intervals = detect_voice_activity(mic_samples, primary_mic.stream_index, cfg)
            mic_energy = analyze_audio_energy_windows(mic_samples, primary_mic.stream_index, cfg)
            audio_energy_windows.extend(mic_energy)

            transcript_segments, selection, transcription_status, trans_warnings = transcribe_audio_stream(
                mic_samples,
                cfg,
                custom_transcriber=custom_transcriber,
            )
            whisper_model_used = selection.model_name
            whisper_device_used = selection.device
            notes.append(selection.rationale)
            warnings.extend(trans_warnings)

            speech_intervals = _merge_transcript_into_speech_intervals(
                vad_intervals,
                transcript_segments,
                primary_mic.stream_index,
            )

            # Also analyze secondary/gameplay audio streams for loudness transients so loud game events aren't ignored
            for astream in media_info.audio_streams:
                if astream.stream_index == primary_mic.stream_index:
                    continue
                notes.append(
                    f"Analyzed secondary audio stream {astream.stream_index} (role='{astream.inferred_role}', "
                    f"title='{astream.title or 'untitled'}') for gameplay audio energy and transients."
                )
                sec_samples = extract_audio_pcm_16k_mono(source_path, astream.stream_index)
                sec_energy = analyze_audio_energy_windows(sec_samples, astream.stream_index, cfg)
                audio_energy_windows.extend(sec_energy)

        scenes, motion_windows = analyze_visual_scenes_and_motion(source_path, media_info, cfg)

        # Compute overall analysis confidence
        conf_components: List[float] = []
        if media_info.video_streams:
            conf_components.append(0.90 if not media_info.video_streams[0].is_vfr else 0.70)
        else:
            conf_components.append(0.25)
        if speech_intervals:
            mean_vad_conf = sum(s.confidence for s in speech_intervals) / float(len(speech_intervals))
            conf_components.append(mean_vad_conf)
        if transcript_segments:
            mean_tx_conf = sum(t.confidence for t in transcript_segments) / float(len(transcript_segments))
            conf_components.append(mean_tx_conf)
        elif transcription_status.startswith("skipped") or transcription_status.startswith("failed"):
            # Lower confidence slightly when Whisper model weights were unavailable offline
            conf_components.append(0.68)

        overall_conf = round(
            sum(conf_components) / float(len(conf_components)) if conf_components else 0.50,
            4,
        )
    finally:
        verify_source_untouched(initial_snap)

    return AnalysisReport(
        schema_version="1.0.0",
        created_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        media_info=media_info,
        whisper_model_used=whisper_model_used,
        whisper_device_used=whisper_device_used,
        transcription_status=transcription_status,
        speech_intervals=speech_intervals,
        transcript_segments=transcript_segments,
        audio_energy_windows=audio_energy_windows,
        scene_boundaries=scenes,
        motion_windows=motion_windows,
        overall_analysis_confidence=overall_conf,
        warnings=warnings,
        notes=notes,
    )

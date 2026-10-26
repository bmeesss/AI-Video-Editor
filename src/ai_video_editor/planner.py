"""Deterministic, chronological edit-decision planner for gaming videos."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .config import EditingConfig
from .models import (
    AnalysisReport,
    EditDecisionModel,
    SubtitleEntry,
    TimelineClip,
    TimelineMarker,
    TimelineTrack,
)


@dataclass
class _FrameSpan:
    start_frame: int  # inclusive
    end_frame: int    # exclusive (so length in frames = end_frame - start_frame)
    keep: bool
    reason: str
    confidence: float

    @property
    def length_frames(self) -> int:
        return self.end_frame - self.start_frame


def seconds_to_frame(seconds: float, fps_num: int, fps_den: int, max_frames: int) -> int:
    """Convert seconds to integer frame index clamped to [0, max_frames]."""
    fps = float(fps_num) / float(max(1, fps_den))
    f = int(round(seconds * fps))
    return max(0, min(max_frames, f))


def frame_to_seconds(frame: int, fps_num: int, fps_den: int) -> float:
    """Convert integer frame index to seconds."""
    fps = float(fps_num) / float(max(1, fps_den))
    return round(float(frame) / fps, 4)


def format_timecode_hms(seconds: float) -> str:
    """Format seconds as HH:MM:SS.ff for human-readable markers and notes."""
    s = max(0.0, float(seconds))
    hours = int(s // 3600)
    minutes = int((s % 3600) // 60)
    secs = s % 60.0
    return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"


def _build_frame_partitions(
    report: AnalysisReport,
    config: EditingConfig,
    total_frames: int,
    fps_num: int,
    fps_den: int,
) -> List[_FrameSpan]:
    """Partition [0, total_frames) into contiguous keep/remove spans using conservative rules."""
    fps = float(fps_num) / float(max(1, fps_den))

    # Per-frame state arrays
    keep_mask = np.zeros(total_frames, dtype=bool)
    stumble_mask = np.zeros(total_frames, dtype=bool)
    reason_codes = np.full(total_frames, fill_value=0, dtype=np.int8)
    # Reason priority codes:
    # 0 = dead_air_candidate
    # 1 = gameplay_audio_activity
    # 2 = active_gameplay_motion
    # 3 = speech_commentary
    # 4 = preserved_intro
    # -1 = obvious_stumble_candidate
    conf_array = np.full(total_frames, fill_value=0.82, dtype=np.float32)

    # 1. Gameplay audio energy & transient protection
    for win in report.audio_energy_windows:
        if win.is_transient_peak or win.rms_dbfs >= (config.silence_dbfs_threshold + 6.0):
            sf = seconds_to_frame(win.start, fps_num, fps_den, total_frames)
            ef = seconds_to_frame(win.end, fps_num, fps_den, total_frames)
            if ef > sf:
                keep_mask[sf:ef] = True
                mask_slice = reason_codes[sf:ef] < 1
                reason_codes[sf:ef][mask_slice] = 1
                conf_array[sf:ef] = np.maximum(conf_array[sf:ef], 0.80)

    # 2. Active visual gameplay motion protection (never cut tense/quiet active gameplay!)
    for mwin in report.motion_windows:
        if mwin.is_active_gameplay:
            sf = seconds_to_frame(mwin.start, fps_num, fps_den, total_frames)
            ef = seconds_to_frame(mwin.end, fps_num, fps_den, total_frames)
            if ef > sf:
                keep_mask[sf:ef] = True
                mask_slice = reason_codes[sf:ef] < 2
                reason_codes[sf:ef][mask_slice] = 2
                conf_array[sf:ef] = np.maximum(conf_array[sf:ef], 0.85)

    # 3. Speech intervals (VAD + transcript) expanded by pre-roll and post-roll safety margins
    for sp in report.speech_intervals:
        padded_start = max(0.0, sp.start - config.speech_pre_roll_seconds)
        padded_end = sp.end + config.speech_post_roll_seconds
        sf = seconds_to_frame(padded_start, fps_num, fps_den, total_frames)
        ef = seconds_to_frame(padded_end, fps_num, fps_den, total_frames)
        if ef > sf:
            keep_mask[sf:ef] = True
            mask_slice = reason_codes[sf:ef] < 3
            reason_codes[sf:ef][mask_slice] = 3
            conf_array[sf:ef] = np.maximum(conf_array[sf:ef], float(sp.confidence))

    # 4. Obvious verbal stumbles (only when remove_obvious_stumbles=True and high confidence)
    if config.remove_obvious_stumbles:
        for seg in report.transcript_segments:
            if seg.is_stumble_candidate and seg.stumble_confidence >= config.min_stumble_confidence:
                sf = seconds_to_frame(seg.start, fps_num, fps_den, total_frames)
                ef = seconds_to_frame(seg.end, fps_num, fps_den, total_frames)
                if ef > sf:
                    keep_mask[sf:ef] = False
                    stumble_mask[sf:ef] = True
                    reason_codes[sf:ef] = -1
                    conf_array[sf:ef] = float(seg.stumble_confidence)

    # 5. Always preserve the original intro at the beginning (non-negotiable rule)
    intro_frames = seconds_to_frame(config.preserve_intro_seconds, fps_num, fps_den, total_frames)
    if intro_frames > 0:
        keep_mask[:intro_frames] = True
        stumble_mask[:intro_frames] = False
        reason_codes[:intro_frames] = 4
        conf_array[:intro_frames] = 0.98

    # Convert frame masks into initial contiguous runs of keep/remove
    raw_spans: List[_FrameSpan] = []
    reason_map = {
        4: "preserved_intro",
        3: "speech_commentary",
        2: "active_gameplay_motion",
        1: "gameplay_audio_activity",
        0: "dead_air_silence",
        -1: "obvious_stumble",
    }

    idx = 0
    while idx < total_frames:
        k_val = bool(keep_mask[idx])
        s_val = bool(stumble_mask[idx])
        end_idx = idx + 1
        while (
            end_idx < total_frames
            and bool(keep_mask[end_idx]) == k_val
            and bool(stumble_mask[end_idx]) == s_val
        ):
            end_idx += 1

        dominant_code = int(np.max(reason_codes[idx:end_idx])) if k_val else (-1 if s_val else 0)
        mean_conf = round(float(np.mean(conf_array[idx:end_idx])), 4)
        raw_spans.append(
            _FrameSpan(
                start_frame=idx,
                end_frame=end_idx,
                keep=k_val,
                reason=reason_map.get(dominant_code, "kept_context"),
                confidence=mean_conf,
            )
        )
        idx = end_idx

    # Enforce minimum silence duration: if a 'dead_air_silence' span is shorter than
    # (min_silence_duration_seconds - pre_roll - post_roll), keep it as a natural pause!
    min_trimmed_silence_sec = max(
        0.5,
        config.min_silence_duration_seconds - config.speech_pre_roll_seconds - config.speech_post_roll_seconds,
    )
    min_silence_frames = max(1, int(round(min_trimmed_silence_sec * fps)))
    min_stumble_frames = max(1, int(round(0.30 * fps)))

    for span in raw_spans:
        if not span.keep:
            if span.reason == "obvious_stumble":
                if span.length_frames < min_stumble_frames:
                    span.keep = True
                    span.reason = "short_pause_preserved"
            else:
                if span.length_frames < min_silence_frames:
                    span.keep = True
                    span.reason = "natural_pause_preserved"

    # Merge adjacent spans with identical keep status
    merged_spans: List[_FrameSpan] = []
    for span in raw_spans:
        if merged_spans and merged_spans[-1].keep == span.keep:
            prev = merged_spans[-1]
            total_len = prev.length_frames + span.length_frames
            weighted_conf = (
                prev.confidence * prev.length_frames + span.confidence * span.length_frames
            ) / float(max(1, total_len))
            # Prefer intro or speech reason label when merging kept spans
            priority = {
                "preserved_intro": 5,
                "speech_commentary": 4,
                "active_gameplay_motion": 3,
                "gameplay_audio_activity": 2,
                "natural_pause_preserved": 1,
                "short_pause_preserved": 1,
                "obvious_stumble": 2,
                "dead_air_silence": 1,
            }
            best_reason = (
                prev.reason
                if priority.get(prev.reason, 0) >= priority.get(span.reason, 0)
                else span.reason
            )
            merged_spans[-1] = _FrameSpan(
                start_frame=prev.start_frame,
                end_frame=span.end_frame,
                keep=prev.keep,
                reason=best_reason,
                confidence=round(weighted_conf, 4),
            )
        else:
            merged_spans.append(span)

    # Enforce min_kept_clip_duration_seconds: never leave tiny micro-clips between cuts;
    # conservatively keep short neighbor bridges rather than making jarring jump cuts.
    min_kept_frames = max(1, int(round(config.min_kept_clip_duration_seconds * fps)))
    if len(merged_spans) > 1:
        for i, span in enumerate(merged_spans):
            if span.keep and span.length_frames < min_kept_frames:
                # Expand into adjacent removed span to avoid a micro-clip
                if i + 1 < len(merged_spans) and not merged_spans[i + 1].keep:
                    merged_spans[i + 1].keep = True
                    merged_spans[i + 1].reason = "bridge_pause_preserved"
                elif i - 1 >= 0 and not merged_spans[i - 1].keep:
                    merged_spans[i - 1].keep = True
                    merged_spans[i - 1].reason = "bridge_pause_preserved"

        # Final merge pass after micro-clip protection
        final_spans: List[_FrameSpan] = []
        for span in merged_spans:
            if final_spans and final_spans[-1].keep == span.keep:
                prev = final_spans[-1]
                final_spans[-1] = _FrameSpan(
                    start_frame=prev.start_frame,
                    end_frame=span.end_frame,
                    keep=prev.keep,
                    reason=prev.reason,
                    confidence=round((prev.confidence + span.confidence) / 2.0, 4),
                )
            else:
                final_spans.append(span)
        merged_spans = final_spans

    # Safety fallback: never remove 100% of the video
    if not any(s.keep for s in merged_spans):
        merged_spans = [
            _FrameSpan(
                start_frame=0,
                end_frame=total_frames,
                keep=True,
                reason="fallback_full_video_preserved",
                confidence=0.50,
            )
        ]

    return merged_spans


def _map_subtitles_to_edited_timeline(
    report: AnalysisReport,
    kept_spans_with_timeline_pos: List[Tuple[_FrameSpan, int, int]],
    fps_num: int,
    fps_den: int,
) -> List[SubtitleEntry]:
    """Map source transcript segments onto the edited timeline's frame/time coordinates."""
    if not report.transcript_segments:
        return []

    fps = float(fps_num) / float(max(1, fps_den))
    total_frames = kept_spans_with_timeline_pos[-1][0].end_frame if kept_spans_with_timeline_pos else 1
    subtitles: List[SubtitleEntry] = []
    sub_idx = 1

    for seg in report.transcript_segments:
        text = seg.text.strip()
        if not text:
            continue
        seg_in_f = seconds_to_frame(seg.start, fps_num, fps_den, total_frames)
        seg_out_f = seconds_to_frame(seg.end, fps_num, fps_den, total_frames)
        if seg_out_f <= seg_in_f:
            seg_out_f = seg_in_f + max(1, int(round(0.5 * fps)))

        for span, tl_start_f, _tl_end_f in kept_spans_with_timeline_pos:
            overlap_start = max(seg_in_f, span.start_frame)
            overlap_end = min(seg_out_f, span.end_frame)
            if overlap_end > overlap_start:
                mapped_tl_in_f = tl_start_f + (overlap_start - span.start_frame)
                mapped_tl_out_f = tl_start_f + (overlap_end - span.start_frame)
                subtitles.append(
                    SubtitleEntry(
                        index=sub_idx,
                        start_sec=round(mapped_tl_in_f / fps, 3),
                        end_sec=round(mapped_tl_out_f / fps, 3),
                        text=text,
                        confidence=seg.confidence,
                    )
                )
                sub_idx += 1
                break

    return subtitles


def build_edit_decision_model(
    report: AnalysisReport,
    title: str,
    config: Optional[EditingConfig] = None,
) -> EditDecisionModel:
    """Create a deterministic, frame-accurate EditDecisionModel from an AnalysisReport."""
    cfg = config or EditingConfig()
    media = report.media_info
    vstream = media.primary_video

    if vstream is not None:
        fps_num = vstream.frame_rate_num
        fps_den = vstream.frame_rate_den
        width = vstream.width
        height = vstream.height
        total_frames = max(1, vstream.total_frames)
    else:
        fps_num, fps_den = 30, 1
        width, height = 1920, 1080
        total_frames = max(1, int(round(media.duration_seconds * 30.0)))

    spans = _build_frame_partitions(
        report=report,
        config=cfg,
        total_frames=total_frames,
        fps_num=fps_num,
        fps_den=fps_den,
    )

    # Separate kept and removed spans while tracking exact edited timeline frame positions
    kept_with_tl: List[Tuple[_FrameSpan, int, int]] = []
    removed_with_tl_anchor: List[Tuple[_FrameSpan, int]] = []

    tl_cursor = 0
    for span in spans:
        if span.keep:
            tl_start = tl_cursor
            tl_end = tl_cursor + span.length_frames
            kept_with_tl.append((span, tl_start, tl_end))
            tl_cursor = tl_end
        else:
            removed_with_tl_anchor.append((span, tl_cursor))

    edited_duration_frames = max(1, tl_cursor)

    # Build primary video track clips (V1)
    v1_clips: List[TimelineClip] = []
    for idx, (span, tl_start, tl_end) in enumerate(kept_with_tl):
        src_in_f = span.start_frame
        src_out_f = span.end_frame - 1  # MLT inclusive out frame
        v1_clips.append(
            TimelineClip(
                clip_id=f"v1_keep_{idx:03d}",
                source_path=media.path,
                source_in_sec=frame_to_seconds(src_in_f, fps_num, fps_den),
                source_out_sec=frame_to_seconds(span.end_frame, fps_num, fps_den),
                source_in_frame=src_in_f,
                source_out_frame=src_out_f,
                timeline_start_sec=frame_to_seconds(tl_start, fps_num, fps_den),
                timeline_end_sec=frame_to_seconds(tl_end, fps_num, fps_den),
                timeline_in_frame=tl_start,
                timeline_out_frame=tl_end - 1,
                duration_frames=span.length_frames,
                track_type="video",
                audio_stream_index=None,
                enabled=True,
                action="keep",
                decision_reason=span.reason,
                confidence=span.confidence,
            )
        )

    tracks: List[TimelineTrack] = [
        TimelineTrack(
            track_id="V1",
            track_name="V1 - Gameplay Video",
            track_type="video",
            is_review_track=False,
            is_muted=False,
            is_hidden=False,
            audio_stream_index=None,
            clips=v1_clips,
        )
    ]

    # Build audio tracks for each source audio stream (up to 2 primary audio tracks: Mic + Game Audio)
    audio_streams = media.audio_streams[:2] if media.audio_streams else []
    for a_idx, astream in enumerate(audio_streams):
        track_id = f"A{a_idx + 1}"
        role_label = astream.inferred_role.capitalize()
        track_name = f"{track_id} - {astream.title or role_label}"
        a_clips: List[TimelineClip] = []
        for idx, (span, tl_start, tl_end) in enumerate(kept_with_tl):
            src_in_f = span.start_frame
            src_out_f = span.end_frame - 1
            a_clips.append(
                TimelineClip(
                    clip_id=f"{track_id.lower()}_keep_{idx:03d}",
                    source_path=media.path,
                    source_in_sec=frame_to_seconds(src_in_f, fps_num, fps_den),
                    source_out_sec=frame_to_seconds(span.end_frame, fps_num, fps_den),
                    source_in_frame=src_in_f,
                    source_out_frame=src_out_f,
                    timeline_start_sec=frame_to_seconds(tl_start, fps_num, fps_den),
                    timeline_end_sec=frame_to_seconds(tl_end, fps_num, fps_den),
                    timeline_in_frame=tl_start,
                    timeline_out_frame=tl_end - 1,
                    duration_frames=span.length_frames,
                    track_type="audio",
                    audio_stream_index=astream.stream_index,
                    enabled=True,
                    action="keep",
                    decision_reason=span.reason,
                    confidence=span.confidence,
                )
            )
        tracks.append(
            TimelineTrack(
                track_id=track_id,
                track_name=track_name,
                track_type="audio",
                is_review_track=False,
                is_muted=False,
                is_hidden=False,
                audio_stream_index=astream.stream_index,
                clips=a_clips,
            )
        )

    # Build removed clips summary, markers/guides, and optional review-layers tracks
    removed_summary: List[TimelineClip] = []
    markers: List[TimelineMarker] = [
        TimelineMarker(
            frame=0,
            position_sec=0.0,
            comment=f"Preserved Intro ({cfg.preserve_intro_seconds:.1f}s target)",
            category=3,  # Kdenlive green/info guide category
        )
    ]

    review_v_clips: List[TimelineClip] = []
    review_a_clips: List[TimelineClip] = []
    review_cursor_frame = 0
    primary_audio_stream_idx = (
        media.primary_mic_stream.stream_index if media.primary_mic_stream is not None else None
    )

    for r_idx, (span, cut_tl_frame) in enumerate(removed_with_tl_anchor):
        src_in_f = span.start_frame
        src_out_f = span.end_frame - 1
        src_in_sec = frame_to_seconds(src_in_f, fps_num, fps_den)
        src_out_sec = frame_to_seconds(span.end_frame, fps_num, fps_den)
        dur_sec = round(src_out_sec - src_in_sec, 2)

        # Record full removed segment in removed_summary (and bin clipzones)
        full_removed_clip = TimelineClip(
            clip_id=f"removed_{r_idx:03d}",
            source_path=media.path,
            source_in_sec=src_in_sec,
            source_out_sec=src_out_sec,
            source_in_frame=src_in_f,
            source_out_frame=src_out_f,
            timeline_start_sec=frame_to_seconds(cut_tl_frame, fps_num, fps_den),
            timeline_end_sec=frame_to_seconds(cut_tl_frame + span.length_frames, fps_num, fps_den),
            timeline_in_frame=cut_tl_frame,
            timeline_out_frame=cut_tl_frame + span.length_frames - 1,
            duration_frames=span.length_frames,
            track_type="video",
            audio_stream_index=None,
            enabled=False,
            action="remove",
            decision_reason=span.reason,
            confidence=span.confidence,
        )
        removed_summary.append(full_removed_clip)

        # Place non-overlapping disabled review clip on review tracks, clamped to edited_duration_frames
        # so the review layer never extends the sequence length or adds trailing black frames.
        rev_tl_start = max(cut_tl_frame, review_cursor_frame)
        if rev_tl_start < edited_duration_frames:
            avail_frames = min(span.length_frames, edited_duration_frames - rev_tl_start)
            if avail_frames >= 1:
                rev_tl_end = rev_tl_start + avail_frames
                rev_src_out_f = src_in_f + avail_frames - 1
                rev_src_out_sec = frame_to_seconds(src_in_f + avail_frames, fps_num, fps_den)
                review_cursor_frame = rev_tl_end

                rev_v_clip = TimelineClip(
                    clip_id=f"v_rev_{r_idx:03d}",
                    source_path=media.path,
                    source_in_sec=src_in_sec,
                    source_out_sec=rev_src_out_sec,
                    source_in_frame=src_in_f,
                    source_out_frame=rev_src_out_f,
                    timeline_start_sec=frame_to_seconds(rev_tl_start, fps_num, fps_den),
                    timeline_end_sec=frame_to_seconds(rev_tl_end, fps_num, fps_den),
                    timeline_in_frame=rev_tl_start,
                    timeline_out_frame=rev_tl_end - 1,
                    duration_frames=avail_frames,
                    track_type="video",
                    audio_stream_index=None,
                    enabled=False,
                    action="remove",
                    decision_reason=span.reason,
                    confidence=span.confidence,
                )
                review_v_clips.append(rev_v_clip)

                if primary_audio_stream_idx is not None:
                    review_a_clips.append(
                        TimelineClip(
                            clip_id=f"a_rev_{r_idx:03d}",
                            source_path=media.path,
                            source_in_sec=src_in_sec,
                            source_out_sec=rev_src_out_sec,
                            source_in_frame=src_in_f,
                            source_out_frame=rev_src_out_f,
                            timeline_start_sec=frame_to_seconds(rev_tl_start, fps_num, fps_den),
                            timeline_end_sec=frame_to_seconds(rev_tl_end, fps_num, fps_den),
                            timeline_in_frame=rev_tl_start,
                            timeline_out_frame=rev_tl_end - 1,
                            duration_frames=avail_frames,
                            track_type="audio",
                            audio_stream_index=primary_audio_stream_idx,
                            enabled=False,
                            action="remove",
                            decision_reason=span.reason,
                            confidence=span.confidence,
                        )
                    )

        markers.append(
            TimelineMarker(
                frame=min(max(0, edited_duration_frames - 1), cut_tl_frame),
                position_sec=frame_to_seconds(cut_tl_frame, fps_num, fps_den),
                comment=(
                    f"Cut #{r_idx + 1}: Removed {dur_sec:.2f}s ({span.reason}) "
                    f"[{format_timecode_hms(src_in_sec)} -> {format_timecode_hms(src_out_sec)}, "
                    f"conf={span.confidence:.2f}]"
                ),
                category=2 if span.reason == "obvious_stumble" else 0,
            )
        )

    if cfg.timeline_mode == "review-layers":
        tracks.append(
            TimelineTrack(
                track_id="V2_Review",
                track_name="V2 - Review Removed Cuts (Hidden/Disabled)",
                track_type="video",
                is_review_track=True,
                is_muted=True,
                is_hidden=True,
                audio_stream_index=None,
                clips=review_v_clips,
            )
        )
        if primary_audio_stream_idx is not None:
            tracks.append(
                TimelineTrack(
                    track_id="A_Review",
                    track_name="A_Review - Removed Audio Cuts (Muted/Disabled)",
                    track_type="audio",
                    is_review_track=True,
                    is_muted=True,
                    is_hidden=True,
                    audio_stream_index=primary_audio_stream_idx,
                    clips=review_a_clips,
                )
            )

    subtitles = (
        _map_subtitles_to_edited_timeline(report, kept_with_tl, fps_num, fps_den)
        if cfg.generate_subtitles
        else []
    )

    # Calculate overall edit confidence and review notes
    clip_confs = [c.confidence for c in v1_clips] + [c.confidence for c in removed_summary]
    mean_clip_conf = sum(clip_confs) / float(len(clip_confs)) if clip_confs else 0.75
    overall_conf = round(0.5 * report.overall_analysis_confidence + 0.5 * mean_clip_conf, 4)

    warnings = list(report.warnings)
    review_notes: List[str] = [
        f"Timeline mode: '{cfg.timeline_mode}'. Kept {len(v1_clips)} segment(s) ({edited_duration_frames} frames) "
        f"and isolated {len(removed_summary)} removed segment(s) from {total_frames} total source frames.",
        "Original intro preserved at start of timeline; all clips remain in strict chronological order.",
        "No automatic rendering was performed. Open the generated .kdenlive project in Kdenlive to review cuts and export manually.",
    ]

    if cfg.timeline_mode == "review-layers":
        review_notes.append(
            "Recoverability: Removed segments are placed on 'V2 - Review Removed Cuts' and 'A_Review' "
            "as disabled clips on hidden/muted tracks, with timeline guides at every cut point. "
            "To restore a cut in Kdenlive, jump to its guide, enable the clip (right-click -> Enable Clip), "
            "and drag it onto V1/A1."
        )

    low_confidence_halt = overall_conf < cfg.min_overall_edit_confidence
    if low_confidence_halt:
        warnings.append(
            f"Overall edit confidence ({overall_conf:.2f}) is below configured minimum "
            f"({cfg.min_overall_edit_confidence:.2f}). Stopping automatic timeline generation; "
            "human review of the analysis report and audio/video streams is required."
        )

    return EditDecisionModel(
        title=title,
        timeline_mode=cfg.timeline_mode,
        source_media=media,
        fps_num=fps_num,
        fps_den=fps_den,
        width=width,
        height=height,
        source_duration_frames=total_frames,
        edited_duration_frames=edited_duration_frames,
        tracks=tracks,
        markers=markers,
        subtitles=subtitles,
        removed_clips_summary=removed_summary,
        overall_confidence=overall_conf,
        requires_human_review=True,
        low_confidence_halt=low_confidence_halt,
        review_notes=review_notes,
        warnings=warnings,
    )

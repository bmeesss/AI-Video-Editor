"""Native Kdenlive Generation 5 (docproperties.version='1.1', MLT 7.23+) XML project generator."""

from __future__ import annotations

import hashlib
import json
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .models import EditDecisionModel, TimelineTrack
from .safety import capture_source_snapshot, safe_write_text, verify_source_untouched
from .subtitles import write_srt_sidecar


DEFAULT_GUIDES_CATEGORIES = [
    {"color": "#9b59b6", "comment": "Category 1", "index": 0},
    {"color": "#3daee9", "comment": "Category 2", "index": 1},
    {"color": "#1abc9c", "comment": "Category 3", "index": 2},
    {"color": "#1cdc9a", "comment": "Category 4", "index": 3},
    {"color": "#c9ce3b", "comment": "Category 5", "index": 4},
    {"color": "#fdbc4b", "comment": "Category 6", "index": 5},
    {"color": "#f39c1f", "comment": "Category 7", "index": 6},
    {"color": "#f47750", "comment": "Category 8", "index": 7},
    {"color": "#da4453", "comment": "Category 9", "index": 8},
]


def normalize_kdenlive_path(path: Path | str) -> str:
    """Normalize a filesystem path to forward-slash form expected by MLT/Kdenlive across Windows & Linux."""
    raw = str(path)
    return raw.replace("\\", "/")


def frame_to_mlt_timecode(frame: int, fps_num: int, fps_den: int) -> str:
    """Convert an integer frame index (or blank frame count) into MLT clock format HH:MM:SS.mmm."""
    f = max(0, int(frame))
    fps = float(fps_num) / float(max(1, fps_den))
    total_ms = int(round((float(f) / fps) * 1000.0))
    hours = total_ms // 3_600_000
    rem = total_ms % 3_600_000
    minutes = rem // 60_000
    rem = rem % 60_000
    seconds = rem // 1000
    millis = rem % 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def mlt_timecode_to_frame(tc: str, fps_num: int, fps_den: int) -> int:
    """Parse an MLT clock string HH:MM:SS.mmm back to the nearest integer frame."""
    parts = tc.strip().split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid MLT clock timecode '{tc}'")
    hours = int(parts[0])
    minutes = int(parts[1])
    sec_float = float(parts[2])
    total_sec = hours * 3600.0 + minutes * 60.0 + sec_float
    fps = float(fps_num) / float(max(1, fps_den))
    return int(round(total_sec * fps))


def _deterministic_uuid(seed_str: str) -> str:
    """Generate a deterministic UUIDv5 string from a seed so project generation is reproducible."""
    ns = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")
    return str(uuid.uuid5(ns, seed_str))


def _add_prop(parent: ET.Element, name: str, value: str = "") -> ET.Element:
    elem = ET.SubElement(parent, "property", {"name": name})
    if value != "":
        elem.text = str(value)
    return elem


def _order_tracks_for_kdenlive(tracks: List[TimelineTrack]) -> List[TimelineTrack]:
    """Order tracks with Audio tracks first (tractor0..A-1) and Video tracks second (tractorA..N-1).

    Within audio tracks: review audio first, then secondary audio (A2), then primary audio (A1).
    Within video tracks: primary video (V1) first, then review video (V2_Review) above it.
    """
    audio_tracks = [t for t in tracks if t.track_type == "audio"]
    video_tracks = [t for t in tracks if t.track_type == "video"]

    # Sort audio so review track is lowest index, followed by A2, then A1 closest to video
    audio_sorted = sorted(
        audio_tracks,
        key=lambda t: (0 if t.is_review_track else 1, -int(t.audio_stream_index or 0)),
    )
    # Sort video so V1 is first (lower video track) and V2_Review is above it
    video_sorted = sorted(
        video_tracks,
        key=lambda t: (1 if t.is_review_track else 0, t.track_id),
    )
    return audio_sorted + video_sorted


def _build_groups_json(ordered_tracks: List[TimelineTrack]) -> str:
    """Build Kdenlive 1.1 AVSplit / Normal group definitions linking kept video and audio clips."""
    v1_idx: Optional[int] = None
    a_indices: List[int] = []

    for idx, tr in enumerate(ordered_tracks):
        if tr.is_review_track:
            continue
        if tr.track_type == "video" and v1_idx is None:
            v1_idx = idx
        elif tr.track_type == "audio":
            a_indices.append(idx)

    if v1_idx is None or not a_indices:
        return "[]"

    v1_track = ordered_tracks[v1_idx]
    # Primary audio track is the last non-review audio track (A1)
    primary_a_idx = a_indices[-1]
    extra_a_indices = a_indices[:-1]

    groups: List[Dict[str, Any]] = []
    for clip in v1_track.clips:
        tl_frame = clip.timeline_in_frame
        av_split = {
            "type": "AVSplit",
            "children": [
                {
                    "data": f"{v1_idx}:{tl_frame}",
                    "leaf": "clip",
                    "type": "Leaf",
                },
                {
                    "data": f"{primary_a_idx}:{tl_frame}",
                    "leaf": "clip",
                    "type": "Leaf",
                },
            ],
        }
        if not extra_a_indices:
            groups.append(av_split)
        else:
            children: List[Dict[str, Any]] = [av_split]
            for extra_idx in extra_a_indices:
                children.append(
                    {
                        "data": f"{extra_idx}:{tl_frame}",
                        "leaf": "clip",
                        "type": "Leaf",
                    }
                )
            groups.append(
                {
                    "type": "Normal",
                    "children": children,
                }
            )
    return json.dumps(groups, indent=4)


def render_kdenlive_xml(
    edm: EditDecisionModel,
    project_dir: Path,
    subtitle_filename: Optional[str] = None,
) -> str:
    """Render a complete Kdenlive 1.1 (.kdenlive) XML document from an EditDecisionModel."""
    fps_num = edm.fps_num
    fps_den = edm.fps_den
    media = edm.source_media
    vstream = media.primary_video

    sar_num = vstream.sample_aspect_num if vstream else 1
    sar_den = vstream.sample_aspect_den if vstream else 1
    dar_num = vstream.display_aspect_num if vstream else 16
    dar_den = vstream.display_aspect_den if vstream else 9
    colorspace = vstream.colorspace if vstream else 709
    progressive = vstream.progressive if vstream else 1

    seed_base = f"{media.file_hash_sha256}:{edm.title}:{edm.timeline_mode}"
    seq_uuid = _deterministic_uuid(f"seq:{seed_base}")
    clip_uuid = _deterministic_uuid(f"clip:{seed_base}")
    doc_id = str(int(hashlib.sha256(seed_base.encode("utf-8")).hexdigest()[:12], 16))[:13]

    src_out_frame = max(0, edm.source_duration_frames - 1)
    src_out_tc = frame_to_mlt_timecode(src_out_frame, fps_num, fps_den)
    edited_out_frame = max(0, edm.edited_duration_frames - 1)
    edited_out_tc = frame_to_mlt_timecode(edited_out_frame, fps_num, fps_den)
    # Background black_track producer out timecode (Kdenlive adds 500s buffer or uses sequence length)
    bg_out_frames = edm.edited_duration_frames + int(round(500.0 * edm.fps))
    bg_out_tc = frame_to_mlt_timecode(bg_out_frames, fps_num, fps_den)

    resource_path = normalize_kdenlive_path(media.path)
    root_dir_str = normalize_kdenlive_path(project_dir.resolve())

    mlt_el = ET.Element(
        "mlt",
        {
            "LC_NUMERIC": "C",
            "producer": "main_bin",
            "version": "7.23.0",
            "root": root_dir_str,
        },
    )

    ET.SubElement(
        mlt_el,
        "profile",
        {
            "description": f"HD {edm.height}p {edm.fps:.2f} fps",
            "width": str(edm.width),
            "height": str(edm.height),
            "progressive": str(progressive),
            "sample_aspect_num": str(sar_num),
            "sample_aspect_den": str(sar_den),
            "display_aspect_num": str(dar_num),
            "display_aspect_den": str(dar_den),
            "frame_rate_num": str(fps_num),
            "frame_rate_den": str(fps_den),
            "colorspace": str(colorspace),
        },
    )

    # 1. Master Project Bin <chain id="chain_bin_0">
    chain_bin = ET.SubElement(
        mlt_el,
        "chain",
        {
            "id": "chain_bin_0",
            "in": "00:00:00.000",
            "out": src_out_tc,
        },
    )
    _add_prop(chain_bin, "length", str(edm.source_duration_frames))
    _add_prop(chain_bin, "eof", "pause")
    _add_prop(chain_bin, "resource", resource_path)
    _add_prop(chain_bin, "mlt_service", "avformat-novalidate")
    _add_prop(chain_bin, "kdenlive:clipname", media.filename)
    _add_prop(chain_bin, "kdenlive:folderid", "-1")
    _add_prop(chain_bin, "kdenlive:id", "4")
    _add_prop(chain_bin, "kdenlive:control_uuid", f"{{{clip_uuid}}}")
    _add_prop(chain_bin, "kdenlive:clip_type", "0")
    _add_prop(chain_bin, "kdenlive:file_size", str(media.file_size_bytes))
    _add_prop(chain_bin, "kdenlive:file_hash", media.file_hash_md5)

    if edm.removed_clips_summary:
        clipzones = [
            {
                "in": c.source_in_frame,
                "out": c.source_out_frame,
                "name": f"Removed #{i + 1} ({c.decision_reason})",
            }
            for i, c in enumerate(edm.removed_clips_summary)
        ]
        _add_prop(chain_bin, "kdenlive:clipzones", json.dumps(clipzones))

    ordered_tracks = _order_tracks_for_kdenlive(edm.tracks)
    audio_track_count = sum(1 for t in ordered_tracks if t.track_type == "audio")
    video_track_count = sum(1 for t in ordered_tracks if t.track_type == "video")

    # 2. <playlist id="main_bin"> (properties MUST precede entries per MLT DTD)
    main_bin = ET.SubElement(mlt_el, "playlist", {"id": "main_bin"})
    _add_prop(main_bin, "kdenlive:docproperties.activeTrack", str(audio_track_count))
    _add_prop(main_bin, "kdenlive:docproperties.audioChannels", "2")
    _add_prop(main_bin, "kdenlive:docproperties.disablepreview", "0")
    _add_prop(main_bin, "kdenlive:docproperties.documentid", doc_id)
    _add_prop(main_bin, "kdenlive:docproperties.enableTimelineZone", "0")
    _add_prop(main_bin, "kdenlive:docproperties.enableexternalproxy", "0")
    _add_prop(main_bin, "kdenlive:docproperties.enableproxy", "0")
    _add_prop(
        main_bin,
        "kdenlive:docproperties.guidesCategories",
        json.dumps(DEFAULT_GUIDES_CATEGORIES, indent=4),
    )
    _add_prop(main_bin, "kdenlive:docproperties.kdenliveversion", "24.08.3")
    _add_prop(main_bin, "kdenlive:docproperties.openTimeline", f"{{{seq_uuid}}}")
    _add_prop(main_bin, "kdenlive:docproperties.Activetimeline", f"{{{seq_uuid}}}")
    _add_prop(main_bin, "kdenlive:docproperties.uuid", f"{{{seq_uuid}}}")
    _add_prop(main_bin, "kdenlive:docproperties.version", "1.1")
    _add_prop(main_bin, "kdenlive:docproperties.verticalzoom", "1")
    _add_prop(main_bin, "kdenlive:docproperties.zoom", "8")
    _add_prop(main_bin, "kdenlive:expandedFolders")
    _add_prop(
        main_bin,
        "kdenlive:documentnotes",
        "\n".join(edm.review_notes),
    )
    _add_prop(main_bin, "xml_retain", "1")

    ET.SubElement(
        main_bin,
        "entry",
        {
            "producer": f"{{{seq_uuid}}}",
            "in": "00:00:00.000",
            "out": edited_out_tc,
        },
    )
    ET.SubElement(
        main_bin,
        "entry",
        {
            "producer": "chain_bin_0",
            "in": "00:00:00.000",
            "out": src_out_tc,
        },
    )

    # 3. Black background producer <producer id="producer0">
    prod_black = ET.SubElement(
        mlt_el,
        "producer",
        {
            "id": "producer0",
            "in": "00:00:00.000",
            "out": bg_out_tc,
        },
    )
    _add_prop(prod_black, "length", "2147483647")
    _add_prop(prod_black, "eof", "continue")
    _add_prop(prod_black, "resource", "black")
    _add_prop(prod_black, "aspect_ratio", "1")
    _add_prop(prod_black, "mlt_service", "color")
    _add_prop(prod_black, "kdenlive:playlistid", "black_track")
    _add_prop(prod_black, "mlt_image_format", "rgba")
    _add_prop(prod_black, "set.test_audio", "0")

    # 4. Per-track <chain>, <playlist>, and <tractor> elements
    default_audio_stream = (
        media.primary_mic_stream.stream_index if media.primary_mic_stream is not None else 1
    )
    filter_counter = 0

    for t_idx, track in enumerate(ordered_tracks):
        chain_id = f"chain_track_{t_idx}"
        pl_main_id = f"playlist{2 * t_idx}"
        pl_sub_id = f"playlist{2 * t_idx + 1}"
        tractor_id = f"tractor{t_idx}"
        a_stream_idx = (
            track.audio_stream_index
            if track.audio_stream_index is not None
            else default_audio_stream
        )

        # Emit dedicated per-track <chain> when the track contains clips
        if track.clips:
            t_chain = ET.SubElement(
                mlt_el,
                "chain",
                {
                    "id": chain_id,
                    "in": "00:00:00.000",
                    "out": src_out_tc,
                },
            )
            _add_prop(t_chain, "length", str(edm.source_duration_frames))
            _add_prop(t_chain, "eof", "pause")
            _add_prop(t_chain, "resource", resource_path)
            _add_prop(t_chain, "mlt_service", "avformat-novalidate")
            _add_prop(t_chain, "kdenlive:clipname", media.filename)
            _add_prop(t_chain, "kdenlive:folderid", "-1")
            _add_prop(t_chain, "kdenlive:id", "4")
            _add_prop(t_chain, "kdenlive:control_uuid", f"{{{clip_uuid}}}")
            _add_prop(t_chain, "kdenlive:clip_type", "0")
            _add_prop(t_chain, "kdenlive:file_size", str(media.file_size_bytes))
            _add_prop(t_chain, "kdenlive:file_hash", media.file_hash_md5)
            _add_prop(t_chain, "xml", "was here")

            if track.is_review_track or not all(c.enabled for c in track.clips):
                # Disabled clip state in Kdenlive: set.test_audio=1 and set.test_image=1
                _add_prop(t_chain, "set.test_audio", "1")
                _add_prop(t_chain, "set.test_image", "1")
                _add_prop(t_chain, "audio_index", "-1")
                if track.track_type == "audio":
                    _add_prop(t_chain, "video_index", "-1")
            elif track.track_type == "audio":
                _add_prop(t_chain, "set.test_audio", "0")
                _add_prop(t_chain, "set.test_image", "1")
                _add_prop(t_chain, "audio_index", str(a_stream_idx))
                _add_prop(t_chain, "video_index", "-1")
            else:
                _add_prop(t_chain, "set.test_audio", "1")
                _add_prop(t_chain, "set.test_image", "0")

        # Main track playlist
        pl_main = ET.SubElement(mlt_el, "playlist", {"id": pl_main_id})
        if track.track_type == "audio":
            _add_prop(pl_main, "kdenlive:audio_track", "1")
        if track.is_review_track or (track.is_hidden and track.is_muted):
            _add_prop(pl_main, "hide", "3")

        cursor_frame = 0
        for clip in track.clips:
            if clip.timeline_in_frame > cursor_frame:
                gap_frames = clip.timeline_in_frame - cursor_frame
                ET.SubElement(
                    pl_main,
                    "blank",
                    {"length": frame_to_mlt_timecode(gap_frames, fps_num, fps_den)},
                )
            in_tc = frame_to_mlt_timecode(clip.source_in_frame, fps_num, fps_den)
            out_tc = frame_to_mlt_timecode(clip.source_out_frame, fps_num, fps_den)
            entry_el = ET.SubElement(
                pl_main,
                "entry",
                {
                    "producer": chain_id,
                    "in": in_tc,
                    "out": out_tc,
                },
            )
            _add_prop(entry_el, "kdenlive:id", "4")
            if not clip.enabled or track.is_review_track:
                _add_prop(entry_el, "kdenlive:audio_index", str(a_stream_idx))
            cursor_frame = clip.timeline_out_frame + 1

        # Companion B-playlist required by Kdenlive track tractors
        pl_sub = ET.SubElement(mlt_el, "playlist", {"id": pl_sub_id})
        if track.track_type == "audio":
            _add_prop(pl_sub, "kdenlive:audio_track", "1")
        if track.is_review_track or (track.is_hidden and track.is_muted):
            _add_prop(pl_sub, "hide", "3")

        # Track <tractor>
        tractor_attrs: Dict[str, str] = {"id": tractor_id, "in": "00:00:00.000"}
        if track.clips:
            last_out_f = max(0, track.clips[-1].timeline_out_frame)
            tractor_attrs["out"] = frame_to_mlt_timecode(last_out_f, fps_num, fps_den)
        tr_el = ET.SubElement(mlt_el, "tractor", tractor_attrs)

        if track.track_type == "audio":
            _add_prop(tr_el, "kdenlive:audio_track", "1")
        _add_prop(tr_el, "kdenlive:track_name", track.track_name)
        _add_prop(tr_el, "kdenlive:trackheight", "67")
        _add_prop(tr_el, "kdenlive:timeline_active", "0" if track.is_review_track else "1")
        _add_prop(tr_el, "kdenlive:collapsed", "0")
        _add_prop(tr_el, "kdenlive:thumbs_format")
        _add_prop(tr_el, "kdenlive:audio_rec")
        if track.is_review_track or (track.is_hidden and track.is_muted):
            _add_prop(tr_el, "hide", "3")

        if track.is_review_track or (track.is_hidden and track.is_muted):
            hide_attr = "both"
        elif track.track_type == "audio":
            hide_attr = "video"
        else:
            hide_attr = "audio"

        ET.SubElement(tr_el, "track", {"hide": hide_attr, "producer": pl_main_id})
        ET.SubElement(tr_el, "track", {"hide": hide_attr, "producer": pl_sub_id})

        if track.track_type == "audio":
            f_vol = ET.SubElement(tr_el, "filter", {"id": f"filter{filter_counter}"})
            filter_counter += 1
            _add_prop(f_vol, "window", "75")
            _add_prop(f_vol, "max_gain", "20dB")
            _add_prop(f_vol, "mlt_service", "volume")
            _add_prop(f_vol, "internal_added", "237")
            _add_prop(f_vol, "disable", "1")

            f_pan = ET.SubElement(tr_el, "filter", {"id": f"filter{filter_counter}"})
            filter_counter += 1
            _add_prop(f_pan, "channel", "-1")
            _add_prop(f_pan, "mlt_service", "panner")
            _add_prop(f_pan, "internal_added", "237")
            _add_prop(f_pan, "start", "0.5")
            _add_prop(f_pan, "disable", "1")

            f_lvl = ET.SubElement(tr_el, "filter", {"id": f"filter{filter_counter}"})
            filter_counter += 1
            _add_prop(f_lvl, "iec_scale", "0")
            _add_prop(f_lvl, "mlt_service", "audiolevel")
            _add_prop(f_lvl, "peak", "1")
            _add_prop(f_lvl, "disable", "1")

    # 5. Sequence <tractor id="{seq_uuid}">
    # Note: In MLT DTD, <tractor> children MUST be ordered: <property>*, <track>+, (<filter>|<transition>)*
    seq_tractor = ET.SubElement(
        mlt_el,
        "tractor",
        {
            "id": f"{{{seq_uuid}}}",
            "in": "00:00:00.000",
            "out": edited_out_tc,
        },
    )
    _add_prop(seq_tractor, "kdenlive:uuid", f"{{{seq_uuid}}}")
    _add_prop(seq_tractor, "kdenlive:clipname", f"{edm.title} - Sequence")
    _add_prop(seq_tractor, "kdenlive:sequenceproperties.hasAudio", "1" if audio_track_count > 0 else "0")
    _add_prop(seq_tractor, "kdenlive:sequenceproperties.hasVideo", "1" if video_track_count > 0 else "0")
    _add_prop(
        seq_tractor,
        "kdenlive:sequenceproperties.tracksCount",
        str(len(ordered_tracks)),
    )
    _add_prop(seq_tractor, "kdenlive:sequenceproperties.activeTrack", str(audio_track_count))
    _add_prop(seq_tractor, "kdenlive:sequenceproperties.documentuuid", f"{{{seq_uuid}}}")
    _add_prop(seq_tractor, "kdenlive:duration", edited_out_tc)
    _add_prop(seq_tractor, "kdenlive:maxduration", str(edm.edited_duration_frames))
    _add_prop(seq_tractor, "kdenlive:producer_type", "17")
    _add_prop(seq_tractor, "kdenlive:id", "3")
    _add_prop(seq_tractor, "kdenlive:clip_type", "0")
    _add_prop(seq_tractor, "kdenlive:folderid", "-1")
    _add_prop(seq_tractor, "kdenlive:sequenceproperties.groups", _build_groups_json(ordered_tracks))

    guides_list = [
        {"pos": m.frame, "comment": m.comment, "type": m.category}
        for m in edm.markers
    ]
    _add_prop(seq_tractor, "kdenlive:sequenceproperties.guides", json.dumps(guides_list, indent=4))

    if subtitle_filename:
        subtitles_list = [{"file": subtitle_filename, "id": 0, "name": "Subtitles"}]
        _add_prop(
            seq_tractor,
            "kdenlive:sequenceproperties.subtitlesList",
            json.dumps(subtitles_list),
        )

    # Sequence <track> elements
    ET.SubElement(seq_tractor, "track", {"producer": "producer0"})
    for t_idx in range(len(ordered_tracks)):
        ET.SubElement(seq_tractor, "track", {"producer": f"tractor{t_idx}"})

    # Internal transitions (audio mix for audio tracks, qtblend for video tracks)
    trans_counter = 0
    for t_idx, track in enumerate(ordered_tracks):
        b_track_idx = t_idx + 1  # +1 because producer0 is track 0
        if track.track_type == "audio":
            tr_mix = ET.SubElement(seq_tractor, "transition", {"id": f"transition{trans_counter}"})
            trans_counter += 1
            _add_prop(tr_mix, "a_track", "0")
            _add_prop(tr_mix, "b_track", str(b_track_idx))
            _add_prop(tr_mix, "mlt_service", "mix")
            _add_prop(tr_mix, "kdenlive_id", "mix")
            _add_prop(tr_mix, "internal_added", "237")
            _add_prop(tr_mix, "always_active", "1")
            _add_prop(tr_mix, "accepts_blanks", "1")
            _add_prop(tr_mix, "sum", "1")
        else:
            tr_blend = ET.SubElement(seq_tractor, "transition", {"id": f"transition{trans_counter}"})
            trans_counter += 1
            _add_prop(tr_blend, "a_track", "0")
            _add_prop(tr_blend, "b_track", str(b_track_idx))
            _add_prop(tr_blend, "compositing", "0")
            _add_prop(tr_blend, "distort", "0")
            _add_prop(tr_blend, "rotate_center", "0")
            _add_prop(tr_blend, "mlt_service", "qtblend")
            _add_prop(tr_blend, "kdenlive_id", "qtblend")
            _add_prop(tr_blend, "internal_added", "237")
            _add_prop(tr_blend, "always_active", "1")

    # Sequence master filters (and optional subtitle filter)
    if subtitle_filename:
        f_sub = ET.SubElement(seq_tractor, "filter", {"id": f"filter{filter_counter}"})
        filter_counter += 1
        _add_prop(f_sub, "mlt_service", "avfilter.subtitles")
        _add_prop(f_sub, "av.filename", subtitle_filename)
        _add_prop(f_sub, "internal_added", "237")

    f_master_vol = ET.SubElement(seq_tractor, "filter", {"id": f"filter{filter_counter}"})
    filter_counter += 1
    _add_prop(f_master_vol, "window", "75")
    _add_prop(f_master_vol, "max_gain", "20dB")
    _add_prop(f_master_vol, "mlt_service", "volume")
    _add_prop(f_master_vol, "internal_added", "237")
    _add_prop(f_master_vol, "disable", "1")

    f_master_pan = ET.SubElement(seq_tractor, "filter", {"id": f"filter{filter_counter}"})
    filter_counter += 1
    _add_prop(f_master_pan, "channel", "-1")
    _add_prop(f_master_pan, "mlt_service", "panner")
    _add_prop(f_master_pan, "internal_added", "237")
    _add_prop(f_master_pan, "start", "0.5")
    _add_prop(f_master_pan, "disable", "1")

    # 6. Final outer projectTractor <tractor id="tractor{len(ordered_tracks)}">
    proj_tractor = ET.SubElement(
        mlt_el,
        "tractor",
        {
            "id": f"tractor{len(ordered_tracks)}",
            "in": "00:00:00.000",
            "out": edited_out_tc,
        },
    )
    _add_prop(proj_tractor, "kdenlive:projectTractor", "1")
    ET.SubElement(
        proj_tractor,
        "track",
        {
            "producer": f"{{{seq_uuid}}}",
            "in": "00:00:00.000",
            "out": edited_out_tc,
        },
    )

    ET.indent(mlt_el, space=" ")
    xml_body = ET.tostring(mlt_el, encoding="unicode")
    return "<?xml version='1.0' encoding='utf-8'?>\n" + xml_body + "\n"


def export_kdenlive_project(
    edm: EditDecisionModel,
    output_dir: Path | str,
    project_basename: str = "project",
) -> Tuple[Path, Optional[Path], Optional[Path]]:
    """Write .kdenlive project and .srt subtitle sidecars non-destructively into output_dir.

    Returns (kdenlive_path, kdenlive_srt_path, standalone_srt_path).
    Refuses to export if edm.low_confidence_halt is True.
    """
    if edm.low_confidence_halt:
        raise RuntimeError(
            "Refusing to generate .kdenlive project because overall edit confidence is below "
            "the configured threshold. Inspect the analysis report and warnings first."
        )

    snap = capture_source_snapshot(edm.source_media.path)
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    kdenlive_filename = (
        project_basename
        if project_basename.endswith(".kdenlive")
        else f"{project_basename}.kdenlive"
    )
    kdenlive_path = out_dir / kdenlive_filename

    kdenlive_srt_path: Optional[Path] = None
    standalone_srt_path: Optional[Path] = None
    subtitle_sidecar_name: Optional[str] = None

    if edm.subtitles:
        subtitle_sidecar_name = f"{kdenlive_filename}.srt"
        kdenlive_srt_path = write_srt_sidecar(
            out_dir / subtitle_sidecar_name,
            edm.subtitles,
            source_media_path=edm.source_media.path,
        )
        stem = kdenlive_filename[: -len(".kdenlive")]
        standalone_srt_path = write_srt_sidecar(
            out_dir / f"{stem}.srt",
            edm.subtitles,
            source_media_path=edm.source_media.path,
        )

    xml_content = render_kdenlive_xml(
        edm=edm,
        project_dir=out_dir,
        subtitle_filename=subtitle_sidecar_name,
    )
    safe_write_text(kdenlive_path, xml_content, source_path=edm.source_media.path)
    verify_source_untouched(snap)

    return kdenlive_path, kdenlive_srt_path, standalone_srt_path

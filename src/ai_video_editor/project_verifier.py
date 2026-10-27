"""Comprehensive MLT DTD + Kdenlive 1.1 (meltBuilder.cpp) semantic verifier for .kdenlive projects."""

from __future__ import annotations

import io
import json
import re
import uuid
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from lxml import etree

from .kdenlive_export import mlt_timecode_to_frame
from .safety import compute_file_hashes


# Corrected MLT 7.x + Kdenlive 1.1 DTD enforcing element ordering and attributes
KDENLIVE_MLT_DTD = """
<!ELEMENT mlt (profile?, (producer | chain | playlist | tractor | transition | filter | consumer)*)>
<!ATTLIST mlt
    LC_NUMERIC CDATA #IMPLIED
    version CDATA #IMPLIED
    root CDATA #IMPLIED
    profile CDATA #IMPLIED
    producer CDATA #IMPLIED
    title CDATA #IMPLIED
>
<!ELEMENT profile EMPTY>
<!ATTLIST profile
    name CDATA #IMPLIED
    description CDATA #IMPLIED
    width CDATA #REQUIRED
    height CDATA #REQUIRED
    progressive CDATA #REQUIRED
    sample_aspect_num CDATA #REQUIRED
    sample_aspect_den CDATA #REQUIRED
    display_aspect_num CDATA #REQUIRED
    display_aspect_den CDATA #REQUIRED
    frame_rate_num CDATA #REQUIRED
    frame_rate_den CDATA #REQUIRED
    colorspace CDATA #IMPLIED
>
<!ELEMENT property (#PCDATA)>
<!ATTLIST property name CDATA #REQUIRED>
<!ELEMENT producer (property | filter)*>
<!ATTLIST producer
    id CDATA #IMPLIED
    in CDATA #IMPLIED
    out CDATA #IMPLIED
    title CDATA #IMPLIED
>
<!ELEMENT chain (property | filter)*>
<!ATTLIST chain
    id CDATA #IMPLIED
    in CDATA #IMPLIED
    out CDATA #IMPLIED
    title CDATA #IMPLIED
>
<!ELEMENT playlist (property*, (entry | blank | producer | playlist | tractor)*)>
<!ATTLIST playlist
    id CDATA #IMPLIED
    in CDATA #IMPLIED
    out CDATA #IMPLIED
    title CDATA #IMPLIED
>
<!ELEMENT entry (property | filter)*>
<!ATTLIST entry
    producer CDATA #REQUIRED
    in CDATA #IMPLIED
    out CDATA #IMPLIED
>
<!ELEMENT blank EMPTY>
<!ATTLIST blank length CDATA #REQUIRED>
<!ELEMENT tractor (property*, track+, (filter | transition)*)>
<!ATTLIST tractor
    id CDATA #IMPLIED
    in CDATA #IMPLIED
    out CDATA #IMPLIED
    title CDATA #IMPLIED
>
<!ELEMENT track (property | filter | transition)*>
<!ATTLIST track
    producer CDATA #REQUIRED
    hide (video | audio | both | none) #IMPLIED
    in CDATA #IMPLIED
    out CDATA #IMPLIED
>
<!ELEMENT transition (property)*>
<!ATTLIST transition
    id CDATA #IMPLIED
    in CDATA #IMPLIED
    out CDATA #IMPLIED
    mlt_service CDATA #IMPLIED
    a_track CDATA #IMPLIED
    b_track CDATA #IMPLIED
>
<!ELEMENT filter (property)*>
<!ATTLIST filter
    id CDATA #IMPLIED
    in CDATA #IMPLIED
    out CDATA #IMPLIED
    mlt_service CDATA #IMPLIED
    track CDATA #IMPLIED
>
"""


@dataclass
class ProjectVerificationReport:
    """Detailed verification result for a generated .kdenlive project and its sidecars."""

    project_path: str
    is_valid: bool
    dtd_valid: bool
    kdenlive_semantics_valid: bool
    source_media_verified: bool
    subtitles_verified: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    checks_passed: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


def _get_props(elem: ET.Element) -> Dict[str, str]:
    return {
        p.attrib["name"]: (p.text or "")
        for p in elem.findall("./property")
        if "name" in p.attrib
    }


def _is_braced_uuid(val: str) -> bool:
    if not (val.startswith("{") and val.endswith("}")):
        return False
    try:
        uuid.UUID(val[1:-1])
        return True
    except Exception:
        return False


def verify_srt_file(srt_path: Path) -> Tuple[bool, List[str], int]:
    """Validate SubRip (.srt) syntax and monotonic timestamps."""
    errors: List[str] = []
    if not srt_path.exists():
        return False, [f"Subtitle file does not exist: {srt_path}"], 0

    text = srt_path.read_text(encoding="utf-8").strip()
    if not text:
        return True, [], 0

    blocks = re.split(r"\n\s*\n", text)
    ts_pattern = re.compile(
        r"^(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})$"
    )
    prev_start_ms = -1
    prev_end_ms = -1

    for idx, block in enumerate(blocks, start=1):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 3:
            errors.append(f"SRT block #{idx} has fewer than 3 lines.")
            continue
        if lines[0] != str(idx):
            errors.append(f"SRT block #{idx} index line is '{lines[0]}', expected '{idx}'.")
        m = ts_pattern.match(lines[1])
        if not m:
            errors.append(f"SRT block #{idx} has invalid timestamp line: '{lines[1]}'.")
            continue
        vals = [int(x) for x in m.groups()]
        start_ms = ((vals[0] * 3600 + vals[1] * 60 + vals[2]) * 1000) + vals[3]
        end_ms = ((vals[4] * 3600 + vals[5] * 60 + vals[6]) * 1000) + vals[7]
        if end_ms <= start_ms:
            errors.append(f"SRT block #{idx} end timestamp <= start timestamp ({lines[1]}).")
        if start_ms < prev_start_ms:
            errors.append(f"SRT block #{idx} is not in chronological order.")
        if start_ms < prev_end_ms:
            errors.append(
                f"SRT block #{idx} overlaps previous subtitle cue (start={start_ms}ms < prev_end={prev_end_ms}ms)."
            )
        prev_start_ms = start_ms
        prev_end_ms = end_ms

    return len(errors) == 0, errors, len(blocks)


def verify_kdenlive_project(
    project_path: Path | str,
    check_source_media_exists: bool = True,
) -> ProjectVerificationReport:
    """Validate a .kdenlive file against both the MLT DTD and Kdenlive 1.1 C++ loader invariants."""
    p = Path(project_path).expanduser().resolve()
    errors: List[str] = []
    warnings: List[str] = []
    checks_passed: List[str] = []
    stats: Dict[str, Any] = {}

    if not p.exists():
        return ProjectVerificationReport(
            project_path=str(p),
            is_valid=False,
            dtd_valid=False,
            kdenlive_semantics_valid=False,
            source_media_verified=False,
            subtitles_verified=False,
            errors=[f"Project file does not exist: {p}"],
        )

    raw_xml = p.read_text(encoding="utf-8")

    # 1. MLT XML DTD validation (verifies element hierarchy, child ordering, and attributes)
    dtd_valid = False
    try:
        lxml_doc = etree.fromstring(raw_xml.encode("utf-8"))
        dtd = etree.DTD(io.StringIO(KDENLIVE_MLT_DTD))
        if dtd.validate(lxml_doc):
            dtd_valid = True
            checks_passed.append("MLT XML DTD element ordering and attribute structure passed.")
        else:
            for err in dtd.error_log.filter_from_errors():
                errors.append(f"MLT DTD error (line {err.line}): {err.message}")
    except Exception as exc:
        errors.append(f"XML parse error: {exc}")
        return ProjectVerificationReport(
            project_path=str(p),
            is_valid=False,
            dtd_valid=False,
            kdenlive_semantics_valid=False,
            source_media_verified=False,
            subtitles_verified=False,
            errors=errors,
        )

    # 2. Kdenlive Generation 5 (docproperties.version='1.1') C++ semantic checks
    root = ET.fromstring(raw_xml)
    profile = root.find("./profile")
    if profile is None:
        errors.append("Missing <profile> element in <mlt>.")
        fps_num, fps_den = 30, 1
    else:
        fps_num = int(profile.attrib.get("frame_rate_num", "30"))
        fps_den = int(profile.attrib.get("frame_rate_den", "1"))
        checks_passed.append(f"Profile verified ({profile.attrib.get('width')}x{profile.attrib.get('height')} @ {fps_num}/{fps_den} fps).")

    # Index all chains, producers, playlists, and tractors by id
    chains: Dict[str, ET.Element] = {
        c.attrib["id"]: c for c in root.findall("./chain") if "id" in c.attrib
    }
    playlists: Dict[str, ET.Element] = {
        pl.attrib["id"]: pl for pl in root.findall("./playlist") if "id" in pl.attrib
    }
    tractors: Dict[str, ET.Element] = {
        tr.attrib["id"]: tr for tr in root.findall("./tractor") if "id" in tr.attrib
    }

    main_bin = playlists.get("main_bin")
    if main_bin is None:
        errors.append("Missing <playlist id='main_bin'>.")
        return ProjectVerificationReport(
            project_path=str(p),
            is_valid=False,
            dtd_valid=dtd_valid,
            kdenlive_semantics_valid=False,
            source_media_verified=False,
            subtitles_verified=False,
            errors=errors,
        )

    bin_props = _get_props(main_bin)
    doc_ver = bin_props.get("kdenlive:docproperties.version", "")
    if doc_ver != "1.1":
        errors.append(f"Expected kdenlive:docproperties.version='1.1', got '{doc_ver}'.")
    else:
        checks_passed.append("kdenlive:docproperties.version='1.1' (Generation 5) verified.")

    seq_uuid = bin_props.get("kdenlive:docproperties.uuid", "")
    if not _is_braced_uuid(seq_uuid):
        errors.append(f"Invalid or missing kdenlive:docproperties.uuid '{seq_uuid}'.")

    # Verify bin entries and collect bin control_uuid mapping (ProjectItemModel::loadBinPlaylist)
    bin_entries = main_bin.findall("./entry")
    bin_control_uuids: Dict[str, str] = {}  # control_uuid -> kdenlive:id
    source_media_verified = True

    for entry in bin_entries:
        prod_id = entry.attrib.get("producer", "")
        if prod_id in chains:
            c_props = _get_props(chains[prod_id])
            kid = c_props.get("kdenlive:id", "")
            cuuid = c_props.get("kdenlive:control_uuid", "")
            resource = c_props.get("resource", "")
            if not kid or not _is_braced_uuid(cuuid):
                errors.append(
                    f"Bin chain '{prod_id}' is missing valid kdenlive:id ('{kid}') or kdenlive:control_uuid ('{cuuid}')."
                )
            else:
                bin_control_uuids[cuuid] = kid

            if check_source_media_exists and resource:
                res_path = Path(resource)
                if not res_path.exists():
                    source_media_verified = False
                    errors.append(f"Source media file referenced by '{prod_id}' does not exist: {res_path}")
                else:
                    md5_hex, _ = compute_file_hashes(res_path)
                    expected_md5 = c_props.get("kdenlive:file_hash", "")
                    if expected_md5 and md5_hex != expected_md5:
                        source_media_verified = False
                        errors.append(
                            f"Source media MD5 mismatch for '{res_path}': project expects {expected_md5}, actual {md5_hex}."
                        )
                    else:
                        checks_passed.append(f"Source media file '{res_path.name}' exists and matches MD5 hash.")

    # Verify outer projectTractor wraps sequence tractor
    all_tractor_list = root.findall("./tractor")
    if not all_tractor_list:
        errors.append("No <tractor> elements found.")
    else:
        last_tractor = all_tractor_list[-1]
        lt_props = _get_props(last_tractor)
        if lt_props.get("kdenlive:projectTractor") != "1":
            errors.append("Final <tractor> is missing <property name='kdenlive:projectTractor'>1</property>.")
        else:
            checks_passed.append("Outer projectTractor verified.")

    seq_tractor = tractors.get(seq_uuid)
    if seq_tractor is None:
        errors.append(f"Sequence tractor '{seq_uuid}' not found.")
        return ProjectVerificationReport(
            project_path=str(p),
            is_valid=False,
            dtd_valid=dtd_valid,
            kdenlive_semantics_valid=False,
            source_media_verified=source_media_verified,
            subtitles_verified=False,
            errors=errors,
        )

    seq_props = _get_props(seq_tractor)
    seq_out_tc = seq_tractor.attrib.get("out", "00:00:00.000")
    seq_duration_frames = mlt_timecode_to_frame(seq_out_tc, fps_num, fps_den) + 1
    stats["sequence_duration_frames"] = seq_duration_frames

    # Verify timeline tracks inside sequence tractor
    seq_tracks = seq_tractor.findall("./track")
    if len(seq_tracks) < 2:
        errors.append("Sequence tractor must contain background track plus at least one timeline track.")

    timeline_tractor_ids = [t.attrib["producer"] for t in seq_tracks[1:] if "producer" in t.attrib]
    used_track_chains: Set[str] = set()
    track_clip_starts: Dict[int, Set[int]] = {}
    active_track_timelines: Dict[str, List[Tuple[int, int, int, int]]] = {}

    for t_pos, tr_id in enumerate(timeline_tractor_ids):
        track_clip_starts[t_pos] = set()
        tr_el = tractors.get(tr_id)
        if tr_el is None:
            errors.append(f"Sequence track references missing tractor '{tr_id}'.")
            continue
        tr_props = _get_props(tr_el)
        is_audio_track = tr_props.get("kdenlive:audio_track") == "1"
        is_review_hidden = tr_props.get("hide") == "3"
        sub_tracks = tr_el.findall("./track")
        if len(sub_tracks) != 2:
            errors.append(f"Track tractor '{tr_id}' must wrap exactly 2 sub-playlists, found {len(sub_tracks)}.")
            continue

        pl_main_id = sub_tracks[0].attrib.get("producer", "")
        pl_main = playlists.get(pl_main_id)
        if pl_main is None:
            errors.append(f"Track tractor '{tr_id}' references missing playlist '{pl_main_id}'.")
            continue

        pl_props = _get_props(pl_main)
        if is_review_hidden:
            if sub_tracks[0].attrib.get("hide") != "both" or pl_props.get("hide") != "3":
                errors.append(
                    f"Review track '{tr_id}' must set both hide='both' on <track> and <property name='hide'>3</property> on <playlist>."
                )

        # Walk playlist entries & blanks to verify frame accuracy, chain separation, and chronological order
        cursor_f = 0
        prev_src_out = -1
        clip_tuples: List[Tuple[int, int, int, int]] = []

        for child in pl_main:
            if child.tag == "blank":
                gap_f = mlt_timecode_to_frame(child.attrib["length"], fps_num, fps_den)
                if not is_review_hidden:
                    errors.append(
                        f"Active non-review track playlist '{pl_main_id}' contains a <blank> gap of {gap_f} frame(s); "
                        "active tracks must be ripple-packed without gaps."
                    )
                cursor_f += gap_f
            elif child.tag == "entry":
                c_id = child.attrib.get("producer", "")
                if c_id not in chains:
                    errors.append(f"Playlist '{pl_main_id}' entry references unknown chain '{c_id}'.")
                    continue
                c_el = chains[c_id]
                c_props = _get_props(c_el)
                cuuid = c_props.get("kdenlive:control_uuid", "")
                kid = c_props.get("kdenlive:id", "")
                if cuuid not in bin_control_uuids or bin_control_uuids[cuuid] != kid:
                    errors.append(
                        f"Timeline chain '{c_id}' control_uuid '{cuuid}' / id '{kid}' does not match any bin clip."
                    )

                in_f = mlt_timecode_to_frame(child.attrib.get("in", "00:00:00.000"), fps_num, fps_den)
                out_f = mlt_timecode_to_frame(child.attrib.get("out", "00:00:00.000"), fps_num, fps_den)
                max_len = int(c_props.get("length", "0"))
                if in_f < 0 or out_f < in_f or (max_len > 0 and out_f >= max_len):
                    errors.append(
                        f"Invalid clip bounds on '{pl_main_id}': in={in_f}, out={out_f}, source_length={max_len}."
                    )
                if in_f <= prev_src_out:
                    errors.append(
                        f"Non-chronological or overlapping source clip on '{pl_main_id}': in_frame={in_f} <= prev_out_frame={prev_src_out}."
                    )
                prev_src_out = out_f

                if is_review_hidden:
                    if c_props.get("set.test_audio") != "1" or c_props.get("set.test_image") != "1":
                        errors.append(
                            f"Review track '{tr_id}' chain '{c_id}' must be in Disabled state (set.test_audio=1, set.test_image=1)."
                        )
                    entry_props = _get_props(child)
                    if "kdenlive:audio_index" not in entry_props:
                        errors.append(
                            f"Disabled review entry on '{pl_main_id}' is missing kdenlive:audio_index property."
                        )

                dur_f = out_f - in_f + 1
                track_clip_starts[t_pos].add(cursor_f)
                clip_tuples.append((cursor_f, cursor_f + dur_f - 1, in_f, out_f))
                cursor_f += dur_f

        # Ensure chain is not shared across different tracks (ProjectClip::giveMasterAndGetTimelineProducer)
        track_chains_here = {e.attrib.get("producer", "") for e in pl_main.findall("./entry")}
        overlap_chains = used_track_chains.intersection(track_chains_here)
        if overlap_chains:
            errors.append(f"Chain(s) {overlap_chains} reused across multiple tracks; Kdenlive requires per-track chains.")
        used_track_chains.update(track_chains_here)

        if is_review_hidden and cursor_f > seq_duration_frames:
            errors.append(
                f"Review track '{tr_id}' duration ({cursor_f} frames) exceeds sequence duration ({seq_duration_frames} frames)."
            )

        if not is_review_hidden and clip_tuples:
            active_track_timelines[tr_id] = clip_tuples

    # Verify A/V synchronization across all active non-review tracks
    if len(active_track_timelines) >= 2:
        ref_id, ref_clips = next(iter(active_track_timelines.items()))
        for other_id, other_clips in active_track_timelines.items():
            if other_clips != ref_clips:
                errors.append(
                    f"A/V frame alignment mismatch between active tracks '{ref_id}' and '{other_id}'."
                )
        if not errors:
            checks_passed.append(
                f"Frame-exact A/V synchronization verified across {len(active_track_timelines)} active tracks."
            )

    # Verify groups JSON (GroupsModel::fromJson)
    groups_raw = seq_props.get("kdenlive:sequenceproperties.groups", "[]")
    try:
        groups_data = json.loads(groups_raw)
        if not isinstance(groups_data, list):
            errors.append("kdenlive:sequenceproperties.groups must be a JSON list.")
        else:
            def _check_group_node(node: Dict[str, Any]) -> None:
                gtype = node.get("type")
                children = node.get("children", [])
                if gtype == "AVSplit" and len(children) != 2:
                    errors.append(f"AVSplit group must have exactly 2 children, found {len(children)}.")
                for ch in children:
                    if ch.get("type") == "Leaf":
                        data_str = str(ch.get("data", ""))
                        t_str, f_str = data_str.split(":")
                        t_idx, f_pos = int(t_str), int(f_str)
                        if f_pos not in track_clip_starts.get(t_idx, set()):
                            errors.append(
                                f"Group leaf '{data_str}' does not match any clip start frame on track {t_idx}."
                            )
                    else:
                        _check_group_node(ch)

            for g in groups_data:
                _check_group_node(g)
            checks_passed.append(f"Verified {len(groups_data)} clip link group(s) in kdenlive:sequenceproperties.groups.")
    except Exception as exc:
        errors.append(f"Invalid kdenlive:sequenceproperties.groups JSON: {exc}")

    # Verify guides JSON
    guides_raw = seq_props.get("kdenlive:sequenceproperties.guides", "[]")
    try:
        guides_data = json.loads(guides_raw)
        if not isinstance(guides_data, list):
            errors.append("kdenlive:sequenceproperties.guides must be a JSON list.")
        else:
            for g in guides_data:
                if "pos" not in g or "comment" not in g or "type" not in g:
                    errors.append(f"Guide entry missing required keys (pos, comment, type): {g}")
            checks_passed.append(f"Verified {len(guides_data)} timeline guide(s)/marker(s).")
    except Exception as exc:
        errors.append(f"Invalid kdenlive:sequenceproperties.guides JSON: {exc}")

    # Verify subtitles sidecar if referenced
    subtitles_verified = True
    subs_raw = seq_props.get("kdenlive:sequenceproperties.subtitlesList", "")
    if subs_raw:
        try:
            subs_list = json.loads(subs_raw)
            for sub_item in subs_list:
                sub_file = p.parent / str(sub_item.get("file", ""))
                ok, sub_errs, sub_count = verify_srt_file(sub_file)
                if not ok:
                    subtitles_verified = False
                    errors.extend(sub_errs)
                else:
                    checks_passed.append(f"Verified subtitle sidecar '{sub_file.name}' ({sub_count} cue(s)).")
        except Exception as exc:
            subtitles_verified = False
            errors.append(f"Invalid subtitlesList metadata: {exc}")

    semantics_valid = len(errors) == 0
    is_valid = dtd_valid and semantics_valid and source_media_verified and subtitles_verified
    stats["tracks_count"] = len(timeline_tractor_ids)
    stats["active_tracks_count"] = len(active_track_timelines)

    return ProjectVerificationReport(
        project_path=str(p),
        is_valid=is_valid,
        dtd_valid=dtd_valid,
        kdenlive_semantics_valid=semantics_valid,
        source_media_verified=source_media_verified,
        subtitles_verified=subtitles_verified,
        errors=errors,
        warnings=warnings,
        checks_passed=checks_passed,
        stats=stats,
    )

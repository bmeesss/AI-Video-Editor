"""Typed data models for media metadata, multi-modal analysis, and Kdenlive editing decisions."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional


TimelineMode = Literal["review-layers", "ripple"]
WhisperModelChoice = Literal["auto", "base", "small", "base.en", "small.en"]
AudioRole = Literal["mixed", "microphone", "gameplay", "secondary", "unknown"]
SegmentAction = Literal["keep", "remove", "review"]


@dataclass(frozen=True)
class VideoStreamInfo:
    stream_index: int
    codec_name: str
    width: int
    height: int
    frame_rate_num: int
    frame_rate_den: int
    fps: float
    duration_seconds: float
    total_frames: int
    pix_fmt: str = "yuv420p"
    colorspace: int = 709
    progressive: int = 1
    sample_aspect_num: int = 1
    sample_aspect_den: int = 1
    display_aspect_num: int = 16
    display_aspect_den: int = 9
    is_vfr: bool = False
    vfr_jitter_ratio: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VideoStreamInfo":
        return cls(**data)


@dataclass(frozen=True)
class AudioStreamInfo:
    stream_index: int
    audio_track_order: int
    codec_name: str
    sample_rate: int
    channels: int
    duration_seconds: float
    title: str = ""
    language: str = ""
    inferred_role: AudioRole = "unknown"
    mean_dbfs: float = -40.0
    peak_dbfs: float = -20.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AudioStreamInfo":
        return cls(**data)


@dataclass(frozen=True)
class MediaInfo:
    path: str
    filename: str
    container_format: str
    duration_seconds: float
    file_size_bytes: int
    file_hash_md5: str
    file_hash_sha256: str
    video_streams: List[VideoStreamInfo] = field(default_factory=list)
    audio_streams: List[AudioStreamInfo] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def primary_video(self) -> Optional[VideoStreamInfo]:
        return self.video_streams[0] if self.video_streams else None

    @property
    def primary_mic_stream(self) -> Optional[AudioStreamInfo]:
        for s in self.audio_streams:
            if s.inferred_role == "microphone":
                return s
        return self.audio_streams[0] if self.audio_streams else None

    @property
    def primary_game_stream(self) -> Optional[AudioStreamInfo]:
        for s in self.audio_streams:
            if s.inferred_role == "gameplay":
                return s
        if len(self.audio_streams) > 1:
            return self.audio_streams[1]
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "filename": self.filename,
            "container_format": self.container_format,
            "duration_seconds": self.duration_seconds,
            "file_size_bytes": self.file_size_bytes,
            "file_hash_md5": self.file_hash_md5,
            "file_hash_sha256": self.file_hash_sha256,
            "video_streams": [v.to_dict() for v in self.video_streams],
            "audio_streams": [a.to_dict() for a in self.audio_streams],
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MediaInfo":
        return cls(
            path=data["path"],
            filename=data["filename"],
            container_format=data["container_format"],
            duration_seconds=float(data["duration_seconds"]),
            file_size_bytes=int(data["file_size_bytes"]),
            file_hash_md5=data["file_hash_md5"],
            file_hash_sha256=data["file_hash_sha256"],
            video_streams=[VideoStreamInfo.from_dict(v) for v in data.get("video_streams", [])],
            audio_streams=[AudioStreamInfo.from_dict(a) for a in data.get("audio_streams", [])],
            warnings=list(data.get("warnings", [])),
        )


@dataclass(frozen=True)
class SpeechInterval:
    start: float
    end: float
    confidence: float
    stream_index: int = 0
    source: str = "silero-vad"

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpeechInterval":
        return cls(**data)


@dataclass(frozen=True)
class TranscriptWord:
    word: str
    start: float
    end: float
    probability: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TranscriptWord":
        return cls(**data)


@dataclass(frozen=True)
class TranscriptSegment:
    segment_id: int
    start: float
    end: float
    text: str
    confidence: float
    words: List[TranscriptWord] = field(default_factory=list)
    is_stumble_candidate: bool = False
    stumble_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "confidence": self.confidence,
            "words": [w.to_dict() for w in self.words],
            "is_stumble_candidate": self.is_stumble_candidate,
            "stumble_reason": self.stumble_reason,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TranscriptSegment":
        return cls(
            segment_id=int(data["segment_id"]),
            start=float(data["start"]),
            end=float(data["end"]),
            text=data["text"],
            confidence=float(data["confidence"]),
            words=[TranscriptWord.from_dict(w) for w in data.get("words", [])],
            is_stumble_candidate=bool(data.get("is_stumble_candidate", False)),
            stumble_reason=str(data.get("stumble_reason", "")),
        )


@dataclass(frozen=True)
class AudioEnergyWindow:
    start: float
    end: float
    rms_dbfs: float
    peak_dbfs: float
    is_quiet: bool
    is_transient_peak: bool
    stream_index: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AudioEnergyWindow":
        return cls(**data)


@dataclass(frozen=True)
class SceneBoundary:
    scene_index: int
    start: float
    end: float
    start_frame: int
    end_frame: int
    detector: str = "AdaptiveDetector"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SceneBoundary":
        return cls(**data)


@dataclass(frozen=True)
class MotionWindow:
    start: float
    end: float
    motion_score: float
    is_active_gameplay: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MotionWindow":
        return cls(**data)


@dataclass(frozen=True)
class AnalysisReport:
    schema_version: str
    created_at_utc: str
    media_info: MediaInfo
    whisper_model_used: str
    whisper_device_used: str
    transcription_status: str
    speech_intervals: List[SpeechInterval] = field(default_factory=list)
    transcript_segments: List[TranscriptSegment] = field(default_factory=list)
    audio_energy_windows: List[AudioEnergyWindow] = field(default_factory=list)
    scene_boundaries: List[SceneBoundary] = field(default_factory=list)
    motion_windows: List[MotionWindow] = field(default_factory=list)
    overall_analysis_confidence: float = 1.0
    warnings: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "created_at_utc": self.created_at_utc,
            "media_info": self.media_info.to_dict(),
            "whisper_model_used": self.whisper_model_used,
            "whisper_device_used": self.whisper_device_used,
            "transcription_status": self.transcription_status,
            "speech_intervals": [s.to_dict() for s in self.speech_intervals],
            "transcript_segments": [t.to_dict() for t in self.transcript_segments],
            "audio_energy_windows": [a.to_dict() for a in self.audio_energy_windows],
            "scene_boundaries": [b.to_dict() for b in self.scene_boundaries],
            "motion_windows": [m.to_dict() for m in self.motion_windows],
            "overall_analysis_confidence": self.overall_analysis_confidence,
            "warnings": list(self.warnings),
            "notes": list(self.notes),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def save_json(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(indent=2), encoding="utf-8", newline="\n")
        return path

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnalysisReport":
        return cls(
            schema_version=str(data.get("schema_version", "1.0")),
            created_at_utc=str(data.get("created_at_utc", "")),
            media_info=MediaInfo.from_dict(data["media_info"]),
            whisper_model_used=str(data.get("whisper_model_used", "none")),
            whisper_device_used=str(data.get("whisper_device_used", "cpu")),
            transcription_status=str(data.get("transcription_status", "unknown")),
            speech_intervals=[SpeechInterval.from_dict(x) for x in data.get("speech_intervals", [])],
            transcript_segments=[TranscriptSegment.from_dict(x) for x in data.get("transcript_segments", [])],
            audio_energy_windows=[AudioEnergyWindow.from_dict(x) for x in data.get("audio_energy_windows", [])],
            scene_boundaries=[SceneBoundary.from_dict(x) for x in data.get("scene_boundaries", [])],
            motion_windows=[MotionWindow.from_dict(x) for x in data.get("motion_windows", [])],
            overall_analysis_confidence=float(data.get("overall_analysis_confidence", 1.0)),
            warnings=list(data.get("warnings", [])),
            notes=list(data.get("notes", [])),
        )

    @classmethod
    def load_json(cls, path: Path) -> "AnalysisReport":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


@dataclass(frozen=True)
class TimelineClip:
    clip_id: str
    source_path: str
    source_in_sec: float
    source_out_sec: float
    source_in_frame: int
    source_out_frame: int
    timeline_start_sec: float
    timeline_end_sec: float
    timeline_in_frame: int
    timeline_out_frame: int
    duration_frames: int
    track_type: Literal["video", "audio"] = "video"
    audio_stream_index: Optional[int] = None
    enabled: bool = True
    action: SegmentAction = "keep"
    decision_reason: str = ""
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TimelineClip":
        return cls(**data)


@dataclass(frozen=True)
class TimelineTrack:
    track_id: str
    track_name: str
    track_type: Literal["video", "audio"]
    is_review_track: bool
    is_hidden: bool
    is_muted: bool
    is_locked: bool = False
    audio_stream_index: Optional[int] = None
    clips: List[TimelineClip] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "track_name": self.track_name,
            "track_type": self.track_type,
            "is_review_track": self.is_review_track,
            "is_hidden": self.is_hidden,
            "is_muted": self.is_muted,
            "is_locked": self.is_locked,
            "audio_stream_index": self.audio_stream_index,
            "clips": [c.to_dict() for c in self.clips],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TimelineTrack":
        return cls(
            track_id=data["track_id"],
            track_name=data["track_name"],
            track_type=data["track_type"],
            is_review_track=bool(data["is_review_track"]),
            is_hidden=bool(data["is_hidden"]),
            is_muted=bool(data["is_muted"]),
            is_locked=bool(data.get("is_locked", False)),
            audio_stream_index=data.get("audio_stream_index"),
            clips=[TimelineClip.from_dict(c) for c in data.get("clips", [])],
        )


@dataclass(frozen=True)
class TimelineMarker:
    frame: int
    position_sec: float
    comment: str
    category: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TimelineMarker":
        return cls(**data)


@dataclass(frozen=True)
class SubtitleEntry:
    index: int
    start_sec: float
    end_sec: float
    text: str
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SubtitleEntry":
        return cls(**data)


@dataclass(frozen=True)
class EditDecisionModel:
    title: str
    timeline_mode: TimelineMode
    source_media: MediaInfo
    fps_num: int
    fps_den: int
    width: int
    height: int
    source_duration_frames: int
    edited_duration_frames: int
    tracks: List[TimelineTrack] = field(default_factory=list)
    markers: List[TimelineMarker] = field(default_factory=list)
    subtitles: List[SubtitleEntry] = field(default_factory=list)
    removed_clips_summary: List[TimelineClip] = field(default_factory=list)
    overall_confidence: float = 1.0
    requires_human_review: bool = True
    low_confidence_halt: bool = False
    review_notes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def fps(self) -> float:
        return float(self.fps_num) / float(max(1, self.fps_den))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "timeline_mode": self.timeline_mode,
            "source_media": self.source_media.to_dict(),
            "fps_num": self.fps_num,
            "fps_den": self.fps_den,
            "width": self.width,
            "height": self.height,
            "source_duration_frames": self.source_duration_frames,
            "edited_duration_frames": self.edited_duration_frames,
            "tracks": [t.to_dict() for t in self.tracks],
            "markers": [m.to_dict() for m in self.markers],
            "subtitles": [s.to_dict() for s in self.subtitles],
            "removed_clips_summary": [c.to_dict() for c in self.removed_clips_summary],
            "overall_confidence": self.overall_confidence,
            "requires_human_review": self.requires_human_review,
            "low_confidence_halt": self.low_confidence_halt,
            "review_notes": list(self.review_notes),
            "warnings": list(self.warnings),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def save_json(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(indent=2), encoding="utf-8", newline="\n")
        return path

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EditDecisionModel":
        return cls(
            title=str(data["title"]),
            timeline_mode=data["timeline_mode"],
            source_media=MediaInfo.from_dict(data["source_media"]),
            fps_num=int(data["fps_num"]),
            fps_den=int(data["fps_den"]),
            width=int(data["width"]),
            height=int(data["height"]),
            source_duration_frames=int(data["source_duration_frames"]),
            edited_duration_frames=int(data["edited_duration_frames"]),
            tracks=[TimelineTrack.from_dict(t) for t in data.get("tracks", [])],
            markers=[TimelineMarker.from_dict(m) for m in data.get("markers", [])],
            subtitles=[SubtitleEntry.from_dict(s) for s in data.get("subtitles", [])],
            removed_clips_summary=[TimelineClip.from_dict(c) for c in data.get("removed_clips_summary", [])],
            overall_confidence=float(data.get("overall_confidence", 1.0)),
            requires_human_review=bool(data.get("requires_human_review", True)),
            low_confidence_halt=bool(data.get("low_confidence_halt", False)),
            review_notes=list(data.get("review_notes", [])),
            warnings=list(data.get("warnings", [])),
        )

    @classmethod
    def load_json(cls, path: Path) -> "EditDecisionModel":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

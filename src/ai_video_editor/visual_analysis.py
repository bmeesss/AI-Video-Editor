"""Scene-change and visual-motion analysis using PySceneDetect, PyAV, and OpenCV.

Important methodological note:
Visual motion and scene cuts are used strictly as protective/contextual signals
(e.g., to avoid cutting silent gameplay moments where active visual gameplay is
occurring on screen, and to align cuts cleanly near scene transitions).
They are NEVER treated as ground-truth proof that a moment is 'funny' or 'exciting'.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import av
import cv2
import numpy as np
from scenedetect import ContentDetector, SceneManager, open_video

from .config import EditingConfig
from .models import MediaInfo, MotionWindow, SceneBoundary
from .safety import capture_source_snapshot, verify_source_untouched


def _detect_scene_cuts_scenedetect(
    path: Path,
    duration_seconds: float,
    fps: float,
    threshold: float,
) -> List[SceneBoundary]:
    """Run PySceneDetect ContentDetector in read-only mode to identify visual scene boundaries."""
    scenes: List[SceneBoundary] = []
    try:
        try:
            video = open_video(str(path), backend="opencv")
        except Exception:
            video = open_video(str(path), backend="pyav")

        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=threshold, min_scene_len=max(5, int(fps * 0.5))))
        scene_manager.detect_scenes(video=video, show_progress=False)
        raw_scene_list = scene_manager.get_scene_list()

        for idx, (start_tc, end_tc) in enumerate(raw_scene_list):
            scenes.append(
                SceneBoundary(
                    scene_index=idx,
                    start=round(float(start_tc.seconds), 4),
                    end=round(float(end_tc.seconds), 4),
                    start_frame=int(start_tc.frame_num),
                    end_frame=int(end_tc.frame_num),
                )
            )
    except Exception:
        scenes = []

    if not scenes and duration_seconds > 0:
        total_frames = max(1, int(round(duration_seconds * fps)))
        scenes.append(
            SceneBoundary(
                scene_index=0,
                start=0.0,
                end=round(duration_seconds, 4),
                start_frame=0,
                end_frame=total_frames,
            )
        )
    return scenes


def _compute_motion_windows(
    path: Path,
    duration_seconds: float,
    fps: float,
    config: EditingConfig,
) -> List[MotionWindow]:
    """Sample frames in read-only mode, downscale to 160x90 grayscale, and compute 1s motion scores."""
    windows: List[MotionWindow] = []
    sample_interval = 1.0 / max(0.5, float(config.motion_sample_fps))

    frame_diffs: List[Tuple[float, float]] = []
    prev_gray: np.ndarray | None = None
    next_sample_time = 0.0

    with av.open(str(path), mode="r") as container:
        if not container.streams.video:
            return []
        vstream = container.streams.video[0]
        tb = float(vstream.time_base) if vstream.time_base else (1.0 / max(1.0, fps))

        for frame_idx, frame in enumerate(container.decode(vstream)):
            if frame.pts is not None and vstream.time_base is not None:
                t_sec = float(frame.pts) * tb
            else:
                t_sec = float(frame_idx) / max(1.0, fps)

            if t_sec + 1e-4 < next_sample_time:
                continue
            next_sample_time = t_sec + sample_interval

            gray_full = frame.to_ndarray(format="gray")
            small_gray = cv2.resize(gray_full, (160, 90), interpolation=cv2.INTER_AREA)

            if prev_gray is not None:
                diff = float(np.mean(cv2.absdiff(small_gray, prev_gray)))
                frame_diffs.append((t_sec, diff))
            else:
                frame_diffs.append((t_sec, 0.0))
            prev_gray = small_gray

    if not frame_diffs and duration_seconds <= 0:
        return []

    max_time = max(duration_seconds, frame_diffs[-1][0] if frame_diffs else 0.0)
    win_len = 1.0
    t_cursor = 0.0
    while t_cursor < max_time - 1e-3:
        w_start = t_cursor
        w_end = min(max_time, t_cursor + win_len)
        scores = [score for (ts, score) in frame_diffs if w_start <= ts < w_end]
        mean_score = float(np.mean(scores)) if scores else 0.0
        windows.append(
            MotionWindow(
                start=round(w_start, 4),
                end=round(w_end, 4),
                motion_score=round(mean_score, 3),
                is_active_gameplay=mean_score >= config.active_gameplay_motion_threshold,
            )
        )
        t_cursor += win_len

    return windows


def analyze_visual_scenes_and_motion(
    source_path: Path | str,
    media_info: MediaInfo,
    config: EditingConfig,
) -> Tuple[List[SceneBoundary], List[MotionWindow]]:
    """Run read-only visual scene and motion analysis with strict source-file integrity verification."""
    if not config.enable_visual_analysis or not media_info.video_streams:
        return [], []

    snap = capture_source_snapshot(source_path)
    path = Path(snap.resolved_path)
    vinfo = media_info.video_streams[0]

    try:
        scenes = _detect_scene_cuts_scenedetect(
            path=path,
            duration_seconds=media_info.duration_seconds,
            fps=vinfo.fps,
            threshold=config.scene_detection_threshold,
        )
        motion_windows = _compute_motion_windows(
            path=path,
            duration_seconds=media_info.duration_seconds,
            fps=vinfo.fps,
            config=config,
        )
    finally:
        verify_source_untouched(snap)

    return scenes, motion_windows

"""Synthetic gaming media generator for deterministic unit, integration, and E2E tests."""

from __future__ import annotations

import subprocess
from pathlib import Path

import imageio_ffmpeg


def create_synthetic_gaming_video(
    output_path: Path,
    duration_seconds: int = 20,
    fps: int = 30,
    dual_audio_tracks: bool = True,
) -> Path:
    """Generate a real MP4/MKV gaming test recording with known speech bursts, silence, and visual scenes.

    Timeline of the 20-second synthetic recording:
      0.0s - 5.0s : Intro scene (blue moving box) + active microphone commentary burst (0.5s - 4.5s)
      5.0s - 10.0s: Dead-air pause (static dark screen + silent mic + quiet game audio) -> candidate for removal
      10.0s - 15.0s: Active gameplay scene (red fast-moving pattern) + commentary burst (10.2s - 14.5s)
      15.0s - 18.0s: Quiet commentary but HIGH visual motion (green scrolling bars) -> protected by visual motion!
      18.0s - 20.0s: Outro commentary burst (18.2s - 19.8s)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    # Video filter:
    # 0-5s: blue background with moving white rectangle
    # 5-10s: static dark gray screen (dead air)
    # 10-15s: red background with moving yellow rectangle
    # 15-18s: green background with rapid moving white bars (high visual motion)
    # 18-20s: purple outro with moving box
    vf_expr = (
        f"color=c=0x101018:s=640x360:r={fps}:d={duration_seconds},"
        "drawbox=enable='between(t,0,5)':x='mod(t*120,500)':y=80:w=120:h=120:c=dodgerblue@1.0:t=fill,"
        "drawbox=enable='between(t,10,15)':x=0:y=0:w=640:h=360:c=darkred@1.0:t=fill,"
        "drawbox=enable='between(t,10,15)':x='mod(t*180,500)':y=100:w=140:h=140:c=gold@1.0:t=fill,"
        "drawbox=enable='between(t,15,18)':x=0:y=0:w=640:h=360:c=darkgreen@1.0:t=fill,"
        "drawbox=enable='between(t,15,18)':x='mod(t*400,520)':y='mod(t*300,240)':w=180:h=180:c=white@1.0:t=fill,"
        "drawbox=enable='between(t,18,20)':x='mod(t*150,500)':y=60:w=120:h=120:c=violet@1.0:t=fill,"
        "format=yuv420p[vout]"
    )

    # Audio Track 1 (Microphone): speech-band bursts at [0.5-4.5], [10.2-14.5], [18.2-19.8], silent otherwise
    mic_expr = (
        "aevalsrc='"
        "0.35*(sin(2*PI*220*t)+0.5*sin(2*PI*440*t)+0.25*sin(2*PI*880*t))"
        "*(between(t,0.5,4.5)+between(t,10.2,14.5)+between(t,18.2,19.8))"
        f"':s=48000:d={duration_seconds},"
        "pan=stereo|c0=c0|c1=c0[mic]"
    )

    if dual_audio_tracks:
        # Audio Track 2 (Game Audio): louder during active scenes [0-5], [10-18], [18-20], very quiet during [5-10]
        game_expr = (
            "aevalsrc='"
            "0.18*(sin(2*PI*110*t)+0.5*sin(2*PI*330*t))"
            "*(between(t,0,5)+between(t,10,20))"
            f"':s=48000:d={duration_seconds},"
            "pan=stereo|c0=c0|c1=c0[game]"
        )
        filter_complex = f"{vf_expr};{mic_expr};{game_expr}"
        cmd = [
            ffmpeg_exe,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-filter_complex",
            filter_complex,
            "-map",
            "[vout]",
            "-map",
            "[mic]",
            "-map",
            "[game]",
            "-metadata:s:a:0",
            "title=Microphone Commentary",
            "-metadata:s:a:1",
            "title=Desktop Game Audio",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-g",
            str(fps),
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            str(output_path),
        ]
    else:
        filter_complex = f"{vf_expr};{mic_expr}"
        cmd = [
            ffmpeg_exe,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-filter_complex",
            filter_complex,
            "-map",
            "[vout]",
            "-map",
            "[mic]",
            "-metadata:s:a:0",
            "title=Mixed Audio",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-g",
            str(fps),
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            str(output_path),
        ]

    subprocess.run(cmd, check=True)
    return output_path

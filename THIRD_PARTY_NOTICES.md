# Third-Party Licenses and Notices

AI-Video-Editor is licensed under the MIT License (`LICENSE`). It integrates with the following free and open-source software packages (installed via PyPI or user-installed desktop applications) without embedding or modifying GPL application code:

| Component | SPDX License | Upstream Project | Notes |
| :--- | :--- | :--- | :--- |
| **faster-whisper** | MIT | `https://github.com/SYSTRAN/faster-whisper` | Includes OpenAI Whisper model weights (MIT) and bundled Silero VAD ONNX model (MIT). |
| **CTranslate2** | MIT | `https://github.com/OpenNMT/CTranslate2` | Transformer inference runtime used by `faster-whisper`. |
| **ONNX Runtime** (`onnxruntime`) | MIT | `https://github.com/microsoft/onnxruntime` | Used for local CPU execution of Silero VAD (`silero_vad_v6.onnx`). |
| **Silero VAD** | MIT | `https://github.com/snakers4/silero-vad` | Voice Activity Detection model weights bundled in `faster-whisper`. |
| **PyAV** (`av`) | BSD-3-Clause | `https://github.com/PyAV-Org/PyAV` | Python bindings for FFmpeg libraries (LGPLv2.1+ / GPL depending on binary wheel build). |
| **imageio-ffmpeg** | BSD-2-Clause | `https://github.com/imageio/imageio-ffmpeg` | Portable FFmpeg binary wrapper used when system `ffmpeg` is not on `PATH`. |
| **PySceneDetect** (`scenedetect`) | BSD-3-Clause | `https://github.com/Breakthrough/PySceneDetect` | Scene cut and visual transition detection. |
| **OpenCV** (`opencv-python` / `opencv-python-headless`) | Apache-2.0 (OpenCV) / MIT (wrapper) | `https://github.com/opencv/opencv-python` | Downscaled visual motion analysis. |
| **NumPy** (`numpy`) | BSD-3-Clause | `https://github.com/numpy/numpy` | Numerical array processing for audio RMS and visual frame deltas. |
| **pytest** | MIT | `https://github.com/pytest-dev/pytest` | Automated testing framework. |
| **Kdenlive** (External Editor) | GPL-3.0-only | `https://invent.kde.org/multimedia/kdenlive` | User-installed desktop video editor. AI-Video-Editor generates plain-text `.kdenlive` MLT XML project files and does not link against or redistribute Kdenlive binaries. |
| **FFmpeg** | LGPL-2.1-or-later / GPL-2.0-or-later | `https://ffmpeg.org/` | Multimedia decoding and stream inspection. |

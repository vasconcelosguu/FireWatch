from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass
class VideoMetadata:
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float


def read_video_metadata(path: str | Path) -> VideoMetadata:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError("Não foi possível abrir o vídeo com OpenCV.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    duration = frame_count / fps if fps > 0 else 0.0
    return VideoMetadata(width, height, fps, frame_count, duration)


def save_uploaded_video(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return Path(tmp.name)

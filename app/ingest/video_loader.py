"""Video loading and frame extraction using OpenCV."""

from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import cv2
import numpy as np


@dataclass
class VideoMetadata:
    path: str
    frame_count: int
    fps: float
    width: int
    height: int
    duration_sec: float


class VideoLoader:
    """Context manager for reading video files via OpenCV."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        self._cap: cv2.VideoCapture | None = None

    def __enter__(self):
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            raise IOError(f"Cannot open video: {self.path}")
        return self

    def __exit__(self, *args):
        if self._cap:
            self._cap.release()

    def metadata(self) -> VideoMetadata:
        """Get video metadata."""
        cap = self._ensure_cap()
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0.0
        return VideoMetadata(
            path=self.path,
            frame_count=frame_count,
            fps=fps,
            width=width,
            height=height,
            duration_sec=duration,
        )

    def frames(self, sample_rate: int = 1) -> Generator[tuple[int, np.ndarray], None, None]:
        """Yield (frame_index, frame_array) tuples, sampling every Nth frame."""
        cap = self._ensure_cap()
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % sample_rate == 0:
                yield idx, frame
            idx += 1

    def _ensure_cap(self) -> cv2.VideoCapture:
        if self._cap is None or not self._cap.isOpened():
            raise RuntimeError("VideoLoader must be used as a context manager")
        return self._cap


def extract_frames(
    video_path: str | Path,
    output_dir: str | Path,
    fps: int = 10,
) -> list[Path]:
    """Extract frames from video at given FPS, saving as JPEG files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    with VideoLoader(video_path) as loader:
        meta = loader.metadata()
        sample_rate = max(1, int(meta.fps / fps))
        for idx, frame in loader.frames(sample_rate=sample_rate):
            path = output_dir / f"frame_{idx:06d}.jpg"
            cv2.imwrite(str(path), frame)
            saved.append(path)

    return saved

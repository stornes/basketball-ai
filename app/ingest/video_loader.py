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


def _scaled_size(
    native_w: int, native_h: int, target_w: int,
) -> tuple[int, int] | None:
    """Return (width, height) scaled to target_w preserving aspect ratio.

    Returns None if no scaling is needed (native already <= target).
    """
    if target_w and native_w > target_w:
        scale = target_w / native_w
        return target_w, int(native_h * scale)
    return None


class VideoLoader:
    """Context manager for reading video files via OpenCV."""

    def __init__(self, path: str | Path, decode_width: int = 0):
        self.path = str(path)
        self._decode_width = decode_width  # target width; 0 = native
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
        """Get video metadata.

        If decode_width is set, width/height reflect the decode resolution
        (what frames() actually yields), not the native video resolution.
        """
        cap = self._ensure_cap()
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        native_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        native_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0.0

        # Compute effective decode resolution
        scaled = _scaled_size(native_w, native_h, self._decode_width)
        width, height = scaled if scaled else (native_w, native_h)

        return VideoMetadata(
            path=self.path,
            frame_count=frame_count,
            fps=fps,
            width=width,
            height=height,
            duration_sec=duration,
        )

    def frames(self, sample_rate: int = 1) -> Generator[tuple[int, np.ndarray], None, None]:
        """Yield (frame_index, frame_array) tuples, sampling every Nth frame.

        Uses grab/retrieve to skip decoding non-sampled frames (~2.7 min
        savings on a 90-min game at sample_rate=3). Background thread handles
        the grab/retrieve loop for parallel I/O.
        """
        import queue
        from concurrent.futures import ThreadPoolExecutor

        cap = self._ensure_cap()
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        q: queue.Queue = queue.Queue(maxsize=30)

        # Pre-compute decode target size once
        target_size = _scaled_size(
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            self._decode_width,
        )

        def _reader():
            idx = 0
            while True:
                grabbed = cap.grab()
                if not grabbed:
                    q.put((False, -1, None))
                    break
                if idx % sample_rate == 0:
                    ret, frame = cap.retrieve()
                    if ret and target_size is not None:
                        frame = cv2.resize(frame, target_size,
                                           interpolation=cv2.INTER_AREA)
                    q.put((ret, idx, frame))
                idx += 1

        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(_reader)
            while True:
                ok, idx, frame = q.get()
                if not ok:
                    break
                yield idx, frame

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

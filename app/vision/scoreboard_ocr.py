"""Scoreboard OCR for quarter boundary detection.

Reads the quarter indicator text (1ST, 2ND, 3RD, 4TH) from the video's
scoreboard overlay at regular intervals to detect precise quarter boundaries
in wall-clock (video) time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import cv2
import numpy as np


# Mapping from OCR text to quarter number
_QUARTER_PATTERNS: dict[str, int] = {
    "1ST": 1, "1st": 1, "IST": 1,
    "2ND": 2, "2nd": 2, "2NO": 2,
    "3RD": 3, "3rd": 3, "3R0": 3,
    "4TH": 4, "4th": 4, "4TM": 4,
}


@dataclass
class ScoreboardRegion:
    """Pixel coordinates for scoreboard elements at 3840x2160 reference.

    Coordinates are for the XbotGo Falcon overlay format.
    Automatically scaled to actual frame size via scale_to().
    """
    # Quarter indicator text region (e.g., "1ST", "2ND")
    quarter_y1: int
    quarter_y2: int
    quarter_x1: int
    quarter_x2: int

    @classmethod
    def xbotgo_4k(cls) -> "ScoreboardRegion":
        """Preset for XbotGo Falcon overlay at 3840x2160."""
        return cls(
            quarter_y1=1900, quarter_y2=1990,
            quarter_x1=480, quarter_x2=690,
        )

    def scale_to(self, width: int, height: int) -> "ScoreboardRegion":
        """Scale coordinates from 4K reference to actual frame size."""
        sx = width / 3840
        sy = height / 2160
        return ScoreboardRegion(
            quarter_y1=int(self.quarter_y1 * sy),
            quarter_y2=int(self.quarter_y2 * sy),
            quarter_x1=int(self.quarter_x1 * sx),
            quarter_x2=int(self.quarter_x2 * sx),
        )


@dataclass
class QuarterBoundary:
    """A detected quarter boundary in video time."""
    quarter: int  # Quarter STARTING at this boundary (2 = Q2 starts here)
    video_timestamp_sec: float


class QuarterBoundaryDetector:
    """Detect quarter boundaries by OCR-ing the scoreboard quarter indicator.

    Reads "1ST", "2ND", "3RD", "4TH" from the scoreboard overlay and
    detects when transitions occur.

    Usage:
        detector = QuarterBoundaryDetector()
        boundaries = detector.detect(video_path, game_start_sec=139)
        # boundaries = [QuarterBoundary(2, 1200.0), QuarterBoundary(3, 2500.0), ...]
    """

    def __init__(
        self,
        region: ScoreboardRegion | None = None,
        sample_interval_sec: float = 15.0,
    ):
        self._region = region or ScoreboardRegion.xbotgo_4k()
        self._sample_interval = sample_interval_sec
        self._reader = None

    def _get_reader(self):
        if self._reader is None:
            import easyocr
            self._reader = easyocr.Reader(
                ["en"], gpu=False, verbose=False,
            )
        return self._reader

    def detect(
        self,
        video_path: str,
        game_start_sec: float = 0.0,
    ) -> list[QuarterBoundary]:
        """Scan video scoreboard and return detected quarter boundaries.

        Args:
            video_path: Path to video file.
            game_start_sec: Video timestamp when the game starts (tip-off).
                Scanning begins from this point.

        Returns:
            List of QuarterBoundary for each detected transition.
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        region = self._region.scale_to(frame_w, frame_h)

        boundaries: list[QuarterBoundary] = []
        last_quarter = 1  # Game starts in Q1

        print(f"  Scanning scoreboard for quarter transitions "
              f"(from {game_start_sec:.0f}s, interval={self._sample_interval}s)...")

        ts = game_start_sec
        while ts < duration:
            frame_idx = int(ts * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                ts += self._sample_interval
                continue

            quarter = self._read_quarter(frame, region)

            if quarter is not None and quarter > last_quarter:
                boundaries.append(QuarterBoundary(
                    quarter=quarter,
                    video_timestamp_sec=ts,
                ))
                print(f"    Q{last_quarter}→Q{quarter} at {ts:.0f}s ({ts/60:.1f}min)")
                last_quarter = quarter

                # All quarters found for regulation game
                if last_quarter >= 4:
                    break

            ts += self._sample_interval

        cap.release()
        return boundaries

    def _read_quarter(self, frame: np.ndarray, region: ScoreboardRegion) -> int | None:
        """OCR the quarter indicator text from a frame.

        Returns quarter number (1-4) or None if unreadable.
        """
        crop = frame[
            region.quarter_y1:region.quarter_y2,
            region.quarter_x1:region.quarter_x2,
        ]
        if crop.size == 0:
            return None

        # 4x upscale for reliable OCR on small text
        crop_big = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

        reader = self._get_reader()
        try:
            results = reader.readtext(crop_big, detail=1)
        except Exception:
            return None

        for _bbox, text, conf in results:
            if conf < 0.4:
                continue
            # Try exact match first
            text_clean = text.strip().upper()
            if text_clean in _QUARTER_PATTERNS:
                return _QUARTER_PATTERNS[text_clean]
            # Try partial match (e.g., "5 1ST" → "1ST")
            for pattern, qnum in _QUARTER_PATTERNS.items():
                if pattern.upper() in text_clean:
                    return qnum

        return None


def quarter_boundaries_to_ranges(
    boundaries: list[QuarterBoundary],
    game_start_sec: float,
    video_duration_sec: float,
) -> list[tuple[float, float]]:
    """Convert boundary list to (start_sec, end_sec) ranges per quarter.

    Returns: list of (start, end) tuples, one per quarter.
    Shots before game_start_sec are excluded (warmup).
    """
    starts = [game_start_sec] + [b.video_timestamp_sec for b in boundaries]
    ends = [b.video_timestamp_sec for b in boundaries] + [video_duration_sec]
    return list(zip(starts, ends))


def timestamp_to_quarter(
    timestamp_sec: float,
    quarter_ranges: list[tuple[float, float]],
) -> int:
    """Map a video timestamp to its quarter number (1-based).

    Returns 0 for timestamps before game start (warmup).
    """
    for i, (start, end) in enumerate(quarter_ranges):
        if start <= timestamp_sec < end:
            return i + 1
    # If past the last range, assign to last quarter
    if quarter_ranges and timestamp_sec >= quarter_ranges[-1][1]:
        return len(quarter_ranges)
    return 0  # Before game start

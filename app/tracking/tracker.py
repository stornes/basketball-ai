"""Player tracking using DeepSORT."""

from dataclasses import dataclass

import numpy as np

from app.vision.detection_types import BoundingBox, Detection


@dataclass
class TrackedPlayer:
    track_id: int
    bbox: BoundingBox
    frame_idx: int
    is_confirmed: bool
    court_position: tuple[float, float] | None = None


class PlayerTracker:
    """Wraps DeepSORT for persistent player ID assignment across frames."""

    def __init__(self, max_age: int = 30, n_init: int = 3):
        from deep_sort_realtime.deepsort_tracker import DeepSort
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_cosine_distance=0.3,
            embedder_gpu=False,  # CPU embedding for MPS compatibility
        )

    def update(
        self, detections: list[Detection], frame: np.ndarray, frame_idx: int
    ) -> list[TrackedPlayer]:
        """Update tracker with new detections, return tracked players."""
        # Filter to person detections only
        person_dets = [d for d in detections if d.class_id == 0]

        if not person_dets:
            # Still update tracker with empty to age out old tracks
            self.tracker.update_tracks([], frame=frame)
            return []

        # Convert to DeepSORT format: ([x, y, w, h], confidence, class_id)
        ds_detections = [
            (
                list(det.bbox.xywh),
                det.confidence,
                det.class_id,
            )
            for det in person_dets
        ]

        tracks = self.tracker.update_tracks(ds_detections, frame=frame)

        tracked = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            ltrb = track.to_ltrb()
            tracked.append(TrackedPlayer(
                track_id=track.track_id,
                bbox=BoundingBox(
                    x1=float(ltrb[0]),
                    y1=float(ltrb[1]),
                    x2=float(ltrb[2]),
                    y2=float(ltrb[3]),
                ),
                frame_idx=frame_idx,
                is_confirmed=True,
            ))

        return tracked

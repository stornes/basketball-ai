"""Player tracking using IoU-based matching.

Replaces DeepSORT to eliminate the MobileNetV2 CPU embedding bottleneck
(was 27-45 min for a full game). Pure IoU matching is sufficient for
basketball's fixed-camera, well-spaced players.
"""

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
    """IoU-based player tracker. No CNN embeddings — pure geometric matching."""

    def __init__(self, max_age: int = 30, n_init: int = 3, iou_threshold: float = 0.3):
        self._next_id = 1
        self._max_age = max_age
        self._n_init = n_init
        self._iou_threshold = iou_threshold
        # Active tracks: {track_id: {"bbox": BoundingBox, "age": int, "hits": int}}
        self._tracks: dict[int, dict] = {}

    def update(
        self, detections: list[Detection], frame_idx: int
    ) -> list[TrackedPlayer]:
        """Update tracker with new detections, return tracked players."""
        person_dets = [d for d in detections if d.class_id == 0]

        if not person_dets:
            self._age_tracks()
            return []

        # Compute IoU cost matrix between existing tracks and new detections
        track_ids = list(self._tracks.keys())
        matched_dets: set[int] = set()

        if track_ids:
            iou_matrix = np.zeros((len(track_ids), len(person_dets)))
            for i, tid in enumerate(track_ids):
                for j, det in enumerate(person_dets):
                    iou_matrix[i, j] = self._tracks[tid]["bbox"].iou(det.bbox)

            # Greedy matching: assign best IoU pairs above threshold
            matched_tracks: set[int] = set()
            indices = np.dstack(np.unravel_index(
                np.argsort(iou_matrix.ravel())[::-1], iou_matrix.shape
            ))[0]
            for ti, di in indices:
                if ti in matched_tracks or di in matched_dets:
                    continue
                if iou_matrix[ti, di] < self._iou_threshold:
                    break
                tid = track_ids[ti]
                self._tracks[tid]["bbox"] = person_dets[di].bbox
                self._tracks[tid]["age"] = 0
                self._tracks[tid]["hits"] += 1
                matched_tracks.add(ti)
                matched_dets.add(di)

        # Create new tracks for unmatched detections
        for j, det in enumerate(person_dets):
            if j not in matched_dets:
                self._create_track(det.bbox)

        self._age_tracks()

        # Return confirmed tracks
        tracked = []
        for tid, info in self._tracks.items():
            if info["hits"] >= self._n_init:
                tracked.append(TrackedPlayer(
                    track_id=tid,
                    bbox=info["bbox"],
                    frame_idx=frame_idx,
                    is_confirmed=True,
                ))
        return tracked

    def _create_track(self, bbox: BoundingBox):
        """Register a new track with the given bounding box."""
        self._tracks[self._next_id] = {"bbox": bbox, "age": 0, "hits": 1}
        self._next_id += 1

    def _age_tracks(self):
        """Increment age for all tracks and remove stale ones."""
        to_remove = []
        for tid, info in self._tracks.items():
            info["age"] += 1
            if info["age"] > self._max_age:
                to_remove.append(tid)
        for tid in to_remove:
            del self._tracks[tid]

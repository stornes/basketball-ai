"""Player tracking using DeepSORT (appearance embedding + Kalman filter).

Wraps deep_sort_realtime.DeepSort to provide the same TrackedPlayer output
interface as the IoU-based PlayerTracker, so the rest of the pipeline is
unaware of which tracker is active.

Key difference from IoU tracking:
    DeepSORT uses MobileNetV2 appearance embeddings extracted from the live
    frame to re-identify players across occlusions and shot clock stoppages.
    The tradeoff is CPU cost (~150-200ms/frame on a CPU-only machine); for
    GPU or MPS inference this is negligible.

Usage::

    tracker = DeepSortTracker(max_age=30, n_init=3)
    players = tracker.update(detections, frame)
"""

import numpy as np

from deep_sort_realtime.deepsort_tracker import DeepSort

from app.tracking.tracker import TrackedPlayer
from app.vision.detection_types import BoundingBox, Detection


class DeepSortTracker:
    """DeepSORT-backed player tracker with TrackedPlayer output interface.

    Args:
        max_age: Frames a lost track is kept alive before deletion.
            Equivalent to the IoU tracker's max_age parameter.
        n_init: Detections required before a track is confirmed.
            Keeps tentative tracks off the output until DeepSORT is confident.
    """

    def __init__(self, max_age: int = 30, n_init: int = 3) -> None:
        self._tracker = DeepSort(max_age=max_age, n_init=n_init)

    def update(
        self, detections: list[Detection], frame: np.ndarray
    ) -> list[TrackedPlayer]:
        """Feed detections and the current frame, return confirmed tracked players.

        DeepSORT requires the raw frame to crop player regions and extract
        appearance embeddings. These embeddings let it maintain stable IDs
        through occlusions — the key advantage over IoU-only matching.

        Args:
            detections: All detections from the current frame. Non-person
                class_ids are filtered out before being forwarded to DeepSORT.
            frame: The current BGR frame as a numpy array (H x W x 3). This
                is used by DeepSORT for appearance embedding extraction and
                must not be None.

        Returns:
            List of confirmed TrackedPlayer objects. Unconfirmed tracks (fewer
            than n_init detections seen) are excluded to match IoU tracker
            semantics.
        """
        person_dets = [d for d in detections if d.class_id == 0]

        # deep_sort_realtime expects [([left, top, width, height], confidence, class), ...]
        raw_detections = [
            (
                [d.bbox.x1, d.bbox.y1, d.bbox.width, d.bbox.height],
                d.confidence,
                d.class_id,
            )
            for d in person_dets
        ]

        tracks = self._tracker.update_tracks(raw_detections, frame=frame)

        players: list[TrackedPlayer] = []
        for track in tracks:
            if not track.is_confirmed():
                continue

            tlbr = track.to_tlbr()
            bbox = BoundingBox(
                x1=float(tlbr[0]),
                y1=float(tlbr[1]),
                x2=float(tlbr[2]),
                y2=float(tlbr[3]),
            )
            players.append(
                TrackedPlayer(
                    track_id=int(track.track_id),
                    bbox=bbox,
                    frame_idx=0,   # DeepSORT is frame-index-agnostic
                    is_confirmed=True,
                )
            )

        return players

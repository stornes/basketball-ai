"""Team classification via jersey colour clustering.

Uses K-Means (k=2) on CIE Lab colour features extracted from player
torso crops to assign players to home/away teams.
"""

from __future__ import annotations

from collections import defaultdict

import cv2
import numpy as np
from sklearn.cluster import KMeans

from app.vision.detection_types import BoundingBox


class TeamClassifier:
    """Classify players into two teams based on jersey colour.

    Usage:
        classifier = TeamClassifier()
        # During frame processing:
        classifier.collect_sample(track_id, frame, bbox)
        # After all frames:
        classifier.classify()
        # Query:
        team = classifier.get_team(track_id)  # "home" | "away" | None
    """

    def __init__(self, sample_interval: int = 5):
        """
        Args:
            sample_interval: Collect a colour sample every N detections per track.
                Lower = more samples but slower. Default 5 (~20% compute overhead).
        """
        self._sample_interval = sample_interval
        self._detection_counts: dict[int, int] = defaultdict(int)
        # {track_id: list of Lab colour vectors}
        self._colour_samples: dict[int, list[np.ndarray]] = defaultdict(list)
        self._track_to_team: dict[int, str] = {}
        self._classified = False

    def collect_sample(
        self, track_id: int, frame: np.ndarray, bbox: BoundingBox
    ) -> None:
        """Extract jersey colour from player torso crop.

        Only samples every `sample_interval`-th detection per track
        to keep compute overhead low.
        """
        self._detection_counts[track_id] += 1
        if self._detection_counts[track_id] % self._sample_interval != 0:
            return

        # Extract torso region (20-50% height, 20-80% width of player bbox)
        x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)
        h = y2 - y1
        w = x2 - x1
        if h < 10 or w < 10:
            return

        torso_y1 = y1 + int(h * 0.2)
        torso_y2 = y1 + int(h * 0.5)
        torso_x1 = x1 + int(w * 0.2)
        torso_x2 = x1 + int(w * 0.8)

        # Clamp to frame bounds
        fh, fw = frame.shape[:2]
        torso_y1 = max(0, min(torso_y1, fh - 1))
        torso_y2 = max(torso_y1 + 1, min(torso_y2, fh))
        torso_x1 = max(0, min(torso_x1, fw - 1))
        torso_x2 = max(torso_x1 + 1, min(torso_x2, fw))

        crop = frame[torso_y1:torso_y2, torso_x1:torso_x2]
        if crop.size == 0:
            return

        # Convert to CIE Lab (perceptually uniform — better clustering than RGB/HSV)
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2Lab)

        # Mean colour across torso crop
        mean_lab = lab.reshape(-1, 3).mean(axis=0).astype(np.float32)
        self._colour_samples[track_id].append(mean_lab)

    def classify(self) -> dict[int, str]:
        """Run K-Means clustering to separate players into two teams.

        Returns:
            Mapping of track_id → "home" | "away".
        """
        # Gather all tracks with sufficient samples
        valid_tracks = {
            tid: samples
            for tid, samples in self._colour_samples.items()
            if len(samples) >= 1
        }

        if len(valid_tracks) < 2:
            # Not enough data — fall back to parity heuristic
            self._track_to_team = {
                tid: "home" if tid % 2 == 0 else "away"
                for tid in self._colour_samples
            }
            self._classified = True
            return self._track_to_team

        # Compute mean colour per track
        track_ids = list(valid_tracks.keys())
        mean_colours = np.array([
            np.mean(valid_tracks[tid], axis=0) for tid in track_ids
        ])

        # K-Means with k=2
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = kmeans.fit_predict(mean_colours)

        # Assign cluster labels to team names
        # Convention: the cluster with more tracks is "home" (home team
        # typically has more camera time in their arena)
        cluster_counts = np.bincount(labels, minlength=2)
        home_cluster = int(np.argmax(cluster_counts))

        self._track_to_team = {}
        for tid, label in zip(track_ids, labels):
            self._track_to_team[tid] = "home" if label == home_cluster else "away"

        # Fill in tracks without enough samples using parity fallback
        for tid in self._colour_samples:
            if tid not in self._track_to_team:
                self._track_to_team[tid] = "home" if tid % 2 == 0 else "away"

        self._classified = True
        return self._track_to_team

    def get_team(self, track_id: int) -> str | None:
        """Get team assignment for a track.

        Returns:
            "home", "away", or None if not classified.
        """
        if not self._classified:
            return None
        return self._track_to_team.get(track_id)

    @property
    def is_classified(self) -> bool:
        return self._classified

    @property
    def track_count(self) -> int:
        """Number of tracks with colour samples."""
        return len(self._colour_samples)

"""Tests for jersey colour team classification."""

import numpy as np
import pytest

from app.tracking.team_classifier import TeamClassifier
from app.vision.detection_types import BoundingBox


def _make_frame(jersey_bgr: tuple[int, int, int], height: int = 480, width: int = 640) -> np.ndarray:
    """Create a frame with a solid-colour torso region."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    # Fill the entire frame with the jersey colour for simplicity
    frame[:] = jersey_bgr
    return frame


def test_classify_two_distinct_teams():
    """Two clearly different jersey colours should produce two teams."""
    classifier = TeamClassifier(sample_interval=1)

    # Team 1: red jerseys (BGR)
    red_frame = _make_frame((0, 0, 200))
    bbox = BoundingBox(100, 100, 200, 300)
    for i in range(5):
        classifier.collect_sample(track_id=1, frame=red_frame, bbox=bbox)
        classifier.collect_sample(track_id=2, frame=red_frame, bbox=bbox)

    # Team 2: blue jerseys (BGR)
    blue_frame = _make_frame((200, 0, 0))
    for i in range(5):
        classifier.collect_sample(track_id=3, frame=blue_frame, bbox=bbox)
        classifier.collect_sample(track_id=4, frame=blue_frame, bbox=bbox)

    teams = classifier.classify()

    # Tracks 1 and 2 should be on the same team
    assert teams[1] == teams[2]
    # Tracks 3 and 4 should be on the same team
    assert teams[3] == teams[4]
    # But different from tracks 1/2
    assert teams[1] != teams[3]


def test_classify_returns_home_away():
    """Team labels should be 'home' or 'away'."""
    classifier = TeamClassifier(sample_interval=1)
    bbox = BoundingBox(50, 50, 150, 250)

    classifier.collect_sample(1, _make_frame((0, 0, 255)), bbox)
    classifier.collect_sample(2, _make_frame((255, 0, 0)), bbox)

    teams = classifier.classify()
    assert set(teams.values()) == {"home", "away"}


def test_get_team_before_classify_returns_none():
    """Before classify() is called, get_team should return None."""
    classifier = TeamClassifier()
    assert classifier.get_team(1) is None
    assert not classifier.is_classified


def test_get_team_unknown_track():
    """After classify(), unknown track IDs should return None."""
    classifier = TeamClassifier(sample_interval=1)
    bbox = BoundingBox(50, 50, 150, 250)

    classifier.collect_sample(1, _make_frame((0, 0, 255)), bbox)
    classifier.collect_sample(2, _make_frame((255, 0, 0)), bbox)
    classifier.classify()

    assert classifier.get_team(999) is None


def test_sample_interval_skips_frames():
    """With sample_interval=3, only every 3rd detection should be sampled."""
    classifier = TeamClassifier(sample_interval=3)
    bbox = BoundingBox(50, 50, 150, 250)
    frame = _make_frame((0, 0, 200))

    for i in range(9):
        classifier.collect_sample(track_id=1, frame=frame, bbox=bbox)

    # 9 detections, interval=3 → samples at detection 3, 6, 9 = 3 samples
    assert len(classifier._colour_samples[1]) == 3


def test_fallback_with_insufficient_tracks():
    """With only 1 track, classifier should fall back to parity heuristic."""
    classifier = TeamClassifier(sample_interval=1)
    bbox = BoundingBox(50, 50, 150, 250)

    classifier.collect_sample(1, _make_frame((0, 0, 255)), bbox)
    teams = classifier.classify()

    assert classifier.is_classified
    # Parity: track 1 is odd → "away"
    assert teams[1] == "away"


def test_small_bbox_ignored():
    """Very small bounding boxes should not produce samples."""
    classifier = TeamClassifier(sample_interval=1)
    tiny_bbox = BoundingBox(100, 100, 105, 105)  # 5x5 px
    frame = _make_frame((0, 0, 200))

    classifier.collect_sample(1, frame, tiny_bbox)
    assert len(classifier._colour_samples[1]) == 0

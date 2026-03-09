"""Tests for shot detection."""

from app.events.shot_detector import ShotDetector
from app.vision.detection_types import BoundingBox, Detection


def _make_ball_detection(x, y, frame_idx):
    return Detection(
        bbox=BoundingBox(x - 10, y - 10, x + 10, y + 10),
        confidence=0.8,
        class_id=32,
        class_name="sports ball",
        frame_idx=frame_idx,
    )


def test_detects_shot_arc():
    """Ball going up then down should trigger a shot detection."""
    detector = ShotDetector(frame_height=480, fps=30.0)

    # Simulate ball arc: starts at y=300, goes up to y=50, then back down
    arc_y = [300, 250, 200, 150, 100, 60, 50, 60, 100, 150, 200]
    shot_detected = False

    for i, y in enumerate(arc_y):
        ball = _make_ball_detection(320, y, frame_idx=i)
        result = detector.update(ball, [], i)
        if result is not None:
            shot_detected = True
            break

    assert shot_detected


def test_no_shot_on_flat_trajectory():
    """Ball moving horizontally should not trigger a shot."""
    detector = ShotDetector(frame_height=480, fps=30.0)

    for i in range(20):
        ball = _make_ball_detection(100 + i * 10, 300, frame_idx=i)
        result = detector.update(ball, [], i)
        assert result is None


def test_no_shot_on_small_movement():
    """Small vertical movement should not trigger."""
    detector = ShotDetector(frame_height=480, fps=30.0)

    for i in range(20):
        y = 300 - i * 2  # only 40px total displacement
        ball = _make_ball_detection(320, y, frame_idx=i)
        result = detector.update(ball, [], i)
        assert result is None


def test_cooldown_prevents_double_detection():
    """After a shot, cooldown should prevent immediate re-detection."""
    detector = ShotDetector(frame_height=480, fps=30.0)

    # First arc
    arc_y = [300, 250, 200, 150, 100, 60, 50, 60, 100, 150, 200]
    first_shot = None
    for i, y in enumerate(arc_y):
        ball = _make_ball_detection(320, y, i)
        result = detector.update(ball, [], i)
        if result:
            first_shot = result
            break

    assert first_shot is not None

    # Immediate second arc should be suppressed by cooldown
    second_shot = None
    for j, y in enumerate(arc_y):
        idx = len(arc_y) + j
        ball = _make_ball_detection(320, y, idx)
        result = detector.update(ball, [], idx)
        if result:
            second_shot = result

    assert second_shot is None  # cooldown prevents it

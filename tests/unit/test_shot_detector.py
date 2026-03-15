"""Tests for shot detection and outcome classification."""

import pytest

from app.events.event_types import ShotOutcome
from app.events.shot_detector import BallPosition, ShotDetector
from app.vision.detection_types import BoundingBox, Detection


def _make_ball_detection(x, y, frame_idx):
    return Detection(
        bbox=BoundingBox(x - 10, y - 10, x + 10, y + 10),
        confidence=0.8,
        class_id=32,
        class_name="sports ball",
        frame_idx=frame_idx,
    )


def _make_basket(cx: float, cy: float, w: float = 200, h: float = 100) -> Detection:
    """Create a basket Detection centered at (cx, cy)."""
    return Detection(
        bbox=BoundingBox(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2),
        confidence=0.9,
        class_id=1,
        class_name="basket",
        frame_idx=0,
    )


def _make_detector(basket: Detection | None = None, frame_height: int = 2160) -> ShotDetector:
    """Create a ShotDetector with a pre-set basket detection."""
    det = ShotDetector(frame_height=frame_height, fps=28.0)
    if basket:
        det._last_basket_detection = basket
    return det


# ── Existing shot detection tests ───────────────────────────


def test_detects_shot_arc():
    """Ball going up then down should trigger a shot detection."""
    detector = ShotDetector(frame_height=480, fps=30.0)

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

    arc_y = [300, 250, 200, 150, 100, 60, 50, 60, 100, 150, 200]
    first_shot = None
    for i, y in enumerate(arc_y):
        ball = _make_ball_detection(320, y, i)
        result = detector.update(ball, [], i)
        if result:
            first_shot = result
            break

    assert first_shot is not None

    second_shot = None
    for j, y in enumerate(arc_y):
        idx = len(arc_y) + j
        ball = _make_ball_detection(320, y, idx)
        result = detector.update(ball, [], idx)
        if result:
            second_shot = result

    assert second_shot is None


# ── Outcome classification tests ────────────────────────────


class TestClassifyOutcome:
    """Tests for _classify_outcome (arc shots with post-apex data)."""

    def test_made_when_ball_near_basket(self):
        """Ball descends through basket → MADE."""
        basket = _make_basket(1000, 500, w=200, h=100)
        det = _make_detector(basket)

        positions = [
            BallPosition(x=950, y=800, frame_idx=0),
            BallPosition(x=970, y=600, frame_idx=3),
            BallPosition(x=990, y=400, frame_idx=6),   # apex
            BallPosition(x=1000, y=480, frame_idx=9),
            BallPosition(x=1005, y=520, frame_idx=12),
        ]
        assert det._classify_outcome(positions) == ShotOutcome.MADE

    def test_missed_when_ball_far_from_basket(self):
        """Ball descends far from basket → MISSED."""
        basket = _make_basket(1000, 500, w=200, h=100)
        det = _make_detector(basket)

        positions = [
            BallPosition(x=200, y=800, frame_idx=0),
            BallPosition(x=220, y=600, frame_idx=3),
            BallPosition(x=240, y=400, frame_idx=6),
            BallPosition(x=250, y=500, frame_idx=9),
            BallPosition(x=260, y=700, frame_idx=12),
        ]
        assert det._classify_outcome(positions) == ShotOutcome.MISSED

    def test_attempted_when_no_basket(self):
        """No basket detection → ATTEMPTED."""
        det = _make_detector(basket=None)

        positions = [
            BallPosition(x=500, y=800, frame_idx=0),
            BallPosition(x=510, y=400, frame_idx=3),
            BallPosition(x=520, y=500, frame_idx=6),
        ]
        assert det._classify_outcome(positions) == ShotOutcome.ATTEMPTED

    def test_proximity_scales_with_basket_size(self):
        """Larger basket bbox → larger proximity threshold."""
        small = _make_basket(1000, 500, w=80, h=40)
        det_small = _make_detector(small)

        large = _make_basket(1000, 500, w=400, h=200)
        det_large = _make_detector(large)

        assert det_large._basket_proximity_px() > det_small._basket_proximity_px()
        assert det_small._basket_proximity_px() >= ShotDetector.BASKET_PROXIMITY_MIN_PX

    def test_proximity_never_below_minimum(self):
        """Even tiny basket uses minimum threshold."""
        tiny = _make_basket(500, 500, w=10, h=5)
        det = _make_detector(tiny)
        assert det._basket_proximity_px() == ShotDetector.BASKET_PROXIMITY_MIN_PX


class TestClassifyBallLossOutcome:
    """Tests for _classify_ball_loss_outcome (ball vanishes mid-arc)."""

    def test_made_when_ball_heading_toward_basket(self):
        """Ball ascending toward basket and disappears → MADE."""
        basket = _make_basket(1000, 500, w=200, h=100)
        det = _make_detector(basket)

        positions = [
            BallPosition(x=950, y=800, frame_idx=0),
            BallPosition(x=970, y=700, frame_idx=3),
            BallPosition(x=985, y=600, frame_idx=6),
            BallPosition(x=995, y=520, frame_idx=9),
        ]
        assert det._classify_ball_loss_outcome(positions) == ShotOutcome.MADE

    def test_missed_when_ball_far_horizontally(self):
        """Ball disappears far from basket horizontally → MISSED."""
        basket = _make_basket(1000, 500, w=200, h=100)
        det = _make_detector(basket)

        positions = [
            BallPosition(x=200, y=800, frame_idx=0),
            BallPosition(x=210, y=700, frame_idx=3),
            BallPosition(x=220, y=600, frame_idx=6),
            BallPosition(x=230, y=520, frame_idx=9),
        ]
        assert det._classify_ball_loss_outcome(positions) == ShotOutcome.MISSED

    def test_attempted_when_no_basket(self):
        """No basket detection → ATTEMPTED."""
        det = _make_detector(basket=None)

        positions = [
            BallPosition(x=500, y=800, frame_idx=0),
            BallPosition(x=510, y=700, frame_idx=3),
        ]
        assert det._classify_ball_loss_outcome(positions) == ShotOutcome.ATTEMPTED

    def test_missed_when_ball_too_low(self):
        """Ball disappears well below basket → MISSED."""
        basket = _make_basket(1000, 300, w=200, h=100)
        det = _make_detector(basket)

        positions = [
            BallPosition(x=990, y=900, frame_idx=0),
            BallPosition(x=995, y=850, frame_idx=3),
            BallPosition(x=998, y=800, frame_idx=6),
        ]
        assert det._classify_ball_loss_outcome(positions) == ShotOutcome.MISSED


class TestCreateShotEvent:
    """Integration: _create_shot_event routes to correct classifier."""

    def test_ball_loss_uses_ball_loss_classifier(self):
        """ball_loss=True routes to _classify_ball_loss_outcome."""
        basket = _make_basket(1000, 500, w=200, h=100)
        det = _make_detector(basket)

        positions = [
            BallPosition(x=980, y=700, frame_idx=0),
            BallPosition(x=990, y=600, frame_idx=3),
            BallPosition(x=995, y=530, frame_idx=6),
        ]
        event = det._create_shot_event(positions, [], frame_idx=9, apex_y=530, ball_loss=True)
        assert event.outcome == ShotOutcome.MADE

    def test_arc_uses_standard_classifier(self):
        """ball_loss=False uses _classify_outcome."""
        basket = _make_basket(1000, 500, w=200, h=100)
        det = _make_detector(basket)

        positions = [
            BallPosition(x=950, y=800, frame_idx=0),
            BallPosition(x=990, y=400, frame_idx=3),
            BallPosition(x=1000, y=490, frame_idx=6),
        ]
        event = det._create_shot_event(positions, [], frame_idx=9, apex_y=400, ball_loss=False)
        assert event.outcome == ShotOutcome.MADE

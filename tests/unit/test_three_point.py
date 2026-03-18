"""Tests for three-point shot classification geometry."""

import math

import pytest

from app.events.three_point import (
    BASKET_CENTER_X,
    BASKET_CENTER_Y,
    THREE_PT_ARC_RADIUS_FT,
    ThreePointClassifier,
)


@pytest.fixture
def classifier():
    return ThreePointClassifier()


def test_none_returns_two_pointer(classifier):
    """None court_position → 2-pointer (can't classify)."""
    assert classifier.is_three_pointer(None, None) is False
    assert classifier.is_three_pointer(25.0, None) is False
    assert classifier.is_three_pointer(None, 20.0) is False


def test_at_basket_is_two(classifier):
    """Shot right at the basket is a 2-pointer."""
    assert classifier.is_three_pointer(25.0, 5.25) is False


def test_inside_paint_is_two(classifier):
    """Shot from the paint is a 2-pointer."""
    assert classifier.is_three_pointer(25.0, 10.0) is False


def test_left_corner_three(classifier):
    """Left corner three: x <= 3ft, y <= 14ft."""
    assert classifier.is_three_pointer(2.0, 5.0) is True


def test_right_corner_three(classifier):
    """Right corner three: x >= 47ft, y <= 14ft."""
    assert classifier.is_three_pointer(48.0, 5.0) is True


def test_corner_two_not_deep_enough(classifier):
    """Close to sideline but above the corner break (y > 14ft) → depends on arc distance."""
    # x=2, y=20 — this is near sideline but above break, check arc distance
    dist = math.hypot(2.0 - BASKET_CENTER_X, 20.0 - BASKET_CENTER_Y)
    expected = dist > THREE_PT_ARC_RADIUS_FT
    assert classifier.is_three_pointer(2.0, 20.0) is expected


def test_arc_three_top_of_key(classifier):
    """Shot from top of the key/beyond arc — well beyond 23.75ft."""
    # 25.0, 30.0 → distance = 30 - 5.25 = 24.75 > 23.75
    assert classifier.is_three_pointer(25.0, 30.0) is True


def test_arc_two_just_inside(classifier):
    """Shot just inside the arc — distance < 23.75ft."""
    # Place at exactly 23ft from basket center (below threshold)
    y = BASKET_CENTER_Y + 23.0
    assert classifier.is_three_pointer(BASKET_CENTER_X, y) is False


def test_arc_three_just_outside(classifier):
    """Shot just outside the arc — distance > 23.75ft."""
    y = BASKET_CENTER_Y + 24.0
    assert classifier.is_three_pointer(BASKET_CENTER_X, y) is True


def test_wing_three(classifier):
    """Three-pointer from the wing (45-degree angle)."""
    # Place at ~24ft from basket at ~45 degrees
    angle = math.radians(45)
    dist = 24.5  # beyond 23.75
    x = BASKET_CENTER_X + dist * math.cos(angle)
    y = BASKET_CENTER_Y + dist * math.sin(angle)
    assert classifier.is_three_pointer(x, y) is True


def test_wing_two(classifier):
    """Two-pointer from mid-range wing."""
    angle = math.radians(45)
    dist = 15.0  # well inside
    x = BASKET_CENTER_X + dist * math.cos(angle)
    y = BASKET_CENTER_Y + dist * math.sin(angle)
    assert classifier.is_three_pointer(x, y) is False


def test_free_throw_line_is_two(classifier):
    """Free throw line (15ft from baseline) is a 2-pointer."""
    assert classifier.is_three_pointer(25.0, 19.0) is False

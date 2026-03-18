"""Tests for steal detection heuristic."""

import pytest

from app.events.event_types import PossessionEvent
from app.events.steal_detector import StealDetector, StealEvent


def _make_possession(
    poss_id: int,
    track_id: int,
    team: str,
    start_frame: int,
    end_frame: int,
    result: str = "turnover",
    fps: float = 30.0,
) -> PossessionEvent:
    return PossessionEvent(
        possession_id=poss_id,
        player_track_id=track_id,
        team=team,
        start_frame=start_frame,
        end_frame=end_frame,
        start_time=start_frame / fps,
        end_time=end_frame / fps,
        result=result,
    )


@pytest.fixture
def detector():
    return StealDetector(fps=30.0)


def test_basic_steal(detector):
    """Rapid cross-team possession change without shot → steal."""
    possessions = [
        _make_possession(1, track_id=10, team="home", start_frame=100, end_frame=130),
        _make_possession(2, track_id=20, team="away", start_frame=135, end_frame=160),
    ]
    events = detector.check(possessions)

    assert len(events) == 1
    assert events[0].stealer_track_id == 20
    assert events[0].stealer_team == "away"
    assert events[0].victim_track_id == 10
    assert events[0].victim_team == "home"


def test_no_steal_after_shot(detector):
    """Possession ending with shot → not a steal."""
    possessions = [
        _make_possession(1, track_id=10, team="home", start_frame=100, end_frame=130, result="shot"),
        _make_possession(2, track_id=20, team="away", start_frame=135, end_frame=160),
    ]
    events = detector.check(possessions)
    assert len(events) == 0


def test_no_steal_same_team(detector):
    """Same-team possession change → not a steal (it's a pass)."""
    possessions = [
        _make_possession(1, track_id=10, team="home", start_frame=100, end_frame=130),
        _make_possession(2, track_id=15, team="home", start_frame=135, end_frame=160),
    ]
    events = detector.check(possessions)
    assert len(events) == 0


def test_no_steal_large_gap(detector):
    """Gap too large (> 2 seconds) → not a steal."""
    possessions = [
        _make_possession(1, track_id=10, team="home", start_frame=100, end_frame=130),
        # 5 seconds later at 30fps = 150 frames gap
        _make_possession(2, track_id=20, team="away", start_frame=280, end_frame=310),
    ]
    events = detector.check(possessions)
    assert len(events) == 0


def test_no_steal_missing_team(detector):
    """Missing team info → no steal."""
    possessions = [
        _make_possession(1, track_id=10, team="", start_frame=100, end_frame=130),
        _make_possession(2, track_id=20, team="away", start_frame=135, end_frame=160),
    ]
    events = detector.check(possessions)
    assert len(events) == 0


def test_multiple_steals(detector):
    """Multiple steal patterns detected."""
    possessions = [
        _make_possession(1, track_id=10, team="home", start_frame=100, end_frame=130),
        _make_possession(2, track_id=20, team="away", start_frame=135, end_frame=160),
        # Later...
        _make_possession(3, track_id=20, team="away", start_frame=300, end_frame=330),
        _make_possession(4, track_id=15, team="home", start_frame=335, end_frame=360),
    ]
    events = detector.check(possessions)
    assert len(events) == 2
    assert events[0].stealer_track_id == 20
    assert events[1].stealer_track_id == 15


def test_deduplication_on_recheck(detector):
    """Calling check() again with same data doesn't duplicate events."""
    possessions = [
        _make_possession(1, track_id=10, team="home", start_frame=100, end_frame=130),
        _make_possession(2, track_id=20, team="away", start_frame=135, end_frame=160),
    ]
    events1 = detector.check(possessions)
    assert len(events1) == 1

    # Re-check with same possessions
    events2 = detector.check(possessions)
    assert len(events2) == 0  # no new events

    # Total accumulated
    assert len(detector.events) == 1


def test_overlapping_possessions_valid(detector):
    """Slightly overlapping possessions (gap < 0, > -1s) → still valid steal."""
    possessions = [
        _make_possession(1, track_id=10, team="home", start_frame=100, end_frame=140),
        # Starts slightly before previous ends (overlap)
        _make_possession(2, track_id=20, team="away", start_frame=138, end_frame=170),
    ]
    events = detector.check(possessions)
    assert len(events) == 1


def test_large_overlap_rejected(detector):
    """Large overlap (> 1 second) → rejected as tracking noise."""
    possessions = [
        _make_possession(1, track_id=10, team="home", start_frame=100, end_frame=200),
        # Starts 2 seconds before previous ends
        _make_possession(2, track_id=20, team="away", start_frame=140, end_frame=250),
    ]
    events = detector.check(possessions)
    assert len(events) == 0

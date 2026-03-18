"""Tests for rebound detection heuristic."""

import pytest

from app.events.event_types import ShotEvent, ShotOutcome
from app.events.rebound_detector import ReboundDetector, ReboundEvent, ReboundType
from app.tracking.tracker import TrackedPlayer
from app.vision.detection_types import BoundingBox, Detection


def _make_ball(x: float, y: float, frame_idx: int = 0) -> Detection:
    return Detection(
        bbox=BoundingBox(x - 10, y - 10, x + 10, y + 10),
        confidence=0.9,
        class_id=32,
        class_name="ball",
        frame_idx=frame_idx,
    )


def _make_player(track_id: int, x: float, y: float, frame_idx: int = 0, team: str | None = None) -> TrackedPlayer:
    return TrackedPlayer(
        track_id=track_id,
        bbox=BoundingBox(x - 20, y - 40, x + 20, y + 40),
        frame_idx=frame_idx,
        is_confirmed=True,
        team=team,
    )


def _make_missed_shot(frame_idx: int = 100, team: str = "home") -> ShotEvent:
    return ShotEvent(
        frame_idx=frame_idx,
        timestamp_sec=frame_idx / 30.0,
        shooter_track_id=1,
        court_position=None,
        outcome=ShotOutcome.MISSED,
        clip_start_frame=frame_idx - 30,
        clip_end_frame=frame_idx + 30,
        team=team,
    )


@pytest.fixture
def detector():
    return ReboundDetector(fps=30.0)


def test_no_pending_miss_returns_none(detector):
    """No rebound if no missed shot registered."""
    ball = _make_ball(100, 100)
    player = _make_player(2, 100, 100, team="away")
    assert detector.update(ball, [player], 200) is None


def test_defensive_rebound(detector):
    """Opponent player grabs ball after miss → DRB."""
    miss = _make_missed_shot(frame_idx=100, team="home")
    detector.on_missed_shot(miss)

    # Advance past MIN_DELAY_FRAMES
    for i in range(ReboundDetector.MIN_DELAY_FRAMES):
        detector.update(None, [], 101 + i)

    # Ball near away player → defensive rebound
    ball = _make_ball(200, 300, frame_idx=110)
    player = _make_player(5, 205, 310, frame_idx=110, team="away")
    event = detector.update(ball, [player], 110)

    assert event is not None
    assert event.rebound_type == ReboundType.DEFENSIVE
    assert event.rebounder_track_id == 5
    assert event.rebounder_team == "away"
    assert event.shooter_team == "home"


def test_offensive_rebound(detector):
    """Same-team player grabs ball after miss → ORB."""
    miss = _make_missed_shot(frame_idx=100, team="home")
    detector.on_missed_shot(miss)

    for i in range(ReboundDetector.MIN_DELAY_FRAMES):
        detector.update(None, [], 101 + i)

    ball = _make_ball(200, 300, frame_idx=110)
    player = _make_player(3, 205, 310, frame_idx=110, team="home")
    event = detector.update(ball, [player], 110)

    assert event is not None
    assert event.rebound_type == ReboundType.OFFENSIVE
    assert event.rebounder_track_id == 3


def test_timeout_no_rebound(detector):
    """No rebound attributed after window expires."""
    miss = _make_missed_shot(frame_idx=100, team="home")
    detector.on_missed_shot(miss)

    # Advance past the rebound window (4 seconds at 30fps = 120 frames)
    max_frames = int(ReboundDetector.REBOUND_WINDOW_SEC * 30.0)
    for i in range(max_frames + 2):
        detector.update(None, [], 101 + i)

    # Now try — should not detect anything (pending miss cleared)
    ball = _make_ball(200, 300)
    player = _make_player(5, 205, 310, team="away")
    assert detector.update(ball, [player], 300) is None


def test_player_too_far_from_ball(detector):
    """Player not close enough to ball → no rebound yet."""
    miss = _make_missed_shot(frame_idx=100, team="home")
    detector.on_missed_shot(miss)

    for i in range(ReboundDetector.MIN_DELAY_FRAMES):
        detector.update(None, [], 101 + i)

    # Ball and player far apart (> REBOUND_PROXIMITY_PX)
    ball = _make_ball(100, 100, frame_idx=110)
    player = _make_player(5, 500, 500, frame_idx=110, team="away")
    assert detector.update(ball, [player], 110) is None


def test_no_ball_no_rebound(detector):
    """No ball detection → no rebound."""
    miss = _make_missed_shot(frame_idx=100, team="home")
    detector.on_missed_shot(miss)

    for i in range(ReboundDetector.MIN_DELAY_FRAMES):
        detector.update(None, [], 101 + i)

    player = _make_player(5, 200, 300, frame_idx=110, team="away")
    assert detector.update(None, [player], 110) is None


def test_made_shot_not_registered(detector):
    """Made shots don't trigger rebound detection."""
    made = ShotEvent(
        frame_idx=100,
        timestamp_sec=100 / 30.0,
        shooter_track_id=1,
        court_position=None,
        outcome=ShotOutcome.MADE,
        clip_start_frame=70,
        clip_end_frame=130,
        team="home",
    )
    detector.on_missed_shot(made)
    assert detector._pending_miss is None


def test_no_team_defaults_defensive(detector):
    """When team info missing, defaults to DRB."""
    miss = ShotEvent(
        frame_idx=100,
        timestamp_sec=100 / 30.0,
        shooter_track_id=1,
        court_position=None,
        outcome=ShotOutcome.MISSED,
        clip_start_frame=70,
        clip_end_frame=130,
        team=None,
    )
    detector.on_missed_shot(miss)

    for i in range(ReboundDetector.MIN_DELAY_FRAMES):
        detector.update(None, [], 101 + i)

    ball = _make_ball(200, 300, frame_idx=110)
    player = _make_player(5, 205, 310, frame_idx=110, team=None)
    event = detector.update(ball, [player], 110)

    assert event is not None
    assert event.rebound_type == ReboundType.DEFENSIVE


def test_events_accumulated(detector):
    """Detector accumulates events over multiple misses."""
    for shot_frame in (100, 300, 500):
        miss = _make_missed_shot(frame_idx=shot_frame, team="home")
        detector.on_missed_shot(miss)

        for i in range(ReboundDetector.MIN_DELAY_FRAMES):
            detector.update(None, [], shot_frame + 1 + i)

        ball = _make_ball(200, 300, frame_idx=shot_frame + 10)
        player = _make_player(5, 205, 310, frame_idx=shot_frame + 10, team="away")
        detector.update(ball, [player], shot_frame + 10)

    assert len(detector.events) == 3

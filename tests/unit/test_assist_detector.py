"""Tests for assist detection heuristic."""

import pytest

from app.events.assist_detector import AssistDetector, AssistEvent
from app.events.event_types import PossessionEvent, ShotEvent, ShotOutcome


def _make_shot(
    frame_idx: int,
    track_id: int,
    team: str,
    outcome: ShotOutcome = ShotOutcome.MADE,
    fps: float = 30.0,
) -> ShotEvent:
    return ShotEvent(
        frame_idx=frame_idx,
        timestamp_sec=frame_idx / fps,
        shooter_track_id=track_id,
        court_position=None,
        outcome=outcome,
        clip_start_frame=frame_idx - 30,
        clip_end_frame=frame_idx + 30,
        team=team,
    )


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
    return AssistDetector(fps=30.0)


def test_basic_assist(detector):
    """Teammate possession before made shot → assist."""
    possessions = [
        _make_possession(1, track_id=10, team="home", start_frame=80, end_frame=95),
    ]
    shot = _make_shot(frame_idx=100, track_id=20, team="home")

    event = detector.check(shot, possessions)

    assert event is not None
    assert event.assister_track_id == 10
    assert event.scorer_track_id == 20
    assert event.assister_team == "home"


def test_no_assist_on_missed_shot(detector):
    """Missed shots don't get assists."""
    possessions = [
        _make_possession(1, track_id=10, team="home", start_frame=80, end_frame=95),
    ]
    shot = _make_shot(frame_idx=100, track_id=20, team="home", outcome=ShotOutcome.MISSED)

    assert detector.check(shot, possessions) is None


def test_no_assist_different_team(detector):
    """Possession by opponent doesn't count as assist."""
    possessions = [
        _make_possession(1, track_id=10, team="away", start_frame=80, end_frame=95),
    ]
    shot = _make_shot(frame_idx=100, track_id=20, team="home")

    assert detector.check(shot, possessions) is None


def test_no_assist_same_player(detector):
    """Player can't assist themselves."""
    possessions = [
        _make_possession(1, track_id=20, team="home", start_frame=80, end_frame=95),
    ]
    shot = _make_shot(frame_idx=100, track_id=20, team="home")

    assert detector.check(shot, possessions) is None


def test_no_assist_outside_window(detector):
    """Possession too old (> 6 seconds) → no assist."""
    # 6 seconds at 30fps = 180 frames
    possessions = [
        _make_possession(1, track_id=10, team="home", start_frame=0, end_frame=10),
    ]
    shot = _make_shot(frame_idx=300, track_id=20, team="home")  # 10 sec later

    assert detector.check(shot, possessions) is None


def test_no_assist_no_team(detector):
    """No assist when shooter has no team."""
    possessions = [
        _make_possession(1, track_id=10, team="home", start_frame=80, end_frame=95),
    ]
    shot = _make_shot(frame_idx=100, track_id=20, team="home")
    shot.team = None

    assert detector.check(shot, possessions) is None


def test_most_recent_assist(detector):
    """Multiple valid possessions → most recent one gets assist."""
    possessions = [
        _make_possession(1, track_id=10, team="home", start_frame=60, end_frame=70),
        _make_possession(2, track_id=15, team="home", start_frame=80, end_frame=95),
    ]
    shot = _make_shot(frame_idx=100, track_id=20, team="home")

    event = detector.check(shot, possessions)

    assert event is not None
    assert event.assister_track_id == 15  # most recent


def test_skip_shooters_own_possession(detector):
    """Skip the possession that resulted in the shot (same player, result='shot')."""
    possessions = [
        _make_possession(1, track_id=10, team="home", start_frame=60, end_frame=70),
        _make_possession(2, track_id=20, team="home", start_frame=80, end_frame=100, result="shot"),
    ]
    shot = _make_shot(frame_idx=100, track_id=20, team="home")

    event = detector.check(shot, possessions)

    assert event is not None
    assert event.assister_track_id == 10  # skipped shooter's own


def test_events_accumulated(detector):
    """Multiple assists accumulate in detector.events."""
    poss1 = _make_possession(1, track_id=10, team="home", start_frame=80, end_frame=95)
    shot1 = _make_shot(frame_idx=100, track_id=20, team="home")

    poss2 = _make_possession(2, track_id=30, team="away", start_frame=280, end_frame=295)
    shot2 = _make_shot(frame_idx=300, track_id=40, team="away")

    detector.check(shot1, [poss1])
    detector.check(shot2, [poss1, poss2])

    assert len(detector.events) == 2

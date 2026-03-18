"""Tests for possession tracking."""

from app.events.possession import PossessionTracker
from app.tracking.tracker import TrackedPlayer
from app.vision.detection_types import BoundingBox, Detection


def _make_ball(x, y, frame_idx):
    return Detection(
        bbox=BoundingBox(x - 10, y - 10, x + 10, y + 10),
        confidence=0.8, class_id=32, class_name="sports ball",
        frame_idx=frame_idx,
    )


def _make_player(track_id, x, y, frame_idx):
    return TrackedPlayer(
        track_id=track_id,
        bbox=BoundingBox(x - 25, y - 60, x + 25, y + 60),
        frame_idx=frame_idx, is_confirmed=True,
    )


def test_assigns_possession_to_closest_player():
    tracker = PossessionTracker(fps=30.0)

    player1 = _make_player(1, 100, 200, 0)
    player2 = _make_player(2, 400, 200, 0)
    ball = _make_ball(120, 200, 0)  # close to player 1

    # Run enough frames to establish possession
    for i in range(10):
        tracker.update(ball, [player1, player2], i)

    assert tracker._current_possessor == 1


def test_emits_event_on_possession_change():
    tracker = PossessionTracker(fps=30.0)

    player1 = _make_player(1, 100, 200, 0)
    player2 = _make_player(2, 400, 200, 0)

    # Player 1 has ball for enough frames
    for i in range(10):
        ball = _make_ball(120, 200, i)
        tracker.update(ball, [player1, player2], i)

    # Ball moves to player 2
    ball_near_p2 = _make_ball(410, 200, 10)
    event = tracker.update(ball_near_p2, [player1, player2], 10)

    # Should get a possession end event for player 1
    assert event is not None
    assert event.player_track_id == 1
    assert event.result == "turnover"


def test_no_event_on_short_possession():
    tracker = PossessionTracker(fps=30.0)

    player1 = _make_player(1, 100, 200, 0)
    ball = _make_ball(120, 200, 0)

    # Only 1 frame (below MIN_POSSESSION_FRAMES=2)
    tracker.update(ball, [player1], 0)

    # Ball disappears
    event = tracker.update(None, [player1], 1)
    assert event is None  # too short to count


def test_end_possession_on_shot():
    tracker = PossessionTracker(fps=30.0)
    player1 = _make_player(1, 100, 200, 0)
    ball = _make_ball(120, 200, 0)

    for i in range(10):
        tracker.update(ball, [player1], i)

    event = tracker.end_possession_on_shot(10)
    assert event is not None
    assert event.result == "shot"

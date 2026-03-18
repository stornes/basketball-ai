"""Tests for the three-state possession state machine."""

from app.events.possession_state import (
    BallState,
    FLIGHT_SPEED_THRESHOLD_PX,
    PossessionStateMachine,
)
from app.events.rebound_detector import ReboundDetector
from app.events.event_types import ShotEvent, ShotOutcome


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _player(track_id: int, x: float, y: float, team: str = "team_a") -> dict:
    return {"track_id": track_id, "team": team, "bbox_center": (x, y)}


def _make_missed_shot(frame_idx: int = 100) -> ShotEvent:
    return ShotEvent(
        frame_idx=frame_idx,
        timestamp_sec=frame_idx / 30.0,
        shooter_track_id=1,
        court_position=None,
        outcome=ShotOutcome.MISSED,
        clip_start_frame=frame_idx - 30,
        clip_end_frame=frame_idx + 30,
        team="team_a",
    )


def _make_made_shot(frame_idx: int = 100) -> ShotEvent:
    return ShotEvent(
        frame_idx=frame_idx,
        timestamp_sec=frame_idx / 30.0,
        shooter_track_id=1,
        court_position=None,
        outcome=ShotOutcome.MADE,
        clip_start_frame=frame_idx - 30,
        clip_end_frame=frame_idx + 30,
        team="team_a",
    )


# ---------------------------------------------------------------------------
# Test 1: Ball near one player → PLAYER_CONTROL
# ---------------------------------------------------------------------------

def test_ball_near_one_player_is_player_control():
    psm = PossessionStateMachine(fps=30.0, proximity_threshold_px=80.0)
    players = [_player(1, 100.0, 200.0)]
    ball_pos = (110.0, 205.0)  # 11px away — well within threshold

    state = psm.update(frame_idx=1, ball_pos=ball_pos, players=players)

    assert state == BallState.PLAYER_CONTROL
    assert psm.controlling_player == 1


# ---------------------------------------------------------------------------
# Test 2: Ball moves rapidly away → FLIGHT
# ---------------------------------------------------------------------------

def test_ball_moves_rapidly_away_is_flight():
    psm = PossessionStateMachine(fps=30.0, proximity_threshold_px=80.0)
    players = [_player(1, 100.0, 200.0)]

    # Frame 0: ball near player → PLAYER_CONTROL
    psm.update(frame_idx=0, ball_pos=(110.0, 205.0), players=players)

    # Frame 1: ball has jumped far — velocity >> threshold
    velocity_px = FLIGHT_SPEED_THRESHOLD_PX * 3  # well above threshold
    fast_ball = (110.0 + velocity_px, 205.0)
    state = psm.update(frame_idx=1, ball_pos=fast_ball, players=players)

    assert state == BallState.FLIGHT


# ---------------------------------------------------------------------------
# Test 3: Ball near multiple players equally → LOOSE_BALL
# ---------------------------------------------------------------------------

def test_ball_near_multiple_players_equally_is_loose_ball():
    psm = PossessionStateMachine(fps=30.0, proximity_threshold_px=80.0)

    # Two players equidistant from ball
    players = [
        _player(1, 50.0, 200.0, team="team_a"),
        _player(2, 150.0, 200.0, team="team_b"),
    ]
    ball_pos = (100.0, 200.0)  # equidistant: 50px each

    state = psm.update(frame_idx=1, ball_pos=ball_pos, players=players)

    assert state == BallState.LOOSE_BALL


# ---------------------------------------------------------------------------
# Test 4: FLIGHT + MADE shot → no rebound trigger
# ---------------------------------------------------------------------------

def test_flight_plus_made_shot_no_rebound_trigger():
    detector = ReboundDetector(fps=30.0)
    made = _make_made_shot(frame_idx=100)

    # With ball_state=FLIGHT, a MADE shot should NOT register (outcome check first)
    detector.on_missed_shot(made, ball_state=BallState.FLIGHT)

    # pending_miss should remain None because outcome is MADE
    assert detector._pending_miss is None


# ---------------------------------------------------------------------------
# Test 5: FLIGHT + MISSED shot → rebound detection triggered
# ---------------------------------------------------------------------------

def test_flight_plus_missed_shot_rebound_triggered():
    detector = ReboundDetector(fps=30.0)
    miss = _make_missed_shot(frame_idx=100)

    detector.on_missed_shot(miss, ball_state=BallState.FLIGHT)

    # Should be registered
    assert detector._pending_miss is not None
    assert detector._pending_miss.frame_idx == 100


# ---------------------------------------------------------------------------
# Test 6: Pass sequence PLAYER_CONTROL(A) → FLIGHT → PLAYER_CONTROL(B)
# ---------------------------------------------------------------------------

def test_pass_sequence_player_control_a_flight_player_control_b():
    psm = PossessionStateMachine(fps=30.0, proximity_threshold_px=80.0)

    player_a = _player(1, 100.0, 200.0, team="team_a")
    player_b = _player(2, 400.0, 200.0, team="team_a")

    # Player A has the ball for a few frames
    for i in range(3):
        state = psm.update(frame_idx=i, ball_pos=(110.0, 205.0), players=[player_a, player_b])
    assert state == BallState.PLAYER_CONTROL
    assert psm.controlling_player == 1

    # Ball is released — fast movement → FLIGHT
    fast_x = 110.0 + FLIGHT_SPEED_THRESHOLD_PX * 4
    state = psm.update(frame_idx=3, ball_pos=(fast_x, 200.0), players=[player_a, player_b])
    assert state == BallState.FLIGHT

    # Ball travels multiple frames toward Player B (still fast, still FLIGHT)
    state = psm.update(frame_idx=4, ball_pos=(410.0, 200.0), players=[player_a, player_b])
    # May still be FLIGHT here depending on velocity; that is correct behaviour.
    # Force ball to "land" — tiny movement from previous position → below threshold.
    state = psm.update(frame_idx=5, ball_pos=(411.0, 200.0), players=[player_a, player_b])
    assert state == BallState.PLAYER_CONTROL
    assert psm.controlling_player == 2


# ---------------------------------------------------------------------------
# Test 7: Contested rebound: FLIGHT → LOOSE_BALL → PLAYER_CONTROL
# ---------------------------------------------------------------------------

def test_contested_rebound_flight_loose_ball_player_control():
    psm = PossessionStateMachine(fps=30.0, proximity_threshold_px=80.0)

    # Ball starts in flight (fast movement from rest)
    # We seed _prev_ball_pos via an initial update
    psm.update(frame_idx=0, ball_pos=(200.0, 100.0), players=[])

    # Fast descent → FLIGHT
    fast_ball = (200.0 + FLIGHT_SPEED_THRESHOLD_PX * 3, 200.0)
    state = psm.update(frame_idx=1, ball_pos=fast_ball, players=[])
    assert state == BallState.FLIGHT

    # Two players contest — equidistant → LOOSE_BALL
    players_contesting = [
        _player(3, fast_ball[0] - 40.0, fast_ball[1], "team_b"),
        _player(4, fast_ball[0] + 40.0, fast_ball[1], "team_a"),
    ]
    # Ball slows down (landed) — no fast velocity this frame
    landed_ball = (fast_ball[0] + 1.0, fast_ball[1])  # tiny movement
    state = psm.update(frame_idx=2, ball_pos=landed_ball, players=players_contesting)
    assert state == BallState.LOOSE_BALL

    # Player 3 wins — moves much closer, player 4 backs off
    winning_players = [
        _player(3, landed_ball[0] + 10.0, landed_ball[1], "team_b"),
        _player(4, landed_ball[0] + 200.0, landed_ball[1], "team_a"),  # far away
    ]
    state = psm.update(frame_idx=3, ball_pos=landed_ball, players=winning_players)
    assert state == BallState.PLAYER_CONTROL
    assert psm.controlling_player == 3


# ---------------------------------------------------------------------------
# Test 8: Ball not detected → maintain last state with decay
# ---------------------------------------------------------------------------

def test_ball_not_detected_maintains_last_state_with_decay():
    from app.events.possession_state import STATE_DECAY_FRAMES

    psm = PossessionStateMachine(fps=30.0, proximity_threshold_px=80.0)

    # Establish PLAYER_CONTROL
    player = _player(1, 100.0, 200.0)
    psm.update(frame_idx=0, ball_pos=(110.0, 200.0), players=[player])
    assert psm.state == BallState.PLAYER_CONTROL

    # Ball disappears — state should be maintained for STATE_DECAY_FRAMES
    for i in range(1, STATE_DECAY_FRAMES + 1):
        state = psm.update(frame_idx=i, ball_pos=None, players=[player])
        assert state == BallState.PLAYER_CONTROL, (
            f"Expected PLAYER_CONTROL at decay frame {i}, got {state}"
        )

    # After decay window, state should become UNKNOWN
    state = psm.update(
        frame_idx=STATE_DECAY_FRAMES + 1, ball_pos=None, players=[player]
    )
    assert state == BallState.UNKNOWN


# ---------------------------------------------------------------------------
# Test: state machine gating with non-FLIGHT state does not register rebound
# ---------------------------------------------------------------------------

def test_loose_ball_state_does_not_register_rebound():
    """When ball_state is LOOSE_BALL (not FLIGHT), on_missed_shot is blocked."""
    detector = ReboundDetector(fps=30.0)
    miss = _make_missed_shot(frame_idx=100)

    detector.on_missed_shot(miss, ball_state=BallState.LOOSE_BALL)

    assert detector._pending_miss is None


def test_no_ball_state_falls_back_to_original_behaviour():
    """Without ball_state, on_missed_shot behaves as before (outcome-only check)."""
    detector = ReboundDetector(fps=30.0)
    miss = _make_missed_shot(frame_idx=100)

    detector.on_missed_shot(miss)  # no ball_state

    assert detector._pending_miss is not None

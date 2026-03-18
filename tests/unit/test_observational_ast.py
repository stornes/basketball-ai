"""Tests for observational AST: ball-in-hands, pass detection, assist attribution."""

from app.events.assist_detector import AssistDetector, AssistEvent
from app.events.ball_possession import BallInHandsDetector, PossessionTransition
from app.events.event_types import (
    PassEvent, PossessionEvent, ShotEvent, ShotOutcome,
)
from app.events.pass_detector import PassDetector
from app.tracking.tracker import TrackedPlayer
from app.vision.detection_types import BoundingBox, Detection


# ── Helpers ──

def _make_ball(x, y, frame_idx):
    return Detection(
        bbox=BoundingBox(x - 10, y - 10, x + 10, y + 10),
        confidence=0.8, class_id=32, class_name="sports ball",
        frame_idx=frame_idx,
    )


def _make_player(track_id, x, y, frame_idx, team=None):
    return TrackedPlayer(
        track_id=track_id,
        bbox=BoundingBox(x - 25, y - 60, x + 25, y + 60),
        frame_idx=frame_idx, is_confirmed=True, team=team,
    )


def _make_shot(frame_idx, track_id, outcome=ShotOutcome.MADE,
               team=None, jersey_number=None):
    return ShotEvent(
        frame_idx=frame_idx,
        timestamp_sec=frame_idx / 30.0,
        shooter_track_id=track_id,
        court_position=None,
        outcome=outcome,
        clip_start_frame=max(0, frame_idx - 45),
        clip_end_frame=frame_idx + 45,
        team=team,
        jersey_number=jersey_number,
    )


# ── Ball-in-Hands Tests ──

class TestBallInHands:
    """ISC-4,5,6,7,8: BallInHandsDetector with bbox overlap."""

    def test_ball_inside_upper_body_returns_holder(self):
        """Ball center inside player upper-body bbox → possession detected."""
        det = BallInHandsDetector(frame_width=1920)
        # Player at (100, 200) with bbox from (75, 140) to (125, 260)
        player = _make_player(1, 100, 200, 0)
        # Ball at (100, 160) — inside upper-body region
        # Upper body: y from 140 to 140+120*0.6=212, x from 75+50*0.2=85 to 125-50*0.2=115
        ball = _make_ball(100, 160, 0)

        det.update(ball, [player], 0)
        det.update(ball, [player], 1)  # second frame confirms

        assert det.current_holder == 1

    def test_ball_outside_player_bbox_returns_none(self):
        """Ball far from any player → no possession holder."""
        det = BallInHandsDetector(frame_width=1920)
        player = _make_player(1, 100, 200, 0)
        # Ball at (500, 500) — far from player
        ball = _make_ball(500, 500, 0)

        det.update(ball, [player], 0)
        det.update(ball, [player], 1)

        assert det.current_holder is None

    def test_ball_in_lower_body_not_counted(self):
        """Ball in lower body (legs) region → not in hands."""
        det = BallInHandsDetector(frame_width=1920)
        # Player bbox: (75, 140) to (125, 260). Height=120.
        # Upper body ends at y=140+120*0.6=212.
        player = _make_player(1, 100, 200, 0)
        # Ball at (100, 250) — in lower body (legs)
        ball = _make_ball(100, 250, 0)

        det.update(ball, [player], 0)
        det.update(ball, [player], 1)

        assert det.current_holder is None

    def test_transition_emitted_on_holder_change(self):
        """Transition emitted when ball moves from Player A to Player B."""
        det = BallInHandsDetector(frame_width=1920)
        p1 = _make_player(1, 100, 200, 0)
        p2 = _make_player(2, 400, 200, 0)

        # Ball with player 1 for 3 frames (confirms possession)
        ball_p1 = _make_ball(100, 160, 0)
        det.update(ball_p1, [p1, p2], 0)
        det.update(ball_p1, [p1, p2], 1)
        det.update(ball_p1, [p1, p2], 2)

        # Ball moves to free space (no player)
        ball_free = _make_ball(250, 200, 3)
        t1 = det.update(ball_free, [p1, p2], 3)

        # Ball arrives at player 2
        ball_p2 = _make_ball(400, 160, 4)
        t2 = det.update(ball_p2, [p1, p2], 4)

        # Should have transitions in the list
        assert len(det.transitions) >= 2
        # First transition: ball enters player 1 (prev=None, new=1)
        assert det.transitions[0].new_holder_id == 1
        # Second transition: ball leaves player 1 (prev=1, new=None or new=2)
        release = det.transitions[1]
        assert release.prev_holder_id == 1

    def test_no_detection_no_change(self):
        """Ball detection None → no state change."""
        det = BallInHandsDetector(frame_width=1920)
        player = _make_player(1, 100, 200, 0)

        result = det.update(None, [player], 0)
        assert result is None
        assert det.current_holder is None


# ── Pass Detection Tests ──

class TestPassDetection:
    """ISC-9,10,11,12,13,14: PassDetector with trajectory tracking."""

    def test_pass_detected_between_players(self):
        """Ball moving from Player A to Player B → PassEvent emitted."""
        det = PassDetector(fps=30.0, frame_width=1920, frame_height=1080)

        # Player A releases ball
        release = PossessionTransition(
            frame_idx=0, prev_holder_id=1, new_holder_id=None,
            ball_x=100.0, ball_y=200.0,
        )
        det.on_transition(release)

        # Track ball in transit
        for i in range(1, 5):
            ball = _make_ball(100 + i * 80, 200, i)  # moving right
            det.track_ball(ball, i)

        # Player B receives ball (x=500 — 400px away, well over 96px threshold)
        arrival = PossessionTransition(
            frame_idx=5, prev_holder_id=None, new_holder_id=2,
            ball_x=500.0, ball_y=200.0,
        )
        result = det.on_transition(arrival)

        assert result is not None
        assert isinstance(result, PassEvent)
        assert result.from_player_track_id == 1
        assert result.to_player_track_id == 2
        assert result.distance_px > 96  # min threshold for 1920px
        assert result.ball_trajectory is not None
        assert len(result.ball_trajectory) >= 2

    def test_pass_rejected_transit_too_long(self):
        """Ball transit > 2 seconds → not a pass."""
        det = PassDetector(fps=30.0, frame_width=1920, frame_height=1080)

        release = PossessionTransition(
            frame_idx=0, prev_holder_id=1, new_holder_id=None,
            ball_x=100.0, ball_y=200.0,
        )
        det.on_transition(release)

        # 3 seconds later (90 frames at 30fps) — too long
        arrival = PossessionTransition(
            frame_idx=90, prev_holder_id=None, new_holder_id=2,
            ball_x=500.0, ball_y=200.0,
        )
        result = det.on_transition(arrival)

        assert result is None

    def test_pass_rejected_distance_too_small(self):
        """Ball travel < 5% of frame width → not a pass (bobble/handoff)."""
        det = PassDetector(fps=30.0, frame_width=1920, frame_height=1080)

        release = PossessionTransition(
            frame_idx=0, prev_holder_id=1, new_holder_id=None,
            ball_x=100.0, ball_y=200.0,
        )
        det.on_transition(release)

        # 50px away — below 96px threshold for 1920 width
        arrival = PossessionTransition(
            frame_idx=5, prev_holder_id=None, new_holder_id=2,
            ball_x=150.0, ball_y=200.0,
        )
        result = det.on_transition(arrival)

        assert result is None

    def test_same_player_release_receive_not_pass(self):
        """Ball leaves and returns to same player → not a pass."""
        det = PassDetector(fps=30.0, frame_width=1920, frame_height=1080)

        release = PossessionTransition(
            frame_idx=0, prev_holder_id=1, new_holder_id=None,
            ball_x=100.0, ball_y=200.0,
        )
        det.on_transition(release)

        arrival = PossessionTransition(
            frame_idx=5, prev_holder_id=None, new_holder_id=1,  # same player
            ball_x=500.0, ball_y=200.0,
        )
        result = det.on_transition(arrival)

        assert result is None

    def test_pass_trajectory_stored(self):
        """PassEvent stores ball centroid path."""
        det = PassDetector(fps=30.0, frame_width=1920, frame_height=1080)

        release = PossessionTransition(
            frame_idx=0, prev_holder_id=1, new_holder_id=None,
            ball_x=100.0, ball_y=200.0,
        )
        det.on_transition(release)

        # Track ball during transit
        for i in range(1, 4):
            ball = _make_ball(100 + i * 150, 200, i)
            det.track_ball(ball, i)

        arrival = PossessionTransition(
            frame_idx=4, prev_holder_id=None, new_holder_id=2,
            ball_x=550.0, ball_y=200.0,
        )
        result = det.on_transition(arrival)

        assert result is not None
        # Trajectory: release + 3 transit + arrival = 5 points
        assert len(result.ball_trajectory) == 5
        # First point is release position
        assert result.ball_trajectory[0] == (100.0, 200.0, 0)
        # Last point is arrival position
        assert result.ball_trajectory[-1] == (550.0, 200.0, 4)

    def test_resolution_scaled_distance(self):
        """At 4K (3840px), min distance is 192px vs 96px at 1920."""
        det_4k = PassDetector(fps=30.0, frame_width=3840, frame_height=2160)
        det_1080 = PassDetector(fps=30.0, frame_width=1920, frame_height=1080)

        assert det_4k.min_distance_px == 192.0
        assert det_1080.min_distance_px == 96.0

    def test_lob_pass_classification(self):
        """Ball that rises significantly during transit → lob pass."""
        det = PassDetector(fps=30.0, frame_width=1920, frame_height=1080)

        release = PossessionTransition(
            frame_idx=0, prev_holder_id=1, new_holder_id=None,
            ball_x=100.0, ball_y=500.0,
        )
        det.on_transition(release)

        # Ball goes high (y decreases significantly) during transit
        for i, y in enumerate([400, 300, 250, 300, 400], start=1):
            ball = _make_ball(100 + i * 80, y, i)
            det.track_ball(ball, i)

        arrival = PossessionTransition(
            frame_idx=6, prev_holder_id=None, new_holder_id=2,
            ball_x=580.0, ball_y=500.0,
        )
        result = det.on_transition(arrival)

        assert result is not None
        assert result.pass_type == "lob"

    def test_pending_pass_expires(self):
        """If ball is tracked too long without arrival, pending pass expires."""
        det = PassDetector(fps=30.0, frame_width=1920, frame_height=1080)

        release = PossessionTransition(
            frame_idx=0, prev_holder_id=1, new_holder_id=None,
            ball_x=100.0, ball_y=200.0,
        )
        det.on_transition(release)

        # Track ball for > 2 seconds (> 60 frames at 30fps)
        for i in range(1, 70):
            ball = _make_ball(100 + i * 5, 200, i)
            det.track_ball(ball, i)

        # After expiry, pending should be None
        assert det._pending is None


# ── Assist Attribution Tests ──

class TestAssistAttribution:
    """ISC-15,16,17: AssistDetector with pass-based and proximity fallback."""

    def test_assist_via_pass_event(self):
        """Assist attributed via observed PassEvent (Tier 1)."""
        detector = AssistDetector(fps=30.0)

        # Pass from player 1 to player 2 at t=3.0s
        pass_events = [
            PassEvent(
                frame_idx=90, timestamp_sec=3.0,
                from_player_track_id=1, to_player_track_id=2,
                from_team="home", to_team="home",
                pass_type="chest", distance_px=400.0,
            ),
        ]

        # Player 2 scores at t=4.0s (within 6s window)
        shot = _make_shot(120, track_id=2, outcome=ShotOutcome.MADE, team="home")

        result = detector.check(shot, [], pass_events=pass_events)

        assert result is not None
        assert result.assister_track_id == 1
        assert result.scorer_track_id == 2
        assert result.source == "pass"

    def test_assist_fallback_to_proximity(self):
        """No PassEvents → falls back to proximity-based assist (Tier 2)."""
        detector = AssistDetector(fps=30.0)

        # No pass events — only possession events
        possessions = [
            PossessionEvent(
                possession_id=1, player_track_id=1, team="home",
                start_frame=60, end_frame=90,
                start_time=2.0, end_time=3.0,
                result="turnover",
            ),
        ]

        # Player 2 scores at t=4.0s
        shot = _make_shot(120, track_id=2, outcome=ShotOutcome.MADE, team="home")

        result = detector.check(shot, possessions, pass_events=None)

        assert result is not None
        assert result.assister_track_id == 1
        assert result.source == "proximity"

    def test_no_assist_on_missed_shot(self):
        """Missed shots don't get assists."""
        detector = AssistDetector(fps=30.0)

        pass_events = [
            PassEvent(
                frame_idx=90, timestamp_sec=3.0,
                from_player_track_id=1, to_player_track_id=2,
                from_team="home", to_team="home",
            ),
        ]

        shot = _make_shot(120, track_id=2, outcome=ShotOutcome.MISSED, team="home")
        result = detector.check(shot, [], pass_events=pass_events)

        assert result is None

    def test_pass_from_other_team_not_assist(self):
        """Pass from opposing team player → not an assist."""
        detector = AssistDetector(fps=30.0)

        pass_events = [
            PassEvent(
                frame_idx=90, timestamp_sec=3.0,
                from_player_track_id=1, to_player_track_id=2,
                from_team="away", to_team="home",  # cross-team
            ),
        ]

        shot = _make_shot(120, track_id=2, outcome=ShotOutcome.MADE, team="home")
        result = detector.check(shot, [], pass_events=pass_events)

        assert result is None

    def test_pass_outside_window_not_assist(self):
        """Pass > 6 seconds before shot → not an assist."""
        detector = AssistDetector(fps=30.0)

        # Pass at t=0s, shot at t=10s → 10s gap > 6s window
        pass_events = [
            PassEvent(
                frame_idx=0, timestamp_sec=0.0,
                from_player_track_id=1, to_player_track_id=2,
                from_team="home", to_team="home",
            ),
        ]

        shot = _make_shot(300, track_id=2, outcome=ShotOutcome.MADE, team="home")
        result = detector.check(shot, [], pass_events=pass_events)

        assert result is None

    def test_pass_preferred_over_proximity(self):
        """When both pass and proximity could match, pass wins (Tier 1 first)."""
        detector = AssistDetector(fps=30.0)

        # Pass from player 3 to player 2
        pass_events = [
            PassEvent(
                frame_idx=90, timestamp_sec=3.0,
                from_player_track_id=3, to_player_track_id=2,
                from_team="home", to_team="home",
            ),
        ]

        # Proximity would match player 1
        possessions = [
            PossessionEvent(
                possession_id=1, player_track_id=1, team="home",
                start_frame=60, end_frame=85,
                start_time=2.0, end_time=2.83,
                result="turnover",
            ),
        ]

        shot = _make_shot(120, track_id=2, outcome=ShotOutcome.MADE, team="home")
        result = detector.check(shot, possessions, pass_events=pass_events)

        assert result is not None
        assert result.assister_track_id == 3  # pass-based, not player 1
        assert result.source == "pass"

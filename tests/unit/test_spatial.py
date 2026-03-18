"""Tests for the spatial utility module — court_distance with/without homography.

TDD: These tests were written before the implementation.
"""

import math

import numpy as np
import pytest

from app.events.spatial import court_distance, project_point


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _identity_homography() -> np.ndarray:
    """3x3 identity matrix — projection returns the same pixel coords (scaled)."""
    return np.eye(3, dtype=np.float64)


def _scale_homography(scale: float) -> np.ndarray:
    """Homography that uniformly scales coordinates by `scale`."""
    H = np.eye(3, dtype=np.float64)
    H[0, 0] = scale
    H[1, 1] = scale
    return H


# ---------------------------------------------------------------------------
# project_point
# ---------------------------------------------------------------------------

class TestProjectPoint:
    def test_identity_projection_returns_same_coords(self):
        H = _identity_homography()
        result = project_point((100.0, 200.0), H)
        assert result is not None
        assert abs(result[0] - 100.0) < 1e-6
        assert abs(result[1] - 200.0) < 1e-6

    def test_scale_homography_scales_coords(self):
        H = _scale_homography(0.1)  # 1px = 0.1 court units
        result = project_point((500.0, 300.0), H)
        assert result is not None
        assert abs(result[0] - 50.0) < 1e-4
        assert abs(result[1] - 30.0) < 1e-4

    def test_none_homography_returns_none(self):
        result = project_point((100.0, 200.0), None)
        assert result is None

    def test_origin_projects_to_origin_on_identity(self):
        H = _identity_homography()
        result = project_point((0.0, 0.0), H)
        assert result is not None
        assert abs(result[0]) < 1e-9
        assert abs(result[1]) < 1e-9


# ---------------------------------------------------------------------------
# court_distance — without homography (pixel fallback)
# ---------------------------------------------------------------------------

class TestCourtDistancePixelFallback:
    def test_pixel_distance_horizontal(self):
        dist = court_distance((0.0, 0.0), (30.0, 0.0))
        assert abs(dist - 30.0) < 1e-9

    def test_pixel_distance_vertical(self):
        dist = court_distance((100.0, 100.0), (100.0, 140.0))
        assert abs(dist - 40.0) < 1e-9

    def test_pixel_distance_diagonal(self):
        dist = court_distance((0.0, 0.0), (3.0, 4.0))
        assert abs(dist - 5.0) < 1e-9

    def test_zero_distance_same_point(self):
        dist = court_distance((50.0, 50.0), (50.0, 50.0))
        assert dist == 0.0

    def test_explicit_none_homography_uses_pixel(self):
        dist = court_distance((0.0, 0.0), (3.0, 4.0), homography=None)
        assert abs(dist - 5.0) < 1e-9


# ---------------------------------------------------------------------------
# court_distance — with homography (projected)
# ---------------------------------------------------------------------------

class TestCourtDistanceProjected:
    def test_scale_homography_converts_pixels_to_feet(self):
        # 10px/foot scale: a distance of 30px = 3 feet
        H = _scale_homography(0.1)  # multiply by 0.1 → pixel/10 = feet
        dist = court_distance((0.0, 0.0), (30.0, 0.0), homography=H)
        assert abs(dist - 3.0) < 1e-4

    def test_scale_homography_diagonal(self):
        # 10px/foot: 30px, 40px → 3ft, 4ft → 5ft
        H = _scale_homography(0.1)
        dist = court_distance((0.0, 0.0), (30.0, 40.0), homography=H)
        assert abs(dist - 5.0) < 1e-4

    def test_identity_homography_same_as_pixel(self):
        H = _identity_homography()
        dist_proj = court_distance((0.0, 0.0), (3.0, 4.0), homography=H)
        dist_pixel = court_distance((0.0, 0.0), (3.0, 4.0))
        assert abs(dist_proj - dist_pixel) < 1e-6

    def test_zero_distance_with_homography(self):
        H = _scale_homography(0.1)
        dist = court_distance((50.0, 50.0), (50.0, 50.0), homography=H)
        assert dist == 0.0


# ---------------------------------------------------------------------------
# court_distance — real-world-ish thresholds
# ---------------------------------------------------------------------------

class TestCourtDistanceThresholds:
    """Verify the physics of threshold choices make sense.

    FIBA half-court: ~47ft x 50ft. Possession proximity ~ 3-4ft.
    Side-view camera: typical player height ~100px → 1ft ≈ 15-20px.
    """

    def test_possession_proximity_4ft_at_10px_per_ft(self):
        """4ft at 10px/ft = 40px separation."""
        H = _scale_homography(0.1)  # 1 unit = 0.1 → 10px = 1ft
        dist = court_distance((0.0, 0.0), (40.0, 0.0), homography=H)
        assert abs(dist - 4.0) < 1e-4

    def test_assist_proximity_6ft_at_10px_per_ft(self):
        """6ft at 10px/ft = 60px separation."""
        H = _scale_homography(0.1)
        dist = court_distance((0.0, 0.0), (60.0, 0.0), homography=H)
        assert abs(dist - 6.0) < 1e-4


# ---------------------------------------------------------------------------
# PossessionStateMachine — accepts optional homography
# ---------------------------------------------------------------------------

class TestPossessionStateMachineWithHomography:
    """The state machine should work identically with court-unit threshold."""

    def test_player_control_with_court_threshold(self):
        from app.events.possession_state import PossessionStateMachine, BallState

        # Scale: 10px = 1 foot. proximity_threshold_ft=4ft → should trigger at 40px.
        H = _scale_homography(0.1)
        psm = PossessionStateMachine(
            fps=30.0,
            proximity_threshold_px=80.0,
            homography=H,
            proximity_threshold_ft=4.0,
        )

        players = [{"track_id": 1, "team": "team_a", "bbox_center": (100.0, 200.0)}]
        # Ball at 35px away → 3.5ft → within 4ft threshold
        ball_pos = (135.0, 200.0)
        state = psm.update(frame_idx=1, ball_pos=ball_pos, players=players)
        assert state == BallState.PLAYER_CONTROL

    def test_loose_ball_beyond_court_threshold(self):
        from app.events.possession_state import PossessionStateMachine, BallState

        H = _scale_homography(0.1)
        psm = PossessionStateMachine(
            fps=30.0,
            proximity_threshold_px=80.0,
            homography=H,
            proximity_threshold_ft=4.0,
        )

        players = [{"track_id": 1, "team": "team_a", "bbox_center": (100.0, 200.0)}]
        # Ball at 60px away → 6.0ft → beyond 4ft threshold → no nearby players
        ball_pos = (160.0, 200.0)
        state = psm.update(frame_idx=1, ball_pos=ball_pos, players=players)
        # No player near ball → LOOSE_BALL (not FLIGHT, not PLAYER_CONTROL)
        assert state == BallState.LOOSE_BALL

    def test_no_homography_falls_back_to_pixel(self):
        from app.events.possession_state import PossessionStateMachine, BallState

        # No homography: original pixel-based threshold still works
        psm = PossessionStateMachine(fps=30.0, proximity_threshold_px=80.0)
        players = [{"track_id": 1, "team": "team_a", "bbox_center": (100.0, 200.0)}]
        ball_pos = (150.0, 200.0)  # 50px away — within 80px threshold
        state = psm.update(frame_idx=1, ball_pos=ball_pos, players=players)
        assert state == BallState.PLAYER_CONTROL


# ---------------------------------------------------------------------------
# ReboundDetector — accepts optional homography
# ---------------------------------------------------------------------------

class TestReboundDetectorWithHomography:
    def _make_ball(self, x, y, frame_idx=0):
        from app.vision.detection_types import BoundingBox, Detection
        return Detection(
            bbox=BoundingBox(x - 10, y - 10, x + 10, y + 10),
            confidence=0.9,
            class_id=32,
            class_name="ball",
            frame_idx=frame_idx,
        )

    def _make_player(self, track_id, x, y, team="home", frame_idx=0):
        from app.tracking.tracker import TrackedPlayer
        from app.vision.detection_types import BoundingBox
        return TrackedPlayer(
            track_id=track_id,
            bbox=BoundingBox(x - 20, y - 40, x + 20, y + 40),
            frame_idx=frame_idx,
            is_confirmed=True,
            team=team,
        )

    def _make_missed_shot(self, frame_idx=100, team="home"):
        from app.events.event_types import ShotEvent, ShotOutcome
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

    def test_rebound_detected_with_court_threshold(self):
        from app.events.rebound_detector import ReboundDetector, ReboundType
        # Scale: 10px = 1ft. rebound_proximity_ft=5ft → detect at ≤50px.
        H = _scale_homography(0.1)
        detector = ReboundDetector(
            fps=30.0,
            homography=H,
            rebound_proximity_ft=5.0,
        )
        miss = self._make_missed_shot(frame_idx=100, team="home")
        detector.on_missed_shot(miss)

        for i in range(detector.MIN_DELAY_FRAMES):
            detector.update(None, [], 101 + i)

        ball = self._make_ball(200, 300, frame_idx=110)
        # 40px away → 4ft → within 5ft threshold
        player = self._make_player(5, 240, 300, frame_idx=110, team="away")
        event = detector.update(ball, [player], 110)

        assert event is not None
        assert event.rebound_type == ReboundType.DEFENSIVE

    def test_rebound_not_detected_beyond_court_threshold(self):
        from app.events.rebound_detector import ReboundDetector
        # Scale: 10px = 1ft. rebound_proximity_ft=3ft → detect at ≤30px.
        H = _scale_homography(0.1)
        detector = ReboundDetector(
            fps=30.0,
            homography=H,
            rebound_proximity_ft=3.0,
        )
        miss = self._make_missed_shot(frame_idx=100, team="home")
        detector.on_missed_shot(miss)

        for i in range(detector.MIN_DELAY_FRAMES):
            detector.update(None, [], 101 + i)

        ball = self._make_ball(200, 300, frame_idx=110)
        # 60px away → 6ft → beyond 3ft threshold
        player = self._make_player(5, 260, 300, frame_idx=110, team="away")
        event = detector.update(ball, [player], 110)

        assert event is None

    def test_no_homography_uses_pixel_threshold(self):
        from app.events.rebound_detector import ReboundDetector, ReboundType
        # Original pixel-only behaviour unchanged
        detector = ReboundDetector(fps=30.0)
        miss = self._make_missed_shot(frame_idx=100, team="home")
        detector.on_missed_shot(miss)

        for i in range(detector.MIN_DELAY_FRAMES):
            detector.update(None, [], 101 + i)

        ball = self._make_ball(200, 300, frame_idx=110)
        # 5px away — well within 100px default threshold
        player = self._make_player(5, 205, 300, frame_idx=110, team="away")
        event = detector.update(ball, [player], 110)

        assert event is not None
        assert event.rebound_type == ReboundType.DEFENSIVE

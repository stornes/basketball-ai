"""Tests for v1.7.0 manifesto fixes: FGA, MIN, AST, shot chart."""

import pandas as pd

from app.analytics.box_score import BoxScoreCompiler
from app.analytics.shot_chart import ShotChartGenerator
from app.events.event_types import ShotEvent, ShotOutcome, PossessionEvent
from app.events.possession import PossessionTracker
from app.events.shot_detector import ShotDetector
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


def _make_basket(x, y, frame_idx, width=40):
    return Detection(
        bbox=BoundingBox(x - width // 2, y - 15, x + width // 2, y + 15),
        confidence=0.9, class_id=1, class_name="basket",
        frame_idx=frame_idx,
    )


def _make_shot(frame_idx, track_id, outcome=ShotOutcome.ATTEMPTED,
               team=None, jersey_number=None, ball_x=None, ball_y=None):
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
        ball_x=ball_x,
        ball_y=ball_y,
    )


# ── MIN Fix Tests ──

class TestMinFix:
    """ISC-1,2,3: MIN calculation uses _track_to_key and sample_rate."""

    def test_min_nonzero_for_shooter_tracks(self):
        """Shooter tracks with jersey numbers produce non-zero MIN."""
        # Create tracks: track_id 100 and 200, both for jersey #4
        tracks = []
        for i in range(100):
            tracks.append(_make_player(100, 500, 500, i, team="home"))
        for i in range(100, 200):
            tracks.append(_make_player(200, 500, 500, i, team="home"))

        # Shot event maps track 100 to jersey #4
        shots = [_make_shot(50, 100, team="home", jersey_number=4)]

        compiler = BoxScoreCompiler()
        box = compiler.compile(shots, [], tracks, fps=30.0, sample_rate=3)

        # Player #4 should have MIN > 0
        home = box.home.players
        p4 = [p for p in home if p.jersey_number == 4]
        assert len(p4) == 1
        assert p4[0].min_played > 0, "MIN should be non-zero for shooter tracks"

    def test_min_includes_sample_rate(self):
        """MIN formula multiplies by sample_rate."""
        tracks = [_make_player(1, 500, 500, i, team="home") for i in range(60)]
        shots = [_make_shot(0, 1, team="home", jersey_number=1)]

        compiler = BoxScoreCompiler()
        box_sr1 = compiler.compile(shots, [], tracks, fps=30.0, sample_rate=1)
        box_sr3 = compiler.compile(shots, [], tracks, fps=30.0, sample_rate=3)

        p_sr1 = [p for p in box_sr1.home.players if p.jersey_number == 1][0]
        p_sr3 = [p for p in box_sr3.home.players if p.jersey_number == 1][0]

        # sample_rate=3 should produce 3× the minutes
        assert abs(p_sr3.min_played - p_sr1.min_played * 3) < 0.01


# ── FGA Fix Tests ──

class TestFgaFix:
    """ISC-4,5,6,7,8: Resolution thresholds, proximity gate, temporal dedup."""

    def test_resolution_scaled_threshold_4k(self):
        """At 4K (2160px height), MIN_VERTICAL_DISPLACEMENT ≈ 86."""
        det = ShotDetector(frame_height=2160, fps=30.0)
        assert det.MIN_VERTICAL_DISPLACEMENT == int(2160 * 0.04)  # 86
        assert det.MIN_VERTICAL_DISPLACEMENT > 80  # significantly more than old 40

    def test_resolution_scaled_threshold_1080p(self):
        """At 1080p, MIN_VERTICAL_DISPLACEMENT ≈ 43."""
        det = ShotDetector(frame_height=1080, fps=30.0)
        assert det.MIN_VERTICAL_DISPLACEMENT == int(1080 * 0.04)  # 43

    def test_cooldown_fps_based(self):
        """Cooldown is 5 seconds real time, scaled by sample_rate."""
        det = ShotDetector(frame_height=1080, fps=30.0, sample_rate=3)
        # 5.0 seconds * 30 fps / 3 sample_rate = 50 frames
        assert det.COOLDOWN_FRAMES == 50

    def test_in_shot_zone_rejects_midcourt(self):
        """Ball at midcourt (far from basket) should not trigger shot detection."""
        det = ShotDetector(frame_height=2160, fps=30.0, frame_width=3840)
        # Place basket at left side of frame
        basket = _make_basket(200, 1800, 0)
        det._last_basket_detection = basket
        # Ball at midcourt (x=1920, center of frame)
        assert not det._in_shot_zone(1920)

    def test_in_shot_zone_accepts_near_basket(self):
        """Ball near basket should pass the shot zone check."""
        det = ShotDetector(frame_height=2160, fps=30.0, frame_width=3840)
        basket = _make_basket(200, 1800, 0)
        det._last_basket_detection = basket
        # Ball near basket (x=300)
        assert det._in_shot_zone(300)

    def test_in_shot_zone_allows_without_basket(self):
        """Without basket detection, all positions pass (can't filter)."""
        det = ShotDetector(frame_height=2160, fps=30.0, frame_width=3840)
        assert det._in_shot_zone(1920)  # no basket → allow

    def test_temporal_dedup_suppresses_duplicate(self):
        """Second shot within 4s window is suppressed."""
        det = ShotDetector(frame_height=1080, fps=30.0, frame_width=1920,
                           sample_rate=1)
        players = [_make_player(1, 100, 200, 0)]

        # Create a trajectory that looks like a shot arc
        # Ball goes up (y decreasing) then down
        arc_positions = []
        for i in range(8):
            if i < 4:
                y = 400 - i * 20  # going up: 400, 380, 360, 340
            else:
                y = 340 + (i - 4) * 15  # coming down: 355, 370, 385, 400
            arc_positions.append(
                type('BP', (), {'x': 100.0, 'y': float(y), 'frame_idx': i})()
            )

        # First shot should succeed
        shot1 = det._create_shot_event(arc_positions, players, 8, 340.0)
        assert shot1 is not None

        # Second shot within 4s should be suppressed
        det._cooldown = 0  # reset cooldown
        shot2 = det._create_shot_event(arc_positions, players, 10, 340.0)
        assert shot2 is None, "Second shot within dedup window should be suppressed"


# ── AST Fix Tests ──

class TestAstFix:
    """ISC-9-13: PossessionTracker with team_map and scaled thresholds."""

    def test_team_map_assigns_correct_teams(self):
        """PossessionTracker uses team_map instead of parity heuristic."""
        team_map = {1: "home", 2: "away", 3: "home"}
        tracker = PossessionTracker(fps=30.0, team_map=team_map)

        player1 = _make_player(1, 100, 200, 0)
        ball = _make_ball(120, 200, 0)

        for i in range(5):
            tracker.update(ball, [player1], i)

        # End possession
        event = tracker.update(None, [player1], 5)
        assert event is not None
        assert event.team == "home"  # from team_map, not parity

    def test_team_map_fallback_to_parity(self):
        """Unknown track IDs fall back to parity heuristic."""
        team_map = {1: "home"}  # track 99 not in map
        tracker = PossessionTracker(fps=30.0, team_map=team_map)
        # track_id=99, odd → "team_b" via fallback
        assert tracker._assign_team(99) == "team_b"

    def test_scaled_distance_produces_more_possessions(self):
        """Wider threshold at 4K (192px) catches possessions the old 80px missed."""
        player = _make_player(1, 100, 200, 0)

        # Ball 150px from player — within 192px (4K) but outside 80px (old)
        ball_far = _make_ball(250, 200, 0)

        # Old-style tracker (simulated by small frame_width)
        narrow = PossessionTracker(fps=30.0, frame_width=1600)  # 80px threshold
        for i in range(5):
            narrow.update(ball_far, [player], i)
        narrow.update(None, [player], 5)

        # New 4K tracker
        wide = PossessionTracker(fps=30.0, frame_width=3840)  # 192px threshold
        for i in range(5):
            wide.update(ball_far, [player], i)
        wide.update(None, [player], 5)

        # Wide should detect possession, narrow should not
        assert len(narrow.events) == 0, "Narrow threshold should miss 150px possession"
        assert len(wide.events) >= 1, "Wide threshold should catch 150px possession"


# ── Shot Chart Fix Tests ──

class TestShotChartFix:
    """ISC-20,21: Basket-relative coords and half-court mirroring."""

    def test_basket_relative_near_basket(self):
        """Shot near near-basket (bottom of frame) maps to low court_y."""
        x, y = ShotChartGenerator.basket_relative_coords(
            ball_x=960, ball_y=900,  # ball near basket
            basket_det_cx=960, basket_det_cy=950,  # basket in lower half
            basket_det_width=40,
            frame_width=1920, frame_height=1080,
        )
        assert 20 < x < 30  # near center court width
        assert y < 10  # near basket (5.25 ft from baseline)

    def test_basket_relative_far_basket(self):
        """Shot at far-basket (top of frame) maps to high court_y."""
        x, y = ShotChartGenerator.basket_relative_coords(
            ball_x=960, ball_y=100,  # ball near far basket
            basket_det_cx=960, basket_det_cy=80,  # basket in upper half
            basket_det_width=30,
            frame_width=1920, frame_height=1080,
        )
        assert y > 80  # far end of court

    def test_half_court_mirroring(self):
        """Far-basket shots (court_y > 47) are mirrored to half-court."""
        df = pd.DataFrame({
            "court_x": [25.0, 25.0],
            "court_y": [10.0, 85.0],  # near basket, far basket
            "outcome": ["made", "made"],
        })
        gen = ShotChartGenerator()
        result = gen._normalize_to_half_court(df)

        # Near-basket shot unchanged
        assert result.iloc[0]["court_y"] == 10.0

        # Far-basket shot mirrored: 94 - 85 = 9
        assert abs(result.iloc[1]["court_y"] - 9.0) < 0.1
        # X also mirrored: 50 - 25 = 25
        assert abs(result.iloc[1]["court_x"] - 25.0) < 0.1

    def test_mirroring_preserves_near_basket(self):
        """Shots with court_y <= 47 are not modified."""
        df = pd.DataFrame({
            "court_x": [10.0, 40.0],
            "court_y": [5.0, 47.0],
            "outcome": ["made", "missed"],
        })
        gen = ShotChartGenerator()
        result = gen._normalize_to_half_court(df)
        assert result.iloc[0]["court_y"] == 5.0
        assert result.iloc[1]["court_y"] == 47.0

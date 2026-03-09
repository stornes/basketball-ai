"""Tests for game metrics computation."""

from app.analytics.metrics import GameMetrics
from app.events.event_types import ShotEvent, ShotOutcome


def test_shot_percentage_all_made(sample_shot_events):
    made_events = [
        ShotEvent(
            frame_idx=i, timestamp_sec=i / 30.0, shooter_track_id=1,
            court_position=(25.0, 20.0), outcome=ShotOutcome.MADE,
            clip_start_frame=0, clip_end_frame=50,
        )
        for i in range(5)
    ]
    metrics = GameMetrics(made_events, [])
    assert metrics.shot_percentage() == 1.0


def test_shot_percentage_none_made():
    missed = [
        ShotEvent(
            frame_idx=i, timestamp_sec=i / 30.0, shooter_track_id=1,
            court_position=None, outcome=ShotOutcome.MISSED,
            clip_start_frame=0, clip_end_frame=50,
        )
        for i in range(3)
    ]
    metrics = GameMetrics(missed, [])
    assert metrics.shot_percentage() == 0.0


def test_shot_percentage_empty():
    metrics = GameMetrics([], [])
    assert metrics.shot_percentage() == 0.0
    assert metrics.shots_attempted == 0
    assert metrics.shots_made == 0


def test_mixed_shot_percentage(sample_shot_events):
    # 2 made, 1 missed from fixtures
    metrics = GameMetrics(sample_shot_events, [])
    assert metrics.shots_attempted == 3
    assert metrics.shots_made == 2
    assert abs(metrics.shot_percentage() - 2 / 3) < 0.001


def test_player_stats(sample_shot_events, sample_possession_events):
    metrics = GameMetrics(sample_shot_events, sample_possession_events)
    stats = metrics.player_stats()
    assert len(stats) > 0
    # Player 1 has 2 shots, player 2 has 1
    p1 = next(s for s in stats if s["player_id"] == 1)
    assert p1["shots_attempted"] == 2
    assert p1["shots_made"] == 2


def test_summary_dict(sample_shot_events, sample_possession_events):
    metrics = GameMetrics(sample_shot_events, sample_possession_events)
    summary = metrics.to_summary_dict()
    assert "total_shots" in summary
    assert "fg_percentage" in summary
    assert "player_stats" in summary
    assert summary["total_shots"] == 3


def test_shots_dataframe(sample_shot_events):
    metrics = GameMetrics(sample_shot_events, [])
    df = metrics.shots_dataframe()
    assert len(df) == 3
    assert "court_x" in df.columns
    assert "outcome" in df.columns


def test_possessions_dataframe(sample_possession_events):
    metrics = GameMetrics([], sample_possession_events)
    df = metrics.possessions_dataframe()
    assert len(df) == 3
    assert "duration" in df.columns

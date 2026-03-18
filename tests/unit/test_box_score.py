"""Tests for box score data models, computed properties, and compiler."""

import pytest

from app.analytics.box_score import (
    BoxScoreCompiler,
    BoxScoreProfile,
    GameBoxScore,
    PlayerBoxScore,
    StatSource,
    TeamBoxScore,
)
from app.events.assist_detector import AssistEvent
from app.events.event_types import PossessionEvent, ShotEvent, ShotOutcome
from app.events.rebound_detector import ReboundEvent, ReboundType
from app.events.steal_detector import StealEvent


# ── Enum tests ──


def test_box_score_profile_values():
    assert BoxScoreProfile.PROFESSIONAL.value == "professional"
    assert BoxScoreProfile.YOUTH.value == "youth"


def test_stat_source_values():
    assert StatSource.DETECTED.value == "detected"
    assert StatSource.HEURISTIC.value == "heuristic"
    assert StatSource.MANUAL.value == "manual"
    assert StatSource.COMPUTED.value == "computed"
    assert StatSource.UNAVAILABLE.value == "unavailable"


# ── PlayerBoxScore tests ──


def test_player_defaults():
    p = PlayerBoxScore()
    assert p.fg == 0
    assert p.fga == 0
    assert p.pts == 0
    assert p.reb == 0
    assert p.team is None
    assert p.stat_sources == {}


def test_pts_computation():
    p = PlayerBoxScore(fg=5, three_p=2, ft=3)
    # (5-2)*2 + 2*3 + 3 = 6 + 6 + 3 = 15
    assert p.pts == 15


def test_pts_no_threes():
    p = PlayerBoxScore(fg=4, three_p=0, ft=0)
    assert p.pts == 8  # 4 * 2


def test_reb_computation():
    p = PlayerBoxScore(orb=3, drb=5)
    assert p.reb == 8


def test_two_p_computed():
    p = PlayerBoxScore(fg=7, fga=15, three_p=3, three_pa=8)
    assert p.two_p == 4
    assert p.two_pa == 7


def test_fg_pct():
    p = PlayerBoxScore(fg=3, fga=10)
    assert abs(p.fg_pct - 0.3) < 0.001


def test_fg_pct_zero_attempts():
    p = PlayerBoxScore(fg=0, fga=0)
    assert p.fg_pct == 0.0


def test_three_p_pct():
    p = PlayerBoxScore(three_p=2, three_pa=5)
    assert abs(p.three_p_pct - 0.4) < 0.001


def test_ft_pct():
    p = PlayerBoxScore(ft=7, fta=10)
    assert abs(p.ft_pct - 0.7) < 0.001


def test_ft_pct_zero():
    p = PlayerBoxScore(ft=0, fta=0)
    assert p.ft_pct == 0.0


# ── Youth KPI tests ──


def test_ast_to_ratio():
    p = PlayerBoxScore(ast=6, to=3)
    assert abs(p.ast_to_ratio - 2.0) < 0.001


def test_ast_to_ratio_zero_turnovers():
    p = PlayerBoxScore(ast=4, to=0)
    assert p.ast_to_ratio == 4.0


def test_defensive_activity_index():
    p = PlayerBoxScore(stl=3, deflections=2)
    assert p.defensive_activity_index == 5


def test_effort_plays():
    p = PlayerBoxScore(orb=2, stl=1, deflections=1, blk=1)
    assert p.effort_plays == 5


def test_reb_per_min():
    p = PlayerBoxScore(orb=3, drb=5, min_played=20.0)
    assert abs(p.reb_per_min - 0.4) < 0.001


def test_reb_per_min_zero():
    p = PlayerBoxScore(orb=3, drb=5, min_played=0.0)
    assert p.reb_per_min == 0.0


def test_impact_line():
    p = PlayerBoxScore(fg=5, three_p=1, ft=2, orb=1, drb=3, ast=4, stl=2, blk=1, to=2)
    # pts = (5-1)*2 + 1*3 + 2 = 8+3+2 = 13
    assert p.impact_line == "13 / 4 / 4 / 2 / 1 / 2"


# ── Serialisation tests ──


def test_player_round_trip():
    p = PlayerBoxScore(
        player_id=1, player_name="Test Player", jersey_number=23,
        team="home", fg=5, fga=10, three_p=2, three_pa=4,
        ft=3, fta=4, orb=1, drb=3, ast=4, to=2, stl=1, blk=1, pf=2,
        stat_sources={"fg": StatSource.DETECTED, "ast": StatSource.HEURISTIC},
    )
    d = p.to_dict()
    p2 = PlayerBoxScore.from_dict(d)
    assert p2.player_id == 1
    assert p2.player_name == "Test Player"
    assert p2.fg == 5
    assert p2.three_p == 2
    assert p2.stat_sources["fg"] == StatSource.DETECTED
    assert p2.pts == p.pts


def test_team_box_score_totals():
    t = TeamBoxScore(
        team_name="Test", team_key="home",
        players=[
            PlayerBoxScore(fg=3, fga=8, three_p=1, three_pa=3, to=2),
            PlayerBoxScore(fg=5, fga=12, three_p=2, three_pa=5, to=1),
        ],
    )
    assert t.total_fg == 8
    assert t.total_fga == 20
    assert t.total_three_p == 3
    assert t.total_to == 3
    assert abs(t.team_fg_pct - 0.4) < 0.001


def test_team_round_trip():
    t = TeamBoxScore(
        team_name="Home", team_key="home",
        players=[PlayerBoxScore(fg=3, fga=8)],
    )
    d = t.to_dict()
    t2 = TeamBoxScore.from_dict(d)
    assert t2.team_name == "Home"
    assert len(t2.players) == 1
    assert t2.players[0].fg == 3


def test_game_round_trip():
    game = GameBoxScore(
        home=TeamBoxScore(team_name="A", team_key="home", players=[
            PlayerBoxScore(fg=5, fga=10),
        ]),
        away=TeamBoxScore(team_name="B", team_key="away", players=[
            PlayerBoxScore(fg=3, fga=8),
        ]),
        profile=BoxScoreProfile.YOUTH,
        game_date="2026-03-14",
        venue="Test Arena",
    )
    d = game.to_dict()
    game2 = GameBoxScore.from_dict(d)
    assert game2.profile == BoxScoreProfile.YOUTH
    assert game2.game_date == "2026-03-14"
    assert game2.home.team_name == "A"
    assert game2.away.players[0].fg == 3


# ── BoxScoreCompiler tests ──


def test_compiler_counts_fg_fga():
    shots = [
        ShotEvent(frame_idx=100, timestamp_sec=3.0, shooter_track_id=1,
                  court_position=None, outcome=ShotOutcome.MADE,
                  clip_start_frame=0, clip_end_frame=50, team="home"),
        ShotEvent(frame_idx=200, timestamp_sec=6.0, shooter_track_id=1,
                  court_position=None, outcome=ShotOutcome.MISSED,
                  clip_start_frame=0, clip_end_frame=50, team="home"),
        ShotEvent(frame_idx=300, timestamp_sec=9.0, shooter_track_id=2,
                  court_position=None, outcome=ShotOutcome.MADE,
                  clip_start_frame=0, clip_end_frame=50, team="away"),
    ]
    compiler = BoxScoreCompiler()
    game = compiler.compile(shots, [], [], 30.0)
    # Player 1 (home): 1 FG, 2 FGA
    home_p1 = game.home.players[0]
    assert home_p1.fg == 1
    assert home_p1.fga == 2
    # Player 2 (away): 1 FG, 1 FGA
    away_p2 = game.away.players[0]
    assert away_p2.fg == 1
    assert away_p2.fga == 1


def test_compiler_counts_turnovers():
    possessions = [
        PossessionEvent(possession_id=1, player_track_id=1, team="home",
                        start_frame=0, end_frame=100, start_time=0, end_time=3,
                        result="turnover"),
        PossessionEvent(possession_id=2, player_track_id=1, team="home",
                        start_frame=110, end_frame=200, start_time=3.5, end_time=6,
                        result="shot"),
        PossessionEvent(possession_id=3, player_track_id=1, team="home",
                        start_frame=210, end_frame=300, start_time=7, end_time=10,
                        result="turnover"),
    ]
    compiler = BoxScoreCompiler()
    game = compiler.compile([], possessions, [], 30.0)
    home_p1 = game.home.players[0]
    assert home_p1.to == 2


def test_compiler_applies_three_point_classification():
    from app.events.three_point import ThreePointClassifier

    shots = [
        # Beyond arc (distance from basket > 23.75ft)
        ShotEvent(frame_idx=100, timestamp_sec=3.0, shooter_track_id=1,
                  court_position=(25.0, 30.0), outcome=ShotOutcome.MADE,
                  clip_start_frame=0, clip_end_frame=50, team="home"),
        # Inside arc (close to basket)
        ShotEvent(frame_idx=200, timestamp_sec=6.0, shooter_track_id=1,
                  court_position=(25.0, 10.0), outcome=ShotOutcome.MADE,
                  clip_start_frame=0, clip_end_frame=50, team="home"),
    ]
    classifier = ThreePointClassifier()
    compiler = BoxScoreCompiler(three_point_classifier=classifier)
    game = compiler.compile(shots, [], [], 30.0)
    p = game.home.players[0]
    assert p.fg == 2
    assert p.three_p == 1
    assert p.three_pa == 1
    assert p.two_p == 1


# ── Compiler: Phase 2 heuristic events ──


def test_compiler_rebounds():
    """Compiler integrates rebound events into ORB/DRB."""
    rebounds = [
        ReboundEvent(frame_idx=110, timestamp_sec=3.7, rebounder_track_id=5,
                      rebounder_team="home", shooter_track_id=1, shooter_team="away",
                      rebound_type=ReboundType.DEFENSIVE, shot_frame_idx=100),
        ReboundEvent(frame_idx=210, timestamp_sec=7.0, rebounder_track_id=5,
                      rebounder_team="home", shooter_track_id=5, shooter_team="home",
                      rebound_type=ReboundType.OFFENSIVE, shot_frame_idx=200),
        ReboundEvent(frame_idx=310, timestamp_sec=10.3, rebounder_track_id=10,
                      rebounder_team="away", shooter_track_id=5, shooter_team="home",
                      rebound_type=ReboundType.DEFENSIVE, shot_frame_idx=300),
    ]
    compiler = BoxScoreCompiler()
    game = compiler.compile([], [], [], 30.0, rebound_events=rebounds)

    p5 = game.home.players[0]
    assert p5.player_id == 5
    assert p5.drb == 1
    assert p5.orb == 1
    assert p5.reb == 2
    assert p5.stat_sources["orb"] == StatSource.HEURISTIC
    assert p5.stat_sources["drb"] == StatSource.HEURISTIC

    p10 = game.away.players[0]
    assert p10.drb == 1
    assert p10.orb == 0


def test_compiler_assists():
    """Compiler integrates assist events into AST."""
    assists = [
        AssistEvent(frame_idx=95, timestamp_sec=3.2, assister_track_id=10,
                    assister_team="home", scorer_track_id=20, scorer_team="home",
                    shot_frame_idx=100),
        AssistEvent(frame_idx=195, timestamp_sec=6.5, assister_track_id=10,
                    assister_team="home", scorer_track_id=20, scorer_team="home",
                    shot_frame_idx=200),
    ]
    compiler = BoxScoreCompiler()
    game = compiler.compile([], [], [], 30.0, assist_events=assists)

    p10 = game.home.players[0]
    assert p10.ast == 2
    assert p10.stat_sources["ast"] == StatSource.HEURISTIC


def test_compiler_steals():
    """Compiler integrates steal events into STL."""
    steals = [
        StealEvent(frame_idx=135, timestamp_sec=4.5, stealer_track_id=20,
                   stealer_team="away", victim_track_id=10, victim_team="home"),
    ]
    compiler = BoxScoreCompiler()
    game = compiler.compile([], [], [], 30.0, steal_events=steals)

    p20 = game.away.players[0]
    assert p20.stl == 1
    assert p20.stat_sources["stl"] == StatSource.HEURISTIC


def test_compiler_all_phase2_events():
    """Compiler handles all Phase 2 events together."""
    shots = [
        ShotEvent(frame_idx=100, timestamp_sec=3.3, shooter_track_id=20,
                  court_position=None, outcome=ShotOutcome.MADE,
                  clip_start_frame=70, clip_end_frame=130, team="home"),
        ShotEvent(frame_idx=200, timestamp_sec=6.7, shooter_track_id=30,
                  court_position=None, outcome=ShotOutcome.MISSED,
                  clip_start_frame=170, clip_end_frame=230, team="away"),
    ]
    rebounds = [
        ReboundEvent(frame_idx=215, timestamp_sec=7.2, rebounder_track_id=20,
                      rebounder_team="home", shooter_track_id=30, shooter_team="away",
                      rebound_type=ReboundType.DEFENSIVE, shot_frame_idx=200),
    ]
    assists = [
        AssistEvent(frame_idx=90, timestamp_sec=3.0, assister_track_id=10,
                    assister_team="home", scorer_track_id=20, scorer_team="home",
                    shot_frame_idx=100),
    ]
    steals = [
        StealEvent(frame_idx=250, timestamp_sec=8.3, stealer_track_id=10,
                   stealer_team="home", victim_track_id=30, victim_team="away"),
    ]
    compiler = BoxScoreCompiler()
    game = compiler.compile(
        shots, [], [], 30.0,
        rebound_events=rebounds,
        assist_events=assists,
        steal_events=steals,
    )

    # Player 20 (home): 1 FG, 1 FGA, 1 DRB
    p20 = [p for p in game.home.players if p.player_id == 20][0]
    assert p20.fg == 1
    assert p20.fga == 1
    assert p20.drb == 1

    # Player 10 (home): 1 AST, 1 STL
    p10 = [p for p in game.home.players if p.player_id == 10][0]
    assert p10.ast == 1
    assert p10.stl == 1

    # Player 30 (away): 0 FG, 1 FGA
    p30 = game.away.players[0]
    assert p30.fg == 0
    assert p30.fga == 1

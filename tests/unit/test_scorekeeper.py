"""Tests for scorekeeper merge system."""

import json
import tempfile
from pathlib import Path

import pytest

from app.analytics.box_score import GameBoxScore, PlayerBoxScore, StatSource, TeamBoxScore
from app.scoring.scorekeeper import (
    ScorekeeperData,
    ScorekeeperPlayerStats,
    merge_scorekeeper,
)


# ── Fixtures ──


def _make_player(
    jersey: int,
    team: str = "home",
    fg: int = 5,
    orb: int = 0,
    drb: int = 0,
    ast: int = 0,
    sources: dict[str, StatSource] | None = None,
) -> PlayerBoxScore:
    """Create a PlayerBoxScore with sensible defaults."""
    p = PlayerBoxScore(
        player_id=jersey,
        jersey_number=jersey,
        team=team,
        fg=fg,
        fga=fg + 2,
    )
    p.stat_sources = sources or {
        "fg": StatSource.DETECTED,
        "fga": StatSource.DETECTED,
        "orb": StatSource.UNAVAILABLE,
        "drb": StatSource.UNAVAILABLE,
        "ast": StatSource.UNAVAILABLE,
        "to": StatSource.UNAVAILABLE,
        "stl": StatSource.UNAVAILABLE,
        "blk": StatSource.UNAVAILABLE,
        "pf": StatSource.UNAVAILABLE,
        "min_played": StatSource.UNAVAILABLE,
        "plus_minus": StatSource.UNAVAILABLE,
    }
    if orb:
        p.orb = orb
    if drb:
        p.drb = drb
    if ast:
        p.ast = ast
    return p


def _make_game(*home_jerseys: int) -> GameBoxScore:
    """Create a GameBoxScore with home players at given jersey numbers."""
    home_players = [_make_player(j) for j in home_jerseys]
    return GameBoxScore(
        home=TeamBoxScore(team_name="Home", team_key="home", players=home_players),
        away=TeamBoxScore(team_name="Away", team_key="away", players=[]),
    )


def _make_scorekeeper_json(data: dict) -> Path:
    """Write scorekeeper data to a temp JSON file and return path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(data, f)
    f.close()
    return Path(f.name)


# ── Tests ──


class TestLoadScorekeeper:
    """Test loading scorekeeper data from JSON."""

    def test_load_valid_json(self):
        path = _make_scorekeeper_json({
            "home": {
                "team_name": "Hawks",
                "players": [
                    {"jersey": 4, "min": 30.0, "oreb": 3, "dreb": 7, "ast": 2},
                    {"jersey": 10, "min": 22.5, "stl": 3},
                ],
            },
            "away": {
                "team_name": "Eagles",
                "players": [
                    {"jersey": 6, "ast": 5, "to": 1},
                ],
            },
        })
        sk = ScorekeeperData.from_json(path)
        assert sk.home_team_name == "Hawks"
        assert len(sk.home_players) == 2
        assert sk.home_players[0].jersey == 4
        assert sk.home_players[0].oreb == 3
        assert sk.home_players[0].stl is None  # not provided
        assert len(sk.away_players) == 1
        assert sk.away_players[0].ast == 5

    def test_validation_rejects_zero_jersey(self):
        path = _make_scorekeeper_json({
            "home": {"players": [{"jersey": 0, "min": 10}]},
        })
        with pytest.raises(ValueError, match="Jersey number must be > 0"):
            ScorekeeperData.from_json(path)

    def test_validation_rejects_negative_min(self):
        path = _make_scorekeeper_json({
            "home": {"players": [{"jersey": 4, "min": -5}]},
        })
        with pytest.raises(ValueError, match="Minutes must be 0–60"):
            ScorekeeperData.from_json(path)

    def test_validation_rejects_negative_counting_stat(self):
        path = _make_scorekeeper_json({
            "home": {"players": [{"jersey": 4, "oreb": -1}]},
        })
        with pytest.raises(ValueError, match="oreb must be >= 0"):
            ScorekeeperData.from_json(path)


class TestMergeScorekeeper:
    """Test merge_scorekeeper function."""

    def test_merge_fills_unavailable_stats(self):
        """UNAVAILABLE stats should be overwritten by manual data."""
        game = _make_game(4, 10)
        sk = ScorekeeperData(
            home_players=[
                ScorekeeperPlayerStats(jersey=4, oreb=3, dreb=7, ast=2, to=4),
            ],
        )
        merge_scorekeeper(game, sk)

        p4 = next(p for p in game.home.players if p.jersey_number == 4)
        assert p4.orb == 3
        assert p4.drb == 7
        assert p4.ast == 2
        assert p4.to == 4

    def test_merge_preserves_detected_stats(self):
        """DETECTED stats must NEVER be overwritten."""
        game = _make_game(4)
        original_fg = game.home.players[0].fg  # 5, DETECTED

        sk = ScorekeeperData(
            home_players=[
                ScorekeeperPlayerStats(jersey=4, fg=10),  # tries to override
            ],
        )
        merge_scorekeeper(game, sk)

        p4 = game.home.players[0]
        assert p4.fg == original_fg  # preserved
        assert p4.stat_sources["fg"] == StatSource.DETECTED

    def test_merge_overwrites_heuristic_stats(self):
        """HEURISTIC stats should be overwritten by manual data."""
        game = _make_game(4)
        p4 = game.home.players[0]
        p4.orb = 1
        p4.stat_sources["orb"] = StatSource.HEURISTIC

        sk = ScorekeeperData(
            home_players=[
                ScorekeeperPlayerStats(jersey=4, oreb=5),
            ],
        )
        merge_scorekeeper(game, sk)

        assert p4.orb == 5
        assert p4.stat_sources["orb"] == StatSource.MANUAL

    def test_merge_creates_unmatched_player(self):
        """Players in scorekeeper but not in box score get new entries."""
        game = _make_game(4)
        assert len(game.home.players) == 1

        sk = ScorekeeperData(
            home_players=[
                ScorekeeperPlayerStats(jersey=99, min=12.0, ast=3, stl=2),
            ],
        )
        merge_scorekeeper(game, sk)

        assert len(game.home.players) == 2
        p99 = next(p for p in game.home.players if p.jersey_number == 99)
        assert p99.min_played == 12.0
        assert p99.ast == 3
        assert p99.stl == 2
        assert p99.stat_sources["ast"] == StatSource.MANUAL

    def test_merge_stat_sources_all_manual(self):
        """Every merged field must have StatSource.MANUAL."""
        game = _make_game(4)
        sk = ScorekeeperData(
            home_players=[
                ScorekeeperPlayerStats(
                    jersey=4, min=32.0, oreb=3, dreb=7, ast=2,
                    to=4, stl=1, blk=0, pf=2, plus_minus=8,
                ),
            ],
        )
        merge_scorekeeper(game, sk)

        p4 = game.home.players[0]
        for key in ("min_played", "orb", "drb", "ast", "to", "stl", "blk", "pf", "plus_minus"):
            assert p4.stat_sources[key] == StatSource.MANUAL, f"{key} not MANUAL"

    def test_partial_data_only_merges_provided_fields(self):
        """Scorekeeper with only some stats should leave others unchanged."""
        game = _make_game(4)
        sk = ScorekeeperData(
            home_players=[
                ScorekeeperPlayerStats(jersey=4, oreb=3),  # only oreb
            ],
        )
        merge_scorekeeper(game, sk)

        p4 = game.home.players[0]
        assert p4.orb == 3
        assert p4.stat_sources["orb"] == StatSource.MANUAL
        # Other stats remain untouched
        assert p4.drb == 0
        assert p4.stat_sources.get("drb") == StatSource.UNAVAILABLE

    def test_merge_away_team(self):
        """Merge works for away team players."""
        game = GameBoxScore(
            home=TeamBoxScore(team_name="Home", team_key="home", players=[]),
            away=TeamBoxScore(
                team_name="Away",
                team_key="away",
                players=[_make_player(6, team="away")],
            ),
        )
        sk = ScorekeeperData(
            away_players=[
                ScorekeeperPlayerStats(jersey=6, ast=5, to=1),
            ],
        )
        merge_scorekeeper(game, sk)

        p6 = game.away.players[0]
        assert p6.ast == 5
        assert p6.to == 1

    def test_plus_minus_can_be_negative(self):
        """plus_minus accepts negative values."""
        game = _make_game(4)
        sk = ScorekeeperData(
            home_players=[
                ScorekeeperPlayerStats(jersey=4, plus_minus=-12),
            ],
        )
        merge_scorekeeper(game, sk)
        assert game.home.players[0].plus_minus == -12


class TestScorekeeperRoundTrip:
    """Test serialisation round-trip."""

    def test_to_json_and_from_json(self):
        sk = ScorekeeperData(
            home_team_name="Hawks",
            home_players=[
                ScorekeeperPlayerStats(jersey=4, oreb=3, dreb=7),
            ],
            away_team_name="Eagles",
            away_players=[
                ScorekeeperPlayerStats(jersey=6, ast=5),
            ],
        )
        path = Path(tempfile.mktemp(suffix=".json"))
        sk.to_json(path)

        loaded = ScorekeeperData.from_json(path)
        assert loaded.home_team_name == "Hawks"
        assert loaded.home_players[0].jersey == 4
        assert loaded.home_players[0].oreb == 3
        assert loaded.away_players[0].ast == 5

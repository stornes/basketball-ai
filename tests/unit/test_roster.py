"""Tests for roster loading and player name lookup."""

import json
from pathlib import Path

from app.config.roster import Roster, RosterPlayer, load_roster


def test_player_name_lookup():
    roster = Roster(
        home_players=[
            RosterPlayer(number=4, name="Mesfin, Asier Fitsum", is_captain=True),
            RosterPlayer(number=7, name="Kahindo, Christelle"),
        ],
        away_players=[
            RosterPlayer(number=6, name="Grønneberg, Ulrik Holm", is_captain=True),
        ],
    )

    assert roster.player_name(4, "home") == "Mesfin, Asier Fitsum"
    assert roster.player_name(6, "away") == "Grønneberg, Ulrik Holm"
    assert roster.player_name(99) is None


def test_player_name_searches_both_teams():
    roster = Roster(
        home_players=[RosterPlayer(number=1, name="Home Player")],
        away_players=[RosterPlayer(number=23, name="Away Player")],
    )

    # No team specified — should search both
    assert roster.player_name(1) == "Home Player"
    assert roster.player_name(23) == "Away Player"


def test_team_name():
    roster = Roster(home_team_name="Lions", away_team_name="Tigers")
    assert roster.team_name("home") == "Lions"
    assert roster.team_name("away") == "Tigers"
    assert roster.team_name("other") == "other"


def test_load_roster_from_json(tmp_path):
    data = {
        "home": {
            "name": "Lions",
            "players": [
                {"number": 1, "name": "Player One", "captain": True},
                {"number": 5, "name": "Player Five"},
            ],
            "staff": ["Coach A"],
        },
        "away": {
            "name": "Tigers",
            "players": [
                {"number": 6, "name": "Player Six", "captain": True},
            ],
            "staff": ["Coach B"],
        },
    }

    path = tmp_path / "roster.json"
    path.write_text(json.dumps(data))

    roster = load_roster(path)
    assert roster.home_team_name == "Lions"
    assert roster.away_team_name == "Tigers"
    assert len(roster.home_players) == 2
    assert len(roster.away_players) == 1
    assert roster.home_players[0].is_captain is True
    assert roster.home_staff == ["Coach A"]
    assert roster.player_name(6, "away") == "Player Six"


def test_quarter_scores():
    # Actual game: Notodden Thunders 69 - EB-85 82
    # Per-quarter: 18-29, 20-12, 19-19, 12-22
    # Cumulative:  18-29, 38-41, 57-60, 69-82
    roster = Roster(
        home_team_name="Notodden Thunders (D)",
        away_team_name="EB-85",
        home_scores=[18, 38, 57, 69],
        away_scores=[29, 41, 60, 82],
    )

    assert roster.has_scores()
    qs = roster.quarter_scores()
    assert len(qs) == 4
    assert qs[0] == {"quarter": "Q1", "home": 18, "away": 29, "home_cumulative": 18, "away_cumulative": 29}
    assert qs[1]["home"] == 20  # 38 - 18
    assert qs[1]["away"] == 12  # 41 - 29
    assert qs[2]["home"] == 19  # 57 - 38
    assert qs[2]["away"] == 19  # 60 - 41
    assert qs[3]["home"] == 12  # 69 - 57
    assert qs[3]["away"] == 22  # 82 - 60
    assert qs[3]["home_cumulative"] == 69
    assert qs[3]["away_cumulative"] == 82


def test_no_scores():
    roster = Roster()
    assert not roster.has_scores()
    assert roster.quarter_scores() == []


def test_load_roster_with_scores(tmp_path):
    data = {
        "home": {
            "name": "Notodden Thunders (D)",
            "players": [],
            "scores_cumulative": [18, 38, 57, 69],
        },
        "away": {
            "name": "EB-85",
            "players": [],
            "scores_cumulative": [29, 41, 60, 82],
        },
    }

    path = tmp_path / "roster.json"
    path.write_text(json.dumps(data))

    roster = load_roster(path)
    assert roster.home_scores == [18, 38, 57, 69]
    assert roster.away_scores == [29, 41, 60, 82]
    assert roster.has_scores()

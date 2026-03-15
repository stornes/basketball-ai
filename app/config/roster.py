"""Roster management — map jersey numbers to player names.

Loads roster data from a JSON file and provides lookups for the pipeline
to replace track IDs with actual player names in reports and charts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RosterPlayer:
    number: int
    name: str
    is_captain: bool = False


@dataclass
class Roster:
    """Game roster with home and away team players and optional scores."""

    home_team_name: str = "Home"
    away_team_name: str = "Away"
    home_players: list[RosterPlayer] = field(default_factory=list)
    away_players: list[RosterPlayer] = field(default_factory=list)
    home_staff: list[str] = field(default_factory=list)
    away_staff: list[str] = field(default_factory=list)
    # Cumulative scores at end of each quarter [Q1, Q2, Q3, Q4, OT1, ...]
    home_scores: list[int] = field(default_factory=list)
    away_scores: list[int] = field(default_factory=list)

    def player_name(self, jersey_number: int, team: str | None = None) -> str | None:
        """Look up player name by jersey number.

        If team is provided, search only that team. Otherwise search both
        (home first, then away). Returns None if not found.
        """
        if team in ("home", None):
            for p in self.home_players:
                if p.number == jersey_number:
                    return p.name
        if team in ("away", None):
            for p in self.away_players:
                if p.number == jersey_number:
                    return p.name
        return None

    def team_name(self, team: str) -> str:
        """Get display name for a team ('home' or 'away')."""
        if team == "home":
            return self.home_team_name
        if team == "away":
            return self.away_team_name
        return team

    def has_scores(self) -> bool:
        """Whether quarter scores have been provided."""
        return bool(self.home_scores) and bool(self.away_scores)

    def quarter_scores(self) -> list[dict]:
        """Per-quarter scoring breakdown.

        Returns list of dicts with quarter, home_score, away_score (per-quarter, not cumulative).
        """
        if not self.has_scores():
            return []

        rows = []
        for i in range(max(len(self.home_scores), len(self.away_scores))):
            home_cum = self.home_scores[i] if i < len(self.home_scores) else self.home_scores[-1]
            away_cum = self.away_scores[i] if i < len(self.away_scores) else self.away_scores[-1]
            home_prev = self.home_scores[i - 1] if i > 0 and i - 1 < len(self.home_scores) else 0
            away_prev = self.away_scores[i - 1] if i > 0 and i - 1 < len(self.away_scores) else 0
            q_label = f"OT{i - 3}" if i >= 4 else f"Q{i + 1}"
            rows.append({
                "quarter": q_label,
                "home": home_cum - home_prev,
                "away": away_cum - away_prev,
                "home_cumulative": home_cum,
                "away_cumulative": away_cum,
            })
        return rows


def load_roster(path: str | Path) -> Roster:
    """Load roster from a JSON file.

    Expected format:
    {
        "home": {
            "name": "Team A",
            "players": [
                {"number": 1, "name": "Player One", "captain": false},
                ...
            ],
            "staff": ["Coach Name"]
        },
        "away": { ... }
    }
    """
    with open(path) as f:
        data = json.load(f)

    roster = Roster()

    if "home" in data:
        home = data["home"]
        roster.home_team_name = home.get("name", "Home")
        roster.home_players = [
            RosterPlayer(
                number=p["number"],
                name=p["name"],
                is_captain=p.get("captain", False),
            )
            for p in home.get("players", [])
        ]
        roster.home_staff = home.get("staff", [])
        roster.home_scores = home.get("scores_cumulative", [])

    if "away" in data:
        away = data["away"]
        roster.away_team_name = away.get("name", "Away")
        roster.away_players = [
            RosterPlayer(
                number=p["number"],
                name=p["name"],
                is_captain=p.get("captain", False),
            )
            for p in away.get("players", [])
        ]
        roster.away_staff = away.get("staff", [])
        roster.away_scores = away.get("scores_cumulative", [])

    return roster

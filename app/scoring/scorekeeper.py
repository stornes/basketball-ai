"""Full scorekeeper system for manual stat entry.

Provides a JSON-based format for recording stats that kamper.basket.no
does not track (OREB, DREB, AST, TO, STL, BLK, PF, MIN, +/-) and a
merge function that enriches pipeline-detected box scores with this
ground-truth data.

Merge precedence:
  - DETECTED stats are NEVER overwritten
  - HEURISTIC stats ARE overwritten (manual is more reliable)
  - UNAVAILABLE stats ARE overwritten (filling the gap)
  - COMPUTED stats are recalculated after merge
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from app.analytics.box_score import GameBoxScore, PlayerBoxScore, StatSource, TeamBoxScore


# ── Stat fields eligible for merge ──

# Fields that the scorekeeper can provide
MERGEABLE_FIELDS: dict[str, str] = {
    "min": "min_played",
    "oreb": "orb",
    "dreb": "drb",
    "ast": "ast",
    "to": "to",
    "stl": "stl",
    "blk": "blk",
    "pf": "pf",
    "plus_minus": "plus_minus",
    # Scoring overrides (only merge if current source is not DETECTED)
    "fg": "fg",
    "fga": "fga",
    "three_p": "three_p",
    "three_pa": "three_pa",
    "ft": "ft",
    "fta": "fta",
}

# Stat source keys used in PlayerBoxScore.stat_sources
_SOURCE_KEY_MAP: dict[str, str] = {
    "min_played": "min_played",
    "orb": "orb",
    "drb": "drb",
    "ast": "ast",
    "to": "to",
    "stl": "stl",
    "blk": "blk",
    "pf": "pf",
    "plus_minus": "plus_minus",
    "fg": "fg",
    "fga": "fga",
    "three_p": "three_p",
    "three_pa": "three_pa",
    "ft": "ft",
    "fta": "fta",
}


@dataclass
class ScorekeeperPlayerStats:
    """Per-player manual stats. All fields optional except jersey."""

    jersey: int = 0
    min: float | None = None
    oreb: int | None = None
    dreb: int | None = None
    ast: int | None = None
    to: int | None = None
    stl: int | None = None
    blk: int | None = None
    pf: int | None = None
    plus_minus: int | None = None
    # Scoring overrides
    fg: int | None = None
    fga: int | None = None
    three_p: int | None = None
    three_pa: int | None = None
    ft: int | None = None
    fta: int | None = None

    def validate(self) -> None:
        """Raise ValueError if data is invalid."""
        if self.jersey <= 0:
            raise ValueError(f"Jersey number must be > 0, got {self.jersey}")
        if self.min is not None and (self.min < 0 or self.min > 60):
            raise ValueError(f"Minutes must be 0–60, got {self.min}")
        for stat_name in ("oreb", "dreb", "ast", "to", "stl", "blk", "pf",
                          "fg", "fga", "three_p", "three_pa", "ft", "fta"):
            val = getattr(self, stat_name)
            if val is not None and val < 0:
                raise ValueError(f"{stat_name} must be >= 0, got {val}")

    @classmethod
    def from_dict(cls, data: dict) -> ScorekeeperPlayerStats:
        return cls(
            jersey=data["jersey"],
            min=data.get("min"),
            oreb=data.get("oreb"),
            dreb=data.get("dreb"),
            ast=data.get("ast"),
            to=data.get("to"),
            stl=data.get("stl"),
            blk=data.get("blk"),
            pf=data.get("pf"),
            plus_minus=data.get("plus_minus"),
            fg=data.get("fg"),
            fga=data.get("fga"),
            three_p=data.get("three_p"),
            three_pa=data.get("three_pa"),
            ft=data.get("ft"),
            fta=data.get("fta"),
        )

    def to_dict(self) -> dict:
        d: dict = {"jersey": self.jersey}
        for key in ("min", "oreb", "dreb", "ast", "to", "stl", "blk", "pf",
                     "plus_minus", "fg", "fga", "three_p", "three_pa", "ft", "fta"):
            val = getattr(self, key)
            if val is not None:
                d[key] = val
        return d


@dataclass
class ScorekeeperData:
    """Full scorekeeper input for both teams."""

    home_team_name: str = ""
    away_team_name: str = ""
    home_players: list[ScorekeeperPlayerStats] = field(default_factory=list)
    away_players: list[ScorekeeperPlayerStats] = field(default_factory=list)

    def validate(self) -> None:
        """Validate all player entries."""
        for p in self.home_players + self.away_players:
            p.validate()

    @classmethod
    def from_json(cls, path: str | Path) -> ScorekeeperData:
        """Load scorekeeper data from a JSON file."""
        with open(path) as f:
            data = json.load(f)

        sk = cls()
        if "home" in data:
            home = data["home"]
            sk.home_team_name = home.get("team_name", "")
            sk.home_players = [
                ScorekeeperPlayerStats.from_dict(p) for p in home.get("players", [])
            ]
        if "away" in data:
            away = data["away"]
            sk.away_team_name = away.get("team_name", "")
            sk.away_players = [
                ScorekeeperPlayerStats.from_dict(p) for p in away.get("players", [])
            ]

        sk.validate()
        return sk

    def to_json(self, path: str | Path) -> None:
        """Write scorekeeper data to a JSON file."""
        data: dict = {}
        if self.home_players:
            data["home"] = {
                "team_name": self.home_team_name,
                "players": [p.to_dict() for p in self.home_players],
            }
        if self.away_players:
            data["away"] = {
                "team_name": self.away_team_name,
                "players": [p.to_dict() for p in self.away_players],
            }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


def _should_overwrite(current_source: StatSource | None) -> bool:
    """Return True if the current stat source should yield to manual data."""
    if current_source is None:
        return True
    return current_source in (StatSource.UNAVAILABLE, StatSource.HEURISTIC)


def _merge_player(
    box: PlayerBoxScore,
    sk: ScorekeeperPlayerStats,
) -> None:
    """Merge scorekeeper stats into a single PlayerBoxScore (mutates in place)."""
    for json_key, box_attr in MERGEABLE_FIELDS.items():
        sk_value = getattr(sk, json_key)
        if sk_value is None:
            continue

        source_key = _SOURCE_KEY_MAP[box_attr]
        current_source = box.stat_sources.get(source_key)

        if _should_overwrite(current_source):
            setattr(box, box_attr, sk_value)
            box.stat_sources[source_key] = StatSource.MANUAL


def merge_scorekeeper(
    game: GameBoxScore,
    scorekeeper: ScorekeeperData,
) -> GameBoxScore:
    """Enrich a GameBoxScore with manual scorekeeper data.

    Mutates the GameBoxScore in place and returns it.

    Merge rules:
      - DETECTED stats are never overwritten
      - HEURISTIC and UNAVAILABLE stats are overwritten
      - Unmatched players (in scorekeeper but not box score) get new entries
      - All merged fields receive StatSource.MANUAL
    """
    for team_key, sk_players, team_box in (
        ("home", scorekeeper.home_players, game.home),
        ("away", scorekeeper.away_players, game.away),
    ):
        # Build jersey → PlayerBoxScore index
        jersey_index: dict[int, PlayerBoxScore] = {}
        for p in team_box.players:
            if p.jersey_number is not None:
                jersey_index[p.jersey_number] = p

        for sk_player in sk_players:
            if sk_player.jersey in jersey_index:
                _merge_player(jersey_index[sk_player.jersey], sk_player)
            else:
                # Create new PlayerBoxScore for unmatched player
                new_player = PlayerBoxScore(
                    player_id=sk_player.jersey,
                    jersey_number=sk_player.jersey,
                    team=team_key,
                )
                _merge_player(new_player, sk_player)
                team_box.players.append(new_player)

    return game

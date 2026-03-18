"""Box score statistics system — NBA-standard with Professional and Youth profiles.

Provides per-player box scores with provenance tracking (StatSource) and
two display profiles: PROFESSIONAL (full 20+ stat columns) and YOUTH
(coaching-focused subset with impact lines and KPIs).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

from app.config.roster import Roster
from app.events.assist_detector import AssistEvent
from app.events.event_types import PossessionEvent, ShotEvent, ShotOutcome
from app.events.rebound_detector import ReboundEvent, ReboundType
from app.events.steal_detector import StealEvent
from app.tracking.tracker import TrackedPlayer


class BoxScoreProfile(Enum):
    """Controls which stats are displayed and which KPIs are computed."""

    PROFESSIONAL = "professional"
    YOUTH = "youth"


class StatSource(Enum):
    """Provenance tracking for each statistic."""

    DETECTED = "detected"  # From video detection pipeline (high confidence)
    HEURISTIC = "heuristic"  # Inferred from video with uncertainty
    MANUAL = "manual"  # Entered by scorekeeper (ground truth)
    COMPUTED = "computed"  # Derived from other stats
    UNAVAILABLE = "unavailable"  # Cannot be determined


@dataclass
class PlayerBoxScore:
    """Per-player box score holding all statistics for both profiles."""

    # Identity
    player_id: int = 0  # track_id or jersey_number
    player_name: str | None = None
    jersey_number: int | None = None
    team: str | None = None  # "home" or "away"

    # Counting stats
    min_played: float = 0.0
    fg: int = 0  # Field goals made
    fga: int = 0  # Field goals attempted
    three_p: int = 0  # Three-pointers made
    three_pa: int = 0  # Three-pointers attempted
    ft: int = 0  # Free throws made
    fta: int = 0  # Free throws attempted
    orb: int = 0  # Offensive rebounds
    drb: int = 0  # Defensive rebounds
    ast: int = 0  # Assists
    to: int = 0  # Turnovers
    stl: int = 0  # Steals
    blk: int = 0  # Blocks
    pf: int = 0  # Personal fouls
    deflections: int = 0  # Hustle metric (youth)
    plus_minus: int = 0  # Point differential while on court

    # Provenance
    stat_sources: dict[str, StatSource] = field(default_factory=dict)

    # ── Computed properties ──

    @property
    def pts(self) -> int:
        """Total points: 2P*2 + 3P*3 + FT."""
        two_p = self.fg - self.three_p
        return two_p * 2 + self.three_p * 3 + self.ft

    @property
    def reb(self) -> int:
        """Total rebounds."""
        return self.orb + self.drb

    @property
    def two_p(self) -> int:
        """Two-point field goals made."""
        return self.fg - self.three_p

    @property
    def two_pa(self) -> int:
        """Two-point field goals attempted."""
        return self.fga - self.three_pa

    @property
    def fg_pct(self) -> float:
        """Field goal percentage."""
        return self.fg / self.fga if self.fga > 0 else 0.0

    @property
    def three_p_pct(self) -> float:
        """Three-point percentage."""
        return self.three_p / self.three_pa if self.three_pa > 0 else 0.0

    @property
    def two_p_pct(self) -> float:
        """Two-point percentage."""
        return self.two_p / self.two_pa if self.two_pa > 0 else 0.0

    @property
    def ft_pct(self) -> float:
        """Free throw percentage."""
        return self.ft / self.fta if self.fta > 0 else 0.0

    # ── Youth KPIs ──

    @property
    def ast_to_ratio(self) -> float:
        """Assist-to-turnover ratio."""
        return self.ast / self.to if self.to > 0 else float(self.ast)

    @property
    def defensive_activity_index(self) -> int:
        """DAI = steals + deflections."""
        return self.stl + self.deflections

    @property
    def effort_plays(self) -> int:
        """Effort = ORB + STL + deflections + BLK."""
        return self.orb + self.stl + self.deflections + self.blk

    @property
    def reb_per_min(self) -> float:
        """Rebounds per minute."""
        return self.reb / self.min_played if self.min_played > 0 else 0.0

    @property
    def impact_line(self) -> str:
        """PTS / REB / AST / STL / BLK / TO."""
        return f"{self.pts} / {self.reb} / {self.ast} / {self.stl} / {self.blk} / {self.to}"

    def to_dict(self) -> dict:
        """JSON-compatible serialisation."""
        return {
            "player_id": self.player_id,
            "player_name": self.player_name,
            "jersey_number": self.jersey_number,
            "team": self.team,
            "min_played": round(self.min_played, 1),
            "fg": self.fg,
            "fga": self.fga,
            "fg_pct": round(self.fg_pct, 3),
            "three_p": self.three_p,
            "three_pa": self.three_pa,
            "three_p_pct": round(self.three_p_pct, 3),
            "ft": self.ft,
            "fta": self.fta,
            "ft_pct": round(self.ft_pct, 3),
            "two_p": self.two_p,
            "two_pa": self.two_pa,
            "two_p_pct": round(self.two_p_pct, 3),
            "orb": self.orb,
            "drb": self.drb,
            "reb": self.reb,
            "ast": self.ast,
            "to": self.to,
            "stl": self.stl,
            "blk": self.blk,
            "pf": self.pf,
            "deflections": self.deflections,
            "pts": self.pts,
            "plus_minus": self.plus_minus,
            "stat_sources": {k: v.value for k, v in self.stat_sources.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> PlayerBoxScore:
        """Deserialise from dict."""
        sources = {
            k: StatSource(v)
            for k, v in data.get("stat_sources", {}).items()
        }
        return cls(
            player_id=data.get("player_id", 0),
            player_name=data.get("player_name"),
            jersey_number=data.get("jersey_number"),
            team=data.get("team"),
            min_played=data.get("min_played", 0.0),
            fg=data.get("fg", 0),
            fga=data.get("fga", 0),
            three_p=data.get("three_p", 0),
            three_pa=data.get("three_pa", 0),
            ft=data.get("ft", 0),
            fta=data.get("fta", 0),
            orb=data.get("orb", 0),
            drb=data.get("drb", 0),
            ast=data.get("ast", 0),
            to=data.get("to", 0),
            stl=data.get("stl", 0),
            blk=data.get("blk", 0),
            pf=data.get("pf", 0),
            deflections=data.get("deflections", 0),
            plus_minus=data.get("plus_minus", 0),
            stat_sources=sources,
        )


@dataclass
class TeamBoxScore:
    """Team-level box score aggregating all players."""

    team_name: str = ""
    team_key: str = ""  # "home" or "away"
    players: list[PlayerBoxScore] = field(default_factory=list)

    # ── Aggregated totals ──

    @property
    def total_fg(self) -> int:
        return sum(p.fg for p in self.players)

    @property
    def total_fga(self) -> int:
        return sum(p.fga for p in self.players)

    @property
    def team_fg_pct(self) -> float:
        return self.total_fg / self.total_fga if self.total_fga > 0 else 0.0

    @property
    def total_three_p(self) -> int:
        return sum(p.three_p for p in self.players)

    @property
    def total_three_pa(self) -> int:
        return sum(p.three_pa for p in self.players)

    @property
    def total_ft(self) -> int:
        return sum(p.ft for p in self.players)

    @property
    def total_fta(self) -> int:
        return sum(p.fta for p in self.players)

    @property
    def total_orb(self) -> int:
        return sum(p.orb for p in self.players)

    @property
    def total_drb(self) -> int:
        return sum(p.drb for p in self.players)

    @property
    def total_reb(self) -> int:
        return sum(p.reb for p in self.players)

    @property
    def total_ast(self) -> int:
        return sum(p.ast for p in self.players)

    @property
    def total_to(self) -> int:
        return sum(p.to for p in self.players)

    @property
    def total_stl(self) -> int:
        return sum(p.stl for p in self.players)

    @property
    def total_blk(self) -> int:
        return sum(p.blk for p in self.players)

    @property
    def total_pf(self) -> int:
        return sum(p.pf for p in self.players)

    @property
    def total_deflections(self) -> int:
        return sum(p.deflections for p in self.players)

    @property
    def total_pts(self) -> int:
        return sum(p.pts for p in self.players)

    def to_dict(self) -> dict:
        return {
            "team_name": self.team_name,
            "team_key": self.team_key,
            "players": [p.to_dict() for p in self.players],
            "totals": {
                "fg": self.total_fg,
                "fga": self.total_fga,
                "fg_pct": round(self.team_fg_pct, 3),
                "three_p": self.total_three_p,
                "three_pa": self.total_three_pa,
                "ft": self.total_ft,
                "fta": self.total_fta,
                "orb": self.total_orb,
                "drb": self.total_drb,
                "reb": self.total_reb,
                "ast": self.total_ast,
                "to": self.total_to,
                "stl": self.total_stl,
                "blk": self.total_blk,
                "pf": self.total_pf,
                "deflections": self.total_deflections,
                "pts": self.total_pts,
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> TeamBoxScore:
        return cls(
            team_name=data.get("team_name", ""),
            team_key=data.get("team_key", ""),
            players=[PlayerBoxScore.from_dict(p) for p in data.get("players", [])],
        )


@dataclass
class GameBoxScore:
    """Full game box score wrapping both teams."""

    home: TeamBoxScore = field(default_factory=TeamBoxScore)
    away: TeamBoxScore = field(default_factory=TeamBoxScore)
    profile: BoxScoreProfile = BoxScoreProfile.PROFESSIONAL
    game_date: str | None = None
    venue: str | None = None
    quarter_scores: list[dict] | None = None
    video_path: str | None = None
    detection_summary: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "profile": self.profile.value,
            "game_date": self.game_date,
            "venue": self.venue,
            "quarter_scores": self.quarter_scores,
            "video_path": self.video_path,
            "detection_summary": self.detection_summary,
            "home": self.home.to_dict(),
            "away": self.away.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> GameBoxScore:
        return cls(
            home=TeamBoxScore.from_dict(data.get("home", {})),
            away=TeamBoxScore.from_dict(data.get("away", {})),
            profile=BoxScoreProfile(data.get("profile", "professional")),
            game_date=data.get("game_date"),
            venue=data.get("venue"),
            quarter_scores=data.get("quarter_scores"),
            video_path=data.get("video_path"),
            detection_summary=data.get("detection_summary", {}),
        )


class BoxScoreCompiler:
    """Aggregates events into per-player box scores.

    Uses ShotEvents for FG/FGA/3P/3PA, PossessionEvents for TO,
    and TrackedPlayer presence for MIN estimation.
    """

    def __init__(
        self,
        roster: Roster | None = None,
        three_point_classifier: object | None = None,
    ):
        self.roster = roster
        self.three_point_classifier = three_point_classifier

    def compile(
        self,
        shot_events: list[ShotEvent],
        possession_events: list[PossessionEvent],
        tracks: list[TrackedPlayer],
        fps: float,
        profile: BoxScoreProfile = BoxScoreProfile.PROFESSIONAL,
        rebound_events: list[ReboundEvent] | None = None,
        assist_events: list[AssistEvent] | None = None,
        steal_events: list[StealEvent] | None = None,
        sample_rate: int = 1,
    ) -> GameBoxScore:
        """Build a GameBoxScore from pipeline event data."""
        # Accumulate per-player stats keyed by (team, player_id).
        # When jersey_number is known, use it as player_id to merge
        # fragmented track IDs for the same physical player.
        players: dict[tuple[str, int], PlayerBoxScore] = {}
        # Map track_id → canonical key for lookups by other detectors
        _track_to_key: dict[tuple[str, int], tuple[str, int]] = {}

        def _get_player(
            team: str | None,
            player_id: int,
            jersey_number: int | None = None,
        ) -> PlayerBoxScore:
            team_key = team or "unknown"

            # Prefer jersey number as canonical key (merges fragmented tracks)
            canonical_id = jersey_number if jersey_number is not None else player_id
            key = (team_key, canonical_id)

            # Track the mapping for later lookups
            _track_to_key[(team_key, player_id)] = key

            if key not in players:
                players[key] = PlayerBoxScore(
                    player_id=canonical_id,
                    team=team_key,
                    jersey_number=jersey_number,
                )
            p = players[key]
            # Update jersey number if newly discovered
            if jersey_number is not None and p.jersey_number is None:
                p.jersey_number = jersey_number
            return p

        def _get_player_by_track(
            team: str | None,
            track_id: int,
        ) -> PlayerBoxScore:
            """Look up player by track_id, resolving to canonical key if known."""
            team_key = team or "unknown"
            key = _track_to_key.get((team_key, track_id), (team_key, track_id))
            if key in players:
                return players[key]
            return _get_player(team, track_id)

        # ── Process shot events → FG, FGA, 3P, 3PA ──
        for shot in shot_events:
            pid = shot.shooter_track_id
            if pid is None:
                continue

            team = shot.team
            p = _get_player(team, pid, jersey_number=shot.jersey_number)

            # FGA always increments
            p.fga += 1
            p.stat_sources.setdefault("fga", StatSource.DETECTED)

            # Classify 2pt vs 3pt
            is_three = False
            if self.three_point_classifier and shot.court_position:
                is_three = self.three_point_classifier.is_three_pointer(
                    shot.court_position[0], shot.court_position[1],
                )
            if is_three:
                p.three_pa += 1
                p.stat_sources.setdefault("three_pa", StatSource.HEURISTIC)

            # FG (made shots)
            if shot.outcome == ShotOutcome.MADE:
                p.fg += 1
                p.stat_sources.setdefault("fg", StatSource.DETECTED)
                if is_three:
                    p.three_p += 1
                    p.stat_sources.setdefault("three_p", StatSource.HEURISTIC)

        # ── Process possession events → TO ──
        for poss in possession_events:
            if poss.result != "turnover":
                continue
            pid = poss.player_track_id
            # Map team convention: "team_a"/"team_b" → use as-is if not "home"/"away"
            team = poss.team
            p = _get_player_by_track(team, pid)
            p.to += 1
            p.stat_sources.setdefault("to", StatSource.DETECTED)

        # ── Process rebound events → ORB, DRB ──
        for reb in rebound_events or []:
            p = _get_player_by_track(reb.rebounder_team, reb.rebounder_track_id)
            if reb.rebound_type == ReboundType.OFFENSIVE:
                p.orb += 1
                p.stat_sources["orb"] = StatSource.HEURISTIC
            else:
                p.drb += 1
                p.stat_sources["drb"] = StatSource.HEURISTIC

        # ── Process assist events → AST ──
        for ast_ev in assist_events or []:
            p = _get_player_by_track(ast_ev.assister_team, ast_ev.assister_track_id)
            p.ast += 1
            p.stat_sources["ast"] = StatSource.HEURISTIC

        # ── Process steal events → STL ──
        for stl_ev in steal_events or []:
            p = _get_player_by_track(stl_ev.stealer_team, stl_ev.stealer_track_id)
            p.stl += 1
            p.stat_sources["stl"] = StatSource.HEURISTIC

        # ── Compute +/- from scoring events × on-court tracks ──
        if tracks and shot_events:
            self._compute_plus_minus(shot_events, tracks, fps, players, _get_player)

        # ── Estimate minutes from track presence ──
        # Aggregate frame counts per canonical player key (not raw track_id).
        # _track_to_key maps (team, track_id) → (team, jersey_number) for
        # tracks that appeared in shot events. This merges fragmented tracks
        # for the same physical player.
        jersey_frame_count: dict[tuple[str, int], int] = {}
        for t in tracks:
            team = t.team or "unknown"
            raw_key = (team, t.track_id)
            canonical = _track_to_key.get(raw_key, raw_key)
            jersey_frame_count[canonical] = jersey_frame_count.get(canonical, 0) + 1

        for key, p in players.items():
            count = jersey_frame_count.get(key, 0)
            if count > 0 and fps > 0:
                # Multiply by sample_rate: we only process every Nth frame
                p.min_played = (count * sample_rate) / fps / 60.0
                p.stat_sources.setdefault("min_played", StatSource.HEURISTIC)

        # ── Resolve names from roster ──
        if self.roster:
            for key, p in players.items():
                team_key = p.team
                if p.jersey_number is not None and team_key in ("home", "away"):
                    name = self.roster.player_name(p.jersey_number, team_key)
                    if name:
                        p.player_name = name

        # ── Set computed stat sources ──
        for p in players.values():
            if p.fg > 0 or p.fga > 0:
                p.stat_sources["pts"] = StatSource.COMPUTED
                p.stat_sources["fg_pct"] = StatSource.COMPUTED
            # Mark unavailable stats
            for unavail in ("ft", "fta", "blk", "pf"):
                p.stat_sources.setdefault(unavail, StatSource.UNAVAILABLE)
            for heuristic in ("orb", "drb", "ast", "stl", "plus_minus"):
                p.stat_sources.setdefault(heuristic, StatSource.UNAVAILABLE)

        # ── Split into home/away teams (single pass) ──
        home_list: list[PlayerBoxScore] = []
        away_list: list[PlayerBoxScore] = []
        for p in players.values():
            if p.team == "home":
                home_list.append(p)
            elif p.team == "away":
                away_list.append(p)

        sort_key = lambda p: (p.jersey_number or 999, p.player_id)
        home_players = sorted(home_list, key=sort_key)
        away_players = sorted(away_list, key=sort_key)

        home_name = self.roster.home_team_name if self.roster else "Home"
        away_name = self.roster.away_team_name if self.roster else "Away"

        # Mark +/- as heuristic for players that have it set
        for p in players.values():
            if p.plus_minus != 0:
                p.stat_sources["plus_minus"] = StatSource.HEURISTIC

        game = GameBoxScore(
            home=TeamBoxScore(
                team_name=home_name,
                team_key="home",
                players=home_players,
            ),
            away=TeamBoxScore(
                team_name=away_name,
                team_key="away",
                players=away_players,
            ),
            profile=profile,
            quarter_scores=(
                self.roster.quarter_scores() if self.roster and self.roster.has_scores() else None
            ),
        )

        return game

    @staticmethod
    def _compute_plus_minus(
        shot_events: list[ShotEvent],
        tracks: list[TrackedPlayer],
        fps: float,
        players: dict[tuple[str, int], "PlayerBoxScore"],
        _get_player,
    ) -> None:
        """Compute +/- for each player based on scoring events and on-court presence.

        For each made shot, determines which players were on court (had a track
        detection within a window around the scoring event) and adjusts their +/-.
        """
        # Build frame→tracks index: for each frame, which (team, track_id) were on court
        # Use a sparse approach: group tracks by frame_idx
        frame_tracks: dict[int, list[tuple[str | None, int]]] = {}
        for t in tracks:
            if t.frame_idx not in frame_tracks:
                frame_tracks[t.frame_idx] = []
            frame_tracks[t.frame_idx].append((t.team, t.track_id))

        # Window (frames) to check for on-court presence around a scoring event
        presence_window = int(fps * 2) if fps > 0 else 60

        for shot in shot_events:
            if shot.outcome != ShotOutcome.MADE:
                continue

            # Determine points scored
            pts = 2  # default 2-pointer
            if shot.court_position:
                # If we had 3pt info on the shot, we'd use it
                # For now, all made shots are treated as 2-pointers for +/-
                # (3pt classification happens at compile level, not on the event)
                pass

            scoring_team = shot.team
            if not scoring_team:
                continue

            # Find all players on court near this shot's frame
            shot_frame = shot.frame_idx
            on_court: set[tuple[str, int]] = set()
            for f in range(shot_frame - presence_window, shot_frame + presence_window + 1):
                if f in frame_tracks:
                    for team, tid in frame_tracks[f]:
                        if team:
                            on_court.add((team, tid))

            # Adjust +/- for each on-court player
            for team, tid in on_court:
                key = (team, tid)
                if key in players:
                    p = players[key]
                    if team == scoring_team:
                        p.plus_minus += pts
                    else:
                        p.plus_minus -= pts

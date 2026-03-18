"""Advanced basketball analytics — Four Factors, Game Score, TS%, eFG%, USG%, Per-36.

Computes championship-level advanced statistics from PlayerBoxScore data.
All formulas follow standard basketball analytics conventions (Dean Oliver,
John Hollinger, Basketball Reference).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.analytics.box_score import GameBoxScore, PlayerBoxScore, TeamBoxScore


# ═══════════════════════════════════════════════════════════════════
# Growth-Mindset Grade Constants
# ═══════════════════════════════════════════════════════════════════

# Player grades (6-tier scale)
GRADE_IMPACT_PLAYER = "Impact Player"
GRADE_RISING_PERFORMER = "Rising Performer"
GRADE_SOLID_FOUNDATION = "Solid Foundation"
GRADE_DEVELOPING = "Developing"
GRADE_FOUNDATION_PHASE = "Foundation Phase"
GRADE_GROWTH_OPPORTUNITY = "Growth Opportunity"
GRADE_NA = "N/A"

# Four Factors use 4-tier scale (subset)
FOUR_FACTOR_GRADES = (GRADE_IMPACT_PLAYER, GRADE_SOLID_FOUNDATION, GRADE_DEVELOPING, GRADE_GROWTH_OPPORTUNITY)
PLAYER_GRADES = (GRADE_IMPACT_PLAYER, GRADE_RISING_PERFORMER, GRADE_SOLID_FOUNDATION,
                 GRADE_DEVELOPING, GRADE_FOUNDATION_PHASE, GRADE_GROWTH_OPPORTUNITY)


# ═══════════════════════════════════════════════════════════════════
# Four Factors (Dean Oliver)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class FourFactors:
    """Team-level Four Factors analysis."""

    efg_pct: float = 0.0       # Effective FG%
    efg_grade: str = ""
    tov_pct: float = 0.0       # Turnover %
    tov_grade: str = ""
    ft_rate: float = 0.0       # Free Throw Rate (FTA/FGA)
    ft_rate_grade: str = ""
    oreb: int = 0              # Offensive rebounds (raw count)
    oreb_grade: str = ""
    est_possessions: float = 0.0
    ppp: float = 0.0           # Points per possession
    offensive_rating: float = 0.0  # PPP × 100

    def to_dict(self) -> dict:
        return {
            "efg_pct": round(self.efg_pct, 3),
            "efg_grade": self.efg_grade,
            "tov_pct": round(self.tov_pct, 3),
            "tov_grade": self.tov_grade,
            "ft_rate": round(self.ft_rate, 3),
            "ft_rate_grade": self.ft_rate_grade,
            "oreb": self.oreb,
            "oreb_grade": self.oreb_grade,
            "est_possessions": round(self.est_possessions, 2),
            "ppp": round(self.ppp, 3),
            "offensive_rating": round(self.offensive_rating, 1),
        }


def compute_four_factors(
    team: TeamBoxScore,
    *,
    has_to_data: bool | None = None,
    has_oreb_data: bool | None = None,
) -> FourFactors:
    """Compute Four Factors from team box score totals.

    Args:
        team: Team box score.
        has_to_data: Whether turnover data is available. If None, auto-detect
            (True if total_to > 0 or any player has TO stat source).
        has_oreb_data: Whether OREB data is available. If None, auto-detect
            (True if total_orb > 0 or any player has ORB stat source).
    """
    fgm = team.total_fg
    fga = team.total_fga
    three_pm = team.total_three_p
    fta = team.total_fta
    to = team.total_to
    oreb = team.total_orb
    pts = team.total_pts

    # Auto-detect data availability from stat sources
    if has_to_data is None:
        has_to_data = to > 0 or _has_stat_source(team, "to")
    if has_oreb_data is None:
        has_oreb_data = oreb > 0 or _has_stat_source(team, "orb")

    # eFG% = (FGM + 0.5 × 3PM) / FGA
    efg = (fgm + 0.5 * three_pm) / fga if fga > 0 else 0.0

    # TOV% = TO / (FGA + 0.44 × FTA + TO)
    tov_denom = fga + 0.44 * fta + to
    tov_pct = to / tov_denom if tov_denom > 0 else 0.0

    # FT Rate = FTA / FGA
    ft_rate = fta / fga if fga > 0 else 0.0

    # Est. Possessions = FGA + 0.44 × FTA + TO − OREB
    est_poss = fga + 0.44 * fta + to - oreb

    # PPP = PTS / Est_Possessions
    ppp = pts / est_poss if est_poss > 0 else 0.0

    return FourFactors(
        efg_pct=efg,
        efg_grade=_grade_efg(efg),
        tov_pct=tov_pct,
        tov_grade=_grade_tov(tov_pct, has_data=has_to_data),
        ft_rate=ft_rate,
        ft_rate_grade=_grade_ft_rate(ft_rate),
        oreb=oreb,
        oreb_grade=_grade_oreb(oreb, has_data=has_oreb_data),
        est_possessions=est_poss,
        ppp=ppp,
        offensive_rating=ppp * 100,
    )


def _has_stat_source(team: TeamBoxScore, key: str) -> bool:
    """Check if any player on the team has real data for the given stat.

    Returns True only if the stat source is DETECTED, HEURISTIC, or MANUAL
    — not UNAVAILABLE or COMPUTED.
    """
    from app.analytics.box_score import StatSource
    _REAL_SOURCES = {StatSource.DETECTED, StatSource.HEURISTIC, StatSource.MANUAL}
    for p in team.players:
        source = p.stat_sources.get(key)
        if source in _REAL_SOURCES:
            return True
    return False


def _grade_efg(val: float) -> str:
    """Growth-mindset grade for eFG%."""
    if val > 0.52:
        return GRADE_IMPACT_PLAYER
    if val > 0.45:
        return GRADE_SOLID_FOUNDATION
    if val > 0.38:
        return GRADE_DEVELOPING
    return GRADE_GROWTH_OPPORTUNITY


def _grade_tov(val: float, *, has_data: bool = True) -> str:
    """Growth-mindset grade for TOV%. Lower is better."""
    if not has_data:
        return GRADE_NA
    if val < 0.12:
        return GRADE_IMPACT_PLAYER
    if val < 0.16:
        return GRADE_SOLID_FOUNDATION
    if val < 0.20:
        return GRADE_DEVELOPING
    return GRADE_GROWTH_OPPORTUNITY


def _grade_ft_rate(val: float) -> str:
    """Growth-mindset grade for FT Rate."""
    if val > 0.35:
        return GRADE_IMPACT_PLAYER
    if val > 0.25:
        return GRADE_SOLID_FOUNDATION
    if val > 0.15:
        return GRADE_DEVELOPING
    return GRADE_GROWTH_OPPORTUNITY


def _grade_oreb(val: int, *, has_data: bool = True) -> str:
    """Growth-mindset grade for OREB."""
    if not has_data:
        return GRADE_NA
    if val > 15:
        return GRADE_IMPACT_PLAYER
    if val > 10:
        return GRADE_SOLID_FOUNDATION
    if val > 5:
        return GRADE_DEVELOPING
    return GRADE_GROWTH_OPPORTUNITY


# ═══════════════════════════════════════════════════════════════════
# Advanced Player Stats
# ═══════════════════════════════════════════════════════════════════

@dataclass
class AdvancedPlayerStats:
    """Computed advanced metrics for a single player."""

    # Identity (copied from PlayerBoxScore)
    player_id: int = 0
    player_name: str | None = None
    jersey_number: int | None = None
    team: str | None = None

    # Raw stats (for reference)
    pts: int = 0
    fg: int = 0
    fga: int = 0
    three_p: int = 0
    three_pa: int = 0
    ft: int = 0
    fta: int = 0
    orb: int = 0
    drb: int = 0
    reb: int = 0
    ast: int = 0
    to: int = 0
    stl: int = 0
    blk: int = 0
    pf: int = 0
    min_played: float = 0.0

    # Advanced metrics
    ts_pct: float = 0.0       # True Shooting %
    efg_pct: float = 0.0      # Effective FG %
    usg_pct: float = 0.0      # Usage Rate
    game_score: float = 0.0   # Hollinger Game Score
    ast_to_ratio: float | str = 0.0  # AST/TO (INF or -)
    reb_pct: float = 0.0      # Rebound share (simplified)
    grade: str = ""            # Letter grade

    # Per-36 projections
    pts_per36: float = 0.0
    reb_per36: float = 0.0
    ast_per36: float = 0.0
    stl_per36: float = 0.0
    to_per36: float = 0.0
    fga_per36: float = 0.0

    def to_dict(self) -> dict:
        return {
            "player_id": self.player_id,
            "player_name": self.player_name,
            "jersey_number": self.jersey_number,
            "team": self.team,
            "pts": self.pts,
            "fg": self.fg,
            "fga": self.fga,
            "three_p": self.three_p,
            "three_pa": self.three_pa,
            "ft": self.ft,
            "fta": self.fta,
            "orb": self.orb,
            "drb": self.drb,
            "reb": self.reb,
            "ast": self.ast,
            "to": self.to,
            "stl": self.stl,
            "blk": self.blk,
            "pf": self.pf,
            "min_played": round(self.min_played, 1),
            "ts_pct": round(self.ts_pct, 3),
            "efg_pct": round(self.efg_pct, 3),
            "usg_pct": round(self.usg_pct, 1),
            "game_score": round(self.game_score, 1),
            "ast_to_ratio": self.ast_to_ratio if isinstance(self.ast_to_ratio, str) else round(self.ast_to_ratio, 1),
            "reb_pct": round(self.reb_pct, 1),
            "grade": self.grade,
            "pts_per36": round(self.pts_per36, 1),
            "reb_per36": round(self.reb_per36, 1),
            "ast_per36": round(self.ast_per36, 1),
            "stl_per36": round(self.stl_per36, 1),
            "to_per36": round(self.to_per36, 1),
            "fga_per36": round(self.fga_per36, 1),
        }


def compute_player_advanced(
    player: PlayerBoxScore,
    team_min: float,
    team_poss: float,
    team_reb: int,
) -> AdvancedPlayerStats:
    """Compute advanced stats for a single player.

    Args:
        player: The player's box score.
        team_min: Sum of all team player minutes.
        team_poss: Estimated team possessions.
        team_reb: Total team rebounds.
    """
    p = player
    stats = AdvancedPlayerStats(
        player_id=p.player_id,
        player_name=p.player_name,
        jersey_number=p.jersey_number,
        team=p.team,
        pts=p.pts,
        fg=p.fg,
        fga=p.fga,
        three_p=p.three_p,
        three_pa=p.three_pa,
        ft=p.ft,
        fta=p.fta,
        orb=p.orb,
        drb=p.drb,
        reb=p.reb,
        ast=p.ast,
        to=p.to,
        stl=p.stl,
        blk=p.blk,
        pf=p.pf,
        min_played=p.min_played,
    )

    # TS% = PTS / (2 × (FGA + 0.44 × FTA))
    ts_denom = 2 * (p.fga + 0.44 * p.fta)
    stats.ts_pct = p.pts / ts_denom if ts_denom > 0 else 0.0

    # eFG% = (FGM + 0.5 × 3PM) / FGA
    stats.efg_pct = (p.fg + 0.5 * p.three_p) / p.fga if p.fga > 0 else 0.0

    # USG% = 100 × ((FGA + 0.44×FTA + TO) × (Tm_MIN/5)) / (MIN × Tm_Poss)
    usg_numer = (p.fga + 0.44 * p.fta + p.to) * (team_min / 5)
    usg_denom = p.min_played * team_poss
    stats.usg_pct = 100 * usg_numer / usg_denom if usg_denom > 0 else 0.0

    # Game Score = PTS + 0.4×FGM − 0.7×FGA − 0.4×(FTA−FTM)
    #            + 0.7×OREB + 0.3×DREB + STL + 0.7×AST + 0.7×BLK − 0.4×PF − TO
    stats.game_score = (
        p.pts
        + 0.4 * p.fg
        - 0.7 * p.fga
        - 0.4 * (p.fta - p.ft)
        + 0.7 * p.orb
        + 0.3 * p.drb
        + p.stl
        + 0.7 * p.ast
        + 0.7 * p.blk
        - 0.4 * p.pf
        - p.to
    )

    # AST/TO
    if p.to == 0 and p.ast > 0:
        stats.ast_to_ratio = "INF"
    elif p.to == 0 and p.ast == 0:
        stats.ast_to_ratio = "-"
    else:
        stats.ast_to_ratio = p.ast / p.to

    # REB% = (Player_REB / Team_REB) × 100
    stats.reb_pct = (p.reb / team_reb * 100) if team_reb > 0 else 0.0

    # Per-36 projections
    if p.min_played >= 1.0:  # At least 1 minute
        factor = 36 / p.min_played
        stats.pts_per36 = p.pts * factor
        stats.reb_per36 = p.reb * factor
        stats.ast_per36 = p.ast * factor
        stats.stl_per36 = p.stl * factor
        stats.to_per36 = p.to * factor
        stats.fga_per36 = p.fga * factor

    # Grade
    stats.grade = compute_grade(stats.game_score, stats.ts_pct, stats.ast_to_ratio)

    return stats


def compute_grade(
    game_score: float,
    ts_pct: float,
    ast_to: float | str,
) -> str:
    """Assign growth-mindset grade from Game Score + efficiency.

    Uses developmental labels instead of letter grades — frames performance
    as current stage, not fixed judgement. Thresholds calibrated to
    Norwegian U16 competition.

    Labels (Pyramid Principle — conclusion first):
        Impact Player    — consistently drives outcomes
        Rising Performer — approaching mastery, not yet elite
        Solid Foundation — good base with clear upside
        Developing       — improving, not yet consistent
        Foundation Phase — learning the game, building habits
        Growth Opportunity — significant room to develop
    """
    if game_score > 15 and ts_pct > 0.55:
        return GRADE_IMPACT_PLAYER
    if game_score > 12 or (game_score > 8 and ts_pct > 0.60):
        return GRADE_RISING_PERFORMER
    if game_score > 8:
        return GRADE_SOLID_FOUNDATION
    if game_score > 4:
        return GRADE_DEVELOPING
    if game_score > 0:
        return GRADE_FOUNDATION_PHASE
    return GRADE_GROWTH_OPPORTUNITY


# ═══════════════════════════════════════════════════════════════════
# Team Advanced Stats Computation
# ═══════════════════════════════════════════════════════════════════

@dataclass
class TeamAdvancedStats:
    """All advanced stats for one team."""

    team_name: str = ""
    team_key: str = ""
    four_factors: FourFactors = field(default_factory=FourFactors)
    player_stats: list[AdvancedPlayerStats] = field(default_factory=list)
    total_pts: int = 0
    total_fg: int = 0
    total_fga: int = 0
    total_three_p: int = 0
    total_three_pa: int = 0
    total_ft: int = 0
    total_fta: int = 0
    total_reb: int = 0
    total_orb: int = 0
    total_drb: int = 0
    total_ast: int = 0
    total_to: int = 0
    total_stl: int = 0
    total_blk: int = 0
    total_pf: int = 0

    def to_dict(self) -> dict:
        return {
            "team_name": self.team_name,
            "team_key": self.team_key,
            "four_factors": self.four_factors.to_dict(),
            "player_stats": [p.to_dict() for p in self.player_stats],
            "totals": {
                "pts": self.total_pts,
                "fg": self.total_fg,
                "fga": self.total_fga,
                "three_p": self.total_three_p,
                "three_pa": self.total_three_pa,
                "ft": self.total_ft,
                "fta": self.total_fta,
                "reb": self.total_reb,
                "orb": self.total_orb,
                "drb": self.total_drb,
                "ast": self.total_ast,
                "to": self.total_to,
                "stl": self.total_stl,
                "blk": self.total_blk,
                "pf": self.total_pf,
            },
        }


def compute_team_advanced(team: TeamBoxScore) -> TeamAdvancedStats:
    """Compute all advanced stats for a team."""
    ff = compute_four_factors(team)

    team_min = sum(p.min_played for p in team.players)
    team_reb = team.total_reb

    player_stats = []
    for p in team.players:
        adv = compute_player_advanced(
            p,
            team_min=team_min,
            team_poss=ff.est_possessions,
            team_reb=team_reb,
        )
        player_stats.append(adv)

    # Sort by game score descending
    player_stats.sort(key=lambda x: x.game_score, reverse=True)

    return TeamAdvancedStats(
        team_name=team.team_name,
        team_key=team.team_key,
        four_factors=ff,
        player_stats=player_stats,
        total_pts=team.total_pts,
        total_fg=team.total_fg,
        total_fga=team.total_fga,
        total_three_p=team.total_three_p,
        total_three_pa=team.total_three_pa,
        total_ft=team.total_ft,
        total_fta=team.total_fta,
        total_reb=team.total_reb,
        total_orb=team.total_orb,
        total_drb=team.total_drb,
        total_ast=team.total_ast,
        total_to=team.total_to,
        total_stl=team.total_stl,
        total_blk=team.total_blk,
        total_pf=team.total_pf,
    )


@dataclass
class GameAdvancedStats:
    """Advanced stats for a full game."""

    home: TeamAdvancedStats = field(default_factory=TeamAdvancedStats)
    away: TeamAdvancedStats = field(default_factory=TeamAdvancedStats)
    quarter_scores: list[dict] | None = None

    def to_dict(self) -> dict:
        return {
            "home": self.home.to_dict(),
            "away": self.away.to_dict(),
            "quarter_scores": self.quarter_scores,
        }


def compute_game_advanced(game: GameBoxScore) -> GameAdvancedStats:
    """Compute advanced stats for the full game."""
    return GameAdvancedStats(
        home=compute_team_advanced(game.home),
        away=compute_team_advanced(game.away),
        quarter_scores=game.quarter_scores,
    )

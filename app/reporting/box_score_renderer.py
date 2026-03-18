"""Box score renderers — Professional (NBA-standard) and Youth (coaching-focused).

Produces text table and JSON output for GameBoxScore data.
"""

from __future__ import annotations

import json

from app.analytics.box_score import (
    GameBoxScore,
    PlayerBoxScore,
    TeamBoxScore,
)


def _fmt_pct(val: float) -> str:
    """Format percentage as .XXX (no leading zero, 3 decimal places)."""
    if val == 0.0:
        return ".000"
    return f"{val:.3f}"


def _fmt_min(minutes: float) -> str:
    """Format minutes as MM:SS."""
    total_sec = int(minutes * 60)
    m, s = divmod(total_sec, 60)
    return f"{m}:{s:02d}"


def _player_display_name(p: PlayerBoxScore) -> str:
    """Format player name for display."""
    if p.player_name:
        # Shorten to "F. Lastname" or just use full name if short
        parts = p.player_name.split(", ")
        if len(parts) == 2:
            last, first = parts
            initial = first[0] if first else ""
            name = f"{initial}. {last}"
        else:
            name = p.player_name
    else:
        name = f"Player #{p.player_id}"

    jersey = f"(#{p.jersey_number})" if p.jersey_number else ""
    return f"{name} {jersey}".strip()


class BoxScoreRenderer:
    """Renders GameBoxScore to text and JSON formats."""

    def render_text(self, game: GameBoxScore) -> str:
        """Render full game box score as formatted text with impact lines and KPIs."""
        lines = []

        # Header
        home_pts = game.home.total_pts
        away_pts = game.away.total_pts
        lines.append(f"{game.home.team_name} vs {game.away.team_name}")
        if game.game_date:
            lines.append(f"{game.game_date} | Final: {home_pts} - {away_pts}")
        else:
            lines.append(f"Final: {home_pts} - {away_pts}")
        lines.append("")

        # Quarter scores
        if game.quarter_scores:
            lines.append(self._render_quarter_table(game))
            lines.append("")

        # Each team with impact lines
        for team in (game.home, game.away):
            lines.append(team.team_name.upper())
            lines.append(self._box_score_table(team))
            lines.append("")

        # KPIs
        all_players = game.home.players + game.away.players
        if all_players:
            lines.append("KEY PERFORMANCE INDICATORS")
            lines.extend(self._kpis(all_players))
            lines.append("")

        return "\n".join(lines)

    def render_json(self, game: GameBoxScore) -> str:
        """Render full game box score as JSON string."""
        return json.dumps(game.to_dict(), indent=2, ensure_ascii=False)

    def _box_score_table(self, team: TeamBoxScore) -> str:
        """Box score table with impact lines below each player row."""
        header = (
            f"{'Player':<24s} {'MIN':>5s}  {'FG':>5s}  {'FT':>5s}  "
            f"{'PTS':>3s} {'REB':>3s} {'AST':>3s} {'TO':>3s} "
            f"{'STL':>3s} {'BLK':>3s} {'PF':>3s} {'DEFL':>4s}"
        )
        lines = [header]
        lines.append("-" * len(header))

        for p in team.players:
            name = _player_display_name(p)[:24]
            fg_str = f"{p.fg}-{p.fga}"
            ft_str = f"{p.ft}-{p.fta}"

            line = (
                f"{name:<24s} {_fmt_min(p.min_played):>5s}  {fg_str:>5s}  "
                f"{ft_str:>5s}  "
                f"{p.pts:>3d} {p.reb:>3d} {p.ast:>3d} {p.to:>3d} "
                f"{p.stl:>3d} {p.blk:>3d} {p.pf:>3d} {p.deflections:>4d}"
            )
            lines.append(line)
            # Impact line
            ast_to = f"{p.ast_to_ratio:.1f}" if p.to > 0 else f"{p.ast:.0f}.0"
            lines.append(
                f"  Impact: {p.impact_line}   "
                f"AST/TO: {ast_to}   DAI: {p.defensive_activity_index}   "
                f"EFF: {p.effort_plays}"
            )

        # Totals
        t = team
        fg_str = f"{t.total_fg}-{t.total_fga}"
        ft_str = f"{t.total_ft}-{t.total_fta}"
        lines.append("-" * len(header))
        total_line = (
            f"{'TOTALS':<24s} {'':>5s}  {fg_str:>5s}  {ft_str:>5s}  "
            f"{t.total_pts:>3d} {t.total_reb:>3d} {t.total_ast:>3d} {t.total_to:>3d} "
            f"{t.total_stl:>3d} {t.total_blk:>3d} {t.total_pf:>3d} {t.total_deflections:>4d}"
        )
        lines.append(total_line)

        return "\n".join(lines)

    def _kpis(self, players: list[PlayerBoxScore]) -> list[str]:
        """Compute and format KPI highlights.

        Covers: AST/TO ratio, Defensive Activity Index,
        Shot Selection (FG%), Effort Plays, and REB/MIN.
        """
        lines = []

        # Filter to real players only (exclude unattributed/jersey=0)
        real = [
            p for p in players
            if p.jersey_number is not None and p.jersey_number != 0
            and not (p.player_name and "unattributed" in p.player_name.lower())
        ]
        if not real:
            real = players

        # Best AST/TO
        with_to = [p for p in real if p.to > 0 or p.ast > 0]
        if with_to:
            best = max(with_to, key=lambda p: p.ast_to_ratio)
            name = _player_display_name(best)
            lines.append(f"  Best AST/TO Ratio:  {name} ({best.ast_to_ratio:.1f})")

        # Most effort plays
        if real:
            best_eff = max(real, key=lambda p: p.effort_plays)
            name = _player_display_name(best_eff)
            lines.append(
                f"  Most Effort Plays:  {name} "
                f"({best_eff.orb} ORB + {best_eff.stl} STL + "
                f"{best_eff.deflections} DEFL + {best_eff.blk} BLK = {best_eff.effort_plays})"
            )

        # Highest DAI
        if real:
            best_dai = max(real, key=lambda p: p.defensive_activity_index)
            name = _player_display_name(best_dai)
            lines.append(
                f"  Highest DAI:        {name} "
                f"({best_dai.stl} STL + {best_dai.deflections} DEFL = "
                f"{best_dai.defensive_activity_index})"
            )

        # Best Shot Selection (FG%) — min 3 attempts to be meaningful
        shooters = [p for p in real if p.fga >= 3]
        if shooters:
            best_fg = max(shooters, key=lambda p: p.fg_pct)
            name = _player_display_name(best_fg)
            lines.append(
                f"  Best Shot Selection: {name} "
                f"({best_fg.fg}-{best_fg.fga}, {best_fg.fg_pct:.0%} FG%)"
            )

        # Best REB/MIN — min 5 minutes played
        with_min = [p for p in real if p.min_played >= 5]
        if with_min:
            best_rpm = max(with_min, key=lambda p: p.reb_per_min)
            name = _player_display_name(best_rpm)
            lines.append(
                f"  Best REB/MIN:        {name} "
                f"({best_rpm.reb} REB in {best_rpm.min_played:.0f} min = "
                f"{best_rpm.reb_per_min:.2f}/min)"
            )

        return lines

    def _render_quarter_table(self, game: GameBoxScore) -> str:
        """Render quarter-by-quarter scoring table."""
        if not game.quarter_scores:
            return ""

        lines = []
        # Header
        q_labels = [q["quarter"] for q in game.quarter_scores]
        header = f"{'':>20s}" + "".join(f" {q:>5s}" for q in q_labels) + f" {'Total':>6s}"
        lines.append(header)

        # Home
        home_scores = [str(q.get("home", 0)) for q in game.quarter_scores]
        home_line = f"{game.home.team_name:>20s}" + "".join(f" {s:>5s}" for s in home_scores)
        home_line += f" {game.home.total_pts:>6d}"
        lines.append(home_line)

        # Away
        away_scores = [str(q.get("away", 0)) for q in game.quarter_scores]
        away_line = f"{game.away.team_name:>20s}" + "".join(f" {s:>5s}" for s in away_scores)
        away_line += f" {game.away.total_pts:>6d}"
        lines.append(away_line)

        return "\n".join(lines)

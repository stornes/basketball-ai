"""Game Report generator — championship-level game analysis.

Orchestrates all 8 report sections:
1. Game Summary (LLM narrative)
2. Team Box Score Totals
3. Four Factors Analysis (Dean Oliver)
4. Individual Box Score
5. Advanced Individual Metrics (TS%, eFG%, USG%, Game Score, Per-36)
6. Player Scouting Reports (LLM narrative)
7. Coaching Assessment (LLM narrative)
8. Final Verdict (MVP selections + game analysis)
+ Appendix: Statistical Methodology
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from app.analytics.advanced_stats import (
    AdvancedPlayerStats,
    FourFactors,
    GameAdvancedStats,
    TeamAdvancedStats,
    compute_game_advanced,
)
from app.analytics.box_score import GameBoxScore, PlayerBoxScore, TeamBoxScore

if TYPE_CHECKING:
    from app.reporting.coach_agent import BaseLLMClient


# ═══════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════

@dataclass
class PlayerAward:
    """An award given to a player in the Final Verdict section."""

    award_name: str
    jersey_number: int | None
    player_name: str
    team: str
    stat_line: str
    reason: str


@dataclass
class CoachingCategory:
    """One row of the coaching assessment table."""

    category: str
    grade: str
    strength: str
    area_for_growth: str


@dataclass
class FilmReport:
    """Complete game report."""

    # Metadata
    home_name: str = ""
    away_name: str = ""
    home_score: int = 0
    away_score: int = 0
    game_date: str = ""
    competition: str = ""

    # Box score data
    box_score_text: str = ""  # Full box score with impact lines + KPIs (for text/markdown output)
    kpi_highlights: list[str] = field(default_factory=list)  # Top KPI lines
    game_box_score: GameBoxScore | None = None  # Structured data for DOCX table rendering

    # Section data
    game_summary: str = ""
    quarter_scores: list[dict] = field(default_factory=list)
    advanced_stats: GameAdvancedStats | None = None
    key_stat_drivers: str = ""  # Per-factor analysis of what drove each Four Factors value
    data_sources_note: str = ""  # Transparency on data provenance
    scouting_reports: dict[str, str] = field(default_factory=dict)  # player_key → narrative
    coaching_assessment: list[CoachingCategory] = field(default_factory=list)
    coaching_overall_grade: str = ""
    awards: list[PlayerAward] = field(default_factory=list)
    losing_factor: str = ""
    winning_adjustment: str = ""
    methodology_notes: list[str] = field(default_factory=list)

    # Quarter-by-quarter coaching narratives (populated by compile_film_report.py or generate)
    quarter_narratives: list[str] = field(default_factory=list)  # Q1-Q4 coaching analysis

    # Chart image paths (populated by compile_film_report.py)
    chart_score_flow: str = ""  # path to score_flow.png
    chart_shot_all: str = ""  # path to shot_chart.png
    chart_shot_home: str = ""  # path to shot_chart_home.png
    chart_shot_away: str = ""  # path to shot_chart_away.png
    chart_shot_home_quarters: list[str] = field(default_factory=list)  # Q1-Q4 paths
    chart_shot_away_quarters: list[str] = field(default_factory=list)  # Q1-Q4 paths

    def to_dict(self) -> dict:
        return {
            "metadata": {
                "home_name": self.home_name,
                "away_name": self.away_name,
                "home_score": self.home_score,
                "away_score": self.away_score,
                "game_date": self.game_date,
                "competition": self.competition,
            },
            "box_score_text": self.box_score_text,
            "kpi_highlights": self.kpi_highlights,
            "game_summary": self.game_summary,
            "quarter_scores": self.quarter_scores,
            "advanced_stats": self.advanced_stats.to_dict() if self.advanced_stats else None,
            "key_stat_drivers": self.key_stat_drivers,
            "data_sources_note": self.data_sources_note,
            "scouting_reports": self.scouting_reports,
            "coaching_assessment": [
                {
                    "category": c.category,
                    "grade": c.grade,
                    "strength": c.strength,
                    "area_for_growth": c.area_for_growth,
                }
                for c in self.coaching_assessment
            ],
            "coaching_overall_grade": self.coaching_overall_grade,
            "awards": [
                {
                    "award_name": a.award_name,
                    "jersey_number": a.jersey_number,
                    "player_name": a.player_name,
                    "team": a.team,
                    "stat_line": a.stat_line,
                    "reason": a.reason,
                }
                for a in self.awards
            ],
            "losing_factor": self.losing_factor,
            "winning_adjustment": self.winning_adjustment,
            "methodology_notes": self.methodology_notes,
            "quarter_narratives": self.quarter_narratives,
        }


# ═══════════════════════════════════════════════════════════════════
# Report Generator
# ═══════════════════════════════════════════════════════════════════

class FilmReportGenerator:
    """Orchestrates generation of a full game report."""

    def __init__(
        self,
        llm_client: BaseLLMClient | None = None,
        competition: str = "",
        game_date: str = "",
    ):
        self.llm = llm_client
        self.competition = competition
        self.game_date = game_date

    def generate(self, game: GameBoxScore) -> FilmReport:
        """Generate the full film report from a GameBoxScore.

        Always produces a complete report with all sections: box score with
        impact lines, KPIs, advanced stats, four factors, scouting reports,
        development-focused coaching assessment, and threshold-based awards.
        """
        # Compute advanced stats — needed for grades, Game Score, and Four Factors
        advanced = compute_game_advanced(game)

        report = FilmReport(
            home_name=game.home.team_name,
            away_name=game.away.team_name,
            home_score=game.home.total_pts,
            away_score=game.away.total_pts,
            game_date=self.game_date,
            competition=self.competition,
            quarter_scores=game.quarter_scores or [],
            advanced_stats=advanced,
        )

        # Cache stat source scan (used by data_sources_note and methodology_notes)
        self._cached_stat_sources = self._scan_stat_sources(game)

        # Structured box score data and text output
        report.game_box_score = game
        report.box_score_text = self._generate_box_score_text(game)
        report.kpi_highlights = self._generate_kpi_highlights(game)

        # Full report sections
        report.key_stat_drivers = self._generate_key_stat_drivers(game, advanced)
        report.data_sources_note = self._generate_data_sources_note(game)
        report.scouting_reports = self._generate_scouting_reports(advanced)
        coaching = self._generate_coaching_assessment(game)
        report.coaching_assessment = coaching[0]
        report.coaching_overall_grade = coaching[1]
        awards_data = self._determine_awards(game)
        report.awards = awards_data[0]
        report.losing_factor = awards_data[1]
        report.winning_adjustment = awards_data[2]
        report.methodology_notes = self._methodology_notes(game)
        report.quarter_narratives = self._generate_quarter_narratives(game, advanced)
        report.game_summary = self._generate_game_summary(game, advanced, report)

        return report

    # ═══════════════════════════════════════════════════════════════════
    # Box score and KPI methods
    # ═══════════════════════════════════════════════════════════════════

    def _generate_box_score_text(self, game: GameBoxScore) -> str:
        """Generate the full box score text with impact lines and KPIs."""
        from app.reporting.box_score_renderer import BoxScoreRenderer
        renderer = BoxScoreRenderer()
        return renderer.render_text(game)

    def _generate_kpi_highlights(self, game: GameBoxScore) -> list[str]:
        """Compute top KPI highlights across both teams."""
        all_players = game.home.players + game.away.players
        real_players = [
            p for p in all_players
            if p.jersey_number is not None and p.jersey_number != 0
            and not (p.player_name and "unattributed" in p.player_name.lower())
        ]
        if not real_players:
            return []

        from app.reporting.box_score_renderer import _player_display_name

        highlights = []

        # Best AST/TO ratio
        with_playmaking = [p for p in real_players if p.to > 0 or p.ast > 0]
        if with_playmaking:
            best = max(with_playmaking, key=lambda p: p.ast_to_ratio)
            name = _player_display_name(best)
            ratio = f"{best.ast_to_ratio:.1f}"
            if best.ast_to_ratio >= 3.0:
                label = "elite decision-maker"
            elif best.ast_to_ratio >= 2.0:
                label = "strong playmaker"
            elif best.ast_to_ratio >= 1.0:
                label = "developing"
            else:
                label = "needs ball-handling work"
            highlights.append(f"Best AST/TO: {name} ({ratio}) -- {label}")

        # Highest Defensive Activity Index
        best_dai = max(real_players, key=lambda p: p.defensive_activity_index)
        if best_dai.defensive_activity_index > 0:
            name = _player_display_name(best_dai)
            dai = best_dai.defensive_activity_index
            if dai > 5:
                label = "disruptive defender"
            elif dai >= 3:
                label = "engaged"
            else:
                label = "passive on defence"
            highlights.append(
                f"Highest DAI: {name} ({best_dai.stl} STL + {best_dai.deflections} DEFL = {dai}) -- {label}"
            )

        # Most Effort Plays
        best_eff = max(real_players, key=lambda p: p.effort_plays)
        if best_eff.effort_plays > 0:
            name = _player_display_name(best_eff)
            eff = best_eff.effort_plays
            if eff > 6:
                label = "exceptional hustle"
            elif eff >= 3:
                label = "good effort"
            else:
                label = "room to grow"
            highlights.append(
                f"Most Effort Plays: {name} "
                f"({best_eff.orb} ORB + {best_eff.stl} STL + {best_eff.deflections} DEFL + {best_eff.blk} BLK = {eff}) -- {label}"
            )

        # Best Shot Selection (FG%)
        shooters = [p for p in real_players if p.fga >= 3]
        if shooters:
            best_fg = max(shooters, key=lambda p: p.fg_pct)
            name = _player_display_name(best_fg)
            pct = f"{best_fg.fg_pct:.0%}"
            if best_fg.fg_pct > 0.45:
                label = "good shot selection"
            elif best_fg.fg_pct > 0.35:
                label = "developing"
            else:
                label = "needs shot selection coaching"
            highlights.append(f"Best FG%: {name} ({pct} on {best_fg.fga} FGA) -- {label}")

        # Top Rebounder (raw count)
        rebounders = [p for p in real_players if p.reb > 0]
        if rebounders:
            best_reb = max(rebounders, key=lambda p: p.reb)
            name = _player_display_name(best_reb)
            highlights.append(f"Top Rebounder: {name} ({best_reb.reb} REB: {best_reb.orb} ORB + {best_reb.drb} DRB)")

        # Best REB/MIN (normalised rebounding rate — min 5 min to filter garbage time)
        with_min = [p for p in real_players if p.min_played >= 5 and p.reb > 0]
        if with_min:
            best_rpm = max(with_min, key=lambda p: p.reb_per_min)
            name = _player_display_name(best_rpm)
            highlights.append(
                f"Best REB/MIN: {name} "
                f"({best_rpm.reb} REB in {best_rpm.min_played:.0f} min = "
                f"{best_rpm.reb_per_min:.2f}/min)"
            )

        return highlights

    def _generate_coaching_assessment(
        self,
        game: GameBoxScore,
    ) -> tuple[list[CoachingCategory], str]:
        """Coaching assessment focused on development.

        Evaluates: Decision-Making (AST/TO), Effort & Hustle (EFF/DAI),
        Shot Selection (FG%), Team Balance, and Defensive Activity.
        Uses growth-mindset language appropriate for all levels.
        """
        categories = []

        for team in [game.home, game.away]:
            real_players = [
                p for p in team.players
                if p.jersey_number is not None and p.jersey_number != 0
                and not (p.player_name and "unattributed" in p.player_name.lower())
            ]
            scorers = sorted([p for p in real_players if p.pts > 0], key=lambda p: p.pts, reverse=True)

            # Decision-Making (AST/TO)
            team_ast = sum(p.ast for p in real_players)
            team_to = sum(p.to for p in real_players)
            team_ratio = team_ast / team_to if team_to > 0 else float(team_ast)

            if team_ratio >= 2.0:
                dm_grade = "Impact Player"
                dm_strength = f"Team AST/TO ratio {team_ratio:.1f} -- strong decision-making"
                dm_growth = "Maintain composure under defensive pressure"
            elif team_ratio >= 1.0:
                dm_grade = "Solid Foundation"
                dm_strength = f"Team AST/TO ratio {team_ratio:.1f} -- building good habits"
                dm_growth = "Reduce unforced turnovers in transition"
            else:
                dm_grade = "Developing"
                dm_strength = f"Aggressive play shows willingness to create"
                dm_growth = f"Team AST/TO ratio {team_ratio:.1f} -- focus on pass-first mentality"

            categories.append(CoachingCategory(
                category=f"Decision-Making ({team.team_name})",
                grade=dm_grade,
                strength=dm_strength,
                area_for_growth=dm_growth,
            ))

            # Effort & Hustle
            team_eff = sum(p.effort_plays for p in real_players)
            eff_per_player = team_eff / len(real_players) if real_players else 0

            if eff_per_player >= 3:
                eff_grade = "Impact Player"
                eff_strength = f"Team effort plays: {team_eff} ({eff_per_player:.1f}/player)"
                eff_growth = "Channel effort into high-percentage plays"
            elif eff_per_player >= 1.5:
                eff_grade = "Solid Foundation"
                eff_strength = f"Good effort baseline ({team_eff} total effort plays)"
                eff_growth = "Encourage more offensive rebounds and deflections"
            else:
                eff_grade = "Developing"
                eff_strength = "Room for hustle improvement"
                eff_growth = f"Only {team_eff} effort plays -- set targets for ORB, STL, DEFL"

            categories.append(CoachingCategory(
                category=f"Effort & Hustle ({team.team_name})",
                grade=eff_grade,
                strength=eff_strength,
                area_for_growth=eff_growth,
            ))

            # Shot Selection
            team_fga = sum(p.fga for p in real_players)
            team_fg = sum(p.fg for p in real_players)
            team_fg_pct = team_fg / team_fga if team_fga > 0 else 0.0

            if team_fg_pct > 0.45:
                ss_grade = "Impact Player"
                ss_strength = f"Team FG% {team_fg_pct:.0%} -- excellent shot selection"
                ss_growth = "Continue taking high-percentage shots"
            elif team_fg_pct > 0.35:
                ss_grade = "Solid Foundation"
                ss_strength = f"Team FG% {team_fg_pct:.0%} -- developing shot discipline"
                ss_growth = "Reduce contested long-range attempts"
            else:
                ss_grade = "Developing"
                ss_strength = f"Willingness to shoot ({team_fga} attempts)"
                ss_growth = f"Team FG% {team_fg_pct:.0%} -- work on getting closer to the basket"

            categories.append(CoachingCategory(
                category=f"Shot Selection ({team.team_name})",
                grade=ss_grade,
                strength=ss_strength,
                area_for_growth=ss_growth,
            ))

            # Team Balance
            if scorers:
                top_scorer_pct = (scorers[0].pts / team.total_pts * 100) if team.total_pts > 0 else 0
                if top_scorer_pct > 50:
                    bal_grade = "Developing"
                    bal_strength = f"Clear primary scorer identified ({scorers[0].player_name})"
                    bal_growth = f"{top_scorer_pct:.0f}% scoring from one player -- involve others"
                elif len(scorers) >= 4:
                    bal_grade = "Impact Player"
                    bal_strength = f"{len(scorers)} players contributed scoring"
                    bal_growth = "Maintain balanced attack"
                else:
                    bal_grade = "Solid Foundation"
                    bal_strength = f"{len(scorers)} scorers contributing"
                    bal_growth = "Develop 1-2 more scoring options"

                categories.append(CoachingCategory(
                    category=f"Team Balance ({team.team_name})",
                    grade=bal_grade,
                    strength=bal_strength,
                    area_for_growth=bal_growth,
                ))

        overall = "Solid Foundation" if any(
            c.grade == "Impact Player" for c in categories
        ) else "Developing"
        return categories, overall

    def _determine_awards(
        self,
        game: GameBoxScore,
    ) -> tuple[list[PlayerAward], str, str]:
        """Awards emphasising effort and decision-making, not just scoring.

        Uses threshold-based qualification: awards that aren't earned get
        skipped rather than force-assigned to the least-bad candidate.
        """
        from app.reporting.box_score_renderer import _player_display_name
        awards = []

        all_players = []
        for team in [game.home, game.away]:
            for p in team.players:
                if p.jersey_number is not None and p.jersey_number != 0:
                    if not (p.player_name and "unattributed" in p.player_name.lower()):
                        all_players.append((p, team))

        if not all_players:
            return [], "", ""

        # ── Award thresholds ──
        # Every award has a minimum qualifying bar. Awards that aren't
        # earned get skipped, not force-assigned to the least-bad candidate.
        # Principle: "would a coach laugh at this?" test.
        MIN_MVP_PTS = 10       # or 8 REB or 4 AST
        MIN_PLAYMAKER_AST = 3  # can't be a playmaker with 1 assist
        MIN_HUSTLE_EFF = 3     # 3+ effort plays
        MIN_SHARPSHOOTER_FGA = 5  # 5+ attempts for FG% to matter

        awarded_jerseys = set()

        # Impact MVP -- highest composite: PTS + REB + AST + STL + BLK
        # Minimum: 10+ PTS, or 8+ REB, or 4+ AST
        mvp_candidates = [
            (p, t) for p, t in all_players
            if p.pts >= MIN_MVP_PTS or p.reb >= 8 or p.ast >= 4
        ]
        if not mvp_candidates:
            # Fall back to best available (MVP always awarded)
            mvp_candidates = all_players

        mvp_sorted = sorted(
            mvp_candidates,
            key=lambda x: x[0].pts + x[0].reb + x[0].ast + x[0].stl + x[0].blk,
            reverse=True,
        )
        mvp_p, mvp_team = mvp_sorted[0]
        awards.append(PlayerAward(
            award_name="Impact MVP",
            jersey_number=mvp_p.jersey_number,
            player_name=mvp_p.player_name or "",
            team=mvp_team.team_name,
            stat_line=mvp_p.impact_line,
            reason="Highest overall impact across scoring, rebounding, and playmaking",
        ))
        awarded_jerseys.add((mvp_p.jersey_number, mvp_team.team_key))

        # Playmaker Award -- best AST/TO ratio
        # Minimum: 3+ assists (you cannot be a playmaker with 1 assist)
        playmakers = [
            (p, t) for p, t in all_players
            if p.ast >= MIN_PLAYMAKER_AST
            and (p.jersey_number, t.team_key) not in awarded_jerseys
        ]
        if playmakers:
            pm_sorted = sorted(playmakers, key=lambda x: x[0].ast_to_ratio, reverse=True)
            pm_p, pm_team = pm_sorted[0]
            awards.append(PlayerAward(
                award_name="Playmaker Award",
                jersey_number=pm_p.jersey_number,
                player_name=pm_p.player_name or "",
                team=pm_team.team_name,
                stat_line=f"{pm_p.ast} AST, {pm_p.to} TO (ratio {pm_p.ast_to_ratio:.1f})",
                reason="Best decision-making and team play",
            ))
            awarded_jerseys.add((pm_p.jersey_number, pm_team.team_key))

        # Hustle Award -- most effort plays (ORB + STL + DEFL + BLK)
        # Minimum: 3+ effort plays
        hustle_candidates = [
            (p, t) for p, t in all_players
            if p.effort_plays >= MIN_HUSTLE_EFF
            and (p.jersey_number, t.team_key) not in awarded_jerseys
        ]
        if hustle_candidates:
            h_sorted = sorted(hustle_candidates, key=lambda x: x[0].effort_plays, reverse=True)
            h_p, h_team = h_sorted[0]
            awards.append(PlayerAward(
                award_name="Hustle Award",
                jersey_number=h_p.jersey_number,
                player_name=h_p.player_name or "",
                team=h_team.team_name,
                stat_line=f"{h_p.effort_plays} effort plays ({h_p.orb} ORB + {h_p.stl} STL + {h_p.deflections} DEFL + {h_p.blk} BLK)",
                reason="Most effort plays through hustle and defensive activity",
            ))
            awarded_jerseys.add((h_p.jersey_number, h_team.team_key))

        # Sharpshooter -- best FG% with volume
        # Minimum: 5+ FGA
        shooters = [
            (p, t) for p, t in all_players
            if p.fga >= MIN_SHARPSHOOTER_FGA
            and (p.jersey_number, t.team_key) not in awarded_jerseys
        ]
        if shooters:
            fg_sorted = sorted(shooters, key=lambda x: x[0].fg_pct, reverse=True)
            fg_p, fg_team = fg_sorted[0]
            awards.append(PlayerAward(
                award_name="Sharpshooter",
                jersey_number=fg_p.jersey_number,
                player_name=fg_p.player_name or "",
                team=fg_team.team_name,
                stat_line=f"{fg_p.fg}-{fg_p.fga} FG ({fg_p.fg_pct:.0%})",
                reason="Best shooting efficiency with volume",
            ))

        # Losing factor and winning adjustment
        loser_team = game.home if game.home.total_pts < game.away.total_pts else game.away
        winner_team = game.away if loser_team is game.home else game.home

        loser_players = [
            p for p in loser_team.players
            if p.jersey_number is not None and p.jersey_number != 0
            and not (p.player_name and "unattributed" in p.player_name.lower())
        ]
        loser_scorers = sorted([p for p in loser_players if p.pts > 0], key=lambda p: p.pts, reverse=True)

        losing_factor = ""
        winning_adj = ""
        if loser_scorers:
            top_pct = (loser_scorers[0].pts / loser_team.total_pts * 100) if loser_team.total_pts > 0 else 0
            team_to = sum(p.to for p in loser_players)
            team_ast = sum(p.ast for p in loser_players)

            if top_pct > 50:
                losing_factor = f"Offensive imbalance: {top_pct:.0f}% of scoring from one player"
                winning_adj = "Develop team offence that creates looks for 3+ players per quarter"
            elif team_to > team_ast * 2:
                losing_factor = f"Ball security: {team_to} turnovers vs {team_ast} assists"
                winning_adj = "Focus on pass-first decision-making drills and reducing live-ball turnovers"
            elif loser_team.total_reb < winner_team.total_reb - 5:
                losing_factor = f"Rebounding deficit: {loser_team.total_reb} vs {winner_team.total_reb}"
                winning_adj = "Emphasise box-out technique and second-effort mentality"
            else:
                margin = abs(game.home.total_pts - game.away.total_pts)
                losing_factor = f"Outscored by {margin} points"
                winning_adj = "Address defensive consistency and offensive shot selection"

        return awards, losing_factor, winning_adj


    def _generate_game_summary(
        self,
        game: GameBoxScore,
        advanced: GameAdvancedStats,
        report: "FilmReport | None" = None,
    ) -> str:
        """Generate executive summary — 6-7 paragraphs filling ~75% of a page.

        Structure follows Barbara Minto's Pyramid Principle:
        - Lead with the governing thought (result + why)
        - Group supporting arguments logically (no repetition across paragraphs)
        - Each paragraph owns ONE analytical angle
        - Every sentence earns its place with data or insight

        Paragraph plan (each paragraph addresses a distinct finding):
        P1: Result, margin characterisation, and the decisive factor (the answer)
        P2: Quarter-by-quarter arc with turning point identification
        P3: Four Factors verdict — which dimensions decided the game
        P4: Star player analysis — offensive structure and dependency
        P5: Supporting cast and team balance comparison
        P6: Coaching assessment and the single adjustment to flip the result
        """
        home = game.home
        away = game.away
        winner = home if home.total_pts > away.total_pts else away
        loser = away if winner is home else home

        def _short(name: str) -> str:
            """Extract short team name: 'Notodden Thunders (D)' → 'Notodden'."""
            return name.split()[0] if name else name

        w_short = _short(winner.team_name)
        l_short = _short(loser.team_name)
        h_short = _short(home.team_name)
        a_short = _short(away.team_name)

        # Collect real players sorted by pts
        all_players = [(p, home.team_name) for p in home.players] + [
            (p, away.team_name) for p in away.players
        ]
        all_players.sort(key=lambda x: x[0].pts, reverse=True)
        real_players = [(p, t) for p, t in all_players
                        if p.pts > 0 and p.jersey_number != 0
                        and not (p.player_name and "unattributed" in p.player_name.lower())]
        winner_scorers = [(p, t) for p, t in real_players if t == winner.team_name]
        loser_scorers = [(p, t) for p, t in real_players if t == loser.team_name]

        if self.llm:
            prompt = self._build_summary_prompt(game, advanced)
            return self.llm.generate(prompt, system="You are a basketball analyst.")

        margin = abs(home.total_pts - away.total_pts)

        # Format date for prose
        display_date = ""
        if self.game_date:
            try:
                from datetime import datetime
                raw = self.game_date.replace("T", " ").strip()
                for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
                    try:
                        dt = datetime.strptime(raw.split("+")[0].strip(), fmt)
                        if dt.hour > 0:
                            display_date = dt.strftime(" on %-d %B %Y at %H:%M")
                        else:
                            display_date = dt.strftime(" on %-d %B %Y")
                        break
                    except ValueError:
                        continue
                if not display_date:
                    display_date = f" on {raw}"
            except Exception:
                display_date = f" on {self.game_date}"

        # ── Four Factors data ──
        hff = advanced.home.four_factors
        aff = advanced.away.four_factors
        w_ff = hff if winner is home else aff
        l_ff = aff if winner is home else hff

        # ── Identify the top scorer and dependency ──
        top_scorer = real_players[0] if real_players else None
        top_name = top_scorer[0].player_name if top_scorer else ""
        top_team = top_scorer[1] if top_scorer else ""
        top_team_short = _short(top_team)
        top_team_bs = home if top_team == home.team_name else away
        top_pct = (top_scorer[0].pts / top_team_bs.total_pts * 100) if (top_scorer and top_team_bs.total_pts > 0) else 0

        # ── Determine the decisive factor for P1's governing thought ──
        # Find which Four Factor had the biggest gap
        factor_gaps = []
        efg_gap = abs(hff.efg_pct - aff.efg_pct)
        if efg_gap > 0.02:
            factor_gaps.append(("shooting efficiency", efg_gap))
        tov_gap = abs(hff.tov_pct - aff.tov_pct) if hff.tov_grade != "N/A" else 0
        if tov_gap > 0.02:
            factor_gaps.append(("ball security", tov_gap))
        ft_gap = abs(hff.ft_rate - aff.ft_rate)
        if ft_gap > 0.05:
            factor_gaps.append(("free-throw generation", ft_gap))
        oreb_gap = abs((hff.oreb or 0) - (aff.oreb or 0))
        if oreb_gap > 3 and hff.oreb_grade != "N/A":
            factor_gaps.append(("second-chance opportunities", oreb_gap / 20))
        factor_gaps.sort(key=lambda x: x[1], reverse=True)
        decisive_factor = factor_gaps[0][0] if factor_gaps else "shooting"

        # ── P1: Governing thought — result + decisive factor ──
        if margin >= 20:
            intensity = "dominated"
        elif margin >= 10:
            intensity = "controlled"
        elif margin >= 5:
            intensity = "beat"
        else:
            intensity = "edged"

        p1 = (
            f"{winner.team_name} {intensity} {loser.team_name} "
            f"{winner.total_pts}-{loser.total_pts} in {self.competition}{display_date}, "
            f"a {margin}-point margin built primarily on superior {decisive_factor}."
        )

        # Add offensive rating comparison for context
        if w_ff.offensive_rating > 0 and l_ff.offensive_rating > 0:
            ortg_gap = w_ff.offensive_rating - l_ff.offensive_rating
            if abs(ortg_gap) > 5:
                p1 += (
                    f" {w_short} generated {w_ff.offensive_rating:.0f} points per 100 possessions "
                    f"against {l_short}'s {l_ff.offensive_rating:.0f}, "
                    f"a {abs(ortg_gap):.0f}-point efficiency gap that proved insurmountable."
                )

        # ── P2: Quarter-by-quarter arc with turning point ──
        p2 = ""
        if game.quarter_scores:
            qs = game.quarter_scores
            # Build running differential per quarter
            diffs = []  # (quarter_name, home_q, away_q, running_diff_after)
            running = 0
            for q in qs:
                h_q = q.get("home", 0)
                a_q = q.get("away", 0)
                running += (h_q - a_q)  # positive = home leads
                diffs.append((q.get("quarter", ""), h_q, a_q, running))

            # Find turning point: the quarter where the winner took decisive control
            # (largest single-quarter margin in winner's favour, or the quarter
            # where the lead became unrecoverable)
            turning_q = None
            turning_margin = 0
            closest_q = None
            closest_diff = 999
            for q_name, h_q, a_q, cum_diff in diffs:
                w_margin_q = (h_q - a_q) if winner is home else (a_q - h_q)
                if w_margin_q > turning_margin:
                    turning_margin = w_margin_q
                    turning_q = (q_name, h_q, a_q)
                # Track the closest point (for narrative tension)
                abs_diff = abs(cum_diff)
                if abs_diff < closest_diff:
                    closest_diff = abs_diff
                    closest_q = q_name

            # Narrate the arc
            q1 = qs[0] if len(qs) > 0 else {}
            q1_h, q1_a = q1.get("home", 0), q1.get("away", 0)
            q1_leader = h_short if q1_h > q1_a else a_short
            q1_margin = abs(q1_h - q1_a)

            p2_parts = []
            if q1_margin >= 5:
                p2_parts.append(
                    f"{q1_leader} opened aggressively, taking a {max(q1_h, q1_a)}-{min(q1_h, q1_a)} "
                    f"lead after the first quarter."
                )
            else:
                p2_parts.append(
                    f"The opening quarter was tightly contested at {q1_h}-{q1_a}."
                )

            # Middle quarters narrative
            if len(qs) >= 3:
                q2 = qs[1]
                q3 = qs[2]
                h2, a2 = q2.get("home", 0), q2.get("away", 0)
                h3, a3 = q3.get("home", 0), q3.get("away", 0)
                h_half2 = h2 + h3
                a_half2 = a2 + a3
                mid_diff = h_half2 - a_half2

                if abs(mid_diff) <= 3:
                    p2_parts.append(
                        f"The middle quarters were evenly matched ({h_short} {h_half2}, "
                        f"{a_short} {a_half2}), with neither side able to pull away."
                    )
                else:
                    mid_winner = h_short if mid_diff > 0 else a_short
                    trend = "extending" if mid_winner == _short(winner.team_name) else "eroding"
                    p2_parts.append(
                        f"{mid_winner} won the middle quarters {max(h_half2, a_half2)}-{min(h_half2, a_half2)}, "
                        f"gradually {trend} the margin."
                    )

            # Turning point — the decisive moment
            if turning_q and turning_margin >= 5:
                tq_name, tq_h, tq_a = turning_q
                p2_parts.append(
                    f"The turning point came in {tq_name}: {w_short} outscored {l_short} "
                    f"{max(tq_h, tq_a)}-{min(tq_h, tq_a)}, "
                    f"a {turning_margin}-point swing that broke the game open."
                )
            elif len(qs) >= 4:
                q4 = qs[3]
                h4, a4 = q4.get("home", 0), q4.get("away", 0)
                q4_diff = abs(h4 - a4)
                q4_winner_short = h_short if h4 > a4 else a_short
                if q4_diff >= 5:
                    p2_parts.append(
                        f"{q4_winner_short} sealed the result in Q4, outscoring the opposition "
                        f"{max(h4, a4)}-{min(h4, a4)}."
                    )

            # Final score trajectory
            if len(qs) >= 4:
                half1_h = sum(q.get("home", 0) for q in qs[:2])
                half1_a = sum(q.get("away", 0) for q in qs[:2])
                half2_h = sum(q.get("home", 0) for q in qs[2:4])
                half2_a = sum(q.get("away", 0) for q in qs[2:4])
                ht_leader = h_short if half1_h > half1_a else a_short
                same_direction = (half1_h > half1_a) == (half2_h > half2_a)
                if same_direction:
                    half2_note = "the same direction"
                else:
                    half2_winner = h_short if half2_h > half2_a else a_short
                    half2_note = f"{half2_winner}'s favour"
                p2_parts.append(
                    f"At the half, {ht_leader} led {max(half1_h, half1_a)}-{min(half1_h, half1_a)}; "
                    f"the second half finished {max(half2_h, half2_a)}-{min(half2_h, half2_a)} "
                    f"in {half2_note}."
                )

            p2 = " ".join(p2_parts)

        # ── P3: Four Factors verdict ──
        p3_parts = []

        # eFG%
        if efg_gap > 0.02:
            better_efg = home if hff.efg_pct > aff.efg_pct else away
            worse_efg = away if better_efg is home else home
            better_ff_efg = hff if better_efg is home else aff
            worse_ff_efg = aff if better_efg is home else hff
            p3_parts.append(
                f"Dean Oliver's Four Factors reveal clear separation. "
                f"{_short(better_efg.team_name)} shot {better_ff_efg.efg_pct:.1%} eFG% "
                f"against {_short(worse_efg.team_name)}'s {worse_ff_efg.efg_pct:.1%} — "
                f"a {efg_gap:.1%} gap that, at {advanced.home.four_factors.est_possessions:.0f} "
                f"estimated possessions, translates to roughly "
                f"{abs(efg_gap * 2 * max(hff.est_possessions, aff.est_possessions)):.0f} "
                f"points of shooting advantage alone."
            )
        else:
            p3_parts.append(
                "Shooting efficiency was comparable between the two sides."
            )

        # FT rate
        h_fta, a_fta = home.total_fta, away.total_fta
        if h_fta + a_fta > 0:
            more_fta = home if h_fta > a_fta else away
            less_fta = away if more_fta is home else home
            if abs(h_fta - a_fta) > 5:
                p3_parts.append(
                    f"{_short(more_fta.team_name)} drew {more_fta.total_fta} free-throw attempts "
                    f"to {less_fta.total_fta}, converting {more_fta.total_ft} "
                    f"({more_fta.total_ft / more_fta.total_fta * 100:.0f}% FT)."
                )
            else:
                p3_parts.append(
                    f"Free-throw generation was balanced ({h_fta} vs {a_fta} attempts)."
                )

        # TOV and OREB
        tov_na = hff.tov_grade == "N/A" and aff.tov_grade == "N/A"
        oreb_na = hff.oreb_grade == "N/A" and aff.oreb_grade == "N/A"
        if not tov_na:
            worse_tov = home if hff.tov_pct > aff.tov_pct else away
            better_tov = away if worse_tov is home else home
            worse_tov_pct = hff.tov_pct if worse_tov is home else aff.tov_pct
            better_tov_pct = aff.tov_pct if worse_tov is home else hff.tov_pct
            if abs(hff.tov_pct - aff.tov_pct) > 0.03:
                p3_parts.append(
                    f"Ball security was a differentiator: {_short(worse_tov.team_name)} "
                    f"turned it over at {worse_tov_pct:.1%} versus "
                    f"{_short(better_tov.team_name)}'s {better_tov_pct:.1%}."
                )
        if not oreb_na:
            h_oreb = hff.oreb or 0
            a_oreb = aff.oreb or 0
            if abs(h_oreb - a_oreb) > 3:
                more_oreb = home if h_oreb > a_oreb else away
                p3_parts.append(
                    f"{_short(more_oreb.team_name)} dominated the offensive glass with "
                    f"{max(h_oreb, a_oreb)} offensive rebounds to {min(h_oreb, a_oreb)}, "
                    f"generating crucial second-chance points."
                )

        p3 = " ".join(p3_parts)

        # ── P4: Star player — offensive structure and dependency ──
        p4_parts = []
        if top_scorer:
            p = top_scorer[0]
            fg_str = f"{p.fg}-{p.fga}" if p.fga > 0 else f"{p.fg} FGM"
            three_str = ""
            if hasattr(p, 'three_p') and p.three_p > 0:
                tpa = getattr(p, 'three_pa', p.three_p)
                three_str = f", {p.three_p}-{tpa} from three"

            ft_str = ""
            if p.fta > 0:
                ft_str = f", {p.ft}-{p.fta} FT"

            # Game Score context from advanced stats
            gs_str = ""
            for team_adv in [advanced.home, advanced.away]:
                for ps in team_adv.player_stats:
                    if ps.jersey_number == p.jersey_number and team_adv.team_name == top_team:
                        gs_str = f" (Game Score {ps.game_score:.1f})"
                        break

            p4_parts.append(
                f"#{p.jersey_number} {top_name} was the game's dominant force, posting "
                f"{p.pts} points on {fg_str} shooting{three_str}{ft_str}{gs_str}."
            )

            if top_pct > 50:
                p4_parts.append(
                    f"That represents {top_pct:.0f}% of {top_team_short}'s total offence — "
                    f"a level of dependency that, while testament to individual quality, "
                    f"signals a structural vulnerability opponents will exploit."
                )
            elif top_pct > 35:
                p4_parts.append(
                    f"Accounting for {top_pct:.0f}% of {top_team_short}'s scoring, "
                    f"the offensive load was heavily concentrated."
                )

        p4 = " ".join(p4_parts)

        # ── P5: Supporting cast and team balance ──
        p5_parts = []

        # Winner's balance
        if len(winner_scorers) >= 2:
            w_top = winner_scorers[0][0]
            w_others = [p for p, _ in winner_scorers[1:] if p.pts > 0]
            w_others_pts = sum(p.pts for p in w_others)
            double_digit = [p for p in w_others if p.pts >= 10]

            if len(double_digit) >= 2:
                dd_names = [f"{p.player_name or f'#{p.jersey_number}'} ({p.pts})" for p in double_digit[:3]]
                p5_parts.append(
                    f"{w_short}'s offence was balanced: beyond {w_top.player_name or f'#{w_top.jersey_number}'}'s "
                    f"{w_top.pts}, {', '.join(dd_names)} all reached double figures."
                )
            elif w_others:
                second = w_others[0]
                p5_parts.append(
                    f"{w_short}'s secondary scoring came from "
                    f"{second.player_name or f'#{second.jersey_number}'} with {second.pts} points."
                )

        # Loser's balance (or lack thereof)
        if len(loser_scorers) >= 2:
            l_top = loser_scorers[0][0]
            l_top_pct = (l_top.pts / loser.total_pts * 100) if loser.total_pts > 0 else 0
            l_others = [p for p, _ in loser_scorers[1:] if p.pts > 0]
            l_second = l_others[0] if l_others else None

            if l_top_pct > 50 and l_second:
                gap = l_top.pts - l_second.pts
                p5_parts.append(
                    f"For {l_short}, the {gap}-point gulf between {l_top.player_name or f'#{l_top.jersey_number}'} "
                    f"({l_top.pts}) and the next scorer "
                    f"{l_second.player_name or f'#{l_second.jersey_number}'} ({l_second.pts}) "
                    f"illustrates the balance problem."
                )
            elif l_second:
                p5_parts.append(
                    f"{l_short}'s secondary options were limited, with "
                    f"{l_second.player_name or f'#{l_second.jersey_number}'} contributing {l_second.pts} "
                    f"as the next highest scorer."
                )

        p5 = " ".join(p5_parts)

        # ── P6: Coaching assessment and the adjustment ──
        p6_parts = []
        if report and report.coaching_overall_grade:
            p6_parts.append(
                f"The coaching assessment grades the overall performance as "
                f"{report.coaching_overall_grade}."
            )

        if report and report.awards:
            mvp = report.awards[0]
            mvp_line = mvp.stat_line if mvp.stat_line else f"{mvp.player_name}"
            p6_parts.append(
                f"#{mvp.jersey_number} {mvp.player_name} ({_short(mvp.team)}) earns the {mvp.award_name} "
                f"selection on the strength of {mvp_line}."
            )
            # Second award if different player
            if len(report.awards) >= 2 and report.awards[1].jersey_number != mvp.jersey_number:
                second = report.awards[1]
                p6_parts.append(
                    f"The {second.award_name} goes to #{second.jersey_number} {second.player_name} "
                    f"({second.stat_line})."
                )

        if report and report.losing_factor:
            p6_parts.append(
                f"The decisive losing factor: {report.losing_factor}."
            )
        if report and report.winning_adjustment:
            p6_parts.append(
                f"To change the outcome, {l_short} must {report.winning_adjustment[0].lower()}{report.winning_adjustment[1:]}."
            )

        p6 = " ".join(p6_parts)

        paragraphs = [p for p in [p1, p2, p3, p4, p5, p6] if p]
        return "\n\n".join(paragraphs)

    def _build_summary_prompt(
        self,
        game: GameBoxScore,
        advanced: GameAdvancedStats,
    ) -> str:
        """Build LLM prompt for game summary."""
        lines = [
            "Write a 6-paragraph executive summary for this game report.",
            "Target length: 75% of a full page (~400-500 words).",
            "Each paragraph covers ONE distinct analytical angle — no repetition.",
            "",
            "PARAGRAPH STRUCTURE:",
            "P1: Result + decisive factor + offensive rating gap (the governing thought).",
            "P2: Quarter-by-quarter arc. Identify the TURNING POINT — the specific",
            "    quarter/run where the game shifted irreversibly. Show basketball IQ.",
            "P3: Four Factors verdict — eFG%, TOV%, FT Rate, OREB. Which dimensions",
            "    decided the game? Quantify the impact (e.g., 'a 7% eFG gap over 80",
            "    possessions translates to ~11 points of shooting advantage').",
            "P4: Star player analysis — stat line, dependency %, and what it means",
            "    structurally for the team's offence.",
            "P5: Supporting cast — team balance comparison. Who stepped up, who didn't.",
            "P6: Coaching assessment grade, MVP selections, and the single strategic",
            "    adjustment that would flip the result.",
            "",
            "CRITICAL RULES:",
            "- UK English throughout (analyse, defence, colour).",
            "- Barbara Minto Pyramid Principle: lead with the answer, group logically.",
            "- Every sentence must contain data or insight — no filler, no hedging.",
            "- Use growth-mindset language. NEVER use letter grades (A, B, C, D).",
            "  Use: Impact Player, Solid Foundation, Developing, Growth Opportunity.",
            "- Show deep basketball knowledge: reference offensive structure, pace,",
            "  half-court vs transition, shot selection, balance vs star-dependency.",
            "- NO repetition between paragraphs. Each paragraph owns its angle.",
            "",
            f"Game: {game.home.team_name} {game.home.total_pts} vs "
            f"{game.away.team_name} {game.away.total_pts}",
            f"Competition: {self.competition}",
            "",
            "Key stats:",
        ]

        for team_adv in [advanced.home, advanced.away]:
            ff = team_adv.four_factors
            lines.append(
                f"  {team_adv.team_name}: eFG% {ff.efg_pct:.1%} ({ff.efg_grade}), "
                f"TOV% {ff.tov_pct:.1%} ({ff.tov_grade}), Off Rating {ff.offensive_rating:.1f}"
            )
            # Top 3 scorers
            for p in team_adv.player_stats[:3]:
                if p.pts > 0:
                    lines.append(
                        f"    #{p.jersey_number} {p.player_name}: "
                        f"{p.pts}pts, GmScr {p.game_score:.1f}, TS% {p.ts_pct:.1%}, "
                        f"Grade: {p.grade}"
                    )

        return "\n".join(lines)

    def _generate_quarter_narratives(
        self,
        game: GameBoxScore,
        advanced: GameAdvancedStats,
    ) -> list[str]:
        """Generate per-quarter coaching analysis narratives.

        Each narrative is 3-5 sentences explaining the flow and tactical
        dynamics of that quarter, combining statistical evidence with
        coaching insight.  When no per-quarter box-score splits are
        available, the narrative synthesises from quarter scores and
        full-game tendencies.
        """
        qs = game.quarter_scores or []
        if not qs:
            return []

        h_name = game.home.team_name
        a_name = game.away.team_name

        # Helper: short team name (first word)
        def _short(name: str) -> str:
            return name.split()[0] if name else name

        h_short = _short(h_name)
        a_short = _short(a_name)

        # Full-game context for cross-referencing
        h_ff = advanced.home.four_factors if advanced else None
        a_ff = advanced.away.four_factors if advanced else None
        h_team = advanced.home if advanced else None
        a_team = advanced.away if advanced else None

        # ── Build ranked causal factors from full-game stats ──
        # These explain WHY one team outperformed the other.
        causal_factors: list[str] = []
        if h_ff and a_ff and h_team and a_team:
            game_winner_home = game.home.total_pts > game.away.total_pts
            w_short = h_short if game_winner_home else a_short
            l_short = a_short if game_winner_home else h_short
            w_ff = h_ff if game_winner_home else a_ff
            l_ff = a_ff if game_winner_home else h_ff
            w_team = h_team if game_winner_home else a_team
            l_team = a_team if game_winner_home else h_team

            # 1. Shooting efficiency (eFG%)
            efg_gap = w_ff.efg_pct - l_ff.efg_pct
            if abs(efg_gap) >= 0.03:
                better = w_short if efg_gap > 0 else l_short
                worse = l_short if efg_gap > 0 else w_short
                causal_factors.append(
                    f"{better}'s shooting efficiency ({max(w_ff.efg_pct, l_ff.efg_pct):.0%} eFG%) "
                    f"significantly outpaced {worse} ({min(w_ff.efg_pct, l_ff.efg_pct):.0%})"
                )

            # 2. Turnovers
            tov_gap = w_ff.tov_pct - l_ff.tov_pct
            if abs(tov_gap) >= 0.03:
                careless = w_short if tov_gap > 0 else l_short
                secure = l_short if tov_gap > 0 else w_short
                causal_factors.append(
                    f"{careless}'s ball-handling issues ({max(w_ff.tov_pct, l_ff.tov_pct):.0%} TOV rate "
                    f"vs {min(w_ff.tov_pct, l_ff.tov_pct):.0%}) created extra possessions for {secure}"
                )

            # 3. Offensive rebounds / second chances
            orb_gap = (w_team.total_orb or 0) - (l_team.total_orb or 0)
            if abs(orb_gap) >= 3:
                dom = w_short if orb_gap > 0 else l_short
                causal_factors.append(
                    f"{dom} dominated the offensive glass "
                    f"({max(w_team.total_orb, l_team.total_orb)} OREB vs "
                    f"{min(w_team.total_orb, l_team.total_orb)}), generating second-chance points"
                )

            # 4. Free throw generation
            ft_gap = w_ff.ft_rate - l_ff.ft_rate
            if abs(ft_gap) >= 0.05:
                aggressor = w_short if ft_gap > 0 else l_short
                causal_factors.append(
                    f"{aggressor}'s aggressiveness drew fouls at a higher rate "
                    f"({max(w_ff.ft_rate, l_ff.ft_rate):.0%} FT rate vs "
                    f"{min(w_ff.ft_rate, l_ff.ft_rate):.0%})"
                )

            # 5. Assists / ball movement
            ast_gap = (w_team.total_ast or 0) - (l_team.total_ast or 0)
            if abs(ast_gap) >= 3:
                mover = w_short if ast_gap > 0 else l_short
                iso = l_short if ast_gap > 0 else w_short
                causal_factors.append(
                    f"{mover}'s ball movement ({max(w_team.total_ast, l_team.total_ast)} AST vs "
                    f"{min(w_team.total_ast, l_team.total_ast)}) created higher-quality looks "
                    f"than {iso}'s more isolation-heavy approach"
                )

        narratives: list[str] = []
        cumul_h = 0
        cumul_a = 0

        for i, q in enumerate(qs):
            q_label = q.get("quarter", f"Q{i + 1}")
            h_pts = q.get("home", 0)
            a_pts = q.get("away", 0)
            cumul_h += h_pts
            cumul_a += a_pts
            diff = h_pts - a_pts
            margin = cumul_h - cumul_a

            # Determine quarter winner
            if diff > 0:
                q_winner = h_short
                q_loser = a_short
                q_diff = diff
            elif diff < 0:
                q_winner = a_short
                q_loser = h_short
                q_diff = abs(diff)
            else:
                q_winner = None
                q_diff = 0

            # Build narrative per quarter
            if i == 0:  # Q1
                if q_winner:
                    opener = (
                        f"{q_winner} established early control, outscoring "
                        f"{q_loser} {max(h_pts, a_pts)}-{min(h_pts, a_pts)} in the opening quarter."
                    )
                    if q_diff >= 8:
                        article = "An" if q_diff in (8, 11, 18) else "A"
                        detail = (
                            f"{article} {q_diff}-point first-quarter margin suggests "
                            f"{q_loser} struggled with initial defensive assignments "
                            f"and transition coverage, allowing {q_winner} to find "
                            f"rhythm before {q_loser} could adjust."
                        )
                    else:
                        detail = (
                            f"The {q_diff}-point edge was built through disciplined "
                            f"half-court execution and early defensive pressure."
                        )
                else:
                    opener = (
                        f"An evenly contested opening quarter finished {h_pts}-{a_pts}, "
                        f"with neither side establishing a clear advantage."
                    )
                    detail = (
                        "Both teams were feeling each other out, running primary "
                        "actions without committing to aggressive defensive schemes."
                    )

                # Add causal factor if available (use first factor for Q1)
                cause = ""
                if causal_factors:
                    cause = f" Full-game data suggests a key driver: {causal_factors[0]}."

                narratives.append(
                    f"{opener} {detail}{cause} "
                    f"Score after Q1: {cumul_h}-{cumul_a}."
                )

            elif i == 1:  # Q2
                if q_winner:
                    if q_winner == _short(game.home.team_name if diff > 0 else game.away.team_name):
                        # Did Q2 winner also win Q1? Or was this a response?
                        q1_diff = qs[0].get("home", 0) - qs[0].get("away", 0)
                        q1_winner_was_home = q1_diff > 0
                        q2_winner_is_home = diff > 0

                        if q1_winner_was_home == q2_winner_is_home:
                            flow = (
                                f"{q_winner} continued to press their advantage "
                                f"in the second quarter ({max(h_pts, a_pts)}-{min(h_pts, a_pts)}), "
                                f"extending their lead."
                            )
                        else:
                            flow = (
                                f"{q_winner} responded emphatically in Q2, winning "
                                f"the quarter {max(h_pts, a_pts)}-{min(h_pts, a_pts)} "
                                f"to claw back into the contest."
                            )
                    else:
                        flow = (
                            f"The second quarter belonged to {q_winner} "
                            f"({max(h_pts, a_pts)}-{min(h_pts, a_pts)}), "
                            f"shifting the momentum."
                        )
                else:
                    flow = (
                        f"Q2 mirrored the opening: a {h_pts}-{a_pts} stalemate "
                        f"that maintained the status quo."
                    )

                half_note = f"Half-time score: {cumul_h}-{cumul_a}."
                if abs(margin) <= 3:
                    half_assessment = "A tight half indicates evenly matched squads — the second half will be decided by adjustments and composure."
                elif abs(margin) <= 8:
                    leader = h_short if margin > 0 else a_short
                    half_assessment = f"{leader} takes a manageable lead into the break, but the deficit is well within striking distance."
                else:
                    leader = h_short if margin > 0 else a_short
                    trailer = a_short if margin > 0 else h_short
                    half_assessment = f"{leader} holds a commanding {abs(margin)}-point cushion. {trailer} needs a tactical reset to stay competitive."

                narratives.append(f"{flow} {half_note} {half_assessment}")

            elif i == 2:  # Q3
                if q_winner:
                    opener = (
                        f"The third quarter saw {q_winner} win the period "
                        f"{max(h_pts, a_pts)}-{min(h_pts, a_pts)}."
                    )
                    if q_diff >= 6:
                        detail = (
                            f"A {q_diff}-point quarter suggests a decisive "
                            f"half-time adjustment — whether in defensive coverage, "
                            f"tempo, or primary ball-handler usage."
                        )
                    else:
                        detail = (
                            "The narrow quarter margin indicates both coaching staffs "
                            "made effective adjustments coming out of the break."
                        )
                else:
                    opener = f"Q3 was perfectly balanced at {h_pts}-{a_pts}."
                    detail = (
                        "Both teams neutralised each other's second-half openers, "
                        "suggesting well-matched coaching adjustments."
                    )

                # Add second causal factor if available
                cause = ""
                if len(causal_factors) >= 2:
                    cause = f" Across the game, {causal_factors[1]}."

                narratives.append(
                    f"{opener} {detail}{cause} "
                    f"Score entering Q4: {cumul_h}-{cumul_a}."
                )

            elif i == 3:  # Q4
                if q_winner:
                    # Was this a close-out or a comeback?
                    entering_margin = (cumul_h - h_pts) - (cumul_a - a_pts)
                    final_margin = cumul_h - cumul_a
                    q_winner_is_home = diff > 0
                    game_winner_is_home = cumul_h > cumul_a

                    if q_winner_is_home == game_winner_is_home:
                        # Q4 winner is game winner
                        if abs(entering_margin) <= 5:
                            opener = (
                                f"{q_winner} seized control in the decisive fourth quarter, "
                                f"outscoring {q_loser} {max(h_pts, a_pts)}-{min(h_pts, a_pts)} "
                                f"to turn a tight game into a comfortable victory."
                            )
                        else:
                            opener = (
                                f"{q_winner} closed the game professionally, winning "
                                f"Q4 {max(h_pts, a_pts)}-{min(h_pts, a_pts)} to seal the result."
                            )
                    else:
                        # Q4 winner lost the game — a late rally that fell short
                        opener = (
                            f"{q_winner} made a late push, winning Q4 "
                            f"{max(h_pts, a_pts)}-{min(h_pts, a_pts)}, "
                            f"but the deficit proved too large to overcome."
                        )
                else:
                    opener = (
                        f"A {h_pts}-{a_pts} fourth quarter maintained the existing gap."
                    )

                closer = (
                    f"Final score: {h_name} {cumul_h} — {a_name} {cumul_a}."
                )

                # Add tactical observation with statistical WHY
                game_winner = h_short if cumul_h > cumul_a else a_short
                game_loser = a_short if cumul_h > cumul_a else h_short
                final_gap = abs(cumul_h - cumul_a)

                # Build causal summary for Q4 from remaining factors
                remaining_causes = causal_factors[2:] if len(causal_factors) > 2 else []
                if remaining_causes:
                    cause_summary = "; ".join(remaining_causes)
                    tactic = (
                        f"The {final_gap}-point final margin was driven by statistical advantages: "
                        f"{cause_summary}. "
                        f"{game_loser} will need to address these areas to compete in future meetings."
                    )
                elif causal_factors:
                    # Reuse the primary factor as a closing summary
                    tactic = (
                        f"The decisive factor in this game: {causal_factors[0]}. "
                        f"{game_loser} must address this disparity to close the gap."
                    )
                elif final_gap >= 10:
                    tactic = (
                        f"The {final_gap}-point final margin reflects "
                        f"{game_winner}'s superior execution across multiple phases. "
                        f"{game_loser} will need to address late-game execution "
                        f"and scoring distribution to compete in future meetings."
                    )
                else:
                    tactic = (
                        "A single-digit margin shows this was a competitive contest throughout, "
                        "decided by marginal execution differences."
                    )

                narratives.append(f"{opener} {tactic} {closer}")

        return narratives

    def _generate_key_stat_drivers(
        self,
        game: GameBoxScore,
        advanced: GameAdvancedStats,
    ) -> str:
        """Generate Key Stat Drivers — one paragraph per Four Factor.

        Explains WHAT drove each factor's value, not just the grade.
        Includes player-level contributions and counterfactuals.
        """
        home = advanced.home
        away = advanced.away
        hff = home.four_factors
        aff = away.four_factors

        if self.llm:
            prompt = self._build_stat_drivers_prompt(game, advanced)
            return self.llm.generate(
                prompt,
                system=(
                    "You are a basketball analyst. Write in UK English. "
                    "Be concise and direct in the style of Barbara Minto — "
                    "lead with the answer, group logically, cut every unnecessary word."
                ),
            )

        # Template fallback — compute drivers from data
        sections = []

        # eFG% driver
        efg_gap = abs(hff.efg_pct - aff.efg_pct)
        efg_leader = home if hff.efg_pct > aff.efg_pct else away
        efg_trailer = away if efg_leader is home else home
        efg_leader_ff = hff if efg_leader is home else aff
        efg_trailer_ff = aff if efg_leader is home else hff

        # Find top 3PT contributor
        all_players = efg_leader.player_stats
        three_pt_leaders = sorted(
            [p for p in all_players if p.three_p > 0],
            key=lambda p: p.three_p, reverse=True,
        )
        three_note = ""
        if three_pt_leaders:
            top3 = three_pt_leaders[0]
            three_note = (
                f" {top3.player_name}'s {top3.three_p}-for-{top3.three_pa} from three "
                f"inflates the team eFG% significantly — each made three counts as "
                f"1.5 field goals in the formula."
            )

        sections.append(
            f"**eFG% (40% weight) — {'The deciding factor' if efg_gap > 0.10 else 'Key differentiator'}.**"
            f" {efg_leader.team_name}'s {efg_leader_ff.efg_pct:.1%} eFG% "
            f"(\"{efg_leader_ff.efg_grade}\") vs {efg_trailer.team_name}'s "
            f"{efg_trailer_ff.efg_pct:.1%} (\"{efg_trailer_ff.efg_grade}\") — "
            f"a {efg_gap:.1%} gap.{three_note}"
        )

        # TOV% driver — skip if no turnover data
        if hff.tov_grade == "N/A" and aff.tov_grade == "N/A":
            sections.append(
                "**TOV% (25% weight) — Data not available.** "
                "No turnover data was recorded for this game. "
                "This factor cannot be assessed."
            )
        else:
            tov_gap = abs(hff.tov_pct - aff.tov_pct)
            if tov_gap < 0.03:
                tov_impact = "This factor was not decisive."
            else:
                tov_better = home if hff.tov_pct < aff.tov_pct else away
                tov_impact = f"{tov_better.team_name} held the edge in ball security."

            sections.append(
                f"**TOV% (25% weight) — "
                f"{'Negligible gap' if tov_gap < 0.03 else 'Slight edge to ' + (home.team_name if hff.tov_pct < aff.tov_pct else away.team_name)}.**"
                f" {home.team_name} {hff.tov_pct:.1%} (\"{hff.tov_grade}\") vs "
                f"{away.team_name} {aff.tov_pct:.1%} (\"{aff.tov_grade}\"). "
                f"{tov_impact}"
            )

        # FT Rate driver
        ft_leader = home if hff.ft_rate > aff.ft_rate else away
        ft_leader_ff = hff if ft_leader is home else aff
        ft_trailer = away if ft_leader is home else home
        ft_trailer_ff = aff if ft_leader is home else hff
        ft_team_box = game.home if ft_leader is home else game.away

        ft_pct = ft_team_box.total_ft / ft_team_box.total_fta if ft_team_box.total_fta > 0 else 0
        ft_counterfactual = ""
        if ft_pct < 0.65 and ft_team_box.total_fta >= 10:
            missed = ft_team_box.total_fta - ft_team_box.total_ft
            at_65 = round(ft_team_box.total_fta * 0.65) - ft_team_box.total_ft
            if at_65 > 0:
                ft_counterfactual = (
                    f" Had they shot 65% from the line, that's {at_65} additional points — "
                    f"reducing the final margin from {abs(game.home.total_pts - game.away.total_pts)} "
                    f"to {abs(game.home.total_pts - game.away.total_pts) - at_65}."
                )

        sections.append(
            f"**FT Rate (15% weight) — {ft_leader.team_name}'s strongest factor.**"
            f" {ft_leader.team_name}'s {ft_leader_ff.ft_rate:.1%} (\"{ft_leader_ff.ft_rate_grade}\") vs "
            f"{ft_trailer.team_name}'s {ft_trailer_ff.ft_rate:.1%} (\"{ft_trailer_ff.ft_rate_grade}\"). "
            f"{ft_team_box.team_name} attempted {ft_team_box.total_fta} free throws "
            f"and converted {ft_pct:.0%}.{ft_counterfactual}"
        )

        # OREB driver — skip if no rebound data
        if hff.oreb_grade == "N/A" and aff.oreb_grade == "N/A":
            sections.append(
                "**OREB (20% weight) — Data not available.** "
                "No offensive rebound data was recorded for this game. "
                "This factor cannot be assessed."
            )
        else:
            oreb_gap = abs(hff.oreb - aff.oreb)
            sections.append(
                f"**OREB (20% weight) — "
                f"{'Negligible impact' if oreb_gap < 5 else 'Notable gap'}.**"
                f" {home.team_name} {hff.oreb} (\"{hff.oreb_grade}\") vs "
                f"{away.team_name} {aff.oreb} (\"{aff.oreb_grade}\"). "
                f"{'Neither team dominated the offensive glass.' if oreb_gap < 5 else ''}"
            )

        return "\n\n".join(sections)

    def _build_stat_drivers_prompt(
        self,
        game: GameBoxScore,
        advanced: GameAdvancedStats,
    ) -> str:
        """Build LLM prompt for Key Stat Drivers."""
        home = advanced.home
        away = advanced.away
        hff = home.four_factors
        aff = away.four_factors

        # Top shooters for context
        home_3pt = sorted(
            [p for p in home.player_stats if p.three_p > 0],
            key=lambda p: p.three_p, reverse=True,
        )[:3]
        away_3pt = sorted(
            [p for p in away.player_stats if p.three_p > 0],
            key=lambda p: p.three_p, reverse=True,
        )[:3]

        lines = [
            "Write Key Stat Drivers for the Four Factors analysis.",
            "One paragraph per factor: eFG% (40% weight), TOV% (25%), FT Rate (15%), OREB (20%).",
            "",
            "For each factor: explain WHAT drove the value, name specific players,",
            "and include one counterfactual where relevant (e.g., 'Had they shot 65% FT...').",
            "",
            f"Score: {home.team_name} {home.total_pts} — {away.team_name} {away.total_pts}",
            f"Margin: {abs(home.total_pts - away.total_pts)} points",
            "",
            f"{home.team_name}: eFG% {hff.efg_pct:.1%} ({hff.efg_grade}), "
            f"TOV% {hff.tov_pct:.1%} ({hff.tov_grade}), "
            f"FT Rate {hff.ft_rate:.1%} ({hff.ft_rate_grade}), "
            f"OREB {hff.oreb} ({hff.oreb_grade})",
            f"  FT: {game.home.total_ft}-{game.home.total_fta} "
            f"({game.home.total_ft / game.home.total_fta:.0%} conversion)" if game.home.total_fta > 0 else "",
        ]
        for p in home_3pt:
            lines.append(f"  {p.player_name}: {p.three_p}-{p.three_pa} 3PT")

        lines.extend([
            "",
            f"{away.team_name}: eFG% {aff.efg_pct:.1%} ({aff.efg_grade}), "
            f"TOV% {aff.tov_pct:.1%} ({aff.tov_grade}), "
            f"FT Rate {aff.ft_rate:.1%} ({aff.ft_rate_grade}), "
            f"OREB {aff.oreb} ({aff.oreb_grade})",
            f"  FT: {game.away.total_ft}-{game.away.total_fta} "
            f"({game.away.total_ft / game.away.total_fta:.0%} conversion)" if game.away.total_fta > 0 else "",
        ])
        for p in away_3pt:
            lines.append(f"  {p.player_name}: {p.three_p}-{p.three_pa} 3PT")

        lines.extend([
            "",
            "Format: Bold header per factor with weight and verdict, then 3-5 sentences.",
            "Write in UK English. Be concise — lead with the conclusion per Minto's Pyramid Principle.",
        ])
        return "\n".join(lines)

    # Counting-stat keys — the stats that matter for "scorekeeper data present"
    _COUNTING_STAT_KEYS = {"ast", "orb", "drb", "stl", "to", "blk", "pf", "min_played"}

    @staticmethod
    def _scan_stat_sources(game: GameBoxScore) -> tuple[bool, bool, bool]:
        """Scan all players for manual and heuristic stat sources.

        Returns (has_manual_counting, has_heuristic_counting, has_manual_scoring).
        - has_manual_counting: True if any counting stat (AST/REB/STL/TO/BLK/PF/MIN)
          has StatSource.MANUAL — i.e. a real scorekeeper provided data.
        - has_heuristic_counting: True if any counting stat has StatSource.HEURISTIC
          — i.e. pipeline detectors produced estimates.
        - has_manual_scoring: True if any scoring stat (fg/ft/three_p) has
          StatSource.MANUAL — i.e. API ground truth was merged.
        """
        from app.analytics.box_score import StatSource

        has_manual_counting = False
        has_heuristic_counting = False
        has_manual_scoring = False
        counting_keys = FilmReportGenerator._COUNTING_STAT_KEYS
        for p in game.home.players + game.away.players:
            for key, source in p.stat_sources.items():
                if key in counting_keys:
                    if source == StatSource.MANUAL:
                        has_manual_counting = True
                    if source == StatSource.HEURISTIC:
                        has_heuristic_counting = True
                else:
                    if source == StatSource.MANUAL:
                        has_manual_scoring = True
            if has_manual_counting and has_heuristic_counting and has_manual_scoring:
                break
        return has_manual_counting, has_heuristic_counting, has_manual_scoring

    def _generate_data_sources_note(self, game: GameBoxScore) -> str:
        """Generate Data Sources and Limitations section.

        Explains provenance of each stat category: API, pipeline, manual, estimated.
        Uses three-tier model: API scoring → pipeline heuristic → manual scorekeeper.
        """
        has_manual_counting, has_heuristic_counting, has_manual_scoring = (
            getattr(self, '_cached_stat_sources', None) or self._scan_stat_sources(game)
        )

        parts = []

        # Tier 1: API scoring data
        if has_manual_scoring:
            parts.append(
                "**Scoring (FGM, 3PM, FTM, FTA):** Norwegian Basketball Federation API "
                "(kamper.basket.no) — ground truth. The API records made shots only; "
                "FGA per player is unknown. Where FGA is shown, it equals FGM (the "
                "minimum possible) which can produce eFG% above 100% for players "
                "with three-pointers — this is mathematically correct but reflects "
                "incomplete attempt data, not superhuman efficiency."
            )
        else:
            parts.append(
                "**Scoring (FGM, FGA):** HoopsVision AI pipeline detection. "
                "FGM classified via ball-through-hoop proximity; FGA from shot "
                "arc detection."
            )

        parts.append(
            "**FGA and shot locations:** HoopsVision AI pipeline (YOLOv8 fine-tuned + "
            "VLM jersey recognition). Detection rate is ~3.5× actual FGA due to false "
            "positives from rebounds and loose-ball events. Useful for relative "
            "quarter-to-quarter comparison, not absolute volume."
        )

        # Tier 2: Heuristic counting stats — distinguish pipeline-detected vs pace-estimated
        if has_heuristic_counting:
            # Check which counting stats are pace-estimated (TO, OREB only) vs pipeline-detected
            from app.analytics.box_score import StatSource as _SS
            _heuristic_keys = set()
            for p in game.home.players + game.away.players:
                for key, src in p.stat_sources.items():
                    if key in self._COUNTING_STAT_KEYS and src == _SS.HEURISTIC:
                        _heuristic_keys.add(key)

            _pipeline_keys = _heuristic_keys - {"to", "orb"}
            _pace_keys = _heuristic_keys & {"to", "orb"}

            if _pipeline_keys:
                parts.append(
                    "**Pipeline-detected counting stats:** "
                    "Inferred from video by the HoopsVision pipeline — rebound_detector "
                    "(offensive/defensive board attribution), assist_detector (pass-to-score "
                    "linkage), steal_detector (possession change after defensive action), "
                    "+/- from on-court presence during scoring events, and minutes from "
                    "track duration. These are heuristic estimates, not ground truth."
                )

            if _pace_keys and not _pipeline_keys:
                estimated_names = []
                if "to" in _pace_keys:
                    estimated_names.append("turnovers (TO)")
                if "orb" in _pace_keys:
                    estimated_names.append("offensive rebounds (OREB)")
                parts.append(
                    f"**Estimated counting stats ({', '.join(estimated_names)}):** "
                    "Pace-based estimates derived from scoring output and typical U16 rates "
                    "(18% turnover rate, 10% offensive rebound rate per estimated possession). "
                    "These are statistical estimates, not observed events. Useful for Four "
                    "Factors analysis but not precise per-player attribution."
                )

        # Tier 3: Manual scorekeeper counting stats
        if has_manual_counting:
            parts.append(
                "**Manual scorekeeper data (all counting stats including BLK, PF):** "
                "Ground-truth observations merged into the pipeline box score. "
                "Where manual data exists, it overrides heuristic estimates."
            )
        elif not has_heuristic_counting:
            parts.append(
                "**Counting stats (OREB, DREB, AST, TO, STL, BLK, PF, MIN, +/-):** "
                "Not available for this game. kamper.basket.no does not track these stats "
                "and no pipeline detections or scorekeeper data are present. The Four "
                "Factors analysis is limited to eFG% and FT Rate; TOV% and OREB factors "
                "show N/A."
            )

        if not has_manual_counting:
            parts.append(
                "**Blocks and personal fouls:** No pipeline detector exists for BLK or PF. "
                "These require manual scorekeeper entry. Without them, the Game Score "
                "formula omits the +0.7×BLK and −0.4×PF terms."
            )

        return "\n\n".join(parts)

    def _generate_scouting_reports(
        self,
        advanced: GameAdvancedStats,
    ) -> dict[str, str]:
        """Generate per-player scouting reports.

        Uses coaching.yaml analyze_individual prompt when LLM is available.
        """
        reports = {}

        # Load coaching.yaml system prompt if LLM is available
        coaching_system = None
        if self.llm:
            try:
                from app.prompts.loader import load_prompts
                coaching_prompts = load_prompts("coaching")
                coaching_system = coaching_prompts["analyze_individual"]["system"]
            except Exception:
                coaching_system = "You are a basketball scout. Write concise, data-driven assessments."

        for team_adv in [advanced.home, advanced.away]:
            for p in team_adv.player_stats:
                if p.jersey_number == 0 or (p.player_name and "unattributed" in p.player_name.lower()):
                    continue  # Skip pseudo-players
                # Skip players with no meaningful stats (no scoring, no counting stats)
                has_stats = (p.pts > 0 or p.fga > 0 or p.ast > 0 or p.reb > 0
                             or p.stl > 0 or p.blk > 0 or p.to > 0)
                if not has_stats:
                    continue

                key = f"{team_adv.team_key}_{p.jersey_number or p.player_id}"

                if self.llm:
                    try:
                        prompt = self._build_scouting_prompt(p, team_adv)
                        reports[key] = self.llm.generate(prompt, system=coaching_system)
                    except Exception as e:
                        print(f"  Warning: LLM scouting failed for {key} ({e}), using template")
                        reports[key] = self._template_scouting_report(p, team_adv)
                else:
                    reports[key] = self._template_scouting_report(p, team_adv)

        return reports

    def _build_scouting_prompt(
        self,
        player: AdvancedPlayerStats,
        team: TeamAdvancedStats,
    ) -> str:
        """Build LLM prompt for a single player's scouting report."""
        p = player
        lines = [
            f"Write a 3-5 sentence scouting report for:",
            f"#{p.jersey_number} {p.player_name} ({team.team_name})",
            f"Stats: {p.pts}pts, {p.fg}-{p.fga} FG, {p.three_p}-{p.three_pa} 3PT, "
            f"{p.ft}-{p.fta} FT",
            f"Counting: {p.reb} REB ({p.orb} OR, {p.drb} DR), {p.ast} AST, "
            f"{p.to} TO, {p.stl} STL, {p.blk} BLK, {p.pf} PF",
            f"Advanced: TS% {p.ts_pct:.1%}, eFG% {p.efg_pct:.1%}, "
            f"Game Score {p.game_score:.1f}, Grade {p.grade}",
            f"Team context: {team.total_pts} total team points",
            "",
            "Include: key counting stats, efficiency context, specific observations, "
            "and a one-line verdict."
        ]
        return "\n".join(lines)

    def _template_scouting_report(
        self,
        player: AdvancedPlayerStats,
        team: TeamAdvancedStats,
    ) -> str:
        """Template-based scouting report (no LLM).

        Produces 2-4 sentences of proper prose — no sentence fragments.
        """
        p = player
        name = p.player_name or f"#{p.jersey_number}"
        pct_of_team = (p.pts / team.total_pts * 100) if team.total_pts > 0 else 0

        # Sentence 1: Scoring line
        scoring_parts = [f"{name} scored {p.pts} points"]
        if p.fga > 0:
            scoring_parts.append(f"on {p.fg}-{p.fga} shooting")
        if p.three_pa > 0:
            scoring_parts.append(f"including {p.three_p}-{p.three_pa} from three")
        if p.fta > 0:
            scoring_parts.append(f"with {p.ft}-{p.fta} from the free throw line")
        s1 = ", ".join(scoring_parts) + "."

        # Sentence 2: Efficiency verdict (growth-mindset framing)
        if p.fga > 0:
            if p.ts_pct > 0.65:
                eff_word = "highly efficient"
            elif p.ts_pct > 0.50:
                eff_word = "productive"
            elif p.ts_pct > 0.35:
                eff_word = "building towards consistency"
            else:
                eff_word = "developing shot selection"
            s2 = (
                f"Finished {eff_word} "
                f"({p.ts_pct:.1%} TS%, {p.efg_pct:.0%} eFG%) "
                f"with a Game Score of {p.game_score:.1f} ({p.grade})."
            )
        else:
            s2 = f"Recorded a Game Score of {p.game_score:.1f} ({p.grade})."

        # Sentence 3: Team context (if dominant or has counting stats)
        s3 = ""
        if pct_of_team > 40:
            s3 = (
                f"Carried {pct_of_team:.0f}% of the team's scoring burden, "
                f"suggesting heavy offensive reliance on a single player."
            )
        elif p.reb > 0 or p.ast > 0 or p.stl > 0:
            counting = []
            if p.reb > 0:
                counting.append(f"{p.reb} rebounds")
            if p.ast > 0:
                counting.append(f"{p.ast} assists")
            if p.stl > 0:
                counting.append(f"{p.stl} steals")
            s3 = f"Also contributed {', '.join(counting)}."

        sentences = [s for s in [s1, s2, s3] if s]
        return " ".join(sentences)


    def _methodology_notes(self, game: GameBoxScore | None = None) -> list[str]:
        """Standard methodology footnotes.

        Adapts notes based on data availability (e.g. scorekeeper merge).
        """
        notes = [
            "Data sources: HoopsVision AI pipeline (v1.6.0-vlm) + kamper.basket.no API",
            "FGM and FT data from Norwegian Basketball Federation API (ground truth)",
            "FGA detection from AI pipeline (YOLOv8 fine-tuned + VLM jersey recognition)",
            "Advanced metrics computed using standard basketball analytics formulas",
        ]

        has_manual_counting, has_heuristic_counting, _ = (
            getattr(self, '_cached_stat_sources', None)
            or (self._scan_stat_sources(game) if game else (False, False, False))
        )

        if has_manual_counting:
            notes.append(
                "Manual scorekeeper data merged for AST/REB/STL/BLK/PF/TO/MIN — "
                "Game Score uses full Hollinger formula with all 11 terms"
            )
        elif has_heuristic_counting:
            notes.append(
                "Pipeline detects OREB/DREB/AST/STL/TO/+/-/MIN heuristically — "
                "Game Score includes these terms. BLK and PF terms omitted (no detector)"
            )
        else:
            notes.append(
                "No counting stats (REB, AST, STL, TO, BLK, PF, MIN) available — "
                "Game Score uses scoring terms only: PTS + 0.4×FGM − 0.7×FGA − 0.4×(FTA−FTM)"
            )

        notes.append(
            "eFG% and TS% can exceed 100% when FGA equals FGM (API records makes only) "
            "and the player has three-pointers — this reflects incomplete attempt data"
        )
        return notes

    def save_json(self, report: FilmReport, path: Path) -> None:
        """Save the report data as JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)

    def save_markdown(self, report: FilmReport, path: Path) -> None:
        """Save the report as Markdown."""
        path.parent.mkdir(parents=True, exist_ok=True)
        md = self._render_markdown(report)
        with open(path, "w") as f:
            f.write(md)

    def _render_markdown(self, report: FilmReport) -> str:
        """Render the full unified report as Markdown."""
        lines = []
        r = report
        adv = r.advanced_stats

        # Title
        lines.append("# Game Report")
        lines.append(f"## {r.home_name} {r.home_score} — {r.away_name} {r.away_score}")
        lines.append(f"**{r.competition}** | {r.game_date}")
        lines.append("")

        # Section 1: Game Summary
        lines.append("## 1. Game Summary")
        lines.append(r.game_summary)
        lines.append("")

        # Section 2: Quarter Scores
        if r.quarter_scores:
            lines.append("## 2. Score by Quarter")
            lines.append(f"| Quarter | {r.home_name} | {r.away_name} |")
            lines.append("|---------|" + "-" * (len(r.home_name) + 2) + "|" + "-" * (len(r.away_name) + 2) + "|")
            for q in r.quarter_scores:
                lines.append(f"| {q.get('quarter', '')} | {q.get('home', '')} | {q.get('away', '')} |")
            lines.append(f"| **Total** | **{r.home_score}** | **{r.away_score}** |")
            lines.append("")

            if r.quarter_narratives:
                for qi, narr in enumerate(r.quarter_narratives):
                    lines.append(f"**Q{qi + 1}:** {narr}")
                    lines.append("")

        # Section 3: Box Score (with impact lines)
        if r.box_score_text:
            lines.append("## 3. Box Score")
            lines.append("```")
            lines.append(r.box_score_text)
            lines.append("```")
            lines.append("")

        # Section 4: Key Performance Indicators
        if r.kpi_highlights:
            lines.append("## 4. Key Performance Indicators")
            for h in r.kpi_highlights:
                lines.append(f"- {h}")
            lines.append("")

        # Section 5: Four Factors + Advanced Metrics
        if adv:
            for team_adv in [adv.home, adv.away]:
                ff = team_adv.four_factors
                lines.append(f"## Four Factors: {team_adv.team_name}")
                lines.append(f"- eFG%: {ff.efg_pct:.1%} ({ff.efg_grade})")
                lines.append(f"- TOV%: {ff.tov_pct:.1%} ({ff.tov_grade})")
                lines.append(f"- FT Rate: {ff.ft_rate:.1%} ({ff.ft_rate_grade})")
                lines.append(f"- OREB: {ff.oreb} ({ff.oreb_grade})")
                lines.append(f"- Est Poss: {ff.est_possessions:.1f}")
                lines.append(f"- Off Rating: {ff.offensive_rating:.1f}")
                lines.append("")

            if r.key_stat_drivers:
                lines.append("### Key Stat Drivers")
                lines.append(r.key_stat_drivers)
                lines.append("")

            for team_adv in [adv.home, adv.away]:
                lines.append(f"### Individual Metrics: {team_adv.team_name}")
                lines.append("| # | Player | PTS | FG | TS% | eFG% | GmScr | Grade |")
                lines.append("|---|--------|-----|----|-----|------|-------|-------|")
                for p in team_adv.player_stats:
                    if p.jersey_number == 0 or (p.player_name and "unattributed" in p.player_name.lower()):
                        continue
                    has_stats = (p.pts > 0 or p.fga > 0 or p.ast > 0 or p.reb > 0
                                 or p.stl > 0 or p.blk > 0 or p.to > 0)
                    if not has_stats:
                        continue
                    ts_str = f"{p.ts_pct:.1%}" if p.fga > 0 else "-"
                    efg_str = f"{p.efg_pct:.1%}" if p.fga > 0 else "-"
                    lines.append(
                        f"| {p.jersey_number or '-'} | {p.player_name or '-'} | "
                        f"{p.pts} | {p.fg}-{p.fga} | {ts_str} | {efg_str} | "
                        f"{p.game_score:.1f} | {p.grade} |"
                    )
                lines.append("")

        # Section 6: Coaching Assessment
        if r.coaching_assessment:
            lines.append("## 6. Coaching Assessment")
            lines.append(f"**Overall: {r.coaching_overall_grade}**")
            lines.append("")
            lines.append("| Category | Grade | Strength | Area for Growth |")
            lines.append("|----------|-------|----------|-----------------|")
            for c in r.coaching_assessment:
                lines.append(f"| {c.category} | {c.grade} | {c.strength} | {c.area_for_growth} |")
            lines.append("")

        # Section 7: Awards / Final Verdict
        if r.awards:
            lines.append("## 7. Awards")
            lines.append("| Award | Player | Stat Line | Reason |")
            lines.append("|-------|--------|-----------|--------|")
            for a in r.awards:
                lines.append(f"| {a.award_name} | #{a.jersey_number} {a.player_name} | {a.stat_line} | {a.reason} |")
            if r.losing_factor:
                lines.append(f"\n**Development Focus:** {r.losing_factor}")
            if r.winning_adjustment:
                lines.append(f"\n**Next Step:** {r.winning_adjustment}")
            lines.append("")

        # Data Sources
        if r.data_sources_note:
            lines.append("## Appendix: Data Sources and Limitations")
            lines.append(r.data_sources_note)
            lines.append("")

        # Methodology
        if r.methodology_notes:
            lines.append("## Appendix: Statistical Methodology")
            for note in r.methodology_notes:
                lines.append(f"- {note}")

        return "\n".join(lines)

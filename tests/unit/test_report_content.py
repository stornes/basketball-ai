"""Content tests for Game Report output.

Verifies report content against v1.0.0-game-report-spec.yaml.
Tests focus on:
- Data provenance honesty (no false claims)
- N/A grading for missing data
- Template quality (commas not periods, proper prose)
- Section structure requirements
- Methodology accuracy

Two scenarios tested:
1. API-only game (kamper.basket.no scoring data, no counting stats)
2. Full scorekeeper game (all stats including BLK, PF)
"""

import pytest

from app.analytics.box_score import (
    GameBoxScore,
    PlayerBoxScore,
    StatSource,
    TeamBoxScore,
)
from app.reporting.film_report import FilmReport, FilmReportGenerator


# ═══════════════════════════════════════════════════════════════════
# Fixtures — API-only game (tier 1 only, no counting stats)
# ═══════════════════════════════════════════════════════════════════


def _api_only_player(
    *,
    player_id: int = 1,
    jersey_number: int = 10,
    player_name: str = "Test Player",
    team: str = "home",
    fg: int = 5,
    fga: int = 5,
    three_p: int = 1,
    three_pa: int = 1,
    ft: int = 2,
    fta: int = 3,
) -> PlayerBoxScore:
    """Player with API scoring data only — all counting stats zero.

    Mimics kamper.basket.no scenario: FGA = FGM (API only reports makes).
    """
    p = PlayerBoxScore(
        player_id=player_id,
        player_name=player_name,
        jersey_number=jersey_number,
        team=team,
        fg=fg,
        fga=fga,
        three_p=three_p,
        three_pa=three_pa,
        ft=ft,
        fta=fta,
        orb=0,
        drb=0,
        ast=0,
        to=0,
        stl=0,
        blk=0,
        pf=0,
        min_played=0.0,
    )
    # Tag scoring stats as MANUAL (API merge) and counting as UNAVAILABLE
    p.stat_sources = {
        "fg": StatSource.MANUAL,
        "three_p": StatSource.MANUAL,
        "ft": StatSource.MANUAL,
        "fta": StatSource.MANUAL,
        "orb": StatSource.UNAVAILABLE,
        "drb": StatSource.UNAVAILABLE,
        "ast": StatSource.UNAVAILABLE,
        "to": StatSource.UNAVAILABLE,
        "stl": StatSource.UNAVAILABLE,
        "blk": StatSource.UNAVAILABLE,
        "pf": StatSource.UNAVAILABLE,
        "min_played": StatSource.UNAVAILABLE,
    }
    return p


def _full_scorekeeper_player(
    *,
    player_id: int = 1,
    jersey_number: int = 10,
    player_name: str = "Test Player",
    team: str = "home",
    fg: int = 5,
    fga: int = 10,
    three_p: int = 1,
    three_pa: int = 3,
    ft: int = 2,
    fta: int = 3,
    orb: int = 2,
    drb: int = 3,
    ast: int = 4,
    to: int = 2,
    stl: int = 1,
    blk: int = 1,
    pf: int = 2,
    min_played: float = 25.0,
) -> PlayerBoxScore:
    """Player with full scorekeeper data — all stats present."""
    p = PlayerBoxScore(
        player_id=player_id,
        player_name=player_name,
        jersey_number=jersey_number,
        team=team,
        fg=fg,
        fga=fga,
        three_p=three_p,
        three_pa=three_pa,
        ft=ft,
        fta=fta,
        orb=orb,
        drb=drb,
        ast=ast,
        to=to,
        stl=stl,
        blk=blk,
        pf=pf,
        min_played=min_played,
    )
    p.stat_sources = {
        "fg": StatSource.MANUAL,
        "three_p": StatSource.MANUAL,
        "ft": StatSource.MANUAL,
        "fta": StatSource.MANUAL,
        "orb": StatSource.MANUAL,
        "drb": StatSource.MANUAL,
        "ast": StatSource.MANUAL,
        "to": StatSource.MANUAL,
        "stl": StatSource.MANUAL,
        "blk": StatSource.MANUAL,
        "pf": StatSource.MANUAL,
        "min_played": StatSource.MANUAL,
    }
    return p


@pytest.fixture
def api_only_game() -> GameBoxScore:
    """Game with API scoring data only — the common kamper.basket.no scenario."""
    home_players = [
        _api_only_player(player_id=1, jersey_number=4, player_name="Star Scorer",
                         fg=17, fga=17, three_p=4, three_pa=5, ft=6, fta=14, team="home"),
        _api_only_player(player_id=2, jersey_number=1, player_name="Bench Player A",
                         fg=3, fga=3, three_p=1, three_pa=1, ft=4, fta=6, team="home"),
        _api_only_player(player_id=3, jersey_number=5, player_name="Bench Player B",
                         fg=4, fga=4, three_p=0, three_pa=0, ft=0, fta=1, team="home"),
    ]
    away_players = [
        _api_only_player(player_id=4, jersey_number=6, player_name="Away Star",
                         fg=9, fga=9, three_p=5, three_pa=5, ft=0, fta=1, team="away"),
        _api_only_player(player_id=5, jersey_number=36, player_name="Away Support",
                         fg=10, fga=10, three_p=1, three_pa=1, ft=3, fta=4, team="away"),
        _api_only_player(player_id=6, jersey_number=4, player_name="Away Guard",
                         fg=8, fga=8, three_p=2, three_pa=2, ft=4, fta=5, team="away"),
    ]
    return GameBoxScore(
        home=TeamBoxScore(team_name="Home Team", team_key="home", players=home_players),
        away=TeamBoxScore(team_name="Away Team", team_key="away", players=away_players),
        quarter_scores=[
            {"quarter": "Q1", "home": 18, "away": 29},
            {"quarter": "Q2", "home": 20, "away": 12},
            {"quarter": "Q3", "home": 19, "away": 19},
            {"quarter": "Q4", "home": 12, "away": 22},
        ],
    )


@pytest.fixture
def api_only_report(api_only_game: GameBoxScore) -> FilmReport:
    """Generated report from API-only game (no LLM)."""
    gen = FilmReportGenerator(
        competition="Test League",
        game_date="March 17, 2026",
    )
    return gen.generate(api_only_game)


@pytest.fixture
def full_scorekeeper_game() -> GameBoxScore:
    """Game with full scorekeeper data."""
    home_players = [
        _full_scorekeeper_player(player_id=1, jersey_number=10, player_name="Home Star",
                                  fg=8, fga=15, three_p=2, three_pa=5, ft=4, fta=5,
                                  orb=2, drb=4, ast=5, to=3, stl=2, blk=1, pf=2,
                                  min_played=30.0, team="home"),
        _full_scorekeeper_player(player_id=2, jersey_number=11, player_name="Home Bench",
                                  fg=3, fga=8, three_p=1, three_pa=3, ft=2, fta=3,
                                  orb=1, drb=2, ast=2, to=1, stl=1, blk=0, pf=3,
                                  min_played=16.0, team="home"),
    ]
    away_players = [
        _full_scorekeeper_player(player_id=3, jersey_number=20, player_name="Away Star",
                                  fg=10, fga=18, three_p=3, three_pa=7, ft=5, fta=6,
                                  orb=3, drb=5, ast=7, to=2, stl=3, blk=2, pf=1,
                                  min_played=32.0, team="away"),
        _full_scorekeeper_player(player_id=4, jersey_number=21, player_name="Away Bench",
                                  fg=2, fga=6, three_p=0, three_pa=2, ft=1, fta=2,
                                  orb=1, drb=3, ast=1, to=2, stl=0, blk=0, pf=4,
                                  min_played=14.0, team="away"),
    ]
    return GameBoxScore(
        home=TeamBoxScore(team_name="Home FC", team_key="home", players=home_players),
        away=TeamBoxScore(team_name="Away FC", team_key="away", players=away_players),
        quarter_scores=[
            {"quarter": "Q1", "home": 15, "away": 20},
            {"quarter": "Q2", "home": 18, "away": 15},
            {"quarter": "Q3", "home": 12, "away": 22},
            {"quarter": "Q4", "home": 20, "away": 18},
        ],
    )


@pytest.fixture
def full_scorekeeper_report(full_scorekeeper_game: GameBoxScore) -> FilmReport:
    """Generated report from full scorekeeper game (no LLM)."""
    gen = FilmReportGenerator(
        competition="Full Stats League",
        game_date="March 17, 2026",
    )
    return gen.generate(full_scorekeeper_game)


# ═══════════════════════════════════════════════════════════════════
# Section 1: Game Summary
# ═══════════════════════════════════════════════════════════════════


class TestGameSummary:
    """Verify game summary meets v1.0.0 spec requirements."""

    def test_minimum_paragraph_count(self, api_only_report: FilmReport):
        """Game summary must have >= 3 paragraphs (executive summary, not a blurb)."""
        paragraphs = [p.strip() for p in api_only_report.game_summary.split("\n\n") if p.strip()]
        assert len(paragraphs) >= 3, (
            f"Game summary has {len(paragraphs)} paragraphs, minimum is 3. "
            f"Content: {api_only_report.game_summary!r}"
        )

    def test_not_single_paragraph(self, api_only_report: FilmReport):
        """Game summary must NOT be a single paragraph."""
        paragraphs = [p.strip() for p in api_only_report.game_summary.split("\n\n") if p.strip()]
        assert len(paragraphs) > 1, "Game summary is a single paragraph — must be 3+"

    def test_contains_score(self, api_only_report: FilmReport):
        """Game summary must mention the final score."""
        summary = api_only_report.game_summary
        assert "69" in summary or "82" in summary or str(api_only_report.home_score) in summary, (
            f"Game summary does not mention a score. Content: {summary!r}"
        )

    def test_excludes_pseudo_players(self, api_only_report: FilmReport):
        """Game summary must not mention pseudo-players."""
        summary = api_only_report.game_summary.lower()
        assert "unattributed" not in summary
        assert "#0 " not in summary


# ═══════════════════════════════════════════════════════════════════
# Section 3: Four Factors — N/A Grading
# ═══════════════════════════════════════════════════════════════════


class TestFourFactorsNAGrading:
    """Verify N/A grading when data is missing."""

    def test_tov_grade_na_when_no_to_data(self, api_only_report: FilmReport):
        """TOV grade must be N/A when no turnover data exists."""
        home_ff = api_only_report.advanced_stats.home.four_factors
        away_ff = api_only_report.advanced_stats.away.four_factors
        assert home_ff.tov_grade == "N/A", (
            f"Home TOV grade should be 'N/A' but is '{home_ff.tov_grade}'"
        )
        assert away_ff.tov_grade == "N/A", (
            f"Away TOV grade should be 'N/A' but is '{away_ff.tov_grade}'"
        )

    def test_oreb_grade_na_when_no_oreb_data(self, api_only_report: FilmReport):
        """OREB grade must be N/A when no offensive rebound data exists."""
        home_ff = api_only_report.advanced_stats.home.four_factors
        away_ff = api_only_report.advanced_stats.away.four_factors
        assert home_ff.oreb_grade == "N/A", (
            f"Home OREB grade should be 'N/A' but is '{home_ff.oreb_grade}'"
        )
        assert away_ff.oreb_grade == "N/A", (
            f"Away OREB grade should be 'N/A' but is '{away_ff.oreb_grade}'"
        )

    def test_tov_grade_growth_mindset_when_data_exists(self, full_scorekeeper_report: FilmReport):
        """TOV grade must be a growth-mindset label (not N/A) when TO data exists."""
        growth_labels = ("Impact Player", "Solid Foundation", "Developing", "Growth Opportunity")
        home_ff = full_scorekeeper_report.advanced_stats.home.four_factors
        away_ff = full_scorekeeper_report.advanced_stats.away.four_factors
        assert home_ff.tov_grade in growth_labels, (
            f"Home TOV grade should be a growth label but is '{home_ff.tov_grade}'"
        )
        assert away_ff.tov_grade in growth_labels, (
            f"Away TOV grade should be a growth label but is '{away_ff.tov_grade}'"
        )

    def test_oreb_grade_growth_mindset_when_data_exists(self, full_scorekeeper_report: FilmReport):
        """OREB grade must be a growth-mindset label when OREB data exists."""
        growth_labels = ("Impact Player", "Solid Foundation", "Developing", "Growth Opportunity")
        home_ff = full_scorekeeper_report.advanced_stats.home.four_factors
        away_ff = full_scorekeeper_report.advanced_stats.away.four_factors
        assert home_ff.oreb_grade in growth_labels, (
            f"Home OREB grade should be a growth label but is '{home_ff.oreb_grade}'"
        )
        assert away_ff.oreb_grade in growth_labels, (
            f"Away OREB grade should be a growth label but is '{away_ff.oreb_grade}'"
        )


# ═══════════════════════════════════════════════════════════════════
# Section 3a: Key Stat Drivers — N/A handling
# ═══════════════════════════════════════════════════════════════════


class TestKeyStatDrivers:
    """Verify key stat drivers content."""

    def test_tov_data_not_available_when_no_data(self, api_only_report: FilmReport):
        """TOV% driver must say 'Data not available' when no TO data."""
        drivers = api_only_report.key_stat_drivers
        assert "Data not available" in drivers, (
            f"Key stat drivers should mention 'Data not available' for TOV%. "
            f"Content: {drivers!r}"
        )

    def test_oreb_data_not_available_when_no_data(self, api_only_report: FilmReport):
        """OREB driver must say 'Data not available' when no OREB data."""
        drivers = api_only_report.key_stat_drivers
        # Check for OREB-specific N/A
        oreb_section = drivers[drivers.find("OREB"):]
        assert "Data not available" in oreb_section or "not available" in oreb_section.lower(), (
            f"OREB section should mention data not available. "
            f"Content: {oreb_section!r}"
        )


# ═══════════════════════════════════════════════════════════════════
# Section 3b: Data Sources — provenance honesty
# ═══════════════════════════════════════════════════════════════════


class TestDataSourcesNote:
    """Verify data sources note does not make false claims."""

    def test_no_counting_stat_claims_when_unavailable(self, api_only_report: FilmReport):
        """Data sources note must NOT claim counting stats exist when all are zero."""
        note = api_only_report.data_sources_note
        assert "Not available" in note or "not available" in note.lower(), (
            f"Data sources note should say counting stats are 'Not available'. "
            f"Content: {note!r}"
        )

    def test_no_scorekeeper_claim_when_api_only(self, api_only_report: FilmReport):
        """Data sources note must not claim 'manual scorekeeper' data when only API exists."""
        note = api_only_report.data_sources_note.lower()
        # Should NOT claim manual scorekeeper counting data
        assert "manual scorekeeper data" not in note or "not available" in note, (
            f"Data sources note falsely claims manual scorekeeper counting data. "
            f"Content: {api_only_report.data_sources_note!r}"
        )

    def test_explains_efg_above_100(self, api_only_report: FilmReport):
        """Data sources note must explain eFG% > 100% when FGA = FGM."""
        note = api_only_report.data_sources_note
        # Must mention that eFG% can exceed 100% and explain why
        assert "100%" in note or "above 100" in note.lower(), (
            f"Data sources note should explain eFG% > 100%. "
            f"Content: {note!r}"
        )

    def test_mentions_scorekeeper_when_present(self, full_scorekeeper_report: FilmReport):
        """Data sources note must mention manual scorekeeper when data exists."""
        note = full_scorekeeper_report.data_sources_note.lower()
        assert "manual" in note or "scorekeeper" in note, (
            f"Data sources note should mention manual scorekeeper. "
            f"Content: {full_scorekeeper_report.data_sources_note!r}"
        )


# ═══════════════════════════════════════════════════════════════════
# Section 5: Scouting Reports — template quality
# ═══════════════════════════════════════════════════════════════════


class TestScoutingReports:
    """Verify scouting report template quality."""

    def test_uses_commas_not_periods_between_clauses(self, api_only_report: FilmReport):
        """Scouting reports must use commas between clauses, not periods.

        Bad:  'scored 44 points. on 17-17 shooting. including 4-5 from three.'
        Good: 'scored 44 points, on 17-17 shooting, including 4-5 from three.'
        """
        for key, report_text in api_only_report.scouting_reports.items():
            # Check for the specific anti-pattern: period followed by clause connector
            assert ". on " not in report_text.lower(), (
                f"Scouting report {key} uses '. on ' (period between clauses): {report_text!r}"
            )
            assert ". including " not in report_text.lower(), (
                f"Scouting report {key} uses '. including ' (period between clauses): {report_text!r}"
            )
            assert ". with " not in report_text.lower() or "Game Score" in report_text, (
                f"Scouting report {key} uses '. with ' between clauses: {report_text!r}"
            )

    def test_scouting_reports_generated(self, api_only_report: FilmReport):
        """At least one scouting report should be generated for players with stats."""
        assert len(api_only_report.scouting_reports) > 0, (
            "No scouting reports generated — expected at least one for players with stats"
        )


# ═══════════════════════════════════════════════════════════════════
# Section 6: Coaching Assessment
# ═══════════════════════════════════════════════════════════════════


class TestCoachingAssessment:
    """Verify coaching assessment categories."""

    def test_includes_decision_making(self, api_only_report: FilmReport):
        """Coaching assessment must include decision-making category."""
        categories = [c.category.lower() for c in api_only_report.coaching_assessment]
        dm_cats = [c for c in categories if "decision" in c]
        assert len(dm_cats) >= 1, (
            f"Decision-Making category should be present. Categories: {categories}"
        )

    def test_includes_effort(self, api_only_report: FilmReport):
        """Coaching assessment must include effort & hustle category."""
        categories = [c.category.lower() for c in api_only_report.coaching_assessment]
        eff_cats = [c for c in categories if "effort" in c or "hustle" in c]
        assert len(eff_cats) >= 1, (
            f"Effort & Hustle category should be present. Categories: {categories}"
        )

    def test_includes_shot_selection(self, api_only_report: FilmReport):
        """Coaching assessment must include shot selection category."""
        categories = [c.category.lower() for c in api_only_report.coaching_assessment]
        ss_cats = [c for c in categories if "shot" in c]
        assert len(ss_cats) >= 1, (
            f"Shot Selection category should be present. Categories: {categories}"
        )

    def test_includes_team_balance(self, api_only_report: FilmReport):
        """Coaching assessment must include team balance category."""
        categories = [c.category.lower() for c in api_only_report.coaching_assessment]
        bal_cats = [c for c in categories if "balance" in c]
        assert len(bal_cats) >= 1, (
            f"Team Balance category should be present. Categories: {categories}"
        )


# ═══════════════════════════════════════════════════════════════════
# Appendix: Methodology — formula accuracy
# ═══════════════════════════════════════════════════════════════════


class TestMethodologyNotes:
    """Verify methodology notes accuracy."""

    def test_scoring_only_when_no_counting_stats(self, api_only_report: FilmReport):
        """Methodology must say 'scoring terms only' when counting stats are zero."""
        notes_text = " ".join(api_only_report.methodology_notes)
        assert "scoring terms only" in notes_text.lower(), (
            f"Methodology should mention 'scoring terms only' when no counting stats. "
            f"Content: {notes_text!r}"
        )

    def test_no_full_hollinger_claim_when_no_counting_stats(self, api_only_report: FilmReport):
        """Methodology must NOT claim 'full Hollinger' when counting stats are zero."""
        notes_text = " ".join(api_only_report.methodology_notes).lower()
        assert "full hollinger" not in notes_text, (
            f"Methodology falsely claims 'full Hollinger' when counting stats are zero. "
            f"Content: {notes_text!r}"
        )

    def test_efg_ts_explanation(self, api_only_report: FilmReport):
        """Methodology must explain that eFG%/TS% can exceed 100%."""
        notes_text = " ".join(api_only_report.methodology_notes)
        assert "100%" in notes_text or "exceed" in notes_text.lower(), (
            f"Methodology should explain eFG%/TS% can exceed 100%. "
            f"Content: {notes_text!r}"
        )


# ═══════════════════════════════════════════════════════════════════
# Section 7: Final Verdict
# ═══════════════════════════════════════════════════════════════════


class TestFinalVerdict:
    """Verify final verdict awards."""

    def test_has_impact_mvp(self, api_only_report: FilmReport):
        """Report must include an Impact MVP award (always awarded)."""
        award_names = [a.award_name for a in api_only_report.awards]
        assert "Impact MVP" in award_names

    def test_no_stat_lvp(self, api_only_report: FilmReport):
        """Report must NOT include a Stat LVP award (growth-mindset: no singling out)."""
        award_names = [a.award_name for a in api_only_report.awards]
        assert "Stat LVP" not in award_names

    def test_no_pseudo_player_awards(self, api_only_report: FilmReport):
        """Awards must never go to pseudo-players."""
        for award in api_only_report.awards:
            assert award.jersey_number != 0, (
                f"Award '{award.award_name}' given to pseudo-player #0"
            )
            if award.player_name:
                assert "unattributed" not in award.player_name.lower(), (
                    f"Award '{award.award_name}' given to unattributed player"
                )

    def test_has_losing_factor(self, api_only_report: FilmReport):
        """Report must include a losing factor."""
        assert api_only_report.losing_factor, "Losing factor should not be empty"

    def test_has_winning_adjustment(self, api_only_report: FilmReport):
        """Report must include a winning adjustment."""
        assert api_only_report.winning_adjustment, "Winning adjustment should not be empty"


# ═══════════════════════════════════════════════════════════════════
# Data provenance: _scan_stat_sources
# ═══════════════════════════════════════════════════════════════════


class TestScanStatSources:
    """Verify the 3-tuple stat source scanning."""

    def test_api_only_returns_scoring_manual_only(self, api_only_game: GameBoxScore):
        """API-only game: has_manual_scoring=True, others False."""
        manual_c, heuristic_c, manual_s = FilmReportGenerator._scan_stat_sources(api_only_game)
        assert manual_s is True, "Should detect manual scoring (API data)"
        assert manual_c is False, "Should NOT detect manual counting (API doesn't provide these)"
        assert heuristic_c is False, "Should NOT detect heuristic counting"

    def test_full_scorekeeper_returns_all_true(self, full_scorekeeper_game: GameBoxScore):
        """Full scorekeeper game: has_manual_counting=True."""
        manual_c, heuristic_c, manual_s = FilmReportGenerator._scan_stat_sources(full_scorekeeper_game)
        assert manual_s is True, "Should detect manual scoring"
        assert manual_c is True, "Should detect manual counting (scorekeeper data)"

"""Tests for advanced basketball analytics — Four Factors, Game Score, TS%, eFG%, USG%, Per-36.

Covers compute_four_factors, compute_player_advanced, compute_grade,
compute_team_advanced, compute_game_advanced, grading thresholds,
and zero/edge cases.
"""

import pytest

from app.analytics.advanced_stats import (
    AdvancedPlayerStats,
    FourFactors,
    GameAdvancedStats,
    TeamAdvancedStats,
    compute_four_factors,
    compute_game_advanced,
    compute_grade,
    compute_player_advanced,
    compute_team_advanced,
)
from app.analytics.box_score import (
    GameBoxScore,
    PlayerBoxScore,
    TeamBoxScore,
)


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════


def _make_team(
    *,
    players: list[PlayerBoxScore] | None = None,
    team_name: str = "TestTeam",
    team_key: str = "home",
) -> TeamBoxScore:
    return TeamBoxScore(
        team_name=team_name,
        team_key=team_key,
        players=players or [],
    )


def _reference_player(**kwargs) -> PlayerBoxScore:
    """Build a PlayerBoxScore with sensible defaults for testing."""
    defaults = dict(
        player_id=1,
        player_name="Test Player",
        jersey_number=10,
        team="home",
        min_played=30.0,
        fg=8,
        fga=15,
        three_p=2,
        three_pa=5,
        ft=4,
        fta=5,
        orb=2,
        drb=4,
        ast=5,
        to=3,
        stl=2,
        blk=1,
        pf=2,
    )
    defaults.update(kwargs)
    return PlayerBoxScore(**defaults)


# ═══════════════════════════════════════════════════════════════════
# Four Factors — reference value verification
# ═══════════════════════════════════════════════════════════════════


class TestFourFactorsReferenceValues:
    """Verify Four Factors formulas against hand-calculated reference values.

    Reference inputs:
        FGM=24, FGA=69, 3PM=6, FTA=22, TO=19, OREB=24, PTS=67
    """

    @pytest.fixture
    def ref_team(self) -> TeamBoxScore:
        """Team whose totals match the reference doc values."""
        # We need total_fg=24, total_fga=69, total_three_p=6,
        # total_fta=22, total_to=19, total_orb=24, total_pts=67.
        #
        # pts = (fg - three_p)*2 + three_p*3 + ft
        # 67 = (24-6)*2 + 6*3 + ft  =>  67 = 36 + 18 + ft  =>  ft = 13
        p = PlayerBoxScore(
            player_id=1,
            team="home",
            fg=24,
            fga=69,
            three_p=6,
            three_pa=15,
            ft=13,
            fta=22,
            orb=24,
            drb=10,
            to=19,
            min_played=40.0,
        )
        return _make_team(players=[p])

    def test_efg_pct(self, ref_team: TeamBoxScore):
        ff = compute_four_factors(ref_team)
        # eFG% = (24 + 0.5*6) / 69 = 27/69 ≈ 0.3913
        assert ff.efg_pct == pytest.approx(0.3913, abs=0.001)

    def test_tov_pct(self, ref_team: TeamBoxScore):
        ff = compute_four_factors(ref_team)
        # TOV% = 19 / (69 + 0.44*22 + 19) = 19 / 97.68 ≈ 0.1945
        assert ff.tov_pct == pytest.approx(0.1945, abs=0.001)

    def test_ft_rate(self, ref_team: TeamBoxScore):
        ff = compute_four_factors(ref_team)
        # FT Rate = 22/69 ≈ 0.3188
        assert ff.ft_rate == pytest.approx(0.3188, abs=0.001)

    def test_est_possessions(self, ref_team: TeamBoxScore):
        ff = compute_four_factors(ref_team)
        # Est Poss = 69 + 0.44*22 + 19 - 24 = 73.68
        assert ff.est_possessions == pytest.approx(73.68, abs=0.01)

    def test_ppp(self, ref_team: TeamBoxScore):
        ff = compute_four_factors(ref_team)
        # PPP = 67 / 73.68 ≈ 0.9094
        assert ff.ppp == pytest.approx(0.9094, abs=0.001)

    def test_offensive_rating(self, ref_team: TeamBoxScore):
        ff = compute_four_factors(ref_team)
        # ORtg = PPP * 100 ≈ 90.9
        assert ff.offensive_rating == pytest.approx(90.94, abs=0.1)


# ═══════════════════════════════════════════════════════════════════
# Four Factors — grading
# ═══════════════════════════════════════════════════════════════════


class TestFourFactorsGrading:
    """Verify grading thresholds for all four factors."""

    def _team_with(self, *, fgm=24, fga=69, three_pm=6, fta=22, to=19, oreb=24, ft=13):
        p = PlayerBoxScore(
            player_id=1,
            team="home",
            fg=fgm,
            fga=fga,
            three_p=three_pm,
            three_pa=10,
            ft=ft,
            fta=fta,
            orb=oreb,
            drb=10,
            to=to,
            min_played=40.0,
        )
        return _make_team(players=[p])

    # eFG grading (growth-mindset): >0.52 Impact Player, >0.45 Solid Foundation, >0.38 Developing, else Growth Opportunity
    def test_efg_grade_impact_player(self):
        # eFG = (fgm + 0.5*3pm) / fga > 0.52
        # (36 + 0.5*8) / 69 = 40/69 = 0.5797
        ff = compute_four_factors(self._team_with(fgm=36, three_pm=8))
        assert ff.efg_grade == "Impact Player"

    def test_efg_grade_solid_foundation(self):
        # (32 + 0.5*6) / 69 = 35/69 = 0.5072
        ff = compute_four_factors(self._team_with(fgm=32, three_pm=6))
        assert ff.efg_grade == "Solid Foundation"

    def test_efg_grade_developing(self):
        # ref values: (24 + 3) / 69 = 0.3913 > 0.38
        ff = compute_four_factors(self._team_with())
        assert ff.efg_grade == "Developing"

    def test_efg_grade_growth_opportunity(self):
        # (20 + 0.5*2) / 69 = 21/69 = 0.3043
        ff = compute_four_factors(self._team_with(fgm=20, three_pm=2))
        assert ff.efg_grade == "Growth Opportunity"

    # TOV grading (lower is better): <0.12 Impact Player, <0.16 Solid Foundation, <0.20 Developing, else Growth Opportunity
    def test_tov_grade_impact_player(self):
        # TOV% = to / (fga + 0.44*fta + to) < 0.12
        # 5 / (69 + 9.68 + 5) = 5/83.68 = 0.0597
        ff = compute_four_factors(self._team_with(to=5))
        assert ff.tov_grade == "Impact Player"

    def test_tov_grade_solid_foundation(self):
        # 12 / (69 + 9.68 + 12) = 12/90.68 = 0.1323
        ff = compute_four_factors(self._team_with(to=12))
        assert ff.tov_grade == "Solid Foundation"

    def test_tov_grade_developing(self):
        # 19 / 97.68 = 0.1945 < 0.20
        ff = compute_four_factors(self._team_with(to=19))
        assert ff.tov_grade == "Developing"

    def test_tov_grade_growth_opportunity(self):
        # 25 / (69 + 9.68 + 25) = 25/103.68 = 0.2411
        ff = compute_four_factors(self._team_with(to=25))
        assert ff.tov_grade == "Growth Opportunity"

    # FT Rate grading: >0.35 Impact Player, >0.25 Solid Foundation, >0.15 Developing, else Growth Opportunity
    def test_ft_rate_grade_impact_player(self):
        # 25/69 = 0.362
        ff = compute_four_factors(self._team_with(fta=25))
        assert ff.ft_rate_grade == "Impact Player"

    def test_ft_rate_grade_solid_foundation(self):
        # 20/69 = 0.2899
        ff = compute_four_factors(self._team_with(fta=20))
        assert ff.ft_rate_grade == "Solid Foundation"

    def test_ft_rate_grade_developing(self):
        # 12/69 = 0.1739
        ff = compute_four_factors(self._team_with(fta=12))
        assert ff.ft_rate_grade == "Developing"

    def test_ft_rate_grade_growth_opportunity(self):
        # 8/69 = 0.1159
        ff = compute_four_factors(self._team_with(fta=8))
        assert ff.ft_rate_grade == "Growth Opportunity"

    # OREB grading: >15 Impact Player, >10 Solid Foundation, >5 Developing, else Growth Opportunity
    def test_oreb_grade_impact_player(self):
        ff = compute_four_factors(self._team_with(oreb=16))
        assert ff.oreb_grade == "Impact Player"

    def test_oreb_grade_solid_foundation(self):
        ff = compute_four_factors(self._team_with(oreb=12))
        assert ff.oreb_grade == "Solid Foundation"

    def test_oreb_grade_developing(self):
        ff = compute_four_factors(self._team_with(oreb=7))
        assert ff.oreb_grade == "Developing"

    def test_oreb_grade_growth_opportunity(self):
        ff = compute_four_factors(self._team_with(oreb=3))
        assert ff.oreb_grade == "Growth Opportunity"


class TestFourFactorsEdgeCases:
    """Zero and degenerate inputs for Four Factors."""

    def test_zero_fga(self):
        p = PlayerBoxScore(player_id=1, team="home", fg=0, fga=0, fta=0, to=0, orb=0)
        team = _make_team(players=[p])
        ff = compute_four_factors(team)
        assert ff.efg_pct == 0.0
        assert ff.ft_rate == 0.0

    def test_zero_possessions(self):
        """When est_possessions <= 0, PPP should be 0."""
        p = PlayerBoxScore(
            player_id=1,
            team="home",
            fg=0,
            fga=0,
            three_p=0,
            fta=0,
            to=0,
            orb=0,
        )
        team = _make_team(players=[p])
        ff = compute_four_factors(team)
        assert ff.est_possessions == 0.0
        assert ff.ppp == 0.0
        assert ff.offensive_rating == 0.0

    def test_to_dict_rounding(self):
        p = PlayerBoxScore(
            player_id=1,
            team="home",
            fg=24,
            fga=69,
            three_p=6,
            three_pa=15,
            ft=13,
            fta=22,
            orb=24,
            drb=10,
            to=19,
        )
        team = _make_team(players=[p])
        ff = compute_four_factors(team)
        d = ff.to_dict()
        # Verify rounding is applied
        assert isinstance(d["efg_pct"], float)
        assert isinstance(d["tov_pct"], float)
        assert isinstance(d["est_possessions"], float)
        assert isinstance(d["ppp"], float)
        assert isinstance(d["offensive_rating"], float)
        assert isinstance(d["oreb"], int)


# ═══════════════════════════════════════════════════════════════════
# AdvancedPlayerStats — formula verification
# ═══════════════════════════════════════════════════════════════════


class TestAdvancedPlayerStats:
    """Verify TS%, eFG%, Game Score, USG%, AST/TO, REB%, Per-36."""

    @pytest.fixture
    def player(self) -> PlayerBoxScore:
        return _reference_player()

    @pytest.fixture
    def team_context(self):
        """Standard team context: 200 team minutes, 70 possessions, 40 team rebounds."""
        return dict(team_min=200.0, team_poss=70.0, team_reb=40)

    def test_ts_pct(self, player, team_context):
        stats = compute_player_advanced(player, **team_context)
        # TS% = pts / (2 * (FGA + 0.44 * FTA))
        # pts = (8-2)*2 + 2*3 + 4 = 12 + 6 + 4 = 22
        # denom = 2 * (15 + 0.44*5) = 2 * 17.2 = 34.4
        # TS% = 22 / 34.4 ≈ 0.6395
        assert stats.ts_pct == pytest.approx(22 / 34.4, abs=0.001)

    def test_efg_pct(self, player, team_context):
        stats = compute_player_advanced(player, **team_context)
        # eFG% = (8 + 0.5*2) / 15 = 9/15 = 0.6
        assert stats.efg_pct == pytest.approx(0.6, abs=0.001)

    def test_game_score(self, player, team_context):
        stats = compute_player_advanced(player, **team_context)
        # GmSc = PTS + 0.4*FGM - 0.7*FGA - 0.4*(FTA-FTM)
        #        + 0.7*OREB + 0.3*DREB + STL + 0.7*AST + 0.7*BLK - 0.4*PF - TO
        # = 22 + 0.4*8 - 0.7*15 - 0.4*(5-4) + 0.7*2 + 0.3*4 + 2 + 0.7*5 + 0.7*1 - 0.4*2 - 3
        # = 22 + 3.2 - 10.5 - 0.4 + 1.4 + 1.2 + 2 + 3.5 + 0.7 - 0.8 - 3
        # = 19.3
        expected = (
            22
            + 0.4 * 8
            - 0.7 * 15
            - 0.4 * (5 - 4)
            + 0.7 * 2
            + 0.3 * 4
            + 2
            + 0.7 * 5
            + 0.7 * 1
            - 0.4 * 2
            - 3
        )
        assert stats.game_score == pytest.approx(expected, abs=0.01)

    def test_usg_pct(self, player, team_context):
        stats = compute_player_advanced(player, **team_context)
        # USG% = 100 * ((FGA + 0.44*FTA + TO) * (Tm_MIN/5)) / (MIN * Tm_Poss)
        # = 100 * ((15 + 0.44*5 + 3) * (200/5)) / (30 * 70)
        # = 100 * (20.2 * 40) / 2100
        # = 100 * 808 / 2100
        # ≈ 38.476
        numer = (15 + 0.44 * 5 + 3) * (200 / 5)
        denom = 30 * 70
        expected = 100 * numer / denom
        assert stats.usg_pct == pytest.approx(expected, abs=0.1)

    def test_ast_to_ratio_normal(self, player, team_context):
        stats = compute_player_advanced(player, **team_context)
        # 5 / 3 ≈ 1.667
        assert isinstance(stats.ast_to_ratio, float)
        assert stats.ast_to_ratio == pytest.approx(5 / 3, abs=0.01)

    def test_ast_to_zero_to_with_ast(self, team_context):
        """Zero turnovers with assists should yield 'INF'."""
        p = _reference_player(ast=5, to=0)
        stats = compute_player_advanced(p, **team_context)
        assert stats.ast_to_ratio == "INF"

    def test_ast_to_zero_both(self, team_context):
        """Zero assists and zero turnovers should yield '-'."""
        p = _reference_player(ast=0, to=0)
        stats = compute_player_advanced(p, **team_context)
        assert stats.ast_to_ratio == "-"

    def test_reb_pct(self, player, team_context):
        stats = compute_player_advanced(player, **team_context)
        # REB% = (reb / team_reb) * 100 = (6 / 40) * 100 = 15.0
        assert stats.reb_pct == pytest.approx(15.0, abs=0.1)

    def test_per36_projections(self, player, team_context):
        stats = compute_player_advanced(player, **team_context)
        factor = 36 / 30.0
        assert stats.pts_per36 == pytest.approx(22 * factor, abs=0.1)
        assert stats.reb_per36 == pytest.approx(6 * factor, abs=0.1)
        assert stats.ast_per36 == pytest.approx(5 * factor, abs=0.1)
        assert stats.stl_per36 == pytest.approx(2 * factor, abs=0.1)
        assert stats.to_per36 == pytest.approx(3 * factor, abs=0.1)
        assert stats.fga_per36 == pytest.approx(15 * factor, abs=0.1)

    def test_identity_fields_copied(self, player, team_context):
        stats = compute_player_advanced(player, **team_context)
        assert stats.player_id == 1
        assert stats.player_name == "Test Player"
        assert stats.jersey_number == 10
        assert stats.team == "home"

    def test_raw_stats_copied(self, player, team_context):
        stats = compute_player_advanced(player, **team_context)
        assert stats.pts == 22
        assert stats.fg == 8
        assert stats.fga == 15
        assert stats.three_p == 2
        assert stats.ft == 4
        assert stats.fta == 5
        assert stats.orb == 2
        assert stats.drb == 4
        assert stats.reb == 6
        assert stats.ast == 5
        assert stats.to == 3
        assert stats.stl == 2
        assert stats.blk == 1
        assert stats.pf == 2
        assert stats.min_played == 30.0


# ═══════════════════════════════════════════════════════════════════
# Advanced Player Stats — edge cases
# ═══════════════════════════════════════════════════════════════════


class TestAdvancedPlayerEdgeCases:

    def test_zero_fga(self):
        """Player with zero field goal attempts."""
        p = _reference_player(fg=0, fga=0, three_p=0, three_pa=0, ft=2, fta=3)
        stats = compute_player_advanced(p, team_min=200, team_poss=70, team_reb=40)
        assert stats.ts_pct == pytest.approx(2 / (2 * 0.44 * 3), abs=0.001)
        assert stats.efg_pct == 0.0

    def test_zero_minutes(self):
        """Player with zero minutes should have zero USG% and no per-36."""
        p = _reference_player(min_played=0.0)
        stats = compute_player_advanced(p, team_min=200, team_poss=70, team_reb=40)
        assert stats.usg_pct == 0.0
        assert stats.pts_per36 == 0.0
        assert stats.reb_per36 == 0.0
        assert stats.ast_per36 == 0.0

    def test_under_one_minute_no_per36(self):
        """Player with less than 1 minute should not get per-36 projections."""
        p = _reference_player(min_played=0.5)
        stats = compute_player_advanced(p, team_min=200, team_poss=70, team_reb=40)
        assert stats.pts_per36 == 0.0

    def test_exactly_one_minute_gets_per36(self):
        """Player with exactly 1 minute should get per-36 projections."""
        p = _reference_player(min_played=1.0, fg=1, fga=2, three_p=0, ft=0, fta=0)
        stats = compute_player_advanced(p, team_min=200, team_poss=70, team_reb=40)
        # pts = (1-0)*2 + 0 + 0 = 2; per36 = 2 * 36 = 72
        assert stats.pts_per36 == pytest.approx(72.0, abs=0.1)

    def test_zero_team_reb(self):
        """Team with zero rebounds should yield reb_pct = 0."""
        p = _reference_player()
        stats = compute_player_advanced(p, team_min=200, team_poss=70, team_reb=0)
        assert stats.reb_pct == 0.0

    def test_zero_team_possessions(self):
        """Zero team possessions should yield zero USG%."""
        p = _reference_player()
        stats = compute_player_advanced(p, team_min=200, team_poss=0, team_reb=40)
        assert stats.usg_pct == 0.0

    def test_zero_everything(self):
        """Completely zeroed-out player."""
        p = PlayerBoxScore(player_id=99, team="away", min_played=0.0)
        stats = compute_player_advanced(p, team_min=0, team_poss=0, team_reb=0)
        assert stats.ts_pct == 0.0
        assert stats.efg_pct == 0.0
        assert stats.usg_pct == 0.0
        assert stats.game_score == 0.0
        assert stats.ast_to_ratio == "-"
        assert stats.reb_pct == 0.0
        assert stats.pts_per36 == 0.0
        assert stats.grade == "Growth Opportunity"


# ═══════════════════════════════════════════════════════════════════
# compute_grade thresholds
# ═══════════════════════════════════════════════════════════════════


class TestComputeGrade:
    """Test all grading branches — growth-mindset labels."""

    def test_grade_impact_player(self):
        # game_score > 15 AND ts_pct > 0.55
        assert compute_grade(16.0, 0.56, 2.0) == "Impact Player"

    def test_grade_impact_player_boundary_misses_on_gs(self):
        # game_score=15 exactly does NOT exceed 15
        assert compute_grade(15.0, 0.56, 2.0) != "Impact Player"

    def test_grade_impact_player_boundary_misses_on_ts(self):
        # ts_pct=0.55 does NOT exceed 0.55
        assert compute_grade(16.0, 0.55, 2.0) != "Impact Player"

    def test_grade_rising_performer_high_gs(self):
        # game_score > 12
        assert compute_grade(13.0, 0.40, 1.0) == "Rising Performer"

    def test_grade_rising_performer_moderate_gs_high_ts(self):
        # game_score > 8 AND ts_pct > 0.60
        assert compute_grade(9.0, 0.61, 1.0) == "Rising Performer"

    def test_grade_solid_foundation(self):
        # game_score > 8 but ts_pct <= 0.60
        assert compute_grade(9.0, 0.50, 1.0) == "Solid Foundation"

    def test_grade_developing(self):
        # game_score > 4
        assert compute_grade(5.0, 0.40, 1.0) == "Developing"

    def test_grade_foundation_phase(self):
        # game_score > 0
        assert compute_grade(1.0, 0.30, 1.0) == "Foundation Phase"

    def test_grade_growth_opportunity(self):
        # game_score <= 0
        assert compute_grade(0.0, 0.30, 1.0) == "Growth Opportunity"
        assert compute_grade(-5.0, 0.30, 1.0) == "Growth Opportunity"

    def test_grade_with_inf_ast_to(self):
        """Grade function should handle string AST/TO without error."""
        assert compute_grade(20.0, 0.60, "INF") == "Impact Player"
        assert compute_grade(-1.0, 0.30, "-") == "Growth Opportunity"

    def test_grade_boundary_exactly_at_threshold(self):
        # game_score=12 exactly: not > 12 for Rising Performer, but > 8 => Solid Foundation
        assert compute_grade(12.0, 0.50, 1.0) == "Solid Foundation"
        # game_score=8 exactly: not > 8 => check > 4 => Developing
        assert compute_grade(8.0, 0.50, 1.0) == "Developing"
        # game_score=4 exactly: not > 4 => check > 0 => Foundation Phase
        assert compute_grade(4.0, 0.50, 1.0) == "Foundation Phase"


# ═══════════════════════════════════════════════════════════════════
# compute_team_advanced
# ═══════════════════════════════════════════════════════════════════


class TestComputeTeamAdvanced:

    def test_basic_team_stats(self):
        p1 = _reference_player(player_id=1, player_name="Star", fg=10, fga=18,
                               three_p=3, ft=5, fta=6, orb=3, drb=5,
                               ast=7, to=2, stl=3, blk=2, pf=1, min_played=32.0)
        p2 = _reference_player(player_id=2, player_name="Bench", fg=3, fga=8,
                               three_p=1, ft=2, fta=3, orb=1, drb=2,
                               ast=2, to=1, stl=1, blk=0, pf=3, min_played=16.0)
        team = _make_team(players=[p1, p2])
        result = compute_team_advanced(team)

        assert result.team_name == "TestTeam"
        assert result.team_key == "home"
        assert result.total_fg == 13
        assert result.total_fga == 26

        # Four factors should be computed
        assert result.four_factors.efg_pct > 0

        # Player stats sorted by game_score descending
        assert len(result.player_stats) == 2
        assert result.player_stats[0].game_score >= result.player_stats[1].game_score

    def test_single_player_team(self):
        p = _reference_player()
        team = _make_team(players=[p])
        result = compute_team_advanced(team)
        assert len(result.player_stats) == 1
        # REB% should be 100% for the only player
        assert result.player_stats[0].reb_pct == pytest.approx(100.0, abs=0.1)

    def test_empty_team(self):
        team = _make_team(players=[])
        result = compute_team_advanced(team)
        assert result.total_pts == 0
        assert len(result.player_stats) == 0

    def test_to_dict(self):
        p = _reference_player()
        team = _make_team(players=[p])
        result = compute_team_advanced(team)
        d = result.to_dict()
        assert "four_factors" in d
        assert "player_stats" in d
        assert "totals" in d
        assert d["team_name"] == "TestTeam"


# ═══════════════════════════════════════════════════════════════════
# compute_game_advanced
# ═══════════════════════════════════════════════════════════════════


class TestComputeGameAdvanced:

    def test_full_game(self):
        home_p = _reference_player(player_id=1, team="home")
        away_p = _reference_player(player_id=2, team="away")
        game = GameBoxScore(
            home=_make_team(players=[home_p], team_key="home", team_name="Home"),
            away=_make_team(players=[away_p], team_key="away", team_name="Away"),
            quarter_scores=[{"q1": {"home": 15, "away": 12}}],
        )
        result = compute_game_advanced(game)
        assert isinstance(result, GameAdvancedStats)
        assert result.home.team_name == "Home"
        assert result.away.team_name == "Away"
        assert result.quarter_scores == [{"q1": {"home": 15, "away": 12}}]

    def test_to_dict(self):
        home_p = _reference_player(player_id=1, team="home")
        away_p = _reference_player(player_id=2, team="away")
        game = GameBoxScore(
            home=_make_team(players=[home_p], team_key="home"),
            away=_make_team(players=[away_p], team_key="away"),
        )
        result = compute_game_advanced(game)
        d = result.to_dict()
        assert "home" in d
        assert "away" in d
        assert "quarter_scores" in d


# ═══════════════════════════════════════════════════════════════════
# AdvancedPlayerStats.to_dict serialisation
# ═══════════════════════════════════════════════════════════════════


class TestAdvancedPlayerStatsToDict:

    def test_to_dict_keys(self):
        p = _reference_player()
        stats = compute_player_advanced(p, team_min=200, team_poss=70, team_reb=40)
        d = stats.to_dict()
        expected_keys = {
            "player_id", "player_name", "jersey_number", "team",
            "pts", "fg", "fga", "three_p", "three_pa", "ft", "fta",
            "orb", "drb", "reb", "ast", "to", "stl", "blk", "pf",
            "min_played", "ts_pct", "efg_pct", "usg_pct", "game_score",
            "ast_to_ratio", "reb_pct", "grade",
            "pts_per36", "reb_per36", "ast_per36", "stl_per36", "to_per36", "fga_per36",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_rounding(self):
        p = _reference_player()
        stats = compute_player_advanced(p, team_min=200, team_poss=70, team_reb=40)
        d = stats.to_dict()
        # ts_pct rounded to 3 decimals
        assert d["ts_pct"] == round(stats.ts_pct, 3)
        # game_score rounded to 1 decimal
        assert d["game_score"] == round(stats.game_score, 1)

    def test_to_dict_ast_to_string(self):
        """When AST/TO is a string, to_dict should preserve it."""
        p = _reference_player(ast=5, to=0)
        stats = compute_player_advanced(p, team_min=200, team_poss=70, team_reb=40)
        d = stats.to_dict()
        assert d["ast_to_ratio"] == "INF"

    def test_to_dict_ast_to_float(self):
        p = _reference_player(ast=6, to=3)
        stats = compute_player_advanced(p, team_min=200, team_poss=70, team_reb=40)
        d = stats.to_dict()
        assert d["ast_to_ratio"] == round(6 / 3, 1)


# ═══════════════════════════════════════════════════════════════════
# FourFactors dataclass defaults
# ═══════════════════════════════════════════════════════════════════


class TestFourFactorsDefaults:

    def test_defaults(self):
        ff = FourFactors()
        assert ff.efg_pct == 0.0
        assert ff.efg_grade == ""
        assert ff.tov_pct == 0.0
        assert ff.est_possessions == 0.0
        assert ff.ppp == 0.0
        assert ff.offensive_rating == 0.0
        assert ff.oreb == 0


class TestAdvancedPlayerStatsDefaults:

    def test_defaults(self):
        s = AdvancedPlayerStats()
        assert s.player_id == 0
        assert s.player_name is None
        assert s.ts_pct == 0.0
        assert s.game_score == 0.0
        assert s.grade == ""
        assert s.pts_per36 == 0.0

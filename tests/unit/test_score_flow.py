"""Tests for score flow chart generation."""

from pathlib import Path

from app.analytics.score_flow import ScoreFlowGenerator
from app.config.roster import Roster


def test_generates_score_flow_chart(tmp_path):
    # Actual game: Notodden Thunders (D) 69 - EB-85 82
    roster = Roster(
        home_team_name="Notodden Thunders (D)",
        away_team_name="EB-85",
        home_scores=[18, 38, 57, 69],
        away_scores=[29, 41, 60, 82],
    )

    out = str(tmp_path / "score_flow.png")
    gen = ScoreFlowGenerator()
    result = gen.generate(roster, out)

    assert Path(result).exists()
    assert Path(result).stat().st_size > 0


def test_returns_empty_string_without_scores(tmp_path):
    roster = Roster(home_team_name="A", away_team_name="B")

    out = str(tmp_path / "score_flow.png")
    gen = ScoreFlowGenerator()
    result = gen.generate(roster, out)

    assert result == ""
    assert not Path(out).exists()


def test_handles_overtime_scores(tmp_path):
    roster = Roster(
        home_team_name="Lions",
        away_team_name="Tigers",
        home_scores=[20, 40, 60, 80, 90],  # 5 periods = OT
        away_scores=[22, 38, 58, 80, 88],
    )

    out = str(tmp_path / "score_flow_ot.png")
    gen = ScoreFlowGenerator()
    result = gen.generate(roster, out)

    assert Path(result).exists()

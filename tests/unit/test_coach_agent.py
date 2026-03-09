"""Tests for coaching agent (template mode only - no LLM needed)."""

from pathlib import Path

from app.reporting.coach_agent import generate_template_report, run_coaching_agent


def test_template_report_generation():
    summary = {
        "total_shots": 10,
        "shots_made": 4,
        "fg_percentage": 0.4,
        "total_possessions": 20,
        "avg_possession_duration": 12.5,
        "turnovers": 5,
        "player_stats": [
            {
                "player_id": 1,
                "shots_attempted": 6,
                "shots_made": 3,
                "fg_percentage": 0.5,
                "possessions": 10,
                "possession_time_sec": 60.0,
                "distance_px": 5000.0,
            }
        ],
    }

    report = generate_template_report(summary)
    assert "Game Analysis Report" in report
    assert "10 shots" in report
    assert "40.0%" in report
    assert "Player 1" in report


def test_run_coaching_agent_template(tmp_path):
    summary = {
        "total_shots": 5,
        "shots_made": 2,
        "fg_percentage": 0.4,
        "total_possessions": 10,
        "avg_possession_duration": 8.0,
        "turnovers": 3,
        "player_stats": [],
    }

    out = str(tmp_path / "report.md")
    result = run_coaching_agent(summary, out, llm_backend="template")

    assert Path(result).exists()
    content = Path(result).read_text()
    assert "Game Analysis Report" in content

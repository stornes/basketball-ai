"""Tests for shot chart generation."""

from pathlib import Path

import pandas as pd

from app.analytics.shot_chart import ShotChartGenerator


def test_generates_chart_with_shots(tmp_path):
    shots_df = pd.DataFrame([
        {"court_x": 25.0, "court_y": 20.0, "outcome": "made"},
        {"court_x": 35.0, "court_y": 30.0, "outcome": "missed"},
        {"court_x": 10.0, "court_y": 15.0, "outcome": "made"},
    ])

    out = str(tmp_path / "shot_chart.png")
    gen = ShotChartGenerator()
    result = gen.generate(shots_df, out)

    assert Path(result).exists()
    assert Path(result).stat().st_size > 0


def test_generates_chart_empty_shots(tmp_path):
    shots_df = pd.DataFrame(columns=["court_x", "court_y", "outcome"])
    out = str(tmp_path / "empty_chart.png")

    gen = ShotChartGenerator()
    result = gen.generate(shots_df, out)

    assert Path(result).exists()
    assert Path(result).stat().st_size > 0

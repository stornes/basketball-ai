"""Coaching report writer — generates a DOCX coaching report from a CoachingReport.

Converts the CoachingReport into a CreateDocx.ts-compatible JSON structure,
then invokes the rendering engine to produce a Word document.

Typography follows the same conventions as docx_renderer.py:
- Font: Arial
- Section headers: navy (#1B3A5C), 16pt bold
- Body: 11pt, 120 after spacing
- Tables: navy header row with white text
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from app.coaching.visual_analyst import ClipAnalysis, CoachingReport


def coaching_report_to_docx_json(report: CoachingReport) -> dict:
    """Convert a CoachingReport to CreateDocx.ts DocumentInput JSON.

    Uses the body array format (table before content) consistent with
    docx_renderer.py conventions.
    """
    sections: list[dict] = []

    # Cover / Title
    sections.append({
        "title": f"Coaching Report — {report.player_name}",
        "content": f"#{report.jersey_number} | {report.team} | {report.game_date}",
        "style": "title",
    })

    # Grade summary table
    grade_rows = _build_grade_table(report.clip_analyses)
    if grade_rows:
        sections.append({
            "title": "Clip Summary",
            "body": [
                {
                    "type": "table",
                    "headers": ["#", "Category", "Context", "Grade"],
                    "rows": grade_rows,
                }
            ],
        })

    # 7 coaching sections
    _append_text_section(sections, "1. Player Profile", report.player_profile)
    _append_text_section(sections, "2. Offensive Game", report.offensive_game)
    _append_text_section(sections, "3. Playmaking and Decision-Making", report.playmaking)
    _append_text_section(sections, "4. Off-Ball Movement", report.off_ball_movement)
    _append_text_section(sections, "5. Defensive Game", report.defensive_game)
    _append_text_section(sections, "6. Intangibles and Effort", report.intangibles)
    _append_text_section(sections, "7. Development Priorities", report.development_plan)

    return {
        "title": f"Coaching Report: {report.player_name}",
        "subtitle": f"#{report.jersey_number} · {report.team} · {report.game_date}",
        "author": "HoopsVision 2026",
        "font": "Arial",
        "sections": sections,
    }


def render_coaching_docx(
    report: CoachingReport,
    output_path: Path,
    *,
    create_docx_tool: str = "~/.claude/skills/DocxCreator/Tools/CreateDocx.ts",
) -> Path:
    """Render a CoachingReport as a DOCX file.

    Converts the report to JSON, writes a temp file, and calls CreateDocx.ts.
    Returns the path to the generated DOCX.
    """
    doc_json = coaching_report_to_docx_json(report)

    json_path = output_path.with_suffix(".docx.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)

    with open(json_path, "w") as f:
        json.dump(doc_json, f, indent=2, ensure_ascii=False)

    tool_path = Path(create_docx_tool).expanduser()

    result = subprocess.run(
        ["bun", "run", str(tool_path), "--output", str(output_path), "--data", str(json_path)],
        capture_output=True,
        text=True,
        timeout=60,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"CreateDocx.ts failed (exit {result.returncode}):\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )

    return output_path


class CoachingReportWriter:
    """Thin wrapper around render_coaching_docx for dependency injection."""

    def __init__(
        self,
        create_docx_tool: str = "~/.claude/skills/DocxCreator/Tools/CreateDocx.ts",
    ):
        self.create_docx_tool = create_docx_tool

    def write_docx(self, report: CoachingReport, output_path: Path) -> Path:
        """Write a coaching report to DOCX and return the output path."""
        return render_coaching_docx(
            report,
            output_path,
            create_docx_tool=self.create_docx_tool,
        )

    def write_json(self, report: CoachingReport, output_path: Path) -> Path:
        """Write the coaching report data as JSON (useful for inspection)."""
        data = {
            "player_name": report.player_name,
            "jersey_number": report.jersey_number,
            "team": report.team,
            "game_date": report.game_date,
            "player_profile": report.player_profile,
            "offensive_game": report.offensive_game,
            "playmaking": report.playmaking,
            "off_ball_movement": report.off_ball_movement,
            "defensive_game": report.defensive_game,
            "intangibles": report.intangibles,
            "development_plan": report.development_plan,
            "clip_analyses": [
                {
                    "clip_path": a.clip_path,
                    "category": a.category,
                    "context": a.context,
                    "situation": a.situation,
                    "what_player_did": a.what_player_did,
                    "what_should_have_done": a.what_should_have_done,
                    "coaching_cue": a.coaching_cue,
                    "grade": a.grade,
                    "error": a.error,
                }
                for a in report.clip_analyses
            ],
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return output_path


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _append_text_section(sections: list[dict], title: str, text: str) -> None:
    """Append a text section to the sections list, skipping empty text."""
    if not text or not text.strip():
        text = "[No analysis available for this section.]"
    sections.append({
        "title": title,
        "body": [
            {"type": "content", "text": text},
        ],
        "pageBreakBefore": True,
    })


def _build_grade_table(analyses: list[ClipAnalysis]) -> list[list[str]]:
    """Build grade summary table rows."""
    rows = []
    grade_colours = {"A": "green", "B": "yellow", "C": "orange", "D": "red", "?": "grey"}
    for i, a in enumerate(analyses, 1):
        rows.append([
            str(i),
            a.category,
            a.context[:60] + ("..." if len(a.context) > 60 else ""),
            a.grade,
        ])
    return rows

"""DOCX renderer for Game Report.

Converts a FilmReport dataclass into a CreateDocx.ts-compatible JSON structure,
then invokes the rendering engine to produce a professional Word document.

Typography follows LL-08 (v3.0.0 manifest):
- Font: Arial throughout (overrides CreateDocx default Cambria/Calibri)
- Section headers: navy (#1B3A5C), 16pt bold
- Sub-headers: navy, 12pt bold
- Body: 11pt, 120 after spacing
- Tables: navy header row with white text, alternating row shading
- Grade colours: A=green, B=yellow-green, C=gold, D=orange-red
- Page breaks before each major section
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from app.analytics.box_score import GameBoxScore, PlayerBoxScore, TeamBoxScore
from app.reporting.film_report import FilmReport


def _fmt_min(minutes: float) -> str:
    """Format minutes as M:SS."""
    total_sec = int(minutes * 60)
    m, s = divmod(total_sec, 60)
    return f"{m}:{s:02d}"


def _player_name_short(p: PlayerBoxScore) -> str:
    """Short display name for table cell."""
    if p.player_name:
        parts = p.player_name.split(", ")
        if len(parts) == 2:
            last, first = parts
            return f"{first[0]}. {last}" if first else last
        return p.player_name
    return f"Player #{p.player_id}"


def _box_score_tables(game: GameBoxScore) -> list[dict]:
    """Build DOCX table sections for each team's box score.

    Each team gets: a stat table (clean rows), followed by impact lines.
    Uses body array format so table renders before content.
    """
    subsections = []
    headers = ["#", "Player", "MIN", "FG", "FT", "PTS", "REB", "AST", "TO", "STL", "BLK", "PF", "DEFL"]

    for team in (game.home, game.away):
        rows = []
        impact_lines = []

        for p in team.players:
            # Skip pseudo-players (jersey 0, unattributed)
            if p.jersey_number is not None and p.jersey_number == 0:
                continue
            if p.player_name and "unattributed" in p.player_name.lower():
                continue
            name = _player_name_short(p)
            jersey = str(p.jersey_number) if p.jersey_number else ""
            rows.append([
                jersey,
                name,
                _fmt_min(p.min_played),
                f"{p.fg}-{p.fga}",
                f"{p.ft}-{p.fta}",
                str(p.pts),
                str(p.reb),
                str(p.ast),
                str(p.to),
                str(p.stl),
                str(p.blk),
                str(p.pf),
                str(p.deflections),
            ])

            # Collect impact line for text block below table
            ast_to = f"{p.ast_to_ratio:.1f}" if p.to > 0 else f"{p.ast:.0f}.0"
            impact_lines.append(
                f"#{jersey} {name}: {p.impact_line}  (AST/TO: {ast_to}, DAI: {p.defensive_activity_index}, EFF: {p.effort_plays})"
            )

        # Totals row
        t = team
        rows.append([
            "",
            "**TOTALS**",
            "",
            f"{t.total_fg}-{t.total_fga}",
            f"{t.total_ft}-{t.total_fta}",
            str(t.total_pts),
            str(t.total_reb),
            str(t.total_ast),
            str(t.total_to),
            str(t.total_stl),
            str(t.total_blk),
            str(t.total_pf),
            str(t.total_deflections),
        ])

        # Impact lines as text below the table
        impact_text = "**Impact Lines** (PTS / REB / AST / STL / BLK / TO)\n\n" + "\n".join(impact_lines)

        # Note unattributed stats if they exist
        unattr = []
        if t.unattributed_ast: unattr.append(f"{t.unattributed_ast} AST")
        if t.unattributed_stl: unattr.append(f"{t.unattributed_stl} STL")
        if t.unattributed_orb + t.unattributed_drb: unattr.append(f"{t.unattributed_orb + t.unattributed_drb} REB")
        if t.unattributed_to: unattr.append(f"{t.unattributed_to} TO")
        if unattr:
            impact_text += f"\n\n*Team totals include {', '.join(unattr)} detected but not attributed to individual players.*"

        # Use body array format so table renders BEFORE content.
        # Legacy format renders content first, then table (wrong order).
        subsections.append({
            "heading": team.team_name,
            "level": 2,
            "body": [
                {"type": "table", "headers": headers, "rows": rows},
                {"type": "content", "text": impact_text},
            ],
        })

    return subsections


# Grade → colour mapping for table cells
GRADE_COLOURS = {
    "A+": "228B22", "A": "228B22", "A-": "228B22",
    "B+": "6B8E23", "B": "6B8E23", "B-": "6B8E23",
    "C+": "DAA520", "C": "DAA520", "C-": "DAA520",
    "D+": "CD5C5C", "D": "CD5C5C", "D-": "CD5C5C",
    "F": "CC0000",
}


def film_report_to_docx_json(report: FilmReport) -> dict[str, Any]:
    """Convert a FilmReport into CreateDocx.ts DocumentInput JSON.

    Single unified renderer. Includes all sections: box score with impact
    lines, KPIs, Four Factors, Individual Metrics, scouting reports,
    coaching assessment, and awards. Sections conditioned on data availability.
    All table+text sections use body array format (table before content).
    """
    r = report
    adv = r.advanced_stats
    sections: list[dict[str, Any]] = []

    # ── Section 1: Game Summary ──────────────────────────────────
    sections.append({
        "heading": "Game Summary",
        "level": 1,
        "content": r.game_summary,
        "pageBreakBefore": True,
    })

    # ── Section 2: Score by Quarter ──────────────────────────────
    if r.quarter_scores:
        q_headers = ["Quarter", r.home_name, r.away_name]
        q_rows = [
            [q.get("quarter", ""), str(q.get("home", "")), str(q.get("away", ""))]
            for q in r.quarter_scores
        ]
        q_rows.append(["**Total**", f"**{r.home_score}**", f"**{r.away_score}**"])

        body_blocks: list[dict] = [
            {"type": "table", "headers": q_headers, "rows": q_rows},
        ]
        if r.quarter_narratives:
            q_narrative = ""
            for qi, narr in enumerate(r.quarter_narratives):
                q_narrative += f"**Q{qi + 1}:** {narr}\n\n"
            body_blocks.append({"type": "content", "text": q_narrative.strip()})

        sections.append({
            "heading": "Score by Quarter",
            "level": 1,
            "body": body_blocks,
            "pageBreakBefore": True,
        })

    # ── Section 2b: Score Flow Chart ─────────────────────────────
    if r.chart_score_flow and Path(r.chart_score_flow).exists():
        sections.append({
            "heading": "Score Flow",
            "level": 2,
            "image": {
                "path": str(Path(r.chart_score_flow).resolve()),
                "caption": f"Cumulative score progression — {r.home_name} vs {r.away_name}",
                "width": 95,
            },
        })

    # ── Section 3: Box Score ─────────────────────────────────────
    if r.game_box_score:
        box_subsections = _box_score_tables(r.game_box_score)
        sections.append({
            "heading": "Box Score",
            "level": 1,
            "pageBreakBefore": True,
            "subsections": box_subsections,
        })
    elif r.box_score_text:
        sections.append({
            "heading": "Box Score",
            "level": 1,
            "content": r.box_score_text,
            "pageBreakBefore": True,
        })

    # ── Section 4: Key Performance Indicators ────────────────────
    if r.kpi_highlights:
        sections.append({
            "heading": "Key Performance Indicators",
            "level": 1,
            "bullets": [h.strip() for h in r.kpi_highlights],
            "pageBreakBefore": True,
        })

    # ── Section 5: Four Factors Analysis ─────────────────────────
    if adv:
        ff_subsections = []
        for team_adv in [adv.home, adv.away]:
            ff = team_adv.four_factors
            ff_subsections.append({
                "heading": team_adv.team_name,
                "level": 3,
                "table": {
                    "headers": ["Factor", "Value", "Grade"],
                    "rows": [
                        ["eFG% (40% weight)", f"{ff.efg_pct:.1%}", ff.efg_grade],
                        ["TOV% (25% weight)", f"{ff.tov_pct:.1%}", ff.tov_grade],
                        ["FT Rate (15% weight)", f"{ff.ft_rate:.1%}", ff.ft_rate_grade],
                        ["OREB (20% weight)", str(ff.oreb), ff.oreb_grade],
                        ["Est. Possessions", f"{ff.est_possessions:.1f}", "—"],
                        ["Offensive Rating", f"{ff.offensive_rating:.1f}", "—"],
                    ],
                },
            })

        sections.append({
            "heading": "Four Factors Analysis",
            "level": 1,
            "content": (
                "Dean Oliver's Four Factors framework evaluates team performance "
                "across the four dimensions that most determine winning: shooting "
                "efficiency (eFG%), ball security (TOV%), free throw generation "
                "(FT Rate), and second-chance opportunities (OREB)."
            ),
            "pageBreakBefore": True,
            "subsections": ff_subsections,
        })

    # ── Section 6: Individual Metrics ────────────────────────────
    if adv:
        any_fga = any(
            p.fga > p.fg
            for team_adv in [adv.home, adv.away]
            for p in team_adv.player_stats
            if p.jersey_number != 0 and not (p.player_name and "unattributed" in p.player_name.lower())
        )
        any_ast = True
        any_min = any(
            p.min_played > 0
            for team_adv in [adv.home, adv.away]
            for p in team_adv.player_stats
            if p.jersey_number != 0 and not (p.player_name and "unattributed" in p.player_name.lower())
        )

        metrics_subsections = []
        for team_adv in [adv.home, adv.away]:
            rows = []
            for p in team_adv.player_stats:
                if p.jersey_number == 0:
                    continue
                if p.player_name and "unattributed" in p.player_name.lower():
                    continue
                has_stats = (
                    p.pts > 0 or p.fga > 0 or p.ast > 0 or p.reb > 0
                    or p.stl > 0 or p.blk > 0 or p.to > 0
                )
                if not has_stats:
                    continue

                fg_str = f"{p.fg}-{p.fga}" if any_fga else str(p.fg)
                ts_str = f"{p.ts_pct:.1%}" if any_fga and p.fga > 0 else "—"
                efg_str = f"{p.efg_pct:.1%}" if any_fga and p.fga > 0 else "—"
                row = [str(p.jersey_number), p.player_name or "—"]
                if any_min:
                    row.append(f"{p.min_played:.0f}")
                row.extend([str(p.pts), fg_str])
                if any_ast:
                    row.append(str(p.ast) if p.ast > 0 else "—")
                row.extend([ts_str, efg_str, f"{p.game_score:.1f}", p.grade])
                rows.append(row)

            headers = ["#", "Player"]
            if any_min:
                headers.append("MIN")
            headers.append("PTS")
            headers.append("FG" if any_fga else "FGM")
            if any_ast:
                headers.append("AST")
            headers.extend(["TS%", "eFG%", "GmScr", "Grade"])

            metrics_subsections.append({
                "heading": team_adv.team_name,
                "level": 2,
                "table": {
                    "headers": headers,
                    "rows": rows,
                },
            })

        sections.append({
            "heading": "Individual Metrics",
            "level": 1,
            "content": (
                "Player-level advanced statistics sorted by Game Score (Hollinger). "
                "TS% measures scoring efficiency accounting for shot type; eFG% adjusts "
                "for three-point value; Game Score provides a single-number performance summary."
            ),
            "pageBreakBefore": True,
            "subsections": metrics_subsections,
        })

    # ── Section 7: Player Scouting Reports ───────────────────────
    if r.scouting_reports:
        scout_subsections = []
        for key, narrative in r.scouting_reports.items():
            parts = key.split("_", 1)
            player_id = parts[1] if len(parts) > 1 else key
            player_name = f"#{player_id}"
            if adv:
                for team_adv in [adv.home, adv.away]:
                    for p in team_adv.player_stats:
                        if str(p.jersey_number) == player_id or str(p.player_id) == player_id:
                            if parts[0] == team_adv.team_key:
                                player_name = f"#{p.jersey_number} {p.player_name or ''}"
                                break

            scout_subsections.append({
                "heading": player_name,
                "level": 3,
                "content": narrative,
            })

        sections.append({
            "heading": "Player Scouting Reports",
            "level": 1,
            "pageBreakBefore": True,
            "subsections": scout_subsections,
        })

    # ── Section 8: Coaching Assessment ───────────────────────────
    if r.coaching_assessment:
        coach_rows = [
            [c.category, c.grade, c.strength, c.area_for_growth]
            for c in r.coaching_assessment
        ]
        sections.append({
            "heading": "Coaching Assessment",
            "level": 1,
            "body": [
                {"type": "content", "text": f"Overall grade: **{r.coaching_overall_grade}**"},
                {"type": "table", "headers": ["Category", "Grade", "Strength", "Area for Growth"], "rows": coach_rows},
            ],
            "pageBreakBefore": True,
        })

    # ── Section 9: Awards / Final Verdict ────────────────────────
    if r.awards:
        award_rows = [
            [a.award_name, f"#{a.jersey_number} {a.player_name}", a.stat_line, a.reason]
            for a in r.awards
        ]
        verdict_body: list[dict] = [
            {"type": "table", "headers": ["Award", "Player", "Stat Line", "Reason"], "rows": award_rows},
        ]
        verdict_content = ""
        if r.losing_factor:
            verdict_content += f"**Development Focus:** {r.losing_factor}\n\n"
        if r.winning_adjustment:
            verdict_content += f"**Next Step:** {r.winning_adjustment}"
        if verdict_content:
            verdict_body.append({"type": "content", "text": verdict_content})

        sections.append({
            "heading": "Awards",
            "level": 1,
            "body": verdict_body,
            "pageBreakBefore": True,
        })

    # ── Section 10: Shot Charts ──────────────────────────────────
    _has_charts = (
        r.chart_shot_home or r.chart_shot_away
        or r.chart_shot_home_quarters or r.chart_shot_away_quarters
    )
    if _has_charts:
        chart_subsections: list[dict[str, Any]] = []

        for label, path in [
            (r.home_name, r.chart_shot_home),
            (r.away_name, r.chart_shot_away),
        ]:
            if path and Path(path).exists():
                chart_subsections.append({
                    "heading": f"{label} — All Quarters",
                    "level": 2,
                    "image": {
                        "path": str(Path(path).resolve()),
                        "caption": f"Shot chart — {label} (green = made with jersey #, red = missed)",
                        "width": 80,
                    },
                })

        for label, q_paths in [
            (r.home_name, r.chart_shot_home_quarters),
            (r.away_name, r.chart_shot_away_quarters),
        ]:
            for i, qpath in enumerate(q_paths, 1):
                if qpath and Path(qpath).exists():
                    chart_subsections.append({
                        "heading": f"{label} — Q{i}",
                        "level": 3,
                        "image": {
                            "path": str(Path(qpath).resolve()),
                            "caption": f"Shot chart — {label} Q{i}",
                            "width": 70,
                        },
                    })

        if chart_subsections:
            sections.append({
                "heading": "Shot Charts",
                "level": 1,
                "content": (
                    "Shot locations detected by the HoopsVision pipeline. "
                    "Green markers indicate made baskets (numbered with shooter's jersey), "
                    "red markers indicate missed attempts."
                ),
                "pageBreakBefore": True,
                "subsections": chart_subsections,
            })

    # ── Appendix: Data Sources and Limitations ───────────────────
    if r.data_sources_note:
        sections.append({
            "heading": "Appendix: Data Sources and Limitations",
            "level": 1,
            "content": r.data_sources_note,
            "pageBreakBefore": True,
        })

    # ── Appendix: Statistical Methodology ────────────────────────
    if r.methodology_notes:
        sections.append({
            "heading": "Appendix: Statistical Methodology",
            "level": 1,
            "bullets": r.methodology_notes,
            "pageBreakBefore": True,
        })

    # ── Assemble document ────────────────────────────────────────
    document: dict[str, Any] = {
        "title": "Game Report",
        "subtitle": f"{r.home_name} {r.home_score} — {r.away_name} {r.away_score}",
        "author": "HoopsVision AI Pipeline",
        "date": f"{r.competition} | {r.game_date}",
        "headerText": f"{r.home_name} vs {r.away_name} — {r.game_date}",
        "toc": True,
        "sections": sections,
    }
    return document


def render_docx(
    report: FilmReport,
    output_path: Path,
    *,
    create_docx_tool: str = "~/.claude/skills/DocxCreator/Tools/CreateDocx.ts",
) -> Path:
    """Render a FilmReport as a DOCX file.

    Converts the report to JSON, writes a temp file, and calls CreateDocx.ts.
    Returns the path to the generated DOCX.
    """
    doc_json = film_report_to_docx_json(report)

    # Write JSON input file alongside the output
    json_path = output_path.with_suffix(".docx.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(doc_json, f, indent=2, ensure_ascii=False)

    # Resolve tool path
    tool_path = Path(create_docx_tool).expanduser()

    # Call CreateDocx.ts
    result = subprocess.run(
        ["bun", "run", str(tool_path), "--output", str(output_path), "--data", str(json_path)],
        capture_output=True,
        text=True,
        timeout=30,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"CreateDocx.ts failed (exit {result.returncode}):\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )

    return output_path

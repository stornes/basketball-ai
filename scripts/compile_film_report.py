"""Compile a game report from saved pipeline data + API ground truth.

Loads box_score.json, merges FT data from kamper.basket.no API,
computes advanced stats, and generates markdown + JSON reports.

Usage:
    python scripts/compile_film_report.py \
        --data-dir data/outputs/v1.6.0-vlm \
        --match-id 8254973 \
        --output-dir data/outputs/v1.6.0-vlm/report
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(override=True)

from app.analytics.box_score import (
    GameBoxScore,
    PlayerBoxScore,
    StatSource,
    TeamBoxScore,
)
from app.analytics.advanced_stats import compute_game_advanced
from app.reporting.film_report import FilmReportGenerator


def load_box_score(data_dir: Path) -> GameBoxScore:
    """Load GameBoxScore from saved JSON."""
    path = data_dir / "box_score.json"
    with open(path) as f:
        data = json.load(f)
    return GameBoxScore.from_dict(data)


def clean_box_score(game: GameBoxScore, data_dir: Path = Path(".")) -> GameBoxScore:
    """Remove unidentified players and players with jerseys not in roster."""
    # Load roster for validation
    roster_path = data_dir / "roster.json"
    valid_jerseys = {"home": set(), "away": set()}
    if roster_path.exists():
        with open(roster_path) as f:
            roster_data = json.load(f)
        for side in ("home", "away"):
            for rp in roster_data.get(side, {}).get("players", []):
                valid_jerseys[side].add(rp["number"])

    for team in [game.home, game.away]:
        roster_valid = [
            p for p in team.players
            if p.jersey_number is not None and (
                not valid_jerseys[team.team_key] or p.jersey_number in valid_jerseys[team.team_key]
            )
        ]
        non_roster = [
            p for p in team.players
            if p.jersey_number is not None and valid_jerseys[team.team_key] and p.jersey_number not in valid_jerseys[team.team_key]
        ]
        unidentified = [p for p in team.players if p.jersey_number is None]

        # Sum stats from removed players
        removed = non_roster + unidentified
        removed_fga = sum(p.fga for p in removed)
        removed_fg = sum(p.fg for p in removed)

        # Log what was dropped. No redistribution. If the pipeline can't
        # attribute a stat to an identified player, that stat is dropped.
        # Principle: you see it and count it. No estimation allowed.
        removed_ast = sum(p.ast for p in removed)
        removed_stl = sum(p.stl for p in removed)
        removed_orb = sum(p.orb for p in removed)
        removed_drb = sum(p.drb for p in removed)
        removed_to = sum(p.to for p in removed)

        if removed_ast + removed_stl + removed_orb + removed_drb + removed_to > 0:
            print(f"  {team.team_name}: dropped unattributed stats: "
                  f"{removed_ast} AST, {removed_stl} STL, {removed_orb} ORB, {removed_drb} DRB, {removed_to} TO")

        game.detection_summary[f"{team.team_key}_unidentified_fga"] = removed_fga
        game.detection_summary[f"{team.team_key}_unidentified_fg"] = removed_fg
        game.detection_summary[f"{team.team_key}_removed_count"] = len(removed)
        game.detection_summary[f"{team.team_key}_dropped_ast"] = removed_ast
        game.detection_summary[f"{team.team_key}_dropped_stl"] = removed_stl
        game.detection_summary[f"{team.team_key}_dropped_orb"] = removed_orb
        game.detection_summary[f"{team.team_key}_dropped_drb"] = removed_drb
        game.detection_summary[f"{team.team_key}_dropped_to"] = removed_to

        team.players = roster_valid

    return game


def merge_api_data(game: GameBoxScore, match_id: int, data_dir: Path = Path(".")) -> GameBoxScore:
    """Merge FT and FGM data from kamper.basket.no API."""
    try:
        from scripts.fetch_game import fetch_incidents, _parse_shots
    except ImportError:
        print("Warning: Could not import fetch_game, skipping API merge")
        return game

    print(f"Fetching ground truth from kamper.basket.no (match {match_id})...")
    incidents = fetch_incidents(match_id)
    shots = _parse_shots(incidents)

    # Build per-player stats from API
    from collections import defaultdict
    api_stats = {"home": defaultdict(lambda: {"fg": 0, "fga": 0, "3p": 0, "3pa": 0, "ft": 0, "fta": 0}),
                 "away": defaultdict(lambda: {"fg": 0, "fga": 0, "3p": 0, "3pa": 0, "ft": 0, "fta": 0})}

    for s in shots:
        team = s["team"]
        name = s["player"] or "Unknown"
        if s["type"] == "ft":
            api_stats[team][name]["fta"] += 1
            if s["made"]:
                api_stats[team][name]["ft"] += 1
        else:
            # Count ALL field goal attempts (made + missed)
            api_stats[team][name]["fga"] += 1
            if s["made"]:
                api_stats[team][name]["fg"] += 1
            if s["type"] == "3pt":
                api_stats[team][name]["3pa"] += 1
                if s["made"]:
                    api_stats[team][name]["3p"] += 1

    # Load roster for name/jersey mapping
    roster_path = data_dir / "roster.json"
    roster_lookup = {}  # (team_key, jersey) → roster_name
    if roster_path.exists():
        with open(roster_path) as f:
            roster_data = json.load(f)
        for side in ("home", "away"):
            for rp in roster_data.get(side, {}).get("players", []):
                roster_lookup[(side, rp["number"])] = rp["name"]

    # Name mapping: API uses "First Last" format, roster uses "Last, First"
    def match_by_name(api_name: str, roster_entries: dict) -> int | None:
        """Match API name to roster jersey number."""
        api_parts = api_name.lower().split()
        for (team, jersey), rname in roster_entries.items():
            roster_parts = rname.lower().replace(",", "").split()
            if api_parts and roster_parts:
                if api_parts[-1] in roster_parts or roster_parts[0] in api_parts:
                    return jersey
        return None

    def find_or_create_player(
        team_bs: TeamBoxScore,
        team_key: str,
        jersey: int,
    ) -> PlayerBoxScore:
        """Find existing player in box score or create new entry."""
        for p in team_bs.players:
            if p.jersey_number == jersey:
                return p
        # Create new player
        name = roster_lookup.get((team_key, jersey))
        p = PlayerBoxScore(
            player_id=jersey,
            jersey_number=jersey,
            team=team_key,
            player_name=name,
        )
        team_bs.players.append(p)
        return p

    # API is ground truth for scoring. Zero all scoring stats first,
    # then apply API data for matched players. Unmatched players get zero
    # scoring because if the API doesn't have them, they didn't score.
    for team_bs in [game.home, game.away]:
        for p in team_bs.players:
            p.fg = 0
            p.three_p = 0
            p.ft = 0
            p.fta = 0
            # Keep pipeline FGA and 3PA as shot attempt detection

    # Check per-team whether API logged any FG misses (FGA > FGM).
    # If not, the scorekeeper only recorded makes — FGA from API is useless.
    for team_key in ("home", "away"):
        has_miss = any(
            s["fga"] > s["fg"]
            for s in api_stats[team_key].values()
            if s["fg"] > 0
        )
        game.detection_summary[f"{team_key}_api_has_fg_misses"] = has_miss

    # Merge into box score
    for team_key, team_bs in [("home", game.home), ("away", game.away)]:
        team_roster = {k: v for k, v in roster_lookup.items() if k[0] == team_key}
        for api_name, stats in api_stats[team_key].items():
            jersey = match_by_name(api_name, team_roster)
            if jersey is None:
                print(f"  Warning: Could not match API player '{api_name}' to roster")
                continue

            player = find_or_create_player(team_bs, team_key, jersey)

            # Override FGM from API (ground truth)
            player.fg = stats["fg"]
            player.stat_sources["fg"] = StatSource.MANUAL
            # Only override FGA if API has real attempt data (misses logged)
            # If API FGA == FGM, the scorekeeper only logged makes — keep pipeline FGA
            if stats["fga"] > stats["fg"]:
                player.fga = stats["fga"]
                player.stat_sources["fga"] = StatSource.MANUAL
            elif player.fga < player.fg:
                # Pipeline FGA is less than API FGM — floor to FGM
                player.fga = player.fg
                player.stat_sources["fga"] = StatSource.MANUAL
            # else: keep pipeline FGA (it's >= FGM, from video detection)

            player.three_p = stats["3p"]
            player.stat_sources["three_p"] = StatSource.MANUAL
            # Same logic for 3PA
            if stats["3pa"] > stats["3p"]:
                player.three_pa = stats["3pa"]
                player.stat_sources["three_pa"] = StatSource.MANUAL
            elif player.three_pa < player.three_p:
                player.three_pa = player.three_p
                player.stat_sources["three_pa"] = StatSource.MANUAL
            # Merge FT data
            player.ft = stats["ft"]
            player.fta = stats["fta"]
            player.stat_sources["ft"] = StatSource.MANUAL
            player.stat_sources["fta"] = StatSource.MANUAL
            print(f"  Merged: {api_name} → #{jersey} {player.player_name}")

    return game


def main():
    parser = argparse.ArgumentParser(description="Compile game report")
    parser.add_argument("--data-dir", required=True, help="Pipeline output directory")
    parser.add_argument("--match-id", type=int, help="Match ID for API data merge")
    parser.add_argument("--output-dir", help="Output directory (default: data-dir/report)")
    parser.add_argument("--competition", default="", help="Competition name")
    parser.add_argument("--game-date", default="", help="Game date string")
    parser.add_argument("--docx", action="store_true", help="Generate DOCX output")
    parser.add_argument(
        "--llm-backend",
        default="gemini",
        choices=["gemini", "grok", "template"],
        help="LLM backend for scouting reports (default: gemini)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else data_dir / "report"

    # Load and clean
    print("Loading box score...")
    game = load_box_score(data_dir)
    game = clean_box_score(game, data_dir)

    print(f"Home: {game.home.team_name} ({len(game.home.players)} identified players)")
    print(f"Away: {game.away.team_name} ({len(game.away.players)} identified players)")

    # Merge API data
    if args.match_id:
        game = merge_api_data(game, args.match_id, data_dir)

    # Estimate team FGA from possessions (since pipeline FGA is noisy from tracker)
    # Assume roughly equal possessions per team, estimate from total score and typical U16 ORtg
    total_pts = game.home.total_pts + game.away.total_pts
    est_poss_per_team = max(total_pts / 1.8, 60)  # ~75 for a 150-point game

    for team in [game.home, game.away]:
        # Est FGA = Poss - 0.44*FTA (simplified, TO and OREB unknown)
        est_fga = est_poss_per_team - 0.44 * team.total_fta

        real_players = [
            p for p in team.players
            if p.jersey_number is not None and p.jersey_number != 0
            and not (p.player_name and "unattributed" in p.player_name.lower())
        ]
        # Use the flag set during API merge — checks API data directly,
        # not the merged box score (pipeline FGA > FGM != API miss logged).
        api_has_fg_misses = game.detection_summary.get(
            f"{team.team_key}_api_has_fg_misses", False
        )

        if not api_has_fg_misses and est_fga > 0:
            # Scorekeeper didn't log FG misses — distribute estimated team FGA
            # across real players proportionally by FGM share.
            team_fgm = sum(p.fg for p in real_players)
            if team_fgm > 0:
                for p in real_players:
                    if p.fg > 0:
                        player_est_fga = max(round(p.fg * est_fga / team_fgm), p.fg)
                        p.fga = player_est_fga
                        p.stat_sources["fga"] = StatSource.HEURISTIC
                    elif p.fga <= 0:
                        # No FGM and no pipeline FGA — give 1 estimated attempt
                        p.fga = 1
                        p.stat_sources["fga"] = StatSource.HEURISTIC
                    # Ensure 3PA >= 3PM
                    if p.three_pa < p.three_p:
                        # Estimate 3PA from team 3-point rate
                        team_3pm = sum(pp.three_p for pp in real_players)
                        team_3pa_est = max(int(est_fga * 0.35), team_3pm)  # ~35% of FGA are 3s
                        if team_3pm > 0:
                            p.three_pa = max(round(p.three_p * team_3pa_est / team_3pm), p.three_p)
                        else:
                            p.three_pa = p.three_p
                        p.stat_sources["three_pa"] = StatSource.HEURISTIC
                # Ensure 3PA <= FGA (runs after all FGA estimates are set)
                for p in real_players:
                    if p.three_pa > p.fga:
                        p.three_pa = p.fga
                print(f"  FGA estimated (no FG misses in API): est_team_FGA={est_fga:.0f}")
                game.detection_summary[f"{team.team_key}_fga_estimated"] = True
        else:
            # API has real FGA data or pipeline FGA is usable.
            # Add Unattributed pseudo-player for any remaining gap.
            current_fga = team.total_fga
            if current_fga < est_fga:
                gap = int(est_fga - current_fga)
                if gap > 0:
                    unid_player = PlayerBoxScore(
                        player_id=0,
                        jersey_number=0,
                        team=team.team_key,
                        player_name="Unattributed attempts",
                        fga=gap,
                        fg=0,
                        three_pa=int(gap * 0.35),
                    )
                    unid_player.stat_sources["fga"] = StatSource.HEURISTIC
                    team.players.append(unid_player)
                    game.detection_summary[f"{team.team_key}_est_fga_gap"] = gap

    # Print team totals after merge
    for team in [game.home, game.away]:
        print(f"\n{team.team_name} totals:")
        print(f"  FG: {team.total_fg}-{team.total_fga}")
        print(f"  3P: {team.total_three_p}-{team.total_three_pa}")
        print(f"  FT: {team.total_ft}-{team.total_fta}")
        print(f"  PTS: {team.total_pts}")
        print(f"  TO: {team.total_to}, OREB: {team.total_orb}")

    # Generate report
    print("\nGenerating film report...")
    llm_client = None
    if args.llm_backend != "template":
        try:
            from app.reporting.coach_agent import GeminiClient, GrokClient
            if args.llm_backend == "grok":
                llm_client = GrokClient()
            else:
                llm_client = GeminiClient()
            print(f"  LLM backend: {args.llm_backend}")
        except Exception as e:
            print(f"  Warning: LLM init failed ({e}), falling back to template")

    generator = FilmReportGenerator(
        llm_client=llm_client,
        competition=args.competition or "ØST GU16B",
        game_date=args.game_date or "2026-03-14",
    )
    report = generator.generate(game)

    # Attach chart image paths from pipeline output
    def _chart(name: str) -> str:
        p = data_dir / name
        return str(p) if p.exists() else ""

    report.chart_score_flow = _chart("score_flow.png")
    report.chart_shot_all = _chart("shot_chart.png")
    report.chart_shot_home = _chart("shot_chart_home.png")
    report.chart_shot_away = _chart("shot_chart_away.png")
    report.chart_shot_home_quarters = [
        _chart(f"shot_chart_home_q{q}.png") for q in range(1, 5)
    ]
    report.chart_shot_away_quarters = [
        _chart(f"shot_chart_away_q{q}.png") for q in range(1, 5)
    ]

    chart_count = sum(1 for p in [
        report.chart_score_flow, report.chart_shot_all,
        report.chart_shot_home, report.chart_shot_away,
        *report.chart_shot_home_quarters, *report.chart_shot_away_quarters,
    ] if p)
    print(f"  Found {chart_count} chart images to embed in report")

    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build ISO filename: "2026-03-14T14:07 — Notodden Thunders (D) 50 — EB-85 16"
    iso_date = report.game_date.replace("/", "-")
    base_name = f"{iso_date} — {report.home_name} {report.home_score} — {report.away_name} {report.away_score}"
    # Sanitise for filesystem (no colons on Windows, keep safe)
    safe_name = base_name.replace(":", "").replace("/", "-")

    json_path = output_dir / f"{safe_name}.json"
    generator.save_json(report, json_path)
    print(f"\nSaved JSON: {json_path}")

    md_path = output_dir / f"{safe_name}.md"
    generator.save_markdown(report, md_path)
    print(f"Saved Markdown: {md_path}")

    # DOCX output
    if args.docx:
        from app.reporting.docx_renderer import render_docx
        docx_path = output_dir / f"{safe_name}.docx"
        try:
            render_docx(report, docx_path)
            print(f"Saved DOCX: {docx_path}")
        except Exception as e:
            print(f"Warning: DOCX generation failed: {e}")

    # Also save advanced stats separately
    if report.advanced_stats:
        adv_path = output_dir / f"{safe_name} — advanced_stats.json"
        with open(adv_path, "w") as f:
            json.dump(report.advanced_stats.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"Saved Advanced Stats: {adv_path}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"FILM REPORT: {report.home_name} {report.home_score} — {report.away_name} {report.away_score}")
    print(f"{'='*60}")
    print(f"\n{report.game_summary}")

    if report.awards:
        print("\nAwards:")
        for a in report.awards:
            print(f"  {a.award_name}: #{a.jersey_number} {a.player_name} ({a.team})")
            print(f"    {a.stat_line}")

    if report.losing_factor:
        print(f"\nLosing Factor: {report.losing_factor}")
    if report.winning_adjustment:
        print(f"Winning Adjustment: {report.winning_adjustment}")


if __name__ == "__main__":
    main()

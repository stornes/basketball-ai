"""Fetch game data from kamper.basket.no (Norwegian Basketball Federation).

Pulls roster, quarter scores, and play-by-play from the BasketLive API
and outputs a roster.json compatible with the HoopsVision pipeline.

Usage:
    python scripts/fetch_game.py --match-id 8254973
    python scripts/fetch_game.py --match-id 8254973 --output data/rosters/game.json
    python scripts/fetch_game.py --date 2026-03-14 --list  # list all games on date
    python scripts/fetch_game.py --match-id 8258410 --play-by-play  # include shot log

API base: https://sf14-terminlister-prod-app.azurewebsites.net
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError

API_BASE = "https://sf14-terminlister-prod-app.azurewebsites.net"
SPORT_ID = 199  # Basketball


def _fetch_json(endpoint: str) -> dict | list:
    """Fetch JSON from the BasketLive API."""
    import gzip
    url = f"{API_BASE}{endpoint}"
    req = Request(url, headers={"Accept": "application/json", "Accept-Encoding": "gzip, deflate"})
    try:
        with urlopen(req, timeout=15) as resp:
            data = resp.read()
            # Handle gzip-encoded responses
            if data[:2] == b'\x1f\x8b':
                data = gzip.decompress(data)
            return json.loads(data.decode())
    except HTTPError as e:
        print(f"API error {e.code}: {url}", file=sys.stderr)
        sys.exit(1)


def list_games_on_date(date_str: str) -> list[dict]:
    """List all basketball games on a given date.

    Args:
        date_str: Date in YYYY-MM-DD format.
    """
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    api_date = f"{dt.month}/{dt.day}/{dt.year}"
    games = _fetch_json(f"/ta/ScheduledMatchesBySport/?sportIds={SPORT_ID}&date={api_date}")
    return games


def fetch_match(match_id: int) -> dict:
    """Fetch match metadata."""
    return _fetch_json(f"/ta/Match?matchId={match_id}")


def fetch_incidents(match_id: int) -> dict:
    """Fetch play-by-play incidents and quarter scores."""
    return _fetch_json(f"/ta/MatchIncidents?matchId={match_id}")


def fetch_team_members(team_id: int) -> list[dict]:
    """Fetch team roster (members list)."""
    return _fetch_json(f"/ta/TeamMembers/?teamId={team_id}")


def _parse_quarter_scores(incidents: dict) -> tuple[list[int], list[int]]:
    """Extract cumulative quarter scores from match incidents.

    Returns (home_cumulative, away_cumulative).
    """
    period_results = incidents.get("matchPeriodResults", [])
    # Sort by partialResultTypeId to get Q1, Q2, Q3, Q4, OT... in order
    period_results.sort(key=lambda x: x.get("partialResultTypeId", 0))

    home_cum = []
    away_cum = []
    home_total = 0
    away_total = 0
    for pr in period_results:
        home_total += pr.get("homeGoals", 0)
        away_total += pr.get("awayGoals", 0)
        home_cum.append(home_total)
        away_cum.append(away_total)
    return home_cum, away_cum


def _parse_shots(incidents: dict) -> list[dict]:
    """Extract shot events from match incidents.

    The API uses parent-child pairs:
      Parent: incidentType="Skudd", incidentSubType="2p"/"3p"/"1p"/"1p bom" etc
      Child:  incidentSubType="Spiller" with firstName/lastName

    Returns list of shot dicts with player, team, time, type, made.
    """
    all_incidents = incidents.get("matchIncidents", [])

    # Period ID → quarter number mapping
    period_map = {}
    for pr in incidents.get("matchPeriodResults", []):
        pid = pr.get("partialResultTypeId")
        ptype = pr.get("partialResultType", "")
        if "1." in ptype:
            period_map[pid] = 1
        elif "2." in ptype:
            period_map[pid] = 2
        elif "3." in ptype:
            period_map[pid] = 3
        elif "4." in ptype:
            period_map[pid] = 4
        else:
            # OT or other
            period_map[pid] = int(ptype.split(".")[0]) if ptype[0].isdigit() else 5

    # Group by parent ID to pair shot type with player
    shot_parents = {}  # matchIncidentId → incident
    children = {}  # parentId → [child incidents]

    for inc in all_incidents:
        if inc.get("incidentType") == "Skudd":
            mid = inc.get("matchIncidentId")
            pid = inc.get("parentId")
            if pid:
                children.setdefault(pid, []).append(inc)
            else:
                shot_parents[mid] = inc

    shots = []
    for parent_id, parent in shot_parents.items():
        sub = parent.get("incidentSubType", "")
        team_code = parent.get("team", "")
        time_val = parent.get("time", 0)
        period_id = parent.get("partialResultTypeId")
        quarter = period_map.get(period_id)

        # Determine shot type and outcome
        is_made = "bom" not in sub.lower()
        if "3p" in sub:
            shot_type = "3pt"
            points = 3 if is_made else 0
        elif "1p" in sub:
            shot_type = "ft"
            points = 1 if is_made else 0
        else:
            shot_type = "2pt"
            points = 2 if is_made else 0

        # Find the player child
        player_name = None
        for child in children.get(parent_id, []):
            if child.get("incidentSubType") == "Spiller":
                first = child.get("firstName", "") or ""
                last = child.get("lastName", "") or ""
                player_name = f"{first} {last}".strip()
                break

        shots.append({
            "player": player_name,
            "team": "home" if team_code == "H" else "away",
            "time": time_val,
            "quarter": quarter,
            "type": shot_type,
            "made": is_made,
            "points": points,
        })

    return shots


def _build_roster_json(
    match: dict,
    incidents: dict,
    include_play_by_play: bool = False,
) -> dict:
    """Build a roster.json compatible with the HoopsVision pipeline.

    Uses match data for team names and the incidents endpoint for
    quarter scores. Player rosters come from the play-by-play
    (names that appear in shot/foul incidents).
    """
    home_name = (match.get("hometeamOverriddenName") or match.get("hometeam", "Home")).strip()
    away_name = (match.get("awayteamOverriddenName") or match.get("awayteam", "Away")).strip()

    home_cum, away_cum = _parse_quarter_scores(incidents)

    # Extract unique players from incidents
    all_incidents = incidents.get("matchIncidents", [])
    home_players: dict[str, dict] = {}  # personId → {name, ...}
    away_players: dict[str, dict] = {}

    for inc in all_incidents:
        person_id = inc.get("personId")
        first = inc.get("firstName")
        last = inc.get("lastName")
        team_code = inc.get("team")
        if not person_id or not first:
            continue

        name = f"{last}, {first}" if last else first
        entry = {"name": name, "person_id": person_id}

        if team_code == "H":
            home_players.setdefault(person_id, entry)
        elif team_code == "B":
            away_players.setdefault(person_id, entry)

    # Build player lists (no jersey numbers from Simple client)
    # For Advanced (BLNO) games, jersey numbers would come from PubNub MatchData
    home_list = [
        {"number": 0, "name": p["name"], "captain": False}
        for p in home_players.values()
    ]
    away_list = [
        {"number": 0, "name": p["name"], "captain": False}
        for p in away_players.values()
    ]

    result = {
        "source": "kamper.basket.no",
        "match_id": match.get("matchId"),
        "match_date": match.get("matchDate"),
        "tournament": match.get("tournamentName"),
        "venue": match.get("activityAreaName"),
        "result": match.get("matchResult", {}).get("matchEndResult"),
        "home": {
            "name": home_name,
            "team_id": match.get("hometeamId"),
            "players": home_list,
            "staff": [],
            "scores_cumulative": home_cum,
        },
        "away": {
            "name": away_name,
            "team_id": match.get("awayteamId"),
            "players": away_list,
            "staff": [],
            "scores_cumulative": away_cum,
        },
    }

    if include_play_by_play:
        shots = _parse_shots(incidents)
        result["play_by_play"] = {
            "shots": shots,
            "total_shots": len(shots),
            "made_shots": len([s for s in shots if s["made"]]),
        }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Fetch game data from kamper.basket.no"
    )
    parser.add_argument("--match-id", type=int, help="Match ID from kamper.basket.no URL")
    parser.add_argument("--date", help="List games on date (YYYY-MM-DD)")
    parser.add_argument("--list", action="store_true", help="List games (with --date)")
    parser.add_argument("--output", "-o", help="Output path for roster.json")
    parser.add_argument("--play-by-play", action="store_true",
                        help="Include shot-by-shot play-by-play data")
    parser.add_argument("--pretty", action="store_true", default=True,
                        help="Pretty-print JSON output")
    args = parser.parse_args()

    if args.date and args.list:
        games = list_games_on_date(args.date)
        print(f"Games on {args.date}: {len(games)}")
        print()
        for g in games:
            mid = g.get("matchId")
            home = (g.get("hometeamOverriddenName") or g.get("hometeam", "?")).strip()
            away = (g.get("awayteamOverriddenName") or g.get("awayteam", "?")).strip()
            result = g.get("result", "")
            tournament = g.get("tournamentName", "")
            time_str = str(g.get("matchStartTime", ""))
            venue = g.get("activityAreaName", "")
            print(f"  {mid:>10} | {time_str:>5} | {home} vs {away} | {result} | {tournament} | {venue}")
        return

    if not args.match_id:
        parser.error("--match-id is required (or use --date --list to find one)")

    print(f"Fetching match {args.match_id}...")
    match = fetch_match(args.match_id)
    incidents = fetch_incidents(args.match_id)

    home = (match.get("hometeamOverriddenName") or match.get("hometeam", "?")).strip()
    away = (match.get("awayteamOverriddenName") or match.get("awayteam", "?")).strip()
    mr = match.get("matchResult", {})
    result_str = mr.get("matchEndResult", "")
    print(f"  {home} {result_str} {away}")
    print(f"  Tournament: {match.get('tournamentName', '?')}")
    print(f"  Date: {match.get('matchDate', '?')}")

    home_cum, away_cum = _parse_quarter_scores(incidents)
    print(f"  Quarter scores: {home_cum} vs {away_cum}")

    all_inc = incidents.get("matchIncidents", [])
    print(f"  Incidents: {len(all_inc)}")

    roster_json = _build_roster_json(match, incidents, include_play_by_play=args.play_by_play)

    if args.play_by_play:
        pbp = roster_json["play_by_play"]
        print(f"  Shots: {pbp['total_shots']} ({pbp['made_shots']} made)")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        indent = 2 if args.pretty else None
        with open(out_path, "w") as f:
            json.dump(roster_json, f, indent=indent, ensure_ascii=False)
        print(f"\nSaved to {out_path}")
    else:
        indent = 2 if args.pretty else None
        print()
        print(json.dumps(roster_json, indent=indent, ensure_ascii=False))


if __name__ == "__main__":
    main()

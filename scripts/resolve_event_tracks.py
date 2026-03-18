"""Resolve jersey numbers for unresolved event tracks (assists, rebounds, steals).

Uses Sherlock deduction (multi-frame VLM reasoning) to identify players who had
events but weren't resolved during the original pipeline run.

Usage:
    .venv/bin/python scripts/resolve_event_tracks.py \
        --data-dir data/outputs/v1.7.0 \
        --vlm-backend grok
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(override=True)

from app.tracking.jersey_number import sherlock_resolve, PlayerDescription


def main():
    parser = argparse.ArgumentParser(description="Resolve event track jersey numbers via Sherlock")
    parser.add_argument("--data-dir", required=True, help="Pipeline output directory")
    parser.add_argument("--vlm-backend", default="grok", choices=["gemini", "grok", "anthropic"])
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # Load existing data
    print("Loading pipeline data...")
    with open(data_dir / "box_score.json") as f:
        bs = json.load(f)
    with open(data_dir / "player_tracks.json") as f:
        all_tracks = json.load(f)
    with open(data_dir / "player_descriptions.json") as f:
        desc_raw = json.load(f)
    roster_path = data_dir / "roster.json"
    with open(roster_path) as f:
        roster_data = json.load(f)

    video_path = bs.get("video_path", "")
    if not Path(video_path).exists():
        video_path = str(Path(__file__).parent.parent / video_path)
    if not Path(video_path).exists():
        print(f"Error: Video not found at {video_path}")
        return

    # Build existing jersey map and descriptions
    existing_map: dict[int, int] = {}
    descriptions: dict[int, PlayerDescription] = {}
    for tid_str, d in desc_raw.items():
        tid = int(tid_str)
        jn = d.get("jersey_number")
        if jn is not None:
            existing_map[tid] = jn
        descriptions[tid] = PlayerDescription(
            track_id=d.get("track_id", tid),
            jersey_number=jn,
            team_color=d.get("team_color"),
            description=d.get("description"),
        )

    print(f"Existing jersey map: {len(existing_map)} resolved tracks")

    # Find unresolved event tracks (AST, STL, ORB, DRB > 0 but no jersey)
    unresolved = []
    for side in ["home", "away"]:
        for p in bs[side]["players"]:
            if p.get("jersey_number") is not None:
                continue
            tid = p["player_id"]
            if tid in existing_map:
                continue
            has_events = (
                p.get("ast", 0) > 0 or p.get("stl", 0) > 0
                or p.get("orb", 0) > 0 or p.get("drb", 0) > 0
            )
            if has_events:
                stats = f"AST={p.get('ast',0)} STL={p.get('stl',0)} ORB={p.get('orb',0)} DRB={p.get('drb',0)}"
                print(f"  Unresolved: {side} tid={tid} ({stats})")
                unresolved.append(SimpleNamespace(
                    shooter_track_id=tid,
                    team=side,
                ))

    if not unresolved:
        print("No unresolved event tracks found.")
        return

    print(f"\nRunning Sherlock on {len(unresolved)} unresolved event tracks...")

    sherlock_map, sherlock_desc = sherlock_resolve(
        video_path=video_path,
        all_tracks=all_tracks,
        fps=28.19,
        roster_home=roster_data.get("home", {}),
        roster_away=roster_data.get("away", {}),
        descriptions=descriptions,
        jersey_map=existing_map,
        unresolved_shots=unresolved,
        vlm_backend=args.vlm_backend,
    )

    # Summary
    print(f"\n{'='*50}")
    print(f"SHERLOCK RESOLVED: {len(sherlock_map)} out of {len(unresolved)} event tracks")
    for tid, jn in sorted(sherlock_map.items()):
        for side in ["home", "away"]:
            for p in bs[side]["players"]:
                if p["player_id"] == tid:
                    stats = f"AST={p.get('ast',0)} STL={p.get('stl',0)} ORB={p.get('orb',0)} DRB={p.get('drb',0)}"
                    print(f"  tid={tid} -> #{jn} ({side}): {stats}")

    # Update box_score.json with resolved jersey numbers
    if sherlock_map:
        updated = 0
        for side in ["home", "away"]:
            for p in bs[side]["players"]:
                tid = p["player_id"]
                if tid in sherlock_map:
                    p["jersey_number"] = sherlock_map[tid]
                    p["stat_sources"] = p.get("stat_sources", {})
                    p["stat_sources"]["jersey"] = "sherlock"
                    updated += 1

        # Save updated box score
        out_path = data_dir / "box_score.json"
        with open(out_path, "w") as f:
            json.dump(bs, f, indent=2, ensure_ascii=False)
        print(f"\nUpdated {updated} players in {out_path}")

        # Update descriptions
        if sherlock_desc:
            for tid, d in sherlock_desc.items():
                desc_raw[str(tid)] = {
                    "track_id": d.track_id,
                    "jersey_number": d.jersey_number,
                    "team_color": d.team_color,
                    "description": d.description,
                }
            with open(data_dir / "player_descriptions.json", "w") as f:
                json.dump(desc_raw, f, indent=2, ensure_ascii=False)
            print(f"Updated {len(sherlock_desc)} player descriptions")

        print("\nRe-run compile_film_report.py to regenerate the report.")
    else:
        print("\nNo new resolutions. The tracks may be too small or ambiguous for VLM.")


if __name__ == "__main__":
    main()

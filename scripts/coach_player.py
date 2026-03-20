"""CLI entry point for the visual coaching system.

Generates a coaching report for a specific player by:
1. Loading pipeline data (box_score.json, shots.json, player_tracks.json, possessions.json)
2. Finding all track IDs that belong to the target player
3. Extracting 20-25 key clips from the video
4. Sending clips to Gemini Vision for analysis
5. Synthesising a 7-section coaching report
6. Saving a DOCX report

Usage:
    python scripts/coach_player.py \\
        --data-dir data/outputs/v2.1.0 \\
        --video data/videos/2026-03-14_notodden-thunders-d_vs_eb-85.MP4 \\
        --player "Victor Stornes" \\
        --jersey 4 \\
        --team away \\
        --output-dir data/coaching/victor-2026-03-14
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(override=True)

from app.coaching.clip_extractor import (
    ClipExtractor,
    group_tracks_by_id,
    resolve_player_track_ids,
)
from app.coaching.visual_analyst import VisualCoachingAnalyst
from app.coaching.report_writer import CoachingReportWriter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a visual coaching report for a specific player."
    )
    parser.add_argument("--data-dir", required=True, help="Pipeline output directory")
    parser.add_argument("--video", required=True, help="Path to game video file")
    parser.add_argument("--player", required=True, help="Player full name")
    parser.add_argument("--jersey", required=True, type=int, help="Jersey number")
    parser.add_argument("--team", required=True, choices=["home", "away"], help="Team side")
    parser.add_argument("--output-dir", required=True, help="Directory to save coaching output")
    parser.add_argument("--max-clips", type=int, default=25, help="Maximum clips to extract")
    parser.add_argument("--clip-duration", type=float, default=10.0, help="Clip duration in seconds")
    parser.add_argument("--model", default="gemini-3-flash-preview", help="Gemini model to use")
    parser.add_argument("--no-docx", action="store_true", help="Skip DOCX rendering, save JSON only")
    return parser.parse_args()


def load_json(path: Path) -> list | dict:
    with open(path) as f:
        return json.load(f)


def get_video_fps(video_path: str) -> float:
    """Get FPS from video file using cv2."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps > 0 else 25.0


def main() -> int:
    args = parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    clips_dir = output_dir / "clips"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading pipeline data from {data_dir}...")

    # Load pipeline data
    box_score = load_json(data_dir / "box_score.json")
    shots_raw: list[dict] = load_json(data_dir / "shots.json")
    all_tracks: list[dict] = load_json(data_dir / "player_tracks.json")
    possessions_raw: list[dict] = load_json(data_dir / "possessions.json")

    # Load player descriptions for track ID resolution
    desc_path = data_dir / "player_descriptions.json"
    player_descriptions: dict = load_json(desc_path) if desc_path.exists() else {}

    # Get game metadata
    game_date = box_score.get("game_date", "")
    team_data = box_score.get(args.team, {})
    team_name = team_data.get("team_name", args.team)

    print(f"Resolving track IDs for {args.player} (#{args.jersey}, {args.team})...")

    # Resolve track IDs for target player
    player_track_ids = resolve_player_track_ids(
        player_descriptions=player_descriptions,
        target_jersey=args.jersey,
        target_team=args.team,
    )

    if not player_track_ids:
        print(f"  Warning: Could not resolve track IDs via player_descriptions.json.")
        print(f"  Falling back to all tracks for team='{args.team}'.")
        # Collect all unique track IDs from the target team
        player_track_ids = list(set(
            int(t["track_id"])
            for t in all_tracks
            if t.get("team") == args.team
        ))
        print(f"  Using {len(player_track_ids)} track IDs for team '{args.team}'.")
    else:
        print(f"  Found {len(player_track_ids)} track ID(s): {player_track_ids}")

    player_track_id_set = set(player_track_ids)

    # Filter tracks to only the target player
    player_tracks = [t for t in all_tracks if int(t["track_id"]) in player_track_id_set]
    print(f"  Player track frames: {len(player_tracks)}")

    # Filter shots to only the target player
    player_shots = [
        s for s in shots_raw
        if s.get("shooter_track_id") is not None
        and int(s["shooter_track_id"]) in player_track_id_set
    ]
    print(f"  Player shot events: {len(player_shots)}")

    # Filter possessions to only the target player
    player_possessions = [
        p for p in possessions_raw
        if int(p.get("player_track_id", -1)) in player_track_id_set
    ]
    print(f"  Player possession events: {len(player_possessions)}")

    # Get video FPS
    print(f"\nOpening video: {args.video}")
    fps = get_video_fps(args.video)
    print(f"  FPS: {fps:.2f}")

    # Extract clips
    print(f"\nExtracting up to {args.max_clips} clips...")
    # Load all tracks for annotation overlays
    all_tracks = None
    tracks_path = data_dir / "player_tracks.json"
    if tracks_path.exists():
        print(f"  Loading tracks for annotation overlays...")
        with open(tracks_path) as f:
            all_tracks = json.load(f)
        print(f"  {len(all_tracks)} track points loaded")

    extractor = ClipExtractor(
        video_path=args.video,
        fps=fps,
        all_tracks=all_tracks,
        player_track_ids=set(player_track_ids),
        player_team=args.team,
    )
    clips = extractor.extract_player_clips(
        player_tracks=player_tracks,
        shot_events=player_shots,
        possession_events=player_possessions,
        max_clips=args.max_clips,
        clip_duration_sec=args.clip_duration,
        output_dir=str(clips_dir),
    )
    print(f"  Extracted {len(clips)} clips:")
    for clip in clips:
        print(f"    [{clip.category}] {clip.context} → {Path(clip.clip_path).name}")

    # Build player context for VLM prompts
    player_context = {
        "player_name": args.player,
        "jersey": args.jersey,
        "team_color": "dark blue" if args.team == "away" else "white",
        "team": team_name,
        "game_date": game_date,
    }

    # Analyse clips with VLM
    print(f"\nAnalysing {len(clips)} clips with {args.model}...")
    analyst = VisualCoachingAnalyst(model=args.model)
    analyses = analyst.analyse_clips(clips, player_context)

    # Count results
    successful = [a for a in analyses if not a.error]
    failed = [a for a in analyses if a.error]
    print(f"  Successful: {len(successful)}, Failed: {len(failed)}")
    if failed:
        for a in failed:
            print(f"    FAILED: {a.context} — {a.error}")

    # Synthesise coaching report
    print("\nSynthesising coaching report...")
    report = analyst.synthesise_report(analyses, player_context)

    # Save outputs
    writer = CoachingReportWriter()

    json_path = output_dir / f"coaching_{args.player.replace(' ', '-').lower()}_{game_date}.json"
    writer.write_json(report, json_path)
    print(f"  JSON report: {json_path}")

    if not args.no_docx:
        docx_path = output_dir / f"coaching_{args.player.replace(' ', '-').lower()}_{game_date}.docx"
        try:
            writer.write_docx(report, docx_path)
            print(f"  DOCX report: {docx_path}")
        except Exception as exc:
            print(f"  Warning: DOCX generation failed: {exc}")
            print(f"  JSON report saved — use --no-docx to skip DOCX.")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

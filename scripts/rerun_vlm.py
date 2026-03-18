"""Re-run VLM jersey number recognition on existing pipeline output.

Modes:
  --mode standard: Single best crop per track, simple VLM prompt
  --mode sherlock: Multi-frame temporal window + deductive reasoning with roster context

Uses player_tracks.json to find bboxes for made-shot track IDs,
extracts crops from the video, sends to VLM, and updates shots.json.
"""

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv(override=True)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.tracking.jersey_number import (
    JerseyNumberReader,
    PlayerDescription,
    VLMBackend,
    _build_sherlock_prompt,
    _call_gemini_multi_image,
    _call_anthropic_multi_image,
    _call_openai_multi_image,
    _call_grok_multi_image,
    _extract_crop_from_video,
    _find_temporal_crops,
    _parse_sherlock_response,
    _parse_vlm_response,
    _VLM_PROMPT,
    _call_gemini,
    _call_anthropic,
    _call_openai,
    _call_grok,
)
from app.vision.detection_types import BoundingBox


def run_standard(args, shots, all_tracks, output_dir):
    """Standard mode: single best crop per track."""
    # Find track IDs to resolve
    if args.all_shots:
        target_shots = [s for s in shots if s.get("shooter_track_id") is not None]
    else:
        target_shots = [s for s in shots
                       if s.get("shooter_track_id") is not None
                       and s.get("outcome") == "made"]

    track_ids = {int(s["shooter_track_id"]) for s in target_shots}
    print(f"Found {len(target_shots)} target shots with {len(track_ids)} unique track IDs")

    # Find the largest bbox per target track ID
    best_per_track: dict[int, dict] = {}
    for t in all_tracks:
        tid = t["track_id"]
        if tid not in track_ids:
            continue
        bbox = t["bbox"]
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if tid not in best_per_track or area > best_per_track[tid]["area"]:
            best_per_track[tid] = {
                "frame_idx": t["frame_idx"],
                "bbox": bbox,
                "area": area,
            }

    print(f"Found best crops for {len(best_per_track)} tracks")

    cap = cv2.VideoCapture(args.video)
    reader = JerseyNumberReader(vlm_backend=args.vlm_backend, max_crops_per_track=1)

    sorted_tracks = sorted(best_per_track.items(), key=lambda x: x[1]["frame_idx"])
    for tid, info in sorted_tracks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, info["frame_idx"])
        ret, frame = cap.read()
        if not ret:
            continue
        bb = BoundingBox(x1=info["bbox"][0], y1=info["bbox"][1],
                        x2=info["bbox"][2], y2=info["bbox"][3])
        reader.force_collect_sample(tid, frame, bb)

    cap.release()
    jersey_map = reader.resolve(track_ids=track_ids)
    return jersey_map, reader.player_descriptions


def run_sherlock(args, shots, all_tracks, output_dir):
    """Sherlock mode: multi-frame deductive reasoning with roster context."""
    # Load roster
    roster_path = output_dir / "roster.json"
    if not roster_path.exists():
        print("ERROR: roster.json required for Sherlock mode")
        sys.exit(1)

    with open(roster_path) as f:
        roster = json.load(f)

    home_players = roster["home"]["players"]
    away_players = roster["away"]["players"]
    home_name = roster["home"]["name"]
    away_name = roster["away"]["name"]

    # Load existing descriptions to build known features
    desc_path = output_dir / "player_descriptions.json"
    existing_desc = {}
    if desc_path.exists():
        with open(desc_path) as f:
            existing_desc = json.load(f)

    # Build known features from already-resolved tracks
    # Map: jersey_number → feature string (for each team)
    home_known: dict[int, str] = {}
    away_known: dict[int, str] = {}
    for tid, d in existing_desc.items():
        num = d.get("jersey_number")
        desc_text = d.get("description", "")
        color = (d.get("team_color") or "").lower()
        if num is None or not desc_text:
            continue
        # Determine team from jersey color
        if "white" in color:
            home_known[num] = desc_text
        elif "blue" in color or "dark" in color:
            away_known[num] = desc_text

    print(f"Known home features: {sorted(home_known.keys())}")
    print(f"Known away features: {sorted(away_known.keys())}")

    # Find unresolved made-shot tracks
    if args.all_shots:
        target_shots = [s for s in shots if s.get("shooter_track_id") is not None]
    else:
        target_shots = [s for s in shots
                       if s.get("shooter_track_id") is not None
                       and s.get("outcome") == "made"]

    all_track_ids = {int(s["shooter_track_id"]) for s in target_shots}

    # Filter to unresolved only (jersey_number is null)
    unresolved_shots = [s for s in target_shots if s.get("jersey_number") is None]
    unresolved_ids = {int(s["shooter_track_id"]) for s in unresolved_shots}

    print(f"\nTotal made-shot tracks: {len(all_track_ids)}")
    print(f"Unresolved tracks: {len(unresolved_ids)}")

    # Pre-filter: identify non-player tracks from existing descriptions
    non_player_ids = set()
    for tid_str, d in existing_desc.items():
        tid = int(tid_str)
        desc_text = (d.get("description") or "").lower()
        color = (d.get("team_color") or "").lower()
        if any(kw in desc_text for kw in [
            "no basketball player", "not visible", "court floor",
            "whistle", "referee", "grey t-shirt", "diadora",
            "grey beard", "black trousers", "black pants"
        ]):
            non_player_ids.add(tid)
        if color in ("grey", "dark grey", "dark", "unknown") and d.get("jersey_number") is None:
            # Likely coach/referee/spectator
            if any(kw in desc_text for kw in ["t-shirt", "trousers", "watch", "wristband"]):
                non_player_ids.add(tid)

    filtered_ids = unresolved_ids - non_player_ids
    print(f"Non-player tracks filtered: {len(non_player_ids)} ({non_player_ids})")
    print(f"Tracks for Sherlock analysis: {len(filtered_ids)}")

    # Get team assignment for each unresolved track from shots
    track_teams: dict[int, str] = {}
    for s in target_shots:
        tid = int(s["shooter_track_id"])
        if tid in filtered_ids:
            track_teams[tid] = s.get("team", "unknown")

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)

    vlm = VLMBackend.get(args.vlm_backend)

    jersey_map: dict[int, int] = {}
    descriptions: dict[int, PlayerDescription] = {}

    for tid in sorted(filtered_ids):
        team = track_teams.get(tid, "unknown")
        print(f"\n{'='*60}")
        print(f"  Track {tid} | Team: {team}")

        # Find temporal crops
        temporal = _find_temporal_crops(all_tracks, tid, fps, window_sec=10.0, max_crops=5)
        if not temporal:
            print(f"  No track data found")
            continue

        # Extract crops from video
        crops = []
        for t in temporal:
            crop_bytes = _extract_crop_from_video(cap, t["frame_idx"], t["bbox"])
            if crop_bytes:
                crops.append(crop_bytes)

        if not crops:
            print(f"  Could not extract any crops")
            continue

        print(f"  Extracted {len(crops)} crops from frames "
              f"{[t['frame_idx'] for t in temporal]}")

        # Determine roster context
        if team == "home":
            roster_players = home_players
            team_name = home_name
            jersey_color = "white with blue accents"
            known = home_known
        elif team == "away":
            roster_players = away_players
            team_name = away_name
            jersey_color = "dark blue"
            known = away_known
        else:
            # Unknown team — try standard VLM on best crop
            print(f"  Unknown team, falling back to standard VLM")
            try:
                response = vlm.call_single(crops[0], _VLM_PROMPT)
                num, color, desc = _parse_vlm_response(response)
                if num is not None:
                    jersey_map[tid] = num
                    print(f"  Standard VLM → #{num}")
                descriptions[tid] = PlayerDescription(tid, num, color, desc)
            except Exception as e:
                print(f"  VLM error: {e}")
            time.sleep(0.2)
            continue

        # Build Sherlock prompt
        prompt = _build_sherlock_prompt(
            team_label=team,
            team_name=team_name,
            jersey_color=jersey_color,
            roster_players=roster_players,
            known_features=known,
            n_images=len(crops),
        )

        try:
            if len(crops) > 1:
                response = vlm.call_multi(crops, prompt)
            else:
                response = vlm.call_single(crops[0], prompt)

            num, confidence, full_text = _parse_sherlock_response(response)

            print(f"  Sherlock deduction:")
            # Print key lines
            for line in full_text.split("\n"):
                line = line.strip()
                if line and any(kw in line.upper() for kw in [
                    "OBSERVATION", "ELIMINAT", "DEDUCTION", "NUMBER"
                ]):
                    print(f"    {line[:120]}")

            if num is not None and confidence in ("high", "medium"):
                jersey_map[tid] = num
                print(f"  → RESOLVED: #{num} (confidence: {confidence})")

                # Update known features for future deductions
                # Extract description from observations
                obs_text = ""
                for line in full_text.split("\n"):
                    if "OBSERVATION" in line.upper():
                        obs_text = line.split(":", 1)[-1].strip() if ":" in line else ""
                        break

                if team == "home":
                    home_known[num] = obs_text or f"track {tid}"
                else:
                    away_known[num] = obs_text or f"track {tid}"
            elif num is not None:
                print(f"  → LOW CONFIDENCE: #{num} ({confidence}) — not accepted")
            else:
                print(f"  → UNRESOLVED")

            descriptions[tid] = PlayerDescription(
                track_id=tid,
                jersey_number=num if confidence in ("high", "medium") else None,
                team_color=jersey_color,
                description=full_text[:500],
            )

        except Exception as e:
            print(f"  Sherlock error: {e}")

        time.sleep(0.3)  # Rate limiting

    cap.release()

    # Also handle non-player tracks — mark them explicitly
    for tid in non_player_ids & unresolved_ids:
        if tid in existing_desc:
            d = existing_desc[str(tid)]
            descriptions[tid] = PlayerDescription(
                track_id=tid,
                jersey_number=None,
                team_color=d.get("team_color"),
                description=f"[NON-PLAYER] {d.get('description', 'unknown')}",
            )

    return jersey_map, descriptions


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Re-run VLM jersey recognition")
    parser.add_argument("--output-dir", required=True, help="Pipeline output directory")
    parser.add_argument("--video", required=True, help="Source video path")
    parser.add_argument("--vlm-backend", default="gemini", choices=["gemini", "anthropic", "openai", "grok"])
    parser.add_argument("--all-shots", action="store_true", help="Resolve all shots, not just made")
    parser.add_argument("--mode", default="standard", choices=["standard", "sherlock"],
                       help="Resolution mode: standard (single crop) or sherlock (multi-frame deductive)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Load shots
    shots_path = output_dir / "shots.json"
    with open(shots_path) as f:
        shots = json.load(f)

    # Load all tracks
    tracks_path = output_dir / "player_tracks.json"
    print(f"Loading player tracks from {tracks_path}...")
    with open(tracks_path) as f:
        all_tracks = json.load(f)
    print(f"Loaded {len(all_tracks)} track entries")

    # Run selected mode
    if args.mode == "sherlock":
        jersey_map, descriptions = run_sherlock(args, shots, all_tracks, output_dir)
    else:
        jersey_map, descriptions = run_standard(args, shots, all_tracks, output_dir)

    print(f"\n{'='*60}")
    print(f"VLM Results: {len(jersey_map)} jersey numbers resolved")
    for tid, num in sorted(jersey_map.items()):
        print(f"  Track {tid} → #{num}")

    # Update shots.json
    updated = 0
    for shot in shots:
        if shot.get("shooter_track_id") is not None:
            tid = int(shot["shooter_track_id"])
            if tid in jersey_map:
                shot["jersey_number"] = jersey_map[tid]
                updated += 1

    with open(shots_path, "w") as f:
        json.dump(shots, f, indent=2)
    print(f"\nUpdated {updated} shots in {shots_path}")

    # Merge with existing descriptions
    desc_path = output_dir / "player_descriptions.json"
    existing = {}
    if desc_path.exists():
        with open(desc_path) as f:
            existing = json.load(f)

    for tid, d in descriptions.items():
        existing[str(tid)] = {
            "track_id": d.track_id,
            "jersey_number": d.jersey_number,
            "team_color": d.team_color,
            "description": d.description,
        }

    with open(desc_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"Saved {len(existing)} player descriptions to {desc_path}")

    # Summary
    print("\n=== Resolution Summary ===")
    with open(shots_path) as f:
        final_shots = json.load(f)
    made = [s for s in final_shots if s["outcome"] == "made"]
    with_jersey = [s for s in made if s.get("jersey_number") is not None]
    print(f"  Made shots: {len(made)}")
    print(f"  With jersey number: {len(with_jersey)} ({100*len(with_jersey)/len(made):.0f}%)")


if __name__ == "__main__":
    main()

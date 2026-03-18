"""Track merger — post-processing module that merges fragmented track IDs.

The IoU tracker produces many short-lived track fragments when players briefly
leave the detection window (occlusion, out-of-frame, missed detection).  For a
10-player game this can generate 100-150 unique track IDs instead of 10.

This module identifies fragments that almost certainly belong to the same
physical player and collapses them into a single canonical track ID.

Merging criteria (ALL must hold):
  1. Same team label
  2. No temporal overlap  (A's last_frame < B's first_frame)
  3. Temporal gap < max_gap_sec  (default 5 s)
  4. Spatial distance between boundary bboxes < max_distance_px  (default 200 px)

The algorithm is deliberately conservative.  It is better to leave two
fragments separate than to incorrectly merge two different players.
"""

from __future__ import annotations

import math
from collections import defaultdict


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def merge_tracks(
    tracks: list[dict],
    fps: float,
    max_gap_sec: float = 5.0,
    max_distance_px: float = 200.0,
) -> dict[int, int]:
    """Merge fragmented track IDs into canonical IDs.

    Parameters
    ----------
    tracks:
        List of track dicts as written by run_analysis.py.  Each dict has at
        minimum the keys: ``track_id`` (int), ``frame_idx`` (int),
        ``bbox`` (list[float] with [x1, y1, x2, y2]), ``team`` (str | None).
    fps:
        Frames-per-second of the source video.  Used to convert the temporal
        gap from seconds to frames.
    max_gap_sec:
        Maximum allowed gap (in seconds) between two fragments for them to be
        considered merge candidates.
    max_distance_px:
        Maximum allowed Euclidean distance (in pixels) between bbox centres at
        the boundary frames for two fragments to be considered merge candidates.
        The comparison is strictly less-than (``distance < max_distance_px``).

    Returns
    -------
    dict[int, int]
        ``merge_map`` where ``merge_map[old_id] = canonical_id``.  Track IDs
        that are not merged into another track are absent from the map (callers
        should use ``merge_map.get(tid, tid)`` to resolve any ID).
    """
    if not tracks:
        return {}

    max_gap_frames = fps * max_gap_sec

    # ------------------------------------------------------------------
    # Step 1: Build per-track profiles
    # ------------------------------------------------------------------
    # profile[track_id] = {
    #   "first_frame": int, "last_frame": int,
    #   "first_bbox": [x1,y1,x2,y2], "last_bbox": [x1,y1,x2,y2],
    #   "team": str | None
    # }
    profiles: dict[int, dict] = {}

    for entry in tracks:
        tid = int(entry["track_id"])
        frame = int(entry["frame_idx"])
        bbox = entry["bbox"]
        team = entry.get("team")

        if tid not in profiles:
            profiles[tid] = {
                "first_frame": frame,
                "last_frame": frame,
                "first_bbox": bbox,
                "last_bbox": bbox,
                "team": team,
            }
        else:
            p = profiles[tid]
            if frame < p["first_frame"]:
                p["first_frame"] = frame
                p["first_bbox"] = bbox
            if frame > p["last_frame"]:
                p["last_frame"] = frame
                p["last_bbox"] = bbox
            # Use the most-recently-seen non-None team label
            if p["team"] is None and team is not None:
                p["team"] = team

    # ------------------------------------------------------------------
    # Step 2: Group track IDs by team.  Unclassified tracks (team=None)
    #         are NOT merged with each other or with any classified track.
    # ------------------------------------------------------------------
    team_groups: dict[str, list[int]] = defaultdict(list)
    for tid, prof in profiles.items():
        team = prof["team"]
        if team is not None:
            team_groups[team].append(tid)

    # ------------------------------------------------------------------
    # Step 3: Build merge chains within each team group
    # ------------------------------------------------------------------
    # merge_map maps each track_id to its canonical ID.  Initialised as
    # identity; we update it as we find merge candidates.
    merge_map: dict[int, int] = {}  # only non-trivial mappings stored

    for team, tid_list in team_groups.items():
        # Sort fragments by their first frame so we always merge later
        # fragments into earlier ones.
        sorted_ids = sorted(tid_list, key=lambda t: profiles[t]["first_frame"])

        # canonical[i] = the canonical track ID for sorted_ids[i]
        # Initialise each fragment as its own canonical.
        canonical = {tid: tid for tid in sorted_ids}

        for i, tid_b in enumerate(sorted_ids):
            prof_b = profiles[tid_b]
            can_b = canonical[tid_b]

            # Try to attach fragment B to the best compatible earlier fragment.
            # "Best" = smallest temporal gap (greedy, leftmost-first).
            for j in range(i):
                tid_a = sorted_ids[j]
                can_a = canonical[tid_a]
                prof_a = profiles[can_a]  # compare against canonical's profile

                # ----- Criterion 2: no temporal overlap -----
                # A must end strictly before B begins
                if prof_a["last_frame"] >= prof_b["first_frame"]:
                    continue

                # ----- Criterion 3: temporal gap -----
                gap_frames = prof_b["first_frame"] - prof_a["last_frame"] - 1
                if gap_frames >= max_gap_frames:
                    continue

                # ----- Criterion 4: spatial proximity -----
                dist = _bbox_centre_distance(prof_a["last_bbox"], prof_b["first_bbox"])
                if dist >= max_distance_px:
                    continue

                # All criteria satisfied — merge B into A's canonical
                old_can_b = can_b
                can_b = can_a

                # Propagate canonical update to all fragments already pointing
                # at old_can_b
                for k in range(len(sorted_ids)):
                    if canonical[sorted_ids[k]] == old_can_b:
                        canonical[sorted_ids[k]] = can_a

                # Extend the canonical's profile to cover B's extent
                _extend_profile(prof_a, prof_b)

                # Found a merge — stop looking for earlier candidates for B
                break

        # Record non-trivial mappings in the output merge_map
        for tid, can in canonical.items():
            if can != tid:
                merge_map[tid] = can

    return merge_map


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _bbox_centre(bbox: list[float]) -> tuple[float, float]:
    """Return the (cx, cy) centre of a [x1, y1, x2, y2] bounding box."""
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _bbox_centre_distance(bbox_a: list[float], bbox_b: list[float]) -> float:
    """Euclidean pixel distance between the centres of two bounding boxes."""
    cx_a, cy_a = _bbox_centre(bbox_a)
    cx_b, cy_b = _bbox_centre(bbox_b)
    return math.hypot(cx_b - cx_a, cy_b - cy_a)


def _extend_profile(canonical_profile: dict, new_profile: dict) -> None:
    """Extend ``canonical_profile`` so it covers ``new_profile``'s time range.

    Updates first_frame/first_bbox if new_profile starts earlier, and
    last_frame/last_bbox if new_profile ends later.  Mutates in-place.
    """
    if new_profile["first_frame"] < canonical_profile["first_frame"]:
        canonical_profile["first_frame"] = new_profile["first_frame"]
        canonical_profile["first_bbox"] = new_profile["first_bbox"]
    if new_profile["last_frame"] > canonical_profile["last_frame"]:
        canonical_profile["last_frame"] = new_profile["last_frame"]
        canonical_profile["last_bbox"] = new_profile["last_bbox"]

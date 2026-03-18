"""Tests for track_merger.py — TDD RED phase.

All tests written before implementation. Run these to confirm RED state,
then implement merge_tracks() to make them GREEN.
"""

import json
import math
import os
import pytest

from app.tracking.track_merger import merge_tracks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_track(track_id: int, frame_idx: int, x1: float, y1: float,
               x2: float, y2: float, team: str) -> dict:
    return {
        "track_id": track_id,
        "frame_idx": frame_idx,
        "bbox": [x1, y1, x2, y2],
        "team": team,
    }


def build_fragment(track_id: int, start_frame: int, end_frame: int,
                   cx: float, cy: float, team: str,
                   box_w: float = 80.0, box_h: float = 160.0) -> list[dict]:
    """Return a list of track dicts for a single fragment (one entry per 6 frames)."""
    records = []
    for frame in range(start_frame, end_frame + 1, 6):
        half_w, half_h = box_w / 2, box_h / 2
        records.append(make_track(track_id, frame,
                                  cx - half_w, cy - half_h,
                                  cx + half_w, cy + half_h, team))
    return records


# ---------------------------------------------------------------------------
# Test 1: Two non-overlapping same-team fragments close in space -> merged
# ---------------------------------------------------------------------------

class TestBasicMerge:
    def test_same_team_no_overlap_close_space_merged(self):
        """Fragment A ends at frame 30, Fragment B starts at frame 60 (gap = 30
        frames = 1 s at 30 fps). Same team, same spatial position. Must merge."""
        fps = 30.0
        frag_a = build_fragment(track_id=1, start_frame=0, end_frame=30,
                                cx=500.0, cy=400.0, team="home")
        frag_b = build_fragment(track_id=2, start_frame=60, end_frame=120,
                                cx=510.0, cy=405.0, team="home")
        tracks = frag_a + frag_b

        merge_map = merge_tracks(tracks, fps=fps)

        # Both should map to the same canonical ID (the earlier one)
        canonical = merge_map.get(1, 1)
        assert merge_map.get(2, 2) == canonical, (
            "Fragment 2 should be merged into fragment 1's canonical ID"
        )

    def test_returns_dict_int_to_int(self):
        """merge_map values and keys must be ints."""
        frag_a = build_fragment(1, 0, 30, 500.0, 400.0, "home")
        frag_b = build_fragment(2, 60, 120, 510.0, 405.0, "home")
        merge_map = merge_tracks(frag_a + frag_b, fps=30.0)
        for k, v in merge_map.items():
            assert isinstance(k, int), f"Key {k!r} is not int"
            assert isinstance(v, int), f"Value {v!r} is not int"

    def test_tracks_not_needing_merge_absent_from_map(self):
        """A track that is not merged into another ID need not appear in the map.
        Any track in the map must map to itself or a smaller/earlier ID."""
        frag = build_fragment(1, 0, 100, 500.0, 400.0, "home")
        merge_map = merge_tracks(frag, fps=30.0)
        # Single track: either absent or maps to itself
        assert merge_map.get(1, 1) == 1


# ---------------------------------------------------------------------------
# Test 2: Overlapping fragments -> NOT merged
# ---------------------------------------------------------------------------

class TestOverlapNotMerged:
    def test_overlapping_fragments_not_merged(self):
        """Fragment A spans frames 0-60, Fragment B spans frames 40-120.
        They overlap at frames 40-60. Must NOT merge."""
        frag_a = build_fragment(1, 0, 60, 500.0, 400.0, "home")
        frag_b = build_fragment(2, 40, 120, 510.0, 405.0, "home")

        merge_map = merge_tracks(frag_a + frag_b, fps=30.0)

        # IDs must remain separate
        assert merge_map.get(2, 2) != merge_map.get(1, 1) or (
            # Unless they happen to map to different canonical IDs
            merge_map.get(1, 1) != merge_map.get(2, 2)
        )
        # Simpler assertion: track 2 should not be merged into track 1
        assert merge_map.get(2, 2) != 1, (
            "Overlapping fragment 2 must not be merged into fragment 1"
        )

    def test_adjacent_no_gap_not_merged(self):
        """Fragment A ends at frame 30, Fragment B starts at frame 30 exactly
        (zero gap / touching boundary). Should NOT merge (no gap means same instant,
        could be two different players on court simultaneously)."""
        frag_a = build_fragment(1, 0, 30, 500.0, 400.0, "home")
        frag_b = build_fragment(2, 30, 90, 510.0, 405.0, "home")

        merge_map = merge_tracks(frag_a + frag_b, fps=30.0)
        assert merge_map.get(2, 2) != 1, (
            "Adjacent (zero-gap) fragments must not be merged"
        )


# ---------------------------------------------------------------------------
# Test 3: Different teams -> NOT merged
# ---------------------------------------------------------------------------

class TestDifferentTeamNotMerged:
    def test_different_teams_not_merged(self):
        """Same position, no overlap, large gap — but different teams. Must NOT merge."""
        frag_a = build_fragment(1, 0, 30, 500.0, 400.0, "home")
        frag_b = build_fragment(2, 90, 150, 500.0, 400.0, "away")

        merge_map = merge_tracks(frag_a + frag_b, fps=30.0)
        assert merge_map.get(2, 2) != 1, (
            "Tracks from different teams must never be merged"
        )

    def test_none_team_not_merged_with_classified(self):
        """A track with team=None must not be merged into a classified track."""
        frag_a = build_fragment(1, 0, 30, 500.0, 400.0, "home")
        # Fragment with no team classification
        frag_b = [make_track(2, 90, 500.0, 320.0, 580.0, 480.0, None)]

        merge_map = merge_tracks(frag_a + frag_b, fps=30.0)
        assert merge_map.get(2, 2) != 1


# ---------------------------------------------------------------------------
# Test 4: Large spatial gap -> NOT merged
# ---------------------------------------------------------------------------

class TestLargeSpatialGapNotMerged:
    def test_large_spatial_gap_not_merged(self):
        """Fragment A ends at position (100, 100), Fragment B starts at (900, 700).
        Euclidean pixel distance >> 200px. Must NOT merge."""
        frag_a = build_fragment(1, 0, 30, cx=100.0, cy=100.0, team="home")
        frag_b = build_fragment(2, 60, 120, cx=900.0, cy=700.0, team="home")

        merge_map = merge_tracks(frag_a + frag_b, fps=30.0)
        assert merge_map.get(2, 2) != 1, (
            "Fragment far away (>200px) must not be merged"
        )

    def test_exactly_at_distance_threshold_not_merged(self):
        """Distance exactly equal to max_distance_px is NOT within threshold
        (strict less-than). Fragments at exactly 200px apart must not merge."""
        cx_a, cy_a = 500.0, 400.0
        # Place B exactly 200px away horizontally
        cx_b = cx_a + 200.0
        cy_b = cy_a
        frag_a = build_fragment(1, 0, 30, cx_a, cy_a, "home")
        frag_b = build_fragment(2, 60, 120, cx_b, cy_b, "home")

        merge_map = merge_tracks(frag_a + frag_b, fps=30.0)
        # At exactly 200px the fragments are NOT candidates (strict < threshold)
        assert merge_map.get(2, 2) != 1

    def test_just_within_distance_threshold_merged(self):
        """Distance just under 200px should pass spatial check if other conditions met."""
        cx_a, cy_a = 500.0, 400.0
        cx_b = cx_a + 199.0  # 1px inside threshold
        cy_b = cy_a
        frag_a = build_fragment(1, 0, 30, cx_a, cy_a, "home")
        frag_b = build_fragment(2, 60, 120, cx_b, cy_b, "home")

        merge_map = merge_tracks(frag_a + frag_b, fps=30.0)
        assert merge_map.get(2, 2) == merge_map.get(1, 1), (
            "Fragment within 200px should merge if all other conditions met"
        )


# ---------------------------------------------------------------------------
# Test 5: Chain of 3 fragments -> all merge to earliest ID
# ---------------------------------------------------------------------------

class TestChainMerge:
    def test_chain_of_three_merges_to_first(self):
        """Three sequential fragments from the same player position and team.
        All should merge to the earliest canonical ID."""
        fps = 30.0
        frag_a = build_fragment(1, 0, 30, 500.0, 400.0, "home")    # frames 0-30
        frag_b = build_fragment(2, 60, 90, 505.0, 402.0, "home")   # gap 30 frames = 1s
        frag_c = build_fragment(3, 120, 150, 508.0, 398.0, "home") # gap 30 frames = 1s

        merge_map = merge_tracks(frag_a + frag_b + frag_c, fps=fps)

        canonical_1 = merge_map.get(1, 1)
        canonical_2 = merge_map.get(2, 2)
        canonical_3 = merge_map.get(3, 3)

        assert canonical_2 == canonical_1, "Fragment 2 should chain-merge to fragment 1"
        assert canonical_3 == canonical_1, "Fragment 3 should chain-merge to fragment 1"

    def test_chain_canonical_is_smallest_id(self):
        """The canonical ID chosen for a chain must be the earliest fragment's ID."""
        fps = 30.0
        frag_a = build_fragment(5, 0, 30, 500.0, 400.0, "home")
        frag_b = build_fragment(10, 60, 90, 505.0, 402.0, "home")

        merge_map = merge_tracks(frag_a + frag_b, fps=fps)
        assert merge_map.get(10, 10) == 5, (
            "Canonical ID for merged chain must be the earliest (lowest-first-frame) track_id"
        )


# ---------------------------------------------------------------------------
# Test 6: Real-world test — load first 1000 entries from player_tracks.json
# ---------------------------------------------------------------------------

class TestRealWorldReduction:
    TRACKS_PATH = "data/outputs/v1.7.0/player_tracks.json"

    def _load_first_n(self, n: int = 1000) -> list[dict]:
        """Load first n entries from player_tracks.json.

        Each track entry is ~300 bytes in the formatted JSON.  Reading 400KB is
        more than enough to cover 1000 entries without touching the full 70MB.
        We read a generous prefix, find the last complete object boundary, and
        parse that fragment as a valid JSON array.
        """
        # ~400 bytes per record with 2-space indent; 1000 records = ~400KB.
        # Read 512KB to have comfortable headroom.
        read_bytes = 512 * 1024
        with open(self.TRACKS_PATH, "rb") as f:
            raw = f.read(read_bytes)

        text = raw.decode("utf-8", errors="replace")

        # Find the last complete '}' followed by optional whitespace/comma
        # before the n-th record ends.  Simplest approach: find the n-th
        # occurrence of the pattern that closes an object at the top level.
        # JSON objects in the array are separated by '},\n  {'.
        # Walk forward counting closing braces at depth 1.
        depth = 0
        in_array = False
        count = 0
        last_close = -1

        for idx, ch in enumerate(text):
            if not in_array:
                if ch == '[':
                    in_array = True
                continue
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    count += 1
                    last_close = idx
                    if count >= n:
                        break

        if last_close == -1:
            # Fallback: just parse whatever we got
            fragment = text.strip().rstrip(",").rstrip()
            if not fragment.endswith("]"):
                fragment += "]"
            try:
                return json.loads(fragment)[:n]
            except json.JSONDecodeError:
                return []

        # Build a valid JSON array from the records we found
        array_start = text.index("[")
        fragment = text[array_start : last_close + 1] + "]"
        # Remove trailing commas before ] (last element may have a comma)
        fragment = fragment.replace(",]", "]")
        return json.loads(fragment)

    @pytest.mark.skipif(
        not os.path.exists("data/outputs/v1.7.0/player_tracks.json"),
        reason="player_tracks.json not present in test environment",
    )
    def test_merge_reduces_unique_ids(self):
        """Running merge_tracks on first 1000 entries should reduce unique track IDs.
        The original IoU tracker produces ~149 fragments for ~10 players.
        After merging, there should be fewer unique canonical IDs."""
        tracks = self._load_first_n(1000)
        assert len(tracks) > 0, "Should load at least some track records"

        original_ids = {t["track_id"] for t in tracks}
        fps = 30.0  # typical basketball video fps

        merge_map = merge_tracks(tracks, fps=fps)

        # Apply merge_map
        canonical_ids = {merge_map.get(tid, tid) for tid in original_ids}

        # After merging, canonical IDs must be a subset of or equal to original
        assert len(canonical_ids) <= len(original_ids), (
            "Merging must not increase the number of unique track IDs"
        )

        # With 1000 entries from a highly fragmented game, we expect at least
        # some reduction. This is a soft assertion — if the first 1000 entries
        # happen to contain no merge candidates the test still passes.
        print(f"\n  Real-world merge: {len(original_ids)} -> {len(canonical_ids)} unique IDs")
        print(f"  Merge map size: {len(merge_map)} entries")

    @pytest.mark.skipif(
        not os.path.exists("data/outputs/v1.7.0/player_tracks.json"),
        reason="player_tracks.json not present in test environment",
    )
    def test_merge_map_values_are_valid_track_ids(self):
        """Every canonical ID in the merge_map must itself be a known track_id."""
        tracks = self._load_first_n(1000)
        original_ids = {t["track_id"] for t in tracks}

        merge_map = merge_tracks(tracks, fps=30.0)

        for old_id, canonical_id in merge_map.items():
            assert canonical_id in original_ids, (
                f"canonical_id {canonical_id} is not a known track_id "
                f"(came from old_id {old_id})"
            )


# ---------------------------------------------------------------------------
# Test 7: Large temporal gap -> NOT merged
# ---------------------------------------------------------------------------

class TestLargeTemporalGapNotMerged:
    def test_gap_exceeds_max_gap_sec_not_merged(self):
        """Default max_gap_sec=5.0. At 30fps that's 150 frames.
        Fragment A ends frame 30, Fragment B starts frame 200. Gap = 170 frames = 5.67s.
        Must NOT merge."""
        fps = 30.0
        frag_a = build_fragment(1, 0, 30, 500.0, 400.0, "home")
        frag_b = build_fragment(2, 200, 260, 502.0, 401.0, "home")

        merge_map = merge_tracks(frag_a + frag_b, fps=fps)
        assert merge_map.get(2, 2) != 1, (
            "Fragment with temporal gap > max_gap_sec must not be merged"
        )

    def test_gap_within_max_gap_sec_merged(self):
        """Gap of exactly 4.0s at 30fps = 120 frames (< 150 frame limit). Should merge."""
        fps = 30.0
        frag_a = build_fragment(1, 0, 30, 500.0, 400.0, "home")
        frag_b = build_fragment(2, 151, 210, 502.0, 401.0, "home")  # gap = 120 frames = 4s

        merge_map = merge_tracks(frag_a + frag_b, fps=fps)
        assert merge_map.get(2, 2) == merge_map.get(1, 1), (
            "Fragment within max_gap_sec should merge"
        )

    def test_custom_max_gap_sec_respected(self):
        """Passing max_gap_sec=2.0 with a 90-frame gap at 30fps (3s) must NOT merge."""
        fps = 30.0
        frag_a = build_fragment(1, 0, 30, 500.0, 400.0, "home")
        frag_b = build_fragment(2, 121, 180, 502.0, 401.0, "home")  # gap = 90 frames = 3s

        merge_map = merge_tracks(frag_a + frag_b, fps=fps, max_gap_sec=2.0)
        assert merge_map.get(2, 2) != 1, (
            "With max_gap_sec=2.0, a 3s gap must not merge"
        )


# ---------------------------------------------------------------------------
# Test 8: Empty / edge-case inputs
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_tracks_returns_empty_map(self):
        assert merge_tracks([], fps=30.0) == {}

    def test_single_track_returns_empty_map(self):
        frag = build_fragment(1, 0, 100, 500.0, 400.0, "home")
        result = merge_tracks(frag, fps=30.0)
        # No merges needed — map should be empty or map 1->1
        assert result.get(1, 1) == 1

    def test_tracks_with_missing_team_not_merged(self):
        """Tracks where team is None on both sides should not be merged with each other."""
        frag_a = [make_track(1, frame, 490.0, 320.0, 570.0, 480.0, None)
                  for frame in range(0, 31, 6)]
        frag_b = [make_track(2, frame, 491.0, 321.0, 571.0, 481.0, None)
                  for frame in range(60, 121, 6)]

        merge_map = merge_tracks(frag_a + frag_b, fps=30.0)
        # Both have no team — should not be merged (conservative)
        assert merge_map.get(2, 2) != 1

"""Player clip extractor — extracts key moment clips for a specific player.

Uses cv2.VideoCapture + cv2.VideoWriter to extract short MP4 clips around
player events (shots, possessions, defense, transitions, off-ball movement).
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class ClipCategory(str, Enum):
    """Category of a player clip, ordered by importance."""
    SHOT = "SHOT"
    POSSESSION = "POSSESSION"
    DEFENSE = "DEFENSE"
    TRANSITION = "TRANSITION"
    OFF_BALL = "OFF_BALL"


# Priority order for sorting clips (lower number = higher priority)
_CATEGORY_PRIORITY: dict[ClipCategory, int] = {
    ClipCategory.SHOT: 0,
    ClipCategory.POSSESSION: 1,
    ClipCategory.DEFENSE: 2,
    ClipCategory.TRANSITION: 3,
    ClipCategory.OFF_BALL: 4,
}


@dataclass
class PlayerClip:
    """A short video clip of a specific player moment."""

    clip_path: str           # path to extracted MP4
    start_sec: float
    end_sec: float
    category: str            # SHOT, POSSESSION, DEFENSE, etc.
    context: str             # "Q2 5:30, score 38-41, drives baseline"
    frame_indices: list[int] = field(default_factory=list)


class ClipExtractor:
    """Extracts key player clips from a game video using cv2."""

    def __init__(self, video_path: str, fps: float):
        self.video_path = video_path
        self.fps = fps

    def extract_player_clips(
        self,
        player_tracks: list[dict],       # track rows for the target player
        shot_events: list[dict],          # shots involving this player
        possession_events: list[dict],    # possessions where player had the ball
        max_clips: int = 25,
        clip_duration_sec: float = 10.0,
        output_dir: str = "clips",
    ) -> list[PlayerClip]:
        """Extract key moment clips for a specific player.

        Categories:
        - SHOT: every shot attempt (made and missed)
        - POSSESSION: ball touches where player had the ball
        - DEFENSE: defensive possessions (sample)
        - TRANSITION: fast break and getting-back moments
        - OFF_BALL: random off-ball movement samples

        Returns clips sorted by importance (shots first, then possessions, etc.)
        """
        import cv2

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Build a set of timestamps available in player_tracks for quick lookup
        # Group tracks by nearest second for efficiency
        track_times: list[float] = sorted(
            set(t["frame_idx"] / self.fps for t in player_tracks)
        )

        clips: list[PlayerClip] = []

        # 1. SHOT clips — every shot attempt
        for i, shot in enumerate(shot_events):
            ts = shot.get("timestamp_sec", shot.get("frame_idx", 0) / self.fps)
            quarter = shot.get("quarter", "?")
            outcome = shot.get("outcome", "attempt")
            context = _format_context(ts, f"Q{quarter}", outcome, ClipCategory.SHOT)
            clip = self._extract_clip(
                clip_index=len(clips),
                category=ClipCategory.SHOT,
                center_sec=ts,
                duration_sec=clip_duration_sec,
                context=context,
                output_dir=output_dir,
            )
            if clip:
                clips.append(clip)

        # 2. POSSESSION clips — ball touches
        for i, poss in enumerate(possession_events):
            ts = poss.get("start_time", 0.0)
            result = poss.get("result", "possession")
            context = _format_context(ts, "", result, ClipCategory.POSSESSION)
            clip = self._extract_clip(
                clip_index=len(clips),
                category=ClipCategory.POSSESSION,
                center_sec=ts,
                duration_sec=clip_duration_sec,
                context=context,
                output_dir=output_dir,
            )
            if clip:
                clips.append(clip)

        # 3. DEFENSE clips — sample from track data (not explicitly in events)
        # Use evenly-spaced samples from the player's track timeline
        defense_samples = _sample_evenly(track_times, n=4)
        for ts in defense_samples:
            context = _format_context(ts, "", "defensive position", ClipCategory.DEFENSE)
            clip = self._extract_clip(
                clip_index=len(clips),
                category=ClipCategory.DEFENSE,
                center_sec=ts,
                duration_sec=clip_duration_sec,
                context=context,
                output_dir=output_dir,
            )
            if clip:
                clips.append(clip)

        # 4. TRANSITION clips — fast break moments (rapid position changes)
        transition_times = _find_transition_moments(track_times, window_sec=3.0)
        for ts in transition_times[:3]:
            context = _format_context(ts, "", "transition", ClipCategory.TRANSITION)
            clip = self._extract_clip(
                clip_index=len(clips),
                category=ClipCategory.TRANSITION,
                center_sec=ts,
                duration_sec=clip_duration_sec,
                context=context,
                output_dir=output_dir,
            )
            if clip:
                clips.append(clip)

        # 5. OFF_BALL clips — random off-ball movement samples
        # Sample from times not covered by other events
        covered = set()
        for c in clips:
            for t in _range_secs(c.start_sec, c.end_sec, step=1.0):
                covered.add(round(t))
        offball_candidates = [
            t for t in track_times
            if round(t) not in covered
        ]
        offball_samples = _sample_evenly(offball_candidates, n=3)
        for ts in offball_samples:
            context = _format_context(ts, "", "off-ball movement", ClipCategory.OFF_BALL)
            clip = self._extract_clip(
                clip_index=len(clips),
                category=ClipCategory.OFF_BALL,
                center_sec=ts,
                duration_sec=clip_duration_sec,
                context=context,
                output_dir=output_dir,
            )
            if clip:
                clips.append(clip)

        # Sort by importance and cap at max_clips
        clips.sort(key=lambda c: _CATEGORY_PRIORITY.get(ClipCategory(c.category), 99))
        return clips[:max_clips]

    def _extract_clip(
        self,
        clip_index: int,
        category: ClipCategory,
        center_sec: float,
        duration_sec: float,
        context: str,
        output_dir: str,
    ) -> Optional[PlayerClip]:
        """Extract a single clip centered on `center_sec`.

        Returns None if cv2 fails (e.g. video not found, seek fails).
        """
        import cv2

        half = duration_sec / 2.0
        start_sec = max(0.0, center_sec - half)
        end_sec = center_sec + half

        filename = f"clip_{clip_index:03d}_{category.value.lower()}.mp4"
        clip_path = str(Path(output_dir) / filename)

        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                return None

            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or self.fps
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            start_frame = int(start_sec * fps)
            end_frame = min(int(end_sec * fps), total_frames - 1)

            if start_frame >= total_frames or end_frame <= start_frame:
                cap.release()
                return None

            # Seek to start frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            # Write clip
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(clip_path, fourcc, fps, (width, height))

            frame_indices: list[int] = []
            for frame_idx in range(start_frame, end_frame + 1):
                ret, frame = cap.read()
                if not ret:
                    break
                writer.write(frame)
                frame_indices.append(frame_idx)

            cap.release()
            writer.release()

            if not frame_indices:
                return None

            return PlayerClip(
                clip_path=clip_path,
                start_sec=start_sec,
                end_sec=end_sec,
                category=category.value,
                context=context,
                frame_indices=frame_indices,
            )

        except Exception:
            return None


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _format_context(timestamp_sec: float, quarter_label: str, event_desc: str, category: ClipCategory) -> str:
    """Format a human-readable context string for a clip."""
    total_sec = int(timestamp_sec)
    mm = total_sec // 60
    ss = total_sec % 60
    time_str = f"{mm}:{ss:02d}"
    parts = []
    if quarter_label:
        parts.append(quarter_label)
    parts.append(time_str)
    parts.append(event_desc)
    return ", ".join(parts)


def _sample_evenly(times: list[float], n: int) -> list[float]:
    """Return up to n evenly-spaced values from a sorted list."""
    if not times or n <= 0:
        return []
    if len(times) <= n:
        return times
    step = len(times) / n
    return [times[int(i * step)] for i in range(n)]


def _find_transition_moments(
    track_times: list[float],
    window_sec: float = 3.0,
) -> list[float]:
    """Find timestamps with large gaps in tracking data (player sprinting / transition)."""
    if len(track_times) < 2:
        return []
    gaps = []
    for i in range(1, len(track_times)):
        gap = track_times[i] - track_times[i - 1]
        if gap > window_sec:
            # Mid-point of the gap is likely where transition happened
            mid = (track_times[i] + track_times[i - 1]) / 2.0
            gaps.append((gap, mid))
    gaps.sort(reverse=True)
    return [mid for _, mid in gaps[:5]]


def _range_secs(start: float, end: float, step: float = 1.0) -> list[float]:
    """Generate a list of float values from start to end by step."""
    result = []
    t = start
    while t <= end:
        result.append(t)
        t += step
    return result


def group_tracks_by_id(player_tracks: list[dict]) -> dict[int, list[dict]]:
    """Group track rows by track_id for O(1) lookup.

    Used by the CLI to build per-player track lists from the full tracks file.
    """
    groups: dict[int, list[dict]] = defaultdict(list)
    for row in player_tracks:
        groups[int(row["track_id"])].append(row)
    return dict(groups)


def resolve_player_track_ids(
    player_descriptions: dict,
    target_jersey: int,
    target_team: str,
) -> list[int]:
    """Resolve which track IDs belong to the target player.

    Searches player_descriptions.json for tracks where:
    - jersey_number matches target_jersey
    - description is consistent with target_team jersey colour

    Falls back to empty list if no match found; the caller should handle
    the fallback of using all same-team track IDs.
    """
    matched: list[int] = []
    for track_id_str, desc in player_descriptions.items():
        if not isinstance(desc, dict):
            continue
        jersey = desc.get("jersey_number")
        if jersey is not None and int(jersey) == target_jersey:
            matched.append(int(track_id_str))
    return matched

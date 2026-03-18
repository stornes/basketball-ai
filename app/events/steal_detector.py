"""Steal detection heuristic — cross-team possession flip without shot.

When possession changes from Team A to Team B without an intervening shot
event and within a short time window, the Team B player gets a steal.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.events.event_types import PossessionEvent


@dataclass
class StealEvent:
    """A detected steal."""

    frame_idx: int
    timestamp_sec: float
    stealer_track_id: int
    stealer_team: str | None
    victim_track_id: int
    victim_team: str | None


class StealDetector:
    """Detects steals from rapid cross-team possession changes.

    Algorithm:
    1. Track consecutive possession events
    2. When possession flips from Team A player to Team B player:
       - No shot event between the two possessions
       - Time gap < STEAL_WINDOW_SEC
       → Team B player gets the steal
    """

    # Max seconds between end of Team A possession and start of Team B possession
    STEAL_WINDOW_SEC = 2.0

    def __init__(self, fps: float):
        self.fps = fps
        self.events: list[StealEvent] = []

    def check(
        self,
        possession_events: list[PossessionEvent],
    ) -> list[StealEvent]:
        """Scan all possession events for steal patterns.

        Returns list of all newly detected steal events.
        Call this once after all possessions are collected, or
        incrementally with the full list (it deduplicates via set tracking).
        """
        new_events: list[StealEvent] = []
        seen_frames: set[int] = {e.frame_idx for e in self.events}

        for i in range(1, len(possession_events)):
            prev = possession_events[i - 1]
            curr = possession_events[i]

            # Skip if previous possession ended with a shot
            if prev.result == "shot":
                continue

            # Must be different teams
            if not prev.team or not curr.team:
                continue
            if prev.team == curr.team:
                continue

            # Time gap must be small (rapid possession change)
            gap = curr.start_time - prev.end_time
            if gap > self.STEAL_WINDOW_SEC:
                continue
            # Negative gap means overlap — still valid (simultaneous tracking)
            if gap < -1.0:
                continue

            # Avoid duplicates
            if curr.start_frame in seen_frames:
                continue

            event = StealEvent(
                frame_idx=curr.start_frame,
                timestamp_sec=curr.start_time,
                stealer_track_id=curr.player_track_id,
                stealer_team=curr.team,
                victim_track_id=prev.player_track_id,
                victim_team=prev.team,
            )
            self.events.append(event)
            new_events.append(event)
            seen_frames.add(curr.start_frame)

        return new_events

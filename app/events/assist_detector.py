"""Assist detection — observational pass-based with proximity fallback.

v1.7.0: Two-tier assist detection:
1. **Preferred (observational):** Use PassEvents — a pass from Player A to
   Player B where B scores within ASSIST_WINDOW_SEC is a true observed assist.
2. **Fallback (heuristic):** When no PassEvents available, fall back to
   proximity-based possession matching (same-team possession by different
   player within time window before made shot).
"""

from __future__ import annotations

from dataclasses import dataclass

from app.events.event_types import PassEvent, PossessionEvent, ShotEvent, ShotOutcome


@dataclass
class AssistEvent:
    """A detected assist."""

    frame_idx: int
    timestamp_sec: float
    assister_track_id: int
    assister_team: str | None
    scorer_track_id: int | None
    scorer_team: str | None
    shot_frame_idx: int
    source: str = "proximity"  # "pass" (observational) or "proximity" (heuristic)


class AssistDetector:
    """Detects assists from pass events or possession proximity.

    Algorithm (two-tier):
    Tier 1 — Pass-based (observational, preferred):
      1. Search PassEvents for a pass TO the scorer within ASSIST_WINDOW_SEC
      2. Pass must be from a same-team player (A ≠ B)
      3. The pass observation IS the assist — no inference needed

    Tier 2 — Proximity-based (heuristic, fallback):
      1. On made shot by Player B, look back through recent possessions
      2. Find most recent same-team possession by Player A (A != B)
      3. If possession ended within ASSIST_WINDOW_SEC before the shot → assist
    """

    # Max seconds before shot to look for assist
    ASSIST_WINDOW_SEC = 6.0

    def __init__(self, fps: float):
        self.fps = fps
        self.events: list[AssistEvent] = []

    def check(
        self,
        shot: ShotEvent,
        possession_events: list[PossessionEvent],
        pass_events: list[PassEvent] | None = None,
    ) -> AssistEvent | None:
        """Check if a made shot has an assist.

        Tries pass-based detection first (observational), then falls back
        to proximity-based detection (heuristic).

        Args:
            shot: The shot event (must be MADE for assist attribution).
            possession_events: All possession events so far.
            pass_events: Optional list of observed pass events. If provided
                and a matching pass is found, the assist is observational.

        Returns:
            AssistEvent if an assist is detected, None otherwise.
        """
        if shot.outcome != ShotOutcome.MADE:
            return None

        if shot.shooter_track_id is None:
            return None

        shooter_team = shot.team
        if not shooter_team:
            return None

        # Tier 1: Try pass-based assist detection (observational)
        if pass_events:
            assist = self._check_pass_based(shot, pass_events, shooter_team)
            if assist:
                return assist

        # Tier 2: Fall back to proximity-based detection (heuristic)
        return self._check_proximity_based(shot, possession_events, shooter_team)

    def _check_pass_based(
        self,
        shot: ShotEvent,
        pass_events: list[PassEvent],
        shooter_team: str,
    ) -> AssistEvent | None:
        """Tier 1: Check for assist via observed pass to scorer.

        Searches for a pass TO the shooter that arrived within
        ASSIST_WINDOW_SEC before the shot. This is a direct observation —
        the camera saw the pass happen.
        """
        shot_time = shot.timestamp_sec
        window_start = shot_time - self.ASSIST_WINDOW_SEC

        # Search backwards for most recent pass TO the shooter
        for pass_evt in reversed(pass_events):
            # Too old
            if pass_evt.timestamp_sec < window_start:
                break

            # Too recent (after the shot)
            if pass_evt.timestamp_sec > shot_time:
                continue

            # Pass must be TO the shooter
            if pass_evt.to_player_track_id != shot.shooter_track_id:
                continue

            # Passer must be on the same team (if team info available)
            if pass_evt.from_team and pass_evt.from_team != shooter_team:
                continue

            # Passer must be different from shooter
            if pass_evt.from_player_track_id == shot.shooter_track_id:
                continue

            # Found an observed assist
            event = AssistEvent(
                frame_idx=pass_evt.frame_idx,
                timestamp_sec=pass_evt.timestamp_sec,
                assister_track_id=pass_evt.from_player_track_id,
                assister_team=pass_evt.from_team,
                scorer_track_id=shot.shooter_track_id,
                scorer_team=shooter_team,
                shot_frame_idx=shot.frame_idx,
                source="pass",
            )
            self.events.append(event)
            return event

        return None

    def _check_proximity_based(
        self,
        shot: ShotEvent,
        possession_events: list[PossessionEvent],
        shooter_team: str,
    ) -> AssistEvent | None:
        """Tier 2: Check for assist via possession proximity (heuristic).

        Falls back to this when no PassEvents are available or no matching
        pass is found. Same algorithm as v1.6.x.
        """
        shot_time = shot.timestamp_sec
        window_start = shot_time - self.ASSIST_WINDOW_SEC

        # Search backwards for most recent same-team possession by different player
        for poss in reversed(possession_events):
            # Skip possessions that ended the current shot (result == "shot")
            if poss.result == "shot" and poss.player_track_id == shot.shooter_track_id:
                continue

            # Too old
            if poss.end_time < window_start:
                break

            # Too recent (after the shot)
            if poss.start_time > shot_time:
                continue

            # Must be same team
            if poss.team != shooter_team:
                continue

            # Must be different player
            if poss.player_track_id == shot.shooter_track_id:
                continue

            # Found an assist (heuristic)
            event = AssistEvent(
                frame_idx=poss.end_frame,
                timestamp_sec=poss.end_time,
                assister_track_id=poss.player_track_id,
                assister_team=poss.team,
                scorer_track_id=shot.shooter_track_id,
                scorer_team=shooter_team,
                shot_frame_idx=shot.frame_idx,
                source="proximity",
            )
            self.events.append(event)
            return event

        return None

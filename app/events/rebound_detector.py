"""Rebound detection heuristic — ORB/DRB from post-miss ball proximity.

After a missed shot, monitors the next few seconds for the first player
to gain ball proximity. If that player's team matches the shooter's team,
it's an offensive rebound (ORB); otherwise defensive (DRB).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from app.events.event_types import ShotEvent, ShotOutcome
from app.events.possession_state import BallState
from app.events.spatial import court_distance
from app.tracking.tracker import TrackedPlayer
from app.vision.detection_types import Detection


class ReboundType(Enum):
    OFFENSIVE = "offensive"
    DEFENSIVE = "defensive"


@dataclass
class ReboundEvent:
    """A detected rebound."""

    frame_idx: int
    timestamp_sec: float
    rebounder_track_id: int
    rebounder_team: str | None
    shooter_track_id: int | None
    shooter_team: str | None
    rebound_type: ReboundType
    shot_frame_idx: int  # the missed shot that triggered this


class ReboundDetector:
    """Detects rebounds by tracking ball proximity after missed shots.

    Algorithm:
    1. On MISSED shot, start monitoring window (default 4 seconds)
    2. Each frame, find closest player to ball
    3. First player within proximity threshold gets the rebound
    4. Compare rebounder team vs shooter team → ORB or DRB
    5. Timeout after window expires → no rebound attributed
    """

    # Max distance (pixels) between ball center and player center for rebound
    REBOUND_PROXIMITY_PX = 100
    # Seconds after miss to look for rebound
    REBOUND_WINDOW_SEC = 4.0
    # Minimum frames after miss before checking (ball still in air)
    MIN_DELAY_FRAMES = 5

    def __init__(
        self,
        fps: float,
        homography: np.ndarray | None = None,
        rebound_proximity_ft: float | None = None,
    ) -> None:
        """Initialise the rebound detector.

        Args:
            fps: Frames per second of the video.
            homography: Optional 3x3 perspective transform from CourtMapper.H.
                When provided, proximity is evaluated in court feet.
            rebound_proximity_ft: Court-plane distance (feet) for rebound
                detection. Only used when homography is supplied.
                Typical value: 4-6 feet. Falls back to REBOUND_PROXIMITY_PX
                when None or when no homography is available.
        """
        self.fps = fps
        self.homography = homography
        self.rebound_proximity_ft = rebound_proximity_ft
        self._pending_miss: ShotEvent | None = None
        self._frames_since_miss = 0
        self._max_frames = int(self.REBOUND_WINDOW_SEC * fps)
        self.events: list[ReboundEvent] = []

    def on_missed_shot(
        self,
        shot: ShotEvent,
        ball_state: BallState | None = None,
    ) -> None:
        """Register a missed shot to start rebound detection.

        Args:
            shot: The shot event. Only MISSED/ATTEMPTED outcomes are registered.
            ball_state: Optional current BallState from PossessionStateMachine.
                When provided, the shot is only registered if ball_state is FLIGHT
                (the ball was visibly airborne). This prevents false rebound
                detection on made shots where ball_state would be LOOSE_BALL.
                When None (default), the original behaviour applies — register
                based on shot outcome alone.
        """
        if shot.outcome not in (ShotOutcome.MISSED, ShotOutcome.ATTEMPTED):
            return

        # If state machine gating is active, require the ball to be in FLIGHT
        if ball_state is not None and ball_state != BallState.FLIGHT:
            return

        self._pending_miss = shot
        self._frames_since_miss = 0

    def update(
        self,
        ball: Detection | None,
        players: list[TrackedPlayer],
        frame_idx: int,
    ) -> ReboundEvent | None:
        """Process one frame. Returns ReboundEvent if rebound detected."""
        if self._pending_miss is None:
            return None

        self._frames_since_miss += 1

        # Timeout — no rebound attributed
        if self._frames_since_miss > self._max_frames:
            self._pending_miss = None
            return None

        # Wait for ball to come down
        if self._frames_since_miss < self.MIN_DELAY_FRAMES:
            return None

        if ball is None or not players:
            return None

        bx, by = ball.bbox.center

        # Determine whether to use court-plane projection
        use_court = (
            self.homography is not None
            and self.rebound_proximity_ft is not None
        )
        active_threshold = (
            self.rebound_proximity_ft if use_court else self.REBOUND_PROXIMITY_PX
        )

        # Find closest player to ball
        closest = None
        min_dist = float("inf")
        for player in players:
            px, py = player.bbox.center
            dist = court_distance(
                (bx, by), (px, py),
                homography=self.homography if use_court else None,
            )
            if dist < min_dist:
                min_dist = dist
                closest = player

        if closest is None or min_dist > active_threshold:
            return None

        # Rebound detected
        miss = self._pending_miss
        shooter_team = miss.team
        rebounder_team = closest.team

        if shooter_team and rebounder_team:
            reb_type = (
                ReboundType.OFFENSIVE
                if rebounder_team == shooter_team
                else ReboundType.DEFENSIVE
            )
        else:
            # Can't determine type without team info — default to defensive
            reb_type = ReboundType.DEFENSIVE

        event = ReboundEvent(
            frame_idx=frame_idx,
            timestamp_sec=frame_idx / self.fps if self.fps > 0 else 0.0,
            rebounder_track_id=closest.track_id,
            rebounder_team=rebounder_team,
            shooter_track_id=miss.shooter_track_id,
            shooter_team=shooter_team,
            rebound_type=reb_type,
            shot_frame_idx=miss.frame_idx,
        )
        self.events.append(event)
        self._pending_miss = None
        return event

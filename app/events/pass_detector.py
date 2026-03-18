"""Pass trajectory detection — observational pass events.

Detects passes by tracking ball movement between two player possessions.
A pass is observed when:
1. Ball leaves Player A's upper-body bbox (possession end)
2. Ball travels through open space (not in any player's bbox)
3. Ball enters Player B's upper-body bbox (new possession)
4. The transit time is < 2 seconds (real pass, not loose ball)
5. The ball travel distance > 5% of frame width (meaningful movement)

This is true observational data — the camera saw the ball move from
one player to another.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from app.events.ball_possession import BallInHandsDetector, PossessionTransition
from app.events.event_types import PassEvent
from app.vision.detection_types import Detection


@dataclass
class _PendingPass:
    """Tracks a potential pass in progress (ball left a player, in transit)."""

    from_player_id: int
    release_frame: int
    release_time: float
    release_x: float
    release_y: float
    trajectory: list[tuple[float, float, int]] = field(default_factory=list)


class PassDetector:
    """Detects passes from ball-in-hands transitions and ball trajectory.

    Algorithm:
    1. When BallInHandsDetector reports ball leaving Player A → start tracking
    2. Record ball centroid path each frame during transit
    3. When ball enters Player B → check distance and time constraints
    4. If valid → emit PassEvent with trajectory
    5. Classify pass type from trajectory shape (lob = high arc, bounce = low dip)

    All thresholds scale with frame dimensions for resolution independence.
    """

    # Max transit time for a pass (seconds)
    MAX_TRANSIT_SEC = 2.0
    # Min ball travel distance as fraction of frame width
    MIN_DISTANCE_RATIO = 0.05  # 96px at 1920, 192px at 3840
    # Trajectory analysis thresholds
    LOB_HEIGHT_RATIO = 0.08  # ball rises > 8% of frame height during transit
    BOUNCE_DIP_RATIO = 0.03  # ball dips > 3% below release height

    def __init__(self, fps: float, frame_width: int = 1920,
                 frame_height: int = 1080):
        self.fps = fps
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.min_distance_px = frame_width * self.MIN_DISTANCE_RATIO
        self._pending: _PendingPass | None = None
        self._pass_count = 0
        self.events: list[PassEvent] = []

    def on_transition(
        self,
        transition: PossessionTransition,
    ) -> PassEvent | None:
        """Process a ball possession transition.

        Called whenever BallInHandsDetector reports a possession change.

        Args:
            transition: The possession transition event.

        Returns:
            PassEvent if a complete pass was detected, None otherwise.
        """
        # Ball left a player → start tracking potential pass
        if transition.prev_holder_id is not None and transition.new_holder_id is None:
            self._pending = _PendingPass(
                from_player_id=transition.prev_holder_id,
                release_frame=transition.frame_idx,
                release_time=transition.frame_idx / self.fps,
                release_x=transition.ball_x,
                release_y=transition.ball_y,
            )
            return None

        # Ball arrived at a new player
        if transition.new_holder_id is not None:
            if self._pending is None:
                # No pending release — ball appeared in player's hands
                # (e.g., first frame, or after detection gap)
                return None

            # Check if this completes a valid pass
            pass_event = self._try_complete_pass(transition)
            self._pending = None
            return pass_event

        return None

    def track_ball(self, ball: Detection | None, frame_idx: int) -> None:
        """Record ball position during transit (between possessions).

        Call this every frame to build the pass trajectory.
        """
        if self._pending is None or ball is None:
            return

        # Check if transit has exceeded time limit
        elapsed = (frame_idx - self._pending.release_frame) / self.fps
        if elapsed > self.MAX_TRANSIT_SEC:
            self._pending = None  # Too long — not a pass
            return

        cx, cy = ball.bbox.center
        self._pending.trajectory.append((cx, cy, frame_idx))

    def _try_complete_pass(
        self,
        arrival: PossessionTransition,
    ) -> PassEvent | None:
        """Check if pending release + arrival constitutes a valid pass."""
        pending = self._pending
        if pending is None:
            return None

        # Same player released and received — not a pass (dribble/bobble)
        if pending.from_player_id == arrival.new_holder_id:
            return None

        # Check transit time
        transit_time = (arrival.frame_idx - pending.release_frame) / self.fps
        if transit_time > self.MAX_TRANSIT_SEC:
            return None
        if transit_time < 0:
            return None

        # Check ball travel distance
        distance = math.hypot(
            arrival.ball_x - pending.release_x,
            arrival.ball_y - pending.release_y,
        )
        if distance < self.min_distance_px:
            return None

        # Valid pass detected
        self._pass_count += 1

        # Build trajectory including release and arrival points
        trajectory = [
            (pending.release_x, pending.release_y, pending.release_frame),
        ]
        trajectory.extend(pending.trajectory)
        trajectory.append(
            (arrival.ball_x, arrival.ball_y, arrival.frame_idx),
        )

        # Classify pass type from trajectory shape
        pass_type = self._classify_pass_type(trajectory)

        event = PassEvent(
            frame_idx=arrival.frame_idx,
            timestamp_sec=arrival.frame_idx / self.fps,
            from_player_track_id=pending.from_player_id,
            to_player_track_id=arrival.new_holder_id,
            pass_type=pass_type,
            ball_trajectory=trajectory,
            distance_px=distance,
        )
        self.events.append(event)
        return event

    def _classify_pass_type(
        self,
        trajectory: list[tuple[float, float, int]],
    ) -> str:
        """Classify pass type from ball trajectory shape.

        - lob: ball rises significantly above release/arrival height
        - bounce: ball dips below release height during transit
        - chest: relatively flat trajectory
        - unknown: not enough trajectory data
        """
        if len(trajectory) < 3:
            return "unknown"

        release_y = trajectory[0][1]
        arrival_y = trajectory[-1][1]
        avg_endpoint_y = (release_y + arrival_y) / 2

        # Get intermediate points (exclude endpoints)
        mid_points = trajectory[1:-1]
        if not mid_points:
            return "unknown"

        min_y = min(p[1] for p in mid_points)  # highest point (y inverted)
        max_y = max(p[1] for p in mid_points)  # lowest point

        height_threshold = self.frame_height * self.LOB_HEIGHT_RATIO
        dip_threshold = self.frame_height * self.BOUNCE_DIP_RATIO

        # Lob: ball goes significantly higher than endpoints
        if avg_endpoint_y - min_y > height_threshold:
            return "lob"

        # Bounce: ball dips significantly below endpoints
        if max_y - avg_endpoint_y > dip_threshold:
            return "bounce"

        return "chest"

"""Ball-in-hands detector — true observational ball possession.

Determines whether a specific player is physically holding or dribbling
the ball by checking if the ball bounding box overlaps with the player's
upper-body region. This is a direct observation (the camera sees the ball
in the player's hands), not a proximity inference.

v1.7.0: Bbox overlap heuristic. VLM fallback for ambiguous cases is
        a future enhancement.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.tracking.tracker import TrackedPlayer
from app.vision.detection_types import Detection


@dataclass
class PossessionTransition:
    """A change in ball possession holder."""

    frame_idx: int
    prev_holder_id: int | None  # None if ball was free
    new_holder_id: int | None  # None if ball became free
    ball_x: float
    ball_y: float


class BallInHandsDetector:
    """Detects ball possession via bbox overlap (observational).

    The ball is considered "in hands" when its center falls within the
    upper-body region of a player's bounding box. The upper-body region
    is defined as the top 60% of height and inner 60% of width — this
    targets the torso and arms where the ball is held during dribbling,
    catching, and shooting.

    Unlike PossessionTracker (which uses proximity distance), this detector
    requires physical overlap — the camera must see the ball and player
    occupying the same pixel region.
    """

    # Upper-body region: top 60% of player height, inner 60% of width
    UPPER_BODY_HEIGHT_RATIO = 0.60
    UPPER_BODY_WIDTH_RATIO = 0.60
    # Minimum consecutive frames to confirm possession (noise filter)
    MIN_CONFIRM_FRAMES = 2

    def __init__(self, frame_width: int = 1920):
        self.frame_width = frame_width
        self._current_holder: int | None = None
        self._holder_frames: int = 0
        self._confirmed: bool = False
        self.transitions: list[PossessionTransition] = []

    def update(
        self,
        ball: Detection | None,
        players: list[TrackedPlayer],
        frame_idx: int,
    ) -> PossessionTransition | None:
        """Process one frame. Returns PossessionTransition on holder change.

        Args:
            ball: Ball detection for this frame (or None if not detected).
            players: Tracked players in this frame.
            frame_idx: Current frame index.

        Returns:
            PossessionTransition if the ball changed hands, None otherwise.
        """
        if ball is None:
            # Ball not detected — possession state unknown, don't change
            return None

        ball_cx, ball_cy = ball.bbox.center

        # Check which player (if any) has the ball in their upper-body region
        holder_id = self._find_holder(ball_cx, ball_cy, players)

        if holder_id == self._current_holder:
            # Same holder (or still no holder)
            if holder_id is not None:
                self._holder_frames += 1
                if self._holder_frames >= self.MIN_CONFIRM_FRAMES and not self._confirmed:
                    self._confirmed = True
            return None

        # Holder changed — emit transition if previous was confirmed
        transition = None
        if self._confirmed or (self._current_holder is None and holder_id is not None):
            transition = PossessionTransition(
                frame_idx=frame_idx,
                prev_holder_id=self._current_holder if self._confirmed else None,
                new_holder_id=holder_id,
                ball_x=ball_cx,
                ball_y=ball_cy,
            )
            self.transitions.append(transition)

        # Reset state for new holder
        self._current_holder = holder_id
        self._holder_frames = 1 if holder_id is not None else 0
        self._confirmed = False

        return transition

    @property
    def current_holder(self) -> int | None:
        """Track ID of the player currently holding the ball (confirmed only)."""
        return self._current_holder if self._confirmed else None

    def _find_holder(
        self,
        ball_cx: float,
        ball_cy: float,
        players: list[TrackedPlayer],
    ) -> int | None:
        """Find which player has the ball in their upper-body region.

        Returns the track_id of the holder, or None if ball is not inside
        any player's upper-body bbox.
        """
        for player in players:
            if self._ball_in_upper_body(ball_cx, ball_cy, player):
                return player.track_id
        return None

    def _ball_in_upper_body(
        self,
        ball_cx: float,
        ball_cy: float,
        player: TrackedPlayer,
    ) -> bool:
        """Check if ball center is inside the player's upper-body region.

        Upper-body region:
        - Vertically: top 60% of player bbox (head + torso + arms)
        - Horizontally: inner 60% centered (excludes extended legs/feet)
        """
        bbox = player.bbox
        p_width = bbox.width
        p_height = bbox.height

        # Upper-body vertical range: top of bbox to 60% down
        upper_y_top = bbox.y1
        upper_y_bottom = bbox.y1 + p_height * self.UPPER_BODY_HEIGHT_RATIO

        # Upper-body horizontal range: centered 60% of width
        margin = p_width * (1 - self.UPPER_BODY_WIDTH_RATIO) / 2
        upper_x_left = bbox.x1 + margin
        upper_x_right = bbox.x2 - margin

        return (
            upper_x_left <= ball_cx <= upper_x_right
            and upper_y_top <= ball_cy <= upper_y_bottom
        )

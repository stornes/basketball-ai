"""Shot detection via ball trajectory heuristics."""

from collections import deque
from dataclasses import dataclass

from app.events.event_types import ShotEvent, ShotOutcome
from app.tracking.tracker import TrackedPlayer
from app.vision.detection_types import Detection


@dataclass
class BallPosition:
    x: float
    y: float
    frame_idx: int


class ShotDetector:
    """Detects shot events from ball trajectory analysis.

    Algorithm:
    1. Track ball center position over time
    2. Detect upward arc (ball y decreasing = moving up in frame coords)
    3. When ball reaches apex and starts descending, record shot attempt
    4. Classify made/missed by checking if ball passes near basket bbox
    """

    # Minimum frames of upward movement to qualify as a shot arc
    MIN_ARC_FRAMES = 4
    # Minimum vertical displacement (pixels) for a shot
    MIN_VERTICAL_DISPLACEMENT = 40
    # Cooldown frames after detecting a shot
    COOLDOWN_FRAMES = 30
    # Frames before/after shot for clip boundaries
    CLIP_PADDING_FRAMES = 45
    # Proximity as a multiple of basket bbox diagonal (resolution-independent)
    BASKET_PROXIMITY_RATIO = 1.5
    # Fallback absolute proximity when basket bbox is unreliable (tiny)
    BASKET_PROXIMITY_MIN_PX = 120

    def __init__(self, frame_height: int, fps: float):
        self.frame_height = frame_height
        self.fps = fps
        self.trajectory: deque[BallPosition] = deque(maxlen=60)
        self._cooldown = 0
        self._shot_count = 0
        self._last_basket_detection: Detection | None = None

    def update(
        self,
        ball_detection: Detection | None,
        players: list[TrackedPlayer],
        frame_idx: int,
        basket_detection: Detection | None = None,
    ) -> ShotEvent | None:
        """Process one frame. Returns ShotEvent if a shot is detected.

        Args:
            ball_detection: Ball detection for this frame (or None).
            players: Tracked players for this frame.
            frame_idx: Current frame index.
            basket_detection: Basket/hoop detection for this frame (or None).
        """
        # Track latest basket position for outcome classification
        if basket_detection is not None:
            self._last_basket_detection = basket_detection

        if self._cooldown > 0:
            self._cooldown -= 1

        if ball_detection is None:
            # Ball lost - if we were tracking an arc, this might be a shot
            if self._cooldown == 0 and len(self.trajectory) >= self.MIN_ARC_FRAMES:
                shot = self._check_shot_on_ball_loss(players, frame_idx)
                if shot:
                    return shot
            return None

        cx, cy = ball_detection.bbox.center
        self.trajectory.append(BallPosition(x=cx, y=cy, frame_idx=frame_idx))

        if self._cooldown > 0 or len(self.trajectory) < self.MIN_ARC_FRAMES:
            return None

        return self._check_arc(players, frame_idx)

    def _check_arc(
        self, players: list[TrackedPlayer], frame_idx: int
    ) -> ShotEvent | None:
        """Check if recent trajectory shows a shot arc."""
        positions = list(self.trajectory)
        if len(positions) < self.MIN_ARC_FRAMES:
            return None

        # Look at recent positions for upward then downward movement
        recent = positions[-self.MIN_ARC_FRAMES * 2:] if len(positions) >= self.MIN_ARC_FRAMES * 2 else positions

        # Find apex (minimum y value = highest point in frame)
        y_values = [p.y for p in recent]
        apex_idx = y_values.index(min(y_values))

        # Need points before and after apex
        if apex_idx < 2 or apex_idx >= len(recent) - 1:
            return None

        # Check vertical displacement
        start_y = y_values[0]
        apex_y = y_values[apex_idx]
        displacement = start_y - apex_y  # positive = ball went up

        if displacement < self.MIN_VERTICAL_DISPLACEMENT:
            return None

        # Ball is in upper portion of frame at apex
        if apex_y > self.frame_height * 0.6:
            return None

        # Check that ball is descending after apex
        post_apex_y = y_values[apex_idx + 1:]
        if not post_apex_y or post_apex_y[-1] <= apex_y:
            return None

        return self._create_shot_event(recent, players, frame_idx, apex_y)

    def _check_shot_on_ball_loss(
        self, players: list[TrackedPlayer], frame_idx: int
    ) -> ShotEvent | None:
        """When ball disappears, check if last trajectory was a shot."""
        positions = list(self.trajectory)
        if len(positions) < self.MIN_ARC_FRAMES:
            return None

        # Check if ball was moving upward before disappearing
        recent = positions[-self.MIN_ARC_FRAMES:]
        y_values = [p.y for p in recent]

        # Consistently going up (y decreasing)
        going_up = all(y_values[i] > y_values[i + 1] for i in range(len(y_values) - 1))
        displacement = y_values[0] - y_values[-1]

        if going_up and displacement > self.MIN_VERTICAL_DISPLACEMENT:
            return self._create_shot_event(
                recent, players, frame_idx, y_values[-1],
                ball_loss=True,
            )

        return None

    def _create_shot_event(
        self,
        arc_positions: list[BallPosition],
        players: list[TrackedPlayer],
        frame_idx: int,
        apex_y: float,
        ball_loss: bool = False,
    ) -> ShotEvent:
        """Create a ShotEvent from detected arc."""
        self._shot_count += 1
        self._cooldown = self.COOLDOWN_FRAMES
        self.trajectory.clear()

        # Find closest player to ball at shot start (shooter)
        shooter_id = self._find_shooter(arc_positions[0], players)

        # Outcome: use ball-loss heuristic when ball disappeared mid-arc
        if ball_loss:
            outcome = self._classify_ball_loss_outcome(arc_positions)
        else:
            outcome = self._classify_outcome(arc_positions)

        start_frame = arc_positions[0].frame_idx
        return ShotEvent(
            frame_idx=start_frame,
            timestamp_sec=start_frame / self.fps,
            shooter_track_id=shooter_id,
            court_position=None,  # Set later by court mapper
            outcome=outcome,
            clip_start_frame=max(0, start_frame - self.CLIP_PADDING_FRAMES),
            clip_end_frame=frame_idx + self.CLIP_PADDING_FRAMES,
        )

    def _basket_proximity_px(self) -> float:
        """Compute proximity threshold from basket bbox size.

        Uses basket diagonal × BASKET_PROXIMITY_RATIO so the threshold
        scales with resolution and camera distance. Falls back to
        BASKET_PROXIMITY_MIN_PX for very small or missing basket detections.
        """
        if self._last_basket_detection is None:
            return self.BASKET_PROXIMITY_MIN_PX
        bbox = self._last_basket_detection.bbox
        diag = (bbox.width ** 2 + bbox.height ** 2) ** 0.5
        return max(diag * self.BASKET_PROXIMITY_RATIO, self.BASKET_PROXIMITY_MIN_PX)

    def _classify_outcome(
        self, arc_positions: list[BallPosition]
    ) -> ShotOutcome:
        """Determine if a shot was made or missed using basket proximity.

        Checks if the ball's descending trajectory passes near the basket
        bounding box. The proximity threshold scales with basket size so it
        works at any resolution (SD, HD, 4K).
        """
        if self._last_basket_detection is None:
            return ShotOutcome.ATTEMPTED

        basket_cx, basket_cy = self._last_basket_detection.bbox.center
        proximity_sq = self._basket_proximity_px() ** 2
        basket_h = self._last_basket_detection.bbox.height

        # Look at post-apex positions (descending ball)
        y_values = [p.y for p in arc_positions]
        apex_idx = y_values.index(min(y_values))

        # Check positions after the apex (squared distance avoids sqrt)
        for pos in arc_positions[apex_idx:]:
            dist_sq = (pos.x - basket_cx) ** 2 + (pos.y - basket_cy) ** 2
            if dist_sq < proximity_sq:
                # Ball near basket — allow 2× basket height below center
                # to tolerate frame skipping (sample_rate=3)
                if pos.y <= basket_cy + basket_h * 2:
                    return ShotOutcome.MADE

        return ShotOutcome.MISSED

    def _classify_ball_loss_outcome(
        self, arc_positions: list[BallPosition]
    ) -> ShotOutcome:
        """Classify outcome when ball disappears mid-arc (ball-loss shots).

        When the ball disappears while ascending, it may have gone through
        the basket (the net obscures it). Check if the ball's last known
        position was heading toward the basket's horizontal band.
        """
        if self._last_basket_detection is None:
            return ShotOutcome.ATTEMPTED

        basket_cx, basket_cy = self._last_basket_detection.bbox.center
        basket_w = self._last_basket_detection.bbox.width
        proximity = self._basket_proximity_px()

        last_pos = arc_positions[-1]

        # Ball must be in the basket's horizontal neighbourhood
        x_dist = abs(last_pos.x - basket_cx)
        if x_dist > proximity:
            return ShotOutcome.MISSED

        # Ball must be above or near basket height (ascending toward hoop)
        if last_pos.y > basket_cy + self._last_basket_detection.bbox.height * 2:
            return ShotOutcome.MISSED

        # Check trajectory direction — ball should be heading toward basket
        if len(arc_positions) >= 2:
            prev = arc_positions[-2]
            dx = last_pos.x - prev.x
            # Ball x should be moving toward basket, or already aligned
            moving_toward = (
                x_dist < basket_w * 2  # already near basket horizontally
                or (dx > 0 and last_pos.x < basket_cx)   # moving right toward basket
                or (dx < 0 and last_pos.x > basket_cx)    # moving left toward basket
            )
            if not moving_toward:
                return ShotOutcome.MISSED

        return ShotOutcome.MADE

    @staticmethod
    def _find_shooter(
        ball_pos: BallPosition, players: list[TrackedPlayer]
    ) -> int | None:
        """Find player closest to ball at start of shot arc."""
        if not players:
            return None

        min_dist = float("inf")
        shooter_id = None
        for player in players:
            cx, cy = player.bbox.center
            dist = ((cx - ball_pos.x) ** 2 + (cy - ball_pos.y) ** 2) ** 0.5
            if dist < min_dist:
                min_dist = dist
                shooter_id = player.track_id

        return shooter_id

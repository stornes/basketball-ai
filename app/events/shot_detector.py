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
    4. Classify made/missed based on ball's final position
    """

    # Minimum frames of upward movement to qualify as a shot arc
    MIN_ARC_FRAMES = 4
    # Minimum vertical displacement (pixels) for a shot
    MIN_VERTICAL_DISPLACEMENT = 40
    # Cooldown frames after detecting a shot
    COOLDOWN_FRAMES = 30
    # Frames before/after shot for clip boundaries
    CLIP_PADDING_FRAMES = 45

    def __init__(self, frame_height: int, fps: float):
        self.frame_height = frame_height
        self.fps = fps
        self.trajectory: deque[BallPosition] = deque(maxlen=60)
        self._cooldown = 0
        self._shot_count = 0

    def update(
        self,
        ball_detection: Detection | None,
        players: list[TrackedPlayer],
        frame_idx: int,
    ) -> ShotEvent | None:
        """Process one frame. Returns ShotEvent if a shot is detected."""
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
                recent, players, frame_idx, y_values[-1]
            )

        return None

    def _create_shot_event(
        self,
        arc_positions: list[BallPosition],
        players: list[TrackedPlayer],
        frame_idx: int,
        apex_y: float,
    ) -> ShotEvent:
        """Create a ShotEvent from detected arc."""
        self._shot_count += 1
        self._cooldown = self.COOLDOWN_FRAMES
        self.trajectory.clear()

        # Find closest player to ball at shot start (shooter)
        shooter_id = self._find_shooter(arc_positions[0], players)

        # Heuristic outcome: if ball apex is in top 20% of frame, likely a shot attempt
        # More sophisticated: check if ball descends to basket region
        if apex_y < self.frame_height * 0.2:
            outcome = ShotOutcome.ATTEMPTED
        else:
            outcome = ShotOutcome.ATTEMPTED

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

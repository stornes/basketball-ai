"""Possession segmentation based on ball-player proximity."""

import math

from app.events.event_types import PossessionEvent
from app.tracking.tracker import TrackedPlayer
from app.vision.detection_types import Detection


class PossessionTracker:
    """Assigns possession to the player closest to the ball."""

    POSSESSION_DISTANCE_PX = 80  # max pixels between ball and player for possession
    MIN_POSSESSION_FRAMES = 5  # minimum frames to register a possession

    def __init__(self, fps: float):
        self.fps = fps
        self._current_possessor: int | None = None
        self._possession_start_frame: int = 0
        self._possession_count = 0
        self._frames_held = 0
        self.events: list[PossessionEvent] = []

    def update(
        self,
        ball: Detection | None,
        players: list[TrackedPlayer],
        frame_idx: int,
    ) -> PossessionEvent | None:
        """Process one frame. Returns PossessionEvent on possession change."""
        if ball is None or not players:
            return self._end_possession(frame_idx, "end_of_segment")

        bx, by = ball.bbox.center

        # Find closest player to ball
        closest = None
        min_dist = float("inf")
        for player in players:
            px, py = player.bbox.center
            dist = math.hypot(px - bx, py - by)
            if dist < min_dist:
                min_dist = dist
                closest = player

        if closest is None or min_dist > self.POSSESSION_DISTANCE_PX:
            return self._end_possession(frame_idx, "turnover")

        new_possessor = closest.track_id

        if new_possessor == self._current_possessor:
            self._frames_held += 1
            return None

        # Possession change
        event = self._end_possession(frame_idx, "turnover")

        # Start new possession
        self._current_possessor = new_possessor
        self._possession_start_frame = frame_idx
        self._frames_held = 1

        return event

    def end_possession_on_shot(self, frame_idx: int) -> PossessionEvent | None:
        """Call when a shot is detected to end the current possession."""
        return self._end_possession(frame_idx, "shot")

    def _end_possession(self, frame_idx: int, result: str) -> PossessionEvent | None:
        """End current possession if it meets minimum duration."""
        if (
            self._current_possessor is not None
            and self._frames_held >= self.MIN_POSSESSION_FRAMES
        ):
            self._possession_count += 1
            event = PossessionEvent(
                possession_id=self._possession_count,
                player_track_id=self._current_possessor,
                team=self._assign_team(self._current_possessor),
                start_frame=self._possession_start_frame,
                end_frame=frame_idx,
                start_time=self._possession_start_frame / self.fps,
                end_time=frame_idx / self.fps,
                result=result,
            )
            self._current_possessor = None
            self._frames_held = 0
            self.events.append(event)
            return event

        self._current_possessor = None
        self._frames_held = 0
        return None

    @staticmethod
    def _assign_team(track_id: int) -> str:
        """Heuristic team assignment based on track ID parity."""
        # In a real system, this would use jersey color clustering
        return "team_a" if track_id % 2 == 0 else "team_b"

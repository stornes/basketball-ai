"""Event data types for shot detection and possession tracking."""

from dataclasses import dataclass
from enum import Enum


class ShotOutcome(Enum):
    ATTEMPTED = "attempted"
    MADE = "made"
    MISSED = "missed"
    UNKNOWN = "unknown"


@dataclass
class ShotEvent:
    frame_idx: int
    timestamp_sec: float
    shooter_track_id: int | None
    court_position: tuple[float, float] | None  # (x, y) in feet
    outcome: ShotOutcome
    clip_start_frame: int
    clip_end_frame: int
    team: str | None = None
    jersey_number: int | None = None
    ball_x: float | None = None  # ball pixel X at shot detection
    ball_y: float | None = None  # ball pixel Y at shot detection


@dataclass
class PassEvent:
    """An observed pass between two players.

    Unlike proximity-based possession inference, a PassEvent represents a
    directly observed ball movement: the ball left Player A's bbox, traveled
    through open space, and entered Player B's bbox. This is true observational
    data — the camera saw the pass happen.
    """

    frame_idx: int  # frame when pass completed (ball reached receiver)
    timestamp_sec: float
    from_player_track_id: int
    to_player_track_id: int
    from_team: str | None = None
    to_team: str | None = None
    pass_type: str = "unknown"  # "chest", "bounce", "lob", "unknown"
    ball_trajectory: list[tuple[float, float, int]] | None = None  # [(x, y, frame_idx), ...]
    distance_px: float = 0.0  # ball travel distance in pixels


@dataclass
class PossessionEvent:
    possession_id: int
    player_track_id: int
    team: str  # "team_a" or "team_b" (heuristic assignment)
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    result: str  # "shot", "turnover", "end_of_segment"

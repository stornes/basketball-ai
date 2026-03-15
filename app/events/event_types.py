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

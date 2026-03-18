"""Game analytics and metrics computation."""

import math

import pandas as pd

from app.events.event_types import PossessionEvent, ShotEvent, ShotOutcome
from app.tracking.tracker import TrackedPlayer
from app.vision.scoreboard_ocr import timestamp_to_quarter


class GameMetrics:
    """Computes game statistics from event logs."""

    def __init__(
        self,
        shot_events: list[ShotEvent],
        possession_events: list[PossessionEvent],
        tracks: list[TrackedPlayer] | None = None,
        fps: float = 30.0,
        quarter_duration_sec: int = 600,
        num_quarters: int | None = None,
        quarter_ranges: list[tuple[float, float]] | None = None,
    ):
        self.shot_events = shot_events
        self.possession_events = possession_events
        self.tracks = tracks or []
        self.fps = fps
        self.quarter_duration_sec = quarter_duration_sec
        self.num_quarters = num_quarters
        self.quarter_ranges = quarter_ranges  # [(start_sec, end_sec), ...] from scoreboard OCR

    @property
    def shots_attempted(self) -> int:
        return len(self.shot_events)

    @property
    def shots_made(self) -> int:
        return sum(1 for s in self.shot_events if s.outcome == ShotOutcome.MADE)

    def shot_percentage(self) -> float:
        if self.shots_attempted == 0:
            return 0.0
        return self.shots_made / self.shots_attempted

    def shots_dataframe(self) -> pd.DataFrame:
        """Shot events as a DataFrame with team and quarter columns."""
        if not self.shot_events:
            return pd.DataFrame(columns=[
                "frame_idx", "timestamp_sec", "shooter_track_id",
                "court_x", "court_y", "outcome", "team", "quarter",
                "jersey_number",
            ])
        rows = []
        for s in self.shot_events:
            cx, cy = s.court_position if s.court_position else (None, None)
            if self.quarter_ranges:
                # Use scoreboard-detected quarter boundaries
                quarter = timestamp_to_quarter(s.timestamp_sec, self.quarter_ranges)
            else:
                # Fallback: formula-based (assumes game time = video time)
                quarter = int(s.timestamp_sec / self.quarter_duration_sec) + 1
                if self.num_quarters is not None:
                    quarter = min(quarter, self.num_quarters)
            rows.append({
                "frame_idx": s.frame_idx,
                "timestamp_sec": s.timestamp_sec,
                "shooter_track_id": s.shooter_track_id,
                "court_x": cx,
                "court_y": cy,
                "outcome": s.outcome.value,
                "team": s.team,
                "quarter": quarter,
                "jersey_number": s.jersey_number,
            })
        return pd.DataFrame(rows)

    def possessions_dataframe(self) -> pd.DataFrame:
        """Possession events as a DataFrame."""
        if not self.possession_events:
            return pd.DataFrame(columns=[
                "possession_id", "player_track_id", "team",
                "start_time", "end_time", "result",
            ])
        rows = [
            {
                "possession_id": p.possession_id,
                "player_track_id": p.player_track_id,
                "team": p.team,
                "start_time": p.start_time,
                "end_time": p.end_time,
                "duration": p.end_time - p.start_time,
                "result": p.result,
            }
            for p in self.possession_events
        ]
        return pd.DataFrame(rows)

    def player_stats(self) -> list[dict]:
        """Per-player statistics."""
        stats: dict[int, dict] = {}

        for s in self.shot_events:
            pid = s.shooter_track_id
            if pid is None:
                continue
            if pid not in stats:
                stats[pid] = {"shots": 0, "made": 0, "possessions": 0, "possession_time": 0.0}
            stats[pid]["shots"] += 1
            if s.outcome == ShotOutcome.MADE:
                stats[pid]["made"] += 1

        for p in self.possession_events:
            pid = p.player_track_id
            if pid not in stats:
                stats[pid] = {"shots": 0, "made": 0, "possessions": 0, "possession_time": 0.0}
            stats[pid]["possessions"] += 1
            stats[pid]["possession_time"] += p.end_time - p.start_time

        # Compute distance run from tracks
        track_positions: dict[int, list[tuple[float, float]]] = {}
        for t in self.tracks:
            if t.track_id not in track_positions:
                track_positions[t.track_id] = []
            track_positions[t.track_id].append(t.bbox.center)

        for pid, positions in track_positions.items():
            if pid not in stats:
                stats[pid] = {"shots": 0, "made": 0, "possessions": 0, "possession_time": 0.0}
            distance_px = sum(
                math.hypot(positions[i + 1][0] - positions[i][0],
                           positions[i + 1][1] - positions[i][1])
                for i in range(len(positions) - 1)
            )
            stats[pid]["distance_px"] = distance_px

        result = []
        for pid, s in stats.items():
            fg_pct = s["made"] / s["shots"] if s["shots"] > 0 else 0.0
            result.append({
                "player_id": pid,
                "shots_attempted": s["shots"],
                "shots_made": s["made"],
                "fg_percentage": round(fg_pct, 3),
                "possessions": s["possessions"],
                "possession_time_sec": round(s["possession_time"], 1),
                "distance_px": round(s.get("distance_px", 0.0), 1),
            })

        return sorted(result, key=lambda x: x["shots_attempted"], reverse=True)

    def to_summary_dict(self) -> dict:
        """Flat summary dict for coaching agent input."""
        poss_df = self.possessions_dataframe()
        return {
            "total_shots": self.shots_attempted,
            "shots_made": self.shots_made,
            "fg_percentage": round(self.shot_percentage(), 3),
            "total_possessions": len(self.possession_events),
            "avg_possession_duration": round(
                poss_df["duration"].mean(), 1
            ) if not poss_df.empty else 0.0,
            "turnovers": len(poss_df[poss_df["result"] == "turnover"]) if not poss_df.empty else 0,
            "player_stats": self.player_stats(),
        }

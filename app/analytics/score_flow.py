"""Score flow chart — cumulative score per team per minute from video shot data."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from app.config.roster import Roster


class ScoreFlowGenerator:
    """Generates a per-minute score flow chart.

    Uses detected shot events from video to determine WHEN scoring happens,
    and the roster's per-quarter totals to determine HOW MUCH was scored.
    Each game minute's score is proportional to the number of detected shots
    in that minute, scaled so quarter totals match the scorecard exactly.
    """

    def __init__(self, quarter_duration_min: int = 10):
        self.quarter_duration_min = quarter_duration_min

    def _map_shots_to_game_minutes(
        self,
        shots_df: pd.DataFrame,
        video_duration_sec: float,
        num_quarters: int,
    ) -> pd.DataFrame:
        """Map video timestamps to game minutes.

        Divides the video into equal segments per quarter, then maps each
        shot's video timestamp to a game minute within that quarter.
        """
        quarter_video_sec = video_duration_sec / num_quarters
        records = []
        for _, shot in shots_df.iterrows():
            t = shot["timestamp_sec"]
            team = shot.get("team")
            # Which video quarter does this shot fall in?
            vq = min(int(t / quarter_video_sec), num_quarters - 1)
            # Position within the video quarter (0.0 to 1.0)
            frac = (t - vq * quarter_video_sec) / quarter_video_sec
            # Map to game minute within this quarter
            game_min = vq * self.quarter_duration_min + frac * self.quarter_duration_min
            records.append({"game_minute": int(game_min), "team": team})
        return pd.DataFrame(records)

    def generate(
        self,
        roster: Roster,
        output_path: str,
        shots_df: pd.DataFrame | None = None,
        video_duration_sec: float | None = None,
    ) -> str:
        """Generate score flow chart PNG.

        If shots_df is provided, uses video shot detection to distribute
        per-quarter scores across minutes. Otherwise falls back to plotting
        quarter-end data points only (no interpolation).
        """
        if not roster.has_scores():
            return ""

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        home_scores = list(roster.home_scores)
        away_scores = list(roster.away_scores)
        num_quarters = max(len(home_scores), len(away_scores))
        total_minutes = num_quarters * self.quarter_duration_min

        # Pad shorter list
        while len(home_scores) < num_quarters:
            home_scores.append(home_scores[-1])
        while len(away_scores) < num_quarters:
            away_scores.append(away_scores[-1])

        # Per-quarter deltas
        home_deltas = [home_scores[0]] + [
            home_scores[i] - home_scores[i - 1] for i in range(1, num_quarters)
        ]
        away_deltas = [away_scores[0]] + [
            away_scores[i] - away_scores[i - 1] for i in range(1, num_quarters)
        ]

        if shots_df is not None and video_duration_sec and not shots_df.empty:
            home_per_min, away_per_min = self._distribute_from_shots(
                shots_df, video_duration_sec, num_quarters,
                home_deltas, away_deltas, total_minutes,
            )
        else:
            # Fallback: step function at quarter boundaries (no interpolation)
            home_per_min = np.zeros(total_minutes)
            away_per_min = np.zeros(total_minutes)
            for q in range(num_quarters):
                end_min = (q + 1) * self.quarter_duration_min - 1
                home_per_min[end_min] = home_deltas[q]
                away_per_min[end_min] = away_deltas[q]

        # Cumulative score per minute — snap quarter endpoints to exact scorecard values
        home_cum = np.concatenate([[0], np.cumsum(home_per_min)])
        away_cum = np.concatenate([[0], np.cumsum(away_per_min)])
        for q in range(num_quarters):
            idx = (q + 1) * self.quarter_duration_min
            home_cum[idx] = home_scores[q]
            away_cum[idx] = away_scores[q]
        minutes = np.arange(total_minutes + 1)

        self._plot(
            minutes, home_cum, away_cum,
            roster, num_quarters, total_minutes, output_path,
        )
        return output_path

    def _distribute_from_shots(
        self,
        shots_df: pd.DataFrame,
        video_duration_sec: float,
        num_quarters: int,
        home_deltas: list[int],
        away_deltas: list[int],
        total_minutes: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Distribute quarter scores across minutes using shot detection timing."""
        mapped = self._map_shots_to_game_minutes(
            shots_df, video_duration_sec, num_quarters,
        )

        home_per_min = np.zeros(total_minutes)
        away_per_min = np.zeros(total_minutes)

        for q in range(num_quarters):
            q_start = q * self.quarter_duration_min
            q_end = (q + 1) * self.quarter_duration_min

            for team_label, deltas, per_min in [
                ("home", home_deltas, home_per_min),
                ("away", away_deltas, away_per_min),
            ]:
                q_pts = deltas[q]
                if q_pts == 0:
                    continue

                # Count shots per minute in this quarter for this team
                q_shots = mapped[
                    (mapped["team"] == team_label)
                    & (mapped["game_minute"] >= q_start)
                    & (mapped["game_minute"] < q_end)
                ]
                shots_per_min = q_shots.groupby("game_minute").size()

                total_q_shots = shots_per_min.sum()
                if total_q_shots == 0:
                    # No shots detected — distribute evenly across the quarter
                    for m in range(q_start, q_end):
                        per_min[m] = q_pts / self.quarter_duration_min
                else:
                    # Distribute points proportional to shot frequency
                    for m, count in shots_per_min.items():
                        per_min[m] = q_pts * (count / total_q_shots)

        return home_per_min, away_per_min

    def _plot(
        self,
        minutes: np.ndarray,
        home_cum: np.ndarray,
        away_cum: np.ndarray,
        roster: Roster,
        num_quarters: int,
        total_minutes: int,
        output_path: str,
    ) -> None:
        """Render the score flow chart."""
        fig, ax = plt.subplots(figsize=(14, 6))
        fig.patch.set_facecolor("#2D3250")
        ax.set_facecolor("#3B4070")

        # Plot lines
        ax.plot(
            minutes, home_cum,
            color="#FF6B6B", linewidth=2.5, label=roster.home_team_name,
            zorder=5,
        )
        ax.plot(
            minutes, away_cum,
            color="#51CF66", linewidth=2.5, label=roster.away_team_name,
            zorder=5,
        )

        # Quarter-end markers
        q_minutes = [i * self.quarter_duration_min for i in range(1, num_quarters + 1)]
        ax.scatter(q_minutes, [home_cum[m] for m in q_minutes], color="#FF6B6B", s=40, zorder=6)
        ax.scatter(q_minutes, [away_cum[m] for m in q_minutes], color="#51CF66", s=40, zorder=6)

        # Quarter boundary lines
        for qi, m in enumerate(q_minutes, 1):
            q_label = f"OT{qi - 4}" if qi > 4 else f"Q{qi}"
            ax.axvline(x=m, color="white", linewidth=0.8, alpha=0.5, zorder=2)
            ax.text(
                m, max(home_cum[-1], away_cum[-1]) * 1.05,
                q_label, ha="center", va="bottom",
                color="white", fontsize=11, fontweight="bold",
            )

        # Styling
        max_score = max(home_cum[-1], away_cum[-1])
        ax.set_ylim(0, max_score * 1.12)
        ax.set_xlim(0, total_minutes)
        ax.set_xlabel("Game Minutes", color="white", fontsize=12)
        ax.set_ylabel("Points", color="white", fontsize=12)
        ax.set_xticks(range(0, total_minutes + 1, 5))
        ax.tick_params(colors="white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color("white")
        ax.spines["left"].set_color("white")

        ax.yaxis.grid(True, alpha=0.2, color="white")
        ax.xaxis.grid(True, alpha=0.1, color="white")
        ax.set_axisbelow(True)

        legend = ax.legend(
            loc="upper left", fontsize=12,
            facecolor="#3B4070", edgecolor="white",
        )
        for text in legend.get_texts():
            text.set_color("white")

        final_home = int(home_cum[-1])
        final_away = int(away_cum[-1])
        title = f"Score Flow — {roster.home_team_name} {final_home} vs {roster.away_team_name} {final_away}"
        ax.set_title(title, fontsize=16, fontweight="bold", color="white", pad=20)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

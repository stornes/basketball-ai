"""Shot chart visualization on NBA court diagram."""

from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class ShotChartGenerator:
    """Renders shots on an NBA half-court diagram."""

    # NBA half court dimensions in feet
    COURT_WIDTH = 50.0
    COURT_LENGTH = 47.0  # half court

    def generate(self, shots_df: pd.DataFrame, output_path: str) -> str:
        """Generate shot chart PNG from shots DataFrame.

        Expected columns: court_x, court_y, outcome
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 9.4))
        self._draw_court(ax)

        if not shots_df.empty and "court_x" in shots_df.columns:
            valid = shots_df.dropna(subset=["court_x", "court_y"])
            if not valid.empty:
                made_mask = valid["outcome"] == "made"
                if made_mask.any():
                    ax.scatter(
                        valid.loc[made_mask, "court_x"],
                        valid.loc[made_mask, "court_y"],
                        c="green", marker="o", s=120, label="Made",
                        alpha=0.7, edgecolors="darkgreen", linewidths=1.5,
                        zorder=5,
                    )
                missed_mask = ~made_mask
                if missed_mask.any():
                    ax.scatter(
                        valid.loc[missed_mask, "court_x"],
                        valid.loc[missed_mask, "court_y"],
                        c="red", marker="x", s=120, label="Missed",
                        alpha=0.7, linewidths=2,
                        zorder=5,
                    )

        ax.set_xlim(-2, self.COURT_WIDTH + 2)
        ax.set_ylim(-2, self.COURT_LENGTH + 2)
        ax.set_aspect("equal")
        ax.legend(loc="upper right")
        ax.set_title("Shot Chart", fontsize=16, fontweight="bold")
        ax.set_xlabel("Court Width (ft)")
        ax.set_ylabel("Court Length (ft)")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return output_path

    def _draw_court(self, ax: plt.Axes) -> None:
        """Draw NBA half-court lines and markings."""
        # Court outline
        court = patches.Rectangle(
            (0, 0), self.COURT_WIDTH, self.COURT_LENGTH,
            linewidth=2, edgecolor="black", facecolor="#E8D5B7", zorder=0,
        )
        ax.add_patch(court)

        # Paint / key (19ft x 16ft, centered)
        paint_x = (self.COURT_WIDTH - 16) / 2
        paint = patches.Rectangle(
            (paint_x, 0), 16, 19,
            linewidth=2, edgecolor="black", facecolor="#D4A574", zorder=1,
        )
        ax.add_patch(paint)

        # Free throw circle (6ft radius, centered at top of key)
        ft_circle = patches.Circle(
            (self.COURT_WIDTH / 2, 19), 6,
            linewidth=2, edgecolor="black", facecolor="none", zorder=2,
        )
        ax.add_patch(ft_circle)

        # Basket (1.5ft from baseline, centered)
        basket = patches.Circle(
            (self.COURT_WIDTH / 2, 5.25), 0.75,
            linewidth=2, edgecolor="orange", facecolor="orange", zorder=3,
        )
        ax.add_patch(basket)

        # Backboard
        ax.plot(
            [self.COURT_WIDTH / 2 - 3, self.COURT_WIDTH / 2 + 3],
            [4, 4],
            color="black", linewidth=3, zorder=3,
        )

        # Three-point line
        # Corner threes are straight lines (3ft from sideline, up to ~14ft)
        ax.plot([3, 3], [0, 14], color="black", linewidth=2, zorder=2)
        ax.plot([self.COURT_WIDTH - 3, self.COURT_WIDTH - 3], [0, 14],
                color="black", linewidth=2, zorder=2)

        # Three-point arc (23.75ft from basket center)
        three_pt_angles = np.linspace(
            np.arccos(14 / 23.75), np.pi - np.arccos(14 / 23.75), 100
        )
        three_pt_x = self.COURT_WIDTH / 2 + 23.75 * np.cos(three_pt_angles)
        three_pt_y = 5.25 + 23.75 * np.sin(three_pt_angles)
        ax.plot(three_pt_x, three_pt_y, color="black", linewidth=2, zorder=2)

        # Restricted area arc (4ft radius from basket)
        restricted_angles = np.linspace(0, np.pi, 50)
        restricted_x = self.COURT_WIDTH / 2 + 4 * np.cos(restricted_angles)
        restricted_y = 5.25 + 4 * np.sin(restricted_angles)
        ax.plot(restricted_x, restricted_y, color="black", linewidth=1.5, zorder=2)

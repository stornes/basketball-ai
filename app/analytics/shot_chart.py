"""Shot chart visualization on NBA court diagram."""

from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Full court length for mirroring far-basket shots
FULL_COURT_LENGTH = 94.0
HALF_COURT_LENGTH = 47.0
COURT_WIDTH = 50.0
BASKET_Y = 5.25  # basket distance from baseline in feet


class ShotChartGenerator:
    """Renders shots on an NBA half-court diagram."""

    # NBA half court dimensions in feet
    COURT_WIDTH = COURT_WIDTH
    COURT_LENGTH = HALF_COURT_LENGTH

    def generate(
        self,
        shots_df: pd.DataFrame,
        output_path: str,
        title: str = "Shot Chart",
    ) -> str:
        """Generate shot chart PNG from shots DataFrame.

        Expected columns: court_x, court_y, outcome
        Handles mirroring: shots with court_y > 47 are mirrored to half-court.
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 9.4))
        self._draw_court(ax)

        if not shots_df.empty and "court_x" in shots_df.columns:
            # Normalize to half-court (mirror far-basket shots)
            plot_df = self._normalize_to_half_court(shots_df)
            valid = plot_df.dropna(subset=["court_x", "court_y"])
            if not valid.empty:
                made_mask = valid["outcome"] == "made"
                if made_mask.any():
                    ax.scatter(
                        valid.loc[made_mask, "court_x"],
                        valid.loc[made_mask, "court_y"],
                        c="green", marker="x", s=120, label="Made",
                        alpha=0.7, linewidths=2.5,
                        zorder=5,
                    )
                    # Annotate made shots with jersey numbers
                    if "jersey_number" in valid.columns:
                        made_with_jersey = valid.loc[
                            made_mask & valid["jersey_number"].notna()
                        ]
                        for _, row in made_with_jersey.iterrows():
                            ax.annotate(
                                str(int(row["jersey_number"])),
                                (row["court_x"], row["court_y"]),
                                textcoords="offset points",
                                xytext=(6, 6),
                                fontsize=8,
                                fontweight="bold",
                                color="#006400",
                                zorder=6,
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
        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.set_xlabel("Court Width (ft)")
        ax.set_ylabel("Court Length (ft)")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return output_path

    @staticmethod
    def _normalize_to_half_court(df: pd.DataFrame) -> pd.DataFrame:
        """Mirror far-basket shots (court_y > 47) to near-basket half-court.

        Far-basket shots are reflected: y → 94 - y, x → 50 - x.
        This puts all shots on the same half-court diagram with basket at y=5.25.
        """
        result = df.copy()
        if "court_x" not in result.columns or "court_y" not in result.columns:
            return result

        far_mask = result["court_y"].notna() & (result["court_y"] > HALF_COURT_LENGTH)
        if far_mask.any():
            result.loc[far_mask, "court_y"] = FULL_COURT_LENGTH - result.loc[far_mask, "court_y"]
            result.loc[far_mask, "court_x"] = COURT_WIDTH - result.loc[far_mask, "court_x"]
        return result

    @staticmethod
    def basket_relative_coords(
        ball_x: float, ball_y: float,
        basket_det_cx: float, basket_det_cy: float,
        basket_det_width: float,
        frame_width: int,
        frame_height: int,
    ) -> tuple[float, float]:
        """Compute court coordinates from ball position relative to basket.

        Uses the detected basket position as anchor point. The basket maps to
        (25.0, 5.25) on the half-court diagram. Ball offset from basket in
        pixels is scaled to feet using basket width as reference (regulation
        basket = 1.5 ft diameter).

        Args:
            ball_x, ball_y: Ball pixel coordinates at shot detection.
            basket_det_cx, basket_det_cy: Basket detection center in pixels.
            basket_det_width: Basket detection bbox width in pixels.
            frame_width, frame_height: Frame dimensions for fallback scaling.

        Returns:
            (court_x, court_y) in feet on a full-court coordinate system.
            Near-basket shots: y ≈ 5-25. Far-basket shots: y ≈ 69-89.
        """
        # Scale factor: pixels per foot from basket size
        # Regulation basket = 18 inches = 1.5 ft diameter
        if basket_det_width > 5:  # minimum viable basket detection
            px_per_ft = basket_det_width / 1.5
        else:
            # Fallback: estimate from frame width (court = 50 ft)
            px_per_ft = frame_width / 50.0

        # Ball offset from basket in feet
        dx_ft = (ball_x - basket_det_cx) / px_per_ft
        # Invert Y: pixel Y increases downward, court Y increases upward
        dy_ft = (basket_det_cy - ball_y) / px_per_ft

        # Determine which basket (near or far) based on basket position in frame
        # If basket is in the lower half of frame → near basket (y ≈ 5.25)
        # If basket is in the upper half → far basket (y ≈ 88.75)
        if basket_det_cy > frame_height / 2:
            # Near basket (bottom of frame)
            court_x = 25.0 + dx_ft
            court_y = BASKET_Y + dy_ft
        else:
            # Far basket (top of frame) — map to far end of full court
            court_x = 25.0 - dx_ft  # mirror X for far basket
            court_y = (FULL_COURT_LENGTH - BASKET_Y) - dy_ft

        # Clamp to court bounds
        court_x = max(0.0, min(court_x, COURT_WIDTH))
        court_y = max(0.0, min(court_y, FULL_COURT_LENGTH))

        return (court_x, court_y)

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

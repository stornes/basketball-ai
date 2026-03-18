"""Three-point classification from court coordinates.

Uses NBA court geometry to classify shots as 2-point or 3-point based on
the shooter's court position (in feet) relative to the basket.

Constants:
    - Three-point arc radius: 23.75 ft from basket center
    - Basket center: (25.0, 5.25) ft on a 50x47 ft half court
    - Corner three: x <= 3 ft from sideline AND y <= 14 ft from baseline
"""

from __future__ import annotations

import math


# NBA court constants (feet)
THREE_PT_ARC_RADIUS_FT = 23.75
BASKET_CENTER_X = 25.0
BASKET_CENTER_Y = 5.25
CORNER_THREE_X_MIN = 3.0  # sideline boundary
CORNER_THREE_X_MAX = 47.0  # 50 - 3
CORNER_THREE_Y_MAX = 14.0


class ThreePointClassifier:
    """Classifies shots as 2-point or 3-point from court coordinates."""

    def __init__(
        self,
        arc_radius: float = THREE_PT_ARC_RADIUS_FT,
        basket_x: float = BASKET_CENTER_X,
        basket_y: float = BASKET_CENTER_Y,
    ):
        self.arc_radius = arc_radius
        self.basket_x = basket_x
        self.basket_y = basket_y

    def is_three_pointer(self, court_x: float | None, court_y: float | None) -> bool:
        """Determine if a shot from (court_x, court_y) is a three-pointer.

        Args:
            court_x: X position on court in feet (0-50, sideline to sideline).
            court_y: Y position on court in feet (0-47, baseline to half court).

        Returns:
            True if the shot is beyond the three-point line, False otherwise.
            Returns False if coordinates are None (court position unknown).
        """
        if court_x is None or court_y is None:
            return False

        # Corner three: close to sideline and below the arc break point
        if court_y <= CORNER_THREE_Y_MAX:
            if court_x <= CORNER_THREE_X_MIN or court_x >= CORNER_THREE_X_MAX:
                return True

        # Arc three: distance from basket center > arc radius
        distance = math.hypot(court_x - self.basket_x, court_y - self.basket_y)
        return distance > self.arc_radius

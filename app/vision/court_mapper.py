"""Court mapping via classical CV: line detection + homography."""

import cv2
import numpy as np


class CourtMapper:
    """Detects court lines and computes pixel-to-court-feet homography."""

    COURT_LENGTH_FT = 94.0
    COURT_WIDTH_FT = 50.0

    def __init__(self, output_scale: float = 10.0):
        self._scale = output_scale
        self.H: np.ndarray | None = None  # 3x3 homography matrix

    def calibrate(self, frame: np.ndarray) -> bool:
        """Attempt to find court corners via Hough lines. Returns True on success."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, threshold1=50, threshold2=150)

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=100,
            minLineLength=100,
            maxLineGap=20,
        )

        if lines is None or len(lines) < 4:
            return False

        h_lines, v_lines = self._classify_lines(lines)

        if len(h_lines) < 2 or len(v_lines) < 2:
            return False

        corners = self._find_court_corners(h_lines, v_lines, frame.shape)
        if corners is None:
            return False

        src = np.float32(corners)
        dst = np.float32([
            [0, 0],
            [self.COURT_WIDTH_FT * self._scale, 0],
            [self.COURT_WIDTH_FT * self._scale, self.COURT_LENGTH_FT * self._scale],
            [0, self.COURT_LENGTH_FT * self._scale],
        ])

        self.H = cv2.getPerspectiveTransform(src, dst)
        return True

    def to_court_coords(self, pixel_x: float, pixel_y: float) -> tuple[float, float] | None:
        """Convert pixel coordinates to court coordinates (in feet)."""
        if self.H is None:
            return None
        pt = np.float32([[[pixel_x, pixel_y]]])
        transformed = cv2.perspectiveTransform(pt, self.H)
        x, y = transformed[0][0]
        return (x / self._scale, y / self._scale)

    def _classify_lines(
        self, lines: np.ndarray
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Classify lines as horizontal or vertical based on angle."""
        h_lines = []
        v_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1)))
            if angle < 20:
                h_lines.append(line[0])
            elif angle > 70:
                v_lines.append(line[0])

        return h_lines, v_lines

    def _find_court_corners(
        self,
        h_lines: list[np.ndarray],
        v_lines: list[np.ndarray],
        frame_shape: tuple,
    ) -> list[list[float]] | None:
        """Find 4 court corners from classified lines."""
        h = frame_shape[0]
        w = frame_shape[1]

        # Sort horizontal lines by y-position (midpoint)
        h_sorted = sorted(h_lines, key=lambda l: (l[1] + l[3]) / 2)
        # Sort vertical lines by x-position (midpoint)
        v_sorted = sorted(v_lines, key=lambda l: (l[0] + l[2]) / 2)

        if len(h_sorted) < 2 or len(v_sorted) < 2:
            return None

        # Take outermost lines
        top_line = h_sorted[0]
        bottom_line = h_sorted[-1]
        left_line = v_sorted[0]
        right_line = v_sorted[-1]

        # Find intersections of boundary lines
        corners = [
            self._line_intersection(top_line, left_line),
            self._line_intersection(top_line, right_line),
            self._line_intersection(bottom_line, right_line),
            self._line_intersection(bottom_line, left_line),
        ]

        if any(c is None for c in corners):
            return None

        # Validate corners are within frame bounds (with margin)
        margin = 50
        for cx, cy in corners:
            if cx < -margin or cx > w + margin or cy < -margin or cy > h + margin:
                return None

        return corners

    @staticmethod
    def _line_intersection(
        line1: np.ndarray, line2: np.ndarray
    ) -> list[float] | None:
        """Find intersection point of two line segments (extended to infinity)."""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return None

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        px = x1 + t * (x2 - x1)
        py = y1 + t * (y2 - y1)

        return [px, py]

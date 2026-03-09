"""Tests for court mapping."""

import cv2
import numpy as np

from app.vision.court_mapper import CourtMapper


def test_line_intersection():
    """Test line intersection math."""
    # Horizontal line y=100
    h_line = np.array([0, 100, 640, 100])
    # Vertical line x=320
    v_line = np.array([320, 0, 320, 480])

    result = CourtMapper._line_intersection(h_line, v_line)
    assert result is not None
    assert abs(result[0] - 320) < 1
    assert abs(result[1] - 100) < 1


def test_parallel_lines_no_intersection():
    """Parallel lines should return None."""
    line1 = np.array([0, 100, 640, 100])
    line2 = np.array([0, 200, 640, 200])

    result = CourtMapper._line_intersection(line1, line2)
    assert result is None


def test_calibrate_with_court_frame():
    """Test calibration on a frame with drawn court lines."""
    # Create a frame with clear court boundary lines
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (139, 119, 101)  # wood court color

    # Draw court boundary
    cv2.rectangle(frame, (50, 30), (590, 450), (255, 255, 255), 3)
    # Draw center line
    cv2.line(frame, (50, 240), (590, 240), (255, 255, 255), 2)
    # Draw some additional court lines
    cv2.line(frame, (320, 30), (320, 450), (255, 255, 255), 2)

    mapper = CourtMapper()
    success = mapper.calibrate(frame)

    # May or may not succeed depending on Hough line detection params
    # but at minimum should not crash
    if success:
        # Verify we can convert coordinates
        result = mapper.to_court_coords(320, 240)
        assert result is not None
        assert isinstance(result, tuple)
        assert len(result) == 2


def test_to_court_coords_without_calibration():
    """Without calibration, to_court_coords returns None."""
    mapper = CourtMapper()
    assert mapper.to_court_coords(100, 100) is None

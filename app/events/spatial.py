"""Spatial utility for court-plane distance calculations.

Provides `court_distance` — the single entry point for all proximity checks
in event detectors. When a homography matrix is available, both points are
projected onto the top-down court plane and the Euclidean distance is returned
in court units (feet, matching CourtMapper's output scale).  When no homography
is provided the function falls back to raw pixel distance, preserving the
existing behaviour of all callers.

Design note: keeping this module dependency-free (only numpy and math) so it
can be imported anywhere in the event layer without circular imports.
"""

from __future__ import annotations

import math

import numpy as np


def project_point(
    point: tuple[float, float],
    homography: np.ndarray | None,
) -> tuple[float, float] | None:
    """Project a single pixel coordinate onto the court plane.

    Args:
        point: (x, y) in pixel space.
        homography: 3x3 perspective transform matrix (from CourtMapper.H).
            If None, returns None — callers should fall back to pixel coords.

    Returns:
        (court_x, court_y) in court units, or None if homography is absent.
    """
    if homography is None:
        return None

    px, py = point
    pt = np.float64([[[px, py]]])
    transformed = np.squeeze(
        np.dot(
            homography,
            np.array([px, py, 1.0], dtype=np.float64),
        )
    )
    # Homogeneous → Cartesian
    w = transformed[2]
    if abs(w) < 1e-12:
        return None
    return (transformed[0] / w, transformed[1] / w)


def court_distance(
    pos_a: tuple[float, float],
    pos_b: tuple[float, float],
    homography: np.ndarray | None = None,
) -> float:
    """Euclidean distance between two positions.

    When a homography matrix is provided both points are projected to the
    top-down court coordinate system first (units: feet, matching
    CourtMapper's output).  Without a homography the raw pixel distance is
    returned unchanged, preserving backward compatibility.

    Args:
        pos_a: First position (x, y), pixel coordinates.
        pos_b: Second position (x, y), pixel coordinates.
        homography: Optional 3x3 perspective transform from CourtMapper.H.

    Returns:
        Distance in court feet when projected, pixels otherwise.
    """
    if homography is not None:
        a_court = project_point(pos_a, homography)
        b_court = project_point(pos_b, homography)
        if a_court is not None and b_court is not None:
            return math.hypot(b_court[0] - a_court[0], b_court[1] - a_court[1])
        # Projection failed — fall through to pixel distance

    return math.hypot(pos_b[0] - pos_a[0], pos_b[1] - pos_a[1])

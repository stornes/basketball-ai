"""Shared test fixtures - synthetic video and detection data."""

import numpy as np
import pytest

from app.events.event_types import PossessionEvent, ShotEvent, ShotOutcome
from app.tracking.tracker import TrackedPlayer
from app.vision.detection_types import BoundingBox, Detection


@pytest.fixture(scope="session")
def synthetic_frame():
    """480x640 BGR frame with 2 'players' and 1 'ball'."""
    import cv2
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (100, 180, 80)  # green court

    # Player 1
    cv2.rectangle(frame, (100, 200), (150, 320), (50, 50, 200), -1)
    # Player 2
    cv2.rectangle(frame, (400, 180), (450, 300), (200, 50, 50), -1)
    # Ball
    cv2.circle(frame, (280, 150), 15, (0, 140, 255), -1)

    return frame


@pytest.fixture(scope="session")
def synthetic_video_path(tmp_path_factory):
    """30-frame synthetic .mp4 video with moving ball."""
    import cv2
    tmp = tmp_path_factory.mktemp("video")
    path = str(tmp / "test_game.mp4")
    h, w = 480, 640

    writer = cv2.VideoWriter(
        path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h)
    )

    for i in range(30):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:] = (100, 180, 80)

        # Players move slightly
        cv2.rectangle(frame, (100 + i, 200), (150 + i, 320), (50, 50, 200), -1)
        cv2.rectangle(frame, (400 - i, 180), (450 - i, 300), (200, 50, 50), -1)

        # Ball follows an arc trajectory
        ball_x = 280 + i * 3
        ball_y = int(150 - 5 * i + 0.3 * i * i)  # parabolic arc
        ball_y = max(20, min(460, ball_y))
        cv2.circle(frame, (ball_x, ball_y), 15, (0, 140, 255), -1)

        writer.write(frame)

    writer.release()
    return path


@pytest.fixture
def sample_detections():
    """List of detections for one frame: 2 persons + 1 ball."""
    return [
        Detection(
            bbox=BoundingBox(100, 200, 150, 320),
            confidence=0.9,
            class_id=0,
            class_name="person",
            frame_idx=0,
        ),
        Detection(
            bbox=BoundingBox(400, 180, 450, 300),
            confidence=0.85,
            class_id=0,
            class_name="person",
            frame_idx=0,
        ),
        Detection(
            bbox=BoundingBox(265, 135, 295, 165),
            confidence=0.7,
            class_id=32,
            class_name="sports ball",
            frame_idx=0,
        ),
    ]


@pytest.fixture
def sample_shot_events():
    """Pre-built list of shot events."""
    return [
        ShotEvent(
            frame_idx=100, timestamp_sec=3.33, shooter_track_id=1,
            court_position=(25.0, 20.0), outcome=ShotOutcome.MADE,
            clip_start_frame=55, clip_end_frame=145,
        ),
        ShotEvent(
            frame_idx=300, timestamp_sec=10.0, shooter_track_id=2,
            court_position=(35.0, 30.0), outcome=ShotOutcome.MISSED,
            clip_start_frame=255, clip_end_frame=345,
        ),
        ShotEvent(
            frame_idx=500, timestamp_sec=16.67, shooter_track_id=1,
            court_position=(25.0, 15.0), outcome=ShotOutcome.MADE,
            clip_start_frame=455, clip_end_frame=545,
        ),
    ]


@pytest.fixture
def sample_possession_events():
    """Pre-built list of possession events."""
    return [
        PossessionEvent(
            possession_id=1, player_track_id=1, team="team_a",
            start_frame=0, end_frame=100, start_time=0.0, end_time=3.33,
            result="shot",
        ),
        PossessionEvent(
            possession_id=2, player_track_id=2, team="team_b",
            start_frame=110, end_frame=300, start_time=3.67, end_time=10.0,
            result="shot",
        ),
        PossessionEvent(
            possession_id=3, player_track_id=1, team="team_a",
            start_frame=310, end_frame=400, start_time=10.33, end_time=13.33,
            result="turnover",
        ),
    ]


@pytest.fixture
def sample_tracks():
    """Sample tracked player data."""
    tracks = []
    for frame in range(10):
        tracks.append(TrackedPlayer(
            track_id=1,
            bbox=BoundingBox(100 + frame * 2, 200, 150 + frame * 2, 320),
            frame_idx=frame,
            is_confirmed=True,
        ))
        tracks.append(TrackedPlayer(
            track_id=2,
            bbox=BoundingBox(400 - frame, 180, 450 - frame, 300),
            frame_idx=frame,
            is_confirmed=True,
        ))
    return tracks

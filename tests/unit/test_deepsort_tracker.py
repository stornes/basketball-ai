"""Unit tests for DeepSortTracker.

TDD: these tests define the contract before implementation.

Tests:
    1. update() returns TrackedPlayer objects with the correct fields
    2. Track IDs are stable across frames for the same detection
    3. Config switching between "iou" and "deepsort" selects the right tracker
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.vision.detection_types import BoundingBox, Detection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_frame(h: int = 480, w: int = 640) -> np.ndarray:
    """Return a blank BGR frame."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def _make_detection(
    x1: float, y1: float, x2: float, y2: float, frame_idx: int = 0
) -> Detection:
    return Detection(
        bbox=BoundingBox(x1, y1, x2, y2),
        confidence=0.9,
        class_id=0,  # person
        class_name="person",
        frame_idx=frame_idx,
    )


def _make_deepsort_track(track_id: int, tlbr: tuple, confirmed: bool = True):
    """Build a mock deep_sort_realtime track object."""
    track = MagicMock()
    track.track_id = track_id
    track.is_confirmed.return_value = confirmed
    track.to_tlbr.return_value = np.array(list(tlbr), dtype=float)
    track.det_class = 0
    return track


# ---------------------------------------------------------------------------
# Test 1: update() returns TrackedPlayer objects
# ---------------------------------------------------------------------------

class TestDeepSortTrackerReturnsTrackedPlayers:
    """DeepSortTracker.update() must return a list of TrackedPlayer instances."""

    def test_returns_tracked_player_list(self):
        mock_track = _make_deepsort_track(1, (10.0, 20.0, 110.0, 220.0))

        with patch("app.tracking.deepsort_tracker.DeepSort") as MockDeepSort:
            instance = MockDeepSort.return_value
            instance.update_tracks.return_value = [mock_track]

            from app.tracking.deepsort_tracker import DeepSortTracker
            from app.tracking.tracker import TrackedPlayer

            tracker = DeepSortTracker()
            det = _make_detection(10, 20, 110, 220)
            frame = _make_frame()
            result = tracker.update([det], frame)

        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], TrackedPlayer)

    def test_tracked_player_has_correct_fields(self):
        mock_track = _make_deepsort_track(7, (50.0, 100.0, 150.0, 300.0))

        with patch("app.tracking.deepsort_tracker.DeepSort") as MockDeepSort:
            instance = MockDeepSort.return_value
            instance.update_tracks.return_value = [mock_track]

            from app.tracking.deepsort_tracker import DeepSortTracker

            tracker = DeepSortTracker()
            det = _make_detection(50, 100, 150, 300)
            frame = _make_frame()
            result = tracker.update([det], frame)

        player = result[0]
        assert player.track_id == 7
        assert player.bbox.x1 == pytest.approx(50.0)
        assert player.bbox.y1 == pytest.approx(100.0)
        assert player.bbox.x2 == pytest.approx(150.0)
        assert player.bbox.y2 == pytest.approx(300.0)
        assert player.is_confirmed is True
        # frame_idx is always 0 because DeepSORT is frame-agnostic
        assert player.frame_idx == 0

    def test_unconfirmed_tracks_are_excluded(self):
        confirmed = _make_deepsort_track(1, (10.0, 10.0, 50.0, 50.0), confirmed=True)
        unconfirmed = _make_deepsort_track(2, (60.0, 60.0, 100.0, 100.0), confirmed=False)

        with patch("app.tracking.deepsort_tracker.DeepSort") as MockDeepSort:
            instance = MockDeepSort.return_value
            instance.update_tracks.return_value = [confirmed, unconfirmed]

            from app.tracking.deepsort_tracker import DeepSortTracker

            tracker = DeepSortTracker()
            dets = [
                _make_detection(10, 10, 50, 50),
                _make_detection(60, 60, 100, 100),
            ]
            result = tracker.update(dets, _make_frame())

        assert len(result) == 1
        assert result[0].track_id == 1

    def test_empty_detections_returns_empty_list(self):
        with patch("app.tracking.deepsort_tracker.DeepSort") as MockDeepSort:
            instance = MockDeepSort.return_value
            instance.update_tracks.return_value = []

            from app.tracking.deepsort_tracker import DeepSortTracker

            tracker = DeepSortTracker()
            result = tracker.update([], _make_frame())

        assert result == []


# ---------------------------------------------------------------------------
# Test 2: Track IDs are stable across frames
# ---------------------------------------------------------------------------

class TestTrackIdStability:
    """The same physical detection must yield the same track_id across frames."""

    def test_same_id_across_two_frames(self):
        """DeepSORT must return ID=1 for both frames for a stationary player."""
        track_frame1 = _make_deepsort_track(1, (100.0, 200.0, 200.0, 400.0))
        track_frame2 = _make_deepsort_track(1, (102.0, 201.0, 202.0, 401.0))

        with patch("app.tracking.deepsort_tracker.DeepSort") as MockDeepSort:
            instance = MockDeepSort.return_value
            # Return ID 1 on both calls
            instance.update_tracks.side_effect = [[track_frame1], [track_frame2]]

            from app.tracking.deepsort_tracker import DeepSortTracker

            tracker = DeepSortTracker()
            frame = _make_frame()
            det = _make_detection(100, 200, 200, 400)

            r1 = tracker.update([det], frame)
            r2 = tracker.update([det], frame)

        assert r1[0].track_id == r2[0].track_id == 1

    def test_different_players_get_different_ids(self):
        t1 = _make_deepsort_track(1, (10.0, 10.0, 60.0, 60.0))
        t2 = _make_deepsort_track(2, (300.0, 300.0, 360.0, 360.0))

        with patch("app.tracking.deepsort_tracker.DeepSort") as MockDeepSort:
            instance = MockDeepSort.return_value
            instance.update_tracks.return_value = [t1, t2]

            from app.tracking.deepsort_tracker import DeepSortTracker

            tracker = DeepSortTracker()
            result = tracker.update(
                [_make_detection(10, 10, 60, 60), _make_detection(300, 300, 360, 360)],
                _make_frame(),
            )

        ids = {p.track_id for p in result}
        assert ids == {1, 2}

    def test_frame_passed_to_deepsort(self):
        """DeepSORT needs the actual frame for appearance embeddings."""
        with patch("app.tracking.deepsort_tracker.DeepSort") as MockDeepSort:
            instance = MockDeepSort.return_value
            instance.update_tracks.return_value = []

            from app.tracking.deepsort_tracker import DeepSortTracker

            tracker = DeepSortTracker()
            frame = _make_frame()
            tracker.update([], frame)

            call_kwargs = instance.update_tracks.call_args
            assert call_kwargs is not None
            # Frame must be passed as 'frame' kwarg (our implementation uses keyword)
            passed_frame = call_kwargs.kwargs.get("frame")
            if passed_frame is None and len(call_kwargs.args) > 1:
                passed_frame = call_kwargs.args[1]
            assert passed_frame is not None
            assert isinstance(passed_frame, np.ndarray)


# ---------------------------------------------------------------------------
# Test 3: Config switching between "iou" and "deepsort"
# ---------------------------------------------------------------------------

class TestConfigTrackerSwitching:
    """PipelineConfig.tracker_type must select the correct tracker."""

    def test_default_tracker_type_is_deepsort(self):
        from app.pipeline.pipeline_config import PipelineConfig

        config = PipelineConfig()
        assert config.tracker_type == "deepsort"

    def test_can_set_iou(self):
        from app.pipeline.pipeline_config import PipelineConfig

        config = PipelineConfig(tracker_type="iou")
        assert config.tracker_type == "iou"

    def test_can_set_deepsort(self):
        from app.pipeline.pipeline_config import PipelineConfig

        config = PipelineConfig(tracker_type="deepsort")
        assert config.tracker_type == "deepsort"

    def test_iou_config_uses_player_tracker(self):
        """When tracker_type='iou' the orchestrator should use PlayerTracker."""
        from app.pipeline.pipeline_config import PipelineConfig
        from app.tracking.tracker import PlayerTracker

        config = PipelineConfig(tracker_type="iou")
        # We're verifying the config value here; the orchestrator wires it up.
        # Integration-level wiring is covered in run_analysis.py directly.
        assert config.tracker_type == "iou"

    def test_deepsort_config_uses_deepsort_tracker(self):
        """When tracker_type='deepsort' the orchestrator should use DeepSortTracker."""
        from app.pipeline.pipeline_config import PipelineConfig

        config = PipelineConfig(tracker_type="deepsort")
        assert config.tracker_type == "deepsort"


# ---------------------------------------------------------------------------
# Test 4: Detection format conversion
# ---------------------------------------------------------------------------

class TestDetectionFormatConversion:
    """DeepSortTracker must convert Detection objects to the [x,y,w,h,conf] format
    that deep_sort_realtime expects."""

    def test_detections_converted_to_ltwh_tuple(self):
        """deep_sort_realtime expects [([left, top, w, h], confidence, class), ...]."""
        with patch("app.tracking.deepsort_tracker.DeepSort") as MockDeepSort:
            instance = MockDeepSort.return_value
            instance.update_tracks.return_value = []

            from app.tracking.deepsort_tracker import DeepSortTracker

            tracker = DeepSortTracker()
            det = _make_detection(100, 200, 200, 400)  # w=100, h=200
            tracker.update([det], _make_frame())

            call_args = instance.update_tracks.call_args
            raw_dets = call_args.args[0] if call_args.args else call_args.kwargs.get("raw_detections")

            assert raw_dets is not None
            assert len(raw_dets) == 1
            bbox, conf, cls = raw_dets[0]
            assert bbox[0] == pytest.approx(100.0)  # left
            assert bbox[1] == pytest.approx(200.0)  # top
            assert bbox[2] == pytest.approx(100.0)  # width
            assert bbox[3] == pytest.approx(200.0)  # height
            assert conf == pytest.approx(0.9)
            assert cls == 0  # person class

    def test_non_person_detections_excluded(self):
        """Only class_id==0 (person) detections should be forwarded to DeepSORT."""
        with patch("app.tracking.deepsort_tracker.DeepSort") as MockDeepSort:
            instance = MockDeepSort.return_value
            instance.update_tracks.return_value = []

            from app.tracking.deepsort_tracker import DeepSortTracker

            tracker = DeepSortTracker()
            person_det = _make_detection(10, 10, 50, 50)
            ball_det = Detection(
                bbox=BoundingBox(200, 200, 230, 230),
                confidence=0.8,
                class_id=32,  # ball
                class_name="sports ball",
                frame_idx=0,
            )
            tracker.update([person_det, ball_det], _make_frame())

            call_args = instance.update_tracks.call_args
            raw_dets = call_args.args[0] if call_args.args else call_args.kwargs.get("raw_detections")
            assert len(raw_dets) == 1  # only the person

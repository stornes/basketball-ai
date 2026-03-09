"""Tests for player/ball detection."""

from unittest.mock import MagicMock, patch

import numpy as np

from app.pipeline.pipeline_config import PipelineConfig
from app.vision.detection_types import BoundingBox, Detection


def test_detection_types():
    bbox = BoundingBox(10, 20, 50, 80)
    assert bbox.center == (30.0, 50.0)
    assert bbox.width == 40.0
    assert bbox.height == 60.0
    assert bbox.area == 2400.0
    assert bbox.xywh == (10, 20, 40.0, 60.0)


def test_detection_dataclass():
    det = Detection(
        bbox=BoundingBox(0, 0, 100, 100),
        confidence=0.95,
        class_id=0,
        class_name="person",
        frame_idx=5,
    )
    assert det.confidence == 0.95
    assert det.class_name == "person"


@patch("ultralytics.YOLO")
def test_detector_processes_frame(mock_yolo_cls, synthetic_frame):
    """Test detector with mocked YOLO model."""
    mock_model = MagicMock()
    mock_yolo_cls.return_value = mock_model

    # Mock YOLO result with proper tensor-like objects
    mock_boxes = MagicMock()
    mock_boxes.xyxy = [MagicMock(cpu=lambda: MagicMock(numpy=lambda: np.array([100, 200, 150, 320])))]
    mock_boxes.conf = [MagicMock(cpu=lambda: MagicMock(numpy=lambda: np.array(0.9)))]
    mock_boxes.cls = [MagicMock(cpu=lambda: MagicMock(numpy=lambda: np.array(0)))]
    mock_boxes.__len__ = lambda self: 1

    mock_result = MagicMock()
    mock_result.boxes = mock_boxes
    mock_result.names = {0: "person", 32: "sports ball"}
    mock_model.predict.return_value = [mock_result]

    from app.vision.detector import PlayerBallDetector
    config = PipelineConfig(device="cpu")
    detector = PlayerBallDetector(config)

    detections = detector.detect_frame(synthetic_frame, frame_idx=0)
    assert len(detections) == 1
    assert detections[0].class_name == "person"

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
def test_detector_processes_batch(mock_yolo_cls, synthetic_frame):
    """Test detector batch detection with mocked YOLO model."""
    mock_model = MagicMock()
    mock_yolo_cls.return_value = mock_model

    def _make_batch_tensor(vals):
        """Create a mock tensor that supports both bulk .cpu().numpy() and per-index access."""
        arr = np.array(vals)
        m = MagicMock()
        m.cpu.return_value.numpy.return_value = arr
        return m

    # Mock YOLO result with batch tensor-like objects
    mock_boxes = MagicMock()
    mock_boxes.xyxy = _make_batch_tensor([[100, 200, 150, 320]])
    mock_boxes.conf = _make_batch_tensor([0.9])
    mock_boxes.cls = _make_batch_tensor([0])
    mock_boxes.__len__ = lambda self: 1

    mock_result = MagicMock()
    mock_result.boxes = mock_boxes
    mock_result.names = {0: "person", 32: "sports ball"}
    mock_model.predict.return_value = [mock_result]

    from app.vision.detector import PlayerBallDetector
    config = PipelineConfig(device="cpu")
    detector = PlayerBallDetector(config)

    batch_detections = detector.detect_batch([synthetic_frame], [0])
    assert len(batch_detections) == 1
    assert len(batch_detections[0]) == 1
    assert batch_detections[0][0].class_name == "person"

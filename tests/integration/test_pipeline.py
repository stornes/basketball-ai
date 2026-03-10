"""Integration tests for the full pipeline on synthetic video."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from app.pipeline.pipeline_config import PipelineConfig


@patch("ultralytics.YOLO")
def test_pipeline_runs_on_synthetic_video(mock_yolo_cls, synthetic_video_path, tmp_path):
    """Full pipeline on synthetic video with mocked YOLO."""
    mock_model = MagicMock()
    mock_yolo_cls.return_value = mock_model

    def _make_tensor(val):
        m = MagicMock()
        m.cpu.return_value.numpy.return_value = np.array(val)
        return m

    def mock_predict(batch_inputs, **kwargs):
        """Return one result per input frame, matching YOLO batch behavior."""
        num_frames = len(batch_inputs) if isinstance(batch_inputs, list) else 1
        results = []
        for _ in range(num_frames):
            mock_result = MagicMock()
            mock_boxes = MagicMock()

            mock_boxes.xyxy = [
                _make_tensor([100, 200, 150, 320]),
                _make_tensor([400, 180, 450, 300]),
                _make_tensor([265, 135, 295, 165]),
            ]
            mock_boxes.conf = [
                _make_tensor(0.9),
                _make_tensor(0.85),
                _make_tensor(0.7),
            ]
            mock_boxes.cls = [
                _make_tensor(0),
                _make_tensor(0),
                _make_tensor(32),
            ]
            mock_boxes.__len__ = lambda self: 3

            mock_result.boxes = mock_boxes
            mock_result.names = {0: "person", 32: "sports ball"}
            results.append(mock_result)
        return results

    mock_model.predict = mock_predict

    config = PipelineConfig(
        device="cpu",
        frame_sample_rate=5,
        output_dir=str(tmp_path / "output"),
        enable_clips=False,  # skip clips for speed
        enable_coaching_agent=True,
        llm_backend="template",
    )

    from app.pipeline.run_analysis import PipelineOrchestrator
    orchestrator = PipelineOrchestrator(config)
    result = orchestrator.run(synthetic_video_path)

    # Verify outputs
    assert result is not None
    assert Path(result.stats_path).exists()
    assert Path(result.possessions_path).exists()
    assert Path(result.chart_path).exists()
    assert Path(result.report_path).exists()

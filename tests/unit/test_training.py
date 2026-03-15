"""Tests for training module."""

import tempfile
from pathlib import Path

from app.training.train import TrainingConfig, find_dataset_yaml


class TestTrainingConfig:
    def test_valid_config_has_no_errors(self, tmp_path):
        data_yaml = tmp_path / "data.yaml"
        data_yaml.write_text("names:\n  0: ball\n  1: player\n")
        config = TrainingConfig(data_yaml=str(data_yaml), device="cpu")
        errors = config.validate()
        assert errors == []

    def test_missing_data_yaml_error(self):
        config = TrainingConfig(data_yaml="/nonexistent/data.yaml", device="cpu")
        errors = config.validate()
        assert any("not found" in e for e in errors)

    def test_invalid_epochs_error(self, tmp_path):
        data_yaml = tmp_path / "data.yaml"
        data_yaml.write_text("names:\n  0: ball\n")
        config = TrainingConfig(data_yaml=str(data_yaml), epochs=0, device="cpu")
        errors = config.validate()
        assert any("Epochs" in e for e in errors)

    def test_invalid_batch_size_error(self, tmp_path):
        data_yaml = tmp_path / "data.yaml"
        data_yaml.write_text("names:\n  0: ball\n")
        config = TrainingConfig(data_yaml=str(data_yaml), batch_size=-1, device="cpu")
        errors = config.validate()
        assert any("Batch size" in e for e in errors)

    def test_invalid_device_error(self, tmp_path):
        data_yaml = tmp_path / "data.yaml"
        data_yaml.write_text("names:\n  0: ball\n")
        config = TrainingConfig(data_yaml=str(data_yaml), device="tpu")
        errors = config.validate()
        assert any("device" in e.lower() for e in errors)


class TestFindDatasetYaml:
    def test_finds_direct_data_yaml(self, tmp_path):
        data_yaml = tmp_path / "data.yaml"
        data_yaml.write_text("names:\n  0: ball\n")
        result = find_dataset_yaml(str(tmp_path))
        assert result == str(data_yaml)

    def test_finds_nested_data_yaml(self, tmp_path):
        nested = tmp_path / "v1"
        nested.mkdir()
        data_yaml = nested / "data.yaml"
        data_yaml.write_text("names:\n  0: ball\n")
        result = find_dataset_yaml(str(tmp_path))
        assert result == str(data_yaml)

    def test_returns_none_when_not_found(self, tmp_path):
        result = find_dataset_yaml(str(tmp_path))
        assert result is None


class TestDetectorClassMap:
    def test_coco_defaults_no_class_map(self):
        """Verify detector uses COCO defaults when no class_map provided."""
        from app.vision.detector import PlayerBallDetector
        from app.pipeline.pipeline_config import PipelineConfig
        from unittest.mock import patch, MagicMock

        with patch("ultralytics.YOLO") as mock_yolo:
            mock_model = MagicMock()
            mock_yolo.return_value = mock_model
            config = PipelineConfig(device="cpu")
            detector = PlayerBallDetector(config)

            assert detector._class_filter == [0, 32]
            assert 0 in detector._person_ids
            assert 32 in detector._ball_ids

    def test_custom_class_map(self):
        """Verify detector uses custom class mapping for fine-tuned model."""
        from app.vision.detector import PlayerBallDetector
        from app.pipeline.pipeline_config import PipelineConfig
        from unittest.mock import patch, MagicMock

        with patch("ultralytics.YOLO") as mock_yolo:
            mock_model = MagicMock()
            mock_yolo.return_value = mock_model
            config = PipelineConfig(
                device="cpu",
                class_map={0: "ball", 1: "player", 2: "hoop"},
            )
            detector = PlayerBallDetector(config)

            assert set(detector._class_filter) == {0, 1, 2}
            assert 1 in detector._person_ids
            assert 0 in detector._ball_ids
            # Normalize: model class 1 → COCO person (0)
            assert detector._normalize_class_id(1) == 0
            # Normalize: model class 0 → COCO ball (32)
            assert detector._normalize_class_id(0) == 32

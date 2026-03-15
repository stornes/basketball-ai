"""Fine-tune YOLOv8 on a basketball-specific dataset."""

from dataclasses import dataclass, field
from pathlib import Path

from app.pipeline.pipeline_config import detect_device


@dataclass
class TrainingConfig:
    """Configuration for YOLO fine-tuning."""

    data_yaml: str = "data/datasets/basketball-yolo-dataset/data.yaml"
    base_model: str = "yolov8n.pt"
    epochs: int = 50
    batch_size: int = 16
    img_size: int = 640
    device: str = field(default_factory=detect_device)
    project: str = "runs/basketball"
    name: str = "finetune"
    patience: int = 10
    lr0: float = 0.01
    freeze: int = 0  # number of backbone layers to freeze (0 = train all)

    def validate(self) -> list[str]:
        """Validate training config, return list of errors."""
        errors = []
        data_path = Path(self.data_yaml)
        if not data_path.exists():
            errors.append(f"Dataset config not found: {self.data_yaml}")
        if self.epochs < 1:
            errors.append(f"Epochs must be >= 1, got {self.epochs}")
        if self.batch_size < 1:
            errors.append(f"Batch size must be >= 1, got {self.batch_size}")
        if self.img_size < 32:
            errors.append(f"Image size must be >= 32, got {self.img_size}")
        if self.device not in ("mps", "cuda", "cpu"):
            errors.append(f"Invalid device: {self.device}")
        return errors


def find_dataset_yaml(dataset_dir: str) -> str | None:
    """Find data.yaml in a dataset directory."""
    base = Path(dataset_dir)
    if (base / "data.yaml").exists():
        return str(base / "data.yaml")
    # Check one level deep
    for child in base.iterdir():
        if child.is_dir() and (child / "data.yaml").exists():
            return str(child / "data.yaml")
    return None


def train(config: TrainingConfig) -> Path:
    """Run YOLO fine-tuning.

    Args:
        config: Training configuration.

    Returns:
        Path to best weights file.
    """
    from ultralytics import YOLO

    errors = config.validate()
    if errors:
        raise ValueError("Training config errors:\n" + "\n".join(f"  - {e}" for e in errors))

    print(f"Fine-tuning {config.base_model} on {config.data_yaml}")
    print(f"Device: {config.device} | Epochs: {config.epochs} | Batch: {config.batch_size}")

    model = YOLO(config.base_model)

    train_kwargs = dict(
        data=config.data_yaml,
        epochs=config.epochs,
        batch=config.batch_size,
        imgsz=config.img_size,
        device=config.device,
        project=config.project,
        name=config.name,
        patience=config.patience,
        lr0=config.lr0,
        freeze=config.freeze,
        exist_ok=True,
        verbose=True,
    )

    try:
        results = model.train(**train_kwargs)
    except RuntimeError as e:
        if "mps" in str(e).lower() and config.device == "mps":
            print(f"MPS training failed: {e}")
            print("Falling back to CPU training...")
            train_kwargs["device"] = "cpu"
            results = model.train(**train_kwargs)
        else:
            raise

    # YOLO may nest save_dir under runs/detect/ — use the actual save_dir from results
    save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else None

    # Search candidate directories for best.pt
    candidates = []
    if save_dir:
        candidates.append(save_dir / "weights")
    candidates.append(Path(config.project) / config.name / "weights")

    for weights_dir in candidates:
        best = weights_dir / "best.pt"
        if best.exists():
            print(f"\nTraining complete! Best weights: {best}")
            return best
        if weights_dir.exists():
            for pt_file in weights_dir.glob("*.pt"):
                print(f"\nTraining complete! Weights: {pt_file}")
                return pt_file

    searched = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"No weights found in: {searched}")


if __name__ == "__main__":
    cfg = TrainingConfig()
    train(cfg)

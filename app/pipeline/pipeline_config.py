"""Central pipeline configuration and device detection."""

from dataclasses import dataclass, field
from pathlib import Path


def detect_device() -> str:
    """Detect best available compute device."""
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


@dataclass
class PipelineConfig:
    device: str = field(default_factory=detect_device)
    yolo_model: str = "yolov8n.pt"
    confidence_threshold: float = 0.35
    iou_threshold: float = 0.5
    frame_sample_rate: int = 3
    batch_size: int = 8
    court_confidence_threshold: float = 0.6
    output_dir: str = "data/outputs"
    enable_clips: bool = True
    enable_coaching_agent: bool = True
    llm_backend: str = "gemini"  # "gemini" | "template"
    gemini_model: str = "gemini-2.0-flash"

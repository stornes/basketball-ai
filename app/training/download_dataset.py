"""Download basketball detection dataset from Roboflow."""

import os
from pathlib import Path

DEFAULT_WORKSPACE = "basketball-yolo-dataset"
DEFAULT_PROJECT = "basketball-yolo-dataset"
DEFAULT_VERSION = 1
DEFAULT_FORMAT = "yolov8"


def get_api_key() -> str:
    """Get Roboflow API key from environment."""
    key = os.environ.get("ROBOFLOW_API_KEY", "")
    if not key:
        raise ValueError(
            "ROBOFLOW_API_KEY not set. Get a free key at https://app.roboflow.com/settings/api\n"
            "Then set it: export ROBOFLOW_API_KEY='your_key_here' or add to .env"
        )
    return key


def download_dataset(
    output_dir: str = "data/datasets",
    workspace: str = DEFAULT_WORKSPACE,
    project: str = DEFAULT_PROJECT,
    version: int = DEFAULT_VERSION,
    model_format: str = DEFAULT_FORMAT,
) -> Path:
    """Download dataset from Roboflow Universe.

    Args:
        output_dir: Base directory for datasets.
        workspace: Roboflow workspace name.
        project: Roboflow project name.
        version: Dataset version number.
        model_format: Export format (default: yolov8).

    Returns:
        Path to the downloaded dataset directory.
    """
    from roboflow import Roboflow

    api_key = get_api_key()
    rf = Roboflow(api_key=api_key)

    print(f"Connecting to Roboflow workspace: {workspace}")
    ws = rf.workspace(workspace)
    proj = ws.project(project)

    print(f"Downloading {project} v{version} in {model_format} format...")
    dataset = proj.version(version).download(
        model_format=model_format,
        location=str(Path(output_dir) / project),
    )

    dataset_path = Path(output_dir) / project
    print(f"Dataset downloaded to: {dataset_path}")

    from app.training.train import find_dataset_yaml

    found = find_dataset_yaml(str(dataset_path))
    if found:
        print(f"Dataset config: {found}")
    else:
        print("Warning: data.yaml not found — you may need to create it manually.")

    return dataset_path


if __name__ == "__main__":
    import sys

    output = sys.argv[1] if len(sys.argv) > 1 else "data/datasets"
    download_dataset(output_dir=output)

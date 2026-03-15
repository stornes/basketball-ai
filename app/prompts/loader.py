"""Load prompt configurations from YAML files."""

import json
from pathlib import Path

import yaml


_PROMPTS_DIR = Path(__file__).parent


def load_prompts(name: str) -> dict[str, dict[str, str]]:
    """Load a prompt config file by name (without .yaml extension)."""
    path = _PROMPTS_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Prompt config not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def format_prompt(template: str, **kwargs) -> str:
    """Format a prompt template with the given variables.

    Dict/list values are auto-serialized to indented JSON.
    """
    formatted = {}
    for key, value in kwargs.items():
        if isinstance(value, (dict, list)):
            formatted[key] = json.dumps(value, indent=2)
        else:
            formatted[key] = value
    return template.format(**formatted)

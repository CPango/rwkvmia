"""Configuration loading helpers for repository-local YAML files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def repo_root() -> Path:
    """Return the repository root based on this module location."""
    return Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return an empty dict for empty files."""
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = repo_root() / config_path

    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve_repo_path(path_value: str | Path) -> Path:
    """Resolve a repository-relative path without requiring it to exist."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return repo_root() / path


def load_paths_config(path: str | Path = "configs/paths.yaml") -> dict[str, Any]:
    """Load path configuration from configs/paths.yaml."""
    return load_yaml(path)


def load_models_config(path: str | Path = "configs/models.yaml") -> dict[str, Any]:
    """Load model configuration from configs/models.yaml."""
    return load_yaml(path)

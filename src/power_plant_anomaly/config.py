"""Load YAML configuration and resolve output directories."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from power_plant_anomaly.paths import DEFAULT_CONFIG_PATH, resolve_project_path


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    path = config_path or DEFAULT_CONFIG_PATH
    if not path.exists():
        return {}
    with path.open() as handle:
        return yaml.safe_load(handle) or {}


def section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    return value if isinstance(value, dict) else {}


def data_path(config: dict[str, Any], key: str) -> Path:
    data = section(config, "data")
    if key not in data:
        msg = f"config data.{key} is not set"
        raise KeyError(msg)
    return resolve_project_path(data[key])


def output_root(config: dict[str, Any]) -> Path:
    output = section(config, "output")
    root = resolve_project_path(output.get("root", "output"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def output_subdir(config: dict[str, Any], key: str, default: str) -> Path:
    output = section(config, "output")
    path = output_root(config) / output.get(key, default)
    path.mkdir(parents=True, exist_ok=True)
    return path


def figures_dir(config: dict[str, Any]) -> Path:
    return output_subdir(config, "figures_dir", "figures")


def tables_dir(config: dict[str, Any]) -> Path:
    return output_subdir(config, "tables_dir", "tables")

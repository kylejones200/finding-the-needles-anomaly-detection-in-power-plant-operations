"""Repository root and path helpers."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"


def resolve_project_path(rel: str | Path) -> Path:
    path = Path(rel)
    return path if path.is_absolute() else PROJECT_ROOT / path

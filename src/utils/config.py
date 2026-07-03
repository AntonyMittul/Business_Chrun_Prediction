"""Project paths and config loading — single source of truth for locations.

Import this anywhere instead of hard-coding paths:

    from src.utils.config import PROJECT_ROOT, load_config
    cfg = load_config()
    df_path = PROJECT_ROOT / cfg["data"]["raw_file"]
"""
from __future__ import annotations

from pathlib import Path

import yaml

# src/utils/config.py -> parents[2] == project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


def load_config(path: Path | str = CONFIG_PATH) -> dict:
    """Load config.yaml as a dict."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

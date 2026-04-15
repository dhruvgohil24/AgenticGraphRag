# src/utils.py

import yaml
import logging
from pathlib import Path

def load_config(config_path: str = "config.yaml") -> dict:
    """Load YAML config from project root."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_logger(name: str) -> logging.Logger:
    """Return a consistently formatted logger."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  [%(levelname)s]  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)

def ensure_dirs(config: dict) -> None:
    """Create local data directories if they don't exist."""
    Path(config["paths"]["chroma_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["paths"]["data_dir"]).mkdir(parents=True, exist_ok=True)
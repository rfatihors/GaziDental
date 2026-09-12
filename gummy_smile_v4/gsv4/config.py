"""Configuration loading. All relative paths resolve against the gummy_smile_v4 root."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"


def load_config(path: Path | str | None = None) -> Dict[str, Any]:
    """Load config.yaml and attach the resolved root directory under ``_root``."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with cfg_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_root"] = cfg_path.resolve().parent.parent
    return cfg


def resolve(cfg: Dict[str, Any], relative: str | Path) -> Path:
    """Resolve a path from the config against the project root."""
    p = Path(relative)
    return p if p.is_absolute() else (cfg["_root"] / p).resolve()


def input_path(cfg: Dict[str, Any], name: str) -> Path:
    """Path of a clinical input file listed under ``inputs`` in the config."""
    return resolve(cfg, Path(cfg["paths"]["inputs"]) / cfg["inputs"][name])

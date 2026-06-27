from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common import *  # noqa: F401,F403
from src.utils.io import load_config


def load_phase1_config(config_path: str) -> dict[str, Any]:
    """Load a Phase 6 config, recursively resolving an optional ``extends``."""
    path = Path(config_path).resolve()
    child = load_config(path)
    extends = child.pop("extends", None)
    if extends is None:
        return child
    parent = load_phase1_config(str((path.parent / str(extends)).resolve()))
    merged = _deep_merge(parent, child)
    merged["_config_path"] = str(path)
    merged["_base_dir"] = str(path.parents[1])
    return merged


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


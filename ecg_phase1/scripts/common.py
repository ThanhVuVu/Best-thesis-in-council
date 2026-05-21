from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ECG_PHASE1_ROOT = Path(__file__).resolve().parents[1]
if str(ECG_PHASE1_ROOT) not in sys.path:
    sys.path.insert(0, str(ECG_PHASE1_ROOT))

from src.utils.io import load_config, resolve_path


def config_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase1.yaml")
    return parser


def load_phase1_config(config_path: str):
    return load_config(config_path)


def cfg_path(config: dict, *keys: str) -> Path:
    value = config
    for key in keys:
        value = value[key]
    return resolve_path(value, config["_base_dir"])


def device_from_torch() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

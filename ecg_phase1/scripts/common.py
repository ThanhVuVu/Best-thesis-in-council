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
    if not torch.cuda.is_available():
        return torch.device("cpu")

    major, minor = torch.cuda.get_device_capability(0)
    arch = f"sm_{major}{minor}"
    supported_arches = torch.cuda.get_arch_list()
    if supported_arches and arch not in supported_arches:
        gpu_name = torch.cuda.get_device_name(0)
        supported = ", ".join(supported_arches)
        raise RuntimeError(
            f"CUDA device {gpu_name} has compute capability {arch}, but this PyTorch "
            f"install was not built with support for it. Supported CUDA arches: {supported}. "
            "On Kaggle, switch the accelerator to T4/P100-compatible PyTorch, or install a "
            "PyTorch wheel that supports this GPU before training."
        )

    return torch.device("cuda")

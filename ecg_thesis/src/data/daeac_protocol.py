from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.utils.io import ensure_dir


SAMPLE_TIME_KEYS = ("r_peak_time_sec", "sample_time_sec")
SAMPLE_ID_KEYS = ("r_peak_sample", "sample")


def create_daeac_after_time_split(
    source_path: str | Path,
    output_path: str | Path,
    threshold_sec: float = 300.0,
    force: bool = False,
) -> dict[str, Any]:
    source = Path(source_path)
    output = Path(output_path)
    if output.exists() and not force:
        return inspect_daeac_time_split(output, threshold_sec)
    if not source.exists():
        raise FileNotFoundError(f"DAEAC full target file not found: {source}")

    with np.load(source, allow_pickle=True) as data:
        time_key = _first_present(data, SAMPLE_TIME_KEYS)
        times = np.asarray(data[time_key], dtype=np.float64)
        sample_count = _sample_count(data)
        if len(times) != sample_count:
            raise ValueError(f"{source}: {time_key} length {len(times)} does not match sample count {sample_count}.")
        mask = times >= float(threshold_sec)
        if not bool(mask.any()):
            raise ValueError(f"{source}: no samples at or after {threshold_sec} seconds.")
        arrays = {
            key: np.asarray(data[key])[mask] if _is_sample_array(key, data[key], sample_count) else np.asarray(data[key])
            for key in data.files
        }

    ensure_dir(output.parent)
    np.savez_compressed(output, **arrays)
    return inspect_daeac_time_split(output, threshold_sec)


def inspect_daeac_time_split(path: str | Path, threshold_sec: float = 300.0) -> dict[str, Any]:
    split_path = Path(path)
    with np.load(split_path, allow_pickle=True) as data:
        time_key = _first_present(data, SAMPLE_TIME_KEYS)
        times = np.asarray(data[time_key], dtype=np.float64)
        y = np.asarray(data["y"], dtype=np.int64) if "y" in data else None
        class_names = [str(value) for value in data["class_names"].tolist()] if "class_names" in data else []
    counts = {}
    if y is not None:
        values = np.bincount(y, minlength=len(class_names) or int(y.max(initial=-1)) + 1)
        counts = {class_names[idx] if idx < len(class_names) else str(idx): int(count) for idx, count in enumerate(values)}
    return {
        "path": str(split_path),
        "samples": int(len(times)),
        "threshold_sec": float(threshold_sec),
        "time_key": time_key,
        "time_min_sec": float(times.min()) if len(times) else None,
        "time_max_sec": float(times.max()) if len(times) else None,
        "class_counts": counts,
    }


def daeac_sample_keys(path: str | Path) -> set[str]:
    with np.load(path, allow_pickle=True) as data:
        record_key = "record" if "record" in data else "record_id" if "record_id" in data else None
        sample_key = _first_present(data, SAMPLE_ID_KEYS)
        if record_key is None:
            raise KeyError(f"{path}: missing record/record_id metadata required for overlap audit.")
        records = np.asarray(data[record_key]).astype(str)
        samples = np.asarray(data[sample_key]).astype(str)
    if len(records) != len(samples):
        raise ValueError(f"{path}: record and {sample_key} lengths differ.")
    return {f"{record}::{sample}" for record, sample in zip(records, samples)}


def audit_daeac_disjoint(left_path: str | Path, right_path: str | Path) -> dict[str, Any]:
    left = daeac_sample_keys(left_path)
    right = daeac_sample_keys(right_path)
    overlap = sorted(left & right)
    return {
        "left_path": str(left_path),
        "right_path": str(right_path),
        "left_samples": len(left),
        "right_samples": len(right),
        "overlap_count": len(overlap),
        "overlap_examples": overlap[:10],
        "disjoint": not overlap,
    }


def _sample_count(data: np.lib.npyio.NpzFile) -> int:
    for key in ("x", "x_daeac", "X", "inputs", "data"):
        if key in data and np.asarray(data[key]).ndim >= 1:
            return int(len(data[key]))
    raise KeyError("Could not find a sample tensor in DAEAC NPZ.")


def _is_sample_array(key: str, value: np.ndarray, sample_count: int) -> bool:
    if key in {"class_names", "class_to_id_json", "config_json"}:
        return False
    array = np.asarray(value)
    return array.ndim >= 1 and len(array) == sample_count


def _first_present(data: np.lib.npyio.NpzFile, keys: tuple[str, ...]) -> str:
    for key in keys:
        if key in data:
            return key
    raise KeyError(f"Missing required metadata; expected one of {list(keys)}.")

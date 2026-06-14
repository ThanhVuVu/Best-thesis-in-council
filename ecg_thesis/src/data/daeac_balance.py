from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from src.data.daeac_dataset import DAEACDataset
from src.utils.io import ensure_dir


METADATA_KEYS = (
    "record",
    "record_id",
    "symbol",
    "sample",
    "r_peak_sample",
    "r_peak_time_sec",
    "fs",
    "domain",
    "lead_index",
    "lead_name",
)


def load_labeled_daeac_arrays(
    path: str | Path,
    input_key: str,
    label_key: str,
    class_names: list[str],
    max_samples: int | None = None,
    seed: int = 42,
) -> tuple[DAEACDataset, np.ndarray, np.ndarray, np.ndarray]:
    dataset = DAEACDataset(path, input_key=input_key, label_key=label_key, class_names=class_names)
    if dataset.y is None:
        raise ValueError(f"{path} does not contain labels; source balancing requires labeled source data.")
    indices = stratified_indices(dataset.y, max_samples=max_samples, seed=seed)
    return dataset, dataset.x[indices], dataset.y[indices], indices


def stratified_indices(labels: np.ndarray, max_samples: int | None, seed: int) -> np.ndarray:
    if max_samples is None or int(max_samples) >= len(labels):
        return np.arange(len(labels), dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    labels = labels.astype(np.int64)
    classes = np.unique(labels)
    per_class = max(1, int(max_samples) // max(len(classes), 1))
    selected: list[np.ndarray] = []
    for cls in classes:
        cls_idx = np.flatnonzero(labels == cls)
        take = min(len(cls_idx), per_class)
        selected.append(rng.choice(cls_idx, size=take, replace=False))
    merged = np.concatenate(selected) if selected else np.zeros(0, dtype=np.int64)
    if len(merged) < int(max_samples):
        remaining = np.setdiff1d(np.arange(len(labels), dtype=np.int64), merged, assume_unique=False)
        take = min(len(remaining), int(max_samples) - len(merged))
        if take > 0:
            merged = np.concatenate([merged, rng.choice(remaining, size=take, replace=False)])
    rng.shuffle(merged)
    return merged.astype(np.int64)


def random_oversample_indices(labels: np.ndarray, multipliers: dict[int, int], seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    labels = labels.astype(np.int64)
    selected: list[np.ndarray] = []
    for cls in sorted(set(int(v) for v in labels.tolist())):
        cls_idx = np.flatnonzero(labels == cls)
        multiplier = int(multipliers.get(cls, 1))
        if multiplier < 1:
            raise ValueError(f"Multiplier for class {cls} must be >= 1, got {multiplier}.")
        selected.extend([cls_idx.copy() for _ in range(multiplier)])
    merged = np.concatenate(selected) if selected else np.zeros(0, dtype=np.int64)
    rng.shuffle(merged)
    return merged.astype(np.int64)


def save_daeac_npz_with_metadata(
    output_path: str | Path,
    source_dataset: DAEACDataset,
    selected_source_indices: np.ndarray | None,
    x: np.ndarray,
    y: np.ndarray,
    class_names: list[str],
    config_json: dict[str, Any],
    input_key: str = "x_daeac",
    is_synthetic: np.ndarray | None = None,
) -> None:
    output = Path(output_path)
    ensure_dir(output.parent)
    payload: dict[str, Any] = {
        input_key: x.astype(np.float32),
        "y": y.astype(np.int64),
        "class_names": np.asarray(class_names, dtype=object),
        "config_json": np.asarray(json.dumps(config_json, sort_keys=True), dtype=object),
    }
    if is_synthetic is not None:
        payload["is_synthetic"] = is_synthetic.astype(bool)
    if selected_source_indices is not None:
        for key in METADATA_KEYS:
            if key in source_dataset.data:
                values = source_dataset.data[key]
                if len(values) == len(source_dataset.x):
                    payload[key] = values[selected_source_indices]
    np.savez_compressed(output, **payload)


def counts_by_name(labels: np.ndarray, class_names: list[str]) -> dict[str, int]:
    counts = Counter(int(v) for v in labels.astype(np.int64).tolist())
    return {name: int(counts.get(idx, 0)) for idx, name in enumerate(class_names)}


def class_name_to_id(class_names: list[str]) -> dict[str, int]:
    return {name: idx for idx, name in enumerate(class_names)}

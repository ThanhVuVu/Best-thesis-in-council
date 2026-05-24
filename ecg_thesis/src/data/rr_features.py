from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

RR_FEATURE_NAMES = np.asarray(["rr_prev", "rr_next", "rr_ratio", "rr_prev_next_ratio"])
EPS = 1e-6


def compute_rr_features(records: np.ndarray, samples: np.ndarray, fs_values: np.ndarray) -> np.ndarray:
    records = np.asarray(records)
    samples = np.asarray(samples).astype(np.float64)
    fs_values = np.asarray(fs_values).astype(np.float64)
    rr = np.zeros((len(samples), 4), dtype=np.float32)

    for record in np.unique(records.astype(str)):
        idx = np.where(records.astype(str) == record)[0]
        if len(idx) == 0:
            continue
        order = idx[np.argsort(samples[idx], kind="stable")]
        rec_samples = samples[order]
        rec_fs = fs_values[order]
        diffs = np.diff(rec_samples) / np.maximum(rec_fs[1:], EPS)
        median_rr = float(np.median(diffs)) if len(diffs) else 1.0
        median_rr = max(median_rr, EPS)

        rr_prev = np.empty(len(order), dtype=np.float64)
        rr_next = np.empty(len(order), dtype=np.float64)
        rr_prev[0] = median_rr
        rr_next[-1] = median_rr
        if len(order) > 1:
            rr_prev[1:] = diffs
            rr_next[:-1] = diffs
        rr_next_safe = np.maximum(rr_next, EPS)
        values = np.stack(
            [
                rr_prev,
                rr_next,
                rr_prev / median_rr,
                rr_prev / rr_next_safe,
            ],
            axis=1,
        )
        rr[order] = values.astype(np.float32)
    return rr


def fit_rr_normalizer(rr_features: np.ndarray) -> dict[str, list[float]]:
    rr = np.asarray(rr_features, dtype=np.float32)
    mean = rr.mean(axis=0)
    std = rr.std(axis=0)
    std = np.where(std < EPS, 1.0, std)
    return {"mean": mean.tolist(), "std": std.tolist(), "feature_names": RR_FEATURE_NAMES.tolist()}


def apply_rr_normalizer(rr_features: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    rr = np.asarray(rr_features, dtype=np.float32)
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    return ((rr - mean) / np.maximum(std, EPS)).astype(np.float32)


def add_rr_features_to_npz(
    input_path: str | Path,
    output_path: str | Path | None = None,
    normalizer: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any] | None]:
    input_path = Path(input_path)
    output_path = Path(output_path) if output_path is not None else input_path
    data = np.load(input_path, allow_pickle=True)
    required = {"record", "sample", "fs"}
    missing = sorted(required - set(data.files))
    if missing:
        raise KeyError(f"{input_path} is missing keys required for RR features: {missing}")

    raw_rr = compute_rr_features(data["record"], data["sample"], data["fs"])
    rr = apply_rr_normalizer(raw_rr, normalizer) if normalizer is not None else raw_rr
    payload = {key: data[key] for key in data.files}
    payload["rr_features"] = rr.astype(np.float32)
    payload["rr_feature_names"] = RR_FEATURE_NAMES
    np.savez_compressed(output_path, **payload)
    return raw_rr, normalizer

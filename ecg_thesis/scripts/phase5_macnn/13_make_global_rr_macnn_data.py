from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from common import cfg_path, load_phase1_config
from src.utils.io import ensure_dir, write_json

EPS = 1e-6
FEATURE_NAMES = np.asarray(
    ["rr_global", "rr_global_sq", "rr_global_exp", "rr_global_exp_minus_1", "rr_global_log"],
    dtype=object,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase5_macnn_daeac.yaml")
    parser.add_argument("--suffix", default="globalrr")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_phase1_config(args.config)
    output_dir = ensure_dir(cfg_path(config, "paths", "output_dir"))
    data_cfg = config["data"]
    keys = ["source_train", "source_test", "target_unlabeled", "target_test"]
    summary: dict[str, Any] = {}

    input_paths = {key: cfg_path(config, "data", key) for key in keys}
    reference = _load_reference_timing(input_paths.values())

    for key in keys:
        src = input_paths[key]
        dst = _output_path(src, str(args.suffix))
        summary[key] = make_global_rr_file(src, dst, force=bool(args.force), reference=reference)
        print(key, summary[key])

    write_json(summary, output_dir / "metrics" / f"macnn_global_rr_data_{args.suffix}_summary.json")


def make_global_rr_file(
    input_path: str | Path,
    output_path: str | Path,
    force: bool = False,
    reference: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> dict[str, Any]:
    input_path = Path(input_path)
    output_path = Path(output_path)
    if output_path.exists() and not force:
        return {"input": str(input_path), "output": str(output_path), "skipped": True, "reason": "output exists"}

    data = np.load(input_path, allow_pickle=True)
    required = {"x_macnn", "y", "record", "sample", "fs"}
    missing = sorted(required - set(data.files))
    if missing:
        raise KeyError(f"{input_path} is missing required keys for global RR conversion: {missing}")

    x = data["x_macnn"].astype(np.float32)
    if x.ndim != 4 or x.shape[1] != 1 or x.shape[2] < 1 or x.shape[3] != 128:
        raise ValueError(f"Expected x_macnn [N, 1, H, 128], got {x.shape} in {input_path}")

    rr_features = compute_global_rr_features(data["record"], data["sample"], data["fs"], reference=reference)
    tiled_rr = np.repeat(rr_features[:, :, None], x.shape[-1], axis=2)
    morphology = x[:, 0, 0:1, :]
    x_global_rr = np.concatenate([morphology, tiled_rr], axis=1)[:, None, :, :].astype(np.float32)

    payload = {name: data[name] for name in data.files if name != "x_macnn"}
    payload["x_macnn"] = x_global_rr
    payload["rr_global_features"] = rr_features.astype(np.float32)
    payload["rr_global_feature_names"] = FEATURE_NAMES
    payload["config_json"] = _updated_config_json(data, input_path)

    ensure_dir(output_path.parent)
    np.savez_compressed(output_path, **payload)
    return {
        "input": str(input_path),
        "output": str(output_path),
        "skipped": False,
        "x_macnn_shape": list(x_global_rr.shape),
        "feature_names": ["morphology", *FEATURE_NAMES.tolist()],
    }


def compute_global_rr_features(
    records: np.ndarray,
    samples: np.ndarray,
    fs_values: np.ndarray,
    reference: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    records = np.asarray(records)
    samples = np.asarray(samples).astype(np.float64)
    fs_values = np.asarray(fs_values).astype(np.float64)
    ref_records, ref_samples, ref_fs_values = reference if reference is not None else (records, samples, fs_values)
    ref_records = np.asarray(ref_records)
    ref_samples = np.asarray(ref_samples).astype(np.float64)
    ref_fs_values = np.asarray(ref_fs_values).astype(np.float64)
    mean_rr_by_record = _mean_rr_by_record(ref_records, ref_samples, ref_fs_values)
    features = np.zeros((len(samples), len(FEATURE_NAMES)), dtype=np.float32)

    for rec in np.unique(records.astype(str)):
        indices = np.flatnonzero(records.astype(str) == rec)
        order = indices[np.argsort(samples[indices])]
        rec_samples = samples[order]
        rec_fs = float(np.median(fs_values[order])) if len(order) else 1.0
        mean_rr = max(float(mean_rr_by_record.get(str(rec), 1.0)), EPS)
        rr_prev = np.empty(len(order), dtype=np.float64)
        rr_prev[0] = mean_rr
        if len(order) > 1:
            rr_prev[1:] = np.diff(rec_samples) / max(rec_fs, EPS)
        rr_global = np.maximum(rr_prev / mean_rr, EPS)
        rr_clipped = np.clip(rr_global, EPS, 10.0)
        values = np.stack(
            [
                rr_global,
                rr_global**2,
                np.exp(np.clip(rr_global, -10.0, 10.0)),
                np.exp(np.clip(rr_global, -10.0, 10.0)) - 1.0,
                np.log(rr_clipped),
            ],
            axis=1,
        )
        features[order] = values.astype(np.float32)
    return features


def _load_reference_timing(paths) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    records, samples, fs_values = [], [], []
    for path in paths:
        data = np.load(path, allow_pickle=True)
        for name in ("record", "sample", "fs"):
            if name not in data.files:
                raise KeyError(f"{path} is missing required key for global RR reference: {name}")
        records.append(data["record"])
        samples.append(data["sample"])
        fs_values.append(data["fs"])
    return np.concatenate(records), np.concatenate(samples), np.concatenate(fs_values)


def _mean_rr_by_record(records: np.ndarray, samples: np.ndarray, fs_values: np.ndarray) -> dict[str, float]:
    means: dict[str, float] = {}
    record_strings = records.astype(str)
    for rec in np.unique(record_strings):
        indices = np.flatnonzero(record_strings == rec)
        order = indices[np.argsort(samples[indices])]
        if len(order) < 2:
            means[str(rec)] = 1.0
            continue
        rec_fs = float(np.median(fs_values[order]))
        diffs = np.diff(samples[order].astype(np.float64)) / max(rec_fs, EPS)
        means[str(rec)] = max(float(np.mean(diffs)), EPS)
    return means


def _output_path(input_path: Path, suffix: str) -> Path:
    if input_path.name.endswith("_macnn.npz"):
        return input_path.with_name(input_path.name.replace("_macnn.npz", f"_{suffix}_macnn.npz"))
    return input_path.with_name(f"{input_path.stem}_{suffix}{input_path.suffix}")


def _updated_config_json(data: np.lib.npyio.NpzFile, input_path: Path) -> np.ndarray:
    previous: dict[str, Any] = {}
    if "config_json" in data.files:
        try:
            previous = json.loads(str(data["config_json"].item()))
        except Exception:
            previous = {"previous_config_json": str(data["config_json"])}
    previous["global_rr_variant"] = {
        "source_file": str(input_path),
        "feature_names": FEATURE_NAMES.tolist(),
        "note": "morphology row plus five global average RR rows from Ran et al. MBE 2024 section 2.4",
    }
    return np.asarray(json.dumps(previous, sort_keys=True), dtype=object)


if __name__ == "__main__":
    main()

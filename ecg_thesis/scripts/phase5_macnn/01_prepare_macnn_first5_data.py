from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common import cfg_path, load_phase1_config
from src.data.macnn_preprocess import audit_first5_split, preprocess_macnn_records
from src.data.physionet import discover_records
from src.data.splits import MITBIH_TEST_RECORDS, MITBIH_TRAIN_RECORDS
from src.utils.io import ensure_dir, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase5_macnn_daeac.yaml")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-macnn", action="store_true")
    parser.add_argument("--skip-catnet-split", action="store_true")
    args = parser.parse_args()
    config = load_phase1_config(args.config)

    processed_dir = ensure_dir(cfg_path(config, "paths", "processed_dir"))
    metrics_dir = ensure_dir(cfg_path(config, "paths", "output_dir") / "metrics")
    summaries = {}
    if not args.skip_macnn:
        mit_dir = cfg_path(config, "paths", "mitbih_raw_dir")
        inc_dir = cfg_path(config, "paths", "incart_raw_dir")
        incart_records = discover_records(inc_dir)
        summaries["mitbih_train"] = preprocess_macnn_records(
            mit_dir, MITBIH_TRAIN_RECORDS, cfg_path(config, "data", "source_train"), "mitbih", config, "all", args.force
        )
        summaries["mitbih_test"] = preprocess_macnn_records(
            mit_dir, MITBIH_TEST_RECORDS, cfg_path(config, "data", "source_test"), "mitbih", config, "all", args.force
        )
        summaries["incart_first5_unlabeled"] = preprocess_macnn_records(
            inc_dir, incart_records, cfg_path(config, "data", "target_unlabeled"), "incart", config, "first5", args.force
        )
        summaries["incart_after5_heldout"] = preprocess_macnn_records(
            inc_dir, incart_records, cfg_path(config, "data", "target_test"), "incart", config, "after5", args.force
        )
        summaries["first5_audit"] = audit_first5_split(
            cfg_path(config, "data", "target_unlabeled"),
            cfg_path(config, "data", "target_test"),
            float(config["data"]["target_adapt_seconds"]),
        )

    if not args.skip_catnet_split:
        incart_test = cfg_path(config, "catnet_first5", "source_test").parent / "incart_test.npz"
        summaries["catnet_first5_split"] = split_catnet_incart(
            incart_test,
            cfg_path(config, "catnet_first5", "target_unlabeled"),
            cfg_path(config, "catnet_first5", "target_test"),
            threshold_sec=float(config["data"]["target_adapt_seconds"]),
            force=args.force,
        )
    write_json(summaries, metrics_dir / "phase5_macnn_preprocess_summary.json")
    print(summaries)


def split_catnet_incart(input_path: Path, adapt_path: Path, heldout_path: Path, threshold_sec: float, force: bool = False) -> dict:
    if (adapt_path.exists() and heldout_path.exists()) and not force:
        return {"skipped": True, "reason": "catnet first5 split exists", "adapt": str(adapt_path), "heldout": str(heldout_path)}
    if not input_path.exists():
        return {"skipped": True, "reason": f"missing base INCART file: {input_path}"}
    data = np.load(input_path, allow_pickle=True)
    times = data["sample"].astype(np.float64) / np.maximum(data["fs"].astype(np.float64), 1e-6)
    adapt_mask = times < threshold_sec
    heldout_mask = times >= threshold_sec
    ensure_dir(adapt_path.parent)
    _save_subset(data, adapt_mask, adapt_path)
    _save_subset(data, heldout_mask, heldout_path)
    return {
        "skipped": False,
        "input": str(input_path),
        "adapt": str(adapt_path),
        "heldout": str(heldout_path),
        "adapt_count": int(adapt_mask.sum()),
        "heldout_count": int(heldout_mask.sum()),
        "adapt_max_r_peak_time_sec": float(times[adapt_mask].max()) if adapt_mask.any() else None,
        "heldout_min_r_peak_time_sec": float(times[heldout_mask].min()) if heldout_mask.any() else None,
    }


def _save_subset(data, mask: np.ndarray, output_path: Path) -> None:
    payload = {}
    n = len(mask)
    for key in data.files:
        value = data[key]
        payload[key] = value[mask] if hasattr(value, "shape") and value.shape[:1] == (n,) else value
    np.savez_compressed(output_path, **payload)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common import cfg_path, load_phase1_config
from src.utils.io import ensure_dir, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase2_dann.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_phase1_config(args.config)

    source_path = cfg_path(config, "data", "target_full")
    unlabeled_path = cfg_path(config, "data", "target_unlabeled")
    test_path = cfg_path(config, "data", "target_test")
    metrics_dir = ensure_dir(cfg_path(config, "paths", "output_dir") / "metrics")

    if unlabeled_path.exists() and test_path.exists() and not args.force:
        print(f"Split files already exist: {unlabeled_path}, {test_path}")
        return

    data = np.load(source_path, allow_pickle=True)
    adapt_records = set(config["incart_split"]["adapt_records"])
    test_records = set(config["incart_split"]["test_records"])
    records = np.asarray([str(r) for r in data["record"]])

    adapt_mask = np.asarray([r in adapt_records for r in records])
    test_mask = np.asarray([r in test_records for r in records])
    overlap = sorted(adapt_records & test_records)
    if overlap:
        raise ValueError(f"Adapt/test record overlap: {overlap}")
    if np.any(adapt_mask & test_mask):
        raise ValueError("Beat-level overlap detected between adapt and test masks")

    _save_subset(data, adapt_mask, unlabeled_path)
    _save_subset(data, test_mask, test_path)
    summary = {
        "source": str(source_path),
        "target_unlabeled": str(unlabeled_path),
        "target_test": str(test_path),
        "adapt_records": sorted(adapt_records),
        "test_records": sorted(test_records),
        "adapt_beats": int(adapt_mask.sum()),
        "test_beats": int(test_mask.sum()),
        "adapt_class_counts": _class_counts(data["y"][adapt_mask]),
        "test_class_counts": _class_counts(data["y"][test_mask]),
    }
    write_json(summary, metrics_dir / "phase2_incart_split.json")
    print(summary)


def _save_subset(data: np.lib.npyio.NpzFile, mask: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    values = {}
    for key in data.files:
        arr = data[key]
        if len(arr.shape) > 0 and len(arr) == len(mask):
            values[key] = arr[mask]
        else:
            values[key] = arr
    np.savez_compressed(path, **values)


def _class_counts(y: np.ndarray) -> dict[str, int]:
    counts = np.bincount(y.astype(np.int64), minlength=3)
    return {str(i): int(v) for i, v in enumerate(counts)}


if __name__ == "__main__":
    main()

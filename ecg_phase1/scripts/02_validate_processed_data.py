from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

from common import cfg_path, load_phase1_config
from src.data.splits import MITBIH_TEST_RECORDS, MITBIH_TRAIN_RECORDS
from src.utils.io import ensure_dir, write_json


def validate_npz(path, expected_records: list[str] | None = None) -> dict:
    data = np.load(path, allow_pickle=True)
    x = data["x"]
    y = data["y"]
    if x.ndim != 3 or x.shape[1] != 1 or x.shape[2] != 250:
        raise AssertionError(f"{path}: bad x shape {x.shape}")
    if x.dtype != np.float32:
        raise AssertionError(f"{path}: bad x dtype {x.dtype}")
    if y.dtype != np.int64:
        raise AssertionError(f"{path}: bad y dtype {y.dtype}")
    if set(np.unique(y).tolist()) - {0, 1, 2}:
        raise AssertionError(f"{path}: invalid labels {np.unique(y)}")
    if not np.isfinite(x).all():
        raise AssertionError(f"{path}: x contains NaN or Inf")
    records = set(str(r) for r in data["record"])
    if expected_records is not None:
        missing = sorted(set(expected_records) - records)
        extra = sorted(records - set(expected_records))
    else:
        missing = []
        extra = []
    return {
        "path": str(path),
        "x_shape": list(x.shape),
        "y_shape": list(y.shape),
        "class_counts": {str(k): int(v) for k, v in Counter(y.tolist()).items()},
        "symbol_counts": {str(k): int(v) for k, v in Counter(data["symbol"].tolist()).items()},
        "record_count": len(records),
        "missing_expected_records": missing,
        "extra_records": extra,
        "lead_counts": {str(k): int(v) for k, v in Counter(data["lead_name"].tolist()).items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase1.yaml")
    args = parser.parse_args()
    config = load_phase1_config(args.config)
    processed = cfg_path(config, "paths", "processed_dir")
    metrics_dir = ensure_dir(cfg_path(config, "paths", "output_dir") / "metrics")

    summaries = {
        "mitbih_train": validate_npz(processed / "mitbih_train.npz", MITBIH_TRAIN_RECORDS),
        "mitbih_test": validate_npz(processed / "mitbih_test.npz", MITBIH_TEST_RECORDS),
        "incart_test": validate_npz(processed / "incart_test.npz"),
    }

    train_records = set(np.load(processed / "mitbih_train.npz", allow_pickle=True)["record"].astype(str))
    test_records = set(np.load(processed / "mitbih_test.npz", allow_pickle=True)["record"].astype(str))
    overlap = sorted(train_records & test_records)
    if overlap:
        raise AssertionError(f"MIT-BIH train/test overlap: {overlap}")
    summaries["mitbih_train_test_overlap"] = overlap

    write_json(summaries, metrics_dir / "processed_validation.json")
    print("Processed data validation passed.")
    for name, summary in summaries.items():
        print(name, summary)


if __name__ == "__main__":
    main()

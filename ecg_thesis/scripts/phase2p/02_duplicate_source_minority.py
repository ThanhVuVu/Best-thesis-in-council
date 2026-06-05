from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import cfg_path, load_phase1_config
from src.data.splits import mitbih_fit_val_records
from src.utils.io import ensure_dir, write_json
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase2p_catnet_paper_uda.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_phase1_config(args.config)
    set_seed(int(config["seed"]))
    src = cfg_path(config, "data", "source_train")
    dst = cfg_path(config, "data", "source_train_duplicated")
    output = ensure_dir(cfg_path(config, "paths", "output_dir"))
    if dst.exists() and not args.force:
        print(f"Duplicated source file exists, skipping: {dst}")
        return

    data = np.load(src, allow_pickle=True)
    y = data["y"].astype(np.int64)
    records = np.asarray([str(r) for r in data["record"]])
    fit_records, val_records = mitbih_fit_val_records()
    fit_mask = np.asarray([rec in set(fit_records) for rec in records])
    class_names = [str(v) for v in data["class_names"].tolist()]
    additional = config["duplication"]["additional_copies"]
    factors = {name: 1 + int(additional.get(name, 0)) for name in class_names}

    indices = []
    for idx, label in enumerate(y):
        if not fit_mask[idx]:
            continue
        cls_name = class_names[int(label)]
        indices.extend([idx] * int(factors[cls_name]))
    indices = np.asarray(indices, dtype=np.int64)
    payload = {}
    n = len(y)
    for key in data.files:
        value = data[key]
        if hasattr(value, "shape") and value.shape[:1] == (n,):
            payload[key] = value[indices]
        else:
            payload[key] = value
    previous = {}
    try:
        previous = json.loads(str(data["config_json"].item()))
    except Exception:
        previous = {}
    previous["phase2p_duplication"] = {
        "source": str(src),
        "fit_records": fit_records,
        "val_records_not_included": val_records,
        "additional_copies": config["duplication"]["additional_copies"],
        "total_multipliers": factors,
        "no_target_labels_used": True,
    }
    payload["config_json"] = np.asarray(json.dumps(previous, sort_keys=True), dtype=object)
    ensure_dir(dst.parent)
    np.savez_compressed(dst, **payload)

    summary = {
        "source": str(src),
        "output": str(dst),
        "fit_records": fit_records,
        "val_records_not_included": val_records,
        "before_fit_counts": _named_counts(y[fit_mask], class_names),
        "after_duplication_counts": _named_counts(payload["y"], class_names),
        "total_multipliers": factors,
        "x_shape": list(payload["x"].shape),
        "no_target_labels_used": True,
    }
    write_json(summary, output / "metrics" / "phase2p_source_duplication_summary.json")
    print(summary)


def _named_counts(y: np.ndarray, class_names: list[str]) -> dict[str, int]:
    counts = np.bincount(y.astype(np.int64), minlength=len(class_names))
    return {name: int(counts[idx]) for idx, name in enumerate(class_names)}


if __name__ == "__main__":
    main()

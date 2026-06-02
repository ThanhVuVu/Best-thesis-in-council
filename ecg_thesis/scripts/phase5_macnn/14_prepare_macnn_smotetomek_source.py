from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from common import cfg_path, load_phase1_config
from src.data.datasets import ECGMACNNDataset, subset_by_records
from src.data.splits import mitbih_fit_val_records
from src.utils.io import ensure_dir, write_json
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase5_macnn_daeac.yaml")
    parser.add_argument("--output", default="data/processed/phase5_macnn/mitbih_fit_macnn_smotetomek_cap_n12000_s4000_v6000.npz")
    parser.add_argument("--target-n", type=int, default=12000)
    parser.add_argument("--target-s", type=int, default=4000)
    parser.add_argument("--target-v", type=int, default=6000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    try:
        from imblearn.combine import SMOTETomek
        from imblearn.over_sampling import SMOTE
        from imblearn.under_sampling import RandomUnderSampler, TomekLinks
    except ImportError as exc:
        raise ImportError("SMOTE-Tomek ablation requires imbalanced-learn. Run: pip install imbalanced-learn") from exc

    config = load_phase1_config(args.config)
    set_seed(int(config["seed"]))
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path(config["_base_dir"]) / output_path
    if output_path.exists() and not args.force:
        print(f"SMOTE-Tomek source file exists, skipping: {output_path}")
        return

    full_train = ECGMACNNDataset(cfg_path(config, "data", "source_train"))
    fit_records, val_records = mitbih_fit_val_records()
    fit_ds = subset_by_records(full_train, fit_records)
    fit_indices = np.asarray(fit_ds.indices, dtype=np.int64)
    x_fit = full_train.x[fit_indices]
    y_fit = full_train.y[fit_indices]
    original_counts = _counts(y_fit)

    targets = {
        0: min(int(args.target_n), int(original_counts.get(0, 0))),
        1: int(args.target_s),
        2: int(args.target_v),
    }
    for cls, target in targets.items():
        if target <= 0:
            raise ValueError(f"Target count for class {cls} must be positive, got {target}")
    if targets[1] <= original_counts.get(1, 0) or targets[2] <= original_counts.get(2, 0):
        raise ValueError(
            "This capped ablation expects S and V targets above original fit counts. "
            f"Original={original_counts}, targets={targets}"
        )

    x_flat = x_fit.reshape(len(x_fit), -1)
    rus_strategy = {
        0: targets[0],
        1: int(original_counts[1]),
        2: int(original_counts[2]),
    }
    rus = RandomUnderSampler(sampling_strategy=rus_strategy, random_state=int(config["seed"]))
    x_rus, y_rus = rus.fit_resample(x_flat, y_fit)

    smote = SMOTE(sampling_strategy=targets, random_state=int(config["seed"]), k_neighbors=5)
    tomek = TomekLinks(sampling_strategy="all")
    smotetomek = SMOTETomek(
        sampling_strategy=targets,
        random_state=int(config["seed"]),
        smote=smote,
        tomek=tomek,
    )
    x_resampled, y_resampled = smotetomek.fit_resample(x_rus, y_rus)
    x_resampled = x_resampled.reshape(-1, *x_fit.shape[1:]).astype(np.float32)
    y_resampled = y_resampled.astype(np.int64)

    ensure_dir(output_path.parent)
    config_json = {
        "method": "RandomUnderSampler + SMOTE-Tomek",
        "source": str(cfg_path(config, "data", "source_train")),
        "fit_records": fit_records,
        "val_records_not_used": val_records,
        "targets": {"N": targets[0], "S": targets[1], "V": targets[2]},
        "note": "Resampling is applied only to MIT-BIH fit records. No target-domain labels are used.",
    }
    np.savez_compressed(
        output_path,
        x_macnn=x_resampled,
        y=y_resampled,
        class_names=np.asarray(config["data"]["class_names"], dtype=object),
        config_json=np.asarray(json.dumps(config_json, sort_keys=True), dtype=object),
    )

    summary = {
        "output": str(output_path),
        "input": str(cfg_path(config, "data", "source_train")),
        "fit_records": fit_records,
        "val_records_not_used": val_records,
        "original_fit_counts": _named_counts(original_counts, config["data"]["class_names"]),
        "after_rus_counts": _named_counts(_counts(y_rus), config["data"]["class_names"]),
        "after_smotetomek_counts": _named_counts(_counts(y_resampled), config["data"]["class_names"]),
        "target_counts_before_tomek": {name: int(targets[idx]) for idx, name in enumerate(config["data"]["class_names"])},
        "x_macnn_shape": list(x_resampled.shape),
        "no_target_labels_used": True,
    }
    output_dir = ensure_dir(cfg_path(config, "paths", "output_dir"))
    write_json(summary, output_dir / "metrics" / f"{output_path.stem}_summary.json")
    print(summary)


def _counts(y: np.ndarray) -> dict[int, int]:
    counts = Counter(int(v) for v in y)
    return {idx: int(counts.get(idx, 0)) for idx in range(3)}


def _named_counts(counts: dict[int, int], class_names: list[str]) -> dict[str, int]:
    return {name: int(counts.get(idx, 0)) for idx, name in enumerate(class_names)}


if __name__ == "__main__":
    main()

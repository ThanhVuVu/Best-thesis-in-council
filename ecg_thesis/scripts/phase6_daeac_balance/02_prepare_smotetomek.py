from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common import cfg_path, load_phase1_config
from src.data.daeac_balance import (
    class_name_to_id,
    counts_by_name,
    load_labeled_daeac_arrays,
    save_daeac_npz_with_metadata,
)
from src.utils.io import ensure_dir, resolve_path, write_json
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase6_daeac_hybrid_mkmmd_smotetomek.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--target-count", type=int, default=None)
    parser.add_argument("--max-source-samples", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    try:
        from imblearn.combine import SMOTETomek
        from imblearn.over_sampling import SMOTE
        from imblearn.under_sampling import RandomUnderSampler, TomekLinks
    except ImportError as exc:
        raise ImportError("SMOTE-Tomek balancing requires imbalanced-learn. Run: pip install imbalanced-learn") from exc

    config = load_phase1_config(args.config)
    set_seed(int(config["seed"]))
    class_names = list(config["data"]["class_names"])
    input_key = str(config["data"].get("input_key", "auto"))
    label_key = str(config["data"].get("label_key", "y"))
    balance_cfg = dict(config["balance"]["smotetomek"])
    target_count = int(args.target_count or balance_cfg["target_count"])
    source_input = _source_input(config)
    output_path = _output_path(args.output, balance_cfg["output"], config)
    if output_path.exists() and not args.force:
        print(f"SMOTE-Tomek source exists, skipping: {output_path}")
        return

    dataset, x_source, y_source, _source_indices = load_labeled_daeac_arrays(
        source_input,
        input_key=input_key,
        label_key=label_key,
        class_names=class_names,
        max_samples=args.max_source_samples,
        seed=int(config["seed"]),
    )
    name_to_id = class_name_to_id(class_names)
    majority_id = name_to_id[str(balance_cfg.get("majority_class", "N"))]
    before_counts = _counts(y_source, len(class_names))
    _validate_smotetomek_inputs(before_counts, target_count, class_names)

    x_flat = x_source.reshape(len(x_source), -1)
    rus_strategy = {
        cls: min(int(before_counts[cls]), target_count) if cls == majority_id else int(before_counts[cls])
        for cls in range(len(class_names))
    }
    rus = RandomUnderSampler(sampling_strategy=rus_strategy, random_state=int(config["seed"]))
    x_rus, y_rus = rus.fit_resample(x_flat, y_source)
    after_rus_counts = _counts(y_rus, len(class_names))

    targets = {cls: max(target_count, int(after_rus_counts[cls])) for cls in range(len(class_names))}
    k_neighbors = _safe_k_neighbors(after_rus_counts, requested=int(balance_cfg.get("k_neighbors", 5)))
    smote = SMOTE(sampling_strategy=targets, random_state=int(config["seed"]), k_neighbors=k_neighbors)
    tomek = TomekLinks(sampling_strategy="all")
    smotetomek = SMOTETomek(
        sampling_strategy=targets,
        random_state=int(config["seed"]),
        smote=smote,
        tomek=tomek,
    )
    x_resampled, y_resampled = smotetomek.fit_resample(x_rus, y_rus)
    x_resampled = x_resampled.reshape(-1, *x_source.shape[1:]).astype(np.float32)
    y_resampled = y_resampled.astype(np.int64)
    after_counts = _counts(y_resampled, len(class_names))

    config_json = {
        "method": "RandomUnderSampler + SMOTE-Tomek",
        "source": str(source_input),
        "target_count": target_count,
        "majority_class": class_names[majority_id],
        "k_neighbors": k_neighbors,
        "max_source_samples": args.max_source_samples,
        "note": "Balancing is applied only to labeled source data. Target-domain data and labels are not used.",
    }
    save_daeac_npz_with_metadata(
        output_path,
        dataset,
        selected_source_indices=None,
        x=x_resampled,
        y=y_resampled,
        class_names=class_names,
        config_json=config_json,
        input_key="x_daeac",
    )
    summary = {
        "output": str(output_path),
        "input": str(source_input),
        "method": "RandomUnderSampler + SMOTE-Tomek",
        "target_count": target_count,
        "majority_class": class_names[majority_id],
        "k_neighbors": k_neighbors,
        "before_counts": counts_by_name(y_source, class_names),
        "after_rus_counts": counts_by_name(y_rus, class_names),
        "after_smotetomek_counts": counts_by_name(y_resampled, class_names),
        "target_counts_before_tomek": {name: int(targets[idx]) for idx, name in enumerate(class_names)},
        "tomek_removed_or_resampling_delta": int(len(y_resampled) - sum(targets.values())),
        "shape": list(x_resampled.shape),
        "no_target_labels_used": True,
    }
    metrics_dir = ensure_dir(cfg_path(config, "paths", "output_dir") / "metrics")
    write_json(summary, metrics_dir / f"{output_path.stem}_summary.json")
    print(summary)


def _counts(y: np.ndarray, num_classes: int) -> dict[int, int]:
    counts = np.bincount(y.astype(np.int64), minlength=num_classes)
    return {idx: int(counts[idx]) for idx in range(num_classes)}


def _validate_smotetomek_inputs(counts: dict[int, int], target_count: int, class_names: list[str]) -> None:
    if target_count <= 0:
        raise ValueError(f"target_count must be positive, got {target_count}.")
    for cls, count in counts.items():
        if count <= 1:
            raise ValueError(f"Class {class_names[cls]} has only {count} samples; SMOTE needs at least 2.")


def _safe_k_neighbors(counts: dict[int, int], requested: int) -> int:
    min_count = min(counts.values())
    return max(1, min(int(requested), int(min_count) - 1))


def _source_input(config: dict) -> Path:
    value = config["balance"].get("source_input")
    if value is None:
        value = config["data"]["source_train"]
    return resolve_path(value, config["_base_dir"])


def _output_path(arg_output: str | None, default_output: str, config: dict) -> Path:
    value = arg_output or default_output
    return resolve_path(value, config["_base_dir"])


if __name__ == "__main__":
    main()

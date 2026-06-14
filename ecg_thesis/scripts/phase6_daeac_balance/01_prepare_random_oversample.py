from __future__ import annotations

import argparse
from pathlib import Path

from common import cfg_path, load_phase1_config
from src.data.daeac_balance import (
    class_name_to_id,
    counts_by_name,
    load_labeled_daeac_arrays,
    random_oversample_indices,
    save_daeac_npz_with_metadata,
)
from src.utils.io import ensure_dir, resolve_path, write_json
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase6_daeac_hybrid_mkmmd_random_balance.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-source-samples", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_phase1_config(args.config)
    set_seed(int(config["seed"]))
    class_names = list(config["data"]["class_names"])
    input_key = str(config["data"].get("input_key", "auto"))
    label_key = str(config["data"].get("label_key", "y"))
    balance_cfg = dict(config["balance"]["random_oversample"])
    source_input = _source_input(config)
    output_path = _output_path(args.output, balance_cfg["output"], config)
    if output_path.exists() and not args.force:
        print(f"Random oversampled source exists, skipping: {output_path}")
        return

    dataset, x_source, y_source, source_indices = load_labeled_daeac_arrays(
        source_input,
        input_key=input_key,
        label_key=label_key,
        class_names=class_names,
        max_samples=args.max_source_samples,
        seed=int(config["seed"]),
    )
    name_to_id = class_name_to_id(class_names)
    multipliers = {name_to_id[name]: int(value) for name, value in dict(balance_cfg["multipliers"]).items()}
    local_indices = random_oversample_indices(y_source, multipliers, seed=int(config["seed"]))
    original_indices = source_indices[local_indices]
    x_balanced = x_source[local_indices]
    y_balanced = y_source[local_indices]

    config_json = {
        "method": "random_oversample",
        "source": str(source_input),
        "multipliers": dict(balance_cfg["multipliers"]),
        "max_source_samples": args.max_source_samples,
        "note": "Balancing is applied only to labeled source data. Target-domain data and labels are not used.",
    }
    save_daeac_npz_with_metadata(
        output_path,
        dataset,
        original_indices,
        x_balanced,
        y_balanced,
        class_names,
        config_json,
        input_key="x_daeac",
    )
    summary = {
        "output": str(output_path),
        "input": str(source_input),
        "method": "random_oversample",
        "multipliers": dict(balance_cfg["multipliers"]),
        "before_counts": counts_by_name(y_source, class_names),
        "after_counts": counts_by_name(y_balanced, class_names),
        "shape": list(x_balanced.shape),
        "no_target_labels_used": True,
    }
    metrics_dir = ensure_dir(cfg_path(config, "paths", "output_dir") / "metrics")
    write_json(summary, metrics_dir / f"{output_path.stem}_summary.json")
    print(summary)


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

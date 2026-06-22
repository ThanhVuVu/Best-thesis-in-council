from __future__ import annotations

import argparse
import csv
from pathlib import Path

from common import cfg_path, load_phase1_config
from src.utils.io import read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    config = load_phase1_config(args.config)
    variants = dict(config["comparison"]["variants"])
    rows = [_variant_row(name, cfg_path(config, "comparison", "variants", name)) for name in variants]
    output = Path(args.output_dir).resolve() if args.output_dir else cfg_path(config, "paths", "output_dir") / "comparison"
    report = {
        "selection_policy": "Checkpoints and epochs are selected only by source-validation Macro-F1.",
        "target_metrics_role": "Post-training description only; never used for tuning or variant selection.",
        "variants": rows,
    }
    write_json(report, output / "prototype_loss_comparison.json")
    output.mkdir(parents=True, exist_ok=True)
    (output / "prototype_loss_comparison.md").write_text(_markdown(rows), encoding="utf-8")
    print(f"Wrote PLAN 3 comparison to {output}")


def _variant_row(name: str, root: Path) -> dict:
    summary = _first_json(root / "metrics", "*_train_summary.json")
    metrics = {
        dataset: _first_json(root / "metrics", f"*_best_{dataset}_metrics.json")
        for dataset in ("source_val", "target_after5", "incart", "svdb")
    }
    errors = {}
    confusion = _first_path(root / "metrics", "*_best_source_val_confusion_matrix.csv")
    if confusion is not None:
        matrix = _read_confusion(confusion)
        errors = {"source_val_S_to_N": matrix.get("S", {}).get("N"), "source_val_V_to_N": matrix.get("V", {}).get("N")}
    return {
        "variant": name,
        "output_dir": str(root),
        "best_epoch": (summary or {}).get("best_epoch"),
        "best_source_val_macro_f1": (summary or {}).get("best_val_macro_f1"),
        "init_source_val_macro_f1": (summary or {}).get("init_val_macro_f1"),
        "best_adapted_source_val_macro_f1": (summary or {}).get("best_adapted_val_macro_f1"),
        "adaptation_gain_over_init": (summary or {}).get("adaptation_gain_over_init"),
        "selected_stage": (summary or {}).get("selected_stage"),
        **errors,
        "metrics": metrics,
    }


def _first_json(root: Path, pattern: str):
    path = _first_path(root, pattern)
    return read_json(path) if path is not None else None


def _first_path(root: Path, pattern: str):
    return next(iter(sorted(root.glob(pattern))), None)


def _read_confusion(path: Path) -> dict[str, dict[str, int]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    predicted = rows[0][1:]
    return {row[0]: {name: int(value) for name, value in zip(predicted, row[1:])} for row in rows[1:]}


def _markdown(rows: list[dict]) -> str:
    lines = [
        "# DAEAC Prototype Loss Replacement",
        "",
        "Checkpoint selection uses source-validation Macro-F1 only. Target metrics are descriptive and must not select a variant.",
        "",
        "| Variant | Selected stage | Best epoch | Init F1 | Selected F1 | Adaptation gain | S→N | V→N | Target F1 (descriptive) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        target = (row["metrics"].get("target_after5") or {}).get("macro_f1")
        lines.append(
            f"| {row['variant']} | {_fmt(row.get('selected_stage'))} | {_fmt(row['best_epoch'])} | "
            f"{_fmt(row.get('init_source_val_macro_f1'))} | {_fmt(row['best_source_val_macro_f1'])} | "
            f"{_fmt(row.get('adaptation_gain_over_init'))} | {_fmt(row.get('source_val_S_to_N'))} | "
            f"{_fmt(row.get('source_val_V_to_N'))} | {_fmt(target)} |"
        )
    lines.extend(["", "Do not tune loss weights, margins, thresholds, checkpoints, or variants from target columns.", ""])
    return "\n".join(lines)


def _fmt(value) -> str:
    if value is None:
        return "—"
    return f"{value:.4f}" if isinstance(value, float) else str(value)


if __name__ == "__main__":
    main()

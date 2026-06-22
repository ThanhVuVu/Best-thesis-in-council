from __future__ import annotations

import argparse
from pathlib import Path

from common import cfg_path, load_phase1_config
from src.utils.io import ensure_dir, read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    config = load_phase1_config(args.config)
    roots = {
        name: cfg_path(config, "comparison", "variants", name)
        for name in ("no_filter", "confidence_global", "confidence_entropy", "class_specific")
    }
    rows = [_variant_row(name, root) for name, root in roots.items()]
    report = {
        "selection_policy": "Checkpoints are selected only by source-validation Macro-F1.",
        "target_metrics_role": "Post-training description only; never threshold, hyperparameter, variant, or checkpoint selection.",
        "clinical_consistency": "Deferred because morphology keys are absent from the DAEAC NPZ contract.",
        "variants": rows,
    }
    output = ensure_dir(Path(args.output_dir).resolve() if args.output_dir else roots["class_specific"])
    write_json(report, output / "pseudo_filter_comparison.json")
    (output / "pseudo_filter_comparison.md").write_text(_markdown(rows), encoding="utf-8")
    print(f"Wrote pseudo-filter comparison to {output}")


def _variant_row(name: str, root: Path) -> dict:
    summary = _first_json(root / "metrics", "*_train_summary.json")
    metrics = {
        dataset: _first_json(root / "metrics", f"*_best_{dataset}_metrics.json")
        for dataset in ("source_val", "target_after5", "incart", "svdb")
    }
    last = (summary or {}).get("history", [{}])[-1] if (summary or {}).get("history") else {}
    return {
        "variant": name,
        "available": summary is not None,
        "best_epoch": (summary or {}).get("best_epoch"),
        "best_source_val_macro_f1": (summary or {}).get("best_val_macro_f1"),
        "final_acceptance_ratio": last.get("pseudo/accepted_ratio"),
        "final_empty_acceptance": last.get("pseudo/empty_acceptance"),
        "final_all_n": last.get("pseudo/all_n"),
        "metrics": {
            dataset: {"accuracy": value.get("accuracy"), "macro_f1": value.get("macro_f1")} if value else None
            for dataset, value in metrics.items()
        },
    }


def _first_json(root: Path, pattern: str):
    matches = sorted(root.glob(pattern)) if root.exists() else []
    return read_json(matches[0]) if matches else None


def _markdown(rows: list[dict]) -> str:
    lines = [
        "# DAEAC PLAN 2 — Pseudo-label Filtering",
        "",
        "Checkpoint selection uses source-validation Macro-F1 only. Target rows are post-training descriptions.",
        "",
        "| Variant | Best epoch | Source-val F1 | Accept ratio | Empty | All-N | DS2-after5 F1 | INCART F1 | SVDB F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        metrics = row["metrics"]
        values = [
            row["variant"], row["best_epoch"], row["best_source_val_macro_f1"],
            row["final_acceptance_ratio"], row["final_empty_acceptance"], row["final_all_n"],
            (metrics["target_after5"] or {}).get("macro_f1"),
            (metrics["incart"] or {}).get("macro_f1"), (metrics["svdb"] or {}).get("macro_f1"),
        ]
        lines.append("| " + " | ".join(_fmt(value) for value in values) + " |")
    lines.extend(["", "Do not use target columns to tune filtering thresholds or select a variant.", ""])
    return "\n".join(lines)


def _fmt(value) -> str:
    if value is None:
        return "—"
    return f"{value:.4f}" if isinstance(value, float) else str(value)


if __name__ == "__main__":
    main()

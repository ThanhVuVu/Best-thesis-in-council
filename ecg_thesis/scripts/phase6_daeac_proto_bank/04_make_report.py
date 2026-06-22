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
        "logging_only": cfg_path(config, "comparison", "logging_output_dir"),
        "weighted_global": cfg_path(config, "comparison", "weighted_output_dir"),
    }
    rows = []
    for variant, root in roots.items():
        summary = _first_json(root / "metrics", "*_train_summary.json")
        metrics = {
            "source_val": _first_json(root / "metrics", "*_best_source_val_metrics.json"),
            "target_full": _first_json(root / "metrics", "*_best_target_full_metrics.json"),
            "incart": _first_json(root / "metrics", "*_best_incart_metrics.json"),
            "svdb": _first_json(root / "metrics", "*_best_svdb_metrics.json"),
        }
        rows.append(
            {
                "variant": variant,
                "available": summary is not None,
                "best_epoch": summary.get("best_epoch") if summary else None,
                "best_source_val_macro_f1": summary.get("best_val_macro_f1") if summary else None,
                "metrics": {
                    name: {"accuracy": value.get("accuracy"), "macro_f1": value.get("macro_f1")}
                    if value
                    else None
                    for name, value in metrics.items()
                },
                "final_prototypes": summary.get("final_prototype_diagnostics") if summary else None,
            }
        )
    report = {
        "selection_policy": "Checkpoints are selected only by source-validation Macro-F1; target metrics are descriptive.",
        "variants": rows,
    }
    output = ensure_dir(Path(args.output_dir).resolve() if args.output_dir else roots["weighted_global"])
    write_json(report, output / "prototype_bank_comparison.json")
    markdown = _markdown(rows)
    (output / "prototype_bank_comparison.md").write_text(markdown, encoding="utf-8")
    print(f"Wrote prototype-bank comparison to {output}")


def _first_json(root: Path, pattern: str):
    matches = sorted(root.glob(pattern)) if root.exists() else []
    return read_json(matches[0]) if matches else None


def _markdown(rows):
    lines = [
        "# DAEAC Reliability-Weighted Prototype Bank",
        "",
        "Checkpoints are selected only by source-validation Macro-F1. Target metrics below are post-training descriptions, never selection signals.",
        "",
        "| Variant | Best epoch | Source-monitor Macro-F1 | Full-DS2 Macro-F1 | INCART Macro-F1 | SVDB Macro-F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        metrics = row["metrics"]
        values = [
            row["variant"],
            _fmt(row["best_epoch"]),
            _fmt(row["best_source_val_macro_f1"]),
            _fmt((metrics["target_full"] or {}).get("macro_f1")),
            _fmt((metrics["incart"] or {}).get("macro_f1")),
            _fmt((metrics["svdb"] or {}).get("macro_f1")),
        ]
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "Interpret prototype reliability, beta, validity, and skipped-update logs together. Do not tune bank settings from target rows.",
            "",
        ]
    )
    return "\n".join(lines)


def _fmt(value):
    if value is None:
        return "—"
    return f"{value:.4f}" if isinstance(value, float) else str(value)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

from common import cfg_path, load_phase1_config
from src.utils.io import read_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase1.yaml")
    args = parser.parse_args()
    config = load_phase1_config(args.config)
    output = cfg_path(config, "paths", "output_dir")
    metrics_dir = output / "metrics"

    validation = _read_optional(metrics_dir / "processed_validation.json")
    train = _read_optional(metrics_dir / "train_summary.json")
    mit = _read_optional(metrics_dir / "mitbih_test_metrics.json")
    inc = _read_optional(metrics_dir / "incart_test_metrics.json")

    lines = [
        "# Phase 1 Report",
        "",
        "## Dataset Statistics",
        "",
        _format_json_block(validation),
        "",
        "## Training Summary",
        "",
        _format_json_block(train),
        "",
        "## In-domain MIT-BIH Results",
        "",
        _format_metrics(mit),
        "",
        "## Cross-domain INCART Results",
        "",
        _format_metrics(inc),
        "",
        "## Initial Conclusion",
        "",
        _conclusion(mit, inc),
        "",
    ]
    report_path = output / "phase1_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {report_path}")


def _read_optional(path: Path):
    return read_json(path) if path.exists() else None


def _format_json_block(obj) -> str:
    if obj is None:
        return "_Not generated yet._"
    import json

    return "```json\n" + json.dumps(obj, indent=2) + "\n```"


def _format_metrics(metrics) -> str:
    if metrics is None:
        return "_Not generated yet._"
    rows = [
        f"- Accuracy: {metrics['accuracy']:.4f}",
        f"- Macro-F1: {metrics['macro_f1']:.4f}",
    ]
    for cls, values in metrics["per_class"].items():
        rows.append(
            f"- {cls}: precision={values['precision']:.4f}, recall={values['recall']:.4f}, "
            f"f1={values['f1']:.4f}, support={values['support']}"
        )
    return "\n".join(rows)


def _conclusion(mit, inc) -> str:
    if mit is None or inc is None:
        return "Final conclusion is pending until both in-domain and cross-domain evaluations are complete."
    drop = mit["macro_f1"] - inc["macro_f1"]
    return (
        f"MIT-BIH macro-F1 is {mit['macro_f1']:.4f}; INCART macro-F1 is {inc['macro_f1']:.4f}; "
        f"the source-only cross-domain drop is {drop:.4f}. Inspect the saved confusion matrices, "
        "example beat plots, and embedding UMAP before making a final domain-shift claim."
    )


if __name__ == "__main__":
    main()

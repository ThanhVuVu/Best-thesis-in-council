from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import cfg_path, load_phase1_config
from src.utils.io import read_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase2_dann.yaml")
    args = parser.parse_args()
    config = load_phase1_config(args.config)
    output = cfg_path(config, "paths", "output_dir")
    metrics_dir = output / "metrics"

    names = {
        "incart_split": "phase2_incart_split.json",
        "source_train": "source_only_inception_train_summary.json",
        "source_mitbih": "source_only_inception_mitbih_test_metrics.json",
        "source_incart": "source_only_inception_incart_heldout_metrics.json",
        "dann_train": "dann_train_summary.json",
        "dann_mitbih": "dann_mitbih_test_metrics.json",
        "dann_incart": "dann_incart_heldout_metrics.json",
    }
    data = {key: _optional(metrics_dir / filename) for key, filename in names.items()}
    lines = [
        "# Phase 2 DANN Report",
        "",
        "## INCART Adapt/Test Split",
        "",
        _json_block(data["incart_split"]),
        "",
        "## Source-only InceptionTime1D",
        "",
        _format_metrics(data["source_mitbih"], "MIT-BIH test"),
        "",
        _format_metrics(data["source_incart"], "INCART held-out"),
        "",
        "## DANN InceptionTime1D",
        "",
        _format_metrics(data["dann_mitbih"], "MIT-BIH test"),
        "",
        _format_metrics(data["dann_incart"], "INCART held-out"),
        "",
        "## Training Summaries",
        "",
        "### Source-only",
        "",
        _json_block(data["source_train"]),
        "",
        "### DANN",
        "",
        _json_block(data["dann_train"]),
        "",
        "## Initial Comparison",
        "",
        _comparison(data["source_incart"], data["dann_incart"]),
        "",
    ]
    report_path = output / "phase2_dann_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {report_path}")


def _optional(path: Path):
    return read_json(path) if path.exists() else None


def _json_block(obj) -> str:
    if obj is None:
        return "_Not generated yet._"
    return "```json\n" + json.dumps(obj, indent=2) + "\n```"


def _format_metrics(metrics, title: str) -> str:
    if metrics is None:
        return f"### {title}\n\n_Not generated yet._"
    rows = [
        f"### {title}",
        "",
        f"- Accuracy: {metrics['accuracy']:.4f}",
        f"- Macro-F1: {metrics['macro_f1']:.4f}",
    ]
    for cls, values in metrics["per_class"].items():
        rows.append(
            f"- {cls}: precision={values['precision']:.4f}, recall={values['recall']:.4f}, "
            f"f1={values['f1']:.4f}, support={values['support']}"
        )
    return "\n".join(rows)


def _comparison(source, dann) -> str:
    if source is None or dann is None:
        return "Final comparison is pending until source-only and DANN INCART held-out metrics are generated."
    delta = dann["macro_f1"] - source["macro_f1"]
    s_delta = dann["per_class"]["S"]["f1"] - source["per_class"]["S"]["f1"]
    v_delta = dann["per_class"]["V"]["f1"] - source["per_class"]["V"]["f1"]
    return (
        f"DANN target Macro-F1 delta vs source-only: {delta:+.4f}. "
        f"S-F1 delta: {s_delta:+.4f}. V-F1 delta: {v_delta:+.4f}. "
        "Treat DANN as successful only if target Macro-F1 improves and minority-class performance, "
        "especially S precision/F1, improves without severe N/V collapse."
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse

from common import cfg_path, load_phase1_config
from src.utils.io import ensure_dir, read_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase4b_sourcefree_ecgfm_leadbridge.yaml")
    args = parser.parse_args()

    config = load_phase1_config(args.config)
    output = ensure_dir(cfg_path(config, "paths", "output_dir"))
    metrics_dir = output / "metrics"
    data = {
        "phase4a_incart": _optional(metrics_dir / "source_only_ecgfm_leadbridge_incart_heldout_metrics.json"),
        "phase4b_train": _optional(metrics_dir / "sourcefree_ecgfm_leadbridge_train_summary.json"),
        "phase4b_mitbih": _optional(metrics_dir / "sourcefree_ecgfm_leadbridge_mitbih_test_metrics.json"),
        "phase4b_incart": _optional(metrics_dir / "sourcefree_ecgfm_leadbridge_incart_heldout_metrics.json"),
    }

    lines = [
        "# Phase 4B Source-Free ECG-FM LeadBridge Report",
        "",
        "## Train Summary",
        _json_block(data["phase4b_train"]),
        "",
        "## Evaluation",
        _format_metrics(data["phase4b_mitbih"], "MIT-BIH test"),
        "",
        _format_metrics(data["phase4b_incart"], "INCART held-out"),
        "",
        "## Phase 4A -> 4B Delta",
        _comparison(data["phase4a_incart"], data["phase4b_incart"]),
        "",
    ]
    report_path = output / "phase4b_sourcefree_ecgfm_leadbridge_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {report_path}")


def _optional(path):
    return read_json(path) if path.exists() else None


def _json_block(data) -> str:
    if data is None:
        return "Not available."
    import json

    return "```json\n" + json.dumps(data, indent=2) + "\n```"


def _format_metrics(metrics, title: str) -> str:
    if metrics is None:
        return f"### {title}\nNot available."
    return (
        f"### {title}\n"
        f"- accuracy: {metrics['accuracy']:.4f}\n"
        f"- macro_f1: {metrics['macro_f1']:.4f}\n"
        f"- S f1: {metrics['per_class']['S']['f1']:.4f}\n"
        f"- V f1: {metrics['per_class']['V']['f1']:.4f}"
    )


def _comparison(source, adapted) -> str:
    if source is None or adapted is None:
        return "Not available."
    macro_delta = adapted["macro_f1"] - source["macro_f1"]
    s_delta = adapted["per_class"]["S"]["f1"] - source["per_class"]["S"]["f1"]
    v_delta = adapted["per_class"]["V"]["f1"] - source["per_class"]["V"]["f1"]
    return (
        f"- INCART macro_f1 delta: {macro_delta:+.4f}\n"
        f"- INCART S f1 delta: {s_delta:+.4f}\n"
        f"- INCART V f1 delta: {v_delta:+.4f}"
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import cfg_path, load_phase1_config
from src.utils.io import read_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase4a_ecgfm_leadbridge.yaml")
    args = parser.parse_args()

    config = load_phase1_config(args.config)
    output = cfg_path(config, "paths", "output_dir")
    metrics_dir = output / "metrics"
    data = {
        "preprocess": _optional(metrics_dir / "phase4a_preprocess_summary.json"),
        "train": _optional(metrics_dir / "source_only_ecgfm_leadbridge_train_summary.json"),
        "phase2_dann": _optional(metrics_dir / "dann_incart_heldout_metrics.json"),
        "phase3_source_rr": _optional(metrics_dir / "source_only_catnet_rr_incart_heldout_metrics.json"),
        "phase3_dann_rr": _optional(metrics_dir / "dann_rr_incart_heldout_metrics.json"),
        "phase4a_mitbih": _optional(metrics_dir / "source_only_ecgfm_leadbridge_mitbih_test_metrics.json"),
        "phase4a_incart": _optional(metrics_dir / "source_only_ecgfm_leadbridge_incart_heldout_metrics.json"),
    }
    lines = [
        "# Phase 4A ECG-FM LeadBridge Report",
        "",
        "## Goal",
        "",
        "Phase 4A tests a frozen ECG-FM foundation encoder with a trainable 1-lead to 12-lead LeadBridge and N/S/V classification head.",
        "",
        "## Preprocessing Summary",
        "",
        _json_block(data["preprocess"]),
        "",
        "## Previous INCART Held-out Baselines",
        "",
        _format_metrics(data["phase2_dann"], "Phase 2 DANN CATNet1D"),
        "",
        _format_metrics(data["phase3_source_rr"], "Phase 3 source-only CATNet1D + RR"),
        "",
        _format_metrics(data["phase3_dann_rr"], "Phase 3 DANN CATNet1D + RR"),
        "",
        "## Phase 4A Results",
        "",
        _format_metrics(data["phase4a_mitbih"], "MIT-BIH test"),
        "",
        _format_metrics(data["phase4a_incart"], "INCART held-out"),
        "",
        "## Main Comparison",
        "",
        _comparison(data["phase3_source_rr"], data["phase4a_incart"]),
        "",
        "## Training Summary",
        "",
        _json_block(data["train"]),
        "",
    ]
    report_path = output / "phase4a_ecgfm_leadbridge_report.md"
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


def _comparison(phase3_source_rr, phase4a_incart) -> str:
    if phase3_source_rr is None or phase4a_incart is None:
        return "Final comparison is pending until Phase 3 and Phase 4A INCART metrics are available."
    macro_delta = phase4a_incart["macro_f1"] - phase3_source_rr["macro_f1"]
    s_delta = phase4a_incart["per_class"]["S"]["f1"] - phase3_source_rr["per_class"]["S"]["f1"]
    return (
        f"Phase 4A target Macro-F1 delta vs Phase 3 source-only RR: {macro_delta:+.4f}. "
        f"S-F1 delta: {s_delta:+.4f}. "
        "Treat Phase 4A as useful if frozen ECG-FM + LeadBridge improves source-only target transfer "
        "or provides a strong checkpoint for source-free Phase 4B."
    )


if __name__ == "__main__":
    main()

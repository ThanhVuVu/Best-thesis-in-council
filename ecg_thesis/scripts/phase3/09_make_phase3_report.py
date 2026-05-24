from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import cfg_path, load_phase1_config
from src.utils.io import read_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase3_rr_dann.yaml")
    args = parser.parse_args()

    config = load_phase1_config(args.config)
    output = cfg_path(config, "paths", "output_dir")
    metrics_dir = output / "metrics"
    data = {
        "rr_stats": _optional(metrics_dir / "phase3_rr_normalization.json"),
        "failure_summary": _optional(metrics_dir / "phase3_failure_summary.json"),
        "p2_source_incart": _optional(metrics_dir / "source_only_catnet_incart_heldout_metrics.json"),
        "p2_dann_incart": _optional(metrics_dir / "dann_incart_heldout_metrics.json"),
        "p3_source_mitbih": _optional(metrics_dir / "source_only_catnet_rr_mitbih_test_metrics.json"),
        "p3_source_incart": _optional(metrics_dir / "source_only_catnet_rr_incart_heldout_metrics.json"),
        "p3_dann_mitbih": _optional(metrics_dir / "dann_rr_mitbih_test_metrics.json"),
        "p3_dann_incart": _optional(metrics_dir / "dann_rr_incart_heldout_metrics.json"),
        "source_train": _optional(metrics_dir / "source_only_catnet_rr_train_summary.json"),
        "dann_train": _optional(metrics_dir / "dann_rr_train_summary.json"),
    }

    lines = [
        "# Phase 3 RR-DANN Report",
        "",
        "## Goal",
        "",
        "Phase 3 tests whether RR rhythm features improve CATNet1D and DANN transfer for MIT-BIH -> INCART N/S/V classification.",
        "",
        "## RR Normalization",
        "",
        _json_block(data["rr_stats"]),
        "",
        "## Phase 2 Failure Summary",
        "",
        _json_block(data["failure_summary"]),
        "",
        "## Phase 2 Baselines on INCART Held-out",
        "",
        _format_metrics(data["p2_source_incart"], "Source-only CATNet1D"),
        "",
        _format_metrics(data["p2_dann_incart"], "DANN CATNet1D"),
        "",
        "## Phase 3 Source-only CATNet1D + RR",
        "",
        _format_metrics(data["p3_source_mitbih"], "MIT-BIH test"),
        "",
        _format_metrics(data["p3_source_incart"], "INCART held-out"),
        "",
        "## Phase 3 DANN CATNet1D + RR",
        "",
        _format_metrics(data["p3_dann_mitbih"], "MIT-BIH test"),
        "",
        _format_metrics(data["p3_dann_incart"], "INCART held-out"),
        "",
        "## Main Comparison",
        "",
        _comparison(data["p2_dann_incart"], data["p3_dann_incart"]),
        "",
        "## Training Summaries",
        "",
        "### Source-only RR",
        "",
        _json_block(data["source_train"]),
        "",
        "### DANN-RR",
        "",
        _json_block(data["dann_train"]),
        "",
    ]
    report_path = output / "phase3_rr_dann_report.md"
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


def _comparison(phase2_dann, phase3_dann) -> str:
    if phase2_dann is None or phase3_dann is None:
        return "Final comparison is pending until DANN-RR evaluation is generated."
    macro_delta = phase3_dann["macro_f1"] - phase2_dann["macro_f1"]
    s_delta = phase3_dann["per_class"]["S"]["f1"] - phase2_dann["per_class"]["S"]["f1"]
    v_delta = phase3_dann["per_class"]["V"]["f1"] - phase2_dann["per_class"]["V"]["f1"]
    return (
        f"DANN-RR target Macro-F1 delta vs Phase 2 DANN: {macro_delta:+.4f}. "
        f"S-F1 delta: {s_delta:+.4f}. V-F1 delta: {v_delta:+.4f}. "
        "Treat Phase 3 as successful if rhythm features improve target Macro-F1 or S-F1 "
        "without severe N/V regression."
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from diagnostics_common import load_diagnostics_config, method_name, output_dir
from src.utils.io import ensure_dir, read_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase6_daeac_diagnostics.yaml")
    parser.add_argument("--method-name", default=None)
    args = parser.parse_args()

    diag_config, _base_config = load_diagnostics_config(args.config)
    method = method_name(diag_config, args.method_name)
    out = output_dir(diag_config)
    report_path = ensure_dir(out) / f"{method}_diagnostics_report.md"

    tables = out / "diagnostics" / "tables"
    calibration = out / "diagnostics" / "calibration"
    pr_roc = out / "diagnostics" / "pr_roc"
    embeddings = out / "diagnostics" / "embeddings"
    pseudo = out / "diagnostics" / "pseudo"

    failure_summary = _maybe_json(tables / f"{method}_failure_summary.json")
    calibration_summary = _maybe_json(calibration / f"{method}_calibration_summary.json")
    pr_roc_summary = _maybe_json(pr_roc / f"{method}_pr_roc_summary.json")
    embedding_summary = _maybe_json(embeddings / f"{method}_embedding_summary.json")
    pseudo_summary = _maybe_json(pseudo / f"{method}_pseudo_summary.json")

    lines = [
        f"# DAEAC Diagnostics Report: {method}",
        "",
        "## Scope",
        "",
        "- This report reads checkpoint outputs and preprocessed DAEAC `.npz` files only.",
        "- It does not train or adapt the model.",
        f"- Output root: `{out}`.",
        "",
        "## Failure Tables",
        "",
    ]
    lines += _failure_section(failure_summary)
    lines += [
        "## Calibration",
        "",
    ]
    lines += _calibration_section(calibration_summary)
    lines += [
        "## PR/ROC Threshold Analysis",
        "",
    ]
    lines += _pr_roc_section(pr_roc_summary)
    lines += [
        "## Embedding Separation",
        "",
    ]
    lines += _embedding_section(embedding_summary)
    lines += [
        "## Pseudo-Label Audit",
        "",
    ]
    lines += _pseudo_section(pseudo_summary)
    lines += [
        "## Artifact Index",
        "",
        f"- Predictions: `{out / 'predictions'}`.",
        f"- Metrics and CSV diagnostics: `{out / 'diagnostics'}`.",
        f"- Figures: `{out / 'figures'}`.",
        f"- Embeddings, if collected with `--save-embeddings`: `{out / 'embeddings'}`.",
        "",
        "## Reproduction Commands",
        "",
        "```bash",
        "python scripts/phase6_daeac_diagnostics/01_collect_predictions.py --config configs/phase6_daeac_diagnostics.yaml --dataset all --save-embeddings",
        "python scripts/phase6_daeac_diagnostics/02_failure_tables.py --config configs/phase6_daeac_diagnostics.yaml",
        "python scripts/phase6_daeac_diagnostics/03_calibration_curves.py --config configs/phase6_daeac_diagnostics.yaml",
        "python scripts/phase6_daeac_diagnostics/04_pr_roc_analysis.py --config configs/phase6_daeac_diagnostics.yaml",
        "python scripts/phase6_daeac_diagnostics/05_embedding_analysis.py --config configs/phase6_daeac_diagnostics.yaml --method umap --max-samples 10000",
        "python scripts/phase6_daeac_diagnostics/06_morphology_panels.py --config configs/phase6_daeac_diagnostics.yaml --top-k 24",
        "python scripts/phase6_daeac_diagnostics/07_pseudo_label_audit.py --config configs/phase6_daeac_diagnostics.yaml",
        "python scripts/phase6_daeac_diagnostics/08_make_diagnostics_report.py --config configs/phase6_daeac_diagnostics.yaml",
        "```",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {report_path}")


def _failure_section(summary: dict[str, Any] | None) -> list[str]:
    if not summary:
        return ["Failure summary not found. Run `02_failure_tables.py` first.", ""]
    lines: list[str] = []
    for dataset, values in summary.items():
        lines += [
            f"### {dataset}",
            "",
            f"- Accuracy: {_fmt(values.get('accuracy'))}.",
            f"- Macro-F1: {_fmt(values.get('macro_f1'))}.",
            "- Top error pairs:",
            "",
        ]
        pairs = list(values.get("top_error_pairs", []))[:6]
        if not pairs:
            lines += ["  No errors found in collected predictions.", ""]
            continue
        for row in pairs:
            lines.append(
                f"  - {row.get('true_class')} -> {row.get('pred_class')}: "
                f"{row.get('count')} errors, avg_conf={_fmt(row.get('avg_confidence'))}, "
                f"avg_entropy={_fmt(row.get('avg_entropy'))}."
            )
        lines.append("")
    return lines


def _calibration_section(summary: dict[str, Any] | None) -> list[str]:
    if not summary:
        return ["Calibration summary not found. Run `03_calibration_curves.py` first.", ""]
    lines: list[str] = []
    for dataset, values in summary.items():
        classwise = values.get("classwise_predicted_ece", {})
        worst = _top_items(classwise)
        lines += [
            f"- `{dataset}` ECE: {_fmt(values.get('ece'))}. Worst predicted-class ECE: {worst}.",
        ]
    lines.append("")
    return lines


def _pr_roc_section(summary: dict[str, Any] | None) -> list[str]:
    if not summary:
        return ["PR/ROC summary not found. Run `04_pr_roc_analysis.py` first.", ""]
    lines: list[str] = []
    for dataset, classes in summary.items():
        lines += [f"### {dataset}", ""]
        for class_name, values in classes.items():
            best = values.get("best_threshold_by_f1") or {}
            lines.append(
                f"- {class_name}: AUPRC={_fmt(values.get('auprc'))}, AUROC={_fmt(values.get('auroc'))}, "
                f"best_F1_threshold={_fmt(best.get('threshold'))}, best_F1={_fmt(best.get('f1'))}."
            )
        lines.append("")
    return lines


def _embedding_section(summary: dict[str, Any] | None) -> list[str]:
    if not summary:
        return ["Embedding summary not found. Run `05_embedding_analysis.py` after collecting embeddings.", ""]
    lines = []
    for dataset, values in summary.items():
        lines.append(
            f"- `{dataset}` silhouette by true class: {_fmt(values.get('silhouette_true_class'))} "
            f"using {values.get('num_embedding_samples')} samples."
        )
    lines.append("")
    return lines


def _pseudo_section(summary: dict[str, Any] | None) -> list[str]:
    if not summary:
        return ["Pseudo-label summary not found. Run `07_pseudo_label_audit.py` first.", ""]
    if not summary.get("available", False):
        return [f"Pseudo-label audit unavailable: {summary.get('reason')} at `{summary.get('train_summary')}`.", ""]
    behavior = summary.get("pseudo_behavior", {})
    lines = [
        f"- Best adaptation epoch: {summary.get('best_epoch')}.",
        f"- Best source-validation Macro-F1: {_fmt(summary.get('best_val_macro_f1'))}.",
        f"- Final pseudo-label ratios: `{json.dumps(behavior.get('final_pseudo_ratios', {}))}`.",
        f"- Final target prediction ratios: `{json.dumps(behavior.get('final_target_pred_ratios', {}))}`.",
    ]
    zero = behavior.get("low_or_zero_pseudo_classes", [])
    if zero:
        lines.append(f"- Classes with zero final selected pseudo labels: `{zero}`.")
    lines.append("")
    return lines


def _maybe_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return read_json(path)


def _fmt(value: Any) -> str:
    if value is None:
        return "not available"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _top_items(values: dict[str, Any], n: int = 2) -> str:
    rows = [(name, item) for name, item in values.items() if isinstance(item, (int, float))]
    rows = sorted(rows, key=lambda item: float(item[1]), reverse=True)[:n]
    if not rows:
        return "not available"
    return ", ".join(f"{name}={_fmt(value)}" for name, value in rows)


if __name__ == "__main__":
    main()

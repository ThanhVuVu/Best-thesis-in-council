from __future__ import annotations

import argparse
from pathlib import Path

from common import cfg_path, load_phase1_config
from src.utils.io import ensure_dir, read_json


BASELINE = {
    "name": "Phase 2 CATNet1D + DANN",
    "incart_heldout_macro_f1": 0.6783,
    "incart_heldout_s_f1": 0.4178,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase2p_catnet_paper_uda.yaml")
    args = parser.parse_args()
    config = load_phase1_config(args.config)
    output = ensure_dir(cfg_path(config, "paths", "output_dir"))
    metrics = output / "metrics"
    report_path = output / "phase2p_catnet_paper_uda_report.md"

    preprocess = _maybe_json(metrics / "phase2p_preprocess_summary.json")
    duplicate = _maybe_json(metrics / "phase2p_source_duplication_summary.json")
    source_train = _maybe_json(metrics / "phase2p_catnet_source_train_summary.json")
    cluster_train = _maybe_json(metrics / "phase2p_cluster_source_train_summary.json")
    pseudo = _maybe_json(metrics / "phase2p_clustered_pseudolabel_stats.json")
    uda_train = _maybe_json(metrics / "phase2p_centroid_uda_train_summary.json")
    eval_summary = _maybe_json(metrics / "phase2p_centroid_uda_eval_summary.json")

    lines = [
        "# Phase 2P CATNet Paper-Style UDA Report",
        "",
        "## Protocol",
        "",
        "- Source/target: MIT-BIH -> INCART.",
        "- Labels: N/S/V only.",
        "- INCART adaptation records: I01-I50.",
        "- INCART held-out records: I51-I75.",
        "- Target held-out labels are used only in final evaluation.",
        "- Preprocessing intentionally differs from Phase 2 DANN, so the comparison is not apples-to-apples.",
        "",
        "## Preprocessing",
        "",
        "- Bandpass: 3-20 Hz.",
        "- Resampling: 256 Hz.",
        "- Segmentation: source-mean-RR heartbeat windows.",
        "- Time features: rr_current, rr_pre_avg, rr_pre8_avg.",
        "- Source fit duplication: N +0, S +5, V +2 additional copies.",
        "",
    ]
    if preprocess:
        lines += ["### Preprocess Summary", "", _code(preprocess), ""]
    if duplicate:
        lines += ["### Duplication Summary", "", _code(duplicate), ""]

    lines += [
        "## Training Summary",
        "",
        _summary_line("Source pretrain best validation Macro-F1", source_train, "best_val_macro_f1"),
        _summary_line("Cluster source best validation Macro-F1", cluster_train, "best_val_macro_f1"),
        _summary_line("UDA best source-validation Macro-F1", uda_train, "best_val_macro_f1"),
        "",
    ]
    if pseudo:
        lines += ["### Target Pseudo-Labels", "", _code(pseudo), ""]

    lines += [
        "## Final Evaluation",
        "",
    ]
    if eval_summary:
        for dataset_name, result in eval_summary.get("datasets", {}).items():
            lines += [
                f"### {dataset_name}",
                "",
                f"- Macro-F1: {_fmt(result.get('macro_f1'))}",
                f"- Accuracy: {_fmt(result.get('accuracy'))}",
                f"- S-F1: {_fmt(result.get('per_class', {}).get('S', {}).get('f1'))}",
                f"- S-precision: {_fmt(result.get('per_class', {}).get('S', {}).get('precision'))}",
                f"- S-recall: {_fmt(result.get('per_class', {}).get('S', {}).get('recall'))}",
                f"- Confusion matrix N/S/V rows -> N/S/V columns: `{result.get('confusion_matrix')}`",
                "",
            ]
    else:
        lines += ["Evaluation summary not found yet. Run `06_eval_centroid_uda.py` first.", ""]

    lines += [
        "## Phase 2 Reference",
        "",
        f"- {BASELINE['name']} INCART held-out Macro-F1: {BASELINE['incart_heldout_macro_f1']:.4f}",
        f"- {BASELINE['name']} INCART held-out S-F1: {BASELINE['incart_heldout_s_f1']:.4f}",
        "- This Phase 2P run changes preprocessing and adaptation losses, so treat differences as experimental evidence rather than a one-variable ablation.",
        "",
        "## Artifact Index",
        "",
        "- Checkpoints: `outputs/checkpoints/phase2p_catnet_*_{best,latest}.pt`.",
        "- Metrics: `outputs/metrics/phase2p_*.json`.",
        "- Predictions and pseudo-labels: `outputs/predictions/phase2p_*.csv`.",
        "- Prototypes: `outputs/prototypes/phase2p_*.pt`.",
        "- Backup snapshots: configured `artifact_backup_dir` or cloud runtime backup directory.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {report_path}")


def _maybe_json(path: Path):
    if not path.exists():
        return None
    return read_json(path)


def _summary_line(label: str, data, key: str) -> str:
    if not data:
        return f"- {label}: not available yet."
    return f"- {label}: {_fmt(data.get(key))}."


def _fmt(value) -> str:
    if value is None:
        return "not available"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _code(data) -> str:
    import json

    return "```json\n" + json.dumps(data, indent=2)[:6000] + "\n```"


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

from common import cfg_path, load_phase1_config
from src.utils.io import read_json


METHODS = [
    ("CATNet1D + DANN first5", "catnet_first5_dann"),
    ("MACNN_SE source-only", "macnn_se_source_only"),
    ("MACNN_SE + DANN", "macnn_se_dann"),
    ("MACNN_SE + DAEAC-style", "macnn_se_daeac"),
    ("MACNN_SE + L_align only", "macnn_se_daeac_align_only"),
    ("MACNN_SE + L_align + L_comp", "macnn_se_daeac_align_compact"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase5_macnn_daeac.yaml")
    args = parser.parse_args()
    config = load_phase1_config(args.config)
    output = cfg_path(config, "paths", "output_dir")
    metrics_dir = output / "metrics"
    lines = [
        "# Phase 5 MACNN_SE + DAEAC-Style Report",
        "",
        "## Goal",
        "",
        "Compare MACNN_SE source-only, MACNN_SE + DANN, DAEAC-style prototype alignment, and CATNet1D + DANN on the INCART first-5-min unlabeled adaptation protocol.",
        "",
        "## Main Metrics",
        "",
        "| Method | Test Domain | Accuracy | Macro-F1 | N-F1 | S-F1 | V-F1 | S->N | S->V |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, stem in METHODS:
        for dataset in ("mitbih_test", "incart_after5_heldout"):
            metrics = _read_metrics(metrics_dir / f"{stem}_{dataset}_metrics.json")
            if metrics is None:
                lines.append(f"| {label} | {dataset} | pending | pending | pending | pending | pending | pending | pending |")
                continue
            cm = metrics["confusion_matrix"]
            s_to_n = cm[1][0] if len(cm) > 1 else 0
            s_to_v = cm[1][2] if len(cm) > 1 and len(cm[1]) > 2 else 0
            pc = metrics["per_class"]
            lines.append(
                f"| {label} | {dataset} | {metrics['accuracy']:.4f} | {metrics['macro_f1']:.4f} | "
                f"{pc['N']['f1']:.4f} | {pc['S']['f1']:.4f} | {pc['V']['f1']:.4f} | {s_to_n} | {s_to_v} |"
            )
    lines.extend([
        "",
        "## Ablation",
        "",
        "| Method | INCART Macro-F1 | INCART S Precision | INCART S Recall | INCART S-F1 |",
        "|---|---:|---:|---:|---:|",
    ])
    for label, stem in METHODS[3:]:
        metrics = _read_metrics(metrics_dir / f"{stem}_incart_after5_heldout_metrics.json")
        if metrics is None:
            lines.append(f"| {label} | pending | pending | pending | pending |")
            continue
        s = metrics["per_class"]["S"]
        lines.append(f"| {label} | {metrics['macro_f1']:.4f} | {s['precision']:.4f} | {s['recall']:.4f} | {s['f1']:.4f} |")
    lines.extend([
        "",
        "## Notes",
        "",
        "- Target adaptation uses INCART beats with `r_peak_time_sec < 300` as unlabeled data.",
        "- Held-out target evaluation uses INCART beats with `r_peak_time_sec >= 300`.",
        "- Target labels must not be used during DANN or DAEAC-style training.",
        "- Metrics with `max_samples` in the filename are smoke/debug outputs, not thesis results.",
    ])
    report_path = output / "phase5_macnn_daeac_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {report_path}")


def _read_metrics(path: Path) -> dict | None:
    if not path.exists():
        return None
    return read_json(path)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

from common import cfg_path, load_phase1_config
from src.utils.io import ensure_dir, read_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase6_daeac_paper.yaml")
    args = parser.parse_args()
    config = load_phase1_config(args.config)
    output = ensure_dir(cfg_path(config, "paths", "output_dir"))
    metrics_dir = output / "metrics"
    lines = [
        "# Phase 6 DAEAC Paper-Faithful Report",
        "",
        "Class order: `N=0, S=1, V=2, F=3`.",
        "",
        "| Method | Dataset | Accuracy | Macro-F1 | N-F1 | S-F1 | V-F1 | F-F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for path in sorted(metrics_dir.glob("*_metrics.json")):
        metrics = read_json(path)
        per = metrics.get("per_class", {})
        lines.append(
            "| {setting} | {dataset} | {accuracy:.4f} | {macro_f1:.4f} | {n:.4f} | {s:.4f} | {v:.4f} | {f:.4f} |".format(
                setting=metrics.get("setting", path.stem),
                dataset=metrics.get("dataset", "target_test"),
                accuracy=float(metrics.get("accuracy", 0.0)),
                macro_f1=float(metrics.get("macro_f1", 0.0)),
                n=float(per.get("N", {}).get("f1", 0.0)),
                s=float(per.get("S", {}).get("f1", 0.0)),
                v=float(per.get("V", {}).get("f1", 0.0)),
                f=float(per.get("F", {}).get("f1", 0.0)),
            )
        )
    lines += [
        "",
        "## Paper Sanity Targets",
        "",
        "- DAEAC-base target accuracy: 0.9462.",
        "- DAEAC-uda target accuracy: 0.9759.",
        "- Reported F1 order in the paper is N, VEB, SVEB, F; this implementation stores class order as N, S, V, F.",
        "",
        "## Notes",
        "",
        "- Target labels are only used by `03_eval.py`.",
        "- Adaptation uses target unlabeled tensors and pseudo labels from auxiliary classifier `h`.",
    ]
    report_path = output / "phase6_daeac_paper_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()

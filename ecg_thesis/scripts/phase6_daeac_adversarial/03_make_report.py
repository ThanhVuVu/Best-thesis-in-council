from __future__ import annotations

import argparse
from pathlib import Path

from common import cfg_path, load_phase1_config
from src.utils.io import ensure_dir, read_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase6_daeac_dann.yaml")
    parser.add_argument("--method-name", default=None)
    args = parser.parse_args()

    config = load_phase1_config(args.config)
    output = ensure_dir(cfg_path(config, "paths", "output_dir"))
    prefix = str(args.method_name or config["training"]["checkpoint_prefix"])
    lines = [
        f"# Phase 6 DAEAC Adversarial Report: {prefix}",
        "",
        "## Protocol",
        "",
        f"- Source train/eval: `{config['data']['source_train']}`",
        f"- Target unlabeled adaptation: `{config['data']['target_unlabeled']}`",
        f"- Target test: `{config['data']['target_test']}`",
        f"- Base checkpoint: `{config['training'].get('init_checkpoint')}`",
        f"- Output dir: `{config['paths']['output_dir']}`",
        "",
        "## Training Summary",
        "",
    ]
    summary_path = output / "metrics" / f"{prefix}_train_summary.json"
    if summary_path.exists():
        summary = read_json(summary_path)
        lines.extend(
            [
                f"- Best checkpoint: `{summary.get('best_checkpoint')}`",
                f"- Latest checkpoint: `{summary.get('latest_checkpoint')}`",
                f"- Best epoch: `{summary.get('best_epoch')}`",
                f"- Best source validation macro F1: `{summary.get('best_source_val_macro_f1')}`",
            ]
        )
    else:
        lines.append(f"- Missing train summary: `{summary_path}`")

    lines.extend(["", "## Evaluation Metrics", ""])
    metric_paths = sorted((output / "metrics").glob(f"{prefix}*_metrics.json"))
    if not metric_paths:
        lines.append("- No evaluation metrics found.")
    for path in metric_paths:
        metrics = read_json(path)
        lines.extend(
            [
                f"### {path.stem}",
                "",
                f"- Dataset: `{metrics.get('dataset')}`",
                f"- Accuracy: `{metrics.get('accuracy')}`",
                f"- Macro F1: `{metrics.get('macro_f1')}`",
                "",
                "| Class | Precision | Recall | F1 | Support |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for name, values in metrics.get("per_class", {}).items():
            lines.append(
                f"| {name} | {values.get('precision'):.6f} | {values.get('recall'):.6f} | "
                f"{values.get('f1'):.6f} | {values.get('support')} |"
            )
        lines.append("")

    report_path = output / f"{prefix}_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote report: {report_path}")


if __name__ == "__main__":
    main()

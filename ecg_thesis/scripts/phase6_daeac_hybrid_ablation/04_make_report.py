from __future__ import annotations

import argparse

from common import cfg_path, load_phase1_config
from src.utils.io import read_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_phase1_config(args.config)
    output = cfg_path(config, "paths", "output_dir")
    prefix = str(config["adaptation"]["checkpoint_prefix"])
    summary_path = output / "metrics" / f"{prefix}_train_summary.json"
    summary = read_json(summary_path) if summary_path.exists() else {}
    lines = [
        f"# Phase 6 Hybrid Ablation: {prefix}", "",
        f"- Ablation: `{config['ablation']['name']}`",
        f"- Target adaptation: `{config['data']['target_unlabeled']}`",
        f"- Primary target test: `{config['data']['target_test']}`",
        f"- Init checkpoint: `{config['adaptation']['init_checkpoint']}`",
        f"- Best epoch: `{summary.get('best_epoch')}`",
        f"- Best source validation Macro-F1: `{summary.get('best_val_macro_f1')}`", "",
        "| Checkpoint | Dataset | Accuracy | Macro-F1 | N-F1 | S-F1 | V-F1 | F-F1 |", "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for path in sorted((output / "metrics").glob(f"{prefix}_*_metrics.json")):
        metric = read_json(path)
        per = metric.get("per_class", {})
        lines.append(
            f"| {metric.get('checkpoint_kind')} | {metric.get('dataset')} | {metric.get('accuracy', 0):.6f} | {metric.get('macro_f1', 0):.6f} | "
            f"{per.get('N', {}).get('f1', 0):.6f} | {per.get('S', {}).get('f1', 0):.6f} | {per.get('V', {}).get('f1', 0):.6f} | {per.get('F', {}).get('f1', 0):.6f} |"
        )
    report = output / f"{prefix}_report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {report}")


if __name__ == "__main__":
    main()

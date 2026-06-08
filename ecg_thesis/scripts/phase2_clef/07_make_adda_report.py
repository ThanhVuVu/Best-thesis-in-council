from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import cfg_path, load_phase1_config
from src.utils.io import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase2_clef_adda.yaml")
    args = parser.parse_args()

    config = load_phase1_config(args.config)
    output = ensure_dir(cfg_path(config, "paths", "output_dir"))
    rows = [
        ("CLEF fine-tune source-only", Path("outputs/phase2_clef/metrics/clef_finetune_incart_source_only_metrics.json")),
        ("CLEF-DANN from fine-tune", Path("outputs/phase2_clef_dann/metrics/clef_dann_from_finetune_incart_heldout_metrics.json")),
        ("CLEF-ADDA from fine-tune", output / "metrics" / "clef_adda_from_finetune_incart_heldout_metrics.json"),
    ]
    lines = [
        "# Phase 2 CLEF ADDA Report",
        "",
        "| Method | INCART accuracy | INCART macro-F1 | N-F1 | S-F1 | V-F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, path in rows:
        metrics = _read_metrics(path)
        if metrics is None:
            lines.append(f"| {label} | pending | pending | pending | pending | pending |")
            continue
        per_class = metrics["per_class"]
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    f"{metrics['accuracy']:.4f}",
                    f"{metrics['macro_f1']:.4f}",
                    f"{per_class['N']['f1']:.4f}",
                    f"{per_class['S']['f1']:.4f}",
                    f"{per_class['V']['f1']:.4f}",
                ]
            )
            + " |"
        )
    report_path = output / "phase2_clef_adda_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {report_path}")


def _read_metrics(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

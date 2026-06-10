from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import cfg_path, load_phase1_config
from src.utils.io import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase2_clef_cdan.yaml")
    parser.add_argument("--checkpoint-prefix", default=None)
    args = parser.parse_args()

    config = load_phase1_config(args.config)
    if args.checkpoint_prefix is not None:
        config["training"]["checkpoint_prefix"] = args.checkpoint_prefix
    output = ensure_dir(cfg_path(config, "paths", "output_dir"))
    metrics_dir = output / "metrics"
    prefix = config["training"].get("checkpoint_prefix", "clef_cdan")
    data = {
        "cdan_train": _optional(metrics_dir / f"{prefix}_train_summary.json"),
        "cdan_mitbih": _optional(metrics_dir / f"{prefix}_mitbih_test_metrics.json"),
        "cdan_incart": _optional(metrics_dir / f"{prefix}_incart_heldout_metrics.json"),
        "source_incart": _optional(Path(config["_base_dir"]) / "outputs/phase2_clef/metrics/clef_finetune_incart_source_only_metrics.json"),
        "dann_incart": _optional(Path(config["_base_dir"]) / "outputs/phase2_clef_dann/metrics/clef_dann_from_finetune_incart_heldout_metrics.json"),
    }
    report = [
        "# Phase 2 CLEF CDAN Report",
        "",
        "## CDAN+E Metrics",
        _format_metrics(data["cdan_mitbih"], "MIT-BIH test"),
        _format_metrics(data["cdan_incart"], "INCART held-out"),
        "",
        "## Baseline Comparison",
        _comparison("CLEF source-only", data["source_incart"], data["cdan_incart"]),
        _comparison("CLEF-DANN", data["dann_incart"], data["cdan_incart"]),
        "",
        "## Train Summary",
        _json_block(data["cdan_train"]),
    ]
    report_path = output / "phase2_clef_cdan_report.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote {report_path}")


def _optional(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _format_metrics(metrics, title: str) -> str:
    if metrics is None:
        return f"### {title}\n\nMissing metrics."
    per_class = metrics.get("per_class", {})
    rows = [
        f"### {title}",
        "",
        f"- Accuracy: {metrics.get('accuracy', float('nan')):.4f}",
        f"- Macro-F1: {metrics.get('macro_f1', float('nan')):.4f}",
    ]
    for class_name in ("N", "S", "V"):
        class_metrics = per_class.get(class_name, {})
        if class_metrics:
            rows.append(f"- {class_name} F1: {class_metrics.get('f1', float('nan')):.4f}")
    return "\n".join(rows)


def _comparison(name: str, baseline, cdan) -> str:
    if baseline is None or cdan is None:
        return f"- {name}: missing comparison metrics."
    macro_delta = cdan["macro_f1"] - baseline["macro_f1"]
    s_delta = cdan.get("per_class", {}).get("S", {}).get("f1", 0.0) - baseline.get("per_class", {}).get("S", {}).get("f1", 0.0)
    v_delta = cdan.get("per_class", {}).get("V", {}).get("f1", 0.0) - baseline.get("per_class", {}).get("V", {}).get("f1", 0.0)
    return f"- CDAN vs {name}: Macro-F1 {macro_delta:+.4f}, S-F1 {s_delta:+.4f}, V-F1 {v_delta:+.4f}"


def _json_block(data) -> str:
    if data is None:
        return "Missing train summary."
    return "```json\n" + json.dumps(data, indent=2) + "\n```"


if __name__ == "__main__":
    main()

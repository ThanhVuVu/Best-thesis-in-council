from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"
for path in (ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import cfg_path, load_phase1_config
from src.utils.io import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase6_daeac_feature_debug.yaml")
    parser.add_argument("--merged-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config = load_phase1_config(args.config)
    merged = Path(args.merged_dir) if args.merged_dir else cfg_path(config, "paths", "output_dir") / "merged"
    output = ensure_dir(Path(args.output_dir) if args.output_dir else merged)
    report = output / "phase6_daeac_feature_debug_report.md"
    lines = [
        "# Phase 6 DAEAC Feature Debug Report",
        "",
        "## Available Tables",
        "",
    ]
    for name in [
        "method_layer_collapse_comparison.csv",
        "method_minority_to_N_summary.csv",
        "method_raw_feature_effect_summary.csv",
        "method_linear_probe_summary.csv",
    ]:
        path = merged / name
        rows = _read_csv(path) if path.exists() else []
        lines.append(f"- `{name}`: {len(rows)} rows")
    lines.extend(["", "## Reading Guide", "", "- Lower minority-to-N separability in late layers indicates feature collapse.", "- High N-neighbor fraction for S/V/F indicates minority samples are embedded inside the N neighborhood.", "- Raw feature effect sizes identify whether the missing signal is RR, pre-R, QRS proxy, or post-R morphology."])
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report written to {report}")


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


if __name__ == "__main__":
    main()

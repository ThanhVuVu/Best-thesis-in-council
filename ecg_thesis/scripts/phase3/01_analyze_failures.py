from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import cfg_path, load_phase1_config
from src.utils.io import ensure_dir, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase3_rr_dann.yaml")
    parser.add_argument("--predictions", default="outputs/predictions/dann_incart_heldout_predictions.csv")
    parser.add_argument("--checkpoint", default=None, help="Accepted for CLI symmetry; predictions CSV is used for analysis.")
    args = parser.parse_args()

    config = load_phase1_config(args.config)
    output = ensure_dir(cfg_path(config, "paths", "output_dir"))
    predictions_path = _resolve(config, args.predictions)
    if not predictions_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {predictions_path}. Run Phase 2 DANN evaluation first.")

    df = pd.read_csv(predictions_path)
    summary = {
        "prediction_file": str(predictions_path),
        "total_rows": int(len(df)),
        "critical_confusions": {
            "S_to_N": int(((df["true_class"] == "S") & (df["pred_class"] == "N")).sum()),
            "S_to_V": int(((df["true_class"] == "S") & (df["pred_class"] == "V")).sum()),
            "N_to_V": int(((df["true_class"] == "N") & (df["pred_class"] == "V")).sum()),
            "V_to_N": int(((df["true_class"] == "V") & (df["pred_class"] == "N")).sum()),
        },
        "record_counts_for_errors": {},
    }
    for key, (true_cls, pred_cls) in {
        "S_to_N": ("S", "N"),
        "S_to_V": ("S", "V"),
        "N_to_V": ("N", "V"),
        "V_to_N": ("V", "N"),
    }.items():
        sub = df[(df["true_class"] == true_cls) & (df["pred_class"] == pred_cls)]
        summary["record_counts_for_errors"][key] = sub["record"].astype(str).value_counts().head(20).to_dict() if "record" in sub else {}

    metrics_dir = ensure_dir(output / "metrics")
    write_json(summary, metrics_dir / "phase3_failure_summary.json")
    print(f"Saved failure summary to {metrics_dir / 'phase3_failure_summary.json'}")


def _resolve(config: dict, path: str) -> Path:
    return cfg_path({"paths": {"value": path}, "_base_dir": config["_base_dir"]}, "paths", "value")


if __name__ == "__main__":
    main()

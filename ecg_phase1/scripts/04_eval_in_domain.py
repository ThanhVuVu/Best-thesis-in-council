from __future__ import annotations

import argparse

from common import cfg_path, device_from_torch, load_phase1_config
from src.data.datasets import ECGBeatDataset
from src.training.train import load_model_from_checkpoint
from src.utils.io import ensure_dir, write_json
from scripts_eval_common import evaluate_and_save


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase1.yaml")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best.pt")
    args = parser.parse_args()
    config = load_phase1_config(args.config)
    device = device_from_torch()
    print(f"Using device: {device}")
    model, _ = load_model_from_checkpoint(cfg_path({"paths": {"x": args.checkpoint}, "_base_dir": config["_base_dir"]}, "paths", "x"), device)
    dataset = ECGBeatDataset(cfg_path(config, "paths", "processed_dir") / "mitbih_test.npz", return_metadata=True)
    output = ensure_dir(cfg_path(config, "paths", "output_dir"))
    evaluate_and_save(model, dataset, device, output, "mitbih_test", "in_domain_source_only")


if __name__ == "__main__":
    main()

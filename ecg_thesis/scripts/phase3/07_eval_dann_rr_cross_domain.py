from __future__ import annotations

import argparse

from torch.utils.data import Subset

from common import cfg_path, device_from_torch, load_phase1_config
from scripts_eval_common import evaluate_and_save
from src.data.datasets import ECGBeatRRDataset
from src.training.train_dann import load_dann_from_checkpoint
from src.utils.io import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase3_rr_dann.yaml")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/dann_rr_best.pt")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    config = load_phase1_config(args.config)
    device = device_from_torch()
    print(f"Using device: {device}")
    checkpoint_path = cfg_path({"paths": {"checkpoint": args.checkpoint}, "_base_dir": config["_base_dir"]}, "paths", "checkpoint")
    model, _ = load_dann_from_checkpoint(checkpoint_path, device)
    dataset = ECGBeatRRDataset(cfg_path(config, "data", "target_test"), return_metadata=True)
    name = "dann_rr_incart_heldout"
    if args.max_samples is not None:
        n = min(int(args.max_samples), len(dataset))
        dataset = Subset(dataset, list(range(n)))
        name = f"{name}_max_samples_{n}"
        print(f"Evaluating subset: {n} samples")
    output = ensure_dir(cfg_path(config, "paths", "output_dir"))
    evaluate_and_save(model, dataset, device, output, name, "dann_rr")


if __name__ == "__main__":
    main()

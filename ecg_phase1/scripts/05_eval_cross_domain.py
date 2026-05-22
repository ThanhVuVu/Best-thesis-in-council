from __future__ import annotations

import argparse

from torch.utils.data import Subset

from common import cfg_path, device_from_torch, load_phase1_config
from src.data.datasets import ECGBeatDataset
from src.training.train import load_model_from_checkpoint
from src.utils.io import ensure_dir
from scripts_eval_common import evaluate_and_save


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase1.yaml")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best.pt")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    config = load_phase1_config(args.config)
    device = device_from_torch()
    print(f"Using device: {device}")
    checkpoint_path = cfg_path({"paths": {"checkpoint": args.checkpoint}, "_base_dir": config["_base_dir"]}, "paths", "checkpoint")
    model, _ = load_model_from_checkpoint(checkpoint_path, device)
    dataset = ECGBeatDataset(cfg_path(config, "paths", "processed_dir") / "incart_test.npz", return_metadata=True)
    dataset_name = "incart_test"
    if args.max_samples is not None:
        n = min(int(args.max_samples), len(dataset))
        dataset = Subset(dataset, list(range(n)))
        dataset_name = f"incart_test_max_samples_{n}"
        print(f"Evaluating subset: {n} samples")
    output = ensure_dir(cfg_path(config, "paths", "output_dir"))
    evaluate_and_save(model, dataset, device, output, dataset_name, "cross_domain_source_only")


if __name__ == "__main__":
    main()

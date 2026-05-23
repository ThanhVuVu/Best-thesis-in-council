from __future__ import annotations

import argparse

from torch.utils.data import Subset

from common import cfg_path, device_from_torch, load_phase1_config
from scripts_eval_common import evaluate_and_save
from src.data.datasets import ECGBeatDataset
from src.training.train import load_model_from_checkpoint
from src.utils.io import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase2_dann.yaml")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/source_only_inception_best.pt")
    parser.add_argument("--dataset", choices=["mitbih", "incart", "both"], default="both")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    config = load_phase1_config(args.config)
    device = device_from_torch()
    print(f"Using device: {device}")
    checkpoint_path = cfg_path({"paths": {"checkpoint": args.checkpoint}, "_base_dir": config["_base_dir"]}, "paths", "checkpoint")
    model, _ = load_model_from_checkpoint(checkpoint_path, device)
    output = ensure_dir(cfg_path(config, "paths", "output_dir"))

    if args.dataset in ("mitbih", "both"):
        dataset = ECGBeatDataset(cfg_path(config, "data", "source_test"), return_metadata=True)
        name = "source_only_inception_mitbih_test"
        dataset, name = _maybe_subset(dataset, name, args.max_samples)
        evaluate_and_save(model, dataset, device, output, name, "source_only_inception")

    if args.dataset in ("incart", "both"):
        dataset = ECGBeatDataset(cfg_path(config, "data", "target_test"), return_metadata=True)
        name = "source_only_inception_incart_heldout"
        dataset, name = _maybe_subset(dataset, name, args.max_samples)
        evaluate_and_save(model, dataset, device, output, name, "source_only_inception")


def _maybe_subset(dataset, name: str, max_samples: int | None):
    if max_samples is None:
        return dataset, name
    n = min(int(max_samples), len(dataset))
    print(f"Evaluating subset {name}: {n} samples")
    return Subset(dataset, list(range(n))), f"{name}_max_samples_{n}"


if __name__ == "__main__":
    main()

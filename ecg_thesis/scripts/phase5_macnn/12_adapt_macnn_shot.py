from __future__ import annotations

import argparse

from common import add_wandb_args, apply_wandb_overrides, cfg_path, device_from_torch, load_phase1_config
from src.data.datasets import ECGMACNNDataset
from src.training.train_shot import train_macnn_shot
from src.utils.io import ensure_dir, write_json
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase5_macnn_daeac.yaml")
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--checkpoint-prefix", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--diversity-target", choices=["uniform", "source_prior", "none"], default=None)
    parser.add_argument("--no-pseudo-labeling", action="store_true")
    parser.add_argument("--entropy-weight", type=float, default=None)
    parser.add_argument("--diversity-weight", type=float, default=None)
    add_wandb_args(parser)
    args = parser.parse_args()

    config = load_phase1_config(args.config)
    apply_wandb_overrides(config, args)
    if args.init_checkpoint is not None:
        config["shot"]["init_checkpoint"] = args.init_checkpoint
    if args.checkpoint_prefix is not None:
        config["shot"]["checkpoint_prefix"] = args.checkpoint_prefix
    if args.epochs is not None:
        config["shot"]["epochs"] = int(args.epochs)
    if args.lr is not None:
        config["shot"]["lr"] = float(args.lr)
    if args.beta is not None:
        config["shot"]["beta"] = float(args.beta)
    if args.diversity_target is not None:
        config["shot"]["diversity_target"] = args.diversity_target
    if args.no_pseudo_labeling:
        config["shot"]["pseudo_labeling"] = False
        config["shot"]["beta"] = 0.0
    if args.entropy_weight is not None:
        config["shot"]["entropy_weight"] = float(args.entropy_weight)
    if args.diversity_weight is not None:
        config["shot"]["diversity_weight"] = float(args.diversity_weight)

    set_seed(int(config["seed"]))
    device = device_from_torch()
    target = ECGMACNNDataset(cfg_path(config, "data", "target_unlabeled"))
    output = ensure_dir(cfg_path(config, "paths", "output_dir"))
    summary = train_macnn_shot(target, config, output, device)
    prefix = config["shot"]["checkpoint_prefix"]
    write_json(summary, output / "metrics" / f"{prefix}_train_summary.json")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse

from torch.utils.data import Subset

from common import add_wandb_args, apply_wandb_overrides, cfg_path, device_from_torch, load_phase1_config
from src.data.datasets import ECGBeatDataset
from src.training.train_source_free import train_source_free
from src.utils.io import ensure_dir, write_json
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase4c_ecgfm_top2_sourcefree.yaml")
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-target-samples", type=int, default=None)
    parser.add_argument("--unfreeze-top-ecgfm-layers", type=int, default=None)
    parser.add_argument("--ecgfm-checkpoint", default=None)
    parser.add_argument("--fairseq-signals-path", default=None)
    add_wandb_args(parser)
    args = parser.parse_args()

    config = load_phase1_config(args.config)
    _apply_overrides(config, args)
    apply_wandb_overrides(config, args)
    set_seed(int(config["seed"]))
    if args.epochs is not None:
        config["source_free"]["epochs"] = int(args.epochs)

    device = device_from_torch()
    print(f"Using device: {device}")
    target = ECGBeatDataset(cfg_path(config, "data", "target_unlabeled"))
    if args.max_target_samples is not None:
        target = Subset(target, list(range(min(int(args.max_target_samples), len(target)))))
    print(f"Phase 4C target unlabeled windows: {len(target)}")

    output = ensure_dir(cfg_path(config, "paths", "output_dir"))
    summary = train_source_free(
        target,
        config,
        output,
        device,
        init_checkpoint=args.init_checkpoint,
        unfreeze_top_ecgfm_layers=int(config["source_free"].get("unfreeze_top_ecgfm_layers", 2)),
        model_kwargs_override={
            "ecgfm_checkpoint_path": args.ecgfm_checkpoint,
            "fairseq_signals_path": args.fairseq_signals_path,
        },
    )
    write_json(summary, output / "metrics" / "ecgfm_top2_sourcefree_train_summary.json")


def _apply_overrides(config: dict, args) -> None:
    if args.init_checkpoint is not None:
        config["source_free"]["init_checkpoint"] = args.init_checkpoint
    if args.unfreeze_top_ecgfm_layers is not None:
        config["source_free"]["unfreeze_top_ecgfm_layers"] = int(args.unfreeze_top_ecgfm_layers)
    if args.ecgfm_checkpoint is not None:
        config["model"]["ecgfm_checkpoint_path"] = args.ecgfm_checkpoint
        config["ecgfm"]["checkpoint_path"] = args.ecgfm_checkpoint
    if args.fairseq_signals_path is not None:
        config["model"]["fairseq_signals_path"] = args.fairseq_signals_path
        config["ecgfm"]["fairseq_signals_path"] = args.fairseq_signals_path


if __name__ == "__main__":
    main()

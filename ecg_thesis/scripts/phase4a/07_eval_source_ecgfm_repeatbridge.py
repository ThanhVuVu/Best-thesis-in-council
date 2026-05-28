from __future__ import annotations

import argparse

from torch.utils.data import Subset

from common import add_wandb_args, apply_wandb_overrides, cfg_path, device_from_torch, load_phase1_config
from scripts_eval_common import evaluate_and_save
from src.data.datasets import ECGBeatDataset
from src.training.train import load_model_from_checkpoint
from src.utils.io import ensure_dir
from src.utils.wandb_logging import init_wandb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase4a_ecgfm_repeatbridge.yaml")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/source_only_ecgfm_repeatbridge_best.pt")
    parser.add_argument("--dataset", choices=["mitbih", "incart", "both"], default="both")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--ecgfm-checkpoint", default=None)
    parser.add_argument("--fairseq-signals-path", default=None)
    add_wandb_args(parser)
    args = parser.parse_args()

    config = load_phase1_config(args.config)
    apply_wandb_overrides(config, args)
    device = device_from_torch()
    print(f"Using device: {device}")
    checkpoint_path = cfg_path({"paths": {"checkpoint": args.checkpoint}, "_base_dir": config["_base_dir"]}, "paths", "checkpoint")
    model, _ = load_model_from_checkpoint(
        checkpoint_path,
        device,
        model_kwargs_override={
            "ecgfm_checkpoint_path": args.ecgfm_checkpoint,
            "fairseq_signals_path": args.fairseq_signals_path,
        },
    )
    output = ensure_dir(cfg_path(config, "paths", "output_dir"))
    wandb_run = init_wandb(
        config,
        job_type="eval",
        default_name=f"eval_{checkpoint_path.stem}",
        extra_config={"checkpoint": str(checkpoint_path), "dataset": args.dataset},
    )

    if args.dataset in ("mitbih", "both"):
        dataset = ECGBeatDataset(cfg_path(config, "data", "source_test"), return_metadata=True)
        dataset, name = _maybe_subset(dataset, "source_only_ecgfm_repeatbridge_mitbih_test", args.max_samples)
        evaluate_and_save(model, dataset, device, output, name, "source_only_ecgfm_repeatbridge", wandb_run=wandb_run)

    if args.dataset in ("incart", "both"):
        dataset = ECGBeatDataset(cfg_path(config, "data", "target_test"), return_metadata=True)
        dataset, name = _maybe_subset(dataset, "source_only_ecgfm_repeatbridge_incart_heldout", args.max_samples)
        evaluate_and_save(model, dataset, device, output, name, "source_only_ecgfm_repeatbridge", wandb_run=wandb_run)
    wandb_run.finish()


def _maybe_subset(dataset, name: str, max_samples: int | None):
    if max_samples is None:
        return dataset, name
    n = min(int(max_samples), len(dataset))
    print(f"Evaluating subset {name}: {n} samples")
    return Subset(dataset, list(range(n))), f"{name}_max_samples_{n}"


if __name__ == "__main__":
    main()

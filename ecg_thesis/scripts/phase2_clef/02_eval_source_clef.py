from __future__ import annotations

import argparse

from torch.utils.data import Subset

from common import cfg_path, device_from_torch, load_phase1_config
from scripts_eval_common import evaluate_and_save
from src.data.datasets import ECGBeatDataset
from src.training.train import load_model_from_checkpoint
from src.utils.io import ensure_dir, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase2_clef_source.yaml")
    parser.add_argument("--mode", choices=["clef_frozen", "clef_finetune"], required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--dataset", choices=["mitbih", "incart", "both"], default="both")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--clef-checkpoint", default=None)
    args = parser.parse_args()

    config = load_phase1_config(args.config)
    _apply_mode(config, args.mode)
    if args.clef_checkpoint is not None:
        config["model"]["clef_checkpoint_path"] = args.clef_checkpoint
    device = device_from_torch()
    print(f"Using device: {device}")

    checkpoint = args.checkpoint or f"outputs/phase2_clef/checkpoints/{args.mode}_best.pt"
    checkpoint_path = cfg_path({"paths": {"checkpoint": checkpoint}, "_base_dir": config["_base_dir"]}, "paths", "checkpoint")
    model, ckpt = load_model_from_checkpoint(
        checkpoint_path,
        device,
        model_kwargs_override={"clef_checkpoint_path": args.clef_checkpoint},
    )
    output = ensure_dir(cfg_path(config, "paths", "output_dir"))
    summary = {"checkpoint": str(checkpoint_path), "mode": args.mode, "datasets": {}, "checkpoint_epoch": ckpt.get("epoch")}

    if args.dataset in ("mitbih", "both"):
        dataset = ECGBeatDataset(cfg_path(config, "data", "source_test"), return_metadata=True)
        dataset, name = _maybe_subset(dataset, f"{args.mode}_mitbih_test", args.max_samples)
        summary["datasets"]["mitbih_test"] = evaluate_and_save(model, dataset, device, output, name, args.mode)

    if args.dataset in ("incart", "both"):
        dataset = ECGBeatDataset(cfg_path(config, "data", "target_test"), return_metadata=True)
        dataset, name = _maybe_subset(dataset, f"{args.mode}_incart_source_only", args.max_samples)
        summary["datasets"]["incart_source_only"] = evaluate_and_save(model, dataset, device, output, name, args.mode)

    write_json(summary, output / "metrics" / f"{args.mode}_eval_summary.json")


def _apply_mode(config: dict, mode: str) -> None:
    mode_cfg = dict(config["modes"][mode])
    config["model"]["freeze_encoder"] = bool(mode_cfg["freeze_encoder"])
    config["model"]["encoder_lr"] = float(mode_cfg["encoder_lr"])
    config["source_only"]["checkpoint_prefix"] = str(mode_cfg["checkpoint_prefix"])
    config["source_only"]["lr"] = float(mode_cfg["lr"])


def _maybe_subset(dataset, name: str, max_samples: int | None):
    if max_samples is None:
        return dataset, name
    n = min(int(max_samples), len(dataset))
    print(f"Evaluating subset {name}: {n} samples")
    return Subset(dataset, list(range(n))), f"{name}_max_samples_{n}"


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from common import add_wandb_args, apply_wandb_overrides, cfg_path, device_from_torch, load_phase1_config
from src.data.daeac_dataset import DAEACDataset, subset_first
from src.training.train_daeac_adversarial import evaluate_daeac_adversarial_model, load_daeac_adversarial_checkpoint
from src.utils.io import ensure_dir, write_json
from src.utils.wandb_logging import init_wandb, log_eval_metrics
from workflow import apply_domain_pair


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase6_daeac_dann.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--method-name", default="daeac_adversarial")
    parser.add_argument("--dataset", default="target", help="source, target, both, external, all, or an external key such as incart/svdb.")
    parser.add_argument("--domain-pair", choices=["ds1_ds2", "ds1_incart", "ds1_svdb", "mitbih_incart", "mitbih_svdb"], default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    add_wandb_args(parser)
    args = parser.parse_args()

    config = load_phase1_config(args.config)
    checkpoint_name = Path(args.checkpoint).name.lower()
    method = "adda" if "adda" in checkpoint_name else "cdan" if "cdan" in checkpoint_name else "dann"
    apply_domain_pair(config, args.domain_pair, method)
    apply_wandb_overrides(config, args)
    output = ensure_dir(cfg_path(config, "paths", "output_dir"))
    device = device_from_torch()
    model, checkpoint = load_daeac_adversarial_checkpoint(args.checkpoint, device)
    wandb_run = init_wandb(
        config,
        job_type="eval_daeac_adversarial",
        default_name=f"{args.method_name}_eval",
        extra_config={"checkpoint": str(args.checkpoint), "checkpoint_method": checkpoint.get("method"), "dataset": args.dataset},
    )

    input_key = str(config["data"].get("input_key", "auto"))
    label_key = str(config["data"].get("label_key", "y"))
    class_names = list(config["data"]["class_names"])
    datasets = _eval_datasets(config, args.dataset)
    if not datasets:
        raise FileNotFoundError(f"No evaluation datasets available for selection {args.dataset!r}.")

    for name, path in datasets:
        ds = DAEACDataset(path, input_key=input_key, label_key=label_key, class_names=class_names)
        ds = subset_first(ds, args.max_samples)
        loader = DataLoader(ds, batch_size=int(config["evaluation"]["batch_size"]), shuffle=False, num_workers=0)
        result = evaluate_daeac_adversarial_model(model, loader, device, class_names, desc=f"{args.method_name} {name}")
        stem = f"{args.method_name}_{name}"
        if args.max_samples is not None:
            stem += f"_max_samples_{args.max_samples}"
        metrics = dict(result["metrics"])
        metrics["dataset"] = name
        metrics["setting"] = args.method_name
        metrics["checkpoint"] = str(args.checkpoint)
        metrics["checkpoint_method"] = checkpoint.get("method")
        write_json(metrics, output / "metrics" / f"{stem}_metrics.json")
        _write_predictions(output / "predictions" / f"{stem}_predictions.csv", result, class_names)
        _write_confusion(output / "metrics" / f"{stem}_confusion_matrix.csv", metrics["confusion_matrix"], class_names)
        log_eval_metrics(wandb_run, metrics, prefix=f"eval/{args.method_name}/{name}")
        print(stem, metrics["paper_metrics"])
    wandb_run.finish()


def _eval_datasets(config: dict, dataset: str) -> list[tuple[str, Path]]:
    external = dict(config.get("data", {}).get("external_targets", {}))
    selected: list[tuple[str, Path]] = []
    if dataset in {"source", "both", "all"}:
        selected.append(("source_eval", cfg_path(config, "data", "source_eval")))
    if dataset in {"target", "both", "all"}:
        selected.append(("target_test", cfg_path(config, "data", "target_test")))
    if dataset in {"external", "all"}:
        for name, value in external.items():
            _append_if_exists(config, selected, name, value)
    elif dataset in external:
        _append_if_exists(config, selected, dataset, external[dataset])
    if not selected:
        valid = ["source", "target", "both", "external", "all", *external.keys()]
        raise ValueError(f"Unknown or unavailable dataset '{dataset}'. Valid values: {valid}")
    return selected


def _append_if_exists(config: dict, selected: list[tuple[str, Path]], name: str, value: str | Path) -> None:
    path = _resolve_data_path(config, value)
    if path.exists():
        selected.append((name, path))
    else:
        print(f"Skipping missing optional external target {name}: {path}")


def _resolve_data_path(config: dict, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(config["_base_dir"]) / path


def _write_predictions(path: Path, result: dict, class_names: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["y_true", "y_pred", "true_class", "pred_class", *[f"prob_{name}" for name in class_names]])
        for y_true, y_pred, prob in zip(result["y_true"], result["y_pred"], result["probabilities"]):
            writer.writerow([int(y_true), int(y_pred), class_names[int(y_true)], class_names[int(y_pred)], *[float(v) for v in np.asarray(prob)]])


def _write_confusion(path: Path, matrix: list[list[int]], class_names: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred", *class_names])
        for name, row in zip(class_names, matrix):
            writer.writerow([name, *row])


if __name__ == "__main__":
    main()

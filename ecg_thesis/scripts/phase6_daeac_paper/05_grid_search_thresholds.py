from __future__ import annotations

import argparse
import copy
import csv
import itertools
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from common import add_wandb_args, apply_wandb_overrides, cfg_path, device_from_torch, load_phase1_config
from src.data.daeac_dataset import DAEACDataset, DAEACTargetUnlabeledDataset, subset_first
from src.training.train_daeac_paper import adapt_daeac, evaluate_daeac_model, load_daeac_checkpoint
from src.utils.io import ensure_dir, read_json, write_json
from src.utils.seed import set_seed


DEFAULT_GRID = {
    "N": [0.985, 0.99, 0.995],
    "S": [0.993, 0.995, 0.997],
    "V": [0.993, 0.995, 0.997],
    "F": [0.9985, 0.999, 0.9995],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase6_daeac_paper.yaml")
    parser.add_argument("--init-checkpoint", default="outputs/phase6_daeac_paper/checkpoints/daeac_base_best.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--checkpoint-prefix", default="daeac_uda_grid")
    parser.add_argument("--score-checkpoint", choices=["latest", "best"], default="latest")
    parser.add_argument("--n-thresholds", nargs="+", type=float, default=None)
    parser.add_argument("--s-thresholds", nargs="+", type=float, default=None)
    parser.add_argument("--v-thresholds", nargs="+", type=float, default=None)
    parser.add_argument("--f-thresholds", nargs="+", type=float, default=None)
    parser.add_argument("--max-source-samples", type=int, default=None)
    parser.add_argument("--max-target-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--limit-trials", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    add_wandb_args(parser)
    args = parser.parse_args()

    base_config = load_phase1_config(args.config)
    apply_wandb_overrides(base_config, args)
    set_seed(int(base_config["seed"]))
    device = device_from_torch()
    output = ensure_dir(cfg_path(base_config, "paths", "output_dir"))
    search_dir = ensure_dir(output / "search" / args.checkpoint_prefix)
    results_csv = search_dir / "threshold_grid_results.csv"

    grid = {
        "N": args.n_thresholds or DEFAULT_GRID["N"],
        "S": args.s_thresholds or DEFAULT_GRID["S"],
        "V": args.v_thresholds or DEFAULT_GRID["V"],
        "F": args.f_thresholds or DEFAULT_GRID["F"],
    }
    trials = list(_grid_trials(grid))
    if args.limit_trials is not None:
        trials = trials[: int(args.limit_trials)]
    print(f"Threshold grid trials: {len(trials)}")
    print("WARNING: this search evaluates target_test after each trial; use it for diagnostics, not paper-faithful final selection.")
    if args.dry_run:
        for idx, thresholds in enumerate(trials):
            print(idx, thresholds)
        return

    input_key = str(base_config["data"].get("input_key", "auto"))
    label_key = str(base_config["data"].get("label_key", "y"))
    class_names = list(base_config["data"]["class_names"])

    source_ds = DAEACDataset(cfg_path(base_config, "data", "source_train"), input_key=input_key, label_key=label_key, class_names=class_names)
    val_ds = DAEACDataset(cfg_path(base_config, "data", "source_eval"), input_key=input_key, label_key=label_key, class_names=class_names)
    target_unlabeled_ds = DAEACTargetUnlabeledDataset(
        cfg_path(base_config, "data", "target_unlabeled"),
        input_key=input_key,
        label_key=label_key,
        class_names=class_names,
    )
    target_test_ds = DAEACDataset(cfg_path(base_config, "data", "target_test"), input_key=input_key, label_key=label_key, class_names=class_names)
    source_ds = subset_first(source_ds, args.max_source_samples)
    val_ds = subset_first(val_ds, args.max_val_samples)
    target_unlabeled_ds = subset_first(target_unlabeled_ds, args.max_target_samples)
    target_test_ds = subset_first(target_test_ds, args.max_eval_samples)
    target_loader = DataLoader(target_test_ds, batch_size=int(base_config["evaluation"]["batch_size"]), shuffle=False, num_workers=0)

    rows: list[dict[str, Any]] = _read_existing_rows(results_csv) if args.resume else []
    done = {(float(r["N"]), float(r["S"]), float(r["V"]), float(r["F"])) for r in rows}

    for idx, thresholds in enumerate(trials):
        key = (thresholds["N"], thresholds["S"], thresholds["V"], thresholds["F"])
        trial_name = f"{args.checkpoint_prefix}_{idx:03d}_{_threshold_suffix(thresholds)}"
        latest_metrics_path = search_dir / f"{trial_name}_latest_target_metrics.json"
        best_metrics_path = search_dir / f"{trial_name}_best_target_metrics.json"
        if args.resume and key in done and latest_metrics_path.exists() and best_metrics_path.exists():
            print(f"[{idx + 1}/{len(trials)}] skip existing {trial_name}")
            continue

        print(f"[{idx + 1}/{len(trials)}] run {trial_name}: {thresholds}")
        config = copy.deepcopy(base_config)
        config["adaptation"]["epochs"] = int(args.epochs)
        config["adaptation"]["init_checkpoint"] = str(args.init_checkpoint)
        config["adaptation"]["checkpoint_prefix"] = trial_name
        config["adaptation"]["pseudo_thresholds"] = dict(thresholds)

        trial_summary = adapt_daeac(source_ds, val_ds, target_unlabeled_ds, config, output, device)
        write_json(trial_summary, search_dir / f"{trial_name}_train_summary.json")

        latest_metrics = _eval_checkpoint(trial_summary["latest_checkpoint"], config, target_loader, device, class_names)
        best_metrics = _eval_checkpoint(trial_summary["best_checkpoint"], config, target_loader, device, class_names)
        write_json(latest_metrics, latest_metrics_path)
        write_json(best_metrics, best_metrics_path)

        row = _result_row(idx, trial_name, thresholds, trial_summary, latest_metrics, best_metrics, args.score_checkpoint)
        rows = [r for r in rows if not (float(r["N"]), float(r["S"]), float(r["V"]), float(r["F"])) == key]
        rows.append(row)
        rows = sorted(rows, key=lambda r: float(r["score"]), reverse=True)
        _write_results(results_csv, rows)
        _write_best_yaml(search_dir / "best_thresholds.yaml", rows[0])
        print(
            f"{trial_name}: latest_macro_f1={latest_metrics['macro_f1']:.4f} "
            f"latest_acc={latest_metrics['accuracy']:.4f} best_macro_f1={best_metrics['macro_f1']:.4f} "
            f"score={row['score']:.4f}"
        )


def _grid_trials(grid: dict[str, list[float]]) -> list[dict[str, float]]:
    names = ["N", "S", "V", "F"]
    return [dict(zip(names, values)) for values in itertools.product(*(grid[name] for name in names))]


def _threshold_suffix(thresholds: dict[str, float]) -> str:
    return "_".join(f"{name}{str(value).replace('.', 'p')}" for name, value in thresholds.items())


def _eval_checkpoint(checkpoint: str | Path, config: dict[str, Any], loader: DataLoader, device, class_names: list[str]) -> dict[str, Any]:
    model = load_daeac_checkpoint(checkpoint, config, device)
    metrics = dict(evaluate_daeac_model(model, loader, device, class_names)["metrics"])
    metrics["checkpoint"] = str(checkpoint)
    return metrics


def _result_row(
    idx: int,
    trial_name: str,
    thresholds: dict[str, float],
    summary: dict[str, Any],
    latest_metrics: dict[str, Any],
    best_metrics: dict[str, Any],
    score_checkpoint: str,
) -> dict[str, Any]:
    scored = latest_metrics if score_checkpoint == "latest" else best_metrics
    latest_per = latest_metrics["per_class"]
    best_per = best_metrics["per_class"]
    final_history = summary["history"][-1] if summary.get("history") else {}
    return {
        "rank": "",
        "trial": idx,
        "trial_name": trial_name,
        "N": thresholds["N"],
        "S": thresholds["S"],
        "V": thresholds["V"],
        "F": thresholds["F"],
        "score_checkpoint": score_checkpoint,
        "score": float(scored["macro_f1"]),
        "latest_accuracy": float(latest_metrics["accuracy"]),
        "latest_macro_f1": float(latest_metrics["macro_f1"]),
        "latest_N_f1": float(latest_per["N"]["f1"]),
        "latest_S_f1": float(latest_per["S"]["f1"]),
        "latest_V_f1": float(latest_per["V"]["f1"]),
        "latest_F_f1": float(latest_per["F"]["f1"]),
        "best_accuracy": float(best_metrics["accuracy"]),
        "best_macro_f1": float(best_metrics["macro_f1"]),
        "best_N_f1": float(best_per["N"]["f1"]),
        "best_S_f1": float(best_per["S"]["f1"]),
        "best_V_f1": float(best_per["V"]["f1"]),
        "best_F_f1": float(best_per["F"]["f1"]),
        "source_best_epoch": int(summary["best_epoch"]),
        "source_best_val_macro_f1": float(summary["best_val_macro_f1"]),
        "final_pseudo_counts": final_history.get("pseudo_counts", []),
        "latest_checkpoint": summary["latest_checkpoint"],
        "best_checkpoint": summary["best_checkpoint"],
    }


def _read_existing_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_results(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_best_yaml(path: Path, row: dict[str, Any]) -> None:
    text = "\n".join(
        [
            "adaptation:",
            "  pseudo_thresholds:",
            f"    N: {row['N']}",
            f"    S: {row['S']}",
            f"    V: {row['V']}",
            f"    F: {row['F']}",
            f"  selected_trial: {row['trial_name']}",
            f"  selected_score_checkpoint: {row['score_checkpoint']}",
            f"  selected_score_macro_f1: {row['score']}",
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()

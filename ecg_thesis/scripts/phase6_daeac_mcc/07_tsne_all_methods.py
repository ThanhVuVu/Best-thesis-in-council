from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import cfg_path, device_from_torch, load_phase1_config
from src.data.daeac_dataset import DAEACDataset, subset_first
from src.training.train_daeac_adversarial import evaluate_daeac_adversarial_model, load_daeac_adversarial_checkpoint
from src.training.train_daeac_paper import evaluate_daeac_model, load_daeac_checkpoint
from src.utils.io import ensure_dir, write_json


@dataclass(frozen=True)
class RunSpec:
    scenario: str
    method: str
    config: Path
    checkpoint: Path
    checkpoint_type: str
    stage: str


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Post-hoc t-SNE visualization for source-only, DANN, CDAN, MCC, "
            "DAN-MKMMD, and hybrid MKMMD+MCC checkpoints."
        )
    )
    parser.add_argument("--manifest", required=True, help="CSV with scenario,method,config,checkpoint[,checkpoint_type,stage].")
    parser.add_argument("--output-dir", default="outputs/phase6_daeac_fcba_latefusion_rr_nsv_tsne_all_methods")
    parser.add_argument("--source-dataset", default="source_eval", choices=["source_eval", "source_train"])
    parser.add_argument("--target-dataset", default="target_test", choices=["target_test", "target_val"])
    parser.add_argument("--max-source-samples", type=int, default=1500)
    parser.add_argument("--max-target-samples", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()

    specs = _read_manifest(args.manifest)
    if not specs:
        raise ValueError(f"No checkpoint rows found in manifest: {args.manifest}")

    device = device_from_torch()
    output_root = ensure_dir(args.output_dir)
    summary: dict[str, list[dict[str, Any]]] = {}
    cached: dict[tuple[str, str], dict[str, Any]] = {}

    for spec in specs:
        print(f"Extracting {spec.scenario} / {spec.method}: {spec.checkpoint}")
        result = _extract_source_target(spec, args, device)
        cached[(spec.scenario, spec.method)] = result
        scenario_dir = ensure_dir(output_root / spec.scenario)
        figure_path = scenario_dir / f"{_safe_name(spec.stage)}_{_safe_name(spec.method)}_tsne.png"
        _plot_tsne_panel(result, figure_path, seed=args.seed, dpi=args.dpi, title=f"{_stage_title(spec.stage)}: {spec.method}")
        summary.setdefault(spec.scenario, []).append(_summary_row(spec, result, figure_path))

    for scenario, rows in summary.items():
        scenario_specs = _ordered_scenario_specs(specs, scenario)
        scenario_results = [(spec, cached[(spec.scenario, spec.method)]) for spec in scenario_specs]
        _plot_grid(scenario, scenario_results, output_root / scenario / "before_after_all_methods_tsne_grid.png", seed=args.seed, dpi=args.dpi)

    write_json(
        {
            "manifest": str(args.manifest),
            "source_dataset": args.source_dataset,
            "target_dataset": args.target_dataset,
            "max_source_samples": args.max_source_samples,
            "max_target_samples": args.max_target_samples,
            "scenarios": summary,
        },
        output_root / "tsne_all_methods_summary.json",
    )
    print(f"t-SNE figures written under {output_root}")


def _read_manifest(path: str | Path) -> list[RunSpec]:
    specs: list[RunSpec] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            scenario = str(row.get("scenario", "")).strip()
            method = str(row.get("method", "")).strip()
            config = str(row.get("config", "")).strip()
            checkpoint = str(row.get("checkpoint", "")).strip()
            checkpoint_type = str(row.get("checkpoint_type", "") or _infer_checkpoint_type(method, checkpoint)).strip().lower()
            stage = str(row.get("stage", "") or _infer_stage(method)).strip().lower()
            if not scenario or not method or not config or not checkpoint:
                raise ValueError(f"Manifest rows need scenario, method, config, checkpoint. Bad row: {row}")
            if checkpoint_type not in {"daeac", "adversarial"}:
                raise ValueError(f"checkpoint_type must be daeac or adversarial, got {checkpoint_type!r} in row {row}")
            if stage not in {"before", "after"}:
                raise ValueError(f"stage must be before or after, got {stage!r} in row {row}")
            specs.append(
                RunSpec(
                    scenario=scenario,
                    method=method,
                    config=Path(config),
                    checkpoint=Path(checkpoint),
                    checkpoint_type=checkpoint_type,
                    stage=stage,
                )
            )
    return specs


def _infer_checkpoint_type(method: str, checkpoint: str) -> str:
    text = f"{method} {checkpoint}".lower()
    return "adversarial" if any(name in text for name in ("dann", "cdan")) else "daeac"


def _infer_stage(method: str) -> str:
    text = method.lower().replace("-", "_").replace(" ", "_")
    return "before" if any(name in text for name in ("source_only", "srconly", "src_only", "base")) else "after"


def _ordered_scenario_specs(specs: list[RunSpec], scenario: str) -> list[RunSpec]:
    selected = [spec for spec in specs if spec.scenario == scenario]
    return sorted(selected, key=lambda spec: (0 if spec.stage == "before" else 1, _method_order(spec.method), spec.method.lower()))


def _method_order(method: str) -> int:
    text = method.lower().replace("-", "_").replace(" ", "_")
    order = {
        "source_only": 0,
        "srconly": 0,
        "src_only": 0,
        "base": 0,
        "dann": 1,
        "cdan": 2,
        "mcc": 3,
        "dan_mkmmd": 4,
        "mkmmd": 4,
        "hybrid_mkmmd_mcc": 5,
        "hybrid": 5,
    }
    for key, value in order.items():
        if key in text:
            return value
    return 99


def _extract_source_target(spec: RunSpec, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    config = load_phase1_config(str(spec.config))
    class_names = list(config["data"]["class_names"])
    batch_size = int(args.batch_size or config.get("evaluation", {}).get("batch_size", 256))
    dataset_kwargs = _dataset_kwargs(config, class_names)
    source_ds = subset_first(
        DAEACDataset(cfg_path(config, "data", args.source_dataset), **dataset_kwargs),
        args.max_source_samples,
    )
    target_ds = subset_first(
        DAEACDataset(cfg_path(config, "data", args.target_dataset), **dataset_kwargs),
        args.max_target_samples,
    )
    source_loader = DataLoader(source_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    target_loader = DataLoader(target_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    if spec.checkpoint_type == "adversarial":
        model, checkpoint = load_daeac_adversarial_checkpoint(spec.checkpoint, device)
        source = evaluate_daeac_adversarial_model(model, source_loader, device, class_names, desc=f"{spec.method} source")
        target = evaluate_daeac_adversarial_model(model, target_loader, device, class_names, desc=f"{spec.method} target")
        checkpoint_method = checkpoint.get("method")
    else:
        model = load_daeac_checkpoint(spec.checkpoint, config, device)
        source = evaluate_daeac_model(model, source_loader, device, class_names)
        target = evaluate_daeac_model(model, target_loader, device, class_names)
        checkpoint_method = "daeac"

    features = np.concatenate([source["features"], target["features"]], axis=0)
    labels = np.concatenate([source["y_true"], target["y_true"]], axis=0)
    preds = np.concatenate([source["y_pred"], target["y_pred"]], axis=0)
    domains = np.asarray(["source"] * len(source["y_true"]) + ["target"] * len(target["y_true"]))
    return {
        "features": features,
        "labels": labels,
        "preds": preds,
        "domains": domains,
        "class_names": class_names,
        "checkpoint_method": checkpoint_method,
        "source_metrics": source["metrics"],
        "target_metrics": target["metrics"],
    }


def _dataset_kwargs(config: dict[str, Any], class_names: list[str]) -> dict[str, Any]:
    data_cfg = dict(config.get("data", {}))
    return {
        "input_key": str(data_cfg.get("input_key", "auto")),
        "label_key": str(data_cfg.get("label_key", "y")),
        "class_names": class_names,
        "rr_mode": str(data_cfg.get("rr_mode", "real")),
        "rr_features_key": str(data_cfg.get("rr_features_key", "rr_features")),
        "return_rr_features": bool(data_cfg.get("return_rr_features", False)),
        "morphology_only": bool(data_cfg.get("morphology_only", False)),
    }


def _plot_tsne_panel(result: dict[str, Any], path: Path, seed: int, dpi: int, title: str) -> None:
    coords = _embed(result["features"], seed)
    class_names = result["class_names"]
    domains = result["domains"]
    labels = result["labels"]
    ensure_dir(path.parent)
    plt.figure(figsize=(7.5, 6.0))
    _scatter_domains(coords, domains, labels, class_names)
    plt.title(title)
    plt.xlabel("t-SNE-1")
    plt.ylabel("t-SNE-2")
    plt.legend(fontsize=7, markerscale=1.5, ncol=2)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()


def _plot_grid(scenario: str, results: list[tuple[RunSpec, dict[str, Any]]], path: Path, seed: int, dpi: int) -> None:
    if not results:
        return
    cols = min(3, len(results))
    rows = int(np.ceil(len(results) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 4.4 * rows), squeeze=False)
    coords_by_method = _joint_embeddings(results, seed)
    xlim, ylim = _shared_limits(list(coords_by_method.values()))
    for ax, (spec, result) in zip(axes.ravel(), results):
        coords = coords_by_method[spec.method]
        _scatter_domains(coords, result["domains"], result["labels"], result["class_names"], ax=ax)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{_stage_title(spec.stage)}\n{spec.method}")
        ax.set_xlabel("t-SNE-1")
        ax.set_ylabel("t-SNE-2")
    for ax in axes.ravel()[len(results) :]:
        ax.axis("off")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", fontsize=8, markerscale=1.5, ncol=min(6, len(labels)))
    fig.suptitle(f"{scenario} before vs after adaptation, joint t-SNE", y=0.98)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    ensure_dir(path.parent)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _joint_embeddings(results: list[tuple[RunSpec, dict[str, Any]]], seed: int) -> dict[str, np.ndarray]:
    features = [result["features"] for _, result in results]
    dims = {int(feature.shape[1]) for feature in features if feature.ndim == 2}
    if len(dims) != 1:
        raise ValueError(f"Joint t-SNE requires equal feature dimensions within a scenario, got {sorted(dims)}.")
    combined = np.concatenate(features, axis=0)
    coords = _embed(combined, seed)
    coords_by_method: dict[str, np.ndarray] = {}
    start = 0
    for spec, result in results:
        stop = start + len(result["features"])
        coords_by_method[spec.method] = coords[start:stop]
        start = stop
    return coords_by_method


def _shared_limits(coords_list: list[np.ndarray]) -> tuple[tuple[float, float], tuple[float, float]]:
    coords = np.concatenate(coords_list, axis=0)
    if coords.size == 0:
        return (-1.0, 1.0), (-1.0, 1.0)
    x_min, y_min = np.min(coords, axis=0)
    x_max, y_max = np.max(coords, axis=0)
    x_pad = max((float(x_max) - float(x_min)) * 0.05, 1.0)
    y_pad = max((float(y_max) - float(y_min)) * 0.05, 1.0)
    return (float(x_min) - x_pad, float(x_max) + x_pad), (float(y_min) - y_pad, float(y_max) + y_pad)


def _embed(features: np.ndarray, seed: int) -> np.ndarray:
    if len(features) < 2:
        return np.zeros((len(features), 2), dtype=np.float32)
    n_components = min(50, int(features.shape[1]), max(1, len(features) - 1))
    reduced = PCA(n_components=n_components, random_state=seed).fit_transform(features) if features.shape[1] > n_components else features
    perplexity = max(2, min(30, (len(reduced) - 1) // 3))
    return TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=perplexity, random_state=seed).fit_transform(reduced)


def _scatter_domains(
    coords: np.ndarray,
    domains: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    ax: plt.Axes | None = None,
) -> None:
    ax = ax or plt.gca()
    markers = {"source": "o", "target": "^"}
    colors = {"N": "tab:blue", "S": "tab:orange", "V": "tab:green"}
    for domain in ("source", "target"):
        for cls, class_name in enumerate(class_names):
            mask = (domains == domain) & (labels == cls)
            if not np.any(mask):
                continue
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=12,
                alpha=0.72,
                linewidths=0,
                marker=markers[domain],
                color=colors.get(class_name, "gray"),
                label=f"{domain}-{class_name}",
            )


def _summary_row(spec: RunSpec, result: dict[str, Any], figure_path: Path) -> dict[str, Any]:
    labels = result["labels"]
    domains = result["domains"]
    return {
        "method": spec.method,
        "stage": spec.stage,
        "config": str(spec.config),
        "checkpoint": str(spec.checkpoint),
        "checkpoint_type": spec.checkpoint_type,
        "checkpoint_method": result["checkpoint_method"],
        "figure": str(figure_path),
        "source_samples": int(np.sum(domains == "source")),
        "target_samples": int(np.sum(domains == "target")),
        "class_counts": {
            name: int(np.sum(labels == idx))
            for idx, name in enumerate(result["class_names"])
        },
        "source_macro_f1": float(result["source_metrics"]["macro_f1"]),
        "target_macro_f1": float(result["target_metrics"]["macro_f1"]),
    }


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip().lower())


def _stage_title(stage: str) -> str:
    return "Before adaptation" if stage == "before" else "After adaptation"


if __name__ == "__main__":
    main()

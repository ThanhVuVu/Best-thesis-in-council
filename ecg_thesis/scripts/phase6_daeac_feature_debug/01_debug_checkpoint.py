from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"
for path in (ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import cfg_path, device_from_torch, load_phase1_config
from src.data.daeac_raw_debug import load_labeled_daeac_dataset, processed_input_features, sample_ids, select_stratified_debug_indices
from src.models.daeac_adversarial import DAEACADDAModel
from src.training.train_daeac_adversarial import load_daeac_adversarial_checkpoint
from src.training.train_daeac_paper import daeac_metrics, load_daeac_checkpoint
from src.utils.daeac_feature_debug_metrics import (
    effect_size_rows,
    knn_purity_rows,
    linear_probe_rows,
    nearest_neighbor_rows,
    pairwise_separability_rows,
    temporal_contrast_rows,
)
from src.utils.io import ensure_dir, write_json
from src.visualization.daeac_feature_debug import (
    plot_feature_effect_heatmap,
    plot_layer_collapse,
    plot_pca_embeddings,
    plot_temporal_contrast,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase6_daeac_feature_debug.yaml")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-type", choices=["base", "adversarial", "auto"], default="auto")
    parser.add_argument("--method-name", required=True)
    parser.add_argument("--processed-dir", default=None)
    parser.add_argument("--raw-cache-dir", default=None)
    parser.add_argument("--dataset", default="target", help="target, incart, svdb, or all")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--target-max-samples", type=int, default=None)
    parser.add_argument("--incart-max-samples", type=int, default=None)
    parser.add_argument("--svdb-max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--adda-encoder", choices=["source", "target"], default="target")
    parser.add_argument("--allow-missing-raw", action="store_true")
    args = parser.parse_args()

    config = load_phase1_config(args.config)
    model_config = load_phase1_config(args.model_config)
    device = device_from_torch()
    output = ensure_dir(Path(args.output_dir) if args.output_dir else cfg_path(config, "paths", "output_dir") / args.method_name)
    figures = ensure_dir(output / "figures")
    processed_dir = Path(args.processed_dir) if args.processed_dir else cfg_path(config, "paths", "processed_dir")
    batch_size = int(args.batch_size or config["analysis"]["batch_size"])
    class_names = list(config["data"]["class_names"])

    model, checkpoint_info = _load_model(args, model_config, device)
    summary: dict[str, Any] = {
        "method_name": args.method_name,
        "checkpoint": str(args.checkpoint),
        "checkpoint_type": args.checkpoint_type,
        "model_config": str(args.model_config),
        "datasets": {},
    }

    combined_pairwise = []
    combined_knn = []
    combined_probe = []
    combined_effect = []
    for dataset_key in _selected_datasets(args.dataset):
        ds = load_labeled_daeac_dataset(
            processed_dir,
            dataset_key,
            input_key=str(config["data"].get("input_key", "auto")),
            label_key=str(config["data"].get("label_key", "y")),
            class_names=class_names,
        )
        max_samples = _dataset_max_samples(dataset_key, args, config["analysis"])
        indices = select_stratified_debug_indices(ds.y, max_samples=max_samples, random_seed=int(config["analysis"].get("random_seed", 42)))
        subset = Subset(ds, indices.tolist())
        loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
        result = _collect_model_outputs(model, loader, device, args.adda_encoder)
        y = ds.y[indices] if ds.y is not None else np.zeros(len(indices), dtype=np.int64)
        metadata = [ds.metadata(int(idx)) for idx in indices]
        ids = sample_ids(dataset_key, ds, indices)

        stem = f"{args.method_name}_{dataset_key}"
        _write_error_groups(output / f"{stem}_error_groups.csv", ids, metadata, y, result["y_pred"], result["probabilities"], class_names)
        raw_rows = _join_raw_cache(args.raw_cache_dir, ids, y, result["y_pred"], class_names, args.allow_missing_raw)
        if raw_rows:
            _write_csv(output / f"{stem}_raw_joined_error_groups.csv", raw_rows)

        processed_features = processed_input_features(ds, indices)
        effect_rows = effect_size_rows(processed_features, y, class_names)
        if raw_rows:
            raw_feature_names = [key for key in raw_rows[0].keys() if key.endswith("_proxy") or key.endswith("_ratio") or key.endswith("_energy") or key.endswith("_area")]
            raw_features = {key: np.asarray([float(row.get(key, 0.0) or 0.0) for row in raw_rows], dtype=np.float32) for key in raw_feature_names}
            effect_rows.extend(effect_size_rows(raw_features, y, class_names))
        temporal_rows = temporal_contrast_rows(ds.x[indices, 0, 0, :], y, class_names)
        pairwise_rows = pairwise_separability_rows(result["features_by_layer"], y, class_names)
        knn_rows = knn_purity_rows(result["features_by_layer"], y, class_names, int(config["analysis"]["knn_k"]))
        probe_rows = linear_probe_rows(
            result["features_by_layer"],
            y,
            class_names,
            float(config["analysis"]["probe_test_size"]),
            int(config["analysis"]["random_seed"]),
            int(config["analysis"]["probe_max_iter"]),
        )
        nn_rows = nearest_neighbor_rows(
            result["features_by_layer"]["gap_embed"],
            y,
            result["y_pred"],
            metadata,
            class_names,
            int(config["analysis"]["nearest_k"]),
        )

        _write_csv(output / f"{stem}_clinical_proxy_effect_size.csv", effect_rows)
        _write_csv(output / f"{stem}_temporal_contrast_curves.csv", temporal_rows)
        _write_csv(output / f"{stem}_layer_pairwise_separability.csv", pairwise_rows)
        _write_csv(output / f"{stem}_layer_knn_purity.csv", knn_rows)
        _write_csv(output / f"{stem}_layer_linear_probe.csv", probe_rows)
        _write_csv(output / f"{stem}_nearest_neighbor_cases.csv", nn_rows)
        metrics = daeac_metrics(y, result["y_pred"], class_names)
        write_json(metrics, output / f"{stem}_metrics.json")
        plot_temporal_contrast(temporal_rows, figures / stem)
        plot_feature_effect_heatmap(effect_rows, figures / stem / "clinical_proxy_effect_size_heatmap.png")
        plot_layer_collapse(pairwise_rows, figures / stem / "layer_collapse_curve.png")
        plot_pca_embeddings(result["features_by_layer"], y, class_names, figures / stem / "embeddings", int(config["visualization"]["max_points"]))

        combined_pairwise.extend(_tag_rows(pairwise_rows, args.method_name, dataset_key))
        combined_knn.extend(_tag_rows(knn_rows, args.method_name, dataset_key))
        combined_probe.extend(_tag_rows(probe_rows, args.method_name, dataset_key))
        combined_effect.extend(_tag_rows(effect_rows, args.method_name, dataset_key))
        summary["datasets"][dataset_key] = {
            "samples": int(len(indices)),
            "sampling": "stratified",
            "class_counts_sampled": _class_counts(y, class_names),
            "metrics": metrics,
            "raw_rows_joined": int(len(raw_rows)),
        }
        ds.close()

    _write_csv(output / "layer_pairwise_separability.csv", combined_pairwise)
    _write_csv(output / "layer_knn_purity.csv", combined_knn)
    _write_csv(output / "layer_linear_probe.csv", combined_probe)
    _write_csv(output / "clinical_proxy_effect_size.csv", combined_effect)
    summary["checkpoint_epoch"] = checkpoint_info.get("epoch")
    summary["checkpoint_method"] = checkpoint_info.get("method", checkpoint_info.get("model_name"))
    write_json(summary, output / "debug_summary.json")
    print(f"debug outputs written to {output}")


def _load_model(args, model_config: dict[str, Any], device: torch.device):
    ckpt_type = args.checkpoint_type
    if ckpt_type == "auto":
        try:
            model, checkpoint = load_daeac_adversarial_checkpoint(args.checkpoint, device)
            return model, checkpoint
        except Exception:
            model = load_daeac_checkpoint(args.checkpoint, model_config, device)
            checkpoint = torch.load(args.checkpoint, map_location=device)
            return model, checkpoint
    if ckpt_type == "adversarial":
        return load_daeac_adversarial_checkpoint(args.checkpoint, device)
    model = load_daeac_checkpoint(args.checkpoint, model_config, device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    return model, checkpoint


def _collect_model_outputs(model, loader: DataLoader, device: torch.device, adda_encoder: str) -> dict[str, Any]:
    model.eval()
    y_pred = []
    probs_all = []
    layers: dict[str, list[np.ndarray]] = {}
    hooks = _register_gap_hooks(_feature_extractor(model, adda_encoder), layers)
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            layers.setdefault("input", []).append(x.detach().cpu().numpy().reshape((x.shape[0], -1)))
            extractor = _feature_extractor(model, adda_encoder)
            native_layers = extractor.forward_layers(x)
            for name, value in native_layers.items():
                layers.setdefault(name, []).append(value.detach().cpu().numpy())
            logits = _class_logits(model, native_layers["gap_embed"])
            probs = torch.softmax(logits, dim=1)
            probs_np = probs.detach().cpu().numpy()
            probs_all.append(probs_np)
            y_pred.append(probs_np.argmax(axis=1))
    for hook in hooks:
        hook.remove()
    return {
        "y_pred": np.concatenate(y_pred) if y_pred else np.zeros(0, dtype=np.int64),
        "probabilities": np.concatenate(probs_all) if probs_all else np.zeros((0, 4), dtype=np.float32),
        "features_by_layer": {name: np.concatenate(values, axis=0) for name, values in layers.items()},
    }


def _feature_extractor(model, adda_encoder: str):
    if isinstance(model, DAEACADDAModel):
        return model.source_encoder if adda_encoder == "source" else model.target_encoder
    return model.feature_extractor


def _class_logits(model, features: torch.Tensor) -> torch.Tensor:
    logits, _ = model.classifier(features, return_logits=True)
    return logits


def _register_gap_hooks(extractor, layers: dict[str, list[np.ndarray]]):
    hook_names = ["input_conv", "aspp_se_1", "residual_1", "aspp_se_2", "residual_2", "transition", "final_aspp_se"]
    hooks = []
    for name in hook_names:
        module = getattr(extractor, name, None)
        if module is None:
            continue

        def hook(_module, _inputs, output, layer_name=name):
            if isinstance(output, torch.Tensor):
                pooled = torch.nn.functional.adaptive_avg_pool2d(output, 1).flatten(1)
                layers.setdefault(layer_name, []).append(pooled.detach().cpu().numpy())

        hooks.append(module.register_forward_hook(hook))
    return hooks


def _join_raw_cache(raw_cache_dir: str | None, sample_id_values: list[str], y: np.ndarray, y_pred: np.ndarray, class_names: list[str], allow_missing: bool) -> list[dict[str, Any]]:
    if raw_cache_dir in (None, "", "none", "None"):
        if allow_missing:
            return []
        raise FileNotFoundError("Missing --raw-cache-dir. Pass --allow-missing-raw for representation-only debug.")
    raw_path = Path(raw_cache_dir) / "raw_clinical_features.csv"
    if not raw_path.exists():
        if allow_missing:
            return []
        raise FileNotFoundError(f"Raw cache CSV not found: {raw_path}")
    cache = {}
    with raw_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            cache[row["sample_id"]] = row
    rows = []
    for idx, sample_id in enumerate(sample_id_values):
        row = dict(cache.get(sample_id, {"sample_id": sample_id}))
        row.update({"true_class": class_names[int(y[idx])], "pred_class": class_names[int(y_pred[idx])], "matched_raw_cache": sample_id in cache})
        rows.append(row)
    return rows


def _write_error_groups(path: Path, ids: list[str], metadata: list[dict[str, Any]], y: np.ndarray, y_pred: np.ndarray, probabilities: np.ndarray, class_names: list[str]) -> None:
    rows = []
    for idx, sample_id in enumerate(ids):
        true_id = int(y[idx])
        pred_id = int(y_pred[idx])
        group = "correct_" + class_names[true_id] if true_id == pred_id else f"{class_names[true_id]}_to_{class_names[pred_id]}"
        rows.append(
            {
                "sample_id": sample_id,
                "sample_index": idx,
                "group": group,
                "true_class": class_names[true_id],
                "pred_class": class_names[pred_id],
                "confidence": float(probabilities[idx, pred_id]),
                "entropy": float(-(probabilities[idx] * np.log(probabilities[idx] + 1.0e-12)).sum()),
                "record": metadata[idx].get("record", ""),
                "symbol": metadata[idx].get("symbol", ""),
                "r_peak_sample": metadata[idx].get("r_peak_sample", metadata[idx].get("sample", "")),
            }
        )
    _write_csv(path, rows)


def _selected_datasets(dataset: str) -> list[str]:
    if dataset == "all":
        return ["target", "incart", "svdb"]
    if dataset in {"target", "incart", "svdb"}:
        return [dataset]
    raise ValueError(f"Unknown dataset {dataset!r}; expected target, incart, svdb, or all")


def _dataset_max_samples(dataset_key: str, args, analysis_cfg: dict) -> int | None:
    explicit = {
        "target": args.target_max_samples,
        "incart": args.incart_max_samples,
        "svdb": args.svdb_max_samples,
    }.get(dataset_key)
    if explicit is not None:
        return int(explicit)
    if args.max_samples is not None:
        return int(args.max_samples)
    defaults = analysis_cfg.get("max_samples_by_dataset", {})
    value = defaults.get(dataset_key, analysis_cfg.get("max_samples"))
    return None if value in (None, "", "null", "None") else int(value)


def _class_counts(y: np.ndarray, class_names: list[str]) -> dict[str, int]:
    counts = np.bincount(y.astype(np.int64), minlength=len(class_names))
    return {name: int(counts[idx]) for idx, name in enumerate(class_names)}


def _tag_rows(rows: list[dict[str, Any]], method_name: str, dataset: str) -> list[dict[str, Any]]:
    return [{"method_name": method_name, "dataset": dataset, **row} for row in rows]


def _write_csv(path: Path, rows: list[dict]) -> None:
    ensure_dir(path.parent)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        if not fieldnames:
            f.write("")
            return
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

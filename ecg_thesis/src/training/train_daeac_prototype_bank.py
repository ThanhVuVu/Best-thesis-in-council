from __future__ import annotations

import copy
import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.models.daeac_paper import ClassifierH
from src.training.daeac_losses import (
    compacting_loss,
    distance_from_name,
    separating_loss,
    weighted_cross_entropy_from_logits,
)
from src.training.daeac_prototype_bank import (
    PrototypeCandidates,
    ReliabilityWeightedPrototypeBank,
    candidate_lists,
    dense_batch_prototypes,
)
from src.training.daeac_pseudo_filter import (
    class_threshold_tensor,
    filter_target_pseudolabels,
    pseudo_safety_reason,
    update_pseudo_safety_state,
    validate_pseudo_filter_config,
)
from src.training.train_daeac_paper import (
    CenterMemory,
    _class_weights,
    _cluster_align_loss,
    _threshold_tensor,
    build_daeac_model,
    compute_global_source_centers,
    compute_global_target_centers,
    evaluate_daeac_model,
    load_daeac_checkpoint,
)
from src.utils.io import ensure_dir, write_json
from src.utils.wandb_logging import init_wandb


VALID_USAGES = {"logging_only", "weighted_global"}


def validate_prototype_bank_config(config: dict[str, Any]) -> str:
    bank_cfg = dict(config.get("prototype_bank", {}))
    if not bool(bank_cfg.get("enabled", False)):
        raise ValueError("prototype_bank.enabled must be true for this workflow.")
    usage = str(bank_cfg.get("usage", "")).lower()
    if usage not in VALID_USAGES:
        raise ValueError(f"prototype_bank.usage must be one of {sorted(VALID_USAGES)}, got '{usage}'.")
    if str(bank_cfg.get("reliability_rule", "")) != "coverage_x_confidence":
        raise ValueError("PLAN 1 supports only reliability_rule=coverage_x_confidence.")
    validate_pseudo_filter_config(config, list(config.get("data", {}).get("class_names", [])))
    return usage


def build_prototype_bank(config: dict[str, Any], device: torch.device) -> ReliabilityWeightedPrototypeBank:
    cfg = config["prototype_bank"]
    return ReliabilityWeightedPrototypeBank(
        num_classes=int(config["data"]["num_classes"]),
        feature_dim=int(config["model"]["feature_dim"]),
        source_momentum=float(cfg["source_momentum"]),
        target_momentum=float(cfg["target_momentum"]),
        reliability_momentum=float(cfg["reliability_momentum"]),
        min_target_count=int(cfg["min_target_count"]),
        beta_max=float(cfg["beta_max"]),
        rampup_epochs=int(cfg["rampup_epochs"]),
    ).to(device)


def train_daeac_prototype_bank(
    source_dataset: Dataset,
    val_dataset: Dataset,
    target_dataset: Dataset,
    config: dict[str, Any],
    output_dir: str | Path,
    device: torch.device,
    resume_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    usage = validate_prototype_bank_config(config)
    cfg = config["adaptation"]
    bank_cfg = config["prototype_bank"]
    class_names = list(config["data"]["class_names"])
    num_classes = int(config["data"]["num_classes"])
    output_dir = Path(output_dir)
    ckpt_dir = ensure_dir(output_dir / "checkpoints")
    metrics_dir = ensure_dir(output_dir / "metrics")
    prefix = str(cfg["checkpoint_prefix"])

    model = build_daeac_model(config, device)
    init_checkpoint = cfg.get("init_checkpoint")
    if not init_checkpoint:
        raise ValueError("adaptation.init_checkpoint is required; this workflow never trains a base model silently.")
    load_daeac_checkpoint(init_checkpoint, config, device, model=model)
    source_loader = DataLoader(source_dataset, batch_size=int(cfg["source_batch_size"]), shuffle=True, num_workers=0)
    source_init_loader = DataLoader(source_dataset, batch_size=int(cfg["source_batch_size"]), shuffle=False, num_workers=0)
    target_loader = DataLoader(target_dataset, batch_size=int(cfg["target_batch_size"]), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=int(cfg["source_batch_size"]), shuffle=False, num_workers=0)
    class_weights = _class_weights(source_dataset, config, cfg, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(cfg["lr_decay_every_steps"]),
        gamma=float(cfg["lr_decay_gamma"]),
    )
    distance_fn = distance_from_name(str(cfg.get("distance", "l2")))
    thresholds = _threshold_tensor(config, cfg, device)
    filter_cfg = validate_pseudo_filter_config(config, class_names)
    filter_enabled = bool(filter_cfg.get("enabled", False))
    filter_thresholds = class_threshold_tensor(filter_cfg, class_names, device) if filter_enabled else thresholds
    bank = build_prototype_bank(config, device)
    cpu_rng_state = torch.random.get_rng_state() if usage == "logging_only" else None
    cuda_rng_state = torch.cuda.get_rng_state_all() if usage == "logging_only" and torch.cuda.is_available() else None
    source_initial, source_initial_counts = _global_source_prototypes(model, source_init_loader, device, num_classes)
    bank.initialize_source(source_initial, source_initial_counts)
    if cpu_rng_state is not None:
        torch.random.set_rng_state(cpu_rng_state)
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state_all(cuda_rng_state)

    legacy_memory = None
    if usage == "logging_only":
        legacy_memory = CenterMemory(num_classes, int(config["model"]["feature_dim"]), device)
        # Preserve the accepted baseline's shuffled initialization passes so
        # the control consumes RNG in exactly the same order before epoch 0.
        legacy_memory.source = compute_global_source_centers(model, source_loader, device, num_classes)
        legacy_memory.target = compute_global_target_centers(model, target_loader, device, num_classes, thresholds)
        legacy_memory.refresh_mixed()

    aux_classifier = ClassifierH(
        feature_dim=int(config["model"]["feature_dim"]),
        num_classes=num_classes,
        dropout=0.0,
    ).to(device)
    start_epoch = 0
    best_macro_f1 = -1.0
    best_epoch = -1
    all_n_streak = 0
    empty_acceptance_streak = 0
    history: list[dict[str, Any]] = []
    if resume_checkpoint is not None:
        resume = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(resume["model_state_dict"])
        bank.load_state_dict(resume["prototype_bank_state_dict"])
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        scheduler.load_state_dict(resume["scheduler_state_dict"])
        start_epoch = int(resume["epoch"]) + 1
        best_macro_f1 = float(resume.get("best_macro_f1", -1.0))
        best_epoch = int(resume.get("best_epoch", -1))
        all_n_streak = int(resume.get("all_n_streak", 0))
        empty_acceptance_streak = int(resume.get("empty_acceptance_streak", 0))
        history = list(resume.get("history", []))
        if legacy_memory is not None:
            legacy_state = resume.get("legacy_center_state")
            if legacy_state is None:
                raise KeyError("logging_only resume checkpoint is missing legacy_center_state.")
            _restore_legacy_memory(legacy_memory, legacy_state, device)

    latest_path = ckpt_dir / f"{prefix}_latest.pt"
    best_path = ckpt_dir / f"{prefix}_best.pt"
    history_path = metrics_dir / f"{prefix}_train_log.csv"
    run = init_wandb(config, job_type="train_daeac_prototype_bank", default_name=prefix)
    for epoch in range(start_epoch, int(cfg["epochs"])):
        model.train()
        aux_classifier.load_state_dict(copy.deepcopy(model.classifier.state_dict()))
        aux_classifier.eval()
        target_iter = _cycle(target_loader)
        batch_rows: list[dict[str, float]] = []
        predicted_counts = torch.zeros(num_classes, dtype=torch.long, device=device)
        accepted_counts = torch.zeros(num_classes, dtype=torch.long, device=device)
        accepted_confidence_sums = torch.zeros(num_classes, device=device)
        predicted_confidence_sums = torch.zeros(num_classes, device=device)
        predicted_entropy_sums = torch.zeros(num_classes, device=device)
        accepted_entropy_sums = torch.zeros(num_classes, device=device)
        rejected_confidence_counts = torch.zeros(num_classes, dtype=torch.long, device=device)
        rejected_entropy_counts = torch.zeros(num_classes, dtype=torch.long, device=device)
        rejected_both_counts = torch.zeros(num_classes, dtype=torch.long, device=device)
        effective_counts = torch.zeros(num_classes, dtype=torch.long, device=device)
        source_skips = torch.zeros(num_classes, dtype=torch.long, device=device)
        target_skips = torch.zeros(num_classes, dtype=torch.long, device=device)

        for x_s, y_s in source_loader:
            target_batch = next(target_iter)
            x_t = target_batch[0] if isinstance(target_batch, (tuple, list)) else target_batch
            x_s, y_s, x_t = x_s.to(device), y_s.to(device), x_t.to(device)
            z_s, logits_s, _ = model(x_s, return_logits=True)
            loss_cls = weighted_cross_entropy_from_logits(logits_s, y_s, class_weights)
            with torch.no_grad():
                z_t_all = model.extract_features(x_t)
                _, probabilities_t = aux_classifier(z_t_all, return_logits=True)
                if filter_enabled:
                    filtered = filter_target_pseudolabels(
                        probabilities_t,
                        mode=str(filter_cfg["mode"]),
                        global_confidence_threshold=float(filter_cfg["global_confidence_threshold"]),
                        class_confidence_thresholds=filter_thresholds,
                        max_normalized_entropy=float(filter_cfg["max_normalized_entropy"]),
                    )
                    confidence_t = filtered.confidence
                    pseudo_t = filtered.pseudo_labels
                    entropy_t = filtered.normalized_entropy
                    confident = filtered.accepted_mask
                    reject_confidence = filtered.rejected_confidence_mask
                    reject_entropy = filtered.rejected_entropy_mask
                    reject_both = filtered.rejected_both_mask
                else:
                    confidence_t, pseudo_t = probabilities_t.max(dim=1)
                    entropy_t = -(probabilities_t * probabilities_t.clamp_min(torch.finfo(probabilities_t.dtype).tiny).log()).sum(dim=1)
                    entropy_t = entropy_t / np.log(num_classes)
                    confident = confidence_t >= thresholds[pseudo_t]
                    reject_confidence = ~confident
                    reject_entropy = torch.zeros_like(confident)
                    reject_both = torch.zeros_like(confident)
            if bool(confident.any()):
                selected_x_t = x_t[confident]
                selected_pseudo_t = pseudo_t[confident]
                z_t = model.extract_features(selected_x_t)
            else:
                selected_pseudo_t = torch.empty(0, dtype=torch.long, device=device)
                z_t = torch.empty(0, bank.feature_dim, device=device)

            source_local, source_counts = dense_batch_prototypes(z_s, y_s, num_classes)
            target_local, target_counts = dense_batch_prototypes(z_t, selected_pseudo_t, num_classes)
            candidates = bank.candidates(source_local, source_counts, target_local, target_counts)
            effective_counts += target_counts * candidates.target_update_mask.to(torch.long)
            source_skips += (~candidates.source_update_mask).to(torch.long)
            target_skips += (~candidates.target_update_mask).to(torch.long)
            source_for_loss, target_for_loss, global_for_loss = _centers_for_usage(
                usage,
                candidates,
                legacy_memory,
                z_s,
                y_s,
                z_t,
                selected_pseudo_t,
                float(cfg["center_ema_gamma"]),
                num_classes,
            )
            loss_align = _cluster_align_loss(source_for_loss, target_for_loss, cfg, distance_fn, device)
            if z_t.numel() > 0:
                z_mix = torch.cat([z_s, z_t], dim=0)
                y_mix = torch.cat([y_s, selected_pseudo_t], dim=0)
            else:
                z_mix, y_mix = z_s, y_s
            loss_sep = separating_loss(global_for_loss, float(cfg["margin"]), distance_fn, device)
            loss_comp = compacting_loss(z_mix, y_mix, global_for_loss, distance_fn, device)
            loss_total = loss_cls + float(cfg["beta1"]) * loss_align + float(cfg["beta2"]) * (loss_sep + loss_comp)
            optimizer.zero_grad(set_to_none=True)
            loss_total.backward()
            optimizer.step()
            scheduler.step()
            bank.commit(candidates)
            if legacy_memory is not None:
                legacy_memory.commit(source_for_loss, target_for_loss, global_for_loss)

            predicted_counts += torch.bincount(pseudo_t, minlength=num_classes)
            accepted_counts += torch.bincount(selected_pseudo_t, minlength=num_classes)
            predicted_confidence_sums.scatter_add_(0, pseudo_t, confidence_t)
            predicted_entropy_sums.scatter_add_(0, pseudo_t, entropy_t)
            if bool(confident.any()):
                accepted_confidence_sums.scatter_add_(0, selected_pseudo_t, confidence_t[confident])
                accepted_entropy_sums.scatter_add_(0, selected_pseudo_t, entropy_t[confident])
            _scatter_mask_counts(rejected_confidence_counts, pseudo_t, reject_confidence)
            _scatter_mask_counts(rejected_entropy_counts, pseudo_t, reject_entropy)
            _scatter_mask_counts(rejected_both_counts, pseudo_t, reject_both)
            batch_rows.append(
                {
                    "loss": float(loss_total.detach().cpu()),
                    "loss_cls": float(loss_cls.detach().cpu()),
                    "loss_align": float(loss_align.detach().cpu()),
                    "loss_sep": float(loss_sep.detach().cpu()),
                    "loss_comp": float(loss_comp.detach().cpu()),
                    "pseudo_selected": float(confident.sum().detach().cpu()),
                }
            )

        reliability = bank.update_reliability(
            predicted_counts,
            accepted_counts,
            accepted_confidence_sums,
            epoch,
        )
        distribution = update_pseudo_safety_state(
            accepted_counts,
            previous_empty_streak=empty_acceptance_streak,
            previous_all_n_streak=all_n_streak,
            near_all_n_ratio=float(filter_cfg.get("near_all_n_ratio", bank_cfg["near_all_n_ratio"])),
        )
        all_n_streak = int(distribution["all_n_streak"])
        empty_acceptance = bool(distribution["empty_acceptance"])
        empty_acceptance_streak = int(distribution["empty_acceptance_streak"])
        val_result = evaluate_daeac_model(model, val_loader, device, class_names)
        row: dict[str, Any] = _mean_rows(batch_rows)
        row.update(
            {
                "epoch": epoch,
                "usage": usage,
                "val_accuracy": float(val_result["metrics"]["accuracy"]),
                "val_macro_f1": float(val_result["metrics"]["macro_f1"]),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "pseudo/accepted_ratio": float(accepted_counts.sum() / predicted_counts.sum().clamp_min(1)),
                "pseudo/rejected_ratio": float((predicted_counts.sum() - accepted_counts.sum()) / predicted_counts.sum().clamp_min(1)),
                "pseudo/empty_acceptance": float(empty_acceptance),
                "pseudo/all_n": float(bool(distribution["all_n"])),
                "pseudo/near_all_n": float(bool(distribution["near_all_n"])),
                "pseudo/all_n_streak": float(all_n_streak),
                "pseudo/empty_acceptance_streak": float(empty_acceptance_streak),
                "prototype/ramp": float(reliability["ramp"]),
                "prototype/valid_source": float(bank.source_valid.sum()),
                "prototype/valid_target": float(bank.target_valid.sum()),
            }
        )
        _add_class_diagnostics(
            row,
            bank,
            class_names,
            predicted_counts,
            accepted_counts,
            accepted_confidence_sums,
            predicted_confidence_sums,
            predicted_entropy_sums,
            accepted_entropy_sums,
            rejected_confidence_counts,
            rejected_entropy_counts,
            rejected_both_counts,
            effective_counts,
            source_skips,
            target_skips,
            reliability,
        )
        history.append(row)
        _write_history_csv(history, history_path)
        run.log({f"adapt/{key}": value for key, value in row.items() if key not in {"epoch", "usage"}}, step=epoch)
        if row["val_macro_f1"] >= best_macro_f1:
            best_macro_f1 = float(row["val_macro_f1"])
            best_epoch = epoch
            _save_checkpoint(
                best_path,
                model,
                bank,
                optimizer,
                scheduler,
                config,
                epoch,
                row,
                best_macro_f1,
                best_epoch,
                all_n_streak,
                empty_acceptance_streak,
                legacy_memory,
                history,
            )
        _save_checkpoint(
            latest_path,
            model,
            bank,
            optimizer,
            scheduler,
            config,
            epoch,
            row,
            best_macro_f1,
            best_epoch,
            all_n_streak,
            empty_acceptance_streak,
            legacy_memory,
            history,
        )
        print(
            f"[{prefix} {epoch + 1}/{cfg['epochs']}] loss={row['loss']:.4f} "
            f"val_macro_f1={row['val_macro_f1']:.4f} accepted={int(accepted_counts.sum())} "
            f"beta={bank.beta.detach().cpu().tolist()}"
        )
        fail_all_n = bool(filter_cfg.get("fail_on_all_n", bank_cfg.get("fail_on_all_n", True)))
        fail_empty = bool(filter_cfg.get("fail_on_empty", False))
        patience = int(filter_cfg.get("safety_patience_epochs", bank_cfg.get("all_n_patience_epochs", 2)))
        safety_reason = pseudo_safety_reason(
            distribution,
            fail_on_empty=fail_empty,
            fail_on_all_n=fail_all_n,
            patience=patience,
        )
        if safety_reason is not None:
            write_json(
                {
                    "reason": safety_reason,
                    "epoch": epoch,
                    "all_n_streak": all_n_streak,
                    "empty_acceptance_streak": empty_acceptance_streak,
                    "latest_checkpoint": str(latest_path),
                    "epoch_metrics": row,
                },
                metrics_dir / f"{prefix}_safety_stop.json",
            )
            run.finish()
            raise RuntimeError(f"Pseudo-label safety stop: {safety_reason} for {patience} consecutive epochs.")

    summary = {
        "usage": usage,
        "init_checkpoint": str(init_checkpoint),
        "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint is not None else None,
        "latest_checkpoint": str(latest_path),
        "best_checkpoint": str(best_path),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_macro_f1,
        "checkpoint_selection": "source_validation_macro_f1",
        "target_test_used_during_training": False,
        "pseudo_filter": filter_cfg,
        "final_prototype_diagnostics": _json_bank_diagnostics(bank, class_names),
        "history": history,
    }
    write_json(summary, metrics_dir / f"{prefix}_train_summary.json")
    run.summary_update({key: value for key, value in summary.items() if key != "history"})
    run.finish()
    return summary


def _centers_for_usage(
    usage: str,
    candidates: PrototypeCandidates,
    legacy_memory: CenterMemory | None,
    z_s: torch.Tensor,
    y_s: torch.Tensor,
    z_t: torch.Tensor,
    pseudo_t: torch.Tensor,
    gamma: float,
    num_classes: int,
) -> tuple[list[torch.Tensor | None], list[torch.Tensor | None], list[torch.Tensor | None]]:
    if usage == "weighted_global":
        return (
            candidate_lists(candidates.source, candidates.source_valid),
            candidate_lists(candidates.target, candidates.target_valid),
            candidate_lists(candidates.global_, candidates.global_valid),
        )
    if legacy_memory is None:
        raise ValueError("logging_only usage requires legacy CenterMemory.")
    local_source = _list_batch_centers(z_s, y_s, num_classes)
    local_target = _list_batch_centers(z_t, pseudo_t, num_classes)
    return legacy_memory.centers_for_loss(local_source, local_target, gamma)


@torch.no_grad()
def _global_source_prototypes(
    model,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    sums = torch.zeros(num_classes, model.feature_dim, device=device)
    counts = torch.zeros(num_classes, dtype=torch.long, device=device)
    model.eval()
    for x, labels in loader:
        features = model.extract_features(x.to(device))
        labels = labels.to(device)
        sums.index_add_(0, labels, features)
        counts += torch.bincount(labels, minlength=num_classes)
    prototypes = sums / counts.clamp_min(1)[:, None]
    return prototypes, counts


def _list_batch_centers(features, labels, num_classes):
    centers = []
    for class_id in range(num_classes):
        mask = labels == class_id
        centers.append(features[mask].mean(dim=0) if bool(mask.any()) else None)
    return centers


def _add_class_diagnostics(
    row,
    bank,
    class_names,
    predicted,
    accepted,
    accepted_confidence_sums,
    predicted_confidence_sums,
    predicted_entropy_sums,
    accepted_entropy_sums,
    rejected_confidence,
    rejected_entropy,
    rejected_both,
    effective_counts,
    source_skips,
    target_skips,
    reliability,
):
    diagnostics = bank.diagnostics()
    for index, name in enumerate(class_names):
        accepted_count = int(accepted[index])
        row[f"pseudo/predicted_{name}"] = float(predicted[index])
        row[f"pseudo/accepted_{name}"] = float(accepted[index])
        row[f"pseudo/rejected_{name}"] = float(predicted[index] - accepted[index])
        row[f"pseudo/acceptance_ratio_{name}"] = (
            float(accepted[index] / predicted[index]) if int(predicted[index]) else 0.0
        )
        row[f"pseudo/mean_predicted_confidence_{name}"] = (
            float(predicted_confidence_sums[index] / predicted[index]) if int(predicted[index]) else 0.0
        )
        row[f"pseudo/mean_confidence_{name}"] = (
            float(accepted_confidence_sums[index] / accepted[index]) if accepted_count else 0.0
        )
        row[f"pseudo/mean_predicted_entropy_{name}"] = (
            float(predicted_entropy_sums[index] / predicted[index]) if int(predicted[index]) else 0.0
        )
        row[f"pseudo/mean_accepted_entropy_{name}"] = (
            float(accepted_entropy_sums[index] / accepted[index]) if accepted_count else 0.0
        )
        row[f"pseudo/rejected_confidence_{name}"] = float(rejected_confidence[index])
        row[f"pseudo/rejected_entropy_{name}"] = float(rejected_entropy[index])
        row[f"pseudo/rejected_both_{name}"] = float(rejected_both[index])
        row[f"prototype/R_t_{name}"] = float(bank.target_reliability[index])
        row[f"prototype/beta_{name}"] = float(bank.beta[index])
        row[f"prototype/source_count_{name}"] = float(bank.source_counts[index])
        row[f"prototype/target_effective_count_{name}"] = float(bank.target_counts[index])
        row[f"prototype/epoch_effective_count_{name}"] = float(effective_counts[index])
        row[f"prototype/source_update_skipped_{name}"] = float(source_skips[index])
        row[f"prototype/target_update_skipped_{name}"] = float(target_skips[index])
        row[f"prototype/coverage_{name}"] = float(reliability["coverage"][index])
        row[f"prototype/observed_R_t_{name}"] = float(reliability["observed_reliability"][index])
        row[f"prototype/ps_pt_l2_{name}"] = float(diagnostics["source_target_l2"][index])
        row[f"prototype/pg_ps_l2_{name}"] = float(diagnostics["global_source_l2"][index])
        row[f"prototype/ps_norm_{name}"] = float(diagnostics["source_norm"][index])
        row[f"prototype/pt_norm_{name}"] = float(diagnostics["target_norm"][index])
        row[f"prototype/pg_norm_{name}"] = float(diagnostics["global_norm"][index])


def _json_bank_diagnostics(bank, class_names):
    diagnostics = bank.diagnostics()
    return {
        name: {
            "R_t": float(bank.target_reliability[index]),
            "beta": float(bank.beta[index]),
            "source_count": int(bank.source_counts[index]),
            "target_count": int(bank.target_counts[index]),
            "source_valid": bool(bank.source_valid[index]),
            "target_valid": bool(bank.target_valid[index]),
            "ps_pt_l2": float(diagnostics["source_target_l2"][index]),
            "pg_ps_l2": float(diagnostics["global_source_l2"][index]),
        }
        for index, name in enumerate(class_names)
    }


def _save_checkpoint(
    path,
    model,
    bank,
    optimizer,
    scheduler,
    config,
    epoch,
    metrics,
    best_macro_f1,
    best_epoch,
    all_n_streak,
    empty_acceptance_streak,
    legacy_memory,
    history,
):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "prototype_bank_state_dict": bank.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": config,
            "epoch": int(epoch),
            "metrics": metrics,
            "best_macro_f1": float(best_macro_f1),
            "best_epoch": int(best_epoch),
            "all_n_streak": int(all_n_streak),
            "empty_acceptance_streak": int(empty_acceptance_streak),
            "legacy_center_state": _legacy_memory_state(legacy_memory),
            "history": history,
        },
        path,
    )


def _legacy_memory_state(memory):
    if memory is None:
        return None
    return {
        "source": [value.detach().cpu() if value is not None else None for value in memory.source],
        "target": [value.detach().cpu() if value is not None else None for value in memory.target],
        "mixed": [value.detach().cpu() if value is not None else None for value in memory.mixed],
    }


def _restore_legacy_memory(memory, state, device):
    memory.source = [value.to(device) if value is not None else None for value in state["source"]]
    memory.target = [value.to(device) if value is not None else None for value in state["target"]]
    memory.mixed = [value.to(device) if value is not None else None for value in state["mixed"]]


def _mean_rows(rows):
    if not rows:
        return {key: 0.0 for key in ("loss", "loss_cls", "loss_align", "loss_sep", "loss_comp", "pseudo_selected")}
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def _write_history_csv(rows, path):
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _cycle(loader):
    while True:
        yield from loader


def _scatter_mask_counts(destination, labels, mask):
    if bool(mask.any()):
        destination.add_(torch.bincount(labels[mask], minlength=len(destination)))

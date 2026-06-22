from __future__ import annotations

import copy
import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.daeac_dataset import DAEACPseudoLabeledDataset
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
from src.training.daeac_prototype_losses import (
    build_margin_matrix,
    directed_target_alignment_loss,
    linear_ramp,
    sample_prototype_margin_loss,
    source_compactness_loss,
    target_compactness_loss,
    target_reliability_weights,
    validate_prototype_loss_config,
)
from src.training.train_daeac_paper import (
    CenterMemory,
    _class_weights,
    _cluster_align_loss,
    _threshold_tensor,
    build_daeac_model,
    compute_global_source_centers,
    compute_global_pseudo_target_centers,
    evaluate_daeac_model,
    load_daeac_checkpoint,
)
from src.utils.io import ensure_dir, write_json
from src.utils.wandb_logging import init_wandb


VALID_USAGES = {"logging_only", "weighted_global"}
VALID_BATCHNORM_MODES = {"train", "freeze_stats", "freeze_all"}
VALID_TARGET_FORWARD_MODES = {"double", "single"}
VALID_EPOCH_DRIVERS = {"source", "target_once"}


def validate_prototype_bank_config(config: dict[str, Any]) -> str:
    bank_cfg = dict(config.get("prototype_bank", {}))
    if not bool(bank_cfg.get("enabled", False)):
        raise ValueError("prototype_bank.enabled must be true for this workflow.")
    usage = str(bank_cfg.get("usage", "")).lower()
    if usage not in VALID_USAGES:
        raise ValueError(f"prototype_bank.usage must be one of {sorted(VALID_USAGES)}, got '{usage}'.")
    if str(bank_cfg.get("reliability_rule", "")) != "coverage_x_confidence":
        raise ValueError("PLAN 1 supports only reliability_rule=coverage_x_confidence.")
    class_names = list(config.get("data", {}).get("class_names", []))
    validate_pseudo_filter_config(config, class_names)
    validate_prototype_loss_config(config, class_names)
    _validate_adaptation_execution_config(config)
    return usage


def _validate_adaptation_execution_config(config: dict[str, Any]) -> None:
    cfg = dict(config.get("adaptation", {}))
    batchnorm_mode = str(cfg.get("batchnorm_mode", "train")).lower()
    target_forward_mode = str(cfg.get("target_forward_mode", "single")).lower()
    epoch_driver = str(cfg.get("epoch_driver", "target_once")).lower()
    if batchnorm_mode not in VALID_BATCHNORM_MODES:
        raise ValueError(f"adaptation.batchnorm_mode must be one of {sorted(VALID_BATCHNORM_MODES)}.")
    if target_forward_mode not in VALID_TARGET_FORWARD_MODES:
        raise ValueError(
            f"adaptation.target_forward_mode must be one of {sorted(VALID_TARGET_FORWARD_MODES)}."
        )
    if epoch_driver not in VALID_EPOCH_DRIVERS:
        raise ValueError(f"adaptation.epoch_driver must be one of {sorted(VALID_EPOCH_DRIVERS)}.")


def _validate_resume_execution_compatibility(checkpoint: dict[str, Any], current_cfg: dict[str, Any]) -> None:
    saved_cfg = dict(checkpoint.get("config", {}).get("adaptation", {}))
    expected_version = int(current_cfg.get("training_semantics_version", 3))
    saved_version = int(saved_cfg.get("training_semantics_version", 1))
    if saved_version != expected_version:
        raise ValueError(
            "Resume checkpoint uses incompatible adaptation training semantics "
            f"(checkpoint={saved_version}, current={expected_version}). Start this corrected workflow "
            "from the source-selected init checkpoint instead of resuming the old adaptation run."
        )
    for key in ("batchnorm_mode", "target_forward_mode", "epoch_driver"):
        saved = str(saved_cfg.get(key, ""))
        current = str(current_cfg.get(key, ""))
        if saved != current:
            raise ValueError(f"Resume checkpoint adaptation.{key}={saved!r} does not match current {current!r}.")


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
    cfg.setdefault("training_semantics_version", 3)
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
    batchnorm_mode = str(cfg.get("batchnorm_mode", "train")).lower()
    target_forward_mode = str(cfg.get("target_forward_mode", "single")).lower()
    epoch_driver = str(cfg.get("epoch_driver", "target_once")).lower()
    configure_adaptation_batchnorm(model, batchnorm_mode)
    source_loader = DataLoader(source_dataset, batch_size=int(cfg["source_batch_size"]), shuffle=True, num_workers=0)
    source_init_loader = DataLoader(source_dataset, batch_size=int(cfg["source_batch_size"]), shuffle=False, num_workers=0)
    target_inference_loader = DataLoader(target_dataset, batch_size=int(cfg["target_batch_size"]), shuffle=False, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=int(cfg["source_batch_size"]), shuffle=False, num_workers=0)
    class_weights = _class_weights(source_dataset, config, cfg, device)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable_parameters, lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(cfg["lr_decay_every_steps"]),
        gamma=float(cfg["lr_decay_gamma"]),
    )
    distance_fn = distance_from_name(str(cfg.get("distance", "l2")))
    thresholds = _threshold_tensor(config, cfg, device)
    filter_cfg = validate_pseudo_filter_config(config, class_names)
    prototype_loss_cfg = validate_prototype_loss_config(config, class_names)
    replacement_losses = prototype_loss_cfg.get("mode") == "replacement"
    margin_matrix = None
    if replacement_losses:
        margin_matrix = build_margin_matrix(prototype_loss_cfg, class_names, device, torch.float32)
    filter_enabled = bool(filter_cfg.get("enabled", False))
    filter_thresholds = class_threshold_tensor(filter_cfg, class_names, device) if filter_enabled else thresholds
    aux_classifier = ClassifierH(
        feature_dim=int(config["model"]["feature_dim"]),
        num_classes=num_classes,
        dropout=float(config["model"].get("dropout", 0.0)),
    ).to(device)
    aux_classifier.load_state_dict(copy.deepcopy(model.classifier.state_dict()))
    aux_classifier.eval()
    pseudo_dataset, pseudo_diagnostics = build_filtered_pseudo_snapshot(
        model=model,
        aux_classifier=aux_classifier,
        target_dataset=target_dataset,
        inference_loader=target_inference_loader,
        thresholds=thresholds,
        filter_cfg=filter_cfg,
        filter_thresholds=filter_thresholds,
        device=device,
        num_classes=num_classes,
    )
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
        legacy_memory.target = compute_global_pseudo_target_centers(
            model,
            DataLoader(pseudo_dataset, batch_size=int(cfg["target_batch_size"]), shuffle=False),
            device,
            num_classes,
        )
        legacy_memory.refresh_mixed()

    start_epoch = 0
    best_macro_f1 = -1.0
    best_epoch = -1
    init_val_macro_f1: float | None = None
    best_adapted_macro_f1 = -1.0
    all_n_streak = 0
    empty_acceptance_streak = 0
    history: list[dict[str, Any]] = []
    if resume_checkpoint is not None:
        resume = torch.load(resume_checkpoint, map_location=device)
        _validate_resume_execution_compatibility(resume, cfg)
        model.load_state_dict(resume["model_state_dict"])
        bank.load_state_dict(resume["prototype_bank_state_dict"])
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        scheduler.load_state_dict(resume["scheduler_state_dict"])
        start_epoch = int(resume["epoch"]) + 1
        best_macro_f1 = float(resume.get("best_macro_f1", -1.0))
        best_epoch = int(resume.get("best_epoch", -1))
        init_value = resume.get("init_val_macro_f1")
        init_val_macro_f1 = float(init_value) if init_value is not None else None
        best_adapted_macro_f1 = float(
            resume.get(
                "best_adapted_macro_f1",
                max((float(row["val_macro_f1"]) for row in resume.get("history", [])), default=-1.0),
            )
        )
        all_n_streak = int(resume.get("all_n_streak", 0))
        empty_acceptance_streak = int(resume.get("empty_acceptance_streak", 0))
        history = list(resume.get("history", []))
        if legacy_memory is not None:
            legacy_state = resume.get("legacy_center_state")
            if legacy_state is None:
                raise KeyError("logging_only resume checkpoint is missing legacy_center_state.")
            _restore_legacy_memory(legacy_memory, legacy_state, device)
        aux_classifier.load_state_dict(copy.deepcopy(model.classifier.state_dict()))
        aux_classifier.eval()
        pseudo_dataset, pseudo_diagnostics = build_filtered_pseudo_snapshot(
            model=model,
            aux_classifier=aux_classifier,
            target_dataset=target_dataset,
            inference_loader=target_inference_loader,
            thresholds=thresholds,
            filter_cfg=filter_cfg,
            filter_thresholds=filter_thresholds,
            device=device,
            num_classes=num_classes,
        )

    latest_path = ckpt_dir / f"{prefix}_latest.pt"
    best_path = ckpt_dir / f"{prefix}_best.pt"
    history_path = metrics_dir / f"{prefix}_train_log.csv"
    run = init_wandb(config, job_type="train_daeac_prototype_bank", default_name=prefix)
    if resume_checkpoint is None:
        init_result = evaluate_daeac_model(model, val_loader, device, class_names)
        init_val_macro_f1 = float(init_result["metrics"]["macro_f1"])
        best_macro_f1 = init_val_macro_f1
        best_epoch = -1
        init_row = {
            "epoch": -1,
            "stage": "initialization",
            "optimizer_steps": 0,
            "val_accuracy": float(init_result["metrics"]["accuracy"]),
            "val_macro_f1": init_val_macro_f1,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        write_json(init_row, metrics_dir / f"{prefix}_init_metrics.json")
        _save_checkpoint(
            best_path,
            model,
            bank,
            optimizer,
            scheduler,
            config,
            -1,
            init_row,
            best_macro_f1,
            best_epoch,
            all_n_streak,
            empty_acceptance_streak,
            legacy_memory,
            history,
            init_val_macro_f1=init_val_macro_f1,
            best_adapted_macro_f1=best_adapted_macro_f1,
        )
        run.summary_update({"init_val_macro_f1": init_val_macro_f1, "selected_stage": "initialization"})
        print(f"[{prefix} init] val_macro_f1={init_val_macro_f1:.4f} selected_as_initial_best")
    for epoch in range(start_epoch, int(cfg["epochs"])):
        set_adaptation_train_mode(model, batchnorm_mode)
        target_loader = DataLoader(
            pseudo_dataset,
            batch_size=int(cfg["target_batch_size"]),
            shuffle=True,
            num_workers=0,
        )
        batch_rows: list[dict[str, float]] = []
        predicted_counts = torch.zeros(num_classes, dtype=torch.long, device=device)
        accepted_counts = torch.zeros(num_classes, dtype=torch.long, device=device)
        accepted_confidence_sums = torch.zeros(num_classes, device=device)
        predicted_confidence_sums = torch.zeros(num_classes, device=device)
        predicted_entropy_sums = torch.zeros(num_classes, device=device)
        accepted_entropy_sums = torch.zeros(num_classes, device=device)
        accepted_weight_sums = torch.zeros(num_classes, device=device)
        rejected_confidence_counts = torch.zeros(num_classes, dtype=torch.long, device=device)
        rejected_entropy_counts = torch.zeros(num_classes, dtype=torch.long, device=device)
        rejected_both_counts = torch.zeros(num_classes, dtype=torch.long, device=device)
        effective_counts = torch.zeros(num_classes, dtype=torch.long, device=device)
        source_skips = torch.zeros(num_classes, dtype=torch.long, device=device)
        target_skips = torch.zeros(num_classes, dtype=torch.long, device=device)
        source_pair_counts = torch.zeros((num_classes, num_classes), device=device)
        source_pair_violations = torch.zeros_like(source_pair_counts)
        target_pair_counts = torch.zeros_like(source_pair_counts)
        target_pair_violations = torch.zeros_like(source_pair_counts)
        source_samples_seen = 0
        target_samples_seen = 0

        if replacement_losses:
            ramps = {
                name: linear_ramp(epoch, int(prototype_loss_cfg["rampup_epochs"][name]))
                for name in ("proto_align", "comp_source", "comp_target", "sep_margin")
            }
        else:
            ramps = {name: 0.0 for name in ("proto_align", "comp_source", "comp_target", "sep_margin")}

        for source_batch, target_batch in paired_adaptation_batches(
            source_loader,
            target_loader,
            epoch_driver=epoch_driver,
        ):
            x_s, y_s = source_batch
            x_t, selected_pseudo_t, selected_confidence_t, selected_entropy_t = target_batch
            x_s, y_s, x_t = x_s.to(device), y_s.to(device), x_t.to(device)
            selected_pseudo_t = selected_pseudo_t.to(device)
            selected_confidence_t = selected_confidence_t.to(device)
            selected_entropy_t = selected_entropy_t.to(device)
            source_samples_seen += int(y_s.numel())
            target_samples_seen += int(x_t.shape[0])
            z_s, logits_s, _ = model(x_s, return_logits=True)
            loss_cls = weighted_cross_entropy_from_logits(logits_s, y_s, class_weights)
            z_t = model.extract_features(x_t)
            selected_weights_t = target_reliability_weights(selected_confidence_t, selected_entropy_t)

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
            if replacement_losses:
                replacement = _compute_replacement_losses(
                    z_s=z_s,
                    y_s=y_s,
                    z_t=z_t,
                    pseudo_t=selected_pseudo_t,
                    target_weights=selected_weights_t,
                    candidates=candidates,
                    cfg=prototype_loss_cfg,
                    margin_matrix=margin_matrix,
                    min_target_count=bank.min_target_count,
                )
                loss_align = replacement["loss_proto_align"]
                loss_sep = replacement["loss_sep_margin"]
                loss_comp = replacement["loss_comp_source"] + replacement["loss_comp_target"]
                weighted_align = ramps["proto_align"] * float(prototype_loss_cfg["lambda_proto_align"]) * loss_align
                weighted_comp_source = ramps["comp_source"] * float(prototype_loss_cfg["lambda_comp_source"]) * replacement["loss_comp_source"]
                weighted_comp_target = ramps["comp_target"] * float(prototype_loss_cfg["lambda_comp_target"]) * replacement["loss_comp_target"]
                weighted_sep = ramps["sep_margin"] * float(prototype_loss_cfg["lambda_sep_margin"]) * loss_sep
                loss_total = loss_cls + weighted_align + weighted_comp_source + weighted_comp_target + weighted_sep
                source_pair_counts += replacement["source_pair_counts"]
                source_pair_violations += replacement["source_pair_violations"]
                target_pair_counts += replacement["target_pair_counts"]
                target_pair_violations += replacement["target_pair_violations"]
            else:
                loss_align = _cluster_align_loss(source_for_loss, target_for_loss, cfg, distance_fn, device)
                if z_t.numel() > 0:
                    z_mix = torch.cat([z_s, z_t], dim=0)
                    y_mix = torch.cat([y_s, selected_pseudo_t], dim=0)
                else:
                    z_mix, y_mix = z_s, y_s
                reduction = str(cfg.get("cluster_loss_reduction", "sum"))
                loss_sep = separating_loss(global_for_loss, float(cfg["margin"]), distance_fn, device, reduction=reduction)
                loss_comp = compacting_loss(z_mix, y_mix, global_for_loss, distance_fn, device, reduction=reduction)
                loss_total = loss_cls + float(cfg["beta1"]) * loss_align + float(cfg["beta2"]) * (loss_sep + loss_comp)
                replacement = _empty_replacement_metrics(z_s)
                weighted_align = float(cfg["beta1"]) * loss_align
                weighted_comp_source = z_s.sum() * 0.0
                weighted_comp_target = float(cfg["beta2"]) * loss_comp
                weighted_sep = float(cfg["beta2"]) * loss_sep
            optimizer.zero_grad(set_to_none=True)
            loss_total.backward()
            optimizer.step()
            scheduler.step()
            bank.commit(candidates)
            if legacy_memory is not None:
                legacy_memory.commit(source_for_loss, target_for_loss, global_for_loss)

            pseudo_t = selected_pseudo_t
            confidence_t = selected_confidence_t
            entropy_t = selected_entropy_t
            confident = torch.ones_like(selected_pseudo_t, dtype=torch.bool)
            reject_confidence = torch.zeros_like(confident)
            reject_entropy = torch.zeros_like(confident)
            reject_both = torch.zeros_like(confident)
            predicted_counts += torch.bincount(pseudo_t, minlength=num_classes)
            accepted_counts += torch.bincount(selected_pseudo_t, minlength=num_classes)
            predicted_confidence_sums.scatter_add_(0, pseudo_t, confidence_t)
            predicted_entropy_sums.scatter_add_(0, pseudo_t, entropy_t)
            if bool(confident.any()):
                accepted_confidence_sums.scatter_add_(0, selected_pseudo_t, confidence_t[confident])
                accepted_entropy_sums.scatter_add_(0, selected_pseudo_t, entropy_t[confident])
                accepted_weight_sums.scatter_add_(0, selected_pseudo_t, selected_weights_t)
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
                    "loss_proto_align": float(replacement["loss_proto_align"].detach().cpu()),
                    "loss_comp_source": float(replacement["loss_comp_source"].detach().cpu()),
                    "loss_comp_target": float(replacement["loss_comp_target"].detach().cpu()),
                    "loss_sep_margin": float(replacement["loss_sep_margin"].detach().cpu()),
                    "loss_sep_source": float(replacement["loss_sep_source"].detach().cpu()),
                    "loss_sep_target": float(replacement["loss_sep_target"].detach().cpu()),
                    "weighted_proto_align": float(weighted_align.detach().cpu()),
                    "weighted_comp_source": float(weighted_comp_source.detach().cpu()),
                    "weighted_comp_target": float(weighted_comp_target.detach().cpu()),
                    "weighted_sep_margin": float(weighted_sep.detach().cpu()),
                    "active_source_comp_samples": float(replacement["active_source_comp_samples"].detach().cpu()),
                    "active_target_comp_samples": float(replacement["active_target_comp_samples"].detach().cpu()),
                    "active_align_classes": float(replacement["active_align_classes"].detach().cpu()),
                    "active_source_sep_samples": float(replacement["active_source_sep_samples"].detach().cpu()),
                    "active_target_sep_samples": float(replacement["active_target_sep_samples"].detach().cpu()),
                    "source_margin_violation_ratio": float(replacement["source_margin_violation_ratio"].detach().cpu()),
                    "target_margin_violation_ratio": float(replacement["target_margin_violation_ratio"].detach().cpu()),
                    "target_weight_mean": float(selected_weights_t.mean().detach().cpu()) if selected_weights_t.numel() else 0.0,
                    "pseudo_selected": float(len(selected_pseudo_t)),
                }
            )

        # Diagnostics describe the complete frozen snapshot inference pass,
        # including rejected samples, rather than only accepted training rows.
        predicted_counts = pseudo_diagnostics["predicted_counts"].to(device)
        accepted_counts = pseudo_diagnostics["accepted_counts"].to(device)
        predicted_confidence_sums = pseudo_diagnostics["predicted_confidence_sums"].to(device)
        predicted_entropy_sums = pseudo_diagnostics["predicted_entropy_sums"].to(device)
        accepted_confidence_sums = pseudo_diagnostics["accepted_confidence_sums"].to(device)
        accepted_entropy_sums = pseudo_diagnostics["accepted_entropy_sums"].to(device)
        accepted_weight_sums = pseudo_diagnostics["accepted_weight_sums"].to(device)
        rejected_confidence_counts = pseudo_diagnostics["rejected_confidence_counts"].to(device)
        rejected_entropy_counts = pseudo_diagnostics["rejected_entropy_counts"].to(device)
        rejected_both_counts = pseudo_diagnostics["rejected_both_counts"].to(device)
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
                "prototype_loss/mode_replacement": float(replacement_losses),
                "prototype_loss/ramp_proto_align": ramps["proto_align"],
                "prototype_loss/ramp_comp_source": ramps["comp_source"],
                "prototype_loss/ramp_comp_target": ramps["comp_target"],
                "prototype_loss/ramp_sep_margin": ramps["sep_margin"],
                "prototype_loss/effective_lambda_proto_align": ramps["proto_align"] * float(prototype_loss_cfg.get("lambda_proto_align", 0.0)),
                "prototype_loss/effective_lambda_comp_source": ramps["comp_source"] * float(prototype_loss_cfg.get("lambda_comp_source", 0.0)),
                "prototype_loss/effective_lambda_comp_target": ramps["comp_target"] * float(prototype_loss_cfg.get("lambda_comp_target", 0.0)),
                "prototype_loss/effective_lambda_sep_margin": ramps["sep_margin"] * float(prototype_loss_cfg.get("lambda_sep_margin", 0.0)),
                "prototype_loss/non_finite": float(
                    not all(
                        np.isfinite(row[key])
                        for key in (
                            "loss",
                            "loss_cls",
                            "loss_proto_align",
                            "loss_comp_source",
                            "loss_comp_target",
                            "loss_sep_margin",
                        )
                    )
                ),
                "execution/batchnorm_frozen": float(batchnorm_mode != "train"),
                "execution/batchnorm_affine_frozen": float(batchnorm_mode == "freeze_all"),
                "execution/single_target_forward": float(target_forward_mode == "single"),
                "execution/target_once_epoch": float(epoch_driver == "target_once"),
                "execution/optimizer_steps": float(len(batch_rows)),
                "execution/source_samples_seen": float(source_samples_seen),
                "execution/target_samples_seen": float(target_samples_seen),
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
            accepted_weight_sums,
            rejected_confidence_counts,
            rejected_entropy_counts,
            rejected_both_counts,
            effective_counts,
            source_skips,
            target_skips,
            reliability,
        )
        _add_pair_diagnostics(
            row,
            class_names,
            source_pair_counts,
            source_pair_violations,
            target_pair_counts,
            target_pair_violations,
        )
        history.append(row)
        best_adapted_macro_f1 = max(best_adapted_macro_f1, float(row["val_macro_f1"]))
        _write_history_csv(history, history_path)
        run.log({f"adapt/{key}": value for key, value in row.items() if key not in {"epoch", "usage"}}, step=epoch)
        if row["val_macro_f1"] > best_macro_f1 + float(cfg.get("checkpoint_min_delta", 0.0)):
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
                init_val_macro_f1=init_val_macro_f1,
                best_adapted_macro_f1=best_adapted_macro_f1,
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
            init_val_macro_f1=init_val_macro_f1,
            best_adapted_macro_f1=best_adapted_macro_f1,
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
        aux_classifier.load_state_dict(copy.deepcopy(model.classifier.state_dict()))
        aux_classifier.eval()
        pseudo_dataset, pseudo_diagnostics = build_filtered_pseudo_snapshot(
            model=model,
            aux_classifier=aux_classifier,
            target_dataset=target_dataset,
            inference_loader=target_inference_loader,
            thresholds=thresholds,
            filter_cfg=filter_cfg,
            filter_thresholds=filter_thresholds,
            device=device,
            num_classes=num_classes,
        )

    summary = {
        "usage": usage,
        "init_checkpoint": str(init_checkpoint),
        "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint is not None else None,
        "latest_checkpoint": str(latest_path),
        "best_checkpoint": str(best_path),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_macro_f1,
        "init_val_macro_f1": init_val_macro_f1,
        "best_adapted_val_macro_f1": best_adapted_macro_f1,
        "adaptation_gain_over_init": (
            best_adapted_macro_f1 - init_val_macro_f1
            if init_val_macro_f1 is not None and best_adapted_macro_f1 >= 0.0
            else None
        ),
        "selected_stage": "initialization" if best_epoch == -1 else "adaptation",
        "checkpoint_selection": "source_validation_macro_f1_including_initialization",
        "target_labels_used_during_training": False,
        "target_inputs_overlap_evaluation": str(config["data"].get("target_protocol", ""))
        in {"first5_adapt_full_test", "full_target_transductive"},
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


def _compute_replacement_losses(
    z_s,
    y_s,
    z_t,
    pseudo_t,
    target_weights,
    candidates,
    cfg,
    margin_matrix,
    min_target_count,
):
    zero = z_s.sum() * 0.0
    empty_pair = torch.zeros_like(margin_matrix)

    if bool(cfg.get("use_comp_source", False)):
        loss_comp_source, comp_source_diag = source_compactness_loss(
            z_s, y_s, candidates.source, candidates.source_valid
        )
    else:
        loss_comp_source = zero
        comp_source_diag = {"active_samples": zero.detach()}

    if bool(cfg.get("use_comp_target", False)):
        loss_comp_target, comp_target_diag = target_compactness_loss(
            z_t, pseudo_t, candidates.global_, candidates.global_valid, target_weights
        )
    else:
        loss_comp_target = z_t.sum() * 0.0
        comp_target_diag = {"active_samples": zero.detach()}

    if bool(cfg.get("use_proto_align", False)):
        loss_align, align_diag = directed_target_alignment_loss(
            z_t,
            pseudo_t,
            target_weights,
            candidates.source,
            candidates.source_valid,
            min_target_count,
        )
    else:
        loss_align = z_t.sum() * 0.0
        align_diag = {"active_classes": zero.detach()}

    if bool(cfg.get("use_sep_margin", False)):
        loss_sep_source, source_sep_diag = sample_prototype_margin_loss(
            z_s, y_s, candidates.source, candidates.source_valid, margin_matrix
        )
        loss_sep_target, target_sep_diag = sample_prototype_margin_loss(
            z_t,
            pseudo_t,
            candidates.global_,
            candidates.global_valid,
            margin_matrix,
            sample_weights=target_weights,
        )
        if float(target_sep_diag["active_samples"]) > 0.0:
            loss_sep = 0.5 * (loss_sep_source + loss_sep_target)
        else:
            loss_sep = loss_sep_source
    else:
        loss_sep_source = zero
        loss_sep_target = z_t.sum() * 0.0
        loss_sep = zero
        source_sep_diag = {
            "active_samples": zero.detach(),
            "pair_counts": empty_pair,
            "pair_violations": empty_pair,
            "violation_ratio": zero.detach(),
        }
        target_sep_diag = dict(source_sep_diag)

    return {
        "loss_proto_align": loss_align,
        "loss_comp_source": loss_comp_source,
        "loss_comp_target": loss_comp_target,
        "loss_sep_margin": loss_sep,
        "loss_sep_source": loss_sep_source,
        "loss_sep_target": loss_sep_target,
        "active_source_comp_samples": comp_source_diag["active_samples"],
        "active_target_comp_samples": comp_target_diag["active_samples"],
        "active_align_classes": align_diag["active_classes"],
        "active_source_sep_samples": source_sep_diag["active_samples"],
        "active_target_sep_samples": target_sep_diag["active_samples"],
        "source_margin_violation_ratio": source_sep_diag["violation_ratio"],
        "target_margin_violation_ratio": target_sep_diag["violation_ratio"],
        "source_pair_counts": source_sep_diag["pair_counts"],
        "source_pair_violations": source_sep_diag["pair_violations"],
        "target_pair_counts": target_sep_diag["pair_counts"],
        "target_pair_violations": target_sep_diag["pair_violations"],
    }


def _empty_replacement_metrics(features):
    zero = features.sum() * 0.0
    return {
        "loss_proto_align": zero,
        "loss_comp_source": zero,
        "loss_comp_target": zero,
        "loss_sep_margin": zero,
        "loss_sep_source": zero,
        "loss_sep_target": zero,
        "active_source_comp_samples": zero.detach(),
        "active_target_comp_samples": zero.detach(),
        "active_align_classes": zero.detach(),
        "active_source_sep_samples": zero.detach(),
        "active_target_sep_samples": zero.detach(),
        "source_margin_violation_ratio": zero.detach(),
        "target_margin_violation_ratio": zero.detach(),
    }


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
    accepted_weight_sums,
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
        row[f"prototype_loss/mean_target_weight_{name}"] = (
            float(accepted_weight_sums[index] / accepted[index]) if accepted_count else 0.0
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
        row[f"prototype/source_valid_{name}"] = float(bank.source_valid[index])
        row[f"prototype/target_valid_{name}"] = float(bank.target_valid[index])


def _add_pair_diagnostics(row, class_names, source_counts, source_violations, target_counts, target_violations):
    for positive, positive_name in enumerate(class_names):
        for negative, negative_name in enumerate(class_names):
            if positive == negative:
                continue
            for domain, counts, violations in (
                ("source", source_counts, source_violations),
                ("target", target_counts, target_violations),
            ):
                count = float(counts[positive, negative])
                ratio = float(violations[positive, negative] / counts[positive, negative]) if count else 0.0
                row[f"prototype_loss/{domain}_margin_violation_{positive_name}_{negative_name}"] = ratio


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
    init_val_macro_f1=None,
    best_adapted_macro_f1=-1.0,
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
            "init_val_macro_f1": float(init_val_macro_f1) if init_val_macro_f1 is not None else None,
            "best_adapted_macro_f1": float(best_adapted_macro_f1),
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


def configure_adaptation_batchnorm(model: torch.nn.Module, mode: str) -> None:
    """Configure BatchNorm parameters once, before optimizer construction."""
    mode = str(mode).lower()
    if mode not in VALID_BATCHNORM_MODES:
        raise ValueError(f"Unknown BatchNorm adaptation mode: {mode}")
    freeze_affine = mode == "freeze_all"
    for module in model.modules():
        if not isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            continue
        if module.affine:
            module.weight.requires_grad_(not freeze_affine)
            module.bias.requires_grad_(not freeze_affine)
        if mode != "train":
            module.eval()


def set_adaptation_train_mode(model: torch.nn.Module, batchnorm_mode: str) -> None:
    """Enter train mode while keeping frozen BatchNorm modules in eval mode."""
    model.train()
    if str(batchnorm_mode).lower() == "train":
        return
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()


def paired_adaptation_batches(source_loader, target_loader, *, epoch_driver: str):
    """Pair domains without silently repeating target samples in target-once mode."""
    epoch_driver = str(epoch_driver).lower()
    if epoch_driver == "source":
        target_iter = _cycle(target_loader)
        for source_batch in source_loader:
            yield source_batch, next(target_iter)
        return
    if epoch_driver == "target_once":
        source_iter = _cycle(source_loader)
        for target_batch in target_loader:
            yield next(source_iter), target_batch
        return
    raise ValueError(f"Unknown adaptation epoch driver: {epoch_driver}")


def build_filtered_pseudo_snapshot(
    *,
    model,
    aux_classifier,
    target_dataset: Dataset,
    inference_loader: DataLoader,
    thresholds: torch.Tensor,
    filter_cfg: dict[str, Any],
    filter_thresholds: torch.Tensor,
    device: torch.device,
    num_classes: int,
) -> tuple[DAEACPseudoLabeledDataset, dict[str, torch.Tensor]]:
    """Run F+h over the complete target set and freeze accepted labels for one epoch."""
    keys = (
        "predicted_counts",
        "accepted_counts",
        "predicted_confidence_sums",
        "predicted_entropy_sums",
        "accepted_confidence_sums",
        "accepted_entropy_sums",
        "accepted_weight_sums",
        "rejected_confidence_counts",
        "rejected_entropy_counts",
        "rejected_both_counts",
    )
    diagnostics = {key: torch.zeros(num_classes, dtype=torch.float32) for key in keys}
    positions: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    confidences: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    offset = 0
    model.eval()
    aux_classifier.eval()
    filter_enabled = bool(filter_cfg.get("enabled", False))
    with torch.no_grad():
        for batch in inference_loader:
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            x = x.to(device)
            features = model.extract_features(x)
            _, probabilities = aux_classifier(features, return_logits=True)
            if filter_enabled:
                result = filter_target_pseudolabels(
                    probabilities,
                    mode=str(filter_cfg["mode"]),
                    global_confidence_threshold=float(filter_cfg["global_confidence_threshold"]),
                    class_confidence_thresholds=filter_thresholds,
                    max_normalized_entropy=float(filter_cfg["max_normalized_entropy"]),
                )
                pseudo = result.pseudo_labels
                confidence = result.confidence
                entropy = result.normalized_entropy
                accepted = result.accepted_mask
                rejected_confidence = result.rejected_confidence_mask
                rejected_entropy = result.rejected_entropy_mask
                rejected_both = result.rejected_both_mask
            else:
                confidence, pseudo = probabilities.max(dim=1)
                entropy = -(probabilities * probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()).sum(dim=1)
                entropy = entropy / np.log(num_classes)
                accepted = confidence > thresholds[pseudo]
                rejected_confidence = ~accepted
                rejected_entropy = torch.zeros_like(accepted)
                rejected_both = torch.zeros_like(accepted)

            accepted_labels = pseudo[accepted]
            accepted_confidence = confidence[accepted]
            accepted_entropy = entropy[accepted]
            accepted_weights = target_reliability_weights(accepted_confidence, accepted_entropy)
            local_positions = torch.arange(offset, offset + len(x), device=device)
            positions.append(local_positions[accepted].cpu())
            labels.append(accepted_labels.cpu())
            confidences.append(accepted_confidence.cpu())
            entropies.append(accepted_entropy.cpu())
            offset += len(x)

            diagnostics["predicted_counts"] += torch.bincount(pseudo.cpu(), minlength=num_classes).float()
            diagnostics["accepted_counts"] += torch.bincount(accepted_labels.cpu(), minlength=num_classes).float()
            diagnostics["predicted_confidence_sums"].scatter_add_(0, pseudo.cpu(), confidence.cpu())
            diagnostics["predicted_entropy_sums"].scatter_add_(0, pseudo.cpu(), entropy.cpu())
            if bool(accepted.any()):
                diagnostics["accepted_confidence_sums"].scatter_add_(0, accepted_labels.cpu(), accepted_confidence.cpu())
                diagnostics["accepted_entropy_sums"].scatter_add_(0, accepted_labels.cpu(), accepted_entropy.cpu())
                diagnostics["accepted_weight_sums"].scatter_add_(0, accepted_labels.cpu(), accepted_weights.cpu())
            for key, mask in (
                ("rejected_confidence_counts", rejected_confidence),
                ("rejected_entropy_counts", rejected_entropy),
                ("rejected_both_counts", rejected_both),
            ):
                if bool(mask.any()):
                    diagnostics[key] += torch.bincount(pseudo[mask].cpu(), minlength=num_classes).float()

    if not labels or sum(len(value) for value in labels) == 0:
        raise RuntimeError("No target samples were accepted while building the epoch pseudo-label snapshot.")
    dataset = DAEACPseudoLabeledDataset(
        target_dataset,
        torch.cat(positions),
        torch.cat(labels),
        torch.cat(confidences),
        torch.cat(entropies),
    )
    return dataset, diagnostics


def forward_target_for_pseudolabels(model, aux_classifier, inputs, *, mode: str):
    """Extract target features once when requested and detach only the pseudo-label branch."""
    mode = str(mode).lower()
    if mode == "single":
        features = model.extract_features(inputs)
        pseudo_features = features.detach()
    elif mode == "double":
        with torch.no_grad():
            features = model.extract_features(inputs)
        pseudo_features = features
    else:
        raise ValueError(f"Unknown target forward mode: {mode}")
    with torch.no_grad():
        _, probabilities = aux_classifier(pseudo_features, return_logits=True)
    return features, probabilities


def _cycle(loader):
    while True:
        yield from loader


def _scatter_mask_counts(destination, labels, mask):
    if bool(mask.any()):
        destination.add_(torch.bincount(labels[mask], minlength=len(destination)))

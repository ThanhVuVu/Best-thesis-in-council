# Current Repo Audit Map

Last updated: 2026-06-27.

This audit map reflects the current implementation state. It is not the old
Phase 0 plan output anymore.

## File/Function Map

### 1. Main DAEAC entrypoints

- `ecg_thesis/scripts/phase6_daeac_paper/00_validate_data.py`
  - Loads config, builds the model, inspects all configured `.npz` inputs, checks RR row neutralization when `rr_mode="real"`, and validates late-fusion RR feature availability.
- `ecg_thesis/scripts/phase6_daeac_paper/01_train_base.py`
  - Builds source train/val datasets and calls `train_daeac_base(...)`.
- `ecg_thesis/scripts/phase6_daeac_paper/02_adapt_uda.py`
  - Builds source, unlabeled target, source-val, and target-val datasets, then calls `adapt_daeac(...)`.
- `ecg_thesis/scripts/phase6_daeac_paper/03_eval.py`
  - Loads a checkpoint and evaluates the requested dataset split.

### 2. Active Kaggle all-domain notebook

- `ecg_thesis/notebooks/phase6_daeac_fcba_latefusion_rr_nsv_daeac_all_domains_kaggle.ipynb`
  - Runs 3-class NSV FCBA + RR late-fusion DAEAC adaptation for:
    - `ds1_ds2`;
    - `ds1_incart`;
    - `ds1_svdb`;
    - `mitbih_incart`;
    - `mitbih_svdb`.
  - Calls `00_validate_data.py`, `02_adapt_uda.py`, and `03_eval.py`.
  - Expects `data/processed/phase6_daeac_record_splits_nsv`.
  - Requires pretrained source checkpoints:
    - `daeac_fcba_rr_nsv_sqrtw_base_best.pt`;
    - `daeac_fcba_rr_nsv_mitbih_sqrtw_base_best.pt`.

### 3. Model class and forward output

- `ecg_thesis/src/models/daeac_paper.py`
- `DAEACNetwork`
  - Default tuple behavior is preserved:
    - `forward(x)` returns `(features, probs)`.
    - `forward(x, return_logits=True)` returns `(features, logits, probs)`.
  - Dict behavior is available:
    - `forward(..., return_dict=True)` returns `features`, `logits`, `probabilities`, and `feature_layers`.
  - `feature_layers` includes:
    - `transition_gap`;
    - `final_aspp_gap`;
    - `pre_adaptation_gap`;
    - `dan_fc`;
    - `gap_embed`.
- `ClassifierH`
  - Default single-head classifier.
- `DualClassifierH`
  - Optional dual-head classifier for RTD reliable pseudo-labeling.
  - Public logits are the average of head 1 and head 2 logits.
- `LateFusionClassifierH`
  - Optional RR late-fusion classifier.
  - Used by the active NSV all-domain notebook branch.
- `DualLateFusionClassifierH`
  - Optional RR late-fusion dual-head classifier.
  - Used by the new NSV `dhrel` all-domain notebook branch.
  - Public logits are the average of the two late-fusion heads.
- `FrequencyConvolutionBlockAttention2D`
  - FCBA attention option used by FCBA configs.

### 4. Feature dimensions by branch

- RTD paper/default DAEAC branch:
  - input shape `[N, 1, 3, 128]`;
  - center feature is 256-d `gap_embed`.
- NSV FCBA + RR late-fusion branch:
  - dataset keeps real 3-row input internally;
  - `morphology_only=True` slices CNN input to `[N, 1, 1, 128]`;
  - 7-d `rr_features` are passed separately to `LateFusionClassifierH`;
  - center/classification feature after late-fusion morphology FC is 128-d.

### 5. Dataset classes

- `ecg_thesis/src/data/daeac_dataset.py`
- `DAEACDataset`
  - Loads `[N, 1, 3, 128]` or morphology-only `[N, 1, 1, 128]`.
  - Defaults to `rr_mode="real"`.
  - Preserves real Row 1/2 by default.
  - Neutralizes Row 1/2 only when `rr_mode="neutralized_legacy"` is explicitly requested.
  - Can return 7-d RR features with `return_rr_features=True`.
  - Can slice morphology-only input with `morphology_only=True`.
- `DAEACTargetUnlabeledDataset`
  - Uses `require_labels=False`.
  - Does not expose target labels from `__getitem__`.
- `DAEACPseudoLabeledDataset`
  - Immutable per-epoch target pseudo-label snapshot.
  - Preserves RR features when the wrapped target dataset returns them.
- `inspect_daeac_npz(...)`
  - Reports shape, class counts, row stats, RR mode, and whether RR rows are neutralized.

### 6. Pseudo-label refresh

- `build_pseudo_labeled_target_dataset(...)`
- Location: `ecg_thesis/src/training/train_daeac_paper.py`
- Confidence-only mode:
  - Uses frozen auxiliary classifier snapshot.
  - Selects target samples by class-specific confidence thresholds.
- Reliable mode:
  - Uses `ReliablePseudoLabelSelector`.
  - Requires `rtd_daeac.dual_head.enabled=true`.
  - Applies confidence, source-distance, and dual-head discrepancy gates.
- Snapshot timing:
  - Initial snapshot before epoch 1.
  - Refreshed after each epoch.
  - If refresh has zero confident target samples, the previous valid snapshot is retained.

### 7. Center memory

- `CenterMemory`
- Location: `ecg_thesis/src/training/train_daeac_paper.py`
- Stores source, target, and mixed centers.
- `centers_for_loss(...)` applies EMA with `center_ema_gamma`.
- `commit(...)` detaches and stores updated centers.
- `compute_global_source_centers(...)` and `compute_global_pseudo_target_centers(...)` initialize global centers.

### 8. Current loss functions

- `ecg_thesis/src/training/daeac_losses.py`
- Source classification losses:
  - `WeightedCrossEntropyByBatchSize`;
  - `CustomFocalLoss`;
  - `ClassBalancedFocalLoss`.
- Loss builder:
  - `build_daeac_classification_loss(...)`;
  - supports `weighted_ce`, `focal`, and `class_balanced_focal`.
- Center losses:
  - `cluster_aligning_loss`;
  - `separating_loss`;
  - `compacting_loss`.
- Adaptation total in current DAEAC path:
  - source classification loss;
  - `beta1 * loss_align`;
  - `beta2 * loss_sep`;
  - `beta2 * loss_comp`.

### 9. V-Measure and target-label access

- `ecg_thesis/src/training/v_measure_validation.py`
- `ericsson_v_measure(source_logits, source_labels, target_logits, ...)`
- Target labels are not passed.
- Target reference labels are computed from target logits, not target ground truth.
- In `adapt_daeac(...)`, target logits are collected by `_daeac_target_logits(...)`, which unpacks only target inputs/RR features.
- Audit result: current V-Measure path is label-free with respect to target labels.

### 10. Checkpointing and early stopping

- `adapt_daeac(...)` in `ecg_thesis/src/training/train_daeac_paper.py`
- Best checkpoint saved when:
  - `row["valid"]`;
  - `row["v_measure"] > best_v_measure + validation.min_delta`.
- Latest checkpoint saved every epoch.
- Config fields:
  - `validation.min_epochs`;
  - `validation.patience`;
  - `validation.min_delta`.
- Current configs commonly use:
  - `min_epochs: 20`;
  - `patience: 10`;
  - `min_delta: 0.0001`.

### 11. Config system

- Loader/common helpers:
  - `ecg_thesis/scripts/phase6_daeac_paper/common.py`
  - `load_phase1_config(...)`
  - `cfg_path(...)`
- Supports recursive `extends`.
- Relevant RTD/core configs:
  - `ecg_thesis/configs/phase6_daeac_paper.yaml`;
  - `ecg_thesis/configs/phase6_daeac_paper_dualhead.yaml`;
  - `ecg_thesis/configs/phase6_daeac_paper_dualhead_reliable.yaml`;
  - `ecg_thesis/configs/phase6_daeac_paper_phase3_focal.yaml`.
- Relevant active all-domain notebook configs:
  - `ecg_thesis/configs/phase6_daeac_fcba_latefusion_rr_nsv_daeac_ds1_ds2.yaml`;
  - `ecg_thesis/configs/phase6_daeac_fcba_latefusion_rr_nsv_daeac_ds1_incart.yaml`;
  - `ecg_thesis/configs/phase6_daeac_fcba_latefusion_rr_nsv_daeac_ds1_svdb.yaml`;
  - `ecg_thesis/configs/phase6_daeac_fcba_latefusion_rr_nsv_daeac_mitbih_incart.yaml`;
  - `ecg_thesis/configs/phase6_daeac_fcba_latefusion_rr_nsv_daeac_mitbih_svdb.yaml`.
- Relevant dual-head reliable all-domain notebook configs:
  - `ecg_thesis/configs/phase6_daeac_fcba_latefusion_rr_nsv_dhrel_ds1_ds2.yaml`;
  - `ecg_thesis/configs/phase6_daeac_fcba_latefusion_rr_nsv_dhrel_ds1_incart.yaml`;
  - `ecg_thesis/configs/phase6_daeac_fcba_latefusion_rr_nsv_dhrel_ds1_svdb.yaml`;
  - `ecg_thesis/configs/phase6_daeac_fcba_latefusion_rr_nsv_dhrel_mitbih_incart.yaml`;
  - `ecg_thesis/configs/phase6_daeac_fcba_latefusion_rr_nsv_dhrel_mitbih_svdb.yaml`.
- Their base configs:
  - `ecg_thesis/configs/phase6_daeac_fcba_latefusion_rr_nsv.yaml`;
  - `ecg_thesis/configs/phase6_daeac_fcba_latefusion_rr_nsv_mitbih_sqrtw.yaml`;
  - `ecg_thesis/configs/phase6_daeac_fcba_latefusion_rr_nsv_mitbih_os.yaml`.

## Implemented Status

- Phase 1A real RR input: implemented.
- Phase 1B return-dict/config-forward refactor: implemented.
- Phase 2A dual-head classifier: implemented.
- Phase 2B reliable pseudo-labeling: implemented.
- Phase 3 source focal/class-balanced focal loss plumbing: implemented.
- FCBA attention: implemented.
- RR late fusion with 7-d `rr_features`: implemented.
- NSV all-domain Kaggle adaptation notebook branch: implemented as an experiment runner with RTD flags disabled.
- NSV dual-head reliable all-domain Kaggle notebook branch: implemented with `S` confidence threshold `0.97`.

## Current Risks

- The active all-domain notebook is 3-class NSV, while RTD core notes often assume 4-class `N/S/V/F`; do not mix their metrics/config assumptions.
- FCBA + RR late fusion is implemented and active in the notebook branch, but is not enabled in RTD core configs by default.
- Late fusion can now be combined with dual-head reliable pseudo-labeling through `DualLateFusionClassifierH`; old single-head checkpoints rely on head-copy initialization.
- `morphology_only=True` means CNN input is `[N, 1, 1, 128]` in the NSV branch, even though the raw dataset still stores `[N, 1, 3, 128]`.
- V-Measure is label-free now, but future checkpoint changes must not switch target validation to labeled supervision.
- Target labels may exist inside `.npz` files, but `DAEACTargetUnlabeledDataset.__getitem__` must keep not exposing them during adaptation.
- Full all-domain notebook runs depend on external Kaggle inputs and pretrained checkpoints, so local Codex verification should stay smoke-only.
- Phase 4 dynamic weighting and Phase 5 task-oriented alignment have config flags, but should not be treated as core-implemented unless code is explicitly audited.

## Recommended Implementation Order

1. Keep `03_current_setup.md` and this audit map synchronized before starting each new phase.
2. If continuing RTD core, implement only one next phase at a time after Phase 2B.
3. Before modifying checkpoint logic, re-audit `v_measure_validation.py` and `adapt_daeac(...)`.
4. Keep NSV all-domain FCBA late-fusion notebook changes separate from RTD core method changes.

## Smoke Commands

```bash
pytest ecg_thesis/tests/test_rtd_daeac_phase1a_rr_rows.py -v
pytest ecg_thesis/tests/test_rtd_daeac_phase1b_forward.py -v
pytest ecg_thesis/tests/test_rtd_daeac_phase2a_dual_head.py -v
pytest ecg_thesis/tests/test_rtd_daeac_phase2b_reliable_pseudo.py -v
pytest ecg_thesis/tests/test_daeac_losses.py -v
pytest ecg_thesis/tests/test_daeac_fcba.py ecg_thesis/tests/test_daeac_rr_late_fusion.py -v
pytest ecg_thesis/tests/test_daeac_late_fusion_dual_head.py -v
pytest ecg_thesis/tests/test_v_measure_validation.py -v
```

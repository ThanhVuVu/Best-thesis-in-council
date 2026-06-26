# Current RTD-DAEAC Setup

Last updated: 2026-06-27.

This file is the compact implementation handoff for the current RTD-DAEAC code
state. It should describe what the repo does now, not the full future plan.

## 1. Problem setup

- Task: ECG heartbeat arrhythmia classification with unsupervised domain adaptation.
- RTD-DAEAC core classes: `N`, `S`, `V`, `F`.
- Active all-domain Kaggle notebook classes: `N`, `S`, `V` only.
- Source domains: `ds1`, `mitbih`.
- Target domains: `ds2`, `incart`, `svdb`.
- Main scenarios: `ds1_ds2`, `ds1_incart`, `ds1_svdb`, `mitbih_incart`, `mitbih_svdb`.
- UDA type: transductive UDA using unlabeled target adaptation data; target test is held out.
- Target adaptation data: `{target}_train.npz`; target validation logits from `{target}_val.npz` are used for Ericsson V-Measure monitoring without exposing target labels.
- Target test data: `{target}_test.npz`.
- The notebook `ecg_thesis/notebooks/phase6_daeac_fcba_latefusion_rr_nsv_daeac_all_domains_kaggle.ipynb` runs the 3-class NSV FCBA + RR late-fusion DAEAC adaptation across all five scenarios.

## 2. Input/data

- Input shape: `[N, 1, 3, 128]`.
- Current input channels/features: 1 Conv2D input channel; 3 feature rows per beat.
- Row 0: normalized heartbeat morphology.
- Row 1: real `pre_rr_ratio`.
- Row 2: real `near_pre_rr_ratio`.
- `DAEACDataset` now defaults to `rr_mode="real"` and preserves stored Row 1/2 values.
- Legacy neutralization is available only through explicit `rr_mode="neutralized_legacy"`.
- `DAEACTargetUnlabeledDataset` inherits the same RR behavior and intentionally does not expose target labels.
- `inspect_daeac_npz()` reports row-wise stats and whether RR rows are neutralized.
- `morphology_only=True` exists for explicit morphology-only runs and changes input rows to 1.
- The active NSV FCBA late-fusion configs use `morphology_only: true`, so the CNN receives `[N, 1, 1, 128]` morphology input while 7 RR features are passed separately to the late-fusion classifier.
- Those configs still keep `rr_mode: real` and `return_rr_features: true`; real Row 1/2 values are not neutralized before the morphology-only slice.
- Main configs should use or inherit `data.rr_mode: real`.
- Sampling rate / beat length: unified 360 Hz; fixed length 128; segment from 0.14s after previous R peak to 0.28s after current R peak.

Relevant files:

```text
ecg_thesis/src/data/daeac_dataset.py
ecg_thesis/tests/test_rtd_daeac_phase1a_rr_rows.py
```

## 3. Current model

- Backbone: DAEAC CNN with initial Conv2D, ASPP-SE blocks, residual blocks, final ASPP-SE, and global average pooling.
- Feature embedding used for centers: 256-d `gap_embed`.
- `return_dict=True` exposes `features`, `logits`, `probabilities`, and `feature_layers`.
- `feature_layers` includes `gap_embed`, `pre_adaptation_gap`, and `dan_fc`.
- Tuple forward behavior is preserved for old code paths.
- Default classifier: `ClassifierH(feature_dim=256, num_classes=4)`.
- Optional dual classifier: `DualClassifierH`, enabled by `rtd_daeac.dual_head.enabled`.
- Optional late-fusion dual classifier: `DualLateFusionClassifierH`, used when `late_fusion.enabled=true` and `rtd_daeac.dual_head.enabled=true`.
- In dual-head mode, public logits are the average of head 1 and head 2 logits; per-head logits/probabilities are exposed in dict output.
- Optional FCBA attention and late-fusion RR classifier code exist, but they are not part of the current RTD-DAEAC core default.
- In the active NSV all-domain notebook, FCBA attention and RR late fusion are enabled:
  - `model.attention: fcba`;
  - `model.input_rows: 1`;
  - `model.late_fusion.enabled: true`;
  - `model.late_fusion.rr_dim: 7`;
  - late-fusion `gap_embed`/center feature dimension is 128 after the morphology FC layer.
- Baseline all-domain notebook configs keep dual head and reliable pseudo-labeling disabled.
- New `dhrel` all-domain configs enable late-fusion dual head plus reliable pseudo-labeling with `S` confidence threshold lowered to `0.97`.

Relevant files:

```text
ecg_thesis/src/models/daeac_paper.py
ecg_thesis/tests/test_rtd_daeac_phase1b_forward.py
ecg_thesis/tests/test_rtd_daeac_phase2a_dual_head.py
ecg_thesis/tests/test_daeac_fcba.py
ecg_thesis/tests/test_daeac_rr_late_fusion.py
ecg_thesis/tests/test_daeac_late_fusion_dual_head.py
```

## 4. Current training/adaptation method

- Main entrypoint logic is in `train_daeac_base()` and `adapt_daeac()`.
- Current baseline behavior remains DAEAC-like center-based UDA when RTD flags are disabled.
- Source pretraining uses source classification loss only.
- Adaptation uses source classification loss plus `beta1 * cluster_aligning_loss`, `beta2 * separating_loss`, and `beta2 * compacting_loss`.
- Config default in `phase6_daeac_paper.yaml`:
  - `losses.source_cls_loss: weighted_ce`;
  - `cluster_loss_reduction: mean`;
  - `beta1: 0.1`;
  - `beta2: 0.1`;
  - `center_ema_gamma: 0.1`.
- The NSV FCBA late-fusion all-domain configs also use weighted CE, `beta1: 0.1`, `beta2: 0.1`, `center_ema_gamma: 0.1`, and `cluster_loss_reduction: mean`.
- DS1-source all-domain configs use sqrt class weights and initialize from `daeac_fcba_rr_nsv_sqrtw_base_best.pt`.
- MITBIH-source all-domain configs extend the MITBIH sqrt-weight config and initialize from `daeac_fcba_rr_nsv_mitbih_sqrtw_base_best.pt`.
- `CenterMemory` maintains source, target, and mixed centers with per-batch EMA.
- Pseudo-label threshold defaults: `N > 0.999`, `S > 0.99`, `V > 0.99`, `F > 0.99`.
- NSV configs use thresholds only for `N`, `S`, `V`.
- Pseudo-label snapshot is built before epoch 1 and refreshed after each epoch.
- If refresh produces no confident target samples, the previous valid snapshot is retained.

Relevant files:

```text
ecg_thesis/src/training/train_daeac_paper.py
ecg_thesis/src/training/daeac_losses.py
ecg_thesis/configs/phase6_daeac_paper.yaml
```

## 5. Implemented RTD-DAEAC phases

### Phase 1A - real RR input

- Implemented.
- `rr_mode="real"` is default.
- `rr_mode="neutralized_legacy"` is explicit legacy behavior only.
- Target unlabeled dataset keeps real RR rows and does not expose labels.
- Tests cover real RR preservation, legacy neutralization, unlabeled target behavior, and row stats.

Smoke:

```bash
pytest ecg_thesis/tests/test_rtd_daeac_phase1a_rr_rows.py -v
```

### Phase 1B - config and forward-output refactor

- Implemented.
- `return_dict=True` exposes logits and `gap_embed` feature layers.
- Default RTD flags are present and off in `phase6_daeac_paper.yaml`.
- Existing tuple forward behavior is preserved.

Smoke:

```bash
pytest ecg_thesis/tests/test_rtd_daeac_phase1b_forward.py -v
```

### Phase 2A - dual-head classifier

- Implemented.
- `DualClassifierH` has independent `fc` and `fc2` heads.
- Dual-head public logits are averaged logits.
- Old single-head checkpoints can initialize dual-head models.
- Config: `phase6_daeac_paper_dualhead.yaml`.

Smoke:

```bash
pytest ecg_thesis/tests/test_rtd_daeac_phase2a_dual_head.py -v
```

### Phase 2B - reliable pseudo-labeling

- Implemented.
- `ReliablePseudoLabelSelector` uses three gates: class confidence threshold, source-center distance threshold, and dual-head discrepancy threshold.
- Source thresholds are percentile-based and finite.
- `PseudoLabelBank` stores accepted pseudo-label metadata by target index.
- Reliable mode requires dual head.
- Target labels are not accessed by reliable selection tests.
- Config: `phase6_daeac_paper_dualhead_reliable.yaml`.

Smoke:

```bash
pytest ecg_thesis/tests/test_rtd_daeac_phase2b_reliable_pseudo.py -v
```

### Phase 3 - class-balanced focal source loss

- Loss implementation exists.
- Supported source classification losses: `weighted_ce`, `focal`, and `class_balanced_focal`.
- `ClassBalancedFocalLoss` uses effective-number class weighting and gamma default `2.35`.
- Configs exist for focal variants.
- Treat as implemented loss plumbing, but verify with smoke before depending on a full experiment.

Smoke:

```bash
pytest ecg_thesis/tests/test_daeac_losses.py -v
```

### FCBA + RR late-fusion NSV notebook branch

- Implemented as a separate experiment branch from the RTD core phases.
- Active notebook:
  - `ecg_thesis/notebooks/phase6_daeac_fcba_latefusion_rr_nsv_daeac_all_domains_kaggle.ipynb`.
- It runs adaptation and target-test evaluation for:
  - `ds1_ds2`;
  - `ds1_incart`;
  - `ds1_svdb`;
  - `mitbih_incart`;
  - `mitbih_svdb`.
- It expects the NSV record-split bundle under `data/processed/phase6_daeac_record_splits_nsv`.
- Required bundle files include `ds1_train/val`, `mitbih_train/val`, and `ds2/incart/svdb` train/val/test `.npz` files.
- It requires two pretrain checkpoints copied into their configured output paths:
  - `daeac_fcba_rr_nsv_sqrtw_base_best.pt`;
  - `daeac_fcba_rr_nsv_mitbih_sqrtw_base_best.pt`.
- It validates every scenario with `scripts/phase6_daeac_paper/00_validate_data.py`, runs adaptation with `02_adapt_uda.py`, evaluates target test with `03_eval.py`, writes an all-domain summary CSV, and zips outputs.
- This branch has `rtd_daeac.enabled: false`, `dual_head.enabled: false`, and `reliable_pseudo.enabled: false` in the adaptation configs.
- Treat this as FCBA + RR late-fusion DAEAC all-domain evaluation, not as the dual-head reliable RTD-DAEAC core.
- Dual-head reliable variant:
  - active notebook: `ecg_thesis/notebooks/phase6_daeac_fcba_latefusion_rr_nsv_dhrel_all_domains_kaggle.ipynb`;
  - configs use `rtd_daeac.enabled: true`, `dual_head.enabled: true`, and `reliable_pseudo.enabled: true`;
  - reliable confidence thresholds are `N: 0.995`, `S: 0.97`, `V: 0.99`;
  - old single-head late-fusion pretrain checkpoints initialize the second fusion head by copying the first head.

Smoke:

```bash
pytest ecg_thesis/tests/test_daeac_fcba.py ecg_thesis/tests/test_daeac_rr_late_fusion.py -v
pytest ecg_thesis/tests/test_daeac_late_fusion_dual_head.py -v
```

## 6. Not yet core-implemented

These are planned or optional pieces and must not be silently enabled:

- Phase 4 dynamic weight controller: config flags exist, but core adaptive weighting should be treated as not implemented unless verified in code.
- Phase 5 task-oriented alignment: config flags exist, but ToAlign source-positive feature replacement should be treated as not implemented unless verified in code.
- Phase 6 checkpoint/V-Measure fallback modes: V-Measure is audited as label-free, but checkpoint logic should still be audited before changes.
- Phase 7 Kaggle notebook/run-script polish for RTD-DAEAC core is not complete.
- Optional ablations such as MCC, implicit alignment, Transformer, Gram-OT, and legacy neutralized RR reproduction are separate from the RTD-DAEAC core.
- FCBA and RR late fusion are implemented for the NSV all-domain notebook branch; late-fusion dual-head reliable pseudo-labeling is available in the new `dhrel` configs.

## 7. Validation/checkpoint

- Current adaptation checkpoint monitor: Ericsson V-Measure.
- Current V-Measure implementation uses source labels and source/target logits; target labels are not used.
- Best checkpoint rule in adaptation:
  - save best checkpoint by maximum valid Ericsson V-Measure;
  - `validation.min_delta: 0.0001`;
  - config includes `min_epochs: 20` and `patience: 10`.
- Adaptation still saves latest checkpoint every epoch.
- Before modifying checkpoint logic, audit:
  - `ecg_thesis/src/training/v_measure_validation.py`;
  - `adapt_daeac()` in `ecg_thesis/src/training/train_daeac_paper.py`;
  - `ecg_thesis/tests/test_v_measure_validation.py`.

Smoke:

```bash
pytest ecg_thesis/tests/test_v_measure_validation.py -v
```

## 8. Current configs to know

- Baseline/current DAEAC with real RR and RTD flags off: `ecg_thesis/configs/phase6_daeac_paper.yaml`
- Dual-head only: `ecg_thesis/configs/phase6_daeac_paper_dualhead.yaml`
- Dual-head reliable pseudo-labeling: `ecg_thesis/configs/phase6_daeac_paper_dualhead_reliable.yaml`
- Active all-domain NSV FCBA late-fusion notebook configs:
  - `ecg_thesis/configs/phase6_daeac_fcba_latefusion_rr_nsv_daeac_ds1_ds2.yaml`
  - `ecg_thesis/configs/phase6_daeac_fcba_latefusion_rr_nsv_daeac_ds1_incart.yaml`
  - `ecg_thesis/configs/phase6_daeac_fcba_latefusion_rr_nsv_daeac_ds1_svdb.yaml`
  - `ecg_thesis/configs/phase6_daeac_fcba_latefusion_rr_nsv_daeac_mitbih_incart.yaml`
  - `ecg_thesis/configs/phase6_daeac_fcba_latefusion_rr_nsv_daeac_mitbih_svdb.yaml`
- Their base configs:
  - `ecg_thesis/configs/phase6_daeac_fcba_latefusion_rr_nsv.yaml`
  - `ecg_thesis/configs/phase6_daeac_fcba_latefusion_rr_nsv_mitbih_sqrtw.yaml`
  - `ecg_thesis/configs/phase6_daeac_fcba_latefusion_rr_nsv_mitbih_os.yaml`
- Dual-head reliable NSV FCBA late-fusion configs:
  - `ecg_thesis/configs/phase6_daeac_fcba_latefusion_rr_nsv_dhrel_ds1_ds2.yaml`
  - `ecg_thesis/configs/phase6_daeac_fcba_latefusion_rr_nsv_dhrel_ds1_incart.yaml`
  - `ecg_thesis/configs/phase6_daeac_fcba_latefusion_rr_nsv_dhrel_ds1_svdb.yaml`
  - `ecg_thesis/configs/phase6_daeac_fcba_latefusion_rr_nsv_dhrel_mitbih_incart.yaml`
  - `ecg_thesis/configs/phase6_daeac_fcba_latefusion_rr_nsv_dhrel_mitbih_svdb.yaml`
- Focal variants:
  - `ecg_thesis/configs/phase6_daeac_paper_phase3_focal.yaml`
  - `ecg_thesis/configs/phase6_daeac_paper_base_focal_standard.yaml`
  - `ecg_thesis/configs/phase6_daeac_paper_base_focal_alpha.yaml`

## 9. Training constraints

- Source pretrain epochs: 50.
- RTD paper baseline adaptation epochs: 300 max in `phase6_daeac_paper.yaml`.
- Active NSV FCBA late-fusion all-domain configs use 50 adaptation epochs.
- Batch size: source/adaptation/eval batch size 256.
- Optimizer/LR: Adam, LR `0.005`, weight decay `0.0001`, StepLR gamma `0.99` every 200 steps.
- Full runs are expected on Kaggle, not locally.
- Codex should not run full training.
- Codex should use smoke tests only.
- Keep Kaggle paths configurable.

## 10. Metrics

- Main metrics: accuracy, macro-F1, per-class precision/recall/F1, paper metrics `Se`, `Pp`, `F1`, confusion matrix, V-Measure for checkpointing.
- Priority class: `F` appears most critical/weakest in paper targets; also monitor minority classes `S`, `V`, `F`.
- Error types to track: per-class false negatives and false positives from confusion matrix, especially missed `F`, `S`, `V` beats and confusion among `S/V/F` versus `N`.

## 11. Immediate implementation context

- The repo is past Phase 0 and has working Phase 1A, Phase 1B, Phase 2A, Phase 2B code/tests.
- The baseline all-domain Kaggle notebook is a 3-class NSV FCBA + RR late-fusion DAEAC branch with RTD flags disabled.
- The `dhrel` all-domain Kaggle notebook enables dual-head reliable pseudo-labeling for the same branch.
- The current safest next implementation target is one phase at a time after reliable pseudo-labeling.
- Do not re-neutralize RR rows in any new code.
- Preserve current behavior when new RTD flags are disabled.
- Do not expose target labels during adaptation.
- Do not implement optional ablations unless explicitly requested.

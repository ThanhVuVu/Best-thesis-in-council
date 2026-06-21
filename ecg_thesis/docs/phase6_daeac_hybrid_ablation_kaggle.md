# Phase 6 Hybrid MKMMD+MCC Ablations On Kaggle

## Common Protocol

All four runs start from `daeac_base_focal_standard_best.pt` and use the same
focal-standard Hybrid MKMMD+MCC baseline. MIT-BIH DS1 is split by record for
source fit/validation. Adaptation sees only the first five minutes of each DS2
record. Primary DS2 evaluation uses only beats at or after minute five.

The best checkpoint is selected by source validation Macro-F1. DS2-after5,
INCART, and SVDB labels are used only by the evaluation script.

## Version Definitions

| Notebook | Only mechanism changed |
|---|---|
| `phase6_daeac_hybrid_cb_mkmmd_kaggle.ipynb` | Replaces global MKMMD with soft class-conditional MKMMD. N/S/V weights are 1 and F weight is 2. |
| `phase6_daeac_hybrid_faware_pseudo_kaggle.ipynb` | Replaces threshold-only pseudo selection with confidence-and-margin-safe top-k quotas. |
| `phase6_daeac_hybrid_minority_mcc_kaggle.ipynb` | Replaces the uniform MCC row average with inverse-soft-prior row weights and an extra F multiplier. |
| `phase6_daeac_hybrid_source_f_prototype_kaggle.ipynb` | Adds source-only cosine prototype contrastive loss for F against N/V. |

These are independent ablations, not cumulative versions.

## Kaggle Inputs

Attach a processed DAEAC dataset containing:

```text
mitdb_ds1_daeac.npz
mitdb_ds2_first5_unlabeled_daeac.npz
mitdb_ds2_daeac.npz
incart_all_daeac.npz       # optional external evaluation
svdb_all_daeac.npz         # optional external evaluation
```

Attach a Kaggle Model or Dataset containing:

```text
daeac_base_focal_standard_best.pt
```

Raw WFDB databases and the feature-debug raw cache are not required. Enable a
GPU. Internet is required only when cloning the repository or installing a
missing dependency.

## Run One Notebook

1. Edit `REPO_URL` and `BRANCH` in the setup cell.
2. Run setup and input-copy cells.
3. Run the static/protocol cell. It creates
   `mitdb_ds2_after5_daeac.npz` and fails if first5/after5 overlap.
4. Run the one-epoch smoke cell. Do not continue if it fails.
5. Run full adaptation. The default is 30 epochs, batch size 256, seed 42,
   and learning rate 0.0005.
6. Run best-checkpoint evaluation. `--dataset all` means source validation,
   DS2-after5, and any attached INCART/SVDB files.
7. Run the persistence cell. The resulting zip is under `/kaggle/working`.

The important outputs are:

```text
checkpoints/*_best.pt
checkpoints/*_latest.pt
metrics/*_train_log.csv
metrics/*_train_summary.json
metrics/*_best_*_metrics.json
metrics/*_best_*_confusion_matrix.csv
predictions/*_best_*_predictions.csv
diagnostics/source_fit_val_split.json
diagnostics/protocol_validation.json
resolved_config.json
*_report.md
```

## Three-Account Schedule

- Account 1: class-balanced MKMMD.
- Account 2: F-aware pseudo labels.
- Account 3: minority-weighted MCC.
- The account that finishes first then runs source F-prototype contrastive.

Do not change checkpoint, seed, split, epoch count, or common loss weights
between accounts.

## Compare Best Runs

Download or attach the four output folders, then run from `ecg_thesis/`:

```bash
python scripts/phase6_daeac_hybrid_ablation/05_compare.py \
  --input-dirs \
    outputs/daeac_hybrid_cb_mkmmd \
    outputs/daeac_hybrid_faware_pseudo \
    outputs/daeac_hybrid_minority_mcc \
    outputs/daeac_hybrid_source_f_prototype
```

For a fair baseline, evaluate the existing Hybrid focal-standard best
checkpoint on `mitdb_ds2_after5_daeac.npz`, INCART, and SVDB. Results previously
reported on full DS2 are transductive and must not be mixed into the after5
comparison table.

The baseline checkpoint can be evaluated with any of the four configs because
the DAEAC architecture and after5 protocol are identical:

```bash
python scripts/phase6_daeac_hybrid_ablation/03_eval.py \
  --config configs/phase6_daeac_hybrid_cb_mkmmd.yaml \
  --checkpoint /path/to/daeac_hybrid_mkmmd_mcc_focal_standard_best.pt \
  --method-name daeac_hybrid_mkmmd_mcc_focal_standard \
  --dataset all \
  --output-dir outputs/phase6_daeac_hybrid_focal_standard_after5_eval
```

## Diagnostic Interpretation

- Class-balanced MKMMD: inspect per-class/layer MMD, active flags, and target
  soft mass. F cannot contribute when its target soft mass is below 1.
- F-aware pseudo labels: compare F candidate and selected counts together with
  selected confidence and margin. More F samples are useful only if safety
  statistics remain stable.
- Minority MCC: inspect target prior, dynamic class weights, and MCC off-diagonal
  loss by class. Confirm N weight falls without making F weight saturate at 3
  for the entire run.
- Source F prototype: track prototype loss and cosine F-N/F-V. Useful geometry
  should reduce both cosine similarities while preserving source validation
  Macro-F1.

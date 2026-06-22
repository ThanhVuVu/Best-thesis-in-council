# Phase 6 Reliability-Weighted Prototype Bank on Kaggle

## Experiment contract

This workflow implements two independent paper-DAEAC ablations:

| Config | Purpose |
|---|---|
| `phase6_daeac_proto_bank_logging.yaml` | Updates, checkpoints, and logs the bank while legacy `CenterMemory` still drives all losses. |
| `phase6_daeac_proto_bank_weighted.yaml` | Uses reliability-weighted `P_s/P_t/P_g` in the existing alignment, separation, and compactness losses. |

Both variants start from the same source-selected `daeac_base_best.pt`. They
use the same seed, source split, optimizer, confidence thresholds, and loss
weights. The best adaptation checkpoint is selected only by source-validation
Macro-F1.

Adaptation sees DS2 first-five-minute samples without labels. Primary target
evaluation uses only DS2 beats at or after minute five. Target labels never
select thresholds, hyperparameters, variants, or checkpoints.

## Kaggle inputs

Enable a GPU and attach a Dataset containing:

```text
mitdb_ds1_daeac.npz
mitdb_ds2_first5_unlabeled_daeac.npz
mitdb_ds2_daeac.npz
incart_all_daeac.npz       # optional external evaluation
svdb_all_daeac.npz         # optional external evaluation
```

Attach a Dataset or Kaggle Model containing the paper baseline checkpoint:

```text
daeac_base_best.pt
```

The notebook copies inputs into `/kaggle/working/Best-thesis-in-council/ecg_thesis`.
It never writes to `/kaggle/input`.

If the base checkpoint is unavailable, set `TRAIN_BASE_IF_MISSING=True` in the
setup cell. This explicitly runs the existing paper source-training script for
300 epochs. The default is `False`, so a missing checkpoint fails before any
adaptation run.

## Notebook workflow

Open `notebooks/phase6_daeac_proto_bank_kaggle.ipynb` and:

1. Set `REPO_URL` and `BRANCH`, unless the repository is already under
   `/kaggle/working`.
2. Keep `FULL_VARIANTS=['logging_only','weighted_global']` for the complete
   ablation. A single variant may be selected when resuming a failed session.
3. Run setup, dependency, and input-copy cells.
4. Run repo/unit checks, after5 creation, and strict protocol validation.
5. Run both one-epoch smoke tests. Do not continue if either fails.
6. Run full adaptation. Defaults are 300 epochs, batch size 256, seed 42, and
   paper learning rate 0.005.
7. Evaluate each source-selected best checkpoint on source validation,
   DS2-after5, and optional INCART/SVDB.
8. Generate the comparison report and persist the zip under `/kaggle/working`.

W&B is disabled by default. Set `WANDB_ENABLED=True` and configure project,
group, and optional entity in the setup cell. Local checkpoints/JSON/CSV remain
the source of truth even when W&B is enabled.

## Resume

Upload a previous `*_latest.pt` and set its discovered path in the notebook's
`RESUME_CHECKPOINTS` mapping. Resume restores model, optimizer, scheduler,
prototype-bank buffers, best source-validation score, all-N streak, and the
legacy center state needed by `logging_only`.

CLI equivalent:

```bash
python scripts/phase6_daeac_proto_bank/02_train.py \
  --config configs/phase6_daeac_proto_bank_weighted.yaml \
  --resume-checkpoint /path/to/daeac_proto_bank_weighted_latest.pt
```

## Important outputs

Each output directory contains:

```text
checkpoints/*_best.pt
checkpoints/*_latest.pt
metrics/*_train_log.csv
metrics/*_train_summary.json
metrics/*_best_*_metrics.json
metrics/*_best_*_confusion_matrix.csv
predictions/*_best_*_predictions.csv
diagnostics/after5_prepare.json
diagnostics/protocol_validation.json
diagnostics/source_fit_val_split.json
resolved_config.json
```

The weighted output directory also receives:

```text
prototype_bank_comparison.json
prototype_bank_comparison.md
```

## Diagnostics

- `R_t_k` is an epoch EMA of accepted coverage multiplied by accepted mean
  confidence. It does not use target labels.
- `beta_k` must remain in `[0, 0.30]` and remains zero until source and target
  prototypes for class `k` are valid.
- A high target-update skip count for S/F means the confidence-filtered batch
  did not meet `min_target_count=4`; the source anchor remains active.
- `ps_pt_l2` measures domain prototype distance. `pg_ps_l2` measures how much
  the global prototype moved away from its trusted source anchor.
- `near_all_n` is a warning. Exact all-N for two consecutive epochs stops the
  run and leaves the latest diagnostic checkpoint/log in place.

Do not increase `beta_max`, lower confidence thresholds, or choose a variant
because of DS2-after5/INCART/SVDB results. Such changes require a predeclared
source-only or label-free selection rule in a later plan.

## Local verification before a long run

```bash
cd ecg_thesis
python scripts/check_repo.py
python -m unittest tests.test_daeac_prototype_bank
python -m unittest tests.test_daeac_prototype_bank_training
python -m unittest tests.test_daeac_losses tests.test_daeac_hybrid_ablation
```

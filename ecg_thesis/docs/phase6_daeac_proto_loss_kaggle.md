# Phase 6 PLAN 3 — Prototype Loss Replacement on Kaggle

## Experiment contract

This workflow replaces the legacy mixed-center losses while keeping the Plan 1
reliability-weighted prototype bank, Plan 2 class-specific confidence/entropy
filter, and epoch-copy auxiliary classifier fixed. It runs eight preregistered
variants:

| Variant | Active prototype loss |
|---|---|
| `legacy_control` | Existing align + center separation + mixed compactness |
| `align_only` | Directed target-batch prototype → detached source prototype |
| `comp_source_only` | Source feature → detached source prototype |
| `comp_target_only` | Weighted accepted target feature → detached global prototype |
| `sep_uniform_only` | Sample-level margin with uniform margin 5.0 |
| `sep_pair_only` | Sample-level margin with S–N/F–N margin 7.5 |
| `full_uniform` | All replacement losses, uniform margin |
| `full_pair` | All replacement losses, pair-specific margin |

Every run starts from the same source-selected `daeac_base_best.pt`. Adaptation
uses DS2 first-five-minute inputs through `DAEACTargetUnlabeledDataset`, which
does not return labels. Best checkpoints are selected only by source-validation
Macro-F1. DS2-after5, INCART, and SVDB labels are post-training descriptions and
must never select thresholds, weights, margins, epochs, or variants.

## Kaggle inputs

Enable a GPU and attach inputs containing exactly one copy of:

```text
mitdb_ds1_daeac.npz
mitdb_ds2_first5_unlabeled_daeac.npz
mitdb_ds2_daeac.npz
daeac_base_best.pt
```

The following external evaluation datasets are optional:

```text
incart_all_daeac.npz
svdb_all_daeac.npz
```

Open `notebooks/phase6_daeac_proto_loss_kaggle.ipynb`, set `REPO_URL` and
`BRANCH`, and run cells in order. Inputs are discovered recursively under
`/kaggle/input` and copied beneath `/kaggle/working`; the notebook never writes
to an attached Dataset. If discovery finds duplicates, set `DATA_INPUT_DIR`
and/or `BASE_CHECKPOINT_INPUT` to exact mounted paths.

The notebook does not silently train a missing base checkpoint. A missing or
ambiguous checkpoint stops before adaptation.

## Checks and smoke runs

Before a long run, the notebook:

1. Runs `scripts/check_repo.py` and DAEAC unit tests.
2. Creates `mitdb_ds2_after5_daeac.npz` from full DS2.
3. Proves first5 and after5 are disjoint.
4. Validates all eight resolved configs and the unlabeled target-loader contract.
5. Runs isolated one-epoch smoke tests for all variants under
   `/kaggle/working/smoke`.

Smoke checkpoints are debugging artifacts, not experiment evidence. Only set
`RUN_FULL=True` after every check passes. A complete run is eight independent
300-epoch adaptations, so it may require multiple Kaggle sessions; run subsets
by temporarily narrowing `VARIANTS` without changing their configs.

## Resume and W&B

Upload prior `*_latest.pt` files as a Kaggle Dataset and set, for example:

```python
RESUME_CHECKPOINTS = {
    "full_pair": "/kaggle/input/<dataset>/daeac_proto_loss_full_pair_latest.pt",
}
```

Resume restores model, optimizer, scheduler, prototype-bank buffers, history,
best source-validation score, and pseudo-label safety streaks. Do not resume a
different variant from that checkpoint.

W&B is disabled by default. Set `ENABLE_WANDB=True`; provide authentication via
Kaggle Secrets and never place an API key in the notebook. Local JSON, CSV, and
checkpoint files remain the source of truth.

## CLI equivalents

Run from `ecg_thesis/`:

```bash
python scripts/check_repo.py
python -m unittest discover -s tests -p "test_daeac_prototype*.py"

python scripts/phase6_daeac_proto_loss/00_prepare_after5.py \
  --config configs/phase6_daeac_proto_loss_legacy_control.yaml

python scripts/phase6_daeac_proto_loss/01_validate.py \
  --config configs/phase6_daeac_proto_loss_full_pair.yaml --strict

python scripts/phase6_daeac_proto_loss/02_train.py \
  --config configs/phase6_daeac_proto_loss_full_pair.yaml

python scripts/phase6_daeac_proto_loss/03_eval.py \
  --config configs/phase6_daeac_proto_loss_full_pair.yaml \
  --checkpoint outputs/phase6_daeac_proto_loss_full_pair/checkpoints/daeac_proto_loss_full_pair_best.pt \
  --dataset all

python scripts/phase6_daeac_proto_loss/04_make_report.py \
  --config configs/phase6_daeac_proto_loss_full_pair.yaml
```

Resume uses:

```bash
python scripts/phase6_daeac_proto_loss/02_train.py \
  --config configs/phase6_daeac_proto_loss_full_pair.yaml \
  --resume-checkpoint /path/to/daeac_proto_loss_full_pair_latest.pt
```

## Outputs and interpretation

Each variant writes its unique directory under
`outputs/phase6_daeac_proto_loss_<variant>/`:

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
resolved_config.json
```

The final notebook cells create:

```text
/kaggle/working/phase6_daeac_proto_loss_report/
/kaggle/working/phase6_daeac_proto_loss_bundle.zip
```

Inspect raw/weighted losses, ramps, accepted target counts, target weights,
active alignment classes, prototype validity, and margin-violation ratios before
interpreting performance. If S has no valid target prototype or `beta_S=0`, the
global S prototype remains the source anchor and S alignment is skipped. Empty
or all-N pseudo-label distributions retain the existing safety stop.

Fallback decisions may use source-validation behavior and label-free target
diagnostics only. Target evaluation columns must not be used to lower filtering
thresholds, change margins/lambdas, select checkpoints, or choose the winning
variant.

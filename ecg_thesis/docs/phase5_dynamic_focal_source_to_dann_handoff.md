# Phase 5 Dynamic Focal Source-Only -> DANN Handoff

## Objective

Run a clean Phase 5 ablation where dynamic weighted focal loss is used at the
MACNN source-only training stage first, then the resulting source-only checkpoint
is used to initialize MACNN DANN.

This is different from the already-tested run:

```text
existing source-only weighted-CE checkpoint -> DANN with dynamic_focal source loss
```

That run did not clearly improve. The next intended experiment is:

```text
MIT-BIH source-only MACNN trained from scratch with dynamic_focal
-> checkpoint macnn_se_source_only_dynamic_focal_lr001_best.pt
-> MACNN DANN initialized from that checkpoint
-> eval on MIT-BIH test and INCART after5 heldout
```

## Why This Matters

Dynamic focal inside DANN alone has little room to help because the DANN run is
initialized from an already-trained source-only checkpoint. In the observed run,
classification loss was already tiny from epoch 1, so imbalance handling did not
substantially reshape the source classifier.

To test the MBE 2024 dynamic weighted focal loss idea fairly, the source-only
classifier itself must be trained with `source_loss=dynamic_focal`.

## Current Code Support

The code already supports this:

- `src/training/train.py`
  - `DynamicWeightedFocalLoss`
- `src/training/train_macnn.py`
  - MACNN source-only supports `source_loss: dynamic_focal`
  - MACNN DAEAC also supports it, but this experiment does not need DAEAC
- `src/training/train_dann.py`
  - DANN source classifier loss supports `source_loss: dynamic_focal`
- `scripts/phase5_macnn/02_train_macnn_source_only.py`
  - CLI flags: `--source-loss dynamic_focal`, `--focal-gamma`
- `scripts/phase5_macnn/04_train_macnn_dann.py`
  - CLI flags: `--source-loss`, `--checkpoint-prefix`
- `scripts/phase5_macnn/05_eval_macnn_dann.py`
  - CLI flag: `--method-name`

## Required Data

Use the original Phase 5 MACNN files:

```text
data/processed/phase5_macnn/mitbih_train_macnn.npz
data/processed/phase5_macnn/mitbih_test_macnn.npz
data/processed/phase5_macnn/incart_first5_unlabeled_macnn.npz
data/processed/phase5_macnn/incart_after5_heldout_macnn.npz
```

Do not regenerate CatNet data. Do not rerun raw WFDB preprocessing unless these
four files are missing.

## Step 1: Train Source-Only With Dynamic Focal

Run from `ecg_thesis/`:

```bash
python scripts/phase5_macnn/02_train_macnn_source_only.py \
  --config configs/phase5_macnn_daeac.yaml \
  --epochs 100 \
  --lr 0.001 \
  --source-loss dynamic_focal \
  --focal-gamma 2.0 \
  --checkpoint-prefix macnn_se_source_only_dynamic_focal_lr001
```

Expected checkpoint:

```text
outputs/checkpoints/macnn_se_source_only_dynamic_focal_lr001_best.pt
```

Eval source-only checkpoint:

```bash
python scripts/phase5_macnn/03_eval_macnn.py \
  --config configs/phase5_macnn_daeac.yaml \
  --checkpoint outputs/checkpoints/macnn_se_source_only_dynamic_focal_lr001_best.pt \
  --method-name macnn_se_source_only_dynamic_focal_lr001 \
  --dataset both
```

Important: compare source-only dynamic focal against the existing source-only
weighted-CE checkpoint. Focus on `macro_f1`, `S` precision/recall/F1, and the
confusion matrix.

## Step 2: Make A Runtime Config For DANN Init

For notebook or Colab/Kaggle use, create a copied config where:

```yaml
dann:
  source_init_checkpoint: outputs/checkpoints/macnn_se_source_only_dynamic_focal_lr001_best.pt

training:
  checkpoint_prefix: macnn_se_dann_from_dynamic_focal_source
  source_loss: weighted_ce
```

Recommended first DANN run uses `source_loss=weighted_ce`, because the source
classifier has already been trained with dynamic focal. This isolates the
question: does a dynamic-focal source checkpoint improve DANN?

Optional second DANN run:

```text
source dynamic_focal checkpoint -> DANN with source_loss dynamic_focal
```

Only run this if the first DANN run is promising.

## Step 3: Train DANN From Dynamic-Focal Source Checkpoint

If using a copied config with the DANN init checkpoint set:

```bash
python scripts/phase5_macnn/04_train_macnn_dann.py \
  --config configs/phase5_macnn_dynamic_focal_source_dann.yaml \
  --epochs 100 \
  --source-loss weighted_ce \
  --checkpoint-prefix macnn_se_dann_from_dynamic_focal_source
```

If not using a copied config, update `config["dann"]["source_init_checkpoint"]`
inside the notebook before calling the DANN script.

Eval:

```bash
python scripts/phase5_macnn/05_eval_macnn_dann.py \
  --config configs/phase5_macnn_dynamic_focal_source_dann.yaml \
  --checkpoint outputs/checkpoints/macnn_se_dann_from_dynamic_focal_source_best.pt \
  --method-name macnn_se_dann_from_dynamic_focal_source \
  --dataset both
```

Also eval latest if the curve is noisy:

```bash
python scripts/phase5_macnn/05_eval_macnn_dann.py \
  --config configs/phase5_macnn_dynamic_focal_source_dann.yaml \
  --checkpoint outputs/checkpoints/macnn_se_dann_from_dynamic_focal_source_latest.pt \
  --method-name macnn_se_dann_from_dynamic_focal_source_latest \
  --dataset both
```

## W&B Tracking

All relevant scripts support W&B flags. Example:

```bash
--wandb \
--wandb-project ecg-thesis \
--wandb-group phase5_macnn_dynamic_focal_source \
--wandb-run-name macnn_se_source_only_dynamic_focal_lr001_train \
--wandb-mode online \
--wandb-tags phase5 macnn dynamic_focal source_only
```

Use distinct run names for:

```text
macnn_se_source_only_dynamic_focal_lr001_train
macnn_se_source_only_dynamic_focal_lr001_eval
macnn_se_dann_from_dynamic_focal_source_train
macnn_se_dann_from_dynamic_focal_source_eval
```

## Decision Criteria

Keep the ablation only if it improves the target-domain result meaningfully over
the current MACNN DANN baseline.

Known current strong baseline:

```text
macnn_se_dann on INCART after5:
accuracy ~= 0.9711
macro_f1 ~= 0.7881
S F1 ~= 0.4796
V F1 ~= 0.8954
```

Main success criteria:

```text
INCART after5 macro_f1 improves
S F1 improves without collapsing S precision
V F1 remains strong
MIT-BIH source test does not degrade severely
```

Do not judge by accuracy alone.

## Things Not To Do

- Do not modify the external DAEAC reference repo.
- Do not rerun CatNet.
- Do not use target held-out labels during training/adaptation.
- Do not treat `incart_after5_heldout` as unlabeled adaptation data.
- Do not overwrite the existing source-only or DANN checkpoints; use the prefixes
  above.

## 2026-06-02 Continuation Status

- Added runtime config:
  `configs/phase5_macnn_dynamic_focal_source_dann.yaml`.
- Updated Kaggle notebook section
  `RUN_DYNAMIC_FOCAL_SOURCE_TO_DANN` to run this ablation and evaluate both
  DANN best/latest checkpoints.
- Local smoke test passed for dynamic focal source-only script with 16 fit and
  16 validation samples.
- Full 100-epoch training was not run locally because this workspace PyTorch is
  CPU-only. Run the notebook section on Kaggle/Colab GPU for the real result.

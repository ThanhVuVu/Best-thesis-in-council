# ECG Thesis Experiments

Beat-level ECG arrhythmia experiments for MIT-BIH to INCART transfer.

Phase 1 trains a source-only baseline on MIT-BIH and evaluates both in-domain on MIT-BIH and cross-domain on INCART using simplified AAMI-like `N/S/V` classes.

Phase 2 adds a CATNet1D source-only baseline and DANN domain adaptation with unlabeled INCART target beats.

Raw PhysioNet data, processed `.npz` files, checkpoints, logs, figures, and predictions are intentionally ignored by git.

## Layout

```text
ecg_thesis/
  configs/
    phase1.yaml
    phase2_dann.yaml
  data/
    processed/          # local only: *.npz
  outputs/              # local only: checkpoints, metrics, figures, logs
  scripts/
    phase1/
    phase2/
    common.py
    scripts_eval_common.py
  src/
    data/
    models/
    training/
    utils/
    visualization/
```

## Phase 1

```bash
cd ecg_thesis
python scripts/phase1/01_eda_raw_data.py --config configs/phase1.yaml
python scripts/phase1/00_prepare_data.py --config configs/phase1.yaml
python scripts/phase1/02_validate_processed_data.py --config configs/phase1.yaml
python scripts/phase1/03_train_source_only.py --config configs/phase1.yaml
python scripts/phase1/04_eval_in_domain.py --config configs/phase1.yaml --checkpoint outputs/checkpoints/best.pt
python scripts/phase1/05_eval_cross_domain.py --config configs/phase1.yaml --checkpoint outputs/checkpoints/best.pt
python scripts/phase1/06_visualize_embeddings.py --config configs/phase1.yaml --checkpoint outputs/checkpoints/best.pt
python scripts/phase1/07_make_phase1_report.py --config configs/phase1.yaml
```

## Checkpoint Safety

Training writes:

```text
outputs/checkpoints/latest.pt
outputs/checkpoints/best.pt
```

To mirror checkpoints to a persistent location, set an environment variable before training:

```bash
export ECG_PHASE1_CHECKPOINT_BACKUP_DIR=/content/drive/MyDrive/thesis-runs/ecg_thesis/checkpoints
python scripts/phase1/03_train_source_only.py --config configs/phase1.yaml
```

On Kaggle, use a writable output folder and save the notebook version after training:

```bash
export ECG_PHASE1_CHECKPOINT_BACKUP_DIR=/kaggle/working/ecg_thesis_checkpoint_backup
python scripts/phase1/03_train_source_only.py --config configs/phase1.yaml
```

## Phase 2

Prepare the target split, train the CATNet1D source-only baseline, then train and evaluate DANN:

```bash
python scripts/phase2/08_split_incart_unlabeled_test.py --config configs/phase2_dann.yaml
python scripts/phase2/09_train_source_only_catnet.py --config configs/phase2_dann.yaml
python scripts/phase2/10_eval_source_only_catnet.py --config configs/phase2_dann.yaml --checkpoint outputs/checkpoints/source_only_catnet_best.pt
python scripts/phase2/11_train_dann.py --config configs/phase2_dann.yaml
python scripts/phase2/12_eval_dann_in_domain.py --config configs/phase2_dann.yaml --checkpoint outputs/checkpoints/dann_best.pt
python scripts/phase2/13_eval_dann_cross_domain.py --config configs/phase2_dann.yaml --checkpoint outputs/checkpoints/dann_best.pt
python scripts/phase2/14_visualize_phase2_embeddings.py --config configs/phase2_dann.yaml --source-checkpoint outputs/checkpoints/source_only_catnet_best.pt --dann-checkpoint outputs/checkpoints/dann_best.pt
python scripts/phase2/15_make_phase2_report.py --config configs/phase2_dann.yaml
```

For smoke tests:

```bash
python scripts/phase2/09_train_source_only_catnet.py --config configs/phase2_dann.yaml --epochs 1 --max-fit-samples 256 --max-val-samples 256
python scripts/phase2/11_train_dann.py --config configs/phase2_dann.yaml --epochs 1 --max-source-samples 256 --max-target-samples 256 --max-val-samples 256
```

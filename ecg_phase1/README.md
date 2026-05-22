# ECG Phase 1

Source-only beat-level ECG arrhythmia baseline.

The pipeline trains on MIT-BIH and evaluates both in-domain on MIT-BIH and cross-domain on INCART using simplified AAMI-like `N/S/V` classes.

## Quick Start

```bash
cd ecg_phase1
python scripts/01_eda_raw_data.py --config configs/phase1.yaml
python scripts/00_prepare_data.py --config configs/phase1.yaml
python scripts/02_validate_processed_data.py --config configs/phase1.yaml
python scripts/03_train_source_only.py --config configs/phase1.yaml
python scripts/04_eval_in_domain.py --config configs/phase1.yaml --checkpoint outputs/checkpoints/best.pt
python scripts/05_eval_cross_domain.py --config configs/phase1.yaml --checkpoint outputs/checkpoints/best.pt
python scripts/06_visualize_embeddings.py --config configs/phase1.yaml --checkpoint outputs/checkpoints/best.pt
python scripts/07_make_phase1_report.py --config configs/phase1.yaml
```

Raw PhysioNet data and processed `.npz` files are intentionally ignored by git.

## Checkpoint Safety

Training always writes:

```text
outputs/checkpoints/latest.pt
outputs/checkpoints/best.pt
```

To mirror checkpoints to a persistent location, set an environment variable before training:

```bash
export ECG_PHASE1_CHECKPOINT_BACKUP_DIR=/content/drive/MyDrive/thesis-runs/ecg_phase1/checkpoints
python scripts/03_train_source_only.py --config configs/phase1.yaml
```

On Kaggle, use a writable output folder and save the notebook version after training:

```bash
export ECG_PHASE1_CHECKPOINT_BACKUP_DIR=/kaggle/working/ecg_phase1_checkpoint_backup
python scripts/03_train_source_only.py --config configs/phase1.yaml
```

# ECG Thesis Experiments

Beat-level ECG arrhythmia experiments for MIT-BIH to INCART transfer.

Phase 1 trains a source-only baseline on MIT-BIH and evaluates both in-domain on MIT-BIH and cross-domain on INCART using simplified AAMI-like `N/S/V` classes.

Phase 2 adds a CATNet1D source-only baseline and DANN domain adaptation with unlabeled INCART target beats.

Phase 3 adds RR/rhythm features on top of CATNet1D and DANN.

Phase 4 tests ECG-FM variants, including 1-lead to 12-lead bridge ablations and source-free adaptation ideas.

Phase 5 adds a DAEAC-style MACNN_SE protocol with INCART first-5-minute
unlabeled adaptation, plus a fair CATNet1D+DANN rerun on the same split.

Raw PhysioNet data, processed `.npz` files, checkpoints, logs, figures, and predictions are intentionally ignored by git.

For the canonical scientific order and status labels, read:

```text
docs/scientific_repo_order.md
```

Current thesis-safe result hierarchy:

```text
Accepted: Phase 1 and Phase 2.
Best accepted result: Phase 2 CATNet1D + DANN.
Full-run pending: Phase 3 RR-aware CATNet1D + DANN.
Advanced/scaffolded: Phase 4 ECG-FM and Phase 5 MACNN branches.
```

## Layout

```text
ecg_thesis/
  configs/
    phase1.yaml
    phase2_dann.yaml
    phase3_rr_dann.yaml
    phase4*.yaml
  data/
    processed/          # local only: *.npz
  docs/                 # plans, handoff notes, documentation index
  notebooks/            # local/ignored Colab and Kaggle notebooks
  outputs/              # local only: checkpoints, metrics, figures, logs
  scripts/
    phase1/
    phase2/
    phase3/
    phase4a/
    phase4b/
    phase4c/
    phase5_macnn/
    common.py
    scripts_eval_common.py
  src/
    data/
    models/
    training/
    utils/
    visualization/
```

## Repo Health Check

Run this after cleanup or before a long Colab/Kaggle run:

```bash
cd ecg_thesis
python scripts/check_repo.py
```

The check is intentionally lightweight: it parses Python files and loads YAML
configs without requiring datasets, checkpoints, GPU, or ECG-FM dependencies.

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

## Phase 3

RR-aware CATNet1D experiments:

```bash
python scripts/phase3/02_prepare_rr_features.py --config configs/phase3_rr_dann.yaml
python scripts/phase3/03_train_source_only_catnet_rr.py --config configs/phase3_rr_dann.yaml
python scripts/phase3/04_eval_source_only_catnet_rr.py --config configs/phase3_rr_dann.yaml --checkpoint outputs/checkpoints/source_only_catnet_rr_best.pt --dataset both
python scripts/phase3/05_train_dann_rr.py --config configs/phase3_rr_dann.yaml
python scripts/phase3/06_eval_dann_rr_in_domain.py --config configs/phase3_rr_dann.yaml --checkpoint outputs/checkpoints/dann_rr_best.pt
python scripts/phase3/07_eval_dann_rr_cross_domain.py --config configs/phase3_rr_dann.yaml --checkpoint outputs/checkpoints/dann_rr_best.pt
python scripts/phase3/09_make_phase3_report.py --config configs/phase3_rr_dann.yaml
```

## Phase 4A

ECG-FM bridge ablations use 5-second windows and an external ECG-FM checkpoint:

```bash
python scripts/phase4a/01_prepare_5s_windows.py --config configs/phase4a_ecgfm_leadbridge.yaml
python scripts/phase4a/08_train_source_ecgfm_leadbridge_weightedlr.py --config configs/phase4a_ecgfm_leadbridge_weightedlr.yaml
python scripts/phase4a/09_eval_source_ecgfm_leadbridge_weightedlr.py --config configs/phase4a_ecgfm_leadbridge_weightedlr.yaml --checkpoint outputs/checkpoints/source_only_ecgfm_leadbridge_weightedlr_best.pt --dataset both
python scripts/phase4a/10_train_source_ecgfm_repeatinitbridge.py --config configs/phase4a_ecgfm_repeatinitbridge.yaml
python scripts/phase4a/11_eval_source_ecgfm_repeatinitbridge.py --config configs/phase4a_ecgfm_repeatinitbridge.yaml --checkpoint outputs/checkpoints/source_only_ecgfm_repeatinitbridge_best.pt --dataset both
python scripts/phase4a/06_train_source_ecgfm_repeatbridge.py --config configs/phase4a_ecgfm_repeatbridge.yaml
python scripts/phase4a/07_eval_source_ecgfm_repeatbridge.py --config configs/phase4a_ecgfm_repeatbridge.yaml --checkpoint outputs/checkpoints/source_only_ecgfm_repeatbridge_best.pt --dataset both
```

## Phase 5

MACNN_SE + DAEAC-style first-5-minute INCART adaptation:

Kaggle notebook: `notebooks/phase5_macnn_kaggle.ipynb`.

```bash
python scripts/phase5_macnn/10_check_phase5_static.py --config configs/phase5_macnn_daeac.yaml
python scripts/phase5_macnn/01_prepare_macnn_first5_data.py --config configs/phase5_macnn_daeac.yaml
python scripts/phase5_macnn/02_train_macnn_source_only.py --config configs/phase5_macnn_daeac.yaml
python scripts/phase5_macnn/03_eval_macnn.py --config configs/phase5_macnn_daeac.yaml --checkpoint outputs/checkpoints/macnn_se_source_only_best.pt --method-name macnn_se_source_only --dataset both
python scripts/phase5_macnn/04_train_macnn_dann.py --config configs/phase5_macnn_daeac.yaml
python scripts/phase5_macnn/05_eval_macnn_dann.py --config configs/phase5_macnn_daeac.yaml --checkpoint outputs/checkpoints/macnn_se_dann_best.pt --dataset both
python scripts/phase5_macnn/06_train_macnn_daeac.py --config configs/phase5_macnn_daeac.yaml
python scripts/phase5_macnn/03_eval_macnn.py --config configs/phase5_macnn_daeac.yaml --checkpoint outputs/checkpoints/macnn_se_daeac_best.pt --method-name macnn_se_daeac --dataset both
python scripts/phase5_macnn/07_train_catnet_dann_first5.py --config configs/phase5_macnn_daeac.yaml
python scripts/phase5_macnn/08_eval_catnet_dann_first5.py --config configs/phase5_macnn_daeac.yaml --checkpoint outputs/checkpoints/catnet_first5_dann_best.pt --dataset both
python scripts/phase5_macnn/11_visualize_phase5_embeddings.py --config configs/phase5_macnn_daeac.yaml --checkpoint outputs/checkpoints/macnn_se_daeac_best.pt --kind macnn --method-name macnn_se_daeac
python scripts/phase5_macnn/09_make_phase5_report.py --config configs/phase5_macnn_daeac.yaml
```

Smoke flags:

```bash
python scripts/phase5_macnn/02_train_macnn_source_only.py --config configs/phase5_macnn_daeac.yaml --epochs 1 --max-fit-samples 32 --max-val-samples 32
python scripts/phase5_macnn/04_train_macnn_dann.py --config configs/phase5_macnn_daeac.yaml --epochs 1 --max-source-samples 32 --max-target-samples 32 --max-val-samples 32
python scripts/phase5_macnn/06_train_macnn_daeac.py --config configs/phase5_macnn_daeac.yaml --epochs 1 --max-source-samples 32 --max-target-samples 32 --max-val-samples 32
```

## Clean Workspace

Generated data, checkpoints, metrics, notebook state, W&B runs, and Python cache
files are ignored by git. To remove cache folders safely on Windows:

```powershell
Get-ChildItem -Path ecg_thesis -Recurse -Directory -Filter __pycache__ |
  Remove-Item -Recurse -Force
```

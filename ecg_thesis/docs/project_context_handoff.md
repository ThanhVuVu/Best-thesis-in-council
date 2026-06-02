# ECG Thesis Project Context Handoff

## Project Summary

This project is about beat-level ECG arrhythmia classification and cross-domain adaptation.

Main transfer setting:

```text
Source domain: MIT-BIH Arrhythmia Database
Target domain: St Petersburg INCART 12-lead Arrhythmia Database
Task: 3-class beat classification, N/S/V
```

The repo currently uses:

```text
ecg_thesis/
  configs/
  data/processed/      # ignored by git, contains *.npz
  docs/
  outputs/             # ignored by git, checkpoints/metrics/figures
  scripts/
    phase1/
    phase2/
  src/
    data/
    models/
    training/
    utils/
    visualization/
```

Important note:

```text
Data and outputs are ignored by git. Kaggle/Colab must copy processed .npz files into ecg_thesis/data/processed.
```

Latest Phase 5 continuation note:

```text
For the next dynamic focal experiment, read:
docs/phase5_dynamic_focal_source_to_dann_handoff.md

Goal: train MACNN source-only from scratch with source_loss=dynamic_focal,
then initialize MACNN DANN from that new source-only checkpoint.
```

---

## Data Pipeline

The processed `.npz` files are the working data format:

```text
data/processed/mitbih_train.npz
data/processed/mitbih_test.npz
data/processed/incart_test.npz
data/processed/incart_unlabeled.npz
data/processed/incart_test_heldout.npz
```

Each `.npz` contains at least:

```python
{
    "x": np.ndarray,             # [N, 1, 250]
    "y": np.ndarray,             # [N]
    "record": np.ndarray,
    "symbol": np.ndarray,
    "sample": np.ndarray,
    "fs": np.ndarray,
    "domain": np.ndarray,
    "lead_index": np.ndarray,
    "lead_name": np.ndarray,
    "class_names": np.ndarray,
    "config_json": np.ndarray,
}
```

Current preprocessing:

```text
Beat-centered, annotation/R-peak based
Single lead: MIT-BIH MLII, INCART II
Input shape: [N, 1, 250]
Class mapping: N/S/V
No wavelet denoising in the main pipeline
No SMOTE/SMOTE-Tomek in the main pipeline
```

Why no wavelet/SMOTE in the main pipeline:

```text
The main experiments isolate model/backbone/domain-adaptation effects.
Wavelet and SMOTE-Tomek are reserved for later ablations because they change the data distribution and can confound interpretation.
```

---

## Phase 1

Phase 1 implemented a source-only baseline:

```text
Backbone: ResNet1D
Training: MIT-BIH source train
Evaluation:
  - MIT-BIH test
  - INCART test
```

Key Phase 1 result:

```text
MIT-BIH test Macro-F1: 0.6070
INCART test Macro-F1: 0.5473
Cross-domain drop: 0.0597
```

Per-class F1:

```text
MIT-BIH N F1: 0.9056
MIT-BIH S F1: 0.2202
MIT-BIH V F1: 0.6953

INCART N F1: 0.8938
INCART S F1: 0.1179
INCART V F1: 0.6303
```

Interpretation:

```text
N is easy.
V is moderately learnable.
S is the main failure mode.
S morphology is close to N and often needs rhythm/context information.
```

---

## Phase 2

The initial idea was InceptionTime1D, then CNN-LSTM, but after inspecting newer notebooks the chosen Phase 2 baseline became:

```text
CATNet1D + DANN
```

CATNet1D is inspired by these notebooks:

```text
Arrhythmia_Classification_Code_28_(7)_60_epochs_loss_MIT_BIH_SMOTETomek.ipynb
AC_Code_25_v2_add_batch_Norm_and_logdir_INCART_SMOTETomek.ipynb
```

But we do not copy the notebooks directly because:

```text
Their ChannelAttention implementation has broken indentation and reports 0 parameters.
Their inner positional_encoding returns inside the first loop.
Their SMOTE-Tomek data is loaded from prebuilt .pkl files.
Their split/protocol is not the same as our cross-domain record-wise evaluation.
```

Implemented clean PyTorch version:

```text
src/models/catnet1d.py
```

CATNet1D architecture:

```text
Input [B, 1, 250]

Conv1D + BatchNorm + ReLU + ChannelAttention1D + Pool
Conv1D + BatchNorm + ReLU + ChannelAttention1D + Pool
Conv1D + BatchNorm + ReLU + ChannelAttention1D + Pool
Conv1D + BatchNorm + ReLU + ChannelAttention1D

Transpose to [B, T, 128]
Sinusoidal positional encoding
TransformerEncoder
Global average pooling
Dense/ReLU/Dropout embedding
Linear classifier
```

Model API:

```python
logits = model(x)
logits, embedding = model(x, return_embedding=True)
embedding = model.forward_features(x)
model.embedding_dim == 128
model.classifier exists
```

Phase 2 DANN:

```text
Feature extractor: CATNet1D
Label classifier: linear
Domain classifier: GRL -> MLP -> source/target
Target labels are ignored during training
Checkpoint selection uses source validation Macro-F1 only
```

Key files:

```text
configs/phase2_dann.yaml
src/models/catnet1d.py
src/models/dann.py
src/training/train.py
src/training/train_dann.py
scripts/phase2/08_split_incart_unlabeled_test.py
scripts/phase2/09_train_source_only_catnet.py
scripts/phase2/10_eval_source_only_catnet.py
scripts/phase2/11_train_dann.py
scripts/phase2/12_eval_dann_in_domain.py
scripts/phase2/13_eval_dann_cross_domain.py
scripts/phase2/14_visualize_phase2_embeddings.py
scripts/phase2/15_make_phase2_report.py
```

---

## Phase 2 Metrics

INCART split:

```text
Adapt records: I01-I50
Held-out test records: I51-I75
Adapt beats: 120334
Held-out test beats: 55255

INCART held-out class counts:
N: 49045
S: 260
V: 5950
```

Source-only CATNet1D:

```text
MIT-BIH test:
Accuracy: 0.8806
Macro-F1: 0.5679
N-F1: 0.9424
S-F1: 0.1180
V-F1: 0.6433

INCART held-out:
Accuracy: 0.7966
Macro-F1: 0.5202
N-F1: 0.8737
S-F1: 0.1567
V-F1: 0.5302
```

DANN CATNet1D:

```text
MIT-BIH test:
Accuracy: 0.8952
Macro-F1: 0.5856
N-F1: 0.9488
S-F1: 0.0941
V-F1: 0.7138

INCART held-out:
Accuracy: 0.8980
Macro-F1: 0.6783
N-F1: 0.9407
S-F1: 0.4178
V-F1: 0.6763
```

Important deltas on INCART held-out:

```text
DANN vs source-only Macro-F1: +0.1581
DANN vs source-only S-F1:     +0.2611
DANN vs source-only V-F1:     +0.1461
DANN vs source-only N-F1:     +0.0670
```

Conclusion:

```text
Phase 2 is successful.
CATNet1D + DANN substantially improves MIT-BIH -> INCART transfer, especially target-domain S and V.
```

Important caveat:

```text
INCART held-out S support is only 260 beats.
The S improvement is promising but should be validated with additional seeds/splits if time permits.
```

---

## Phase 2 Confusion Matrices

DANN on INCART held-out:

```text
True N: 43931 N, 274 S, 4840 V
True S: 64 N, 155 S, 41 V
True V: 363 N, 53 S, 5534 V
```

Recall:

```text
N recall = 0.8957
S recall = 0.5962
V recall = 0.9301
```

DANN on MIT-BIH test:

```text
True N: 41108 N, 1887 S, 1244 V
True S: 1005 N, 188 S, 644 V
True V: 302 N, 83 S, 2835 V
```

Recall:

```text
N recall = 0.9292
S recall = 0.1023
V recall = 0.8804
```

Interpretation:

```text
DANN is effective on target INCART.
However, S remains poorly separated on MIT-BIH source test.
This suggests S is not consistently represented by global domain alignment alone.
```

Main remaining failure modes:

```text
INCART: N -> V false positives are high.
MIT-BIH: S -> N and S -> V are high.
S still needs rhythm/context information.
```

---

## Figure Notes

Training curves:

```text
DANN source validation Macro-F1 remains stable and peaks around epoch 19.
Domain accuracy decreases toward ~0.51.
Domain loss approaches ~0.69.
This suggests successful domain confusion without full class collapse.
```

UMAP:

```text
DANN appears to mix source/target domains better than source-only.
V remains the clearest class.
S remains scattered and boundary-like rather than a clean cluster.
```

Title bug:

```text
Older generated UMAP figures may still say "ResNet1D embeddings".
This was caused by a hard-coded title in plot_embeddings.py.
It has been fixed so new figures can use:
  - Source-only CATNet1D embeddings
  - DANN CATNet1D embeddings
```

Regenerate figures after pulling latest code:

```bash
python scripts/phase2/14_visualize_phase2_embeddings.py \
  --config configs/phase2_dann.yaml \
  --source-checkpoint outputs/checkpoints/source_only_catnet_best.pt \
  --dann-checkpoint outputs/checkpoints/dann_best.pt
```

---

## Kaggle Commands

After cloning:

```python
%cd /kaggle/working/Best-thesis-in-council/ecg_thesis
!pip install -r requirements.txt
```

Copy `.npz` data:

```python
!mkdir -p data/processed
!find /kaggle/input -name "*.npz"
!cp /kaggle/input/YOUR_DATASET_NAME/*.npz data/processed/
```

Validate:

```python
!python scripts/phase1/02_validate_processed_data.py --config configs/phase1.yaml
```

Create INCART split if needed:

```python
!python scripts/phase2/08_split_incart_unlabeled_test.py --config configs/phase2_dann.yaml
```

Train source-only CATNet:

```python
!python scripts/phase2/09_train_source_only_catnet.py --config configs/phase2_dann.yaml
```

Evaluate source-only CATNet:

```python
!python scripts/phase2/10_eval_source_only_catnet.py \
  --config configs/phase2_dann.yaml \
  --checkpoint outputs/checkpoints/source_only_catnet_best.pt
```

Train DANN:

```python
!python scripts/phase2/11_train_dann.py --config configs/phase2_dann.yaml
```

Evaluate DANN:

```python
!python scripts/phase2/12_eval_dann_in_domain.py \
  --config configs/phase2_dann.yaml \
  --checkpoint outputs/checkpoints/dann_best.pt

!python scripts/phase2/13_eval_dann_cross_domain.py \
  --config configs/phase2_dann.yaml \
  --checkpoint outputs/checkpoints/dann_best.pt
```

Visualize and report:

```python
!python scripts/phase2/14_visualize_phase2_embeddings.py \
  --config configs/phase2_dann.yaml \
  --source-checkpoint outputs/checkpoints/source_only_catnet_best.pt \
  --dann-checkpoint outputs/checkpoints/dann_best.pt

!python scripts/phase2/15_make_phase2_report.py --config configs/phase2_dann.yaml
```

---

## Recommended Phase 3 Direction

Phase 3 should focus on:

```text
Rhythm-aware and class-aware domain adaptation for S-class improvement.
```

Motivation:

```text
Phase 2 DANN improves target S substantially, but S is still unstable and not a clean embedding cluster.
S errors suggest single-beat morphology is insufficient.
```

Recommended order:

1. Failure case analysis.
2. Add RR interval features.
3. Train CATNet + RR source-only.
4. Train CATNet + RR + DANN.
5. Add beat context if RR alone is not enough.
6. Try class-aware DANN or prototype alignment.
7. Calibration/thresholding only after model behavior is stable.

Suggested Phase 3 structure:

```text
scripts/phase3/
  01_analyze_failures.py
  02_prepare_rr_features.py
  03_train_source_only_catnet_rr.py
  04_eval_source_only_catnet_rr.py
  05_train_dann_rr.py
  06_eval_dann_rr_in_domain.py
  07_eval_dann_rr_cross_domain.py
  08_visualize_phase3_embeddings.py
  09_make_phase3_report.py

src/data/
  rr_features.py
  context_features.py

src/models/
  catnet_rr1d.py
  catnet_context1d.py

configs/
  phase3_rr_dann.yaml

docs/
  phase3_rr_dann_plan.md
```

RR features to add:

```text
RR_prev = R_i - R_{i-1}
RR_next = R_{i+1} - R_i
RR_ratio = RR_prev / median_RR_record
RR_prev_next_ratio = RR_prev / RR_next
```

New `.npz` keys:

```python
"rr_features": np.ndarray  # [N, 4]
"rr_feature_names": np.ndarray
```

Phase 3 primary metric:

```text
INCART held-out Macro-F1
```

Phase 3 primary class metric:

```text
INCART held-out S-F1
```

Critical counts to report:

```text
S -> N
S -> V
N -> V
V -> N
```

Thesis narrative if Phase 3 works:

```text
Global domain adaptation improves transfer, but supraventricular beats require rhythm-aware features.
Adding RR/context information improves S-class discrimination under domain shift.
```

---

## Suggested Prompt For A New Chat

Use this in a fresh chat:

```text
I am working on an ECG arrhythmia classification thesis repo.
The main project is ecg_thesis.
Please read ecg_thesis/docs/project_context_handoff.md first.
Then continue from Phase 3: RR/context features + CATNet1D + DANN, based on the current codebase.
Do not restart Phase 1 or Phase 2 unless needed for verification.
```

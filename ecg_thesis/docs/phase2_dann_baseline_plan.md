# PHASE 2 Plan - CATNet1D Baseline and DANN Domain Adaptation

## 0. Goal

Build Phase 2 on top of Phase 1.

Phase 1 already provides:

- MIT-BIH beat-level source dataset
- INCART beat-level target dataset
- annotation-centered beat extraction
- single-lead setup: MIT-BIH MLII and INCART II
- source-only ResNet1D baseline
- in-domain and cross-domain evaluation
- confusion matrices, example beat plots, and UMAP embeddings

Phase 2 has two goals:

1. Replace the previous Phase 2 InceptionTime1D/CNN-LSTM idea with a stronger CNN-Transformer baseline inspired by the CAT-Net notebooks:
   - `Arrhythmia_Classification_Code_28_(7)_60_epochs_loss_MIT_BIH_SMOTETomek.ipynb`
   - `AC_Code_25_v2_add_batch_Norm_and_logdir_INCART_SMOTETomek.ipynb`
2. Test a clean unsupervised domain adaptation baseline using DANN with the CATNet1D encoder.

This phase is still a baseline and analysis phase, not the final proposed method.

Important:

```text
Do not copy the notebooks directly.
Implement a clean PyTorch version with correct ChannelAttention1D, correct positional encoding, and DANN-compatible embeddings.
```

---

## 1. Evidence From Phase 1

### Phase 1 metrics

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

- `N` is easy and dominates the dataset.
- `V` has clearer morphology and is learned moderately well.
- `S` is the main failure mode.
- `S` morphology is often close to `N`.
- UMAP suggests features are influenced by both class and domain.

Research framing:

```text
Class-imbalanced and morphology-ambiguous domain transfer for beat-level ECG classification,
especially for supraventricular beats.
```

---

## 2. Main Research Questions

1. Does a CAT-Net inspired CNN-Transformer backbone improve MIT-BIH -> INCART transfer over ResNet1D?
2. Does standard global DANN improve target-domain Macro-F1 over CATNet1D source-only?
3. Does DANN improve the difficult `S` class, or only the easier `N`/`V` classes?
4. Does DANN reduce domain separability without destroying class/morphology structure?
5. If DANN fails or helps only slightly, does this support morphology-aware or class-aware adaptation for Phase 3?

---

## 3. Data Setup

Reuse the processed Phase 1 beat-level files:

```text
data/processed/mitbih_train.npz
data/processed/mitbih_test.npz
data/processed/incart_test.npz
```

Current `.npz` input:

```python
x.shape = [N, 1, 250]
y.shape = [N]
```

The notebooks use `(300, 1)` beats. For the main Phase 2 experiment, keep the current Phase 1 preprocessing:

```text
Input length = 250
Classes = N/S/V
No wavelet regeneration
No SMOTE-Tomek regeneration
```

Why:

```text
We want to isolate the effect of the backbone and DANN.
Changing beat length, denoising, balancing, and model at the same time would make the result hard to interpret.
```

### Target split

INCART must be split record-wise into:

```text
data/processed/incart_unlabeled.npz
data/processed/incart_test_heldout.npz
```

Recommended split:

```text
INCART adaptation records: I01-I50
INCART held-out test records: I51-I75
```

Rules:

- Do not use target labels during DANN training.
- Do not use target test labels for checkpoint selection or hyperparameter tuning.
- Do not random split INCART beats.
- Use record-wise split.

---

## 4. Label Space

Use the same 3-class mapping from Phase 1:

```python
CLASS_TO_ID = {
    "N": 0,
    "S": 1,
    "V": 2,
}
```

Do not switch to the 5-class `N/S/V/F/Q` setup in the main Phase 2 experiment. The notebooks use 5 classes, but our current cross-domain protocol is 3-class `N/S/V`.

The 5-class setup can be a later experiment after the 3-class MIT-BIH -> INCART comparison is stable.

---

## 5. Baseline Strategy

Recommended order:

1. Implement `CATNet1D` source-only.
2. Train `CATNet1D` on MIT-BIH source train.
3. Evaluate source-only on MIT-BIH test and INCART held-out.
4. Use `CATNet1D` as DANN feature extractor.
5. Compare:
   - Phase 1 ResNet1D source-only
   - CATNet1D source-only
   - DANN with CATNet1D encoder
   - optional ablations

Rationale:

```text
If the source-only backbone is weak, DANN may fail for reasons unrelated to domain adaptation.
```

---

## 6. CATNet1D Source-only Baseline

### Notebook inspiration

The newer notebooks contain a CNN-Transformer architecture:

```text
Conv1D + BatchNorm + ChannelAttention
Conv1D + BatchNorm + ChannelAttention
Conv1D + BatchNorm + ChannelAttention
Conv1D + BatchNorm + ChannelAttention
+ positional encoding
TransformerEncoder
Flatten
Dense 128
Dropout
Dense output
```

The notebook summaries show approximately:

```text
Total params: 1,189,637
```

But the notebook implementation has issues:

- `ChannelAttention` has broken indentation and reports `0` parameters.
- `positional_encoding` inside `build_model()` returns inside the first loop.
- Transformer FFN is simplified to one Dense layer.
- It uses prebuilt SMOTE-Tomek `.pkl` files instead of a reproducible preprocessing pipeline in the notebook.

Therefore:

```text
Implement the idea, not the exact notebook code.
```

### Main PyTorch model

Create:

```text
src/models/catnet1d.py
```

Class:

```python
CATNet1D
```

Current repo input:

```python
x.shape = [B, 1, 250]
```

The model should internally work in PyTorch channel-first format.

Recommended architecture for input length 250:

```text
Input [B, 1, 250]

Conv1D 16 filters, kernel=21, padding=same
BatchNorm1D
ReLU
ChannelAttention1D(16, reduction=8)
MaxPool1D kernel=3, stride=2, padding=1

Conv1D 32 filters, kernel=23, padding=same
BatchNorm1D
ReLU
ChannelAttention1D(32, reduction=8)
MaxPool1D kernel=3, stride=2, padding=1

Conv1D 64 filters, kernel=25, padding=same
BatchNorm1D
ReLU
ChannelAttention1D(64, reduction=8)
MaxPool1D kernel=3, stride=2, padding=1

Conv1D 128 filters, kernel=27, padding=same
BatchNorm1D
ReLU
ChannelAttention1D(128, reduction=8)

Transpose to [B, T, 128]
Add sinusoidal positional encoding
TransformerEncoder layer(s)
Flatten or temporal pooling
Dense 128
Dropout
Classifier num_classes=3
```

Because input length is 250, after three stride-2 pooling layers the temporal length will be about:

```text
250 -> 125 -> 63 -> 32
```

So the notebook's fixed `38 x 128` flatten size must not be hardcoded. Infer the feature length dynamically or use temporal pooling.

Preferred embedding design:

```text
Transformer output [B, T, 128]
Global average pooling over T
Dense 128
embedding [B, 128]
classifier [B, 3]
```

This is cleaner for DANN than flattening, and avoids length-specific classifier dimensions.

Model API requirements:

```python
logits = model(x)
logits, embedding = model(x, return_embedding=True)
embedding = model.forward_features(x)
model.embedding_dim == 128
model.classifier exists
```

---

## 7. ChannelAttention1D

Implement a correct 1D channel attention block.

Input:

```python
x.shape = [B, C, T]
```

Logic:

```text
avg_pool = mean over T -> [B, C]
max_pool = max over T -> [B, C]
shared MLP:
    Linear(C, C // reduction)
    ReLU
    Linear(C // reduction, C)
attention = sigmoid(MLP(avg_pool) + MLP(max_pool))
attention = attention[:, :, None]
output = x * attention
```

This fixes the notebook issue where `ChannelAttention` had `0` parameters.

---

## 8. Positional Encoding

Implement sinusoidal positional encoding correctly.

Input to Transformer:

```python
x.shape = [B, T, D]
```

Use:

```text
PE[pos, 2i]   = sin(pos / 10000^(2i/D))
PE[pos, 2i+1] = cos(pos / 10000^(2i/D))
```

The notebook's inner `positional_encoding()` returns inside the first loop. Do not reproduce that bug.

Implementation options:

- precompute `max_len` buffer in `__init__`
- slice `pe[:, :T, :]` in forward

---

## 9. Transformer Encoder

Use PyTorch:

```python
torch.nn.TransformerEncoderLayer(
    d_model=128,
    nhead=4,
    dim_feedforward=128,
    dropout=0.2,
    activation="relu",
    batch_first=True,
)
```

Then:

```python
torch.nn.TransformerEncoder(layer, num_layers=1)
```

This is a cleaner version of the notebook Transformer block.

Initial settings:

```yaml
d_model: 128
num_heads: 4
dff: 128
num_transformer_layers: 1
dropout: 0.2
```

---

## 10. DANN Architecture

DANN has three main components:

```text
Feature extractor
|-- Label classifier
`-- Gradient Reversal -> Domain classifier
```

### Feature extractor

Use `CATNet1D`.

Input:

```python
x.shape = [B, 1, 250]
```

Output:

```python
feature.shape = [B, 128]
```

### Label classifier

Use a classifier compatible with source-only `CATNet1D.classifier` so DANN can copy classifier weights.

Recommended:

```text
feature -> Linear(num_classes=3)
```

### Domain classifier

Predict source vs target:

```text
feature
-> Gradient Reversal Layer
-> Linear
-> ReLU
-> Dropout
-> Linear(num_domains=2)
```

Domain labels:

```python
source = 0
target = 1
```

---

## 11. Losses

For each step:

```text
source batch: (x_s, y_s)
target batch: (x_t)
```

Source classification loss:

```python
loss_cls = CrossEntropy(class_logits_source, y_s, weight=class_weights)
```

Optional focal loss ablation:

```text
focal_gamma = 2.0
```

Domain loss:

```python
domain_y_source = zeros(B_source)
domain_y_target = ones(B_target)
domain_y = concat(domain_y_source, domain_y_target)
loss_domain = CrossEntropy(domain_logits, domain_y)
```

Total DANN loss:

```python
loss_total = loss_cls + alpha * loss_domain
```

Initial setting:

```text
alpha = 0.2
```

Ablation:

```text
alpha = 0.1, 0.5, 1.0
```

---

## 12. Lambda Schedule and Warmup

Use progressive GRL schedule by default:

```python
p = current_step / total_steps
lambd = 2 / (1 + exp(-gamma * p)) - 1
```

Initial setting:

```yaml
gamma: 3.0
warmup_epochs: 3
```

During warmup:

```text
lambda = 0
alpha = 0
```

Rationale:

```text
Strong adversarial pressure too early can hurt class separability.
```

---

## 13. Configuration

Update:

```text
configs/phase2_dann.yaml
```

Recommended model/source-only/DANN sections:

```yaml
model:
  backbone: catnet1d
  embedding_dim: 128
  num_classes: 3
  num_domains: 2
  dropout: 0.2
  d_model: 128
  num_heads: 4
  dff: 128
  num_transformer_layers: 1
  attention_reduction: 8

source_only:
  enabled: true
  model: catnet1d
  checkpoint_prefix: source_only_catnet
  epochs: 50
  batch_size: 64
  lr: 0.001
  weight_decay: 0.0001
  use_class_weights: true
  early_stopping_patience: 10

training:
  epochs: 50
  source_batch_size: 64
  target_batch_size: 64
  lr: 0.0003
  weight_decay: 0.0001
  optimizer: adamw
  source_loss: weighted_ce
  focal_gamma: 2.0
  use_class_weights: true
  early_stopping_patience: 10
  checkpoint_metric: source_val_macro_f1

dann:
  alpha: 0.2
  lambda_schedule: progressive
  gamma: 3.0
  fixed_lambda: 1.0
  warmup_epochs: 3
  source_init_checkpoint: outputs/checkpoints/source_only_catnet_best.pt
```

Keep the existing `paths`, `data`, `incart_split`, `evaluation`, and `visualization` sections.

---

## 14. Required Experiments

### Experiment 0: Phase 1 recap

Use existing results:

```text
Source-only ResNet1D, MIT-BIH -> MIT-BIH
Source-only ResNet1D, MIT-BIH -> INCART
```

### Experiment 1: CATNet1D source-only

Train:

```text
CATNet1D on MIT-BIH labeled train
```

Evaluate:

```text
MIT-BIH test
INCART held-out test
```

Purpose:

```text
Determine whether CNN + Transformer morphology modeling improves transfer before adaptation.
```

### Experiment 2: DANN with CATNet1D

Train:

```text
MIT-BIH labeled train + INCART unlabeled adaptation records
```

Evaluate:

```text
MIT-BIH test
INCART held-out test
```

Purpose:

```text
Check whether global domain alignment improves target Macro-F1 over CATNet1D source-only.
```

### Experiment 3: Loss ablation

Run:

```text
weighted CE
focal loss
```

### Experiment 4: DANN schedule and alpha ablation

Test:

```text
alpha = 0.1, 0.2, 0.5, 1.0
lambda fixed vs progressive
warmup_epochs = 0, 3, 5
```

### Optional Experiment 5: Notebook-style preprocessing ablation

Only after the main CATNet1D + DANN baseline is complete.

Test separately:

```text
beat length 250 vs 300
current preprocessing vs wavelet-denoised preprocessing
class-weighted CE/focal/weighted sampler vs SMOTE-Tomek
3-class N/S/V vs 5-class N/S/V/F/Q
```

Important:

```text
Do not mix these changes into the main Phase 2 baseline.
```

---

## 15. Metrics

Report:

1. Accuracy
2. Macro-F1
3. Per-class precision
4. Per-class recall
5. Per-class F1
6. Confusion matrix
7. Domain classifier accuracy
8. Source validation Macro-F1 used for checkpoint selection

Primary metric:

```text
INCART held-out Macro-F1
```

Secondary focus:

```text
S-F1
S precision
S recall
V-F1
```

Do not claim success from lower domain accuracy alone.

---

## 16. Required Result Tables

Create:

```text
outputs/phase2_dann_report.md
```

Main table:

| Method | Backbone | Source | Target Unlabeled | Test Domain | Accuracy | Macro-F1 | N-F1 | S-F1 | V-F1 |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| Source-only | ResNet1D | MIT-BIH | None | MIT-BIH | | | | | |
| Source-only | ResNet1D | MIT-BIH | None | INCART held-out | | | | | |
| Source-only | CATNet1D | MIT-BIH | None | MIT-BIH | | | | | |
| Source-only | CATNet1D | MIT-BIH | None | INCART held-out | | | | | |
| DANN | CATNet1D | MIT-BIH | INCART adapt | MIT-BIH | | | | | |
| DANN | CATNet1D | MIT-BIH | INCART adapt | INCART held-out | | | | | |

Domain table:

| Method | Domain Accuracy | Target Macro-F1 | Notes |
|---|---:|---:|---|
| Source-only encoder | | | |
| DANN encoder | | | |

---

## 17. Required Figures

Save figures to:

```text
outputs/figures/phase2/
```

Required:

1. DANN training curves:
   - total loss
   - classification loss
   - domain loss
   - source validation macro-F1
   - domain accuracy
   - lambda
   - alpha
2. Confusion matrix comparison on INCART held-out:
   - source-only CATNet1D
   - DANN CATNet1D
3. UMAP before adaptation:
   - source-only CATNet1D embeddings
4. UMAP after DANN:
   - DANN embeddings
5. Per-class F1 bar chart:
   - ResNet1D source-only
   - CATNet1D source-only
   - DANN
6. Failure case plots:
   - N predicted as S
   - S predicted as N
   - V confused with N/S

UMAP is qualitative only.

---

## 18. CLI Commands

Implement runnable scripts:

```bash
python scripts/phase2/08_split_incart_unlabeled_test.py --config configs/phase2_dann.yaml

python scripts/phase2/09_train_source_only_catnet.py --config configs/phase2_dann.yaml

python scripts/phase2/10_eval_source_only_catnet.py \
  --config configs/phase2_dann.yaml \
  --checkpoint outputs/checkpoints/source_only_catnet_best.pt

python scripts/phase2/11_train_dann.py --config configs/phase2_dann.yaml

python scripts/phase2/12_eval_dann_in_domain.py \
  --config configs/phase2_dann.yaml \
  --checkpoint outputs/checkpoints/dann_best.pt

python scripts/phase2/13_eval_dann_cross_domain.py \
  --config configs/phase2_dann.yaml \
  --checkpoint outputs/checkpoints/dann_best.pt

python scripts/phase2/14_visualize_phase2_embeddings.py \
  --config configs/phase2_dann.yaml \
  --source-checkpoint outputs/checkpoints/source_only_catnet_best.pt \
  --dann-checkpoint outputs/checkpoints/dann_best.pt

python scripts/phase2/15_make_phase2_report.py --config configs/phase2_dann.yaml
```

Support debug flags:

```text
--epochs
--max-fit-samples
--max-val-samples
--max-source-samples
--max-target-samples
```

---

## 19. Checkpoints and Outputs

Save source-only:

```text
outputs/checkpoints/source_only_catnet_best.pt
outputs/checkpoints/source_only_catnet_latest.pt
```

Save DANN:

```text
outputs/checkpoints/dann_best.pt
outputs/checkpoints/dann_latest.pt
```

Expected outputs:

```text
outputs/
|-- checkpoints/
|   |-- source_only_catnet_best.pt
|   |-- source_only_catnet_latest.pt
|   |-- dann_best.pt
|   `-- dann_latest.pt
|-- metrics/
|   |-- source_only_catnet_mitbih_test.json
|   |-- source_only_catnet_incart_heldout.json
|   |-- dann_mitbih_test.json
|   |-- dann_incart_heldout.json
|   `-- phase2_ablation_results.csv
|-- figures/
|   `-- phase2/
|       |-- training_curves.png
|       |-- confusion_source_only_vs_dann_incart.png
|       |-- umap_source_only_catnet.png
|       |-- umap_dann.png
|       |-- per_class_f1_comparison.png
|       `-- failure_cases.png
`-- phase2_dann_report.md
```

---

## 20. Acceptance Criteria

### CATNet1D baseline

- `CATNet1D` is implemented in `src/models/catnet1d.py`.
- `CATNet1D` is registered in `src/models/__init__.py`.
- Correct `ChannelAttention1D` is implemented.
- Correct sinusoidal positional encoding is implemented.
- Source-only CATNet1D trains on MIT-BIH.
- Source-only CATNet1D is evaluated on MIT-BIH test and INCART held-out test.
- It is compared against Phase 1 ResNet1D.

### DANN

- GRL is implemented.
- DANN supports CATNet1D encoder.
- DANN initializes from `source_only_catnet_best.pt` when available.
- Source and target dataloaders are used together.
- Target labels are ignored during training.
- Best and latest checkpoints are saved.

### Evaluation

- DANN is evaluated on MIT-BIH test.
- DANN is evaluated on INCART held-out test.
- Metrics are saved in JSON.
- Predictions and confusion matrices are saved.

DANN is considered useful only if:

```text
INCART held-out Macro-F1 improves over source-only CATNet1D,
S-F1 or S precision improves meaningfully,
V-F1 does not collapse,
N-F1 does not drop severely.
```

---

## 21. Interpretation Guide

### Case 1: CATNet1D improves source-only transfer

Conclusion:

```text
CNN + Transformer morphology/context modeling helps beat-level ECG transfer.
```

Next step:

```text
Use CATNet1D as the default encoder for DA.
```

### Case 2: DANN improves target Macro-F1 and S/V F1

Conclusion:

```text
Domain adaptation helps cross-domain ECG beat classification.
```

Next step:

```text
Build morphology-aware or class-aware DA on top of DANN.
```

### Case 3: DANN reduces domain separability but target F1 does not improve

Conclusion:

```text
Global alignment alone is insufficient.
It may align domains without preserving class-discriminative morphology.
```

Next step:

```text
Use class-aware, prototype-based, or morphology-aware alignment.
```

### Case 4: S remains poor across all models

Conclusion:

```text
Single-beat morphology is insufficient for supraventricular beats.
```

Next step:

```text
Add RR interval features or previous/current/next beat context.
```

---

## 22. Important Pitfalls

- Do not use target labels during DANN training.
- Do not tune hyperparameters on target held-out test labels.
- Do not random split target beats.
- Do not evaluate only accuracy.
- Do not overclaim from UMAP alone.
- Do not assume lower domain accuracy means better classification.
- Do not let DANN collapse class structure.
- Do not copy the notebook ChannelAttention bug.
- Do not copy the notebook positional encoding bug.
- Do not add wavelet preprocessing into the main CATNet1D experiment.
- Do not add SMOTE-Tomek into the main CATNet1D experiment.
- Do not change to 5-class AAMI until the 3-class Phase 2 comparison is complete.

---

## 23. Final Deliverable

Create:

```text
outputs/phase2_dann_report.md
```

The report must answer:

1. Does CATNet1D improve over ResNet1D source-only?
2. Does DANN improve MIT-BIH -> INCART held-out target performance?
3. Which class benefits most?
4. Which class fails most?
5. Does DANN reduce domain separability in embedding space?
6. Does global alignment preserve class/morphology structure?
7. Does the result motivate morphology-aware or class-aware adaptation for Phase 3?

---

## 24. Summary for Codex

Implement Phase 2 as:

```text
1. CATNet1D source-only baseline inspired by the newer MIT-BIH/INCART notebooks
2. Correct PyTorch ChannelAttention1D and positional encoding
3. DANN with CATNet1D encoder
4. Class-aware loss ablations
5. Clean INCART record-wise adaptation/test split
6. Full comparison against Phase 1 ResNet1D
```

Do not include wavelet preprocessing or SMOTE-Tomek in the main experiment. Treat them as later ablations after the CATNet1D baseline is understood.

The key research point is not merely whether domain accuracy drops. The key question is whether target Macro-F1 and minority-class morphology transfer improve, especially for the `S` class.

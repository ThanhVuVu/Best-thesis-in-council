# PHASE 3 Plan - Rhythm-Aware and Class-Aware Domain Adaptation

## 0. Goal

Build Phase 3 on top of the successful Phase 2 result:

```text
Source domain: MIT-BIH Arrhythmia Database
Target domain: St Petersburg INCART 12-lead Arrhythmia Database
Task: 3-class beat classification, N/S/V
Best Phase 2 method: CATNet1D + DANN
```

Phase 2 showed that global adversarial domain adaptation is useful:

```text
Source-only CATNet1D on INCART held-out Macro-F1: 0.5202
DANN CATNet1D on INCART held-out Macro-F1:        0.6783
Delta:                                             +0.1581
```

The largest improvement came from the difficult `S` class:

```text
Source-only CATNet1D INCART held-out S-F1: 0.1567
DANN CATNet1D INCART held-out S-F1:        0.4178
Delta:                                    +0.2611
```

However, `S` remains unstable:

- `S` support in INCART held-out is only 260 beats.
- UMAP shows `S` is scattered and boundary-like rather than a clean cluster.
- MIT-BIH source-test `S` recall remains low after DANN.
- `S` morphology is often close to `N`, so single-beat morphology alone may be insufficient.

Phase 3 therefore focuses on:

```text
Rhythm-aware and class-aware domain adaptation for better S-class discrimination.
```

Main thesis claim to test:

```text
Global domain adaptation improves cross-domain transfer, but supraventricular beats require rhythm-aware features.
Adding RR/context information should improve S-class discrimination under domain shift.
```

---

## 1. Research Questions

1. Does adding RR interval information improve source-only CATNet1D performance on `S`?
2. Does CATNet1D + RR + DANN improve INCART held-out Macro-F1 over Phase 2 DANN?
3. Does RR information reduce `S -> N` and `S -> V` errors?
4. Does RR information help without causing `N -> V` or `V -> N` regression?
5. If RR helps only partially, does multi-beat context provide additional improvement?
6. Does class-aware alignment improve minority-class transfer beyond global DANN?

Primary target-domain metric:

```text
INCART held-out Macro-F1
```

Primary class metric:

```text
INCART held-out S-F1
```

Critical confusion counts:

```text
S -> N
S -> V
N -> V
V -> N
```

---

## 2. Phase 2 Baselines To Compare Against

Use the existing Phase 2 numbers as fixed baselines.

### Source-only CATNet1D

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

### DANN CATNet1D

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

Phase 3 should not be considered successful from accuracy alone. It must improve or preserve the class-level behavior that matters:

```text
Target Macro-F1 improves over 0.6783, or
Target S-F1 improves over 0.4178 without severe N/V collapse.
```

---

## 3. Data Setup

Reuse the current processed files:

```text
data/processed/mitbih_train.npz
data/processed/mitbih_test.npz
data/processed/incart_unlabeled.npz
data/processed/incart_test_heldout.npz
```

Current beat tensor:

```python
x.shape = [N, 1, 250]
y.shape = [N]
```

Phase 3 adds rhythm features to the same examples.

New `.npz` keys:

```python
"rr_features": np.ndarray       # [N, 4]
"rr_feature_names": np.ndarray  # ["rr_prev", "rr_next", "rr_ratio", "rr_prev_next_ratio"]
```

Do not change the main preprocessing at the same time:

- Keep annotation-centered beat windows.
- Keep input length 250.
- Keep single-lead setup: MIT-BIH MLII and INCART II.
- Keep 3-class `N/S/V`.
- Do not add wavelet denoising to the main Phase 3 experiment.
- Do not add SMOTE/SMOTE-Tomek to the main Phase 3 experiment.

Reason:

```text
Phase 3 should isolate the effect of rhythm/context features and class-aware adaptation.
```

---

## 4. RR Feature Design

For each beat `i` within a record:

```text
RR_prev = R_i - R_{i-1}
RR_next = R_{i+1} - R_i
RR_ratio = RR_prev / median_RR_record
RR_prev_next_ratio = RR_prev / RR_next
```

Use seconds rather than raw sample counts when possible:

```python
rr_prev_seconds = (sample_i - sample_prev) / fs
rr_next_seconds = (sample_next - sample_i) / fs
```

Recommended raw feature vector:

```python
[
    rr_prev_seconds,
    rr_next_seconds,
    rr_prev_seconds / median_rr_seconds_for_record,
    rr_prev_seconds / rr_next_seconds,
]
```

### Boundary handling

For the first beat in a record:

```text
RR_prev = median_RR_record
```

For the last beat in a record:

```text
RR_next = median_RR_record
```

For invalid or zero denominators:

```text
clip denominator to epsilon, e.g. 1e-6
```

### Normalization

Fit normalization statistics only on MIT-BIH training data:

```text
mean_rr_features
std_rr_features
```

Apply the same statistics to:

```text
MIT-BIH test
INCART unlabeled/adaptation
INCART held-out test
```

Save the statistics for reproducibility:

```text
outputs/metrics/phase3_rr_normalization.json
```

Important:

```text
Do not fit RR normalization on INCART held-out labels or target test distribution.
```

It is acceptable to transform target unlabeled data using source-fitted statistics.

---

## 5. Implementation Files

Create:

```text
configs/phase3_rr_dann.yaml

src/data/rr_features.py
src/models/catnet_rr1d.py

scripts/phase3/01_analyze_failures.py
scripts/phase3/02_prepare_rr_features.py
scripts/phase3/03_train_source_only_catnet_rr.py
scripts/phase3/04_eval_source_only_catnet_rr.py
scripts/phase3/05_train_dann_rr.py
scripts/phase3/06_eval_dann_rr_in_domain.py
scripts/phase3/07_eval_dann_rr_cross_domain.py
scripts/phase3/08_visualize_phase3_embeddings.py
scripts/phase3/09_make_phase3_report.py
```

Optional later files:

```text
src/data/context_features.py
src/models/catnet_context1d.py
scripts/phase3/10_prepare_context_windows.py
scripts/phase3/11_train_dann_context.py
```

---

## 6. Model Design: CATNetRR1D

Use Phase 2 `CATNet1D` as the waveform encoder.

Waveform branch:

```text
x [B, 1, 250]
-> CATNet1D.forward_features(x)
-> waveform_embedding [B, 128]
```

RR branch:

```text
rr_features [B, 4]
-> Linear(4, 32)
-> BatchNorm1d(32)
-> ReLU
-> Dropout
-> Linear(32, 32)
-> ReLU
-> rr_embedding [B, 32]
```

Fusion:

```text
concat([waveform_embedding, rr_embedding]) -> [B, 160]
-> Linear(160, 128)
-> ReLU
-> Dropout
-> fused_embedding [B, 128]
-> Linear(128, 3)
```

Required API:

```python
logits = model(x, rr_features)
logits, embedding = model(x, rr_features, return_embedding=True)
embedding = model.forward_features(x, rr_features)
model.embedding_dim == 128
model.classifier exists
```

Rationale:

```text
Keep DANN compatibility by preserving a single 128-dimensional fused embedding.
```

---

## 7. Training Strategy

Recommended order:

1. Analyze Phase 2 failure cases.
2. Add RR features to all processed `.npz` files.
3. Train source-only CATNetRR1D.
4. Evaluate source-only CATNetRR1D on MIT-BIH and INCART held-out.
5. Train CATNetRR1D + DANN.
6. Evaluate DANN-RR on MIT-BIH and INCART held-out.
7. Visualize embeddings and confusion matrix changes.
8. If RR is insufficient, add multi-beat context.
9. If global alignment is insufficient, add class-aware/prototype alignment.

Checkpoint selection:

```text
Use source validation Macro-F1 only.
Do not select checkpoints using INCART held-out labels.
```

Target labels:

```text
Ignored during DANN training.
Used only for final evaluation on INCART held-out.
```

---

## 8. DANN-RR Design

DANN-RR should mirror Phase 2 DANN, replacing the encoder with `CATNetRR1D`.

Inputs:

```text
source batch: x_s, rr_s, y_s
target batch: x_t, rr_t
```

Feature extractor:

```text
CATNetRR1D.forward_features(x, rr_features) -> [B, 128]
```

Label classifier:

```text
fused_embedding -> Linear(128, 3)
```

Domain classifier:

```text
fused_embedding
-> Gradient Reversal
-> Linear
-> ReLU
-> Dropout
-> Linear(2)
```

Loss:

```text
loss_total = loss_cls + alpha * loss_domain
```

Start with Phase 2 settings:

```yaml
dann:
  alpha: 0.2
  lambda_schedule: progressive
  gamma: 3.0
  warmup_epochs: 3
```

Do not tune aggressively before establishing the main comparison.

---

## 9. Class-Aware Extensions

Only add these after the basic RR source-only and RR-DANN experiments are complete.

### Option A: Class-weighted source classification

Continue using weighted CE from Phase 2.

Track specifically:

```text
S precision
S recall
S-F1
S -> N
S -> V
```

### Option B: Focal loss ablation

Test:

```yaml
source_loss: focal
focal_gamma: 2.0
```

Compare against:

```yaml
source_loss: weighted_ce
```

### Option C: Pseudo-label class-aware alignment

Use target pseudo-labels only when confidence is high:

```text
max_softmax_probability >= 0.9
```

For each class, align source and confident target prototypes:

```text
prototype_c_source = mean(source_embeddings where y_s == c)
prototype_c_target = mean(target_embeddings where pseudo_y_t == c and confidence >= threshold)
loss_proto = sum_c distance(prototype_c_source, prototype_c_target)
```

Initial prototype loss:

```text
MSE or cosine distance
```

Initial weight:

```yaml
prototype_loss_weight: 0.05
```

Important safeguards:

- Do not use target held-out labels.
- Do not trust early pseudo-labels too much.
- Enable prototype loss only after warmup.
- Report pseudo-label class counts and confidence distribution.

### Option D: Conditional DANN

Condition domain classifier on:

```text
embedding
predicted class probabilities
```

This can be useful if global alignment mixes classes incorrectly, but it is more complex than prototype alignment. Treat it as optional.

---

## 10. Multi-Beat Context Extension

Use only if RR alone does not improve `S` enough.

Context input:

```text
previous beat, current beat, next beat
```

Possible tensor:

```python
x_context.shape = [N, 3, 250]
```

Two implementation options:

### Option A: Channel-stacked context

Treat previous/current/next beats as channels:

```text
[B, 3, 250] -> CATNetContext1D
```

Pros:

- Simple.
- Minimal sequence-modeling change.

Cons:

- The model may treat context beats as channels rather than temporal sequence.

### Option B: Shared beat encoder + temporal context model

```text
each beat [B, 1, 250] -> shared CATNet1D encoder -> [B, 3, 128]
-> small Transformer/GRU over beat positions
-> context_embedding [B, 128]
```

Pros:

- Cleaner representation of beat sequence.

Cons:

- More code and more compute.

Recommendation:

```text
Start with RR features first. Add context only if the Phase 3 RR result still leaves S unstable.
```

---

## 11. Configuration Draft

Create:

```text
configs/phase3_rr_dann.yaml
```

Suggested sections:

```yaml
project:
  phase: phase3_rr_dann
  seed: 42

paths:
  mitbih_train: data/processed/mitbih_train.npz
  mitbih_test: data/processed/mitbih_test.npz
  incart_unlabeled: data/processed/incart_unlabeled.npz
  incart_heldout: data/processed/incart_test_heldout.npz
  checkpoint_dir: outputs/checkpoints
  metrics_dir: outputs/metrics
  figures_dir: outputs/figures/phase3

data:
  input_length: 250
  num_channels: 1
  num_classes: 3
  class_names: [N, S, V]
  rr_feature_names:
    - rr_prev
    - rr_next
    - rr_ratio
    - rr_prev_next_ratio

model:
  backbone: catnet_rr1d
  waveform_embedding_dim: 128
  rr_embedding_dim: 32
  embedding_dim: 128
  num_classes: 3
  dropout: 0.2
  d_model: 128
  num_heads: 4
  dff: 128
  num_transformer_layers: 1
  attention_reduction: 8

source_only:
  checkpoint_prefix: source_only_catnet_rr
  epochs: 50
  batch_size: 64
  lr: 0.001
  weight_decay: 0.0001
  optimizer: adamw
  source_loss: weighted_ce
  use_class_weights: true
  early_stopping_patience: 10
  checkpoint_metric: source_val_macro_f1

training:
  checkpoint_prefix: dann_rr
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
  source_init_checkpoint: outputs/checkpoints/source_only_catnet_rr_best.pt

prototype_alignment:
  enabled: false
  confidence_threshold: 0.9
  loss_weight: 0.05
  warmup_epochs: 5
```

---

## 12. Required Experiments

### Experiment 0: Phase 2 recap

Use existing reports and metrics:

```text
CATNet1D source-only
CATNet1D + DANN
```

### Experiment 1: Failure analysis

Analyze Phase 2 predictions:

```text
DANN on MIT-BIH test
DANN on INCART held-out
```

Required outputs:

```text
outputs/metrics/phase3_failure_summary.json
outputs/figures/phase3/failure_examples_s_to_n.png
outputs/figures/phase3/failure_examples_s_to_v.png
outputs/figures/phase3/failure_examples_n_to_v.png
outputs/figures/phase3/failure_examples_v_to_n.png
```

Questions:

- Are `S -> N` errors associated with near-normal RR?
- Are `S -> V` errors associated with abnormal morphology?
- Are `N -> V` errors concentrated in specific INCART records?
- Are failures record-specific, class-specific, or both?

### Experiment 2: CATNetRR1D source-only

Train:

```text
MIT-BIH labeled train with waveform + RR features
```

Evaluate:

```text
MIT-BIH test
INCART held-out
```

Purpose:

```text
Measure whether RR features help before domain adaptation.
```

### Experiment 3: DANN-RR

Train:

```text
MIT-BIH labeled train + INCART unlabeled adaptation records
```

Evaluate:

```text
MIT-BIH test
INCART held-out
```

Purpose:

```text
Measure whether rhythm-aware global domain adaptation beats Phase 2 DANN.
```

### Experiment 4: Loss ablation

Run only after the main DANN-RR result:

```text
weighted CE
focal loss
```

### Experiment 5: Prototype/class-aware alignment

Run only if:

```text
DANN-RR improves Macro-F1 but S remains unstable, or
DANN-RR improves N/V but not S.
```

Compare:

```text
DANN-RR
DANN-RR + prototype alignment
```

### Optional Experiment 6: Beat context

Run only if RR features are insufficient:

```text
CATNetContext1D source-only
CATNetContext1D + DANN
```

---

## 13. Metrics

Report for each model and test domain:

1. Accuracy
2. Macro-F1
3. Per-class precision
4. Per-class recall
5. Per-class F1
6. Confusion matrix
7. Domain classifier accuracy for DANN variants
8. Source validation Macro-F1 for checkpoint selection

Additional Phase 3 diagnostics:

```text
RR feature distribution by class and domain
RR feature distribution for correct vs incorrect S predictions
Pseudo-label class counts if prototype alignment is enabled
Pseudo-label confidence histograms if prototype alignment is enabled
```

Do not claim success from:

- Higher accuracy alone.
- Lower domain accuracy alone.
- UMAP appearance alone.
- Improvement on `N` while `S` collapses.

---

## 14. Required Result Tables

Create:

```text
outputs/phase3_rr_dann_report.md
```

Main table:

| Method | Input | Source | Target Unlabeled | Test Domain | Accuracy | Macro-F1 | N-F1 | S-F1 | V-F1 |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| Source-only | CATNet1D | MIT-BIH | None | MIT-BIH |  |  |  |  |  |
| Source-only | CATNet1D | MIT-BIH | None | INCART held-out |  |  |  |  |  |
| DANN | CATNet1D | MIT-BIH | INCART adapt | MIT-BIH |  |  |  |  |  |
| DANN | CATNet1D | MIT-BIH | INCART adapt | INCART held-out |  |  |  |  |  |
| Source-only | CATNet1D + RR | MIT-BIH | None | MIT-BIH |  |  |  |  |  |
| Source-only | CATNet1D + RR | MIT-BIH | None | INCART held-out |  |  |  |  |  |
| DANN | CATNet1D + RR | MIT-BIH | INCART adapt | MIT-BIH |  |  |  |  |  |
| DANN | CATNet1D + RR | MIT-BIH | INCART adapt | INCART held-out |  |  |  |  |  |

Critical confusion table:

| Method | Test Domain | S -> N | S -> V | N -> V | V -> N |
|---|---|---:|---:|---:|---:|
| CATNet1D source-only | INCART held-out |  |  |  |  |
| CATNet1D DANN | INCART held-out |  |  |  |  |
| CATNet1D + RR source-only | INCART held-out |  |  |  |  |
| CATNet1D + RR DANN | INCART held-out |  |  |  |  |

RR diagnostic table:

| Domain | Class | Count | rr_prev mean | rr_next mean | rr_ratio mean | rr_prev_next_ratio mean |
|---|---|---:|---:|---:|---:|---:|
| MIT-BIH train | N |  |  |  |  |  |
| MIT-BIH train | S |  |  |  |  |  |
| MIT-BIH train | V |  |  |  |  |  |
| INCART held-out | N |  |  |  |  |  |
| INCART held-out | S |  |  |  |  |  |
| INCART held-out | V |  |  |  |  |  |

---

## 15. Required Figures

Save figures to:

```text
outputs/figures/phase3/
```

Required:

1. RR feature distributions by class and domain.
2. Confusion matrix comparison:
   - Phase 2 CATNet1D source-only
   - Phase 2 DANN
   - CATNetRR1D source-only
   - DANN-RR
3. Per-class F1 comparison across Phase 2 and Phase 3.
4. DANN-RR training curves:
   - total loss
   - classification loss
   - domain loss
   - source validation Macro-F1
   - domain accuracy
   - lambda
   - alpha
5. UMAP for CATNetRR1D source-only embeddings.
6. UMAP for DANN-RR embeddings.
7. Failure case plots for:
   - `S -> N`
   - `S -> V`
   - `N -> V`
   - `V -> N`

Optional:

```text
Pseudo-label confidence histogram
Prototype distance by class over training
Record-wise Macro-F1 on INCART held-out
```

---

## 16. CLI Commands

Prepare RR features:

```bash
python scripts/phase3/02_prepare_rr_features.py --config configs/phase3_rr_dann.yaml
```

Analyze Phase 2 failures:

```bash
python scripts/phase3/01_analyze_failures.py \
  --config configs/phase3_rr_dann.yaml \
  --checkpoint outputs/checkpoints/dann_best.pt
```

Train source-only CATNetRR1D:

```bash
python scripts/phase3/03_train_source_only_catnet_rr.py \
  --config configs/phase3_rr_dann.yaml
```

Evaluate source-only CATNetRR1D:

```bash
python scripts/phase3/04_eval_source_only_catnet_rr.py \
  --config configs/phase3_rr_dann.yaml \
  --checkpoint outputs/checkpoints/source_only_catnet_rr_best.pt
```

Train DANN-RR:

```bash
python scripts/phase3/05_train_dann_rr.py \
  --config configs/phase3_rr_dann.yaml
```

Evaluate DANN-RR in-domain:

```bash
python scripts/phase3/06_eval_dann_rr_in_domain.py \
  --config configs/phase3_rr_dann.yaml \
  --checkpoint outputs/checkpoints/dann_rr_best.pt
```

Evaluate DANN-RR cross-domain:

```bash
python scripts/phase3/07_eval_dann_rr_cross_domain.py \
  --config configs/phase3_rr_dann.yaml \
  --checkpoint outputs/checkpoints/dann_rr_best.pt
```

Visualize:

```bash
python scripts/phase3/08_visualize_phase3_embeddings.py \
  --config configs/phase3_rr_dann.yaml \
  --source-checkpoint outputs/checkpoints/source_only_catnet_rr_best.pt \
  --dann-checkpoint outputs/checkpoints/dann_rr_best.pt
```

Make report:

```bash
python scripts/phase3/09_make_phase3_report.py \
  --config configs/phase3_rr_dann.yaml
```

Debug flags should mirror Phase 2:

```text
--epochs
--max-fit-samples
--max-val-samples
--max-source-samples
--max-target-samples
```

---

## 17. Expected Outputs

```text
outputs/
|-- checkpoints/
|   |-- source_only_catnet_rr_best.pt
|   |-- source_only_catnet_rr_latest.pt
|   |-- dann_rr_best.pt
|   `-- dann_rr_latest.pt
|-- metrics/
|   |-- phase3_rr_normalization.json
|   |-- phase3_failure_summary.json
|   |-- source_only_catnet_rr_mitbih_test.json
|   |-- source_only_catnet_rr_incart_heldout.json
|   |-- dann_rr_mitbih_test.json
|   |-- dann_rr_incart_heldout.json
|   `-- phase3_ablation_results.csv
|-- figures/
|   `-- phase3/
|       |-- rr_feature_distributions.png
|       |-- confusion_phase2_phase3_incart.png
|       |-- per_class_f1_phase2_phase3.png
|       |-- training_curves_dann_rr.png
|       |-- umap_source_only_catnet_rr.png
|       |-- umap_dann_rr.png
|       |-- failure_examples_s_to_n.png
|       |-- failure_examples_s_to_v.png
|       |-- failure_examples_n_to_v.png
|       `-- failure_examples_v_to_n.png
`-- phase3_rr_dann_report.md
```

---

## 18. Acceptance Criteria

### Data

- RR features are computed for all required processed `.npz` files.
- RR features preserve the original example order.
- RR feature names are stored.
- Boundary beats are handled deterministically.
- Normalization statistics are fitted on MIT-BIH train only.
- A validation script confirms `x`, `y`, metadata, and `rr_features` lengths match.

### Model

- `CATNetRR1D` is implemented.
- `CATNetRR1D` uses the existing `CATNet1D` waveform encoder.
- `CATNetRR1D.forward_features(x, rr_features)` returns `[B, 128]`.
- `CATNetRR1D` supports `return_embedding=True`.
- DANN-RR can use the fused embedding without special-case code.

### Training

- Source-only CATNetRR1D trains on MIT-BIH.
- DANN-RR trains on MIT-BIH labeled + INCART unlabeled.
- Target labels are ignored during DANN-RR training.
- Best checkpoint selection uses source validation Macro-F1 only.
- Best and latest checkpoints are saved.

### Evaluation

- Source-only CATNetRR1D is evaluated on MIT-BIH test and INCART held-out.
- DANN-RR is evaluated on MIT-BIH test and INCART held-out.
- Metrics are saved as JSON.
- Confusion matrices and predictions are saved.
- Phase 3 report compares against Phase 2.

### Success Criteria

Preferred success:

```text
DANN-RR improves INCART held-out Macro-F1 over Phase 2 DANN
and improves or preserves INCART held-out S-F1.
```

Acceptable success:

```text
DANN-RR improves INCART held-out S-F1 meaningfully
without severe Macro-F1, N-F1, or V-F1 regression.
```

Negative result but still useful:

```text
RR features do not improve S.
This supports moving to explicit multi-beat context or class-aware/prototype alignment.
```

---

## 19. Interpretation Guide

### Case 1: RR source-only improves S

Conclusion:

```text
Rhythm information is useful even without target adaptation.
```

Next:

```text
Use CATNetRR1D as the default Phase 3 encoder.
```

### Case 2: DANN-RR improves Macro-F1 and S-F1

Conclusion:

```text
Rhythm-aware domain adaptation improves supraventricular beat transfer.
```

Thesis framing:

```text
Single-beat morphology and global alignment are not enough; rhythm-aware features improve minority-class transfer.
```

### Case 3: RR helps source but not target

Conclusion:

```text
RR features may be domain-shifted or globally aligned poorly.
```

Next:

```text
Try class-aware/prototype alignment or record-normalized RR variants.
```

### Case 4: DANN-RR improves N/V but not S

Conclusion:

```text
Global alignment still favors dominant or clearer classes.
```

Next:

```text
Add class-aware alignment using confident target pseudo-labels.
```

### Case 5: RR fails broadly

Conclusion:

```text
Simple RR features are insufficient.
```

Next:

```text
Add previous/current/next beat context.
```

---

## 20. Important Pitfalls

- Do not use INCART held-out labels for training, checkpoint selection, or normalization fitting.
- Do not tune repeatedly on INCART held-out and then report it as an untouched test result.
- Do not fit RR normalization on all domains together.
- Do not random split INCART beats.
- Do not report only accuracy.
- Do not claim DANN success only because domain accuracy is near 0.5.
- Do not let `S` improvements hide severe `V` collapse.
- Do not introduce wavelet denoising, SMOTE-Tomek, 5-class labels, and RR features all at once.
- Do not use pseudo-label class-aware losses before the basic DANN-RR baseline is understood.
- Do not overinterpret UMAP as quantitative evidence.

---

## 21. Final Deliverable

Create:

```text
outputs/phase3_rr_dann_report.md
```

The report must answer:

1. Do RR features improve CATNet1D source-only performance?
2. Does DANN-RR improve MIT-BIH -> INCART held-out transfer over Phase 2 DANN?
3. Does the `S` class improve?
4. Which confusion pairs improve or regress?
5. Are RR feature distributions meaningfully different across `N/S/V`?
6. Does DANN-RR preserve class structure while reducing domain separability?
7. Is class-aware/prototype alignment needed after RR?
8. Is multi-beat context needed after RR?

---

## 22. Summary for Codex

Implement Phase 3 as:

```text
1. Analyze Phase 2 failure cases.
2. Add RR interval features to the existing processed datasets.
3. Build CATNetRR1D by fusing CATNet1D waveform embeddings with RR embeddings.
4. Train and evaluate CATNetRR1D source-only.
5. Train and evaluate CATNetRR1D + DANN.
6. Compare directly against Phase 2 CATNet1D and CATNet1D + DANN.
7. Add class-aware/prototype alignment only after the basic RR-DANN result is clear.
8. Add multi-beat context only if RR features are insufficient.
```

The key research question is whether rhythm-aware information improves cross-domain minority-class classification, especially supraventricular `S` beats.

# Revised Coding-Agent Prompt - Paper-Style Preprocessing + Phase 2 CATNet + Paper UDA

You are a coding agent working inside the existing ECG thesis repository:

```text
ecg_thesis/
```

Read these first:

```text
../RULES.md
docs/scientific_repo_order.md
docs/current_status_report.md
docs/project_context_handoff.md
docs/phase2_dann_baseline_plan.md
```

## Current Context

The current best accepted thesis result is:

```text
Phase 2 CATNet1D + DANN on INCART held-out
Protocol: MIT-BIH -> INCART, N/S/V, beat-level [N, 1, 250]
Macro-F1: 0.6783
S-F1:     0.4178
```

This task is not to replace that accepted baseline. Implement a separate
experiment branch inspired by:

```text
Cross-Database and Cross-Channel Electrocardiogram Arrhythmia Heartbeat
Classification Based on Unsupervised Domain Adaptation
by Md Niaz Imtiaz and Naimul Khan
```

Requested design:

```text
Preprocessing: paper-style preprocessing from the attached prompt
Feature extractor: CATNet1D-style feature extractor from Phase 2
Classifier: paper-style bi-classifier adapted to CATNet1D features
Training method: paper-style DRO + classifier discrepancy + centroid UDA
Labels: keep thesis N/S/V for this run
```

Paper-specific choices intentionally adapted for this repo:

```text
Keep from the paper:
  bandpass 3-20 Hz
  resampling to 256 Hz
  source-mean-RR heartbeat segmentation
  three time features as saved preprocessing metadata
  minority duplication factors
  source cluster organization
  confident target pseudo-label selection by confidence and prototype distance
  compacting loss
  separating loss
  inter-domain cluster discrepancy loss
  running combined loss

Changed by user request:
  use Phase 2 CATNet1D-style feature extractor instead of paper residual CNN
  adapt the paper bi-classifier to CATNet1D embeddings
  keep N/S/V instead of paper N/V/S/F
```

Do not implement a broad ablation plan. Do not suggest a new repository
structure. Focus on making this one branch runnable and well reported.

## Non-Negotiable Rules

Keep the main label space:

```text
N/S/V
```

Do not switch the main pipeline to `N/S/V/F` or `N/S/V/F/Q`. If `F` is needed
later, it must be a separate explicitly labeled paper-reproduction branch.

Target labels:

```text
Do not use target labels during adaptation training.
Do not tune pseudo-label thresholds using INCART held-out labels.
Do not select checkpoints using INCART held-out metrics.
Use target held-out labels only for final evaluation and analysis.
```

Always report:

```text
Accuracy
Macro-F1
per-class precision/recall/F1
confusion matrix
S precision
S recall
S-F1
S -> N
S -> V
N -> V
V -> N
```

## Experiment Scope

Implement a separate branch, for example:

```text
Phase 2P: CATNet1D + paper-style preprocessing + centroid UDA
```

Use unique config, checkpoints, metrics, and report names:

```text
configs/phase2p_catnet_paper_uda.yaml
checkpoint prefix: phase2p_catnet_paper_uda
report: outputs/phase2p_catnet_paper_uda_report.md
```

Do not overwrite accepted Phase 2 checkpoints or processed files.

## Data Protocol

Source:

```text
MIT-BIH labeled source train
MIT-BIH source test for in-domain evaluation
```

Target:

```text
INCART unlabeled/adaptation data
INCART held-out/test data for final evaluation
```

Use the current repo target split:

```text
INCART adapt/unlabeled: I01-I50
INCART held-out/test:  I51-I75
```

## Required Preprocessing

Create separate processed files under a new directory, for example:

```text
data/processed/phase2p_paper_preprocess/
```

Do not modify the existing Phase 1/2 `.npz` files in place.

### 1. Load Raw WFDB Data

Reuse existing helpers where possible:

```text
src/data/physionet.py
src/data/splits.py
```

Lead selection:

```text
MIT-BIH: MLII preferred, II fallback
INCART: II preferred, MLII fallback
fallback_lead_index: 0
```

### 2. Label Mapping

Use thesis `N/S/V` mapping:

```text
N, L, R, e, j -> N
A, a, J, S    -> S
V, E          -> V
```

Ignore other symbols.

### 3. Bandpass Signal Denoising

Use the paper-style filtering from the attached prompt:

```text
bandpass filter
passband: 3-20 Hz
stable digital filter implementation
```

Make this configurable:

```yaml
denoise:
  method: bandpass
  low_hz: 3.0
  high_hz: 20.0
  order: 4
```

Do not implement DWT/wavelet denoising for this prompt. The attached prompt's
preprocessing uses a 3-20 Hz bandpass filter.

### 4. Resampling

After filtering, resample all datasets to:

```text
256 Hz
```

Use a stable FIR/polyphase method such as `scipy.signal.resample_poly`.
Correctly transform R-peak indices after resampling and verify that beat
segmentation remains aligned.

Make this configurable:

```yaml
resampling:
  target_fs: 256
  method: resample_poly
```

### 5. Heartbeat Segmentation

Segment beats using R-peak as reference:

```text
fixed-length heartbeat segment
segment length derived from source-domain mean RR after resampling
output waveform shape: [N, 1, segment_length]
```

Default behavior:

```text
compute arithmetic mean RR interval from MIT-BIH source R-peaks after resampling
use that value to define a fixed segment length
use R-peak as the reference point for extraction
ensure every segment has identical length
```

Also allow explicit window seconds for reproducibility:

```yaml
segmentation:
  mode: source_mean_rr
  explicit_window_seconds: null
  explicit_left_seconds: 0.30
  explicit_right_seconds: 0.40
```

If implementing exact source-mean-RR segmentation is ambiguous, preserve the
paper spirit and make the chosen left/right convention explicit in
`config_json` and the preprocessing summary.

Skip boundary-crossing beats and count them in the preprocessing summary.

### 6. Time Feature Extraction

For each beat, compute the 3 temporal features described in the attached
prompt:

```text
current RR interval
average pre-RR interval from the beginning of the record to the current beat
average of the last 8 pre-RR intervals before the current beat
```

Save them as:

```text
time_features: [N, 3]
time_feature_names
```

Normalize time features using MIT-BIH source-train statistics only:

```text
fit mean/std on MIT-BIH source train
apply the same stats to MIT-BIH test, INCART adapt, and INCART held-out
```

Important:

```text
Use these time features in the paper-style classifier. CATNet1D should extract
deep waveform features, then concatenate the 3 normalized time features before
the final classifier layers, matching the paper's feature-fusion idea.
```

### 7. Beat Normalization

Apply:

```text
per-beat z-score
```

### 8. Imbalance Handling

Implement paper-style imbalance handling for this branch:

```text
duplicate minority samples using paper-style augmentation factors
```

Paper wording:

```text
V, S, and F data are duplicated by factors of 2, 5, and 10 and incorporated
into the datasets.
```

These are additional duplicate copies. Therefore the final total multipliers
are:

```text
V total multiplier = 3
S total multiplier = 6
F total multiplier = 11
```

Because this branch keeps thesis labels `N/S/V`, use additional copies:

```text
N: add 0 copies, total multiplier 1
V: add 2 copies, total multiplier 3
S: add 5 copies, total multiplier 6
```

Do not use the paper's `F x10` factor unless the user explicitly requests a
separate `N/S/V/F` branch.

Constraints:

```text
For thesis-safe adaptation, apply duplication only to the MIT-BIH source
fit/training split.
Do not duplicate MIT-BIH test.
Do not duplicate INCART adaptation/test files.
Do not use target held-out labels.
Do not duplicate source validation records if a fit/val split exists.
```

Note:

```text
The paper reports augmentation by the same factors for all three databases.
This repo branch does not duplicate target evaluation data because it would
alter the held-out test distribution and complicate comparison with the thesis
protocol.
```

Save a separate oversampled source-train file and log:

```text
class counts before duplication
class counts after duplication
oversampling factors
no_target_labels_used = true
```

Do not use SMOTE-Tomek or RandomUnderSampler in this prompt.

### 9. Saved Data Format

Save `.npz` files compatible with repo dataset patterns:

```python
{
    "x": np.ndarray,        # [N, 1, segment_length]
    "time_features": np.ndarray,  # [N, 3]
    "time_feature_names": np.ndarray,
    "y": np.ndarray,        # [N]
    "record": np.ndarray,
    "symbol": np.ndarray,
    "sample": np.ndarray,
    "fs": np.ndarray,
    "domain": np.ndarray,
    "lead_index": np.ndarray,
    "lead_name": np.ndarray,
    "class_names": np.ndarray,  # ["N", "S", "V"]
    "config_json": np.ndarray,
}
```

## Model Requirements

Use a CATNet1D-style feature extractor from Phase 2, not the paper's residual
CNN. Use the paper-style bi-classifier adapted to the CATNet embedding.

Feature extractor:

```text
Input: [B, 1, segment_length]
CATNet1D-style CNN + ChannelAttention + positional encoding + Transformer
Deep waveform embedding z: [B, 128]
```

Classifier:

```text
Use paper-style two parallel classifier heads.
Before each head's final classification layer, concatenate:
  CATNet embedding z [B, 128]
  normalized time_features [B, 3]

fused feature h: [B, 131]
classifier_1(h) -> logits1 [B, 3]
classifier_2(h) -> logits2 [B, 3]
final_logits = average(logits1, logits2)
```

Each classifier head should be an MLP compatible with the paper's idea of three
fully connected layers, but adapted to the smaller CATNet embedding dimension.
Make hidden dimensions configurable.

Implement classifier discrepancy:

```text
probabilities1 = softmax(logits1)
probabilities2 = softmax(logits2)
L_discrepancy = Euclidean/L2 distance between probabilities1 and probabilities2
```

Required model API:

```python
final_logits = model(x, time_features)
outputs = model(x, time_features, return_all=True)
final_logits, embedding = model(x, time_features, return_embedding=True)
embedding = model.forward_features(x)
model.embedding_dim == 128
model.classifier1 exists
model.classifier2 exists
```

If current CATNet1D already supports variable input length through dynamic
pooling, reuse it. If not, make the smallest compatible change without
breaking length-250 Phase 2 behavior.

Do not implement:

```text
cross-channel V5
ESTDB
N/S/V/F branch
```

## Paper-Inspired UDA Method

Implement the paper's DRO, bi-classifier discrepancy, and centroid/pseudo-label
UDA method adapted to a CATNet1D feature extractor.

### Stage 1: Source Pretraining

Train CATNet1D on the paper-style MIT-BIH source data:

```text
input: [N, 1, segment_length]
time_features: [N, 3]
labels: N/S/V
losses:
  DRO-applied weighted cross-entropy
  classifier discrepancy loss
optional source fit file: paper-style duplicated source train
checkpoint selection: source validation Macro-F1 only
```

Stage 1 objective:

```text
L_pretrain = L_DRO_weighted_CE + lambda_disc * L_classifier_discrepancy
```

Implement DRO for the weighted CE loss. Keep the DRO parameters configurable and
document the exact DRO formulation used in the report. If using a practical
group-DRO implementation, use source class labels as groups unless a more
paper-faithful grouping is clearly implemented.

Save:

```text
outputs/checkpoints/phase2p_catnet_source_best.pt
outputs/metrics/phase2p_catnet_source_train_summary.json
outputs/logs/phase2p_catnet_source_train_log.csv
```

### Stage 2: Initial Source Prototype Computation

Extract source embeddings and compute initial source prototypes:

```text
P_source_N
P_source_S
P_source_V
```

Also compute:

```text
source intra-class distance statistics
source prototype pairwise distances
N-S prototype distance
```

Save:

```text
outputs/prototypes/phase2p_source_prototypes.pt
outputs/metrics/phase2p_source_prototype_stats.json
```

### Stage 3: Source Cluster Organization

The paper performs an additional source-domain cluster organization stage before
target pseudo-label selection. Implement this stage with the CATNet
bi-classifier model.

Train on source data with:

```text
source weighted cross-entropy
source compacting loss
source separating loss
```

Definitions:

```text
Source compacting:
  pull source embeddings toward source prototype of the true class

Source separating:
  push different source class prototypes apart by a margin
  pay special attention to N vs S
```

The source classification term should use the bi-classifier final prediction
or the average of both classifier losses. Keep this choice configurable and
document it.

After this stage, recompute source prototypes and source intra-class distance
statistics. These recomputed values should be used for target pseudo-label
selection.

Save:

```text
outputs/checkpoints/phase2p_catnet_cluster_source_best.pt
outputs/metrics/phase2p_cluster_source_train_summary.json
outputs/prototypes/phase2p_source_prototypes_clustered.pt
outputs/metrics/phase2p_source_prototype_clustered_stats.json
```

### Stage 4: Target Pseudo-Label Selection

Run inference on INCART unlabeled/adaptation data only.

Select confident target pseudo-labels using:

```text
softmax confidence threshold
distance to predicted source prototype threshold
classifier discrepancy threshold
```

Use the paper's confident prediction criteria adapted to N/S/V:

```text
softmax score > 0.99
feature distance to predicted source centroid < mean source intra-cluster distance
classifier discrepancy < mean source classifier discrepancy
```

Use configurable defaults:

```yaml
pseudolabel:
  confidence_thresholds:
    N: 0.99
    S: 0.90
    V: 0.95
  distance_threshold:
    mode: source_intra_class_quantile
    quantile: 0.95
  discrepancy_threshold:
    mode: source_mean
  min_target_per_class: 20
```

Do not tune thresholds using INCART held-out labels.

Save:

```text
outputs/predictions/phase2p_target_pseudolabels.csv
outputs/metrics/phase2p_pseudolabel_stats.json
```

### Stage 5: Target Prototype Computation

Compute target prototypes from selected target pseudo-labels:

```text
P_target_N
P_target_S
P_target_V
```

If a class has fewer than `min_target_per_class`, skip target prototype
alignment for that class and log the reason.

Save:

```text
outputs/prototypes/phase2p_target_prototypes.pt
outputs/metrics/phase2p_target_prototype_stats.json
```

### Stage 6: Domain Adaptation

Train with source labeled data and target unlabeled data.

Use these losses:

```text
1. source classification loss with DRO-weighted CE
2. source compacting loss
3. selected-target compacting loss
4. inter-domain prototype discrepancy loss
5. cluster separating loss
6. running combined loss
7. optional classifier discrepancy regularization during adaptation
```

Definitions:

```text
Source compacting:
  pull source embeddings toward source prototype of true class

Target compacting:
  for selected target samples only, pull embeddings toward pseudo-label prototype

Inter-domain discrepancy:
  align source and target prototypes of the same class

Cluster separating:
  push different class prototypes apart by margin
  pay special attention to N vs S

Running combined loss:
  compute global average prototypes from source and target prototypes after
  cluster computation
  during adaptation, compute current batch class prototypes where available
  penalize deviation between current batch average prototypes and global
  average prototypes
```

Config:

```yaml
uda:
  use_dro_for_source_ce: true
  discrepancy_weight: 0.0
  compact_source_weight: 0.05
  compact_target_weight: 0.05
  inter_domain_weight: 0.05
  running_combined_weight: 0.05
  separation_weight: 0.01
  separation_margin: 1.0
  warmup_epochs: 3
```

Save:

```text
outputs/checkpoints/phase2p_catnet_paper_uda_best.pt
outputs/checkpoints/phase2p_catnet_paper_uda_latest.pt
outputs/logs/phase2p_catnet_paper_uda_train_log.csv
outputs/metrics/phase2p_catnet_paper_uda_train_summary.json
```

## Evaluation

Evaluate:

```text
MIT-BIH test
INCART held-out/test
```

Save:

```text
outputs/metrics/phase2p_source_mitbih_test_metrics.json
outputs/metrics/phase2p_source_incart_heldout_metrics.json
outputs/metrics/phase2p_uda_mitbih_test_metrics.json
outputs/metrics/phase2p_uda_incart_heldout_metrics.json
outputs/predictions/phase2p_uda_incart_heldout_predictions.csv
outputs/figures/phase2p_catnet_paper_uda/
```

Compare against:

```text
Phase 2 CATNet1D + DANN on INCART held-out
Macro-F1: 0.6783
S-F1:     0.4178
```

But state clearly:

```text
This branch changes preprocessing from [N,1,250] no-wavelet to paper-style
bandpass/resample-256/source-mean-RR segmentation/time-feature extraction and
source duplication, so it is not a one-variable apples-to-apples ablation.
```

## Required Report

Create:

```text
outputs/phase2p_catnet_paper_uda_report.md
```

The report must state:

```text
source dataset
target unlabeled/adaptation dataset
target held-out/test dataset
preprocessing details
whether 256 Hz signal resampling was used
whether source minority duplication was used
model architecture
pseudo-label classifier discrepancy threshold
DRO configuration
pseudo-label thresholds
prototype losses and weights
checkpoint selection rule
full vs smoke/debug run
```

The report must answer:

```text
Did paper-style preprocessing + CATNet improve over source-only?
Did centroid/pseudo-label UDA improve over source-only?
Did the bi-classifier discrepancy criteria select reliable target pseudo-labels?
What happened to S precision, S recall, and S-F1?
Did S -> N or S -> V decrease?
Were enough confident S target pseudo-labels selected?
Did prototype distances suggest better class-aware alignment?
Did N or V collapse?
```

## Implementation Instructions

Before coding, inspect existing files and produce a short implementation plan.
Then implement step by step without waiting for extra approval unless blocked.

Minimum expected additions:

```text
configs/phase2p_catnet_paper_uda.yaml
scripts/phase2p/01_prepare_paper_preprocess.py
scripts/phase2p/02_duplicate_source_minority.py
scripts/phase2p/03_train_source_catnet.py
scripts/phase2p/04_compute_prototypes_and_pseudolabels.py
scripts/phase2p/05_train_centroid_uda.py
scripts/phase2p/06_eval_centroid_uda.py
scripts/phase2p/07_make_report.py
```

Reusable helpers may be added under existing namespaces if needed:

```text
src/data/paper_preprocess.py
src/training/prototypes.py
src/training/pseudolabels.py
src/training/prototype_losses.py
```

Keep all paths, checkpoint prefixes, metrics, and reports unique.

## Warnings

Do not:

```text
use INCART held-out labels during adaptation
overwrite accepted Phase 2 checkpoints
claim direct apples-to-apples comparison with Phase 2 without noting
preprocessing changes
switch main thesis labels to N/S/V/F
implement cross-channel V5 or ESTDB
implement broad ablation matrix
implement DWT denoising or SMOTE-Tomek for this prompt
report only accuracy
```

Do:

```text
save pseudo-label statistics
save prototype distances
save classifier discrepancy statistics
save DRO configuration and group statistics
report Macro-F1 and per-class F1
inspect S behavior explicitly
save time-feature normalization statistics
```

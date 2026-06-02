# Repository Rules

This file summarizes the non-negotiable rules for code, configs, scripts,
notebooks, and reports in this repository.

The thesis package root is:

```text
ecg_thesis/
```

Do not modify external reference repositories when implementing thesis code.

## 1. Scientific Protocol

- The main thesis task is ECG arrhythmia classification under domain shift.
- Source domain is MIT-BIH.
- Target domain is INCART.
- The main label space is 3-class `N/S/V`.
- Do not silently switch to 5-class `N/S/V/F/Q`.
- Do not include class `F` in main Phase 1/2/3/5 training or evaluation unless an explicit ablation says so.
- Report Macro-F1 and per-class F1. Accuracy alone is never sufficient.
- Always inspect `S` behavior:
  - `S` precision
  - `S` recall
  - `S-F1`
  - `S -> N`
  - `S -> V`
- Do not claim success if `N` improves but `S` or `V` collapses.

## 2. Data Rules

- Raw data, processed `.npz`, checkpoints, logs, figures, predictions, W&B runs,
  and notebook runtime state are local/generated artifacts and must stay out of git.
- Processed files must preserve example order and metadata alignment.
- Every processed dataset must have matching lengths for inputs, labels, and metadata.
- Main beat-level Phase 1/2/3 data uses:

```text
x: [N, 1, 250]
classes: N/S/V
MIT-BIH lead: MLII
INCART lead: II
```

- Phase 3 RR data must add:

```text
rr_features: [N, 4]
rr_feature_names
```

- RR normalization statistics must be fitted on MIT-BIH train only.
- Never fit normalization on INCART held-out/test labels or combined source+target test distribution.
- Phase 5 MACNN data uses:

```text
x_macnn: [N, 1, 3, 128]
channel 0: ECG morphology
channel 1: pre-RR ratio repeated to length 128
channel 2: near-pre-RR ratio repeated to length 128
```

- Phase 5 INCART protocol is:

```text
target unlabeled/adapt: r_peak_time_sec < 300
target held-out/test:   r_peak_time_sec >= 300
```

- The first-5 split audit must pass:

```text
adapt max r_peak_time_sec < 300
heldout min r_peak_time_sec >= 300
```

## 3. Target Label Rules

- Target labels are never used during unsupervised adaptation training.
- DANN may use target inputs for domain loss only.
- DAEAC-style/prototype adaptation may use target pseudo-labels only.
- SHOT/source-free adaptation may use target unlabeled data only.
- Target held-out labels are used only for final evaluation and analysis.
- Do not select checkpoints using INCART held-out labels.
- Do not tune repeatedly on held-out metrics and then describe the held-out set as untouched.

## 4. Split Rules

- MIT-BIH train/test split must follow the repo split definitions in `src/data/splits.py`.
- Phase 2/3 INCART record-wise split must not be replaced by random beat split.
- Phase 5 uses first-5-minute INCART split by time, not random split.
- Do not compare results across different target protocols as direct apples-to-apples results.
  For example:

```text
Phase 2 record-wise INCART split != Phase 5 first-5-minute split
```

## 5. Training Rules

- Source-only models train on labeled MIT-BIH only.
- DANN trains on MIT-BIH labeled + INCART unlabeled.
- DAEAC-style adaptation trains with source labeled + target unlabeled/pseudo-labeled behavior as implemented.
- SHOT trains with source model checkpoint + target unlabeled only; source data is not used during SHOT adaptation.
- For source-only and DANN, checkpoint selection should use source validation Macro-F1.
- Early stopping patience currently applies to:

```text
MACNN source-only: 30 epochs without val_macro_f1 improvement
MACNN DANN:        30 epochs without source_val_macro_f1 improvement
```

- Current DAEAC-style and SHOT runs are fixed-epoch unless explicitly changed.
- Smoke runs must use `--epochs 1` and small `--max-*` sample flags, and their metrics are not thesis results.
- Metrics files with `max_samples` in their names are debug outputs, not final results.

## 6. Checkpoint Rules

- Every experiment variant must use a unique checkpoint prefix.
- Do not overwrite accepted checkpoints with ablations.
- Recommended Phase 5 prefix examples:

```text
macnn_se_source_only_lr001
macnn_se_dann
macnn_se_daeac
macnn_se_shot_im
macnn_se_shot_full_uniform
macnn_se_shot_full_prior
```

- When running notebooks in parallel, make checkpoint prefixes unique per notebook/run.
- Long cloud runs should copy or zip outputs to persistent storage before the session ends.

## 7. Evaluation Rules

- Evaluate every accepted model on:

```text
MIT-BIH test
INCART held-out/test
```

- Save metrics as JSON.
- Save predictions when scripts support it.
- Save confusion matrix data or figures when scripts support it.
- Report at least:

```text
Accuracy
Macro-F1
per-class precision
per-class recall
per-class F1
confusion matrix
```

- For DANN, do not claim success from low domain accuracy alone.
- For UMAP/embedding plots, treat them as qualitative diagnostics only.

## 8. Report Rules

- Generated reports stay in `outputs/`.
- Human-facing plans and handoffs stay in `ecg_thesis/docs/`.
- A report must clearly state:
  - data protocol
  - source dataset
  - target unlabeled/adaptation dataset
  - target test dataset
  - model/checkpoint used
  - whether the run is full or smoke/debug
- Do not mix final results and smoke results in the same table without labeling them.
- When changing protocol, report it explicitly.

## 9. Notebook Rules

- Notebooks must not require manual hidden state from previous sessions.
- Every notebook must:
  - clone or locate the repo explicitly
  - install only needed dependencies
  - copy/locate required data and checkpoints explicitly
  - create or state the config it uses
  - run a static/data check before training
  - use unique checkpoint prefixes for ablations
  - zip or copy outputs to persistent storage
- Kaggle notebooks should write outputs under `/kaggle/working`.
- Colab notebooks should copy important outputs back to Google Drive.
- Do not install huge/full dependency sets if minimal dependencies are enough.
- Notebook smoke cells must be clearly separated from full-run cells.

## 10. Code Rules

- Keep new implementation inside `ecg_thesis/`.
- Reuse existing helpers, config loading, dataset classes, and metric functions when possible.
- Do not introduce broad refactors while adding an experiment.
- Do not break existing phase scripts when adding new phases.
- Add new experiments as separate scripts/config sections unless explicitly replacing an old one.
- Model forward APIs should support existing evaluator patterns when possible:

```python
logits = model(x)
logits, embedding = model(x, return_embedding=True)
embedding = model.forward_features(x)
```

- MACNN_SE is a special case where default forward returns:

```python
features, logits = model(x)
```

Use helper functions such as `macnn_logits` and `macnn_features_logits` for MACNN training/evaluation.

## 11. Verification Rules

- Before a long cloud run, run:

```bash
cd ecg_thesis
python scripts/check_repo.py
```

- For Phase 5 MACNN data, run:

```bash
python scripts/phase5_macnn/10_check_phase5_static.py \
  --config configs/phase5_macnn_daeac.yaml \
  --check-files
```

- For notebooks, validate that required data and checkpoints exist before training.
- If a script cannot find required data/checkpoints, fail clearly rather than silently training from scratch unless that behavior is explicitly requested.

## 12. Interpretation Rules

- Macro-F1 can fluctuate because `S` is rare; a non-monotonic validation curve is not automatically a bug.
- Best checkpoint should be judged together with confusion matrix and per-class metrics.
- If a model improves target Macro-F1 but destroys `S`, it is not a thesis success.
- If target held-out is used for many ablation choices, describe the result as ablation/model selection rather than untouched final test performance.
- Negative results are useful if they clarify whether morphology, rhythm, class-aware alignment, or source-free adaptation is needed.

## 13. Current Preferred Phase 5 Practice

- Use `lr=0.001` for MACNN source-only unless intentionally testing the paper-style `0.005`.
- Use `epochs=100` for Phase 5 source-only, DANN, and DAEAC-style training unless a run explicitly overrides it.
- SHOT should start from an already trained source-only checkpoint such as:

```text
outputs/checkpoints/macnn_se_source_only_lr001_best.pt
```

- SHOT variants should not rerun source-only training.
- SHOT default first pass:

```text
epochs: 15
lr: 0.0003
variants:
  SHOT-IM
  SHOT full uniform
  SHOT full source-prior
```


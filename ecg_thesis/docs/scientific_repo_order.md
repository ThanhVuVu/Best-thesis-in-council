# Scientific Repository Order

This document defines the canonical reading and execution order for the ECG
thesis repository. It separates accepted thesis evidence from unfinished
experiments, scaffolds, and optional ablations.

## 1. Current Assessment

The repository is mostly organized scientifically at the code level:

```text
configs/       phase-specific YAML configs
scripts/       executable workflows grouped by phase
src/           reusable data/model/training/visualization code
docs/          human-facing plans and handoffs
outputs/       local generated reports, metrics, checkpoints, figures
data/          local processed data
```

The phase folders are already ordered by time:

```text
phase1 -> phase2 -> phase3 -> phase4a/b/c -> phase5_macnn
```

The main weakness is documentation order and scientific status labeling:

```text
Phase 2 is the strongest accepted result.
Phase 3 is implemented but current local results are smoke/debug only.
Phase 4A/B/C are advanced ECG-FM scaffolds without accepted final metrics.
Phase 5 is a separate first-5-minute MACNN protocol and should not be mixed
directly with the Phase 2/3 record-wise INCART protocol.
```

Therefore, the repo does not need broad file movement. It needs a canonical
scientific order for reading, running, and reporting.

## 2. Canonical Thesis Timeline

Use this as the official scientific timeline:

| Order | Phase | Scientific Role | Status |
|---:|---|---|---|
| 0 | Protocol and rules | Defines task, label space, split rules, reporting rules | Required |
| 1 | Phase 1 | Source-only ResNet1D baseline | Accepted baseline |
| 2 | Phase 2 | CATNet1D source-only and CATNet1D + DANN | Best accepted result |
| 3 | Phase 3 | RR-aware CATNet1D + DANN | Implemented, full run pending |
| 4 | Deep research / failure analysis | Literature and evidence plan for explaining failure modes | Planning |
| 5 | Phase 4A | ECG-FM LeadBridge source-only variants | Scaffolded, not accepted |
| 6 | Phase 4B | Source-free ECG-FM adaptation | Scaffolded, depends on Phase 4A |
| 7 | Phase 4C | Top-layer ECG-FM source-free adaptation | Scaffolded, depends on Phase 4A/B |
| 8 | Phase 5 | MACNN_SE first-5-minute protocol | Separate protocol, advanced branch |
| 9 | Phase 5 ablations | Dynamic focal, global RR, SMOTE-Tomek | Optional ablations |

## 3. Accepted Evidence Hierarchy

When writing the thesis, use this evidence hierarchy:

### Tier A: Accepted Results

Use as thesis-safe evidence:

```text
Phase 1 ResNet1D source-only
Phase 2 CATNet1D source-only
Phase 2 CATNet1D + DANN
```

Current best accepted result:

```text
Phase 2 CATNet1D + DANN on INCART held-out
Macro-F1: 0.6783
S-F1:     0.4178
```

### Tier B: Implemented But Not Final

Use as code-ready next work, not as final result:

```text
Phase 3 RR-aware CATNet1D + DANN
```

Reason:

```text
The local report currently records smoke/debug runs only.
Full train/eval needs to be rerun on real beat-level processed files.
```

### Tier C: Advanced Scaffolds

Use as optional extensions:

```text
Phase 4A ECG-FM LeadBridge
Phase 4B source-free ECG-FM adaptation
Phase 4C top-layer ECG-FM adaptation
```

Reason:

```text
The code/config exists, but accepted final metrics are not available locally.
```

### Tier D: Separate Protocol Or Ablation

Do not compare directly against Phase 2/3 as if it were the same target setup:

```text
Phase 5 MACNN_SE first-5-minute protocol
Phase 5 dynamic focal
Phase 5 global RR
Phase 5 SMOTE-Tomek
```

Reason:

```text
Phase 5 uses a first-5-minute INCART split, while Phase 2/3 use record-wise
INCART split. These protocols answer related but different questions.
```

## 4. Canonical Reading Order

For a new collaborator or future agent, read files in this order:

1. `../RULES.md`
2. `docs/scientific_repo_order.md`
3. `docs/current_status_report.md`
4. `docs/project_context_handoff.md`
5. `docs/phase2_dann_baseline_plan.md`
6. `docs/phase3_rr_dann_plan.md`
7. `docs/phase5_dynamic_focal_source_to_dann_handoff.md` only if working on Phase 5
8. Root `README.md` for commands

If present, also read:

```text
docs/preprocessing_methods_summary.md
docs/deep_research_failure_analysis_plan.md
```

These are supporting documents, not execution entrypoints.

## 5. Canonical Execution Order

### Step 0: Repo Check

```bash
cd ecg_thesis
python scripts/check_repo.py
```

### Step 1: Prepare And Validate Beat-Level Data

```bash
python scripts/phase1/00_prepare_data.py --config configs/phase1.yaml
python scripts/phase1/02_validate_processed_data.py --config configs/phase1.yaml
```

Expected files:

```text
data/processed/mitbih_train.npz
data/processed/mitbih_test.npz
data/processed/incart_test.npz
```

### Step 2: Phase 1 Baseline

Run only if reproducing the baseline:

```bash
python scripts/phase1/03_train_source_only.py --config configs/phase1.yaml
python scripts/phase1/04_eval_in_domain.py --config configs/phase1.yaml --checkpoint outputs/checkpoints/best.pt
python scripts/phase1/05_eval_cross_domain.py --config configs/phase1.yaml --checkpoint outputs/checkpoints/best.pt
python scripts/phase1/07_make_phase1_report.py --config configs/phase1.yaml
```

### Step 3: Phase 2 Best Accepted Baseline

```bash
python scripts/phase2/08_split_incart_unlabeled_test.py --config configs/phase2_dann.yaml
python scripts/phase2/09_train_source_only_catnet.py --config configs/phase2_dann.yaml
python scripts/phase2/10_eval_source_only_catnet.py --config configs/phase2_dann.yaml --checkpoint outputs/checkpoints/source_only_catnet_best.pt
python scripts/phase2/11_train_dann.py --config configs/phase2_dann.yaml
python scripts/phase2/12_eval_dann_in_domain.py --config configs/phase2_dann.yaml --checkpoint outputs/checkpoints/dann_best.pt
python scripts/phase2/13_eval_dann_cross_domain.py --config configs/phase2_dann.yaml --checkpoint outputs/checkpoints/dann_best.pt
python scripts/phase2/15_make_phase2_report.py --config configs/phase2_dann.yaml
```

### Step 4: Phase 3 Main Next Experiment

Run after Phase 2 data and split are available:

```bash
python scripts/phase3/02_prepare_rr_features.py --config configs/phase3_rr_dann.yaml
python scripts/phase3/03_train_source_only_catnet_rr.py --config configs/phase3_rr_dann.yaml
python scripts/phase3/04_eval_source_only_catnet_rr.py --config configs/phase3_rr_dann.yaml --checkpoint outputs/checkpoints/source_only_catnet_rr_best.pt --dataset both
python scripts/phase3/05_train_dann_rr.py --config configs/phase3_rr_dann.yaml
python scripts/phase3/06_eval_dann_rr_in_domain.py --config configs/phase3_rr_dann.yaml --checkpoint outputs/checkpoints/dann_rr_best.pt
python scripts/phase3/07_eval_dann_rr_cross_domain.py --config configs/phase3_rr_dann.yaml --checkpoint outputs/checkpoints/dann_rr_best.pt
python scripts/phase3/09_make_phase3_report.py --config configs/phase3_rr_dann.yaml
```

### Step 5: Advanced Branches

Run only after Phase 3 is understood or if thesis time allows:

```text
Phase 4A/B/C ECG-FM branch
Phase 5 MACNN first-5-minute branch
```

Do not mix Phase 2/3 record-wise results and Phase 5 first-5-minute results in
the same table without explicit protocol labels.

## 6. Reporting Order

Use this report structure:

1. Problem and protocol: MIT-BIH -> INCART, N/S/V, Macro-F1, per-class F1.
2. Data preprocessing: beat-level `[N,1,250]`, lead selection, z-score.
3. Phase 1: source-only baseline exposes S weakness.
4. Phase 2: CATNet1D + DANN is strongest accepted baseline.
5. Failure analysis: S behavior, confusion pairs, embedding diagnostics.
6. Phase 3: RR-aware hypothesis and final result if rerun fully.
7. Optional advanced results:
   - ECG-FM if final metrics exist.
   - MACNN first-5-minute if protocol is clearly separated.
8. Limitations and negative results.

## 7. Naming And Status Rules

Use these labels consistently:

```text
accepted
full-run pending
smoke/debug only
scaffolded
optional ablation
separate protocol
```

Never label a result as final if:

```text
the filename contains max_samples
the run used --epochs 1
the target held-out labels were used to select checkpoints
the protocol differs from the table it is compared against
```

## 8. Recommended Reorder Without Moving Code

Keep the existing code layout. It is already stable and path-compatible.

Recommended documentation order:

```text
docs/
  README.md
  scientific_repo_order.md
  current_status_report.md
  project_context_handoff.md
  phase2_dann_baseline_plan.md
  phase3_rr_dann_plan.md
  deep_research_failure_analysis_plan.md
  preprocessing_methods_summary.md
  phase5_dynamic_focal_source_to_dann_handoff.md
```

Recommended thesis result order:

```text
Phase 1 -> Phase 2 -> Failure Analysis -> Phase 3 -> Optional Phase 4/5
```

This keeps the repo scientifically readable without breaking imports,
notebooks, configs, or existing command paths.


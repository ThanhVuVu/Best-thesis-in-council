# Repository Coding Rules

This file defines the coding, file organization, branching, notebook, and
verification rules for this thesis repository. It intentionally focuses on
clean code and reproducible project structure, not detailed scientific method
rules.

The thesis package root is:

```text
ecg_thesis/
```

## 1. Scope

- Keep all thesis implementation inside `ecg_thesis/`.
- Do not modify external dataset folders or reference repositories when adding
  thesis code.
- Keep generated artifacts out of git:
  - raw or processed datasets
  - `.npz` files
  - checkpoints
  - logs
  - figures
  - predictions
  - W&B runs
  - notebook runtime state
  - Python cache folders
- Prefer small, local changes over broad refactors.
- Do not rename, move, or delete existing phase scripts unless explicitly
  requested.

## 2. Repository Layout

Use this layout as the stable project structure:

```text
ecg_thesis/
  README.md
  requirements.txt
  requirements_phase4a.txt

  configs/          # YAML configs, one phase/experiment/ablation per file
  data/             # local data root; generated files stay ignored
  docs/             # human-facing plans, handoffs, status notes
  notebooks/        # clean Colab/Kaggle driver notebooks
  outputs/          # generated checkpoints, metrics, logs, figures
  scripts/          # runnable experiment workflows
  src/              # reusable Python package code
```

The expected `src/` layout is:

```text
src/
  data/             # loading, preprocessing, splits, dataset classes
  models/           # model definitions
  training/         # training loops, losses, metrics, evaluation helpers
  utils/            # config, IO, seed, logging helpers
  visualization/    # reusable plotting code
```

The expected `scripts/` layout is:

```text
scripts/
  common.py
  scripts_eval_common.py
  check_repo.py
  phase1/
  phase2/
  phase2p/
  phase3/
  phase4a/
  phase4b/
  phase4c/
  phase5_macnn/
```

## 3. Scripts Versus Source Code

- Put reusable code in `ecg_thesis/src/`.
- Put executable workflow steps in `ecg_thesis/scripts/<phase>/`.
- A script should orchestrate a run; it should not contain large reusable model,
  dataset, metric, or training abstractions.
- If a helper is useful across at least two phases, move it to `src/`.
- If a helper is only useful for one phase, keep it in that phase folder.
- Existing `scripts/<phase>/common.py` files may remain as lightweight
  compatibility shims, but avoid adding large shared logic to them.

## 4. Phase And Ablation Organization

- Add new phases or ablations by adding new files or folders, not by editing
  accepted phase files.
- Prefer this pattern for a new branch experiment:

```text
ecg_thesis/configs/<experiment_name>.yaml
ecg_thesis/scripts/<experiment_name>/
  01_prepare.py
  02_train.py
  03_eval.py
  04_make_report.py
```

- Use numbered script names when the workflow is sequential.
- Keep numbers stable after they are introduced.
- Do not mix unrelated ablations in the same script.
- Do not place long training or evaluation logic directly in notebooks.

## 5. Branch And Merge Rules

- Assume multiple experiment branches may be developed from `main` and merged
  later.
- To reduce merge conflicts, new branch work should mostly add:
  - a new config file
  - a new script folder or new scripts inside an existing phase folder
  - optional new reusable modules in `src/`
  - optional documentation under `docs/`
- Avoid frequent edits to shared files:
  - `ecg_thesis/scripts/common.py`
  - `ecg_thesis/scripts/scripts_eval_common.py`
  - `ecg_thesis/src/training/train.py`
  - `ecg_thesis/src/training/train_dann.py`
  - `ecg_thesis/README.md`
  - root rule files
- If a shared file must change, keep the change narrow and backward compatible.
- Do not overwrite or remove APIs used by existing phase scripts.
- Prefer adding new functions over changing existing function behavior, unless
  the old behavior is clearly a bug.
- Do not merge smoke/debug assumptions into configs intended for full runs.

## 6. Config Rules

- Every experiment must have a config file or a clearly separated config
  section.
- Prefer one YAML file per phase, protocol, or ablation.
- Do not silently mutate an accepted config for a new experiment.
- Config paths should be relative to `ecg_thesis/` unless an absolute cloud path
  is explicitly required.
- Every experiment variant must use a unique `checkpoint_prefix`.
- Prefer experiment-specific output folders:

```yaml
paths:
  output_dir: outputs/<experiment_name>
```

- Keep generated outputs under `outputs/`.
- Keep processed data under `data/processed/`.
- Do not put machine-specific secrets, API keys, local credentials, or private
  absolute paths in committed configs.

## 7. Script Rules

- Scripts must be runnable from `ecg_thesis/`.
- Every run script should accept `--config`.
- Long-running scripts should support smoke-test flags when practical:
  - `--epochs 1`
  - `--max-fit-samples`
  - `--max-source-samples`
  - `--max-target-samples`
  - `--max-val-samples`
- Scripts should fail clearly when required data or checkpoints are missing.
- Evaluation scripts must not silently train from scratch.
- Training scripts should save:
  - best checkpoint
  - latest checkpoint
  - metrics or summary JSON
  - train log when practical
- Report-generation scripts should read existing outputs and write reports under
  `outputs/`.

### Weights & Biases tracking

- New train and eval workflows should support Weights & Biases tracking unless
  there is a clear reason not to.
- Prefer the existing helpers in `src/utils/wandb_logging.py`:
  - add `add_wandb_args(parser)` to runnable scripts when practical
  - call `apply_wandb_overrides(config, args)` after loading config
  - use `init_wandb(...)` for train/eval runs
  - use `log_eval_metrics(...)` for evaluation metrics
- Configs may define W&B under:

```yaml
logging:
  wandb:
    enabled: false
    project: ecg-thesis
    entity: null
    run_name: null
    group: null
    mode: null
    tags: []
    log_artifacts: false
```

- Default configs should keep `logging.wandb.enabled: false` unless the
  experiment is explicitly intended to run with W&B by default.
- Long cloud notebooks should expose cells or command flags for enabling W&B,
  including project, group, run name, mode, and tags.
- Training runs should log epoch-level losses, primary metrics, learning rates,
  and best-epoch summaries. Adaptation runs should also log adaptation-specific
  metrics such as domain accuracy, entropy, pseudo-label counts, or alignment
  losses when available.
- Evaluation runs should log accuracy, macro F1, per-class precision/recall/F1,
  and dataset or split names.
- Artifact logging is opt-in via `log_artifacts: true`; never rely on W&B as the
  only copy of checkpoints, metrics, predictions, or reports.
- Do not commit W&B runtime folders, local run cache, API keys, private entity
  names, or machine-specific credentials.

## 8. Source Code Rules

- Reuse existing dataset, metric, config, IO, seed, logging, and training helpers
  before adding new ones.
- Keep functions explicit and boring.
- Avoid hidden global state.
- Use `pathlib.Path` for paths.
- Use structured readers and writers for JSON, YAML, CSV, and NPZ files.
- Avoid ad hoc string parsing when a structured API is available.
- Keep imports stable and simple.
- Prefer type hints for new reusable functions.
- Keep comments short and useful; explain non-obvious logic, not obvious syntax.
- Avoid broad style-only rewrites while implementing an experiment.
- Preserve existing model forward APIs where possible, especially evaluator
  patterns such as:

```python
logits = model(x)
logits, embedding = model(x, return_embedding=True)
embedding = model.forward_features(x)
```

- If a model needs a special forward convention, provide helper functions so
  training and evaluation scripts do not duplicate fragile unpacking logic.

## 9. Notebook Rules

- Notebooks are driver interfaces for Colab or Kaggle, not the source of truth
  for training logic.
- A clean notebook may be committed if it is reproducible from a fresh session.
- Notebook outputs, checkpoints, temporary files, and hidden runtime state must
  stay out of git.
- Every committed notebook should:
  - clone or locate the repo explicitly
  - install only required dependencies
  - locate or copy required datasets/checkpoints explicitly
  - run a static or data check before long training
  - call repo scripts rather than duplicate large code blocks
  - use a unique checkpoint prefix for ablations
  - save or zip important outputs before the cloud session ends
- Kaggle notebooks should write generated files under `/kaggle/working`.
- Colab notebooks should copy important outputs to Google Drive or another
  persistent location.
- Separate smoke-test cells from full-run cells.

## 10. Gitignore And Artifact Rules

- Keep raw datasets, processed arrays, checkpoints, predictions, figures, logs,
  W&B runs, and caches ignored.
- If notebooks should be versioned, track only clean notebooks and keep
  `.ipynb_checkpoints/` ignored.
- Do not commit:
  - `*.npz`
  - `*.pt`
  - `*.pth`
  - `*.ckpt`
  - `outputs/`
  - `data/processed/`
  - `wandb/`
  - `.ipynb_checkpoints/`
- If a generated result becomes thesis evidence, summarize it in `docs/` or a
  report note instead of committing bulky artifacts.

## 11. Documentation Rules

- Use `ecg_thesis/docs/` for human-facing plans, handoffs, status notes, and
  protocol explanations.
- Use `ecg_thesis/README.md` for stable run commands and high-level project
  orientation.
- Do not turn README into a scratchpad for branch-specific notes.
- Branch-specific or experimental notes should go into a dedicated file under
  `docs/`.
- Generated reports should stay under `outputs/`.
- If a script, config, or notebook becomes a canonical workflow, document the
  command needed to run it.

## 12. Verification Rules

- Before committing or launching a long cloud run, run:

```bash
cd ecg_thesis
python scripts/check_repo.py
```

- New Python files must parse successfully.
- New YAML configs must load successfully.
- New scripts should be smoke-testable when practical.
- Do not claim a workflow is cloud-ready unless it has clear path handling,
  dependency setup, data checks, and output persistence.
- If verification cannot be run because data, checkpoints, dependencies, or GPU
  access are missing, state that clearly in the handoff or final response.

## 13. Agent Behavior Rules

- Read this file before making structural or code changes.
- Inspect the relevant existing phase before adding a new one.
- Follow the local style already present in nearby files.
- Make the smallest change that solves the task cleanly.
- Protect user changes in the working tree; do not revert unrelated edits.
- Prefer adding isolated experiment files over editing shared files.
- After edits, report:
  - files changed
  - verification run
  - any verification not run and why

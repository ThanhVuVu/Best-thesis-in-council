# Phase 6 PLAN 2 — Pseudo-label Filtering on Kaggle

## Scientific contract

The workflow runs four independent ablations: no filtering, global confidence,
global confidence plus normalized entropy, and class-specific confidence plus
normalized entropy. All runs start from the same source-selected
`daeac_base_best.pt`. Adaptation uses target inputs without labels, and the best
checkpoint is selected only by source-validation Macro-F1.

Target DS2-after-five-minute and optional external labels are evaluated only
after training. They must not be used to tune thresholds, select variants, or
resume from a preferred epoch. Clinical consistency is intentionally deferred:
the current DAEAC NPZ files contain RR ratios but not the required morphology
keys.

## Kaggle setup

Enable a GPU and attach inputs containing exactly one copy of:

```text
mitdb_ds1_daeac.npz
mitdb_ds2_first5_unlabeled_daeac.npz
mitdb_ds2_daeac.npz
daeac_base_best.pt
```

`incart_all_daeac.npz` and `svdb_all_daeac.npz` are optional. Open
`notebooks/phase6_daeac_pseudo_filter_kaggle.ipynb`, set `REPO_URL` and
`BRANCH`, then run cells in order. The notebook copies immutable inputs into
`/kaggle/working/Best-thesis-in-council/ecg_thesis`; it never writes under
`/kaggle/input` and never silently trains a missing baseline.

The preparation cell derives `mitdb_ds2_after5_daeac.npz` from the complete DS2
file and verifies it is disjoint from the first-five-minute adaptation set.

## Smoke and full runs

The notebook always runs repository checks, unit tests, strict protocol checks,
and isolated one-epoch smoke tests first. Smoke outputs live under
`/kaggle/working/smoke` and are not full experiment evidence.

After smoke tests pass, set:

```python
RUN_FULL = True
```

For interrupted runs, attach the previous latest checkpoints and fill:

```python
RESUME_CHECKPOINTS = {
    "confidence_entropy": "/kaggle/input/<dataset>/daeac_pseudo_filter_confidence_entropy_latest.pt",
}
```

Resume restores the model, optimizer, scheduler, prototype bank, history, and
empty/all-N safety streaks. To enable W&B, set `ENABLE_WANDB=True`; authentication
must be provided through Kaggle Secrets, never committed to the repository.

## Outputs

Each variant writes best/latest checkpoints, resolved config, CSV train log,
metrics, predictions, confusion matrices, and safety diagnostics under its
unique `outputs/phase6_daeac_pseudo_filter_<variant>` directory. The final cells
write the comparison report and archive to:

```text
/kaggle/working/phase6_daeac_pseudo_filter_report/
/kaggle/working/phase6_daeac_pseudo_filter_bundle.zip
```

The comparison includes target metrics only as post-training descriptions.
Interpret entropy, acceptance, all-N, reliability, beta, and skipped prototype
updates together; do not rank variants using target rows.

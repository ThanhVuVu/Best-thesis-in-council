# Documentation Index

This folder keeps human-facing project notes. Generated reports stay in
`outputs/` because they depend on local runs and are ignored by git.

## Core Context

- `scientific_repo_order.md` - canonical reading, execution, and reporting
  order; use this first when deciding what result is accepted vs pending.
- `project_context_handoff.md` - compact project state, key decisions, and
  known results.
- `current_status_report.md` - latest repo inspection, phase-by-phase status,
  trusted results, local-output caveats, and next steps.
- `phase2_dann_baseline_plan.md` - CATNet1D and DANN baseline plan.
- `phase3_rr_dann_plan.md` - RR/context follow-up plan built on Phase 2.
- Phase 5 MACNN/DAEAC-style implementation is documented in the root
  `README.md` and configured by `configs/phase5_macnn_daeac.yaml`.

## Recommended Reading Order

Read in this order when resuming the project:

```text
../RULES.md
scientific_repo_order.md
current_status_report.md
project_context_handoff.md
phase2_dann_baseline_plan.md
phase3_rr_dann_plan.md
```

Read Phase 5 notes only when working on the separate MACNN first-5-minute
branch:

```text
phase5_dynamic_focal_source_to_dann_handoff.md
```

## Local Generated Reports

When available, scripts write reports to:

```text
outputs/phase2_dann_report.md
outputs/phase3_rr_dann_report.md
outputs/phase4a_ecgfm_leadbridge_report.md
outputs/phase4b_sourcefree_ecgfm_leadbridge_report.md
outputs/phase4c_ecgfm_top2_sourcefree_report.md
outputs/phase5_macnn_daeac_report.md
```

Those files are intentionally local-only. Copy final tables into thesis notes or
the handoff document once the run is accepted.

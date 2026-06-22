# Phase 6 DAEAC Paper-Faithful Kaggle Guide

This workflow assumes the DAEAC preprocessed `.npz` files already contain a
DAEAC input array with shape `[N, 1, 3, 128]` or `[N, 3, 128]`, labels in paper
order `N=0, S=1, V=2, F=3`, and optional `class_names=["N","S","V","F"]`.

## Expected Kaggle Layout

```text
/kaggle/working/Best-thesis-in-council/ecg_thesis
/kaggle/input/<your-dataset>/*.npz
```

Copy the `.npz` files into:

```text
ecg_thesis/data/processed/phase6_daeac_paper/
```

Default filenames:

```text
mitdb_ds1_daeac.npz
mitdb_ds2_first5_unlabeled_daeac.npz
mitdb_ds2_daeac.npz
```

The provided Kaggle dataset can also include optional files such as:

```text
mitdb_all_daeac.npz
incart_all_daeac.npz
svdb_all_daeac.npz
```

The default config runs the MITDB DS1 -> DS2 paper experiment. For all five
domain-pair runs, use `configs/phase6_daeac_pair_*.yaml`; the complete lifecycle
and split protocol are in `docs/phase6_daeac_adaptation_implementation.md`.

The default config uses `input_key: auto`, which detects common keys such as
`x_daeac`, `x_macnn`, `x`, `X`, `inputs`, `data`, `samples`, or `beats`. Edit
`configs/phase6_daeac_paper.yaml` only if you want to force one specific key.

## Smoke Run

```bash
python scripts/check_repo.py
python scripts/phase6_daeac_paper/00_validate_data.py --config configs/phase6_daeac_paper.yaml
python scripts/phase6_daeac_paper/01_train_base.py --config configs/phase6_daeac_paper.yaml --epochs 1 --max-source-samples 512 --max-val-samples 512
python scripts/phase6_daeac_paper/02_adapt_uda.py --config configs/phase6_daeac_paper.yaml --epochs 1 --max-source-samples 512 --max-target-samples 512 --max-val-samples 512
```

## Full Run

```bash
python scripts/phase6_daeac_paper/01_train_base.py --config configs/phase6_daeac_paper.yaml
python scripts/phase6_daeac_paper/02_adapt_uda.py --config configs/phase6_daeac_paper.yaml
python scripts/phase6_daeac_paper/03_eval.py --config configs/phase6_daeac_paper.yaml --checkpoint outputs/phase6_daeac_paper/checkpoints/daeac_base_latest.pt --method-name daeac_base --dataset target
python scripts/phase6_daeac_paper/03_eval.py --config configs/phase6_daeac_paper.yaml --checkpoint outputs/phase6_daeac_paper/checkpoints/daeac_uda_latest.pt --method-name daeac_uda --dataset target
python scripts/phase6_daeac_paper/04_make_report.py --config configs/phase6_daeac_paper.yaml
```

## Save Outputs

```bash
zip -r /kaggle/working/phase6_daeac_paper_outputs.zip outputs/phase6_daeac_paper
```

Tips:

- Run the smoke cells before the full 300 + 300 epoch run.
- Keep generated files under `/kaggle/working` so they can be downloaded.
- If pseudo-label counts are all zero, inspect the source-only checkpoint and
  thresholds before launching a long run.
- Validation intentionally fails if data appears to use the reference repo class
  order `N,V,S,F`; convert labels before training.

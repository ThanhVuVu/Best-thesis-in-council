# Phase 6 DAEAC Center-MK-MMD Adaptation

## Goal

This experiment ports the MK-MMD idea from the reference `DAN` Caffe repo into
the existing DAEAC adaptation pipeline without changing the DAEAC architecture
or pseudo-label training loop.

The original attached MK-MMD specification proposed applying MK-MMD to
`GAP_embed` and other intermediate ASPP-SE features. For this phase, that is
intentionally not implemented. Instead, MK-MMD replaces only the distance used
inside DAEAC cluster-aligning loss:

```text
old: L_align = mean_c || C_s^c - C_t^c ||_2
new: L_align = mean_c MK-MMD({C_s^c}, {C_t^c})
```

All other DAEAC losses remain unchanged:

```text
L_total = L_cls + beta1 * L_align + beta2 * (L_sep + L_comp)
```

Target labels are not used during adaptation, gamma setup, checkpoint
selection, or threshold tuning.

## DAN Reference Used

The `DAN` folder is a Caffe implementation. Its reusable concepts are:

- `MMDLoss` layer.
- Gaussian multi-kernel MMD.
- DAN defaults: `kernel_num=5`, `kernel_mul=2.0`, `fix_gamma=false`.
- Linear-time unbiased MMD can be negative for minibatches.

The Caffe/CUDA code is not reused directly. The PyTorch implementation lives in:

```text
src/training/mk_mmd.py
```

Because DAEAC cluster alignment compares one source center and one target center
per class, this implementation uses the exact single-center RKHS distance:

```text
MK-MMD_center(c) = mean_u [2 - 2 * exp(-||C_s^c - C_t^c||^2 / gamma_u)]
```

## Strategies

### Strategy A: DAN-default Center-MK-MMD

Config:

```text
configs/phase6_daeac_mkmmd_center_a.yaml
```

Uses:

```yaml
kernel_num: 5
kernel_mul: 2.0
gamma_mode: adaptive_from_valid_center_pairs
```

### Strategy B: Wider-band Center-MK-MMD

Config:

```text
configs/phase6_daeac_mkmmd_center_b.yaml
```

Uses a wider bandwidth grid:

```yaml
kernel_num: 17
kernel_mul: 1.41421356237
gamma_mode: adaptive_from_valid_center_pairs
```

### Strategy C: Fixed-gamma Center-MK-MMD

Config:

```text
configs/phase6_daeac_mkmmd_center_c.yaml
```

Computes gamma once from initial source/target centers after loading the base
checkpoint:

```yaml
kernel_num: 5
kernel_mul: 2.0
gamma_mode: fixed_from_initial_centers
```

## Commands

Smoke run:

```bash
python scripts/phase6_daeac_paper/02_adapt_uda.py \
  --config configs/phase6_daeac_mkmmd_center_a.yaml \
  --epochs 1 \
  --max-source-samples 512 \
  --max-target-samples 512 \
  --max-val-samples 512
```

Full runs:

```bash
python scripts/phase6_daeac_paper/02_adapt_uda.py --config configs/phase6_daeac_mkmmd_center_a.yaml
python scripts/phase6_daeac_paper/02_adapt_uda.py --config configs/phase6_daeac_mkmmd_center_b.yaml
python scripts/phase6_daeac_paper/02_adapt_uda.py --config configs/phase6_daeac_mkmmd_center_c.yaml
```

Evaluate MITDB target:

```bash
python scripts/phase6_daeac_paper/03_eval.py \
  --config configs/phase6_daeac_mkmmd_center_a.yaml \
  --checkpoint outputs/phase6_daeac_mkmmd_center_a/checkpoints/daeac_mkmmd_center_a_latest.pt \
  --method-name daeac_mkmmd_center_a \
  --dataset target
```

Evaluate external targets:

```bash
python scripts/phase6_daeac_paper/03_eval.py \
  --config configs/phase6_daeac_mkmmd_center_a.yaml \
  --checkpoint outputs/phase6_daeac_mkmmd_center_a/checkpoints/daeac_mkmmd_center_a_latest.pt \
  --method-name daeac_mkmmd_center_a \
  --dataset external
```

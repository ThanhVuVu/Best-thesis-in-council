# Phase 6 DAEAC Data Preprocess Context

This document records the preprocessing protocol used to create the external
DAEAC-ready files currently stored outside git at:

```text
D:\DAEAC\daeac
```

These `.npz` files are generated artifacts. They must not be committed. The
goal of this note is to keep enough context in `main` for a future run to
understand, validate, and reproduce the same data contract.

## Files

The external data folder contains:

```text
incart_all_daeac.npz
mitdb_all_daeac.npz
mitdb_ds1_daeac.npz
mitdb_ds2_daeac.npz
mitdb_ds2_first5_unlabeled_daeac.npz
svdb_all_daeac.npz
```

Phase 6 configs expect these files to be copied into:

```text
ecg_thesis/data/processed/phase6_daeac_paper/
```

Random-balance and SMOTE-Tomek variants then create additional source-only
balanced files under:

```text
ecg_thesis/data/processed/phase6_daeac_balance/
```

## Artifact Manifest

Observed file metadata from `D:\DAEAC\daeac`:

| File | Shape of `x` | Records | Class counts N/S/V/F |
|---|---:|---:|---:|
| `mitdb_ds1_daeac.npz` | `[50987, 1, 3, 128]` | 22 | 45841 / 944 / 3788 / 414 |
| `mitdb_ds2_daeac.npz` | `[49675, 1, 3, 128]` | 22 | 44231 / 1837 / 3219 / 388 |
| `mitdb_ds2_first5_unlabeled_daeac.npz` | `[8175, 1, 3, 128]` | 22 | 7313 / 229 / 533 / 100 |
| `mitdb_all_daeac.npz` | `[100662, 1, 3, 128]` | 44 | 90072 / 2781 / 7007 / 802 |
| `incart_all_daeac.npz` | `[175712, 1, 3, 128]` | 75 | 153538 / 1959 / 19996 / 219 |
| `svdb_all_daeac.npz` | `[184321, 1, 3, 128]` | 78 | 162168 / 12193 / 9937 / 23 |

Every file stores:

```text
x
y
record
symbol
sample
r_peak_sample
r_peak_sample_360hz
r_peak_time_sec
fs_original
lead_index
lead_name
pre_rr_ratio
near_pre_rr_ratio
class_names
class_to_id_json
config_json
```

The `config_json` embedded in each file is the source-of-truth preprocess
manifest for that artifact.

## Label Mapping

The artifacts use the paper-faithful 4-class AAMI order:

```text
N = 0
S = 1
V = 2
F = 3
```

The embedded mapping is:

```json
{
  "F": 3,
  "N": 0,
  "S": 1,
  "V": 2
}
```

Symbols are mapped as:

```json
{
  "A": "S",
  "E": "V",
  "F": "F",
  "J": "S",
  "L": "N",
  "N": "N",
  "R": "N",
  "S": "S",
  "V": "V",
  "a": "S",
  "e": "N",
  "j": "N"
}
```

Symbols outside this map are ignored.

## Record Splits

### MIT-BIH DS1

`mitdb_ds1_daeac.npz` contains:

```text
101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124,
201, 203, 205, 207, 208, 209, 215, 220, 223, 230
```

### MIT-BIH DS2

`mitdb_ds2_daeac.npz` contains:

```text
100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212,
213, 214, 219, 221, 222, 228, 231, 232, 233, 234
```

`mitdb_ds2_first5_unlabeled_daeac.npz` is a first-five-minute subset of the same
DS2 record list. Its labels are stored for auditing, but adaptation code must
load it through `DAEACTargetUnlabeledDataset`, which intentionally does not
return labels.

### MIT-BIH All

`mitdb_all_daeac.npz` is the union of DS1 and DS2.

### INCART

`incart_all_daeac.npz` contains:

```text
I01 through I75
```

### SVDB

`svdb_all_daeac.npz` contains:

```text
800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812,
820, 821, 822, 823, 824, 825, 826, 827, 828, 829,
840, 841, 842, 843, 844, 845, 846, 847, 848, 849,
850, 851, 852, 853, 854, 855, 856, 857, 858, 859,
860, 861, 862, 863, 864, 865, 866, 867, 868, 869,
870, 871, 872, 873, 874, 875, 876, 877, 878, 879,
880, 881, 882, 883, 884, 885, 886, 887, 888, 889,
890, 891, 892, 893, 894
```

## Signal And Lead Selection

Each artifact uses a single selected lead.

Preferred leads:

```json
{
  "mitdb": ["MLII", "II"],
  "incart": ["II", "MLII"],
  "svdb": ["MLII", "II", "ECG1"]
}
```

If a preferred lead is unavailable, the fallback lead index is:

```text
0
```

Observed selected leads in the generated artifacts:

```text
MIT-BIH: MLII, lead_index 0
INCART: II, lead_index 1
SVDB: ECG1, lead_index 0
```

Original sampling rates stored in the files:

```text
MIT-BIH: 360 Hz
INCART: 257 Hz
SVDB: 128 Hz
```

All databases are unified to:

```text
360 Hz
```

The embedded artifact notes state that FFT-based `scipy.signal.resample` is used
for sampling-rate unification and segment resizing.

## Per-Beat Sample Construction

Each valid annotated beat becomes one tensor:

```text
[1, 3, 128]
```

The stacked 3 channels are:

```text
channel 0: heartbeat morphology segment
channel 1: repeated pre_RR_ratio
channel 2: repeated near_pre_RR_ratio
```

The final batch tensor is:

```text
[N, 1, 3, 128]
```

### Boundary And RR Rules

The first beat is skipped because the previous RR interval is undefined.

The artifact notes also state:

```text
The first beat and the next beat without previous RR history are skipped because RR ratio denominators are undefined.
```

When implementing regeneration, make the handling of beat `i=1` explicit and
match the stored artifact counts during validation.

### Segment Window

For beat `i` with current R peak `R_i` and previous R peak `R_{i-1}`, the raw
segment window is:

```text
start = R_{i-1} + 0.14 seconds
end   = R_i     + 0.28 seconds
```

The segment is then resized to:

```text
128 samples
```

### Morphology Normalization

The morphology segment is normalized per segment:

```text
x = (x - mean(x)) / std(x)
```

Use a small epsilon guard for near-zero standard deviation.

### RR Ratio Channels

The two RR-derived scalar values are repeated across all 128 positions.

The artifact names are:

```text
pre_rr_ratio
near_pre_rr_ratio
```

The matching project convention in `src/data/macnn_preprocess.py` is:

```text
median_rr = median(diff(r_peaks) / fs) for the record
pre_rr_ratio = ((R_i - R_{i-1}) / fs) / median_rr
near_pre_rr_ratio = ((R_{i-1} - R_{i-2}) / fs) / median_rr
```

If `R_{i-2}` is unavailable, regeneration must follow the same boundary skip
policy used by the artifacts rather than silently substituting an arbitrary
value.

## First-Five-Minute Target Subset

`mitdb_ds2_first5_unlabeled_daeac.npz` is the adaptation subset:

```text
r_peak_time_sec < 300.0
```

It is generated from MIT-BIH DS2 records. Labels remain in the file for audit
and diagnostics, but adaptation scripts must ignore them.

## Intended Phase 6 Usage

Base Phase 6 config:

```text
configs/phase6_daeac_paper.yaml
```

Main split usage:

```text
source_train/source_eval: mitdb_ds1_daeac.npz
target_unlabeled:        mitdb_ds2_first5_unlabeled_daeac.npz
target_test:             mitdb_ds2_daeac.npz
external_targets:        incart_all_daeac.npz, svdb_all_daeac.npz
```

MCC and Hybrid MKMMD MCC reuse the same data contract:

```text
configs/phase6_daeac_mcc.yaml
configs/phase6_daeac_hybrid_mkmmd_mcc.yaml
```

Diagnostics reuse the same data and checkpoint contract:

```text
configs/phase6_daeac_diagnostics.yaml
scripts/phase6_daeac_diagnostics/
```

## Copying The External Artifacts Into The Repo Runtime

Local Windows example:

```powershell
New-Item -ItemType Directory -Force ecg_thesis\data\processed\phase6_daeac_paper
Copy-Item D:\DAEAC\daeac\*.npz ecg_thesis\data\processed\phase6_daeac_paper\
```

Kaggle example:

```bash
mkdir -p data/processed/phase6_daeac_paper
cp /kaggle/input/<dataset-name>/daeac/*.npz data/processed/phase6_daeac_paper/
```

## Validation Commands

From `ecg_thesis/`:

```bash
python scripts/check_repo.py
python scripts/phase6_daeac_paper/00_validate_data.py --config configs/phase6_daeac_paper.yaml
```

For MCC protocol checks:

```bash
python scripts/phase6_daeac_paper/00_validate_data.py --config configs/phase6_daeac_hybrid_mkmmd_mcc.yaml
python scripts/phase6_daeac_mcc/00_check_protocol.py --config configs/phase6_daeac_mcc.yaml --strict
```

Expected validation shape:

```text
x shape [N, 1, 3, 128]
class_names ["N", "S", "V", "F"]
labels in 0..3
```

## Source Balancing From The Phase 6 Artifacts

Random balance variants generate source-only balanced files from
`mitdb_ds1_daeac.npz`:

```bash
python scripts/phase6_daeac_balance/01_prepare_random_oversample.py \
  --config configs/phase6_daeac_hybrid_mkmmd_random_balance.yaml
```

Default multipliers:

```text
N: 1
S: 5
V: 2
F: 10
```

SMOTE-Tomek variants use:

```bash
python scripts/phase6_daeac_balance/02_prepare_smotetomek.py \
  --config configs/phase6_daeac_hybrid_mkmmd_smotetomek.yaml
```

Generated balanced files are artifacts and must not be committed.

## Git Rules

Commit this context, configs, scripts, and reusable source code.

Do not commit:

```text
*.npz
*.pt
*.pth
*.ckpt
data/processed/
outputs/
predictions/
figures/
logs/
wandb/
__pycache__/
```

# Phase 6 DAEAC adaptation implementation

## Paper-faithful epoch lifecycle

1. Load pretrained `F` and `H`, construct `h`, copy `H -> h`, and keep `h` outside the optimizer.
2. Run the complete unlabeled target partition through `F+h`. Acceptance uses strict class thresholds: `N > 0.999`; `S`, `V`, and `F > 0.99`.
3. Freeze accepted target positions and pseudo labels in an immutable snapshot. Compute initial source and target global centers from the source set and this snapshot.
4. Train one epoch using source batches and the frozen pseudo-target loader. The target loader drives the epoch, so every accepted target sample is visited once. Source batches cycle only when necessary.
5. Update source/target centers with EMA coefficient `c=0.1`, then use `(Cs+Ct)/2` as the mixed center.
6. Optimize `Lcls + 0.1*Lalign + 0.1*(Lsep+Lcomp)`. Original DAEAC losses use sum reduction. Cluster losses depend only on features and centers, so `H` receives only the `Lcls` gradient while `F` receives the complete objective.
7. At the epoch boundary copy `H -> h`, re-run the complete target partition, and freeze the snapshot for the next epoch.

Adam uses learning rate `0.005`, weight decay `0.0001`, and a `0.99` learning-rate multiplier every 200 optimizer iterations. Adaptation runs for 300 epochs with source and target batch sizes of 256.

## Domain pairs

| Config | Source pool | Unlabeled adaptation | Held-out target test |
|---|---|---|---|
| `phase6_daeac_pair_ds1_ds2.yaml` | All MITDB DS1 | DS2 `<300s` | All DS2 |
| `phase6_daeac_pair_ds1_incart.yaml` | All MITDB DS1 | All INCART | All INCART |
| `phase6_daeac_pair_ds1_svdb.yaml` | All MITDB DS1 | All SVDB | All SVDB |
| `phase6_daeac_pair_mitbih_incart.yaml` | All MITDB DS1+DS2 | All INCART | All INCART |
| `phase6_daeac_pair_mitbih_svdb.yaml` | All MITDB DS1+DS2 | All SVDB | All SVDB |

Here `MITBIH` means the complete 44-record `DS1+DS2` source pool after excluding paced records 102, 104, 107, and 217. Every source record participates in both pretraining and adaptation. Four DS1 records are also used as an overlapping source-only monitoring subset for checkpoint selection; they are not removed from source training.

For DS1 to DS2, `00_prepare_after5.py` creates the `<300s` unlabeled adaptation view and evaluation uses the complete DS2 file. Cross-dataset runs use the complete target file for both adaptation inputs and final evaluation. In both scenarios `DAEACTargetUnlabeledDataset` hides target labels during training; labels are read only by the evaluation loader.

## Standard implementation versus ablations

The five `phase6_daeac_pair_*.yaml` files use `prototype_bank.usage=logging_only` and `prototype_losses.mode=legacy`; therefore the original EMA centers and original DAEAC objective drive optimization. Prototype-bank, pseudo-filter, pair-margin, MK-MMD, MCC, and adversarial configs remain explicitly named extensions and should not be reported as the standard DAEAC implementation.

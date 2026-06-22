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
| `phase6_daeac_pair_ds1_ds2.yaml` | MITDB DS1 | DS2 `<300s` | DS2 `>=300s` |
| `phase6_daeac_pair_ds1_incart.yaml` | MITDB DS1 | INCART `<300s` | INCART `>=300s` |
| `phase6_daeac_pair_ds1_svdb.yaml` | MITDB DS1 | SVDB `<300s` | SVDB `>=300s` |
| `phase6_daeac_pair_mitbih_incart.yaml` | MITDB DS1+DS2 | INCART `<300s` | INCART `>=300s` |
| `phase6_daeac_pair_mitbih_svdb.yaml` | MITDB DS1+DS2 | SVDB `<300s` | SVDB `>=300s` |

Here `MITBIH` means the 44-record `DS1+DS2` source pool after excluding paced records 102, 104, 107, and 217. For source checkpoint selection, four DS1 records remain source-validation records; the other 18 DS1 and all 22 DS2 records form the MITBIH fit partition.

`00_prepare_after5.py` creates both target partitions from the full target NPZ and rejects any sample overlap. Target labels are never returned by the adaptation dataset.

## Standard implementation versus ablations

The five `phase6_daeac_pair_*.yaml` files use `prototype_bank.usage=logging_only` and `prototype_losses.mode=legacy`; therefore the original EMA centers and original DAEAC objective drive optimization. Prototype-bank, pseudo-filter, pair-margin, MK-MMD, MCC, and adversarial configs remain explicitly named extensions and should not be reported as the standard DAEAC implementation.

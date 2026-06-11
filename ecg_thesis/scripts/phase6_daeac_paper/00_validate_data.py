from __future__ import annotations

import argparse

import torch

from common import cfg_path, load_phase1_config
from src.data.daeac_dataset import DAEACDataset, DAEACTargetUnlabeledDataset, inspect_daeac_npz
from src.models.daeac_paper import DAEACNetwork


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase6_daeac_paper.yaml")
    parser.add_argument("--check-files", action="store_true", default=True)
    args = parser.parse_args()
    config = load_phase1_config(args.config)
    input_key = str(config["data"].get("input_key", "x_daeac"))
    label_key = str(config["data"].get("label_key", "y"))
    class_names = list(config["data"]["class_names"])

    model = DAEACNetwork(num_classes=int(config["data"]["num_classes"]))
    features, logits, probs = model(torch.zeros(2, 1, 3, 128), return_logits=True)
    assert tuple(features.shape) == (2, 256), features.shape
    assert tuple(logits.shape) == (2, 4), logits.shape
    assert tuple(probs.shape) == (2, 4), probs.shape
    assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-6)
    print("DAEAC model forward OK: features=(2,256), logits=(2,4), probs=(2,4)")

    for key, require_labels in (
        ("source_train", True),
        ("source_eval", True),
        ("target_unlabeled", False),
        ("target_test", True),
    ):
        if key not in config["data"]:
            continue
        summary = inspect_daeac_npz(
            cfg_path(config, "data", key),
            input_key=input_key,
            label_key=label_key,
            class_names=class_names,
            require_labels=require_labels,
        )
        print(f"{key}: {summary}")

    target = DAEACTargetUnlabeledDataset(
        cfg_path(config, "data", "target_unlabeled"),
        input_key=input_key,
        label_key=label_key,
        class_names=class_names,
    )
    item = target[0]
    if not (isinstance(item, tuple) and len(item) == 2):
        raise AssertionError("Target unlabeled dataset must return only (x, index).")
    source = DAEACDataset(cfg_path(config, "data", "source_train"), input_key=input_key, label_key=label_key, class_names=class_names)
    print(f"Target unlabeled safety OK. Source samples={len(source)}, target_unlabeled samples={len(target)}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse

import numpy as np
import torch

from common import cfg_path, load_phase1_config
from src.models.macnn_se import MACNN_SE


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase5_macnn_daeac.yaml")
    parser.add_argument("--check-files", action="store_true")
    args = parser.parse_args()
    config = load_phase1_config(args.config)
    model = MACNN_SE(num_classes=3)
    for height in (3, 6):
        features, logits = model(torch.zeros(2, 1, height, 128))
        assert tuple(features.shape) == (2, 256), features.shape
        assert tuple(logits.shape) == (2, 3), logits.shape
    print("MACNN_SE forward OK for H=3 and H=6: features=(2,256), logits=(2,3)")

    if args.check_files:
        for key in ("source_train", "source_test", "target_unlabeled", "target_test"):
            path = cfg_path(config, "data", key)
            data = np.load(path, allow_pickle=True)
            shape = data["x_macnn"].shape
            assert len(shape) == 4 and shape[1] == 1 and shape[2] >= 3 and shape[3] == 128, (key, shape)
            assert len(data["x_macnn"]) == len(data["y"]), key
            print(f"{key}: x_macnn={data['x_macnn'].shape}, y={data['y'].shape}")
        adapt = np.load(cfg_path(config, "data", "target_unlabeled"), allow_pickle=True)
        heldout = np.load(cfg_path(config, "data", "target_test"), allow_pickle=True)
        threshold = float(config["data"]["target_adapt_seconds"])
        assert float(adapt["r_peak_time_sec"].max()) < threshold
        assert float(heldout["r_peak_time_sec"].min()) >= threshold
        print("INCART first-5 split audit OK")


if __name__ == "__main__":
    main()

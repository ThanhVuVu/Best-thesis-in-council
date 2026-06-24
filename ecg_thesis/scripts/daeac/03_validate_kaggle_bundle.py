from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    args = parser.parse_args()
    bundle = Path(args.bundle)
    manifest = json.loads((bundle / "record_split_manifest.json").read_text(encoding="utf-8"))
    for domain, details in manifest["domains"].items():
        role = details["role"]
        for split, audit in details["audit"]["splits"].items():
            path = bundle / f"{domain}_{split}.npz"
            if not path.exists():
                raise FileNotFoundError(path)
            with np.load(path, allow_pickle=True) as data:
                records = sorted(set(np.asarray(data["record"]).astype(str)))
                if records != sorted(audit["records"]):
                    raise ValueError(f"{path}: record manifest mismatch")
                should_have_labels = role == "source" or split == "test"
                if ("y" in data.files) != should_have_labels:
                    raise ValueError(f"{path}: label exposure policy mismatch")
    print({"bundle": str(bundle), "status": "valid", "domains": sorted(manifest["domains"])})


if __name__ == "__main__":
    main()

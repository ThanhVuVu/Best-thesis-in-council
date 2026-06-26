from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.data.record_splits import audit_record_split, balanced_record_split


class RecordSplitTests(unittest.TestCase):
    def test_split_is_deterministic_disjoint_and_complete(self) -> None:
        counts = {f"r{i:02d}": [100 + i, i % 5, (i * 3) % 7, 1 if i in {2, 7, 13} else 0] for i in range(20)}
        sizes = {"train": 12, "val": 4, "test": 4}
        first = balanced_record_split(counts, sizes, seed=42, trials=200)
        second = balanced_record_split(counts, sizes, seed=42, trials=200)
        self.assertEqual(first, second)
        audit = audit_record_split(counts, first, sizes)
        self.assertTrue(audit["valid"])
        self.assertEqual(audit["record_overlap"], [])

    def test_invalid_sizes_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            balanced_record_split({"a": [1, 0, 0, 0]}, {"train": 0}, seed=42)

    def test_drop_classes_filters_sample_aligned_arrays_only(self) -> None:
        module = _load_prepare_bundle_module()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.npz"
            np.savez_compressed(
                path,
                x=np.arange(5 * 2).reshape(5, 2),
                y=np.asarray([0, 3, 1, 3, 2], dtype=np.int64),
                record=np.asarray(["a", "b", "c", "d", "e"], dtype=object),
                class_names=np.asarray(["N", "S", "V", "F"], dtype=object),
                config_json=np.asarray("{}", dtype=object),
            )

            removed = module._drop_classes(path, ["F"], output_class_names=["N", "S", "V"])

            self.assertEqual(removed, 2)
            with np.load(path, allow_pickle=True) as data:
                self.assertEqual(data["y"].tolist(), [0, 1, 2])
                self.assertEqual(data["record"].tolist(), ["a", "c", "e"])
                self.assertEqual(data["class_names"].tolist(), ["N", "S", "V"])
                self.assertEqual(str(data["class_to_id_json"].tolist()), '{"N": 0, "S": 1, "V": 2}')
                self.assertIn('"class_names": ["N", "S", "V"]', str(data["config_json"].tolist()))


def _load_prepare_bundle_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "daeac" / "02_prepare_record_split_bundle.py"
    spec = importlib.util.spec_from_file_location("prepare_record_split_bundle", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    unittest.main()

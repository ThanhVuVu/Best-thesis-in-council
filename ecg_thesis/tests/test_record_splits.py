from __future__ import annotations

import unittest

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


if __name__ == "__main__":
    unittest.main()

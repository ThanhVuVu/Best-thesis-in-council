from __future__ import annotations

import unittest

import numpy as np
from sklearn.metrics import v_measure_score

from src.training.v_measure_validation import aggregate_v_measure, ericsson_v_measure


class VMeasureValidationTests(unittest.TestCase):
    def test_matches_sklearn_and_is_deterministic(self) -> None:
        source_logits = np.asarray([[5, 0], [0, 5]], dtype=np.float32)
        target_logits = np.asarray([[4, 0], [0, 4], [3, 0], [0, 3]], dtype=np.float32)
        source_labels = np.asarray([0, 1])
        first = ericsson_v_measure(source_logits, source_labels, target_logits, num_classes=2, random_state=42)
        second = ericsson_v_measure(source_logits, source_labels, target_logits, num_classes=2, random_state=42)
        self.assertEqual(first["v_measure"], second["v_measure"])
        self.assertTrue(np.array_equal(first["cluster_labels"], second["cluster_labels"]))
        self.assertAlmostEqual(first["v_measure"], v_measure_score(first["reference_labels"], first["cluster_labels"]))
        self.assertAlmostEqual(first["v_measure"], first["v_measure_manual"])

    def test_aggregate_excludes_per_sample_arrays(self) -> None:
        result = ericsson_v_measure(np.eye(2), np.arange(2), np.eye(2), num_classes=2)
        aggregate = aggregate_v_measure(result)
        self.assertIn("v_measure", aggregate)
        self.assertNotIn("cluster_labels", aggregate)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.data.daeac_dataset import DAEACDataset, DAEACTargetUnlabeledDataset, inspect_daeac_npz


class RTDDAEACPhase1ARRRowsTests(unittest.TestCase):
    def test_real_rr_mode_keeps_stored_rows_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path, x = _write_npz(Path(directory) / "beats.npz")
            ds = DAEACDataset(path)

            self.assertEqual(ds.rr_mode, "real")
            np.testing.assert_allclose(ds.x[:, 0, 1, :], x[:, 0, 1, :])
            np.testing.assert_allclose(ds.x[:, 0, 2, :], x[:, 0, 2, :])
            self.assertFalse(np.allclose(ds.x[:, 0, 1, :], 1.0))
            self.assertFalse(np.allclose(ds.x[:, 0, 2, :], 1.0))
            ds.close()

    def test_neutralized_legacy_mode_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path, _ = _write_npz(Path(directory) / "beats.npz")
            ds = DAEACDataset(path, rr_mode="neutralized_legacy")

            self.assertEqual(ds.rr_mode, "neutralized_legacy")
            self.assertTrue(np.allclose(ds.x[:, 0, 1, :], 1.0))
            self.assertTrue(np.allclose(ds.x[:, 0, 2, :], 1.0))
            ds.close()

    def test_target_unlabeled_keeps_real_rr_without_exposing_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path, x = _write_npz(Path(directory) / "beats.npz")
            ds = DAEACTargetUnlabeledDataset(path)
            item = ds[0]

            self.assertIsInstance(item, tuple)
            self.assertEqual(len(item), 2)
            np.testing.assert_allclose(item[0].numpy()[0, 1, :], x[0, 0, 1, :])
            np.testing.assert_allclose(item[0].numpy()[0, 2, :], x[0, 0, 2, :])
            ds.close()

    def test_inspect_reports_rr_row_stats_and_neutralization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path, _ = _write_npz(Path(directory) / "beats.npz")
            real = inspect_daeac_npz(path)
            legacy = inspect_daeac_npz(path, rr_mode="neutralized_legacy")

            self.assertEqual(real["rr_mode"], "real")
            self.assertIn("row1_pre_rr_ratio", real["row_stats"])
            self.assertFalse(real["rr_rows_neutralized"])
            self.assertTrue(legacy["rr_rows_neutralized"])


def _write_npz(path: Path) -> tuple[Path, np.ndarray]:
    x = np.zeros((4, 1, 3, 128), dtype=np.float32)
    x[:, 0, 0, :] = np.linspace(-1.0, 1.0, 128, dtype=np.float32)
    x[:, 0, 1, :] = np.linspace(0.75, 1.25, 128, dtype=np.float32)
    x[:, 0, 2, :] = np.linspace(0.5, 1.5, 128, dtype=np.float32)
    y = np.asarray([0, 1, 2, 3], dtype=np.int64)
    np.savez_compressed(
        path,
        x=x,
        y=y,
        record=np.asarray(["101", "101", "102", "102"], dtype=object),
        class_names=np.asarray(["N", "S", "V", "F"], dtype=object),
    )
    return path, x


if __name__ == "__main__":
    unittest.main()

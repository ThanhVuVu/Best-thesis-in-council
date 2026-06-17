from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import resample

from src.data.daeac_dataset import DAEACDataset
from src.data.physionet import choose_lead, read_record


DATASET_FILENAMES = {
    "target": "mitdb_ds2_daeac.npz",
    "incart": "incart_all_daeac.npz",
    "svdb": "svdb_all_daeac.npz",
}


def dataset_key_to_raw_key(dataset_key: str) -> str:
    if dataset_key == "target":
        return "mitdb"
    return dataset_key


def load_labeled_daeac_dataset(
    processed_dir: str | Path,
    dataset_key: str,
    input_key: str = "auto",
    label_key: str = "y",
    class_names: list[str] | None = None,
) -> DAEACDataset:
    filename = DATASET_FILENAMES.get(dataset_key, dataset_key)
    path = Path(processed_dir) / filename
    if not path.exists():
        raise FileNotFoundError(f"DAEAC processed file not found for dataset={dataset_key}: {path}")
    return DAEACDataset(path, input_key=input_key, label_key=label_key, class_names=class_names, return_metadata=True)


def sample_ids(dataset_key: str, dataset: DAEACDataset, indices: list[int] | np.ndarray | None = None) -> list[str]:
    values = indices if indices is not None else range(len(dataset))
    ids: list[str] = []
    for idx in values:
        meta = dataset.metadata(int(idx))
        ids.append(make_sample_id(dataset_key, meta, int(idx)))
    return ids


def make_sample_id(dataset_key: str, metadata: dict[str, Any], idx: int) -> str:
    record = str(metadata.get("record", metadata.get("record_id", "")))
    sample = metadata.get("r_peak_sample", metadata.get("sample", idx))
    return f"{dataset_key}:{record}:{int(sample)}:{idx}"


def build_raw_cache_for_dataset(
    dataset_key: str,
    dataset: DAEACDataset,
    raw_dir: str | Path,
    preferred_leads: list[str],
    fallback_lead_index: int,
    cfg: dict[str, Any],
    max_samples: int | None = None,
) -> tuple[list[dict[str, Any]], np.ndarray | None]:
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw WFDB directory does not exist for {dataset_key}: {raw_path}")
    limit = min(len(dataset), int(max_samples)) if max_samples is not None else len(dataset)
    record_cache: dict[str, tuple[np.ndarray, float, int, str]] = {}
    record_dir_cache: dict[str, Path] = {}
    rows: list[dict[str, Any]] = []
    windows: list[np.ndarray] = []
    save_windows = bool(cfg.get("save_raw_windows", False))
    for idx in range(limit):
        meta = dataset.metadata(idx)
        record = str(meta.get("record", meta.get("record_id", "")))
        if record not in record_cache:
            record_dir = _resolve_record_dir(raw_path, record, record_dir_cache)
            wfdb_record = read_record(record_dir, record)
            signal = wfdb_record.p_signal
            if signal is None:
                raise ValueError(f"Record {record} has no physical signal")
            lead_idx, lead_name, _ = choose_lead(list(wfdb_record.sig_name), preferred_leads, int(fallback_lead_index))
            record_cache[record] = (np.asarray(signal[:, lead_idx], dtype=np.float32), float(wfdb_record.fs), int(lead_idx), str(lead_name))
        signal_1d, fs, lead_idx, lead_name = record_cache[record]
        r_peak = int(meta.get("r_peak_sample", meta.get("sample")))
        window = extract_raw_window(signal_1d, r_peak, fs, cfg)
        features = clinical_proxy_features(window, cfg)
        features.update(
            {
                "sample_id": make_sample_id(dataset_key, meta, idx),
                "dataset": dataset_key,
                "sample_index": int(idx),
                "record": record,
                "symbol": str(meta.get("symbol", "")),
                "class_id": int(dataset.y[idx]) if dataset.y is not None else -1,
                "r_peak_sample": r_peak,
                "r_peak_time_sec": float(meta.get("r_peak_time_sec", r_peak / max(fs, 1.0))),
                "fs_original": fs,
                "lead_index": int(lead_idx),
                "lead_name": lead_name,
                "pre_rr_ratio": float(dataset.x[idx, 0, 1, 0]),
                "near_pre_rr_ratio": float(dataset.x[idx, 0, 2, 0]),
            }
        )
        rows.append(features)
        if save_windows:
            windows.append(window.astype(np.float32))
    return rows, np.stack(windows).astype(np.float32) if windows else None


def _resolve_record_dir(raw_dir: Path, record: str, cache: dict[str, Path]) -> Path:
    if record in cache:
        return cache[record]
    direct = raw_dir / f"{record}.hea"
    if direct.exists():
        cache[record] = raw_dir
        return raw_dir
    matches = sorted(raw_dir.glob(f"**/{record}.hea"))
    if matches:
        cache[record] = matches[0].parent
        return matches[0].parent
    raise FileNotFoundError(
        f"Could not find WFDB record {record}.hea under {raw_dir}. "
        "Pass the directory that contains the .hea/.dat/.atr files, or a parent directory containing them."
    )


def extract_raw_window(signal: np.ndarray, r_peak_sample: int, fs: float, cfg: dict[str, Any]) -> np.ndarray:
    target_fs = int(cfg.get("target_fs", 360))
    pre = float(cfg.get("pre_r_sec", 0.35))
    post = float(cfg.get("post_r_sec", 0.45))
    start = int(round(r_peak_sample - pre * fs))
    end = int(round(r_peak_sample + post * fs))
    left_pad = max(0, -start)
    right_pad = max(0, end - len(signal))
    start = max(0, start)
    end = min(len(signal), end)
    window = signal[start:end].astype(np.float32)
    if left_pad or right_pad:
        window = np.pad(window, (left_pad, right_pad), mode="edge")
    expected_len = max(int(round((pre + post) * target_fs)), 2)
    if int(round(fs)) != target_fs or len(window) != expected_len:
        window = resample(window, expected_len).astype(np.float32)
    return zscore(window)


def clinical_proxy_features(window: np.ndarray, cfg: dict[str, Any]) -> dict[str, float]:
    x = np.asarray(window, dtype=np.float64)
    n = len(x)
    fs = float(cfg.get("target_fs", 360))
    pre_r_sec = float(cfg.get("pre_r_sec", 0.35))
    post_r_sec = float(cfg.get("post_r_sec", 0.45))
    r_idx = int(round(n * pre_r_sec / max(pre_r_sec + post_r_sec, 1.0e-8)))
    qrs_half = max(int(round(0.06 * fs)), 1)
    pre_len = max(int(round(0.20 * fs)), 1)
    post_len = max(int(round(0.28 * fs)), 1)
    qrs = x[max(0, r_idx - qrs_half) : min(n, r_idx + qrs_half)]
    pre = x[max(0, r_idx - pre_len) : r_idx]
    post = x[r_idx : min(n, r_idx + post_len)]
    slope = np.diff(x, prepend=x[0])
    qrs_slope = np.diff(qrs, prepend=qrs[0]) if len(qrs) else np.zeros(1)
    return {
        "raw_mean": float(np.mean(x)),
        "raw_std": float(np.std(x)),
        "raw_peak_to_peak": float(np.ptp(x)),
        "pre_r_energy": float(np.mean(pre**2)) if len(pre) else 0.0,
        "pre_r_peak_abs": float(np.max(np.abs(pre))) if len(pre) else 0.0,
        "qrs_energy_proxy": float(np.mean(qrs**2)) if len(qrs) else 0.0,
        "qrs_area_proxy": float(np.sum(np.abs(qrs))) if len(qrs) else 0.0,
        "qrs_peak_to_peak_proxy": float(np.ptp(qrs)) if len(qrs) else 0.0,
        "qrs_max_slope_proxy": float(np.max(np.abs(qrs_slope))) if len(qrs_slope) else 0.0,
        "post_r_energy": float(np.mean(post**2)) if len(post) else 0.0,
        "post_r_area": float(np.sum(np.abs(post))) if len(post) else 0.0,
        "full_max_slope": float(np.max(np.abs(slope))) if len(slope) else 0.0,
    }


def processed_input_features(dataset: DAEACDataset, indices: np.ndarray) -> dict[str, np.ndarray]:
    x = dataset.x[indices]
    morph = x[:, 0, 0, :]
    pre_rr = x[:, 0, 1, 0]
    near_pre_rr = x[:, 0, 2, 0]
    slope = np.diff(morph, axis=1, prepend=morph[:, :1])
    return {
        "processed_morph_mean": morph.mean(axis=1),
        "processed_morph_std": morph.std(axis=1),
        "processed_morph_peak_to_peak": np.ptp(morph, axis=1),
        "processed_morph_energy": np.mean(morph**2, axis=1),
        "processed_morph_max_slope": np.max(np.abs(slope), axis=1),
        "pre_rr_ratio": pre_rr,
        "near_pre_rr_ratio": near_pre_rr,
    }


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return ((values - float(values.mean())) / max(float(values.std()), 1.0e-8)).astype(np.float32)

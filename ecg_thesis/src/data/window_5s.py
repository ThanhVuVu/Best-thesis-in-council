from __future__ import annotations

import json
from collections import Counter
from math import gcd
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import butter, filtfilt, resample_poly
from tqdm import tqdm

from src.data.label_mapping import CLASS_NAMES, ID_TO_CLASS, map_symbol
from src.data.physionet import choose_lead, read_annotation, read_record
from src.utils.io import ensure_dir


def preprocess_5s_windows(
    raw_dir: str | Path,
    records: list[str],
    output_path: str | Path,
    dataset: str,
    config: dict[str, Any],
    force: bool = False,
) -> dict[str, Any]:
    output = Path(output_path)
    if output.exists() and not force:
        return {"output": str(output), "skipped": True, "reason": "processed file exists"}

    ensure_dir(output.parent)
    data_cfg = config["data"]
    lead_cfg = config["lead_selection"]
    target_fs = int(data_cfg["target_fs"])
    window_samples = int(data_cfg["window_samples"])
    half_window = window_samples // 2
    preferred = lead_cfg["preferred_leads"][dataset]
    fallback_index = int(lead_cfg.get("fallback_lead_index", 0))

    x_values = []
    y_values = []
    record_values = []
    symbol_values = []
    sample_values = []
    fs_values = []
    domain_values = []
    lead_index_values = []
    lead_name_values = []

    raw_symbol_counts: Counter[str] = Counter()
    mapped_counts: Counter[str] = Counter()
    ignored_counts: Counter[str] = Counter()
    skipped_boundary = 0
    selected_leads: Counter[str] = Counter()
    fallback_records: list[str] = []
    failures = []

    for rec in tqdm(records, desc=f"phase4a preprocess {dataset}"):
        try:
            wfdb_record = read_record(raw_dir, rec)
            ann = read_annotation(raw_dir, rec)
            signal = wfdb_record.p_signal
            if signal is None:
                raise ValueError(f"Record {rec} has no physical signal")
        except Exception as exc:
            failures.append({"record": rec, "error": repr(exc)})
            continue

        lead_idx, lead_name, used_fallback = choose_lead(list(wfdb_record.sig_name), preferred, fallback_index)
        if used_fallback:
            fallback_records.append(rec)
        selected_leads[lead_name] += 1

        fs = float(wfdb_record.fs)
        signal_1d = np.asarray(signal[:, lead_idx], dtype=np.float32)
        signal_1d = _replace_nonfinite(signal_1d)
        if data_cfg.get("bandpass", {}).get("enabled", True):
            signal_1d = _bandpass_filter(
                signal_1d,
                fs=fs,
                low_hz=float(data_cfg["bandpass"]["low_hz"]),
                high_hz=float(data_cfg["bandpass"]["high_hz"]),
                order=int(data_cfg["bandpass"]["order"]),
            )
        signal_target = _resample_to_target(signal_1d, fs=fs, target_fs=target_fs)

        for rpeak, symbol in zip(ann.sample, ann.symbol):
            raw_symbol_counts[symbol] += 1
            label = map_symbol(symbol)
            if label is None:
                ignored_counts[symbol] += 1
                continue
            center = int(round(float(rpeak) * target_fs / fs))
            start = center - half_window
            end = start + window_samples
            if start < 0 or end > len(signal_target):
                skipped_boundary += 1
                continue
            window = signal_target[start:end].astype(np.float32)
            window = _window_zscore(window)
            x_values.append(window[None, :])
            y_values.append(label)
            record_values.append(rec)
            symbol_values.append(symbol)
            sample_values.append(int(rpeak))
            fs_values.append(fs)
            domain_values.append(dataset)
            lead_index_values.append(int(lead_idx))
            lead_name_values.append(lead_name)
            mapped_counts[ID_TO_CLASS[label]] += 1

    if not x_values:
        raise ValueError(f"No 5s windows were extracted for {dataset} -> {output}")

    x = np.stack(x_values).astype(np.float32)
    y = np.asarray(y_values, dtype=np.int64)
    config_json = json.dumps(
        {
            "dataset": dataset,
            "records": records,
            "data": data_cfg,
            "lead_selection": lead_cfg,
        },
        sort_keys=True,
    )
    np.savez_compressed(
        output,
        x=x,
        y=y,
        record=np.asarray(record_values, dtype=object),
        symbol=np.asarray(symbol_values, dtype=object),
        sample=np.asarray(sample_values, dtype=np.int64),
        fs=np.asarray(fs_values, dtype=np.float32),
        domain=np.asarray(domain_values, dtype=object),
        lead_index=np.asarray(lead_index_values, dtype=np.int64),
        lead_name=np.asarray(lead_name_values, dtype=object),
        class_names=np.asarray(CLASS_NAMES, dtype=object),
        config_json=np.asarray(config_json, dtype=object),
    )
    return {
        "output": str(output),
        "skipped": False,
        "dataset": dataset,
        "records": records,
        "num_windows": int(len(y)),
        "x_shape": list(x.shape),
        "class_counts": dict(mapped_counts),
        "raw_symbol_counts": dict(raw_symbol_counts),
        "ignored_symbol_counts": dict(ignored_counts),
        "skipped_boundary": int(skipped_boundary),
        "selected_lead_counts": dict(selected_leads),
        "fallback_records": fallback_records,
        "failures": failures,
    }


def _replace_nonfinite(signal: np.ndarray) -> np.ndarray:
    if np.isfinite(signal).all():
        return signal
    finite = signal[np.isfinite(signal)]
    fill = float(np.median(finite)) if len(finite) else 0.0
    return np.nan_to_num(signal, nan=fill, posinf=fill, neginf=fill).astype(np.float32)


def _bandpass_filter(signal: np.ndarray, fs: float, low_hz: float, high_hz: float, order: int) -> np.ndarray:
    nyquist = 0.5 * fs
    high = min(high_hz / nyquist, 0.99)
    low = max(low_hz / nyquist, 1e-5)
    if not 0 < low < high < 1:
        return signal.astype(np.float32)
    b, a = butter(order, [low, high], btype="bandpass")
    return filtfilt(b, a, signal).astype(np.float32)


def _resample_to_target(signal: np.ndarray, fs: float, target_fs: int) -> np.ndarray:
    source_fs = int(round(fs))
    if source_fs == target_fs:
        return signal.astype(np.float32)
    divisor = gcd(source_fs, target_fs)
    up = target_fs // divisor
    down = source_fs // divisor
    return resample_poly(signal, up, down).astype(np.float32)


def _window_zscore(window: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = float(window.mean())
    std = float(window.std())
    if std < eps:
        return (window - mean).astype(np.float32)
    return ((window - mean) / std).astype(np.float32)

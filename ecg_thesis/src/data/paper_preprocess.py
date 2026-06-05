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

EPS = 1e-6
TIME_FEATURE_NAMES = np.asarray(["rr_current", "rr_pre_avg", "rr_pre8_avg"], dtype=object)


def compute_source_mean_rr_samples(
    raw_dir: str | Path,
    records: list[str],
    target_fs: int,
) -> int:
    rr_values = []
    for rec in records:
        ann = read_annotation(raw_dir, rec)
        wfdb_record = read_record(raw_dir, rec)
        fs = float(wfdb_record.fs)
        samples = np.asarray(ann.sample, dtype=np.float64)
        if len(samples) < 2:
            continue
        resampled = np.round(samples * float(target_fs) / max(fs, EPS))
        rr_values.extend(np.diff(resampled).tolist())
    if not rr_values:
        raise ValueError("Cannot compute source mean RR: no valid R-peak intervals found")
    return max(int(round(float(np.mean(rr_values)))), 8)


def preprocess_paper_records(
    raw_dir: str | Path,
    records: list[str],
    output_path: str | Path,
    dataset: str,
    config: dict[str, Any],
    segment_length: int,
    time_normalizer: dict[str, Any] | None = None,
    force: bool = False,
) -> tuple[dict[str, Any], np.ndarray]:
    output = Path(output_path)
    if output.exists() and not force:
        data = np.load(output, allow_pickle=True)
        return {"output": str(output), "skipped": True, "reason": "processed file exists"}, data["time_features"].astype(np.float32)

    ensure_dir(output.parent)
    data_cfg = config["paper_preprocess"]
    lead_cfg = config["lead_selection"]
    target_fs = int(data_cfg["resampling"]["target_fs"])
    low_hz = float(data_cfg["denoise"]["low_hz"])
    high_hz = float(data_cfg["denoise"]["high_hz"])
    filter_order = int(data_cfg["denoise"].get("order", 4))
    preferred = lead_cfg["preferred_leads"][dataset]
    fallback_index = int(lead_cfg.get("fallback_lead_index", 0))
    left = int(segment_length // 2)
    right = int(segment_length - left)

    x_values: list[np.ndarray] = []
    y_values: list[int] = []
    time_values: list[np.ndarray] = []
    record_values: list[str] = []
    symbol_values: list[str] = []
    sample_values: list[int] = []
    fs_values: list[float] = []
    domain_values: list[str] = []
    lead_index_values: list[int] = []
    lead_name_values: list[str] = []

    raw_symbol_counts: Counter[str] = Counter()
    mapped_counts: Counter[str] = Counter()
    ignored_counts: Counter[str] = Counter()
    selected_leads: Counter[str] = Counter()
    fallback_records: list[str] = []
    skipped_boundary = 0
    failures = []

    for rec in tqdm(records, desc=f"phase2p preprocess {dataset}"):
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
        signal_1d = _replace_nonfinite(np.asarray(signal[:, lead_idx], dtype=np.float32))
        filtered = _bandpass_filter(signal_1d, fs, low_hz, high_hz, filter_order)
        signal_target = _resample_to_target(filtered, fs, target_fs)
        rpeaks_target = np.round(np.asarray(ann.sample, dtype=np.float64) * float(target_fs) / max(fs, EPS)).astype(np.int64)
        raw_time_features = _time_features_from_rpeaks(rpeaks_target, target_fs)

        for i, (rpeak_orig, rpeak, symbol) in enumerate(zip(ann.sample, rpeaks_target, ann.symbol)):
            raw_symbol_counts[symbol] += 1
            label = map_symbol(symbol)
            if label is None:
                ignored_counts[symbol] += 1
                continue
            start = int(rpeak) - left
            end = int(rpeak) + right
            if start < 0 or end > len(signal_target):
                skipped_boundary += 1
                continue
            beat = _zscore(signal_target[start:end].astype(np.float32))
            x_values.append(beat[None, :].astype(np.float32))
            y_values.append(int(label))
            time_values.append(raw_time_features[i])
            record_values.append(str(rec))
            symbol_values.append(str(symbol))
            sample_values.append(int(rpeak_orig))
            fs_values.append(fs)
            domain_values.append(str(dataset))
            lead_index_values.append(int(lead_idx))
            lead_name_values.append(str(lead_name))
            mapped_counts[ID_TO_CLASS[label]] += 1

    if not x_values:
        raise ValueError(f"No paper-style beats were extracted for {dataset} -> {output}")

    x = np.stack(x_values).astype(np.float32)
    y = np.asarray(y_values, dtype=np.int64)
    raw_time = np.stack(time_values).astype(np.float32)
    if time_normalizer is None:
        time_features = raw_time
    else:
        time_features = apply_time_normalizer(raw_time, time_normalizer)

    config_json = json.dumps(
        {
            "dataset": dataset,
            "records": records,
            "segment_length": int(segment_length),
            "target_fs": target_fs,
            "left_samples": left,
            "right_samples": right,
            "paper_preprocess": data_cfg,
            "lead_selection": lead_cfg,
            "time_normalizer": time_normalizer,
        },
        sort_keys=True,
    )
    np.savez_compressed(
        output,
        x=x,
        time_features=time_features.astype(np.float32),
        raw_time_features=raw_time.astype(np.float32),
        time_feature_names=TIME_FEATURE_NAMES,
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
    summary = {
        "output": str(output),
        "skipped": False,
        "dataset": dataset,
        "records": records,
        "num_beats": int(len(y)),
        "x_shape": list(x.shape),
        "time_features_shape": list(time_features.shape),
        "segment_length": int(segment_length),
        "target_fs": target_fs,
        "class_counts": dict(mapped_counts),
        "raw_symbol_counts": dict(raw_symbol_counts),
        "ignored_symbol_counts": dict(ignored_counts),
        "skipped_boundary": int(skipped_boundary),
        "selected_lead_counts": dict(selected_leads),
        "fallback_records": fallback_records,
        "failures": failures,
    }
    return summary, raw_time


def fit_time_normalizer(time_features: np.ndarray) -> dict[str, Any]:
    values = np.asarray(time_features, dtype=np.float32)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std = np.where(std < EPS, 1.0, std)
    return {"mean": mean.tolist(), "std": std.tolist(), "feature_names": TIME_FEATURE_NAMES.tolist()}


def apply_time_normalizer(time_features: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    values = np.asarray(time_features, dtype=np.float32)
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    return ((values - mean) / np.maximum(std, EPS)).astype(np.float32)


def _time_features_from_rpeaks(rpeaks: np.ndarray, target_fs: int) -> np.ndarray:
    if len(rpeaks) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    rr = np.empty(len(rpeaks), dtype=np.float64)
    if len(rpeaks) > 1:
        diffs = np.diff(rpeaks.astype(np.float64)) / float(target_fs)
        fallback = float(np.median(diffs)) if len(diffs) else 1.0
        rr[0] = fallback
        rr[1:] = diffs
    else:
        rr[0] = 1.0
    out = np.zeros((len(rpeaks), 3), dtype=np.float32)
    for i in range(len(rpeaks)):
        hist = rr[: i + 1]
        out[i, 0] = float(rr[i])
        out[i, 1] = float(hist.mean())
        out[i, 2] = float(rr[max(0, i - 7) : i + 1].mean())
    return out


def _replace_nonfinite(signal: np.ndarray) -> np.ndarray:
    if np.isfinite(signal).all():
        return signal.astype(np.float32)
    finite = signal[np.isfinite(signal)]
    fill = float(np.median(finite)) if len(finite) else 0.0
    return np.nan_to_num(signal, nan=fill, posinf=fill, neginf=fill).astype(np.float32)


def _bandpass_filter(signal: np.ndarray, fs: float, low_hz: float, high_hz: float, order: int) -> np.ndarray:
    nyquist = 0.5 * fs
    low = max(float(low_hz) / nyquist, 1e-5)
    high = min(float(high_hz) / nyquist, 0.99)
    if not 0 < low < high < 1:
        return signal.astype(np.float32)
    b, a = butter(int(order), [low, high], btype="bandpass")
    return filtfilt(b, a, signal).astype(np.float32)


def _resample_to_target(signal: np.ndarray, fs: float, target_fs: int) -> np.ndarray:
    source_fs = int(round(fs))
    if source_fs == int(target_fs):
        return signal.astype(np.float32)
    divisor = gcd(source_fs, int(target_fs))
    up = int(target_fs) // divisor
    down = source_fs // divisor
    return resample_poly(signal, up, down).astype(np.float32)


def _zscore(values: np.ndarray) -> np.ndarray:
    mean = float(values.mean())
    std = float(values.std())
    if std < EPS:
        return (values - mean).astype(np.float32)
    return ((values - mean) / std).astype(np.float32)


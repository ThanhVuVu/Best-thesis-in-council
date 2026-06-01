from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from src.data.label_mapping import CLASS_NAMES, ID_TO_CLASS, map_symbol
from src.data.physionet import choose_lead, read_annotation, read_record
from src.utils.io import ensure_dir

EPS = 1e-6


def preprocess_macnn_records(
    raw_dir: str | Path,
    records: list[str],
    output_path: str | Path,
    dataset: str,
    config: dict[str, Any],
    split_rule: str = "all",
    force: bool = False,
) -> dict[str, Any]:
    output = Path(output_path)
    if output.exists() and not force:
        return {"output": str(output), "skipped": True, "reason": "processed file exists"}

    ensure_dir(output.parent)
    data_cfg = config["data"]
    macnn_cfg = config["macnn_input"]
    lead_cfg = config["lead_selection"]
    preferred = lead_cfg["preferred_leads"][dataset]
    fallback_index = int(lead_cfg.get("fallback_lead_index", 0))

    x_values: list[np.ndarray] = []
    y_values: list[int] = []
    record_values: list[str] = []
    symbol_values: list[str] = []
    sample_values: list[int] = []
    fs_values: list[float] = []
    rpeak_time_values: list[float] = []
    lead_index_values: list[int] = []
    lead_name_values: list[str] = []

    raw_symbol_counts: Counter[str] = Counter()
    mapped_counts: Counter[str] = Counter()
    ignored_counts: Counter[str] = Counter()
    selected_leads: Counter[str] = Counter()
    fallback_records: list[str] = []
    skipped_boundary = 0
    skipped_split = 0
    skipped_no_prev = 0

    for rec in tqdm(records, desc=f"preprocess macnn {dataset}:{split_rule}"):
        wfdb_record = read_record(raw_dir, rec)
        ann = read_annotation(raw_dir, rec)
        signal = wfdb_record.p_signal
        if signal is None:
            raise ValueError(f"Record {rec} has no physical signal")

        lead_idx, lead_name, used_fallback = choose_lead(list(wfdb_record.sig_name), preferred, fallback_index)
        if used_fallback:
            fallback_records.append(rec)
        selected_leads[lead_name] += 1
        signal_1d = signal[:, lead_idx].astype(np.float32)
        fs = float(wfdb_record.fs)
        samples = np.asarray(ann.sample, dtype=np.int64)
        symbols = list(ann.symbol)
        median_rr = _median_rr_seconds(samples, fs)

        for i, (rpeak, symbol) in enumerate(zip(samples, symbols)):
            raw_symbol_counts[symbol] += 1
            label = map_symbol(symbol)
            if label is None:
                ignored_counts[symbol] += 1
                continue
            rpeak_time = float(rpeak / fs)
            if not _include_time(rpeak_time, split_rule, float(data_cfg["target_adapt_seconds"])):
                skipped_split += 1
                continue
            if i == 0:
                skipped_no_prev += 1
                continue

            prev_rpeak = int(samples[i - 1])
            prev_prev_rpeak = int(samples[i - 2]) if i >= 2 else None
            x_macnn = make_macnn_sample(
                signal_1d=signal_1d,
                prev_rpeak=prev_rpeak,
                current_rpeak=int(rpeak),
                prev_prev_rpeak=prev_prev_rpeak,
                fs=fs,
                median_rr_seconds=median_rr,
                segment_start_after_prev_sec=float(macnn_cfg["segment_start_after_prev_sec"]),
                segment_end_after_current_sec=float(macnn_cfg["segment_end_after_current_sec"]),
                output_length=int(macnn_cfg["output_length"]),
                normalize=str(macnn_cfg.get("normalize", "zscore")),
            )
            if x_macnn is None:
                skipped_boundary += 1
                continue

            x_values.append(x_macnn)
            y_values.append(int(label))
            record_values.append(str(rec))
            symbol_values.append(str(symbol))
            sample_values.append(int(rpeak))
            fs_values.append(fs)
            rpeak_time_values.append(rpeak_time)
            lead_index_values.append(int(lead_idx))
            lead_name_values.append(str(lead_name))
            mapped_counts[ID_TO_CLASS[label]] += 1

    if not x_values:
        raise ValueError(f"No MACNN samples were extracted for {dataset}:{split_rule} -> {output}")

    x = np.stack(x_values).astype(np.float32)
    y = np.asarray(y_values, dtype=np.int64)
    np.savez_compressed(
        output,
        x_macnn=x,
        y=y,
        record_id=np.asarray(record_values, dtype=object),
        record=np.asarray(record_values, dtype=object),
        symbol=np.asarray(symbol_values, dtype=object),
        r_peak_sample=np.asarray(sample_values, dtype=np.int64),
        sample=np.asarray(sample_values, dtype=np.int64),
        fs=np.asarray(fs_values, dtype=np.float32),
        r_peak_time_sec=np.asarray(rpeak_time_values, dtype=np.float32),
        lead_index=np.asarray(lead_index_values, dtype=np.int64),
        lead_name=np.asarray(lead_name_values, dtype=object),
        class_names=np.asarray(CLASS_NAMES, dtype=object),
        config_json=np.asarray(json.dumps({"dataset": dataset, "split_rule": split_rule}, sort_keys=True), dtype=object),
    )

    times = np.asarray(rpeak_time_values, dtype=np.float32)
    return {
        "output": str(output),
        "skipped": False,
        "dataset": dataset,
        "split_rule": split_rule,
        "records": records,
        "num_beats": int(len(y)),
        "x_macnn_shape": list(x.shape),
        "class_counts": dict(mapped_counts),
        "raw_symbol_counts": dict(raw_symbol_counts),
        "ignored_symbol_counts": dict(ignored_counts),
        "skipped_boundary": int(skipped_boundary),
        "skipped_split": int(skipped_split),
        "skipped_no_prev_rpeak": int(skipped_no_prev),
        "r_peak_time_sec_min": float(times.min()) if len(times) else None,
        "r_peak_time_sec_max": float(times.max()) if len(times) else None,
        "selected_lead_counts": dict(selected_leads),
        "fallback_records": fallback_records,
    }


def make_macnn_sample(
    signal_1d: np.ndarray,
    prev_rpeak: int,
    current_rpeak: int,
    prev_prev_rpeak: int | None,
    fs: float,
    median_rr_seconds: float,
    segment_start_after_prev_sec: float,
    segment_end_after_current_sec: float,
    output_length: int,
    normalize: str = "zscore",
) -> np.ndarray | None:
    start = int(round(prev_rpeak + segment_start_after_prev_sec * fs))
    end = int(round(current_rpeak + segment_end_after_current_sec * fs))
    if start < 0 or end > len(signal_1d) or end <= start + 1:
        return None

    segment = np.asarray(signal_1d[start:end], dtype=np.float32)
    morphology = _resample_1d(segment, output_length)
    if normalize == "zscore":
        morphology = (morphology - morphology.mean()) / max(float(morphology.std()), EPS)

    pre_rr = max((current_rpeak - prev_rpeak) / max(fs, EPS), EPS)
    if prev_prev_rpeak is None:
        near_pre_rr = median_rr_seconds
    else:
        near_pre_rr = max((prev_rpeak - prev_prev_rpeak) / max(fs, EPS), EPS)
    median_rr = max(float(median_rr_seconds), EPS)
    pre_rr_ratio = pre_rr / median_rr
    near_pre_rr_ratio = near_pre_rr / median_rr

    stacked = np.stack(
        [
            morphology,
            np.full(output_length, pre_rr_ratio, dtype=np.float32),
            np.full(output_length, near_pre_rr_ratio, dtype=np.float32),
        ],
        axis=0,
    )
    return stacked[None, :, :].astype(np.float32)


def audit_first5_split(adapt_path: str | Path, heldout_path: str | Path, threshold_sec: float = 300.0) -> dict[str, Any]:
    adapt = np.load(adapt_path, allow_pickle=True)
    heldout = np.load(heldout_path, allow_pickle=True)
    adapt_times = adapt["r_peak_time_sec"].astype(np.float64)
    heldout_times = heldout["r_peak_time_sec"].astype(np.float64)
    ok = bool(adapt_times.max(initial=-np.inf) < threshold_sec and heldout_times.min(initial=np.inf) >= threshold_sec)
    return {
        "ok": ok,
        "threshold_sec": float(threshold_sec),
        "adapt_count": int(len(adapt_times)),
        "heldout_count": int(len(heldout_times)),
        "adapt_max_r_peak_time_sec": float(adapt_times.max()) if len(adapt_times) else None,
        "heldout_min_r_peak_time_sec": float(heldout_times.min()) if len(heldout_times) else None,
    }


def _include_time(rpeak_time: float, split_rule: str, threshold_sec: float) -> bool:
    if split_rule == "all":
        return True
    if split_rule == "first5":
        return rpeak_time < threshold_sec
    if split_rule == "after5":
        return rpeak_time >= threshold_sec
    raise ValueError(f"Unsupported split_rule: {split_rule}")


def _median_rr_seconds(samples: np.ndarray, fs: float) -> float:
    if len(samples) < 2:
        return 1.0
    rr = np.diff(samples.astype(np.float64)) / max(float(fs), EPS)
    return max(float(np.median(rr)), EPS)


def _resample_1d(values: np.ndarray, output_length: int) -> np.ndarray:
    if len(values) == output_length:
        return values.astype(np.float32)
    old_x = np.linspace(0.0, 1.0, num=len(values), dtype=np.float32)
    new_x = np.linspace(0.0, 1.0, num=output_length, dtype=np.float32)
    return np.interp(new_x, old_x, values).astype(np.float32)

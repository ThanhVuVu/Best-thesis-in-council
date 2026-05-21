from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from src.data.beat_extract import extract_beat
from src.data.label_mapping import CLASS_NAMES, ID_TO_CLASS, map_symbol
from src.data.physionet import choose_lead, discover_records, read_annotation, read_record
from src.utils.io import ensure_dir


def _empty_object_array(values: list[Any]) -> np.ndarray:
    return np.asarray(values, dtype=object)


def preprocess_records(
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

    for rec in tqdm(records, desc=f"preprocess {dataset}"):
        wfdb_record = read_record(raw_dir, rec)
        ann = read_annotation(raw_dir, rec)
        signal = wfdb_record.p_signal
        if signal is None:
            raise ValueError(f"Record {rec} has no physical signal")

        lead_idx, lead_name, used_fallback = choose_lead(list(wfdb_record.sig_name), preferred, fallback_index)
        if used_fallback:
            fallback_records.append(rec)
        selected_leads[lead_name] += 1
        signal_1d = signal[:, lead_idx]
        fs = float(wfdb_record.fs)

        for rpeak, symbol in zip(ann.sample, ann.symbol):
            raw_symbol_counts[symbol] += 1
            label = map_symbol(symbol)
            if label is None:
                ignored_counts[symbol] += 1
                continue
            beat = extract_beat(
                signal_1d=signal_1d,
                rpeak=int(rpeak),
                fs=fs,
                target_fs=int(data_cfg["target_sampling_rate"]),
                left_samples_target_fs=int(data_cfg["left_window_samples_360hz"]),
                right_samples_target_fs=int(data_cfg["right_window_samples_360hz"]),
                beat_length=int(data_cfg["beat_length"]),
                normalize=data_cfg["normalize"],
            )
            if beat is None:
                skipped_boundary += 1
                continue

            x_values.append(beat)
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
        raise ValueError(f"No beats were extracted for {dataset} -> {output}")

    x = np.stack(x_values).astype(np.float32)
    y = np.asarray(y_values, dtype=np.int64)
    config_json = json.dumps({
        "dataset": dataset,
        "records": records,
        "data": data_cfg,
        "lead_selection": lead_cfg,
    }, sort_keys=True)

    np.savez_compressed(
        output,
        x=x,
        y=y,
        record=_empty_object_array(record_values),
        symbol=_empty_object_array(symbol_values),
        sample=np.asarray(sample_values, dtype=np.int64),
        fs=np.asarray(fs_values, dtype=np.float32),
        domain=_empty_object_array(domain_values),
        lead_index=np.asarray(lead_index_values, dtype=np.int64),
        lead_name=_empty_object_array(lead_name_values),
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
        "class_counts": dict(mapped_counts),
        "raw_symbol_counts": dict(raw_symbol_counts),
        "ignored_symbol_counts": dict(ignored_counts),
        "skipped_boundary": int(skipped_boundary),
        "selected_lead_counts": dict(selected_leads),
        "fallback_records": fallback_records,
    }
    return summary


def all_incart_records(raw_dir: str | Path) -> list[str]:
    return discover_records(raw_dir)

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import wfdb


def discover_records(raw_dir: str | Path) -> list[str]:
    folder = Path(raw_dir)
    records = []
    for header in sorted(folder.glob("*.hea")):
        stem = header.stem
        if (folder / f"{stem}.dat").exists() and (folder / f"{stem}.atr").exists():
            records.append(stem)
    return records


def record_base(raw_dir: str | Path, record: str) -> str:
    return str(Path(raw_dir) / record)


def read_header(raw_dir: str | Path, record: str) -> Any:
    return wfdb.rdheader(record_base(raw_dir, record))


def read_record(raw_dir: str | Path, record: str) -> Any:
    return wfdb.rdrecord(record_base(raw_dir, record))


def read_annotation(raw_dir: str | Path, record: str) -> Any:
    return wfdb.rdann(record_base(raw_dir, record), "atr")


def choose_lead(sig_names: list[str], preferred: list[str], fallback_index: int = 0) -> tuple[int, str, bool]:
    for lead in preferred:
        if lead in sig_names:
            idx = sig_names.index(lead)
            return idx, sig_names[idx], False
    idx = min(max(fallback_index, 0), len(sig_names) - 1)
    return idx, sig_names[idx], True


def summarize_raw_dataset(raw_dir: str | Path, dataset: str, preferred_leads: list[str], fallback_index: int) -> dict[str, Any]:
    from src.data.label_mapping import LABEL_MAP

    raw_path = Path(raw_dir)
    records = discover_records(raw_path)
    fs_counter: Counter[str] = Counter()
    n_sig_counter: Counter[int] = Counter()
    lead_name_counter: Counter[str] = Counter()
    selected_lead_counter: Counter[str] = Counter()
    symbol_counter: Counter[str] = Counter()
    mapped_counter: Counter[str] = Counter()
    fallback_records: list[str] = []
    per_record = []
    failures = []

    for rec in records:
        try:
            header = read_header(raw_path, rec)
            ann = read_annotation(raw_path, rec)
        except Exception as exc:
            failures.append({"record": rec, "error": repr(exc)})
            continue

        lead_idx, lead_name, used_fallback = choose_lead(header.sig_name, preferred_leads, fallback_index)
        if used_fallback:
            fallback_records.append(rec)
        symbols = Counter(ann.symbol)
        mapped = Counter(LABEL_MAP[s] for s in ann.symbol if s in LABEL_MAP)

        fs_counter[str(float(header.fs))] += 1
        n_sig_counter[int(header.n_sig)] += 1
        lead_name_counter.update(header.sig_name)
        selected_lead_counter[lead_name] += 1
        symbol_counter.update(symbols)
        mapped_counter.update(mapped)
        per_record.append({
            "record": rec,
            "fs": float(header.fs),
            "n_sig": int(header.n_sig),
            "sig_len": int(header.sig_len),
            "duration_min": float(header.sig_len / header.fs / 60.0),
            "leads": list(header.sig_name),
            "selected_lead_index": int(lead_idx),
            "selected_lead_name": lead_name,
            "used_fallback_lead": used_fallback,
            "annotations": int(len(ann.sample)),
            "mapped_counts": dict(mapped),
            "mapped_total": int(sum(mapped.values())),
            "ignored": int(sum(symbols.values()) - sum(mapped.values())),
        })

    durations = [r["duration_min"] for r in per_record]
    total_annotations = int(sum(symbol_counter.values()))
    total_mapped = int(sum(mapped_counter.values()))
    return {
        "dataset": dataset,
        "raw_dir": str(raw_path),
        "records_discovered": len(records),
        "records_read_ok": len(per_record),
        "failures": failures,
        "fs_counts": dict(fs_counter),
        "n_signal_counts": {str(k): v for k, v in n_sig_counter.items()},
        "duration_min": {
            "min": min(durations) if durations else None,
            "max": max(durations) if durations else None,
            "total_hours": sum(durations) / 60.0 if durations else None,
        },
        "lead_name_counts": dict(lead_name_counter),
        "selected_lead_counts": dict(selected_lead_counter),
        "fallback_records": fallback_records,
        "symbol_counts": dict(symbol_counter),
        "mapped_nsv_counts": dict(mapped_counter),
        "total_annotations": total_annotations,
        "total_mapped_nsv": total_mapped,
        "total_ignored": total_annotations - total_mapped,
        "per_record": per_record,
    }

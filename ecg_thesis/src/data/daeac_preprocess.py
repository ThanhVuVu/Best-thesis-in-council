from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import resample
from tqdm import tqdm

from src.data.physionet import choose_lead, read_annotation, read_record
from src.utils.io import ensure_dir

EPS = 1e-8

CLASS_NAMES = ["N", "S", "V", "F"]
CLASS_TO_ID = {"N": 0, "S": 1, "V": 2, "F": 3}
ID_TO_CLASS = {v: k for k, v in CLASS_TO_ID.items()}

AAMI_SYMBOL_TO_CLASS = {
    "N": "N",
    "L": "N",
    "R": "N",
    "e": "N",
    "j": "N",
    "A": "S",
    "a": "S",
    "J": "S",
    "S": "S",
    "V": "V",
    "E": "V",
    "F": "F",
}


def build_daeac_input_tensor(morph: np.ndarray, pre_rr_ratio: float, near_pre_rr_ratio: float) -> np.ndarray:
    morph = np.asarray(morph, dtype=np.float32)
    length = int(morph.shape[0])
    pre_rr_row = np.full(length, float(pre_rr_ratio), dtype=np.float32)
    near_pre_rr_row = np.full(length, float(near_pre_rr_ratio), dtype=np.float32)
    return np.stack([morph, pre_rr_row, near_pre_rr_row], axis=0)[None, :, :].astype(np.float32)


def compute_rr_features_from_diffs(rr_diffs: np.ndarray, beat_index: int, target_fs: float) -> np.ndarray | None:
    rr_diffs = np.asarray(rr_diffs, dtype=np.float64) / max(float(target_fs), EPS)
    i = int(beat_index)
    if i < 5 or i + 5 > len(rr_diffs):
        return None
    rr_avg = float(rr_diffs.mean())
    rr_anterior = float(rr_diffs[i - 1])
    rr_posterior = float(rr_diffs[i])
    rr_local = float(np.concatenate([rr_diffs[i - 5 : i], rr_diffs[i : i + 5]]).mean())
    return np.asarray(
        [
            rr_anterior - rr_avg,
            rr_posterior - rr_avg,
            rr_local - rr_avg,
            rr_anterior / max(rr_avg, EPS),
            rr_posterior / max(rr_avg, EPS),
            rr_local / max(rr_avg, EPS),
            rr_anterior / max(rr_posterior, EPS),
        ],
        dtype=np.float32,
    )


def zscore_segment(x: np.ndarray, eps: float = EPS) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return ((x - x.mean()) / (x.std() + float(eps))).astype(np.float32)


def generate_morph_noise(signal_length: int, noise_type: str, rng: np.random.Generator, noise_cfg: dict[str, Any] | None = None) -> np.ndarray:
    cfg = noise_cfg or {}
    t = np.linspace(0.0, 1.0, int(signal_length), endpoint=False)
    if noise_type == "low_freq":
        params = dict(cfg.get("low_freq", {}))
        amp = rng.uniform(float(params.get("amp_min", 0.0)), float(params.get("amp_max", 0.05)))
        freq = rng.uniform(float(params.get("freq_min", 0.2)), float(params.get("freq_max", 1.0)))
        phase = rng.uniform(0.0, 2.0 * np.pi)
        noise = amp * np.sin(2.0 * np.pi * freq * t + phase)
    elif noise_type == "high_freq":
        params = dict(cfg.get("high_freq", {}))
        amp = rng.uniform(float(params.get("amp_min", 0.0)), float(params.get("amp_max", 0.03)))
        freq = rng.uniform(float(params.get("freq_min", 8.0)), float(params.get("freq_max", 20.0)))
        phase = rng.uniform(0.0, 2.0 * np.pi)
        noise = amp * np.sin(2.0 * np.pi * freq * t + phase)
    elif noise_type == "white_noise":
        params = dict(cfg.get("white_noise", {}))
        amp = rng.uniform(float(params.get("std_min", 0.0)), float(params.get("std_max", 0.03)))
        noise = amp * rng.standard_normal(int(signal_length))
    else:
        raise ValueError(f"Unknown noise_type: {noise_type}")
    return noise.astype(np.float32)


def augment_morphology(
    morph: np.ndarray,
    rng: np.random.Generator,
    re_zscore: bool = True,
    noise_cfg: dict[str, Any] | None = None,
) -> np.ndarray:
    morph = np.asarray(morph, dtype=np.float32)
    cfg = noise_cfg or {}
    choices = list(cfg.get("choices", ["low_freq", "high_freq", "white_noise"]))
    noise_type = str(rng.choice(choices))
    morph_aug = morph + generate_morph_noise(morph.shape[0], noise_type, rng, cfg)
    return (zscore_segment(morph_aug) if re_zscore else morph_aug.astype(np.float32)).astype(np.float32)


def apply_oversampling_on_daeac_tensor(
    X: np.ndarray,
    y: np.ndarray,
    minority_classes: tuple[int, ...] | list[int] = (1, 2, 3),
    copies_per_sample: int = 3,
    seed: int = 42,
    re_zscore_after_noise: bool = True,
    noise_cfg: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    assert X.ndim == 4, f"Expected X shape [N, 1, 3, 128], got {X.shape}"
    assert X.shape[1] == 1, f"Expected channel dimension 1, got {X.shape}"
    assert X.shape[2] == 3, f"Expected 3 rows: morphology/pre_rr/near_pre_rr, got {X.shape}"
    assert X.shape[3] == 128, f"Expected fixed length 128, got {X.shape}"
    rng = np.random.default_rng(int(seed))
    minority = {int(cls) for cls in minority_classes}
    X_out: list[np.ndarray] = []
    y_out: list[int] = []
    is_augmented: list[bool] = []
    original_index: list[int] = []
    for idx, (x_i, y_i) in enumerate(zip(X, y)):
        x_i = x_i.astype(np.float32, copy=True)
        label = int(y_i)
        X_out.append(x_i)
        y_out.append(label)
        is_augmented.append(False)
        original_index.append(int(idx))
        if label in minority:
            for _ in range(int(copies_per_sample)):
                x_aug = x_i.copy()
                x_aug[0, 0, :] = augment_morphology(
                    x_aug[0, 0, :],
                    rng,
                    re_zscore=bool(re_zscore_after_noise),
                    noise_cfg=noise_cfg,
                )
                X_out.append(x_aug)
                y_out.append(label)
                is_augmented.append(True)
                original_index.append(int(idx))
    return (
        np.stack(X_out, axis=0).astype(np.float32),
        np.asarray(y_out, dtype=np.int64),
        np.asarray(is_augmented, dtype=bool),
        np.asarray(original_index, dtype=np.int64),
    )


def map_symbol_daeac(symbol: str) -> int | None:
    mapped = AAMI_SYMBOL_TO_CLASS.get(symbol)
    if mapped is None:
        return None
    return CLASS_TO_ID[mapped]


def preprocess_daeac_records(
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
    prep_cfg = config["preprocessing"]
    target_fs = int(prep_cfg["unified_sampling_rate"])
    fixed_length = int(prep_cfg["heartbeat_segment"]["fixed_length"])
    start_after_prev_sec = float(prep_cfg["heartbeat_segment"]["start_after_previous_r_peak_seconds"])
    end_after_current_sec = float(prep_cfg["heartbeat_segment"]["end_after_current_r_peak_seconds"])
    normalize = str(prep_cfg.get("normalize", "per_segment_zscore"))
    target_adapt_seconds = float(config["data"].get("target_adapt_seconds", 300.0))

    lead_cfg = config["lead_selection"]
    preferred = lead_cfg["preferred_leads"][dataset]
    fallback_index = int(lead_cfg.get("fallback_lead_index", 0))

    x_values: list[np.ndarray] = []
    y_values: list[int] = []
    record_values: list[str] = []
    symbol_values: list[str] = []
    sample_orig_values: list[int] = []
    sample_target_values: list[int] = []
    rpeak_time_values: list[float] = []
    fs_values: list[float] = []
    lead_index_values: list[int] = []
    lead_name_values: list[str] = []
    pre_rr_ratio_values: list[float] = []
    near_pre_rr_ratio_values: list[float] = []
    rr_feature_values: list[np.ndarray] = []
    rr_features_enabled = bool(dict(prep_cfg.get("rr_features", {})).get("enabled", False))

    raw_symbol_counts: Counter[str] = Counter()
    mapped_counts: Counter[str] = Counter()
    ignored_counts: Counter[str] = Counter()
    selected_leads: Counter[str] = Counter()
    fallback_records: list[str] = []
    failures: list[dict[str, str]] = []
    skipped_boundary = 0
    skipped_split = 0
    skipped_no_previous_rpeak = 0
    skipped_no_rr_history = 0

    for rec in tqdm(records, desc=f"DAEAC preprocess {dataset}:{split_rule}"):
        try:
            wfdb_record = read_record(raw_dir, rec)
            ann = read_annotation(raw_dir, rec)
            signal = wfdb_record.p_signal
            if signal is None:
                raise ValueError(f"Record {rec} has no physical signal")
        except Exception as exc:
            failures.append({"record": str(rec), "error": repr(exc)})
            continue

        lead_idx, lead_name, used_fallback = choose_lead(list(wfdb_record.sig_name), preferred, fallback_index)
        if used_fallback:
            fallback_records.append(str(rec))
        selected_leads[lead_name] += 1

        fs = float(wfdb_record.fs)
        signal_1d = _replace_nonfinite(np.asarray(signal[:, lead_idx], dtype=np.float32))
        signal_target = _fft_resample_to_fs(signal_1d, fs, target_fs)
        rpeaks_orig = np.asarray(ann.sample, dtype=np.int64)
        rpeaks_target = np.round(rpeaks_orig.astype(np.float64) * target_fs / max(fs, EPS)).astype(np.int64)
        rr_diffs = np.diff(rpeaks_target.astype(np.float64))

        for i, (rpeak_orig, rpeak_target, symbol) in enumerate(zip(rpeaks_orig, rpeaks_target, ann.symbol)):
            raw_symbol_counts[symbol] += 1
            label = map_symbol_daeac(symbol)
            if label is None:
                ignored_counts[symbol] += 1
                continue

            rpeak_time_sec = float(rpeak_orig / max(fs, EPS))
            if not _include_time(rpeak_time_sec, split_rule, target_adapt_seconds):
                skipped_split += 1
                continue
            if i == 0:
                skipped_no_previous_rpeak += 1
                continue
            if i < 2:
                skipped_no_rr_history += 1
                continue
            rr_features = compute_rr_features_from_diffs(rr_diffs, i, target_fs) if rr_features_enabled else None
            if rr_features_enabled and rr_features is None:
                skipped_no_rr_history += 1
                continue

            start = int(round(rpeaks_target[i - 1] + start_after_prev_sec * target_fs))
            end = int(round(rpeak_target + end_after_current_sec * target_fs))
            if start < 0 or end > len(signal_target) or end <= start + 1:
                skipped_boundary += 1
                continue

            morphology = resample(signal_target[start:end].astype(np.float32), fixed_length).astype(np.float32)
            if normalize == "per_segment_zscore":
                morphology = _zscore(morphology)
            elif normalize not in {"none", "false"}:
                raise ValueError(f"Unsupported preprocessing.normalize: {normalize}")

            current_rr = float(rr_diffs[i - 1])
            previous_rrs = rr_diffs[: i - 1]
            near_start = max(0, i - 11)
            near_previous_rrs = rr_diffs[near_start : i - 1]
            if len(previous_rrs) == 0 or len(near_previous_rrs) == 0:
                skipped_no_rr_history += 1
                continue
            pre_rr_ratio = current_rr / max(float(previous_rrs.mean()), EPS)
            near_pre_rr_ratio = current_rr / max(float(near_previous_rrs.mean()), EPS)

            x = build_daeac_input_tensor(morphology, pre_rr_ratio, near_pre_rr_ratio)
            x_values.append(x.astype(np.float32))
            y_values.append(int(label))
            record_values.append(str(rec))
            symbol_values.append(str(symbol))
            sample_orig_values.append(int(rpeak_orig))
            sample_target_values.append(int(rpeak_target))
            rpeak_time_values.append(rpeak_time_sec)
            fs_values.append(fs)
            lead_index_values.append(int(lead_idx))
            lead_name_values.append(str(lead_name))
            pre_rr_ratio_values.append(float(pre_rr_ratio))
            near_pre_rr_ratio_values.append(float(near_pre_rr_ratio))
            if rr_features_enabled and rr_features is not None:
                rr_feature_values.append(rr_features)
            mapped_counts[ID_TO_CLASS[label]] += 1

    if not x_values:
        raise ValueError(f"No DAEAC samples were extracted for {dataset}:{split_rule} -> {output}")

    x_array = np.stack(x_values).astype(np.float32)
    y_array = np.asarray(y_values, dtype=np.int64)
    config_json = json.dumps(
        {
            "dataset": dataset,
            "records": records,
            "split_rule": split_rule,
            "class_to_id": CLASS_TO_ID,
            "aami_symbol_to_class": AAMI_SYMBOL_TO_CLASS,
            "preprocessing": prep_cfg,
            "lead_selection": lead_cfg,
            "decision_notes": [
                "FFT-based scipy.signal.resample is used for database sampling-rate unification and segment resizing.",
                "The first beat and the next beat without previous RR history are skipped because RR ratio denominators are undefined.",
                "Target labels are stored for evaluation/auditing only; adaptation code must ignore y for unlabeled target files.",
            ],
        },
        sort_keys=True,
    )
    payload: dict[str, Any] = dict(
        x=x_array,
        y=y_array,
        record=np.asarray(record_values, dtype=object),
        symbol=np.asarray(symbol_values, dtype=object),
        sample=np.asarray(sample_orig_values, dtype=np.int64),
        r_peak_sample=np.asarray(sample_orig_values, dtype=np.int64),
        r_peak_sample_360hz=np.asarray(sample_target_values, dtype=np.int64),
        r_peak_time_sec=np.asarray(rpeak_time_values, dtype=np.float32),
        fs_original=np.asarray(fs_values, dtype=np.float32),
        lead_index=np.asarray(lead_index_values, dtype=np.int64),
        lead_name=np.asarray(lead_name_values, dtype=object),
        pre_rr_ratio=np.asarray(pre_rr_ratio_values, dtype=np.float32),
        near_pre_rr_ratio=np.asarray(near_pre_rr_ratio_values, dtype=np.float32),
        is_augmented=np.zeros(len(y_array), dtype=bool),
        original_index=np.arange(len(y_array), dtype=np.int64),
        class_names=np.asarray(CLASS_NAMES, dtype=object),
        class_to_id_json=np.asarray(json.dumps(CLASS_TO_ID, sort_keys=True), dtype=object),
        config_json=np.asarray(config_json, dtype=object),
    )
    if rr_features_enabled:
        payload["rr_features"] = np.stack(rr_feature_values).astype(np.float32)
    np.savez_compressed(output, **payload)

    times = np.asarray(rpeak_time_values, dtype=np.float32)
    return {
        "output": str(output),
        "skipped": False,
        "dataset": dataset,
        "split_rule": split_rule,
        "records": records,
        "num_beats": int(len(y_array)),
        "x_shape": list(x_array.shape),
        "class_counts": dict(mapped_counts),
        "raw_symbol_counts": dict(raw_symbol_counts),
        "ignored_symbol_counts": dict(ignored_counts),
        "skipped_boundary": int(skipped_boundary),
        "skipped_split": int(skipped_split),
        "skipped_no_previous_rpeak": int(skipped_no_previous_rpeak),
        "skipped_no_rr_history": int(skipped_no_rr_history),
        "r_peak_time_sec_min": float(times.min()) if len(times) else None,
        "r_peak_time_sec_max": float(times.max()) if len(times) else None,
        "selected_lead_counts": dict(selected_leads),
        "fallback_records": fallback_records,
        "failures": failures,
    }


def validate_daeac_npz(path: str | Path, expected_max_time_sec: float | None = None) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    x = data["x"]
    y = data["y"]
    class_counts = Counter(np.asarray(y, dtype=np.int64).tolist())
    result = {
        "path": str(path),
        "x_shape": list(x.shape),
        "y_shape": list(y.shape),
        "shape_ok": bool(x.ndim == 4 and x.shape[1:] == (1, 3, 128) and len(y) == len(x)),
        "finite_x": bool(np.isfinite(x).all()),
        "finite_rr": bool(np.isfinite(data["pre_rr_ratio"]).all() and np.isfinite(data["near_pre_rr_ratio"]).all()),
        "class_counts": {CLASS_NAMES[int(k)]: int(v) for k, v in sorted(class_counts.items())},
        "class_names": [str(v) for v in data["class_names"].tolist()],
    }
    if "rr_features" in data.files:
        rr_features = np.asarray(data["rr_features"], dtype=np.float32)
        result["rr_features_shape"] = list(rr_features.shape)
        result["finite_rr_features"] = bool(rr_features.shape == (len(x), 7) and np.isfinite(rr_features).all())
    if expected_max_time_sec is not None:
        times = data["r_peak_time_sec"].astype(np.float64)
        result["max_time_sec"] = float(times.max()) if len(times) else None
        result["time_rule_ok"] = bool(len(times) == 0 or np.all(times < expected_max_time_sec))
    return result


def save_daeac_subset_npz(
    source_path: str | Path,
    output_path: str | Path,
    indices: np.ndarray | list[int],
    *,
    is_augmented: np.ndarray | None = None,
    original_index: np.ndarray | None = None,
) -> dict[str, Any]:
    indices = np.asarray(indices, dtype=np.int64)
    with np.load(source_path, allow_pickle=True) as data:
        payload: dict[str, Any] = {}
        for key in data.files:
            value = data[key]
            if key == "config_json":
                payload[key] = value
            elif _is_sample_axis_array(value, len(data["x"])):
                payload[key] = value[indices]
            else:
                payload[key] = value
    n = int(len(indices))
    payload["is_augmented"] = (
        np.zeros(n, dtype=bool) if is_augmented is None else np.asarray(is_augmented, dtype=bool)
    )
    payload["original_index"] = (
        indices.astype(np.int64) if original_index is None else np.asarray(original_index, dtype=np.int64)
    )
    output = Path(output_path)
    ensure_dir(output.parent)
    np.savez_compressed(output, **payload)
    return validate_daeac_npz(output)


def save_oversampled_daeac_npz(
    source_path: str | Path,
    output_path: str | Path,
    *,
    minority_classes: list[int] | tuple[int, ...] = (1, 2, 3),
    copies_per_sample: int = 3,
    seed: int = 42,
    re_zscore_after_noise: bool = True,
    noise_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    with np.load(source_path, allow_pickle=True) as data:
        X_aug, y_aug, is_augmented, original_index = apply_oversampling_on_daeac_tensor(
            data["x"],
            data["y"],
            minority_classes=minority_classes,
            copies_per_sample=copies_per_sample,
            seed=seed,
            re_zscore_after_noise=re_zscore_after_noise,
            noise_cfg=noise_cfg,
        )
        payload: dict[str, Any] = {}
        for key in data.files:
            value = data[key]
            if key == "x":
                payload[key] = X_aug
            elif key == "y":
                payload[key] = y_aug
            elif key == "config_json":
                payload[key] = value
            elif _is_sample_axis_array(value, len(data["x"])):
                payload[key] = value[original_index]
            else:
                payload[key] = value
    payload["is_augmented"] = is_augmented
    payload["original_index"] = original_index
    output = Path(output_path)
    ensure_dir(output.parent)
    np.savez_compressed(output, **payload)
    return validate_daeac_npz(output)


def _fft_resample_to_fs(signal: np.ndarray, fs: float, target_fs: int) -> np.ndarray:
    if int(round(fs)) == int(target_fs):
        return signal.astype(np.float32)
    target_len = max(1, int(round(len(signal) * float(target_fs) / max(float(fs), EPS))))
    return resample(signal.astype(np.float32), target_len).astype(np.float32)


def _is_sample_axis_array(value: np.ndarray, sample_count: int) -> bool:
    return isinstance(value, np.ndarray) and value.ndim >= 1 and len(value) == int(sample_count)


def _replace_nonfinite(signal: np.ndarray) -> np.ndarray:
    if np.isfinite(signal).all():
        return signal.astype(np.float32)
    finite = signal[np.isfinite(signal)]
    fill = float(np.median(finite)) if len(finite) else 0.0
    return np.nan_to_num(signal, nan=fill, posinf=fill, neginf=fill).astype(np.float32)


def _include_time(rpeak_time: float, split_rule: str, threshold_sec: float) -> bool:
    if split_rule == "all":
        return True
    if split_rule == "first5":
        return rpeak_time < threshold_sec
    if split_rule == "after5":
        return rpeak_time >= threshold_sec
    raise ValueError(f"Unsupported split_rule: {split_rule}")


def _zscore(values: np.ndarray) -> np.ndarray:
    mean = float(values.mean())
    std = float(values.std())
    if std < EPS:
        return (values - mean).astype(np.float32)
    return ((values - mean) / std).astype(np.float32)

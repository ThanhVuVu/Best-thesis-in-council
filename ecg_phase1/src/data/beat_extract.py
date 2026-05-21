from __future__ import annotations

import numpy as np
from scipy.signal import resample


def normalize_beat(beat: np.ndarray, mode: str = "per_beat_zscore") -> np.ndarray:
    beat = beat.astype(np.float32, copy=False)
    if mode == "none":
        return beat
    if mode != "per_beat_zscore":
        raise ValueError(f"Unsupported normalization mode: {mode}")
    return (beat - beat.mean()) / (beat.std() + 1e-8)


def extract_beat(
    signal_1d: np.ndarray,
    rpeak: int,
    fs: float,
    target_fs: int,
    left_samples_target_fs: int,
    right_samples_target_fs: int,
    beat_length: int,
    normalize: str,
) -> np.ndarray | None:
    if abs(fs - target_fs) < 1e-6:
        start = int(rpeak) - left_samples_target_fs
        end = int(rpeak) + right_samples_target_fs
        if start < 0 or end > len(signal_1d):
            return None
        beat = signal_1d[start:end]
        if len(beat) != beat_length:
            return None
    else:
        left_seconds = left_samples_target_fs / target_fs
        right_seconds = right_samples_target_fs / target_fs
        left_native = int(round(left_seconds * fs))
        right_native = int(round(right_seconds * fs))
        start = int(rpeak) - left_native
        end = int(rpeak) + right_native
        if start < 0 or end > len(signal_1d):
            return None
        segment = signal_1d[start:end]
        if len(segment) < 2:
            return None
        beat = resample(segment, beat_length)

    beat = normalize_beat(np.asarray(beat), normalize)
    return beat.reshape(1, beat_length).astype(np.float32)

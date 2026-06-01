"""Rolling z-score anomaly flags."""

from __future__ import annotations

import numpy as np


def rolling_zscore_flags(
    series: np.ndarray, window: int, threshold: float
) -> np.ndarray:
    s = np.asarray(series, dtype=float)
    n = len(s)
    w = max(window, 2)
    out = np.zeros(n, dtype=float)
    for i in range(n):
        start = max(0, i - w + 1)
        sl = s[start : i + 1]
        mean = float(sl.mean())
        var = float(((sl - mean) ** 2).sum() / len(sl))
        std = max(var**0.5, 1e-12)
        z = (s[i] - mean) / std
        if abs(z) > threshold:
            out[i] = 1.0
    return out

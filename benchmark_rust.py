#!/usr/bin/env python3
"""Python vs Rust kernel benchmark."""

from __future__ import annotations

import time
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from compute_kernel import rolling_zscore_flags  # noqa: E402

def main() -> None:
    s = np.ascontiguousarray(np.sin(np.arange(5000) * 0.01) + 100.0)
    window, threshold = 24, 3.0
    t0 = time.perf_counter()
    for _ in range(200):
        rolling_zscore_flags(s, window, threshold)
    py_s = time.perf_counter() - t0
    try:
        import finding_the_needles_anomaly_detection_in_power_plant_operations_rs as rs
    except ImportError:
        print("Build: maturin develop --release -m rust/py/Cargo.toml")
        print(f"Python {py_s:.3f}s")
        return
    rs_s = rs.bench_kernel_py(s, window, threshold, 2000)
    print(f"Python {py_s:.3f}s Rust {rs_s:.3f}s speedup {py_s / max(rs_s, 1e-9):.1f}x")
    np.testing.assert_allclose(
        rolling_zscore_flags(s, window, threshold),
        np.asarray(rs.rolling_zscore_flags_py(s, window, threshold)),
        rtol=1e-10,
    )
    print("Correctness: OK")

if __name__ == "__main__":
    main()

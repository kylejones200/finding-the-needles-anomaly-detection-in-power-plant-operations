# Finding the Needles: Anomaly Detection in Power Plant Operations

Published: 2025-10-06  
Medium: [Finding the Needles: Anomaly Detection in Power Plant Operations](https://medium.com/@kyle-t-jones/finding-the-needles-anomaly-detection-in-power-plant-operations-1c5b18e2a56f)

Compare ensemble and statistical anomaly detectors on public benchmarks and optional plant production time series.

## Quick start

```bash
uv sync
uv run plant-anomaly compare --save-plots
uv run plant-anomaly timeseries --demo --save-plots
```

Plots and tables go under `output/` (see [`output/README.md`](output/README.md)).

Equivalent: `uv run python -m power_plant_anomaly compare --save-plots`

## Project layout

```
.
├── src/power_plant_anomaly/     # installable package
│   ├── cli.py                   # CLI entry point
│   ├── config.py                # paths (output/, data/)
│   ├── data/loaders.py          # NAB URL + production CSV
│   ├── detection/
│   │   ├── compare.py           # anomsmith multi-model comparison
│   │   └── timeseries.py        # isolation forest, autoencoder, statistical
│   └── viz/plots.py             # figures → output/figures/
├── notebooks/                   # exploratory notebooks
├── tests/                       # pytest
├── output/                      # generated plots & tables (gitignored)
├── data/                        # local CSV inputs (gitignored)
├── docs/                        # Medium exports & article drafts
├── config.yaml
├── pyproject.toml
├── uv.lock
├── rust/                   # Rust port (core + PyO3 + CLI bench)
├── benchmark_rust.py       # Python vs Rust benchmark
├── src/compute_kernel.py   # Python/numpy reference kernel
```

## Commands

| Command | Description |
|---------|-------------|
| `plant-anomaly compare` | Fit Isolation Forest, LOF, PCA, Z-score, and IQR on NAB temperature data |
| `plant-anomaly timeseries` | Run isolation forest, PyTorch autoencoder, and statistical rules on annual production |
| `plant-anomaly timeseries --demo` | Same pipeline on synthetic data (no `data/pr_OK.csv` required) |

Add `--plot` for interactive windows; omit it to only write files under `output/`.

## Data

- **compare** — downloads the NAB machine-temperature CSV automatically.
- **timeseries** — expects `data/pr_OK.csv` with year columns; see [`data/README.md`](data/README.md).

## Development

```bash
uv sync --all-groups
uv run pytest
uv run ruff check src tests
```

## Troubleshooting

| Issue | What to do |
|-------|------------|
| `uv` not found | [Install uv](https://docs.astral.sh/uv/getting-started/installation/) |
| Missing `pr_OK.csv` | Use `--demo` or add the file under `data/` |
| Headless / SSH | Use `--save-plots` instead of `--plot` |

## Rust performance port

Side-by-side **Python vs Rust** implementation of the numeric hot loop — rolling z-score anomaly flags. Reference PyO3 benchmark: **see `benchmark_rust.py`** on a release build (local machine; run `benchmark_rust.py` to reproduce).

| Path | Role |
|------|------|
| `src/compute_kernel.py` | Python/numpy reference kernel |
| `rust/core/` | Pure Rust library |
| `rust/py/` | PyO3 bindings |
| `rust/bench/` | Standalone CLI benchmark |
| `benchmark_rust.py` | Python vs Rust timing + correctness check |

```bash
# Rust-only CLI benchmark
cd rust && cargo run --release -p finding_the_needles_anomaly_detection_in_power_plant_operations_bench

# Python vs Rust (PyO3)
pip install maturin numpy
maturin develop --release -m rust/py/Cargo.toml
python benchmark_rust.py
```

Python ML training, solvers, and orchestration stay in Python; Rust targets the numeric hot loops. Stochastic generators validate output shapes; deterministic kernels match at tight floating-point tolerance.


## Disclaimer

Educational/demo code only. Not financial, safety, or engineering advice. Use at your own risk. Verify results independently before any production or operational use.

## License

MIT — see [LICENSE](LICENSE).

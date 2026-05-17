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
└── uv.lock
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

## Disclaimer

Educational/demo code only. Not financial, safety, or engineering advice. Use at your own risk. Verify results independently before any production or operational use.

## License

MIT — see [LICENSE](LICENSE).

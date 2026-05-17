# Data

Input files are not committed. Update paths in [`config.yaml`](../config.yaml) if you use different names.

| File | Used by | Notes |
|------|---------|--------|
| `pr_OK.csv` | `plant-anomaly timeseries` | Plant production export with year columns (`1990`, `1991`, …) |

**Minimum layout**

```
data/
└── pr_OK.csv
```

Without local data, run the demo:

```bash
uv run plant-anomaly timeseries --demo --save-plots
```

The `compare` command downloads the [NAB machine temperature](https://github.com/numenta/NAB) benchmark automatically (no local file required).

# Output

Generated artifacts are written here and are not committed to git.

| Subfolder | Contents |
|-----------|----------|
| `figures/` | PNG plots (detector panels, method comparison, threshold curves) |
| `tables/` | CSV exports (model comparison summary) |

Regenerate:

```bash
uv sync
uv run plant-anomaly compare --save-plots
uv run plant-anomaly timeseries --demo --save-plots
```

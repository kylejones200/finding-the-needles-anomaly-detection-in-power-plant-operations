"""Load public NAB series and optional local production CSV."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

NAB_MACHINE_TEMPERATURE_URL = (
    "https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/"
    "machine_temperature_system_failure.csv"
)


def load_nab_temperature(url: str | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    """Load NAB machine-temperature series with a quantile-based pseudo ground truth."""
    source = url or NAB_MACHINE_TEMPERATURE_URL
    df = pd.read_csv(source)
    df["time"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.set_index("time")[["value"]].dropna()
    threshold = df["value"].quantile(0.98)
    df["anomaly"] = (df["value"] > threshold).astype(int)
    return df, df["anomaly"].values


def load_production_timeseries(csv_path: Path) -> pd.Series:
    """Aggregate year columns from a plant production export into an annual series."""
    df = pd.read_csv(csv_path)
    year_cols = [col for col in df.columns if col.isdigit()]
    if not year_cols:
        msg = f"No year columns found in {csv_path}"
        raise ValueError(msg)
    totals = df[year_cols].apply(pd.to_numeric, errors="coerce").sum(axis=0)
    ts = pd.Series(
        data=totals.values,
        index=pd.to_datetime(totals.index, format="%Y"),
    ).sort_index()
    return ts.interpolate(method="linear")


def synthetic_production_series(n_years: int = 30, seed: int = 42) -> pd.Series:
    """Demo annual production series when local CSV is unavailable."""
    rng = np.random.default_rng(seed)
    years = pd.date_range("1995", periods=n_years, freq="YS")
    trend = np.linspace(100, 180, n_years)
    noise = rng.normal(0, 8, n_years)
    values = trend + noise
    if n_years > 12:
        values[12] *= 0.4
    if n_years > 22:
        values[22] *= 1.8
    return pd.Series(values, index=years)

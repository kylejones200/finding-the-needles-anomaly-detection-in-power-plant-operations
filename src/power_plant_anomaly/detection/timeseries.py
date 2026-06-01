"""Isolation Forest, autoencoder, and statistical anomaly detection on a series."""

from __future__ import annotations

import logging
import sys
from pathlib import Path as _Path

_src_root = _Path(__file__).resolve().parents[2]
if str(_src_root) not in sys.path:
    sys.path.insert(0, str(_src_root))
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from power_plant_anomaly.viz.plots import (
    plot_anomaly_methods_comparison,
    plot_threshold_selection,
)

logger = logging.getLogger(__name__)


@dataclass
class TimeseriesDetectionResult:
    ts: pd.Series
    anomalies_iso: np.ndarray
    anomalies_ae: np.ndarray
    anomalies_stat: np.ndarray
    reconstruction_error: np.ndarray
    threshold_ae: float
    threshold_df: pd.DataFrame


class DenseAutoencoder(nn.Module):
    def __init__(self, input_dim: int, encoding_dim: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU(),
            nn.Linear(encoding_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _create_sequences(data: np.ndarray, window: int) -> np.ndarray:
    sequences = [data[i : i + window] for i in range(len(data) - window + 1)]
    return np.array(sequences)


def _train_autoencoder(
    model: nn.Module,
    x_train: np.ndarray,
    *,
    epochs: int = 80,
    batch_size: int = 16,
    lr: float = 0.001,
    validation_split: float = 0.2,
    patience: int = 12,
) -> nn.Module:
    x_t = torch.FloatTensor(x_train)
    n_val = max(1, int(len(x_t) * validation_split))
    x_val, x_tr = x_t[-n_val:], x_t[:-n_val]
    loader = DataLoader(TensorDataset(x_tr, x_tr), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best, wait = float("inf"), 0
    for _ in range(epochs):
        model.train()
        for xb, _ in loader:
            optimizer.zero_grad()
            criterion(model(xb), xb).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(x_val), x_val).item()
        if val_loss < best:
            best, wait = val_loss, 0
        else:
            wait += 1
            if wait >= patience:
                break
    return model


def detect_statistical(
    ts: pd.Series,
    z_threshold: float = 3,
    iqr_factor: float = 1.5,
) -> np.ndarray:
    z_scores = np.abs((ts.values - ts.mean()) / ts.std())
    anomalies_z = z_scores > z_threshold
    q1, q3 = ts.quantile(0.25), ts.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - iqr_factor * iqr, q3 + iqr_factor * iqr
    anomalies_iqr = (ts.values < lower) | (ts.values > upper)
    ma = ts.rolling(5, min_periods=1).mean()
    ma_std = ts.rolling(5, min_periods=1).std()
    anomalies_ma = np.abs(ts.values - ma.values) > (2 * ma_std.values)
    return anomalies_z | anomalies_iqr | anomalies_ma


def run_timeseries_detection(
    ts: pd.Series,
    *,
    window_size: int = 10,
    contamination: float = 0.1,
    ae_percentile: float = 95,
    random_state: int = 42,
) -> TimeseriesDetectionResult:
    features = pd.DataFrame(
        {
            "value": ts.values,
            "rolling_mean_3": ts.rolling(3, min_periods=1).mean().values,
            "rolling_std_3": ts.rolling(3, min_periods=1).std().fillna(0).values,
            "diff": ts.diff().fillna(0).values,
            "pct_change": ts.pct_change().fillna(0).values,
        }
    )
    features_scaled = StandardScaler().fit_transform(features)
    iso = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100,
    )
    anomalies_iso = iso.fit_predict(features_scaled) == -1
    sequences = _create_sequences(ts.values, window_size)
    scaler_ae = StandardScaler()
    x_scaled = scaler_ae.fit_transform(sequences.reshape(-1, 1)).reshape(sequences.shape)
    x_flat = x_scaled.reshape(len(x_scaled), -1)
    torch.manual_seed(random_state)
    model = DenseAutoencoder(input_dim=window_size, encoding_dim=5)
    _train_autoencoder(model, x_flat)
    model.eval()
    with torch.no_grad():
        reconstructed = model(torch.FloatTensor(x_flat)).numpy()
    reconstruction_error = np.mean((x_flat - reconstructed) ** 2, axis=1)
    threshold_ae = float(np.percentile(reconstruction_error, ae_percentile))
    anomalies_ae_window = reconstruction_error > threshold_ae
    anomalies_ae = np.zeros(len(ts), dtype=bool)
    for i, flag in enumerate(anomalies_ae_window):
        if flag:
            anomalies_ae[i : i + window_size] = True

    anomalies_stat = detect_statistical(ts)
    thresholds = np.linspace(reconstruction_error.min(), reconstruction_error.max(), 50)
    threshold_df = pd.DataFrame(
        {
            "threshold": thresholds,
            "anomaly_count": [(reconstruction_error > t).sum() for t in thresholds],
            "anomaly_rate": [(reconstruction_error > t).mean() for t in thresholds],
        }
    )
    logger.info(
        "Isolation Forest: %d anomalies (%.2f%%)",
        anomalies_iso.sum(),
        100 * anomalies_iso.mean(),
    )
    logger.info(
        "Autoencoder: %d periods (%.2f%%)",
        anomalies_ae.sum(),
        100 * anomalies_ae.mean(),
    )
    logger.info(
        "Statistical: %d anomalies (%.2f%%)",
        anomalies_stat.sum(),
        100 * anomalies_stat.mean(),
    )
    return TimeseriesDetectionResult(
        ts=ts,
        anomalies_iso=anomalies_iso,
        anomalies_ae=anomalies_ae,
        anomalies_stat=anomalies_stat,
        reconstruction_error=reconstruction_error,
        threshold_ae=threshold_ae,
        threshold_df=threshold_df,
    )


def save_timeseries_figures(
    result: TimeseriesDetectionResult,
    figures_dir: Path,
    *,
    show: bool = False,
) -> dict[str, Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "threshold": plot_threshold_selection(
            result.threshold_df,
            result.threshold_ae,
            out_path=figures_dir / "threshold_selection.png",
            show=show,
        ),
        "comparison": plot_anomaly_methods_comparison(
            result.ts,
            result.anomalies_iso,
            result.anomalies_ae,
            result.anomalies_stat,
            out_path=figures_dir / "anomaly_comparison.png",
            show=show,
        ),
    }
    return paths


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    np.random.seed(42)
    ts = pd.Series(
        np.sin(np.linspace(0, 20, 120)) + np.random.normal(0, 0.1, 120),
        index=pd.date_range("2010-01-01", periods=120, freq="MS"),
    )
    result = run_timeseries_detection(ts, window_size=5)
    logger.info(
        "Detection complete: iso=%d ae=%d stat=%d",
        result.anomalies_iso.sum(),
        result.anomalies_ae.sum(),
        result.anomalies_stat.sum(),
    )


if __name__ == "__main__":
    main()

"""Anomaly detection workflows."""

from power_plant_anomaly.detection.compare import run_model_comparison
from power_plant_anomaly.detection.timeseries import run_timeseries_detection

__all__ = ["run_model_comparison", "run_timeseries_detection"]

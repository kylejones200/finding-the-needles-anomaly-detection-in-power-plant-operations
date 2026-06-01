"""Compare anomaly detectors with anomsmith on the NAB temperature benchmark."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import anomsmith as am
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from power_plant_anomaly.data.loaders import load_nab_temperature
from power_plant_anomaly.viz.plots import plot_comparison_bars, plot_detector_panels

logger = logging.getLogger(__name__)


def run_model_comparison(
    *,
    figures_dir: Path | None = None,
    show: bool = False,
    contamination: float = 0.05,
    quantile: float = 0.95,
) -> dict[str, Any]:
    """Fit multiple detectors and return metrics plus optional figure paths."""
    df, ground_truth = load_nab_temperature()
    y = df["value"]
    threshold_rule = am.ThresholdRule(method="quantile", value=quantile, quantile=quantile)
    detectors = {
        "IsolationForest": am.IsolationForestDetector(contamination=contamination, random_state=42),
        "LOF": am.LOFDetector(contamination=contamination, random_state=42),
        "PCA": am.PCADetector(contamination=contamination, random_state=42),
        "ZScore": am.ZScoreScorer(),
        "IQR": am.IQRScorer(),
    }
    results: dict[str, Any] = {}
    for name, detector in detectors.items():
        detector.fit(y)
        result = am.detect_anomalies(y, detector, threshold_rule)
        precision = precision_score(ground_truth, result["flag"])
        recall = recall_score(ground_truth, result["flag"])
        f1 = f1_score(ground_truth, result["flag"])
        results[name] = {
            "result": result,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "n_anomalies": int(result["flag"].sum()),
        }
        logger.info(
            "%s: precision=%.3f recall=%.3f f1=%.3f anomalies=%d",
            name,
            precision,
            recall,
            f1,
            results[name]["n_anomalies"],
        )

    summary = pd.DataFrame(
        {
            name: {
                "Precision": results[name]["precision"],
                "Recall": results[name]["recall"],
                "F1": results[name]["f1"],
                "Anomalies": results[name]["n_anomalies"],
            }
            for name in results
        }
    ).T
    logger.info("\n%s", summary)
    saved: dict[str, Path] = {}
    if figures_dir is not None:
        figures_dir.mkdir(parents=True, exist_ok=True)
        plot_df = pd.DataFrame(
            {"value": df["value"], "ground_truth": df["anomaly"]},
            index=df.index,
        )
        for name, data in results.items():
            plot_df[f"{name}_flag"] = data["result"]["flag"].values

        saved["panels"] = plot_detector_panels(
            plot_df,
            results,
            title="Anomaly Detection Model Comparison — NAB Temperature",
            out_path=figures_dir / "nab_detector_panels.png",
            show=show,
        )
        saved["metrics"] = plot_comparison_bars(
            results,
            out_path=figures_dir / "nab_metrics_comparison.png",
            show=show,
        )

    return {"results": results, "summary": summary, "figures": saved}

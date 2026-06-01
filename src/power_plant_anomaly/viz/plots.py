"""Save comparison figures to output/figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _finalize(path: Path | None, show: bool) -> Path | None:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()
    return path


def plot_detector_panels(
    plot_df: pd.DataFrame,
    results: dict[str, Any],
    *,
    title: str,
    out_path: Path | None = None,
    show: bool = False,
) -> Path | None:
    fig, axes = plt.subplots(len(results), 1, figsize=(16, 4 * len(results)), sharex=True)
    if len(results) == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=16, y=1.0)
    for ax, (name, data) in zip(axes, results.items(), strict=True):
        ax.plot(plot_df.index, plot_df["value"], "b-", alpha=0.7, linewidth=1, label="Value")
        flagged = plot_df[plot_df[f"{name}_flag"] == 1]
        ax.scatter(
            flagged.index,
            flagged["value"],
            color="red",
            s=50,
            alpha=0.7,
            label=f"{name} ({data['n_anomalies']})",
        )
        gt = plot_df[plot_df["ground_truth"] == 1]
        ax.scatter(
            gt.index,
            gt["value"],
            color="orange",
            marker="x",
            s=100,
            alpha=0.5,
            label="Ground truth",
            zorder=10,
        )
        ax.text(
            0.02,
            0.98,
            f"P={data['precision']:.3f} R={data['recall']:.3f} F1={data['f1']:.3f}",
            transform=ax.transAxes,
            va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
        ax.set_ylabel("Value")
        ax.set_title(name)
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time")
    plt.tight_layout()
    return _finalize(out_path, show)


def plot_comparison_bars(
    results: dict[str, Any],
    *,
    out_path: Path | None = None,
    show: bool = False,
) -> Path | None:
    if out_path is None and not show:
        return None
    names = list(results.keys())
    metrics = ("precision", "recall", "f1")
    x = np.arange(len(names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, metric in enumerate(metrics):
        values = [results[n][metric] for n in names]
        ax.bar(x + i * width, values, width, label=metric.title())
    ax.set_xticks(x + width)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Anomaly detection model performance (NAB)")
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    return _finalize(out_path, show)


def plot_threshold_selection(
    threshold_df: pd.DataFrame,
    threshold_ae: float,
    *,
    out_path: Path | None = None,
    show: bool = False,
) -> Path | None:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(threshold_df["threshold"], threshold_df["anomaly_rate"], linewidth=2)
    ax.axvline(threshold_ae, color="red", linestyle="--", label=f"Selected ({threshold_ae:.4f})")
    ax.set_xlabel("Reconstruction error threshold")
    ax.set_ylabel("Anomaly rate")
    ax.set_title("Threshold selection for autoencoder", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    return _finalize(out_path, show)


def plot_anomaly_methods_comparison(
    ts: pd.Series,
    anomalies_iso: np.ndarray,
    anomalies_ae: np.ndarray,
    anomalies_stat: np.ndarray,
    *,
    out_path: Path | None = None,
    show: bool = False,
) -> Path | None:
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    panels = [
        ("Original time series", ts.values, None),
        ("Isolation Forest", ts.values, anomalies_iso),
        ("Autoencoder", ts.values, anomalies_ae),
        ("Statistical", ts.values, anomalies_stat),
    ]
    for ax, (title, values, mask) in zip(axes, panels, strict=True):
        ax.plot(ts.index, values, "b-", linewidth=1.5 if mask is not None else 2, alpha=0.7)
        if mask is not None:
            ax.scatter(
                ts.index[mask],
                np.asarray(values)[mask],
                color="red",
                s=50,
                marker="x",
                label="Anomalies",
            )
            ax.legend()
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel("Production")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Year")
    plt.tight_layout()
    return _finalize(out_path, show)

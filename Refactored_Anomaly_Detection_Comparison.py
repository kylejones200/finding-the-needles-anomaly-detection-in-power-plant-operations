"""Generated from Jupyter notebook: Anomaly Detection Model Comparison with anomsmith and plotsmith

Magics and shell lines are commented out. Run with a normal Python interpreter."""


# --- code cell ---

import anomsmith as am
import pandas as pd
import plotsmith as ps
from sklearn.metrics import f1_score, precision_score, recall_score


def main():
    # anomsmith provides a unified API for anomaly detection across multiple libraries
    # plotsmith provides enhanced visualization capabilities


    # --- code cell ---

    # Load NASA SMAP dataset
    # Note: anomsmith doesn't have a built-in dataset loader yet, so we use pandas
    url = "https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/machine_temperature_system_failure.csv"
    df = pd.read_csv(url)

    # The CSV already has 'timestamp' and 'value' columns
    # Parse timestamp and set as index
    df["time"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df.index = df["time"]
    df = df[["value"]].dropna()

    # Create anomaly labels (assuming threshold-based ground truth for comparison)
    anomaly_threshold = df["value"].quantile(0.98)
    df["anomaly"] = (df["value"] > anomaly_threshold).astype(int)
    ground_truth = df["anomaly"].values

    print(f"Loaded {len(df)} observations")
    print(
        f"Anomalies in ground truth: {ground_truth.sum()} ({100 * ground_truth.sum() / len(ground_truth):.2f}%)"
    )
    print(df.head())


    # --- code cell ---

    # anomsmith provides a unified API for anomaly detection
    # Instead of manually implementing each library's interface, we use anomsmith's detectors
    # Available detectors: IsolationForestDetector, LOFDetector, PCADetector, RobustCovarianceDetector
    # Available scorers: ZScoreScorer, IQRScorer

    # Prepare time series data
    y = df["value"]

    # Define threshold rule - using quantile method
    threshold_rule = am.ThresholdRule(method="quantile", value=0.95, quantile=0.95)

    # Compare multiple detection methods using anomsmith's unified API
    detectors = {
        "IsolationForest": am.IsolationForestDetector(contamination=0.05, random_state=42),
        "LOF": am.LOFDetector(contamination=0.05, random_state=42),
        "PCA": am.PCADetector(contamination=0.05, random_state=42),
        "ZScore": am.ZScoreScorer(),
        "IQR": am.IQRScorer(),
    }

    results_dict = {}

    for name, detector in detectors.items():
        # Fit detector/scorer
        detector.fit(y)

        # Detect anomalies using anomsmith's unified workflow
        result = am.detect_anomalies(y, detector, threshold_rule)

        # Calculate metrics if ground truth is available
        if ground_truth is not None:
            precision = precision_score(ground_truth, result["flag"])
            recall = recall_score(ground_truth, result["flag"])
            f1 = f1_score(ground_truth, result["flag"])

            results_dict[name] = {
                "result": result,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "n_anomalies": result["flag"].sum(),
            }

            print(
                f"{name}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}, Anomalies={result['flag'].sum()}"
            )
        else:
            results_dict[name] = {"result": result, "n_anomalies": result["flag"].sum()}
            print(f"{name}: Anomalies={result['flag'].sum()}")


    # --- code cell ---

    # Create summary DataFrame for comparison
    if ground_truth is not None:
        summary_df = pd.DataFrame(
            {
                name: {
                    "Precision": results_dict[name]["precision"],
                    "Recall": results_dict[name]["recall"],
                    "F1": results_dict[name]["f1"],
                    "Anomalies": results_dict[name]["n_anomalies"],
                }
                for name in results_dict.keys()
            }
        ).T

        print("\nModel Comparison Summary:")
        print(summary_df)


    # --- code cell ---

    # plotsmith makes it easy to visualize anomaly detection results
    # Since plotsmith doesn't have plot_anomaly_comparison, we'll create a comparison plot manually

    # Create a DataFrame with all detection results for visualization
    plot_df = pd.DataFrame(
        {"value": df["value"], "ground_truth": df["anomaly"]}, index=df.index
    )

    # Add each detector's results
    for name, data in results_dict.items():
        plot_df[f"{name}_flag"] = data["result"]["flag"].values
        plot_df[f"{name}_score"] = data["result"]["score"].values

    # Use plotsmith's plot_timeseries for the main time series
    ps.plot_timeseries(
        plot_df[["value"]],
        title="NASA SMAP Dataset: Time Series with Anomaly Detections",
        xlabel="Time",
        ylabel="Value",
        figsize=(16, 8),
    )


    # --- code cell ---

    # Visualize individual model results
    # Create subplots for each detector's anomaly flags
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        len(results_dict), 1, figsize=(16, 4 * len(results_dict)), sharex=True
    )
    fig.suptitle(
        "Anomaly Detection Model Comparison - NASA SMAP Dataset", fontsize=16, y=1.00
    )

    for idx, (name, data) in enumerate(results_dict.items()):
        ax = axes[idx]

        # Plot the time series
        ax.plot(
            plot_df.index, plot_df["value"], "b-", alpha=0.7, label="Value", linewidth=1
        )

        # Highlight anomalies detected by this model
        anomalies = plot_df[plot_df[f"{name}_flag"] == 1]
        ax.scatter(
            anomalies.index,
            anomalies["value"],
            color="red",
            marker="o",
            s=50,
            alpha=0.7,
            label=f"{name} Anomalies ({data['n_anomalies']})",
        )

        # Highlight ground truth anomalies
        gt_anomalies = plot_df[plot_df["ground_truth"] == 1]
        ax.scatter(
            gt_anomalies.index,
            gt_anomalies["value"],
            color="orange",
            marker="x",
            s=100,
            alpha=0.5,
            label="Ground Truth",
            zorder=10,
        )

        if ground_truth is not None:
            metrics_text = f"Precision: {data['precision']:.3f}, Recall: {data['recall']:.3f}, F1: {data['f1']:.3f}"
            ax.text(
                0.02,
                0.98,
                metrics_text,
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

        ax.set_ylabel("Value")
        ax.set_title(f"{name} Detector")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time")
    plt.tight_layout()
    plt.show()


    # --- code cell ---

    # Use plotsmith's plot_bar to compare model performance metrics
    if ground_truth is not None:
        # Create a grouped bar chart for metrics comparison
        # We'll create separate plots for each metric
        metrics_list = ["Precision", "Recall", "F1"]
        for metric in metrics_list:
            metric_values = [
                results_dict[name][metric.lower()] for name in results_dict.keys()
            ]
            ps.plot_bar(
                x=list(results_dict.keys()),
                height=metric_values,
                title=f"Anomaly Detection Model Performance: {metric}",
                xlabel="Model",
                ylabel=metric,
                figsize=(10, 6),
            )


if __name__ == "__main__":
    main()

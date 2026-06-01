"""Command-line entry points."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from power_plant_anomaly import __version__
from power_plant_anomaly.config import data_path, figures_dir, load_config, tables_dir
from power_plant_anomaly.data.loaders import load_production_timeseries, synthetic_production_series
from power_plant_anomaly.detection.compare import run_model_comparison
from power_plant_anomaly.detection.timeseries import (
    run_timeseries_detection,
    save_timeseries_figures,
)
from power_plant_anomaly.paths import DEFAULT_CONFIG_PATH

logger = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")


def cmd_compare(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config))
    save = args.save_plots or not args.plot
    fig_dir = figures_dir(config) if save else None
    out = run_model_comparison(figures_dir=fig_dir, show=args.plot)
    if args.save_plots and out["figures"]:
        for name, path in out["figures"].items():
            logger.info("%s → %s", name, path)
    summary_path = tables_dir(config) / "nab_model_comparison.csv"
    out["summary"].to_csv(summary_path)
    logger.info("Summary table → %s", summary_path)
    return 0


def cmd_timeseries(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config))
    if args.demo:
        ts = synthetic_production_series()
        logger.info("Using synthetic demo series (%d points)", len(ts))
    else:
        path = data_path(config, "production_csv")
        if not path.exists():
            logger.error(
                "Missing %s — place pr_OK.csv in data/ or run with --demo",
                path,
            )
            return 1
        ts = load_production_timeseries(path)
    result = run_timeseries_detection(ts)
    save = args.save_plots or not args.plot
    if save or args.plot:
        paths = save_timeseries_figures(
            result,
            figures_dir(config),
            show=args.plot,
        )
        for name, path in paths.items():
            logger.info("%s → %s", name, path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Anomaly detection for power plant operations",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "-c",
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to config.yaml",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    compare = sub.add_parser(
        "compare",
        help="Compare anomsmith detectors on NAB machine-temperature data",
    )
    compare.add_argument(
        "--save-plots",
        action="store_true",
        help="Write figures to output/figures/",
    )
    compare.add_argument("--plot", action="store_true", help="Show interactive plots")
    compare.set_defaults(func=cmd_compare)
    ts = sub.add_parser(
        "timeseries",
        help="Isolation Forest, autoencoder, and statistical detection on annual production",
    )
    ts.add_argument("--demo", action="store_true", help="Run on synthetic data (no CSV required)")
    ts.add_argument("--save-plots", action="store_true", help="Write figures to output/figures/")
    ts.add_argument("--plot", action="store_true", help="Show interactive plots")
    ts.set_defaults(func=cmd_timeseries)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

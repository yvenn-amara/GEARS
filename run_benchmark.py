#!/usr/bin/env python3
"""
run_benchmark.py — Run the persistence-bootstrap vs. GMM rolling-origin
benchmark across GEARS's 11 public EV datasets and write a single tidy
results table.

This is Session 4 of the persistence-vs-GMM benchmark prompt. It drives
Session 3's harness (``gears.evaluation.benchmark``) across all viable
datasets; it does not reimplement any rolling-origin, windowing, or
skip-reason logic itself — see ``gears/evaluation/benchmark.py`` (Session 3)
and ``gears/evaluation/windowing.py`` (Session 2) for that.

Supported dataset names (place the raw CSVs in ``data/preprocessed_data/``,
one file per name — see Session 1):
    acn, boulder, caltech, domestics, dundee, jpl, office, palo_alto,
    paris, perth, sap

``acn`` is excluded from the default sweep — it is the row-wise union of
caltech + jpl + office (verified, Section 1.7); including it alongside
those three would triple-count overlapping sessions.

Usage
-----
    # Smoke-test on the smallest, fastest dataset first (Session 3's own
    # acceptance criterion) before scaling up.
    python scripts/run_benchmark.py --dataset office --quick

    # Full sweep, all defaults.
    python scripts/run_benchmark.py \\
        --datasets acn,boulder,caltech,domestics,dundee,jpl,office,palo_alto,paris,perth,sap \\
        --exclude acn --horizons 1,2,3 --x-grid 1,2,3,4,8,16,52

    # Reduced grid density (see Section 4.2 runtime controls) -- e.g. on a
    # single-core / time-boxed machine, step origins every 5 days instead
    # of daily and use fewer Monte Carlo scenarios:
    python scripts/run_benchmark.py --step-days 5 --n-scenarios 20

    python scripts/run_benchmark.py --help
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import warnings
from pathlib import Path

# Allow running from any working directory without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_benchmark")

ALL_DATASETS = [
    "acn", "boulder", "caltech", "domestics", "dundee", "jpl",
    "office", "palo_alto", "paris", "perth", "sap",
]
DEFAULT_EXCLUDE = ["acn"]
DEFAULT_DATA_DIR = "data/preprocessed_data"
DEFAULT_OUTPUT = "results/benchmark/all_results.parquet"


# ── CLI -----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the GEARS persistence-bootstrap vs. GMM rolling-origin benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                    help="Directory containing the <dataset>.csv files.")
    p.add_argument("--dataset", default=None,
                    help="Run a single dataset only (overrides --datasets/--exclude), e.g. 'sap'.")
    p.add_argument("--datasets", default=",".join(ALL_DATASETS),
                    help="Comma-separated dataset names to include.")
    p.add_argument("--exclude", default=",".join(DEFAULT_EXCLUDE),
                    help="Comma-separated dataset names to exclude (default: acn -- "
                         "the union of caltech+jpl+office, Section 1.7).")
    p.add_argument("--horizons", default="1,2,3",
                    help="Comma-separated horizon day_offsets.")
    p.add_argument("--x-grid", default="1,2,3,4,8,16,52",
                    help="Comma-separated history depths X (weeks of look-back).")
    p.add_argument("--n-scenarios", type=int, default=50,
                    help="Monte Carlo scenario draws per (origin, day_offset, X, method) cell.")
    p.add_argument("--step-days", type=int, default=1,
                    help="Stride between consecutive origins (1 = every day). See Section 4.2: "
                         "raise this if the full-density grid is too slow.")
    p.add_argument("--eval-window-days", type=int, default=None,
                    help="Override the evaluation window length uniformly for every dataset "
                         "(default: per-dataset -- 30d, 14d for paris; see EVAL_WINDOW_OVERRIDES).")
    p.add_argument("--min-sessions-for-fit", type=int, default=10)
    p.add_argument("--n-components", type=int, default=1,
                    help="Fixed GMM component count for windowed fits (Section 1.3).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default=DEFAULT_OUTPUT,
                    help="Path to write the combined tidy results parquet.")
    p.add_argument("--quick", action="store_true",
                    help="Smoke-test settings (step-days=7, n-scenarios=5, x-grid=1,4,16) -- "
                         "for catching harness bugs cheaply before a full run (Section 4.2).")
    p.add_argument("--sarima-check", action="store_true",
                    help="Also run the secondary, lightweight SARIMA end-to-end sanity check "
                         "(Section 3.2) on a couple of origins per dataset. Off by default -- "
                         "it is not the main result and meaningfully adds to runtime.")
    p.add_argument("--sarima-output", default="results/benchmark/sarima_sanity_check.parquet",
                    help="Path to write the SARIMA sanity-check results, if --sarima-check is set.")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


# ── Dataset resolution ---------------------------------------------------------

def _resolve_dataset_list(args: argparse.Namespace) -> list[str]:
    if args.dataset:
        return [args.dataset]
    include = [d.strip() for d in args.datasets.split(",") if d.strip()]
    exclude = {d.strip() for d in args.exclude.split(",") if d.strip()}
    resolved = [d for d in include if d not in exclude]
    if not resolved:
        raise ValueError("No datasets left to run after applying --exclude.")
    return resolved


def _load_dataset(data_dir: Path, name: str) -> pd.DataFrame:
    from gears.data.loader import load_sessions

    path = data_dir / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset '{name}' not found at {path}.\n"
            f"  Place the 11 preprocessed CSVs under {data_dir}/ (Session 1)."
        )
    return load_sessions(str(path), verbose=False)


# ── Main ------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    from gears.evaluation.benchmark import (
        run_rolling_origin_benchmark,
        run_sarima_sanity_check,
        summarize_sarima_sanity_check,
        eval_window_for,
    )

    if args.quick:
        step_days = 7
        n_scenarios = 5
        x_grid = [1, 4, 16]
        logger.info("--quick: step_days=%d, n_scenarios=%d, x_grid=%s", step_days, n_scenarios, x_grid)
    else:
        step_days = args.step_days
        n_scenarios = args.n_scenarios
        x_grid = [int(x.strip()) for x in args.x_grid.split(",") if x.strip()]

    horizons = [int(h.strip()) for h in args.horizons.split(",") if h.strip()]
    data_dir = Path(args.data_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset_names = _resolve_dataset_list(args)
    logger.info("Datasets to run: %s", dataset_names)

    frames: list[pd.DataFrame] = []
    sanity_frames: list[pd.DataFrame] = []
    failures: dict[str, str] = {}

    for name in dataset_names:
        t0 = time.time()
        try:
            df = _load_dataset(data_dir, name)
        except FileNotFoundError as exc:
            logger.error("%s", exc)
            failures[name] = str(exc)
            continue

        eval_window_days = args.eval_window_days if args.eval_window_days is not None else eval_window_for(name)
        logger.info(
            "Running %s: %d sessions, eval_window=%dd, step_days=%d, x_grid=%s, horizons=%s, n_scenarios=%d",
            name, len(df), eval_window_days, step_days, x_grid, horizons, n_scenarios,
        )
        try:
            res = run_rolling_origin_benchmark(
                df, name,
                x_grid=x_grid,
                horizons=horizons,
                eval_window_days=eval_window_days,
                min_sessions_for_fit=args.min_sessions_for_fit,
                n_scenarios=n_scenarios,
                n_components=args.n_components,
                step_days=step_days,
                random_state=args.seed,
                verbose=False,
            )
        except Exception as exc:  # noqa: BLE001 -- keep going across datasets
            logger.error("Dataset '%s' failed: %s", name, exc)
            failures[name] = str(exc)
            continue

        elapsed = time.time() - t0
        n_ok = int((res["status"] == "ok").sum())
        n_skip = int((res["status"] != "ok").sum())
        logger.info("%s: done in %.1fs -- %d ok rows, %d skip rows.", name, elapsed, n_ok, n_skip)
        frames.append(res)

        if args.sarima_check:
            normalized = df["arrival_time"].dt.normalize()
            dataset_end = normalized.max()
            max_h = max(horizons)
            first_origin = dataset_end - pd.Timedelta(days=eval_window_days)
            last_origin = dataset_end - pd.Timedelta(days=max_h)
            sanity_origins = pd.date_range(first_origin, last_origin, periods=2) if last_origin > first_origin else [last_origin]
            try:
                sanity = run_sarima_sanity_check(
                    df, name, sanity_origins, horizons=horizons,
                    n_scenarios=10, random_state=args.seed, verbose=False,
                )
                sanity_frames.append(sanity)
                metrics = summarize_sarima_sanity_check(sanity)
                logger.info("%s: SARIMA sanity check -- %s", name, metrics)
            except Exception as exc:  # noqa: BLE001
                logger.warning("%s: SARIMA sanity check failed: %s", name, exc)

    if not frames:
        logger.error("No dataset produced results. Nothing written.")
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)
    combined.to_parquet(output_path, index=False)
    logger.info("Wrote combined results (%d rows, %d datasets) -> %s",
                len(combined), combined["dataset"].nunique(), output_path)

    if sanity_frames:
        sanity_path = Path(args.sarima_output)
        sanity_path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(sanity_frames, ignore_index=True).to_parquet(sanity_path, index=False)
        logger.info("Wrote SARIMA sanity-check results -> %s", sanity_path)

    if failures:
        logger.warning("Datasets that failed to run: %s", list(failures))

    # -- Quick console summary -------------------------------------------------
    print("\n" + "=" * 72)
    print("  GEARS persistence-vs-GMM benchmark -- run summary")
    print("=" * 72)
    for name, grp in combined.groupby("dataset"):
        n_ok = int((grp["status"] == "ok").sum())
        n_skip = int((grp["status"] != "ok").sum())
        print(f"  {name:<12} ok={n_ok:>7}  skipped={n_skip:>6}")
    if failures:
        print(f"  FAILED: {list(failures)}")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()

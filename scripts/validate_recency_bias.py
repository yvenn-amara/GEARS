"""
Standalone validation for EVSessionModel's recency-weighted fit (Session 2):
reproduces the diagnosed negative energy-bias failure mode at long history
windows (X=52) on real held-out data, and checks whether recency weighting
actually shrinks it -- plus honest checks at other strata (a required
short-window check at X=8, and a different location_type stratum).

This deliberately does NOT modify gears/evaluation/benchmark.py -- the
benchmark harness itself is out of scope for this session. It reuses the
harness's own lower-level building block
(gears.evaluation.windowing.sessions_in_last_n_occurrences) and mirrors its
rolling-origin protocol (train on data <= origin only, one pool per
(origin, day_offset, X) cell, oracle session count, Monte Carlo scenario
draws) but is scoped to measuring total-energy bias specifically, comparing
an unweighted GMM arm against a recency-weighted one.

Usage:
    python scripts/validate_recency_bias.py --data /path/to/sample_df.pkl
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from gears.data.loader import load_sessions
from gears.evaluation.windowing import sessions_in_last_n_occurrences
from gears.models.session_model import EVSessionModel

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("validate_recency_bias")


def evaluate_config(
    df: pd.DataFrame,
    location_type: str,
    X: int,
    half_life_days: float,
    eval_window_days: int = 30,
    horizons: tuple[int, ...] = (1, 2, 3),
    step_days: int = 5,
    n_scenarios: int = 30,
    n_components: int = 1,
    random_state: int = 42,
) -> dict:
    """Rolling-origin comparison of plain vs. recency-weighted GMM total-energy
    bias for one (location_type, X) configuration. Mirrors
    gears.evaluation.benchmark's rolling-origin protocol (train on data <=
    origin only, a fresh last-X-occurrences pool per (origin, day_offset),
    oracle true_count), scoped to total-energy bias only.
    """
    sub = (
        df[df["location_type"] == location_type]
        .sort_values("arrival_time")
        .reset_index(drop=True)
    )
    normalized_dates = sub["arrival_time"].dt.normalize()
    by_date = {d: g for d, g in sub.groupby(normalized_dates)}
    empty_true = sub.iloc[0:0]

    dataset_end = normalized_dates.max()
    max_h = max(horizons)
    first_origin = dataset_end - pd.Timedelta(days=eval_window_days)
    last_origin = dataset_end - pd.Timedelta(days=max_h)
    origins = pd.date_range(first_origin, last_origin, freq=f"{step_days}D")

    plain_bias, recency_bias = [], []
    plain_rel, recency_rel = [], []
    n_cells, n_skipped = 0, 0

    for origin in origins:
        train_df = sub[normalized_dates <= origin]
        for day_offset in horizons:
            target_date = origin + pd.Timedelta(days=day_offset)
            true_sessions = by_date.get(target_date, empty_true)
            true_count = len(true_sessions)
            if true_count == 0:
                n_skipped += 1
                continue

            pool, info = sessions_in_last_n_occurrences(train_df, target_date, n=X)
            if info["insufficient_history"] or info["n_sessions"] < 10:
                n_skipped += 1
                continue

            true_energy_total = float(true_sessions["energy"].sum())

            try:
                gmm_plain = EVSessionModel(
                    n_components=n_components, stratify_by=["day_of_week"],
                    random_state=random_state,
                ).fit(pool)
                gmm_recency = EVSessionModel(
                    n_components=n_components, stratify_by=["day_of_week"],
                    random_state=random_state,
                    recency=True, half_life_days=half_life_days,
                ).fit(pool)
            except RuntimeError as e:
                logger.warning(
                    "Fit failed for %s X=%d origin=%s target=%s: %s",
                    location_type, X, origin, target_date, e,
                )
                n_skipped += 1
                continue

            n_cells += 1
            for model, biases, rels in (
                (gmm_plain, plain_bias, plain_rel),
                (gmm_recency, recency_bias, recency_rel),
            ):
                energies = np.empty(n_scenarios)
                for scenario in range(n_scenarios):
                    sampled = model.sample(n_sessions=true_count, date=target_date, seed=scenario)
                    energies[scenario] = sampled["energy"].sum()
                bias = float(energies.mean() - true_energy_total)
                biases.append(bias)
                rels.append(bias / true_energy_total if true_energy_total else np.nan)

    return {
        "location_type": location_type,
        "X": X,
        "half_life_days": half_life_days,
        "n_cells": n_cells,
        "n_skipped": n_skipped,
        "plain_mean_bias_kwh": float(np.mean(plain_bias)) if plain_bias else float("nan"),
        "recency_mean_bias_kwh": float(np.mean(recency_bias)) if recency_bias else float("nan"),
        "plain_mean_rel_bias_pct": float(np.nanmean(plain_rel)) * 100 if plain_rel else float("nan"),
        "recency_mean_rel_bias_pct": float(np.nanmean(recency_rel)) * 100 if recency_rel else float("nan"),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, help="Path to sample_df.pkl (or any load_sessions source).")
    p.add_argument("--eval-window-days", type=int, default=30)
    p.add_argument("--step-days", type=int, default=5)
    p.add_argument("--n-scenarios", type=int, default=30)
    p.add_argument("--out", default="results/recency/recency_validation.csv")
    args = p.parse_args()

    df = load_sessions(args.data, verbose=False)

    # (location_type, X, half_life_days) -- half-life values per the
    # empirical rule of thumb documented on EVSessionModel.recency:
    # half_life_days ~ 14 at X=8, ~21 at X=52.
    configs = [
        ("home", 52, 21.0),   # primary: diagnosed long-history failure mode
        ("home", 8, 14.0),    # required short-window check
        ("work", 52, 21.0),   # additional stratum, different location_type
    ]

    results = []
    for location_type, X, half_life_days in configs:
        print(f"Evaluating location_type={location_type} X={X} half_life_days={half_life_days} ...")
        res = evaluate_config(
            df, location_type, X, half_life_days,
            eval_window_days=args.eval_window_days,
            step_days=args.step_days,
            n_scenarios=args.n_scenarios,
        )
        results.append(res)
        print(res)

    out = pd.DataFrame(results)
    print(out.to_string(index=False))
    out.to_csv(args.out, index=False)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()

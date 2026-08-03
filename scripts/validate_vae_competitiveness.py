#!/usr/bin/env python3
"""
validate_vae_competitiveness.py — Session 3 validation script.

Adds a third arm (VAE, via ``EVSessionModel(model_type="vae")``) to the
existing rolling-origin methodology from ``gears/evaluation/benchmark.py``,
WITHOUT modifying that module (its formal 3-arm wiring is Session 4's job,
per the refactor plan). This mirrors the precedent already set by Session 2's
``scripts/validate_recency_bias.py``: reuse the lower-level building blocks
(``sessions_in_last_n_occurrences``, ``distribution_comparison``,
``crps_ensemble``) directly.

Win-rate scoring follows notebook 4's own convention exactly (cell 16/17):
per cell, the summary score is the mean of the three Wasserstein distances
(hour, duration, energy); a method "wins" a cell if its score is lower than
persistence-bootstrap's.

Runtime note (see notebooks/4_persistence_vs_session_model_benchmark.ipynb's own run
config note): this sandbox is single-CPU with no torch acceleration, and the
VAE arm additionally *trains a fresh neural net per cell* (unlike persistence
and GMM, which are cheap closed-form fits). A literal daily-origin/50-scenario
grid across the full X=[1,2,3,4,8,16,52] grid is not tractable here. This
script defaults to a deliberately reduced grid (few origins, few X values,
fewer scenarios) -- exactly the kind of reduction notebook 4 already used and
documented for the same reason. Reported numbers are real, just from a
smaller grid; this is stated explicitly in the output, never silently.
"""

from __future__ import annotations

import argparse
import logging
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from gears.evaluation.benchmark import crps_ensemble, eval_window_for
from gears.evaluation.windowing import sessions_in_last_n_occurrences
from gears.models.persistence_sampler import PersistenceSessionSampler
from gears.models.session_model import EVSessionModel
from gears.output.aggregator import _overlap_profile_24h
from gears.utils import distribution_comparison

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("validate_vae")

#: Minimum session duration (h) used when deriving mean-power (energy /
#: duration) for the load-profile reconstruction, matching the floor
#: already used in ``OutputAggregator.build_load_profiles``'s
#: ``"mean_power"`` mode and in ``EVSessionModel._raw_to_sessions``.
_MIN_DURATION_H = 0.08


def session_load_profile(sessions: pd.DataFrame, n_sessions_per_day: float) -> np.ndarray:
    """
    Reconstruct a 24-hour average power profile (kW) for one day's worth of
    sessions, reusing ``OutputAggregator``'s own ``"mean_power"`` convention
    (each session draws ``energy / duration`` kW throughout its connection
    window; midnight-crossing sessions wrap correctly) via the package's own
    ``_overlap_profile_24h`` -- rather than reinventing that overlap math.

    Parameters
    ----------
    sessions : pd.DataFrame
        Must have ``hour``, ``duration`` (h), ``energy`` (kWh) columns --
        i.e. exactly what ``distribution_comparison`` already consumes.
    n_sessions_per_day : float
        Number of sessions this batch represents *per day*. For a single
        real day (``true_sessions``) or a single-day scenario draw
        (``sampled``, which already draws ``true_count`` sessions), this
        equals ``len(sessions)``, so the profile is *not* Monte-Carlo
        averaged across multiple simulated days -- it is that one day's
        (real or simulated) profile.

    Returns
    -------
    np.ndarray, shape (24,)
        Average power (kW) per hour-of-day.
    """
    if len(sessions) == 0 or n_sessions_per_day <= 0:
        return np.zeros(24)
    hour_col = "hour" if "hour" in sessions.columns else "arrival_hour"
    arrivals = sessions[hour_col].to_numpy(dtype=float)
    durations = np.maximum(sessions["duration"].to_numpy(dtype=float), _MIN_DURATION_H)
    powers = sessions["energy"].to_numpy(dtype=float) / durations
    return _overlap_profile_24h(arrivals, durations, powers, n_sessions_per_day, len(sessions))


def profile_errors(true_profile: np.ndarray, sampled_profile: np.ndarray) -> tuple[float, float]:
    """Return (rmse_kw, nrmse) between two 24h load profiles.

    ``nrmse`` divides by the true profile's mean power, giving a
    scale-free error comparable across cells/datasets of very different
    absolute power levels (a "home" context vs a high-power "public" one).
    """
    rmse = float(np.sqrt(np.mean((true_profile - sampled_profile) ** 2)))
    denom = float(true_profile.mean())
    nrmse = rmse / denom if denom > 1e-9 else np.nan
    return rmse, nrmse


RESULT_COLUMNS = [
    "dataset", "origin", "target_date", "day_offset", "X", "method", "scenario",
    "true_count", "wasserstein_hour", "wasserstein_duration", "wasserstein_energy",
    "crps_total_energy", "crps_mean_duration", "profile_rmse_kw", "profile_nrmse",
    "status", "n_pool_sessions", "n_pool_occurrences", "fit_seconds", "sample_seconds",
]


def _skip_row(dataset, origin, target_date, day_offset, X, true_count, status):
    row = {c: np.nan for c in RESULT_COLUMNS}
    row.update(dataset=dataset, origin=origin, target_date=target_date,
               day_offset=day_offset, X=X, true_count=true_count, status=status)
    return row


def evaluate_cell(
    dataset_name, train_df, origin, target_date, day_offset, X, true_sessions,
    n_scenarios, min_sessions_for_fit, random_state, vae_kwargs,
):
    rows = []
    true_count = len(true_sessions)
    pool, info = sessions_in_last_n_occurrences(train_df, target_date, n=X)
    n_pool_sessions = info["n_sessions"]
    n_pool_occurrences = info["n_available_occurrences"]

    if info["insufficient_history"]:
        rows.append(_skip_row(dataset_name, origin, target_date, day_offset, X,
                               true_count, "insufficient_history"))
        return rows
    if n_pool_sessions < min_sessions_for_fit:
        rows.append(_skip_row(dataset_name, origin, target_date, day_offset, X,
                               true_count, "insufficient_volume"))
        return rows
    if true_count == 0:
        rows.append(_skip_row(dataset_name, origin, target_date, day_offset, X,
                               true_count, "no_target_sessions"))
        return rows

    pool_persistence = pool.rename(columns={"hour": "arrival_hour"})
    arms = []

    t0 = time.perf_counter()
    try:
        persistence = PersistenceSessionSampler(random_state=random_state).fit(pool_persistence)
        arms.append(("persistence", persistence, time.perf_counter() - t0))
    except (ValueError, RuntimeError) as e:
        logger.warning("persistence fit failed: %s", e)

    t0 = time.perf_counter()
    try:
        gmm = EVSessionModel(
            n_components=1, stratify_by=["day_of_week"], random_state=random_state,
        ).fit(pool)
        arms.append(("gmm", gmm, time.perf_counter() - t0))
    except RuntimeError as e:
        logger.warning("gmm fit failed: %s", e)

    t0 = time.perf_counter()
    try:
        vae = EVSessionModel(
            n_components=1, stratify_by=["day_of_week"], model_type="vae",
            random_state=random_state, **vae_kwargs,
        ).fit(pool)
        arms.append(("vae", vae, time.perf_counter() - t0))
    except (RuntimeError, ValueError) as e:
        logger.warning("vae fit failed: %s", e)

    true_energy_total = float(true_sessions["energy"].sum())
    true_duration_mean = float(true_sessions["duration"].mean())
    true_profile = session_load_profile(true_sessions, true_count)

    for method_name, model, fit_seconds in arms:
        scenario_dists = []
        energies = np.empty(n_scenarios)
        durations = np.empty(n_scenarios)
        profile_rmses = np.empty(n_scenarios)
        profile_nrmses = np.empty(n_scenarios)
        t_samp0 = time.perf_counter()
        for scenario in range(n_scenarios):
            sampled = model.sample(n_sessions=true_count, date=target_date, seed=scenario)
            dist_cmp = distribution_comparison(
                true_sessions, sampled, features=["hour", "duration", "energy"]
            )
            scenario_dists.append(dist_cmp)
            energies[scenario] = sampled["energy"].sum()
            durations[scenario] = sampled["duration"].mean()
            sampled_profile = session_load_profile(sampled, true_count)
            profile_rmses[scenario], profile_nrmses[scenario] = profile_errors(
                true_profile, sampled_profile
            )
        sample_seconds = time.perf_counter() - t_samp0

        crps_energy = crps_ensemble(energies, true_energy_total)
        crps_duration = crps_ensemble(durations, true_duration_mean)

        for scenario, dist_cmp in enumerate(scenario_dists):
            row = {c: np.nan for c in RESULT_COLUMNS}
            wmap = {r["feature"]: r["wasserstein"] for _, r in dist_cmp.iterrows()}
            row.update(
                dataset=dataset_name, origin=origin, target_date=target_date,
                day_offset=day_offset, X=X, method=method_name, scenario=scenario,
                true_count=true_count, status="ok",
                wasserstein_hour=wmap.get("hour"), wasserstein_duration=wmap.get("duration"),
                wasserstein_energy=wmap.get("energy"),
                crps_total_energy=crps_energy, crps_mean_duration=crps_duration,
                profile_rmse_kw=profile_rmses[scenario], profile_nrmse=profile_nrmses[scenario],
                n_pool_sessions=n_pool_sessions, n_pool_occurrences=n_pool_occurrences,
                fit_seconds=fit_seconds, sample_seconds=sample_seconds / n_scenarios,
            )
            rows.append(row)
    return rows


def run(
    df, dataset_name, x_grid, horizons, n_origins, step_days, n_scenarios,
    min_sessions_for_fit, random_state, vae_kwargs, eval_window_days=None,
):
    if eval_window_days is None:
        eval_window_days = eval_window_for(dataset_name, default=n_origins * step_days + 5)
    df = df.sort_values("arrival_time").reset_index(drop=True)
    normalized_dates = df["arrival_time"].dt.normalize()
    dataset_end = normalized_dates.max()
    max_h = max(horizons)
    first_origin = dataset_end - pd.Timedelta(days=eval_window_days)
    last_origin = dataset_end - pd.Timedelta(days=max_h)
    origins = pd.date_range(first_origin, last_origin, freq=f"{step_days}D")
    if n_origins is not None:
        origins = origins[-n_origins:]

    by_date = {d: g for d, g in df.groupby(normalized_dates)}
    empty_true = df.iloc[0:0]

    rows = []
    for origin in origins:
        train_df = df[normalized_dates <= origin]
        for day_offset in horizons:
            target_date = origin + pd.Timedelta(days=day_offset)
            true_sessions = by_date.get(target_date, empty_true)
            for X in x_grid:
                rows.extend(evaluate_cell(
                    dataset_name, train_df, origin, target_date, day_offset, X,
                    true_sessions, n_scenarios, min_sessions_for_fit, random_state,
                    vae_kwargs,
                ))
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def summarize(result: pd.DataFrame) -> None:
    ok = result[result["status"] == "ok"].copy()
    print(f"\n{len(ok)} ok rows, {(result['status'] != 'ok').sum()} skip rows "
          f"({result['status'].value_counts().to_dict()})")
    if ok.empty:
        print("No ok rows -- nothing to summarize.")
        return

    feat_cols = ["wasserstein_hour", "wasserstein_duration", "wasserstein_energy"]
    cell_metric = (
        ok.groupby(["dataset", "origin", "target_date", "day_offset", "X", "method"])[feat_cols]
        .mean()
    )
    cell_metric["total"] = cell_metric.mean(axis=1)
    cell_metric = cell_metric.reset_index()
    pivot = cell_metric.pivot_table(
        index=["dataset", "origin", "target_date", "day_offset", "X"],
        columns="method", values="total",
    )

    for method in ["gmm", "vae"]:
        if method not in pivot.columns or "persistence" not in pivot.columns:
            continue
        paired = pivot[[method, "persistence"]].dropna()
        if paired.empty:
            print(f"\n{method}: no paired cells with persistence.")
            continue
        wins = (paired[method] < paired["persistence"]).mean() * 100
        print(f"\n{method} vs persistence: {len(paired)} paired cells, "
              f"win rate = {wins:.1f}%")
        print(f"  mean Wasserstein score -- {method}: {paired[method].mean():.4f}, "
              f"persistence: {paired['persistence'].mean():.4f}")

    # --- Load-profile reconstruction (separate from the Wasserstein-based
    # win-rate above -- this does not redefine notebook 4's established
    # scoring convention, it reports an additional, practically-motivated
    # metric: does the reconstructed 24h charging power curve match?) ---
    profile_cell = (
        ok.groupby(["dataset", "origin", "target_date", "day_offset", "X", "method"])
        ["profile_nrmse"].mean().reset_index()
    )
    profile_pivot = profile_cell.pivot_table(
        index=["dataset", "origin", "target_date", "day_offset", "X"],
        columns="method", values="profile_nrmse",
    )
    print("\n--- Load-profile reconstruction (NRMSE = RMSE / mean true power) ---")
    for method in ["gmm", "vae"]:
        if method not in profile_pivot.columns or "persistence" not in profile_pivot.columns:
            continue
        paired = profile_pivot[[method, "persistence"]].dropna()
        if paired.empty:
            print(f"{method}: no paired cells with persistence.")
            continue
        wins = (paired[method] < paired["persistence"]).mean() * 100
        print(f"{method} vs persistence: {len(paired)} paired cells, "
              f"profile win rate = {wins:.1f}%")
        print(f"  mean profile NRMSE -- {method}: {paired[method].mean():.4f}, "
              f"persistence: {paired['persistence'].mean():.4f}")

    print("\nTiming (mean over ok rows, seconds):")
    timing = ok.groupby("method")[["fit_seconds", "sample_seconds"]].mean()
    print(timing.to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Path to a CSV/pkl file loadable via load_sessions.")
    p.add_argument("--dataset-name", required=True)
    p.add_argument("--x-grid", default="4,8,16")
    p.add_argument("--horizons", default="1,2")
    p.add_argument("--n-origins", type=int, default=6)
    p.add_argument("--step-days", type=int, default=5)
    p.add_argument("--n-scenarios", type=int, default=20)
    p.add_argument("--min-sessions-for-fit", type=int, default=10)
    p.add_argument("--vae-epochs", type=int, default=50)
    p.add_argument("--vae-hidden-dim", type=int, default=128)
    p.add_argument("--vae-latent-dim", type=int, default=8)
    p.add_argument("--vae-batch-size", type=int, default=128)
    p.add_argument("--vae-beta", type=float, default=1.0)
    p.add_argument("--eval-window-days", type=int, default=None)
    p.add_argument("--filter", default=None, help="pandas query() expression applied after load.")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    warnings.filterwarnings("ignore")
    from gears.data.loader import load_sessions

    df = load_sessions(args.data, verbose=True)
    if args.filter:
        df = df.query(args.filter).reset_index(drop=True)
        print(f"[filter] {args.filter!r} -> {len(df):,} rows")

    vae_kwargs = {
        "vae_epochs": args.vae_epochs, "vae_hidden_dim": args.vae_hidden_dim,
        "vae_latent_dim": args.vae_latent_dim, "vae_batch_size": args.vae_batch_size,
        "vae_beta": args.vae_beta,
    }

    t0 = time.perf_counter()
    result = run(
        df, args.dataset_name,
        x_grid=[int(x) for x in args.x_grid.split(",")],
        horizons=[int(h) for h in args.horizons.split(",")],
        n_origins=args.n_origins, step_days=args.step_days,
        n_scenarios=args.n_scenarios, min_sessions_for_fit=args.min_sessions_for_fit,
        random_state=42, vae_kwargs=vae_kwargs, eval_window_days=args.eval_window_days,
    )
    print(f"\nTotal wall time: {time.perf_counter() - t0:.1f}s")
    summarize(result)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix == ".parquet":
            out_path = out_path.with_suffix(".csv")
        result.to_csv(out_path, index=False)
        print(f"Saved to {out_path}")

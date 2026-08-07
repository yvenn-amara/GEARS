#!/usr/bin/env python3
"""
validate_shared_vae_hypothesis.py — Phase 2 / Session 10 validation script.

Tests the specific, falsifiable hypothesis written down (but never executed) in
REFACTOR_STATE.md's Phase 1 "Session 3 — VAE competitiveness" write-up ("why
persistence keeps winning"): the existing rolling-origin harness
(``gears/evaluation/benchmark.py``) retrains a fresh VAE from scratch on each
individual cell's own narrow windowed pool, discarding the VAE's one real
architectural advantage over GMM -- a single network shared across many
contexts. This script does NOT modify ``benchmark.py`` (same precedent as
Session 2's ``validate_recency_bias.py`` and Session 3's
``validate_vae_competitiveness.py``): it reuses the harness's own lower-level
building blocks (``sessions_in_last_n_occurrences``, ``distribution_comparison``,
``crps_ensemble``) directly.

For each rolling origin, this fits ONE shared ``EVSessionModel(model_type="vae")``
across ALL of that origin's training history (mirroring how a VAE is fit
normally, outside the benchmark's per-cell design -- see
``EVSessionModel._fit_vae``), then scores it against each held-out day exactly
like the existing arms. Since the shared model doesn't depend on the
lookback window X, its score is computed once per (origin, day_offset) and
compared against persistence at every X in the grid during analysis -- not
duplicated as if it were a separate fit per X.

For a direct, apples-to-apples control, the existing per-cell design
(``vae_percell``: a fresh, tiny single-cell VAE, same as the "vae" arm in
``benchmark.py``) and ``persistence`` are also evaluated on the exact same
cells, using the same hyperparameters as the shared arm, so any difference is
attributable to the shared-vs-per-cell fitting design, not to a hyperparameter
change.

Win-rate scoring follows notebook 4 / validate_vae_competitiveness.py's own
convention: per cell, the summary score is the mean of the three Wasserstein
distances (hour, duration, energy); a method "wins" a cell if its score is
lower than persistence-bootstrap's. The profile-NRMSE angle
(``session_load_profile`` / ``profile_errors``, mirroring
``validate_vae_competitiveness.py``'s versions, duplicated here per this
script's own standalone-script precedent) is reported as a separate,
independent comparison, not folded into the Wasserstein-based score.

Runtime note: fitting ONE shared VAE per origin (reused across every
day_offset and X) is dramatically cheaper than the per-cell design's
fit-per-(origin, day_offset, X) -- this is precisely the compute asymmetry
that makes the hypothesis worth testing. The per-cell control arm
(``vae_percell``) is still the expensive part; this script defaults to a
deliberately reduced grid (few origins, few X values, few scenarios),
following the same reduction precedent Session 3/Session 4 already documented
and stated explicitly in this script's own output, never silently.
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
logger = logging.getLogger("validate_shared_vae")

#: Mirrors validate_vae_competitiveness.py's floor (OutputAggregator's
#: "mean_power" convention / EVSessionModel._raw_to_sessions' own clip).
_MIN_DURATION_H = 0.08


def session_load_profile(sessions: pd.DataFrame, n_sessions_per_day: float) -> np.ndarray:
    """Reconstruct a 24h average power profile (kW); see
    validate_vae_competitiveness.py's identical function for the full
    docstring -- duplicated here, not imported, since scripts/ is not a
    package (no __init__.py), matching this repo's existing precedent of
    each validate_*.py script being self-contained."""
    if len(sessions) == 0 or n_sessions_per_day <= 0:
        return np.zeros(24)
    hour_col = "hour" if "hour" in sessions.columns else "arrival_hour"
    arrivals = sessions[hour_col].to_numpy(dtype=float)
    durations = np.maximum(sessions["duration"].to_numpy(dtype=float), _MIN_DURATION_H)
    powers = sessions["energy"].to_numpy(dtype=float) / durations
    return _overlap_profile_24h(arrivals, durations, powers, n_sessions_per_day, len(sessions))


def profile_errors(true_profile: np.ndarray, sampled_profile: np.ndarray) -> tuple[float, float]:
    """Return (rmse_kw, nrmse) between two 24h load profiles."""
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


def _score_arm(
    model, method_name, dataset_name, origin, target_date, day_offset, X,
    true_sessions, true_count, true_energy_total, true_duration_mean, true_profile,
    n_scenarios, n_pool_sessions, n_pool_occurrences, fit_seconds,
) -> list[dict]:
    """Draw n_scenarios samples from a fitted model and score every scenario
    against the realized day. Shared by every arm (persistence, vae_percell,
    vae_shared) so the scenario/scoring loop has one implementation, not
    three copies."""
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
    sample_seconds = (time.perf_counter() - t_samp0) / n_scenarios

    crps_energy = crps_ensemble(energies, true_energy_total)
    crps_duration = crps_ensemble(durations, true_duration_mean)

    rows = []
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
            fit_seconds=fit_seconds, sample_seconds=sample_seconds,
        )
        rows.append(row)
    return rows


def evaluate_origin(
    dataset_name, train_df, origin, horizons, x_grid, by_date, empty_true,
    n_scenarios, min_sessions_for_fit, random_state, vae_kwargs,
) -> list[dict]:
    """Evaluate one rolling origin: fit ONE shared VAE across all of
    train_df (every day-of-week context, full history <= origin -- nothing
    after origin is ever visible, same no-leakage rule as benchmark.py),
    then score it against every day_offset. Also runs the existing
    per-cell persistence/vae_percell arms at every X, for a direct control.
    """
    rows: list[dict] = []

    # -- Shared VAE: ONE fit for this whole origin, reused across every
    # day_offset and every X below -- this is the entire point being
    # tested (the per-cell "vae" arm in benchmark.py cannot do this by
    # design: it retrains from scratch per (origin, day_offset, X) cell).
    t0 = time.perf_counter()
    shared_vae = None
    shared_fit_seconds = time.perf_counter() - t0
    try:
        shared_vae = EVSessionModel(
            stratify_by=["day_of_week"], model_type="vae",
            random_state=random_state, **vae_kwargs,
        ).fit(train_df)
        shared_fit_seconds = time.perf_counter() - t0
    except (RuntimeError, ValueError) as e:
        logger.warning("shared VAE fit failed for %s origin=%s: %s", dataset_name, origin, e)

    n_train_sessions = len(train_df)

    for day_offset in horizons:
        target_date = origin + pd.Timedelta(days=day_offset)
        true_sessions = by_date.get(target_date, empty_true)
        true_count = len(true_sessions)

        if true_count == 0:
            rows.append(_skip_row(dataset_name, origin, target_date, day_offset,
                                   np.nan, true_count, "no_target_sessions"))
            continue

        true_energy_total = float(true_sessions["energy"].sum())
        true_duration_mean = float(true_sessions["duration"].mean())
        true_profile = session_load_profile(true_sessions, true_count)

        # -- vae_shared: X is NaN by construction (see module docstring) --
        if shared_vae is not None:
            rows.extend(_score_arm(
                shared_vae, "vae_shared", dataset_name, origin, target_date, day_offset,
                np.nan, true_sessions, true_count, true_energy_total, true_duration_mean,
                true_profile, n_scenarios, n_train_sessions, np.nan, shared_fit_seconds,
            ))
        else:
            rows.append(_skip_row(dataset_name, origin, target_date, day_offset,
                                   np.nan, true_count, "vae_shared_fit_failed"))

        # -- persistence + vae_percell: existing per-cell design, X-indexed,
        # same pool-construction/skip-gate rules as benchmark.py._evaluate_cell.
        for X in x_grid:
            pool, info = sessions_in_last_n_occurrences(train_df, target_date, n=X)
            n_pool_sessions = info["n_sessions"]
            n_pool_occurrences = info["n_available_occurrences"]

            if info["insufficient_history"]:
                rows.append(_skip_row(dataset_name, origin, target_date, day_offset, X,
                                       true_count, "insufficient_history"))
                continue
            if n_pool_sessions < min_sessions_for_fit:
                rows.append(_skip_row(dataset_name, origin, target_date, day_offset, X,
                                       true_count, "insufficient_volume"))
                continue

            pool_persistence = pool.rename(columns={"hour": "arrival_hour"})

            t0 = time.perf_counter()
            try:
                persistence = PersistenceSessionSampler(
                    random_state=random_state
                ).fit(pool_persistence)
                rows.extend(_score_arm(
                    persistence, "persistence", dataset_name, origin, target_date, day_offset, X,
                    true_sessions, true_count, true_energy_total, true_duration_mean,
                    true_profile, n_scenarios, n_pool_sessions, n_pool_occurrences,
                    time.perf_counter() - t0,
                ))
            except (ValueError, RuntimeError) as e:
                logger.warning("persistence fit failed: %s", e)
                rows.append(_skip_row(dataset_name, origin, target_date, day_offset, X,
                                       true_count, "persistence_fit_failed"))

            t0 = time.perf_counter()
            try:
                vae_percell = EVSessionModel(
                    stratify_by=["day_of_week"], model_type="vae",
                    random_state=random_state, **vae_kwargs,
                ).fit(pool)
                rows.extend(_score_arm(
                    vae_percell, "vae_percell", dataset_name, origin, target_date, day_offset, X,
                    true_sessions, true_count, true_energy_total, true_duration_mean,
                    true_profile, n_scenarios, n_pool_sessions, n_pool_occurrences,
                    time.perf_counter() - t0,
                ))
            except (RuntimeError, ValueError) as e:
                logger.warning("per-cell vae fit failed: %s", e)
                rows.append(_skip_row(dataset_name, origin, target_date, day_offset, X,
                                       true_count, "vae_fit_failed"))

    return rows


def run(
    df, dataset_name, x_grid, horizons, n_origins, step_days, n_scenarios,
    min_sessions_for_fit, random_state, vae_kwargs, eval_window_days=None,
) -> pd.DataFrame:
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

    rows: list[dict] = []
    for origin in origins:
        train_df = df[normalized_dates <= origin]
        rows.extend(evaluate_origin(
            dataset_name, train_df, origin, horizons, x_grid, by_date, empty_true,
            n_scenarios, min_sessions_for_fit, random_state, vae_kwargs,
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

    # Per-cell mean score (mean over scenarios) for each (dataset, origin,
    # target_date, day_offset, X, method). vae_shared rows have X=NaN.
    cell = (
        ok.groupby(["dataset", "origin", "target_date", "day_offset", "X", "method"],
                    dropna=False)[[*feat_cols, "profile_nrmse"]]
        .mean()
    )
    cell["wasserstein_total"] = cell[feat_cols].mean(axis=1)
    cell = cell.reset_index()

    # X-indexed methods (persistence, vae_percell) pivot normally.
    x_indexed = cell[cell["method"] != "vae_shared"]
    pivot = x_indexed.pivot_table(
        index=["dataset", "origin", "target_date", "day_offset"],
        columns=["method", "X"], values="wasserstein_total",
    )
    profile_pivot = x_indexed.pivot_table(
        index=["dataset", "origin", "target_date", "day_offset"],
        columns=["method", "X"], values="profile_nrmse",
    )

    # vae_shared: one score per (dataset, origin, target_date, day_offset),
    # not X-indexed -- joined against every X below.
    shared = cell[cell["method"] == "vae_shared"].set_index(
        ["dataset", "origin", "target_date", "day_offset"]
    )[["wasserstein_total", "profile_nrmse"]]
    shared.columns = ["vae_shared", "vae_shared_profile"]

    x_values = sorted({X for (_m, X) in pivot.columns if not pd.isna(X)})

    print("\n=== Wasserstein-based win rate vs. persistence, by X ===")
    print("(vae_shared is the SAME fit at every X -- it doesn't depend on X; "
          "shown paired against each X's persistence cells for comparability.)")
    for X in x_values:
        row_report = {"X": int(X)}
        if ("persistence", X) not in pivot.columns:
            continue
        persistence_scores = pivot[("persistence", X)]
        for method in ["vae_percell", "vae_shared"]:
            if method == "vae_shared":
                other = shared["vae_shared"]
            elif (method, X) in pivot.columns:
                other = pivot[(method, X)]
            else:
                continue
            paired = pd.DataFrame({"other": other, "persistence": persistence_scores}).dropna()
            if paired.empty:
                continue
            wins = (paired["other"] < paired["persistence"]).mean() * 100
            row_report[f"{method}_n"] = len(paired)
            row_report[f"{method}_win_pct"] = round(wins, 1)
            row_report[f"{method}_mean"] = round(paired["other"].mean(), 4)
        row_report["persistence_mean"] = round(persistence_scores.mean(), 4)
        print(row_report)

    print("\n=== vae_shared vs. vae_percell directly (does sharing help?) ===")
    for X in x_values:
        if ("vae_percell", X) not in pivot.columns:
            continue
        paired = pd.DataFrame({
            "vae_shared": shared["vae_shared"], "vae_percell": pivot[("vae_percell", X)],
        }).dropna()
        if paired.empty:
            continue
        wins = (paired["vae_shared"] < paired["vae_percell"]).mean() * 100
        print(f"X={int(X)}: {len(paired)} paired cells, vae_shared beats vae_percell "
              f"{wins:.1f}% of the time "
              f"(mean {paired['vae_shared'].mean():.4f} vs {paired['vae_percell'].mean():.4f})")

    print("\n=== Profile-NRMSE win rate vs. persistence, by X (separate from Wasserstein) ===")
    for X in x_values:
        if ("persistence", X) not in profile_pivot.columns:
            continue
        persistence_scores = profile_pivot[("persistence", X)]
        for method in ["vae_percell", "vae_shared"]:
            if method == "vae_shared":
                other = shared["vae_shared_profile"]
            elif (method, X) in profile_pivot.columns:
                other = profile_pivot[(method, X)]
            else:
                continue
            paired = pd.DataFrame({"other": other, "persistence": persistence_scores}).dropna()
            if paired.empty:
                continue
            wins = (paired["other"] < paired["persistence"]).mean() * 100
            print(f"X={int(X)}, {method}: {len(paired)} paired cells, profile win rate "
                  f"{wins:.1f}% (mean NRMSE {paired['other'].mean():.4f} vs "
                  f"persistence {paired['persistence'].mean():.4f})")

    print("\nTiming (mean over ok rows, seconds):")
    timing = ok.groupby("method")[["fit_seconds", "sample_seconds"]].mean()
    print(timing.to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Path to a CSV/pkl file loadable via load_sessions.")
    p.add_argument("--dataset-name", required=True)
    p.add_argument("--x-grid", default="1,4,16")
    p.add_argument("--horizons", default="1,2")
    p.add_argument("--n-origins", type=int, default=3)
    p.add_argument("--step-days", type=int, default=7)
    p.add_argument("--n-scenarios", type=int, default=15)
    p.add_argument("--min-sessions-for-fit", type=int, default=10)
    p.add_argument("--vae-epochs", type=int, default=50)
    p.add_argument("--vae-hidden-dim", type=int, default=256)
    p.add_argument("--vae-latent-dim", type=int, default=16)
    p.add_argument("--vae-batch-size", type=int, default=512)
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

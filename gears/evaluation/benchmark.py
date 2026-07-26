"""
Rolling-origin benchmark harness: persistence-bootstrap vs. windowed GMM.

For each dataset, for a sliding evaluation window near its chronological end,
for each horizon day (``day_offset`` within ``H in {1, 2, 3}``) and each
history depth ``X`` (number of most-recent same-weekday occurrences pooled),
this compares :class:`~gears.models.persistence_sampler.PersistenceSessionSampler`
against a windowed :class:`~gears.models.gmm.EVSessionGMM` on how well their
sampled sessions match the truly realized ones -- using the true, known
session count for both arms (the "oracle-count" design), plus a small
secondary end-to-end SARIMA sanity check.

See the benchmark prompt, Section 1 (ground truth) and Session 3, for the
full specification this module implements. In particular:

- **No leakage across the origin boundary.** ``train_df`` is fixed at
  ``origin`` for the whole horizon; nothing after ``origin`` is ever visible
  to either arm, for any day inside the horizon.
- **Each day inside a multi-day horizon gets its own last-X-occurrences
  pool**, built from ``train_df`` (i.e. still <= origin), never reused
  across the horizon.
- **Two skip reasons are logged separately, never merged or silently
  degraded**: ``insufficient_history`` (fewer than X same-weekday
  occurrences exist at all -- a calendar problem) vs.
  ``insufficient_volume`` (occurrences exist but the pooled sessions are
  too few -- below ``MIN_SESSIONS_FOR_FIT``). A degraded X is never quietly
  substituted and relabeled as the requested X.
- A third, harness-level skip reason, ``no_target_sessions``, additionally
  guards against target days with zero realized sessions (common on sparse
  datasets like ``office.csv``) -- comparing an empty realized distribution
  against anything is undefined, not a "GMM vs. persistence" result, so
  these cells are skipped and logged rather than silently producing NaNs
  or crashing inside scipy.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd

from gears.evaluation.windowing import sessions_in_last_n_occurrences
from gears.models.gmm import EVSessionGMM
from gears.models.persistence_sampler import PersistenceSessionSampler
from gears.utils import distribution_comparison, forecast_metrics

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Grid / gate defaults (see Section 1 ground truth + Session 3 spec)
# ---------------------------------------------------------------------------

#: History depths, in "number of most recent occurrences of the target
#: weekday", i.e. weeks of look-back (Section 0.2, assumption 1).
DEFAULT_X_GRID: list[int] = [1, 2, 3, 4, 8, 16, 52]

#: Horizon days evaluated from each origin (day_offset values, not a single
#: fixed H -- every offset 1..max(horizons) is evaluated per origin).
DEFAULT_HORIZONS: list[int] = [1, 2, 3]

#: Minimum pooled sessions required to attempt a fit -- matches
#: EVSessionGMM's own n<10 convention (Section 1.3), reused rather than
#: inventing a new threshold (Section 0.2, assumption 6).
MIN_SESSIONS_FOR_FIT: int = 10

#: Default evaluation window length in days (Section 0.2, assumption 5).
DEFAULT_EVAL_WINDOW_DAYS: int = 30

#: Recommended number of Monte Carlo scenario draws per cell (Section 3.1).
DEFAULT_N_SCENARIOS: int = 50

#: Shortened evaluation windows for short-span datasets, per Section 1.7
#: (a fixed 30-day window leaves almost nothing to train on for paris.csv,
#: whose entire span is 59 days).
EVAL_WINDOW_OVERRIDES: dict[str, int] = {"paris": 14}

#: Datasets excluded from the default sweep -- acn.csv is the row-wise union
#: of caltech.csv + jpl.csv + office.csv (Section 1.7); including it
#: alongside those three would double/triple-count overlapping sessions.
EXCLUDED_DATASETS: set[str] = {"acn"}

#: Tidy long-format results schema (Section 3.4) -- one row per scenario
#: draw for "ok" cells, one row per skipped cell for skip reasons.
RESULT_COLUMNS: list[str] = [
    "dataset", "origin", "target_date", "day_offset", "weekday", "X",
    "method", "scenario", "true_count",
    "wasserstein_hour", "wasserstein_duration", "wasserstein_energy",
    "kl_hour", "kl_duration", "kl_energy",
    "ks_stat_hour", "ks_pvalue_hour",
    "ks_stat_duration", "ks_pvalue_duration",
    "ks_stat_energy", "ks_pvalue_energy",
    "crps_total_energy", "crps_mean_duration",
    "status", "n_pool_sessions", "n_pool_occurrences",
]

#: Recognised skip reasons: the two prescribed by Section 3.1, plus the
#: harness-level zero-target-sessions guard and defensive fit-failure
#: reasons for genuinely unexpected model errors (never conflated with the
#: two prescribed reasons above).
SKIP_REASONS: set[str] = {
    "insufficient_history",
    "insufficient_volume",
    "no_target_sessions",
    "persistence_fit_failed",
    "gmm_fit_failed",
}


def eval_window_for(dataset_name: str, default: int = DEFAULT_EVAL_WINDOW_DAYS) -> int:
    """Return the evaluation window length (days) for ``dataset_name``,
    applying :data:`EVAL_WINDOW_OVERRIDES` when present."""
    return EVAL_WINDOW_OVERRIDES.get(dataset_name, default)


# ---------------------------------------------------------------------------
# CRPS (Section 3.3)
# ---------------------------------------------------------------------------

def crps_ensemble(samples: np.ndarray, true_value: float) -> float:
    """
    Standard unbiased empirical CRPS estimator from an ensemble; no extra
    dependency needed.

    Parameters
    ----------
    samples : array-like
        Ensemble draws of an aggregate statistic (e.g. one total-energy
        value per Monte Carlo scenario).
    true_value : float
        The single realized value being scored against.

    Returns
    -------
    float
        CRPS (lower is better). ``nan`` if ``samples`` is empty.
    """
    samples = np.asarray(samples, dtype=float)
    m = len(samples)
    if m == 0 or np.isnan(true_value):
        return float("nan")
    term1 = np.mean(np.abs(samples - true_value))
    if m == 1:
        return float(term1)
    term2 = np.mean(np.abs(samples[:, None] - samples[None, :])) / 2
    return float(term1 - term2)


# ---------------------------------------------------------------------------
# Row builders
# ---------------------------------------------------------------------------

def _skip_row(
    dataset: str, origin: pd.Timestamp, target_date: pd.Timestamp, day_offset: int,
    X: int, true_count: int, status: str,
    n_pool_sessions: float = np.nan, n_pool_occurrences: float = np.nan,
) -> dict:
    row = {c: np.nan for c in RESULT_COLUMNS}
    row.update(
        dataset=dataset, origin=origin, target_date=target_date, day_offset=day_offset,
        weekday=target_date.dayofweek, X=X, true_count=true_count, status=status,
        n_pool_sessions=n_pool_sessions, n_pool_occurrences=n_pool_occurrences,
    )
    return row


def _ok_row(
    dataset: str, origin: pd.Timestamp, target_date: pd.Timestamp, day_offset: int,
    X: int, method: str, scenario: int, true_count: int,
    dist_cmp: pd.DataFrame, crps_total_energy: float, crps_mean_duration: float,
    n_pool_sessions: int, n_pool_occurrences: int,
) -> dict:
    row = {c: np.nan for c in RESULT_COLUMNS}
    metrics: dict = {}
    for _, r in dist_cmp.iterrows():
        feat = r["feature"]  # 'hour' | 'duration' | 'energy'
        metrics[f"wasserstein_{feat}"] = r["wasserstein"]
        metrics[f"kl_{feat}"] = r["kl_divergence"]
        metrics[f"ks_stat_{feat}"] = r["ks_statistic"]
        metrics[f"ks_pvalue_{feat}"] = r["ks_pvalue"]
    row.update(
        dataset=dataset, origin=origin, target_date=target_date, day_offset=day_offset,
        weekday=target_date.dayofweek, X=X, method=method, scenario=scenario,
        true_count=true_count, status="ok",
        crps_total_energy=crps_total_energy, crps_mean_duration=crps_mean_duration,
        n_pool_sessions=n_pool_sessions, n_pool_occurrences=n_pool_occurrences,
        **metrics,
    )
    return row


# ---------------------------------------------------------------------------
# Per-cell evaluation (one (origin, day_offset, X) combination)
# ---------------------------------------------------------------------------

def _evaluate_cell(
    dataset_name: str,
    train_df: pd.DataFrame,
    origin: pd.Timestamp,
    target_date: pd.Timestamp,
    day_offset: int,
    X: int,
    true_sessions: pd.DataFrame,
    n_scenarios: int,
    min_sessions_for_fit: int,
    n_components: int,
    random_state: int,
) -> list[dict]:
    """Evaluate a single (origin, day_offset, X) cell for both arms.

    Returns a list of result rows: either one skip row, or
    ``n_scenarios`` ok rows per successfully-fitted arm (1 or 2 arms).
    """
    rows: list[dict] = []
    true_count = len(true_sessions)

    pool, info = sessions_in_last_n_occurrences(train_df, target_date, n=X)
    n_pool_sessions = info["n_sessions"]
    n_pool_occurrences = info["n_available_occurrences"]

    # -- Skip reason 1: not enough calendar occurrences exist at all. -----
    if info["insufficient_history"]:
        rows.append(_skip_row(
            dataset_name, origin, target_date, day_offset, X, true_count,
            "insufficient_history", n_pool_sessions, n_pool_occurrences,
        ))
        return rows

    # -- Skip reason 2: occurrences exist, but pooled sessions too thin. --
    if n_pool_sessions < min_sessions_for_fit:
        rows.append(_skip_row(
            dataset_name, origin, target_date, day_offset, X, true_count,
            "insufficient_volume", n_pool_sessions, n_pool_occurrences,
        ))
        return rows

    # -- Harness guard: nothing realized on the target day to compare against.
    if true_count == 0:
        rows.append(_skip_row(
            dataset_name, origin, target_date, day_offset, X, true_count,
            "no_target_sessions", n_pool_sessions, n_pool_occurrences,
        ))
        return rows

    # Same pool feeds both arms (Section 1, assumption 2). GMM consumes the
    # pool's own "hour" column; PersistenceSessionSampler expects
    # "arrival_hour" -- rename, don't refit or re-window.
    pool_persistence = pool.rename(columns={"hour": "arrival_hour"})

    arms: list[tuple[str, object]] = []

    try:
        persistence = PersistenceSessionSampler(random_state=random_state).fit(pool_persistence)
        arms.append(("persistence", persistence))
    except (ValueError, RuntimeError) as e:
        logger.warning("Persistence fit failed for %s X=%d target=%s: %s",
                        dataset_name, X, target_date, e)
        rows.append(_skip_row(
            dataset_name, origin, target_date, day_offset, X, true_count,
            "persistence_fit_failed", n_pool_sessions, n_pool_occurrences,
        ))

    try:
        # stratify_by=["day_of_week"], never [] -- Section 1.3: the pool is
        # already filtered to a single weekday by construction, and an
        # empty stratify_by list crashes pandas groupby([]).
        # n_components=1, not "auto" -- Section 1.3: the default BIC search
        # (min_components=2) is a poor fit for small windowed pools.
        gmm = EVSessionGMM(
            n_components=n_components, stratify_by=["day_of_week"],
            random_state=random_state,
        ).fit(pool)
        arms.append(("gmm", gmm))
    except RuntimeError as e:
        logger.warning("GMM fit failed for %s X=%d target=%s: %s",
                        dataset_name, X, target_date, e)
        rows.append(_skip_row(
            dataset_name, origin, target_date, day_offset, X, true_count,
            "gmm_fit_failed", n_pool_sessions, n_pool_occurrences,
        ))

    true_energy_total = float(true_sessions["energy"].sum())
    true_duration_mean = float(true_sessions["duration"].mean())

    for method_name, model in arms:
        scenario_dists: list[pd.DataFrame] = []
        energies = np.empty(n_scenarios)
        durations = np.empty(n_scenarios)

        for scenario in range(n_scenarios):
            # seed=scenario, matching Section 3.1's pseudocode literally so
            # both arms are evaluated against the same scenario-index grid.
            sampled = model.sample(n_sessions=true_count, date=target_date, seed=scenario)
            dist_cmp = distribution_comparison(
                true_sessions, sampled, features=["hour", "duration", "energy"]
            )
            scenario_dists.append(dist_cmp)
            energies[scenario] = sampled["energy"].sum()
            durations[scenario] = sampled["duration"].mean()

        crps_energy = crps_ensemble(energies, true_energy_total)
        crps_duration = crps_ensemble(durations, true_duration_mean)

        for scenario, dist_cmp in enumerate(scenario_dists):
            rows.append(_ok_row(
                dataset_name, origin, target_date, day_offset, X, method_name, scenario,
                true_count, dist_cmp, crps_energy, crps_duration,
                n_pool_sessions, n_pool_occurrences,
            ))

    return rows


# ---------------------------------------------------------------------------
# Main harness entry point
# ---------------------------------------------------------------------------

def run_rolling_origin_benchmark(
    df: pd.DataFrame,
    dataset_name: str,
    x_grid: Sequence[int] = DEFAULT_X_GRID,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    eval_window_days: int | None = None,
    min_sessions_for_fit: int = MIN_SESSIONS_FOR_FIT,
    n_scenarios: int = DEFAULT_N_SCENARIOS,
    n_components: int = 1,
    step_days: int = 1,
    random_state: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run the rolling-origin persistence-vs-GMM benchmark for one dataset.

    For every origin in a sliding window near the dataset's chronological
    end, trains once on all data <= origin (no retraining mid-horizon), then
    evaluates each day_offset in ``horizons`` (each getting its own
    last-X-occurrences pool, still built from data <= origin) for every
    history depth in ``x_grid``.

    Parameters
    ----------
    df : pd.DataFrame
        Validated sessions DataFrame (output of ``gears.load_sessions``).
    dataset_name : str
        Used to tag rows and to look up an eval-window override
        (:data:`EVAL_WINDOW_OVERRIDES`).
    x_grid : sequence of int
        History depths to sweep (default: Section 1's 7-value grid).
    horizons : sequence of int
        day_offset values evaluated from each origin (default: 1, 2, 3).
    eval_window_days : int, optional
        Length of the sliding evaluation window. Defaults to
        :func:`eval_window_for` (30, or a dataset-specific override).
    min_sessions_for_fit : int
        Minimum pooled sessions to attempt a fit (default 10, matching
        EVSessionGMM's own convention).
    n_scenarios : int
        Monte Carlo scenario draws per (origin, day_offset, X, method) cell.
    n_components : int
        Fixed GMM component count for windowed fits (default 1; see
        Section 1.3 for why "auto" is a poor fit here).
    step_days : int
        Stride between consecutive origins (1 = every day).
    random_state : int
        Base random seed for both arms.
    verbose : bool
        Show a progress bar over origins.

    Returns
    -------
    pd.DataFrame
        Tidy long-format results, columns = :data:`RESULT_COLUMNS`. One row
        per scenario draw for "ok" cells; one row per skipped cell
        otherwise (see module docstring for the skip reasons).
    """
    if eval_window_days is None:
        eval_window_days = eval_window_for(dataset_name)

    max_h = max(horizons)
    df = df.sort_values("arrival_time").reset_index(drop=True)
    normalized_dates = df["arrival_time"].dt.normalize()

    dataset_end = normalized_dates.max()
    first_origin = dataset_end - pd.Timedelta(days=eval_window_days)
    last_origin = dataset_end - pd.Timedelta(days=max_h)
    if last_origin < first_origin:
        raise ValueError(
            f"{dataset_name}: eval_window_days={eval_window_days} is too short "
            f"for max horizon {max_h} (need eval_window_days >= max(horizons))."
        )

    origins = pd.date_range(first_origin, last_origin, freq=f"{step_days}D")

    # Precompute per-day session groups once, for O(1) "true_sessions" lookup
    # -- this is pure bookkeeping (it does not change which rows are
    # visible to a given origin's train_df; that filter is applied fresh
    # per origin below).
    by_date = {d: g for d, g in df.groupby(normalized_dates)}
    empty_true = df.iloc[0:0]

    rows: list[dict] = []

    iterator = origins
    if verbose:
        try:
            from tqdm import tqdm
            iterator = tqdm(origins, desc=f"{dataset_name} (eval_window={eval_window_days}d)")
        except ImportError:
            pass

    for origin in iterator:
        # Nothing after origin is ever visible -- trained once, reused for
        # every day_offset in this origin's horizon (no retraining
        # mid-horizon, Section 3.1).
        train_df = df[normalized_dates <= origin]

        for day_offset in horizons:
            target_date = origin + pd.Timedelta(days=day_offset)
            true_sessions = by_date.get(target_date, empty_true)

            for X in x_grid:
                rows.extend(_evaluate_cell(
                    dataset_name, train_df, origin, target_date, day_offset, X,
                    true_sessions, n_scenarios, min_sessions_for_fit,
                    n_components, random_state,
                ))

    result = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    if verbose:
        n_ok = (result["status"] == "ok").sum()
        n_skip = (result["status"] != "ok").sum()
        logger.info("%s: %d ok rows, %d skip rows.", dataset_name, n_ok, n_skip)
    return result


def run_benchmark_for_datasets(
    datasets: dict[str, pd.DataFrame],
    exclude: Sequence[str] = tuple(EXCLUDED_DATASETS),
    **kwargs,
) -> pd.DataFrame:
    """
    Run :func:`run_rolling_origin_benchmark` across several datasets and
    concatenate into a single tidy results table.

    Parameters
    ----------
    datasets : dict[str, pd.DataFrame]
        Mapping of dataset name -> validated sessions DataFrame.
    exclude : sequence of str
        Dataset names to skip (default: :data:`EXCLUDED_DATASETS`, i.e.
        ``acn`` -- the union of caltech/jpl/office; see Section 1.7).
    **kwargs
        Forwarded to :func:`run_rolling_origin_benchmark`.

    Returns
    -------
    pd.DataFrame
        Concatenation of all per-dataset results (single combined tidy
        table, Section 3.4).
    """
    frames = []
    for name, df in datasets.items():
        if name in exclude:
            logger.info("Skipping %s (excluded).", name)
            continue
        frames.append(run_rolling_origin_benchmark(df, name, **kwargs))
    if not frames:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Secondary check: full pipeline with real SARIMA counts (Section 3.2)
# ---------------------------------------------------------------------------

def run_sarima_sanity_check(
    df: pd.DataFrame,
    dataset_name: str,
    origins: Sequence[pd.Timestamp],
    horizons: Sequence[int] = (1, 2, 3),
    n_scenarios: int = 10,
    random_state: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Lightweight, secondary integration/plumbing check -- NOT the main
    scientific result (Section 3.2). For each of a small number of
    ``origins``, fits the actual ``GEARSModel`` end-to-end (GMM + the real
    SARIMA/SessionForecaster count forecaster) on data <= origin, then
    compares forecasted vs. true session counts for the next
    ``max(horizons)`` days.

    This bypasses the windowed persistence-vs-GMM comparison entirely: it
    exists to confirm the full pipeline (unrelated to the windowing logic
    under test above) still runs end-to-end and produces sane counts, using
    ``gears.utils.forecast_metrics``.

    Parameters
    ----------
    df : pd.DataFrame
        Validated sessions DataFrame.
    dataset_name : str
    origins : sequence of pd.Timestamp
        A small number of origins to check (Section 3.2 recommends 1-2 per
        dataset) -- this is deliberately not swept over the full grid.
    horizons : sequence of int
    n_scenarios : int
        Forecaster scenario draws (kept small -- this is a plumbing check).
    random_state : int
    verbose : bool

    Returns
    -------
    pd.DataFrame
        One row per (origin, day_offset): dataset, origin, target_date,
        day_offset, true_count, forecast_mean_count, status.
    """
    from gears.pipeline import GEARSModel

    df = df.sort_values("arrival_time").reset_index(drop=True)
    normalized_dates = df["arrival_time"].dt.normalize()
    by_date = {d: g for d, g in df.groupby(normalized_dates)}
    empty_true = df.iloc[0:0]
    max_h = max(horizons)

    rows: list[dict] = []
    for origin in origins:
        train_df = df[normalized_dates <= origin]
        model = GEARSModel(n_components=1, forecaster_method="sarima", random_state=random_state)
        try:
            model.fit(train_df, verbose=False)
        except Exception as e:  # noqa: BLE001 - sanity check must not abort the benchmark on any model failure
            logger.warning("SARIMA sanity check: fit failed for %s origin=%s: %s",
                            dataset_name, origin, e)
            rows.append({"dataset": dataset_name, "origin": origin, "target_date": pd.NaT,
                              "day_offset": np.nan, "true_count": np.nan,
                              "forecast_mean_count": np.nan, "status": f"fit_failed: {e}"})
            continue

        try:
            sim = model.simulate_short_term(
                start_date=origin + pd.Timedelta(days=1), horizon=max_h,
                n_scenarios=n_scenarios, seed=random_state,
            )
        except Exception as e:  # noqa: BLE001 - sanity check must not abort the benchmark on any simulate failure
            logger.warning("SARIMA sanity check: simulate failed for %s origin=%s: %s",
                            dataset_name, origin, e)
            rows.append({"dataset": dataset_name, "origin": origin, "target_date": pd.NaT,
                              "day_offset": np.nan, "true_count": np.nan,
                              "forecast_mean_count": np.nan, "status": f"simulate_failed: {e}"})
            continue

        for day_offset in horizons:
            target_date = origin + pd.Timedelta(days=day_offset)
            true_count = len(by_date.get(target_date, empty_true))
            if not sim.empty and "date" in sim.columns and "scenario" in sim.columns:
                day_sim = sim[sim["date"] == target_date.date()]
                counts_by_scenario = day_sim.groupby("scenario").size()
                counts_by_scenario = counts_by_scenario.reindex(range(n_scenarios), fill_value=0)
                forecast_mean = float(counts_by_scenario.mean())
            else:
                forecast_mean = 0.0
            rows.append({
                "dataset": dataset_name, "origin": origin, "target_date": target_date,
                "day_offset": day_offset, "true_count": true_count,
                "forecast_mean_count": forecast_mean, "status": "ok",
            })

    return pd.DataFrame(rows)


def summarize_sarima_sanity_check(sanity_df: pd.DataFrame) -> dict:
    """
    Aggregate :func:`run_sarima_sanity_check` output into overall
    ``forecast_metrics`` (RMSE/MAE/MAPE/sMAPE/bias) on forecasted vs. true
    counts, using ``gears.utils.forecast_metrics``.

    Parameters
    ----------
    sanity_df : pd.DataFrame
        Output of :func:`run_sarima_sanity_check`.

    Returns
    -------
    dict
        ``forecast_metrics`` output, or ``{}`` if no "ok" rows are present.
    """
    ok = sanity_df[sanity_df["status"] == "ok"]
    if ok.empty:
        return {}
    y_true = ok["true_count"].to_numpy(dtype=float)
    y_pred = ok["forecast_mean_count"].to_numpy(dtype=float)
    return forecast_metrics(y_true, y_pred)

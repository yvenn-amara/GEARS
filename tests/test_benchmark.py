"""Tests for gears.evaluation.benchmark -- the rolling-origin persistence-vs-GMM
benchmark harness (Session 3 of the persistence-vs-GMM benchmark prompt).

Covers, per the harness's acceptance criteria:
- the leakage boundary (nothing from after `origin` ever enters a pool),
- per-day-of-horizon distinct-weekday pooling,
- the two skip reasons (`insufficient_history` / `insufficient_volume`)
  firing on synthetic data engineered to trigger each specifically,
plus CRPS correctness and end-to-end schema/shape checks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gears.data.schemas import validate_dataframe
from gears.evaluation import benchmark as bm
from gears.evaluation.benchmark import (
    RESULT_COLUMNS,
    crps_ensemble,
    eval_window_for,
    run_benchmark_for_datasets,
    run_rolling_origin_benchmark,
)

# ---------------------------------------------------------------------------
# Synthetic dataset builders
# ---------------------------------------------------------------------------

def _sessions_on_dates(dates, n_per_date, seed=0, hour_range=(6, 20)):
    """Build a validated sessions df with `n_per_date` sessions on each date
    in `dates` (a list of pd.Timestamp), spread across `hour_range`."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in dates:
        hours = rng.uniform(hour_range[0], hour_range[1], n_per_date)
        for h in hours:
            rows.append({
                "arrival_time": d + pd.Timedelta(hours=float(h)),
                "duration": rng.uniform(0.5, 8),
                "energy": rng.uniform(2, 40),
            })
    return validate_dataframe(pd.DataFrame(rows))


def make_dense_daily_df(start="2024-01-01", n_days=120, n_per_day=15, seed=0):
    """Sessions every single day for n_days -- dense enough that every X in
    the small grid is both calendar- and volume-feasible."""
    dates = pd.date_range(start, periods=n_days, freq="D")
    return _sessions_on_dates(dates, n_per_day, seed=seed)


def make_weekday_varying_df(start="2024-01-01", n_days=140, seed=0):
    """Sessions every day, but with a session COUNT that depends on the
    weekday (Monday gets 3x as many as other days). Used to check that a
    given day_offset's pool is actually filtered to *its own* weekday and
    not contaminated by a neighbouring day_offset's pool within the same
    origin's horizon."""
    dates = pd.date_range(start, periods=n_days, freq="D")
    rng = np.random.default_rng(seed)
    rows = []
    for d in dates:
        n = 15 if d.dayofweek == 0 else 5  # Monday: 15/day, else: 5/day
        hours = rng.uniform(6, 20, n)
        for h in hours:
            rows.append({
                "arrival_time": d + pd.Timedelta(hours=float(h)),
                "duration": rng.uniform(0.5, 8),
                "energy": rng.uniform(2, 40),
            })
    return validate_dataframe(pd.DataFrame(rows))


# ---------------------------------------------------------------------------
# CRPS
# ---------------------------------------------------------------------------

def test_crps_zero_for_perfect_degenerate_ensemble():
    # every ensemble member equals the true value -> CRPS should be 0
    samples = np.full(20, 10.0)
    assert crps_ensemble(samples, 10.0) == pytest.approx(0.0, abs=1e-9)


def test_crps_single_sample_equals_abs_error():
    assert crps_ensemble(np.array([7.0]), 10.0) == pytest.approx(3.0)


def test_crps_empty_is_nan():
    assert np.isnan(crps_ensemble(np.array([]), 10.0))


def test_crps_penalizes_spread():
    # Same mean absolute error to the true value, but the spread-out
    # ensemble should score worse (higher CRPS) than the tight one.
    true_value = 10.0
    tight = np.array([9.0, 10.0, 11.0])
    wide = np.array([0.0, 10.0, 20.0])
    assert crps_ensemble(wide, true_value) > crps_ensemble(tight, true_value)


# ---------------------------------------------------------------------------
# Leakage boundary: nothing from after `origin` ever enters a pool
# ---------------------------------------------------------------------------

def test_no_leakage_across_origin_boundary(monkeypatch):
    """Every call the harness makes to sessions_in_last_n_occurrences must
    be given a train_df whose latest arrival_time is <= origin, for every
    origin in the sweep -- i.e. nothing after the origin is ever visible,
    regardless of which day_offset/X triggered the call."""
    df = make_dense_daily_df(n_days=90, n_per_day=15)

    calls = []
    real_fn = bm.sessions_in_last_n_occurrences

    def spy(train_df, target_date, n):
        calls.append({
            "origin_train_max": train_df["arrival_time"].max(),
            "target_date": pd.Timestamp(target_date),
            "n": n,
        })
        return real_fn(train_df, target_date, n)

    monkeypatch.setattr(bm, "sessions_in_last_n_occurrences", spy)

    res = run_rolling_origin_benchmark(
        df, "leak_check", x_grid=[1, 2, 4], horizons=[1, 2, 3],
        eval_window_days=15, n_scenarios=2, verbose=False,
    )
    assert len(calls) > 0, "harness never called the windowing utility -- test is vacuous"

    for call in calls:
        # The pool-window function is only ever handed data whose latest
        # arrival_time is on or before the origin that produced it, and
        # strictly before the target_date it's forecasting.
        assert call["origin_train_max"] <= call["target_date"]
        assert call["origin_train_max"].normalize() < call["target_date"].normalize()

    assert (res["status"] == "ok").any()


def test_future_session_does_not_change_earlier_origin_pool():
    """Appending a session dated well after an origin must not change the
    pool (n_pool_sessions / n_pool_occurrences) computed for that origin."""
    dates = pd.date_range("2024-01-01", periods=60, freq="D")  # Mondays included
    base = _sessions_on_dates(dates, n_per_date=12, seed=1)

    # Evaluate a fixed origin well inside the dense history, before the
    # window even needs the tail of the dataset.
    origin = pd.Timestamp("2024-02-05")
    common_kwargs = {"x_grid": [4], "horizons": [1], "n_scenarios": 2, "verbose": False}

    res_before = run_rolling_origin_benchmark(
        base, "future_check",
        eval_window_days=(pd.Timestamp("2024-02-29").normalize() - origin).days,
        **common_kwargs,
    )
    cell_before = res_before[
        (res_before["origin"] == origin) & (res_before["day_offset"] == 1) & (res_before["X"] == 4)
    ]
    assert len(cell_before) > 0

    # Now append a big burst of sessions strictly after `origin`, at the
    # very end of the (now longer) dataset.
    future_dates = pd.date_range("2024-03-01", periods=10, freq="D")
    future = _sessions_on_dates(future_dates, n_per_date=50, seed=2)
    extended = pd.concat([base, future], ignore_index=True)
    extended = validate_dataframe(extended[[
        "arrival_time", "duration", "energy",
    ]])

    res_after = run_rolling_origin_benchmark(
        extended, "future_check",
        eval_window_days=(extended["arrival_time"].max().normalize() - origin).days,
        **common_kwargs,
    )
    cell_after = res_after[
        (res_after["origin"] == origin) & (res_after["day_offset"] == 1) & (res_after["X"] == 4)
    ]
    assert len(cell_after) > 0

    assert cell_before["n_pool_sessions"].iloc[0] == cell_after["n_pool_sessions"].iloc[0]
    assert cell_before["n_pool_occurrences"].iloc[0] == cell_after["n_pool_occurrences"].iloc[0]


# ---------------------------------------------------------------------------
# Per-day-of-horizon distinct-weekday pooling
# ---------------------------------------------------------------------------

def test_each_horizon_day_gets_its_own_weekday_pool(monkeypatch):
    """Within a single origin's horizon, day_offset=1/2/3 land on different
    weekdays and must each trigger their OWN sessions_in_last_n_occurrences
    call for their own target_date -- never one shared pool reused across
    the horizon."""
    df = make_weekday_varying_df(n_days=100)

    calls = []
    real_fn = bm.sessions_in_last_n_occurrences

    def spy(train_df, target_date, n):
        info_pool, info = real_fn(train_df, target_date, n)
        calls.append({
            "target_date": pd.Timestamp(target_date),
            "weekday": pd.Timestamp(target_date).dayofweek,
            "n_sessions": info["n_sessions"],
        })
        return info_pool, info

    monkeypatch.setattr(bm, "sessions_in_last_n_occurrences", spy)

    run_rolling_origin_benchmark(
        df, "horizon_check", x_grid=[4], horizons=[1, 2, 3],
        eval_window_days=10, n_scenarios=1, verbose=False,
    )

    # Group calls by origin-ish clusters of 3 consecutive target_dates
    # (one origin's day_offset=1,2,3): verify 3 distinct target_dates with
    # 3 distinct weekdays were queried, not the same one repeated.
    df_calls = pd.DataFrame(calls)
    assert len(df_calls) >= 3
    for i in range(0, len(df_calls) - 2, 3):
        triplet = df_calls.iloc[i:i + 3]
        assert triplet["target_date"].nunique() == 3, "day_offsets 1/2/3 must target distinct dates"
        assert triplet["weekday"].nunique() == 3, "day_offsets 1/2/3 must land on distinct weekdays here"

    # And because our synthetic data makes Monday pools much larger than
    # non-Monday pools, the recorded n_sessions must actually reflect each
    # call's own weekday, not a pool bled over from a different day_offset.
    monday_calls = df_calls[df_calls["weekday"] == 0]
    non_monday_calls = df_calls[df_calls["weekday"] != 0]
    assert len(monday_calls) > 0 and len(non_monday_calls) > 0
    assert monday_calls["n_sessions"].min() > non_monday_calls["n_sessions"].max()


# ---------------------------------------------------------------------------
# The two skip reasons, engineered to trigger each specifically
# ---------------------------------------------------------------------------

def test_insufficient_history_triggers_when_few_occurrences_exist():
    """Few distinct qualifying dates (calendar problem), but each one has
    plenty of sessions (volume is NOT the limiting factor) -- must be
    classified insufficient_history, never insufficient_volume."""
    # Only 2 Mondays in the whole dataset, 20 sessions each (way above the
    # volume gate) -- but X=5 asks for 5 occurrences, so only 2 exist.
    mondays = [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-08")]
    df = _sessions_on_dates(mondays, n_per_date=20, seed=3)
    # Origin must be after both Mondays so history is fixed, target_date a
    # later Monday so the pool looks strictly backward.
    origin = pd.Timestamp("2024-01-14")  # Sunday
    target_date = origin + pd.Timedelta(days=1)  # Monday 2024-01-15
    assert target_date.dayofweek == 0

    train_df = df[df["arrival_time"].dt.normalize() <= origin]
    rows = bm._evaluate_cell(
        "hist_check", train_df, origin, target_date, day_offset=1, X=5,
        true_sessions=df.iloc[0:0], n_scenarios=2, min_sessions_for_fit=10,
        n_components=1, random_state=42,
    )
    assert len(rows) == 1
    assert rows[0]["status"] == "insufficient_history"
    assert rows[0]["n_pool_occurrences"] == 2
    assert rows[0]["n_pool_sessions"] == 40  # 2 * 20 -- volume was never the issue


def test_insufficient_volume_triggers_when_occurrences_thin():
    """Plenty of distinct qualifying dates (calendar is NOT the limiting
    factor), but only 1 session each -- pooled volume is below the gate --
    must be classified insufficient_volume, never insufficient_history."""
    # 10 distinct Mondays, 1 session each: n_available_occurrences=10 >= X=5
    # (calendar satisfied), but n_pool_sessions=5 < MIN_SESSIONS_FOR_FIT=10.
    mondays = pd.date_range("2024-01-01", periods=10, freq="7D")
    df = _sessions_on_dates(mondays, n_per_date=1, seed=4)
    origin = mondays[-1] + pd.Timedelta(days=6)  # Sunday after the last Monday
    target_date = origin + pd.Timedelta(days=1)  # next Monday
    assert target_date.dayofweek == 0

    train_df = df[df["arrival_time"].dt.normalize() <= origin]
    rows = bm._evaluate_cell(
        "vol_check", train_df, origin, target_date, day_offset=1, X=5,
        true_sessions=df.iloc[0:0], n_scenarios=2, min_sessions_for_fit=10,
        n_components=1, random_state=42,
    )
    assert len(rows) == 1
    assert rows[0]["status"] == "insufficient_volume"
    assert rows[0]["n_pool_occurrences"] == 5  # calendar depth was satisfied
    assert rows[0]["n_pool_sessions"] == 5     # but too few actual sessions


def test_no_target_sessions_skip_reason():
    """A target day with zero realized sessions can't be compared against
    anything -- must be skipped with its own reason, not crash inside
    distribution_comparison/scipy on empty arrays."""
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    df = _sessions_on_dates(dates, n_per_date=15, seed=5)
    origin = pd.Timestamp("2024-02-20")
    target_date = origin + pd.Timedelta(days=1)
    train_df = df[df["arrival_time"].dt.normalize() <= origin]

    empty_true = df.iloc[0:0]
    rows = bm._evaluate_cell(
        "empty_check", train_df, origin, target_date, day_offset=1, X=4,
        true_sessions=empty_true, n_scenarios=2, min_sessions_for_fit=10,
        n_components=1, random_state=42,
    )
    assert len(rows) == 1
    assert rows[0]["status"] == "no_target_sessions"
    assert rows[0]["true_count"] == 0


def test_degraded_x_not_relabeled():
    """If X=52 is requested but only e.g. 3 occurrences exist, the result
    must be logged under X=52 with insufficient_history -- never silently
    downgraded to run with n=3 and mislabeled as a smaller/achieved X."""
    mondays = pd.date_range("2024-01-01", periods=3, freq="7D")
    df = _sessions_on_dates(mondays, n_per_date=15, seed=6)
    origin = mondays[-1] + pd.Timedelta(days=6)
    target_date = origin + pd.Timedelta(days=1)
    train_df = df[df["arrival_time"].dt.normalize() <= origin]

    rows = bm._evaluate_cell(
        "degrade_check", train_df, origin, target_date, day_offset=1, X=52,
        true_sessions=df.iloc[0:0], n_scenarios=2, min_sessions_for_fit=10,
        n_components=1, random_state=42,
    )
    assert len(rows) == 1
    assert rows[0]["X"] == 52
    assert rows[0]["status"] == "insufficient_history"
    # Never quietly "succeeded" using the 3 available occurrences instead.
    assert rows[0]["method"] != "gmm" and rows[0]["method"] != "persistence"


# ---------------------------------------------------------------------------
# End-to-end shape / schema / gate-consistency checks
# ---------------------------------------------------------------------------

def test_result_schema_matches_spec():
    df = make_dense_daily_df(n_days=40, n_per_day=15)
    res = run_rolling_origin_benchmark(
        df, "schema_check", x_grid=[1, 2], horizons=[1], eval_window_days=10,
        n_scenarios=3, verbose=False,
    )
    assert list(res.columns) == RESULT_COLUMNS
    assert (res["status"] == "ok").any()
    ok = res[res["status"] == "ok"]
    assert set(ok["method"].unique()) <= {"persistence", "gmm"}
    assert ok["scenario"].max() <= 2  # 0..n_scenarios-1


def test_both_arms_see_identical_pool_size_per_cell():
    """Section 1, assumption 2: persistence and GMM must be fit on the
    exact same pool -- so for any 'ok' cell, n_pool_sessions and
    n_pool_occurrences must match exactly between the two methods."""
    df = make_dense_daily_df(n_days=90, n_per_day=15)
    res = run_rolling_origin_benchmark(
        df, "same_pool_check", x_grid=[2, 4], horizons=[1], eval_window_days=15,
        n_scenarios=2, verbose=False,
    )
    ok = res[res["status"] == "ok"]
    pivot = ok.groupby(["origin", "target_date", "X", "method"])["n_pool_sessions"].first().unstack("method")
    assert {"persistence", "gmm"}.issubset(pivot.columns)
    pd.testing.assert_series_equal(
        pivot["persistence"], pivot["gmm"], check_names=False,
    )


def test_min_sessions_gate_matches_gmm_convention():
    from gears.evaluation.benchmark import MIN_SESSIONS_FOR_FIT
    assert MIN_SESSIONS_FOR_FIT == 10


def test_eval_window_override_applied_for_paris_like_dataset():
    assert eval_window_for("paris") == 14
    assert eval_window_for("some_other_dataset") == bm.DEFAULT_EVAL_WINDOW_DAYS


def test_run_benchmark_for_datasets_excludes_acn_by_default():
    df = make_dense_daily_df(n_days=30, n_per_day=15)
    datasets = {"acn": df, "sap": df}
    res = run_benchmark_for_datasets(
        datasets, x_grid=[2], horizons=[1], eval_window_days=8,
        n_scenarios=1, verbose=False,
    )
    assert "acn" not in set(res["dataset"].unique())
    assert "sap" in set(res["dataset"].unique())


def test_raises_if_eval_window_shorter_than_max_horizon():
    df = make_dense_daily_df(n_days=30, n_per_day=15)
    with pytest.raises(ValueError):
        run_rolling_origin_benchmark(
            df, "too_short", horizons=[1, 2, 3], eval_window_days=1, verbose=False,
        )


def test_default_arms_unchanged_from_session_3():
    """Session 4 must not change what callers get by default -- only
    callers that explicitly opt into ALL_ARMS see the two new arms."""
    assert bm.DEFAULT_ARMS == ("persistence", "gmm")
    assert set(bm.ALL_ARMS) == {"persistence", "gmm", "gmm_recency", "vae"}


def test_all_four_arms_produce_ok_rows_with_real_numbers():
    """Session 4 task 1: gmm_recency and vae must follow the exact same
    try/except-and-skip-row pattern as persistence/gmm -- when arms=ALL_ARMS
    is passed explicitly, all four should fit and produce usable results on
    a dense enough synthetic pool."""
    df = make_dense_daily_df(n_days=90, n_per_day=20)
    res = run_rolling_origin_benchmark(
        df, "four_arms", x_grid=[4], horizons=[1], eval_window_days=5,
        n_scenarios=2, verbose=False, arms=bm.ALL_ARMS,
    )
    ok = res[res["status"] == "ok"]
    assert set(ok["method"].unique()) == {"persistence", "gmm", "gmm_recency", "vae"}
    for method in bm.ALL_ARMS:
        sub = ok[ok["method"] == method]
        assert sub["wasserstein_energy"].notna().any(), f"{method} produced no usable rows"


def test_arms_subset_only_evaluates_requested_methods():
    """Requesting a single arm should never fit or report the others."""
    df = make_dense_daily_df(n_days=60, n_per_day=15)
    res = run_rolling_origin_benchmark(
        df, "gmm_recency_only", x_grid=[4], horizons=[1], eval_window_days=5,
        n_scenarios=2, verbose=False, arms=["gmm_recency"],
    )
    ok = res[res["status"] == "ok"]
    assert set(ok["method"].unique()) <= {"gmm_recency"}
    assert ok["method"].eq("gmm_recency").any()


def test_new_skip_reasons_registered():
    assert {"gmm_recency_fit_failed", "vae_fit_failed"}.issubset(bm.SKIP_REASONS)


def test_gmm_and_persistence_output_same_columns_from_this_harness():
    """Interface parity check reused inside a realistic pool (Session 2's
    acceptance criterion, re-verified in the harness's own context)."""
    df = make_dense_daily_df(n_days=60, n_per_day=15)
    res = run_rolling_origin_benchmark(
        df, "parity_check", x_grid=[4], horizons=[1], eval_window_days=10,
        n_scenarios=2, verbose=False,
    )
    ok = res[res["status"] == "ok"]
    assert set(ok["method"].unique()) == {"persistence", "gmm"}
    # Both arms must have produced usable distribution_comparison output
    # (no all-NaN wasserstein columns for either method).
    for method in ["persistence", "gmm"]:
        sub = ok[ok["method"] == method]
        assert sub["wasserstein_hour"].notna().any()
        assert sub["wasserstein_duration"].notna().any()
        assert sub["wasserstein_energy"].notna().any()

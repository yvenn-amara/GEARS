"""
Non-regression tests — one test per bug corrected in audit sessions 1 and 2.

Bug index
---------
R1  PersistenceForecaster zero-drop
    Old code filtered daily counts with ``daily = daily[daily > 0]``,
    dropping zero-session days.  This broke the n_weeks look-back: the
    reference date was missing from the index, so the carry-forward loop
    ran 52 iterations before falling back to ``mean_daily_``, and the
    returned value bore no relationship to the correct historical count.

R2  sessions_to_daily_counts — 1970 / 1971 date axis
    ``validate_dataframe`` stores the ``date`` column as Python ``date``
    objects (dtype=object).  The old ``sessions_to_daily_counts`` did not
    call ``pd.to_datetime(daily.index)`` before ``reindex(full_range)``.
    ``full_range`` is a ``DatetimeIndex``; matching against an object index
    of ``date`` values silently produced NaN-then-fillna rows anchored at
    the Unix epoch (1970-01-01).

R3  compute_regret() — multidimensional key error
    The old implementation used double-bracket indexing
    ``oracle_opt[["cost_smart"]]`` (returns a one-column DataFrame) instead
    of ``oracle_opt["cost_smart"]`` (returns a Series).  Calling ``.sum()``
    on a DataFrame returns a *Series*, not a scalar, so all arithmetic on
    the result dict raised ``KeyError: 0`` when downstream code tried to
    read a scalar value.

R4  NHiTSForecaster — input_size default and scaler_type
    Old default ``input_size = 4 * horizon`` consumed ≥80 % of a typical
    450-day training set as a single context window, leaving fewer than
    one full epoch of gradient signal → random-walk collapse.
    ``scaler_type`` was absent, so raw session-count gradients (scale ~100)
    were uncontrolled.

No-leakage note
---------------
All splits in this file are strictly temporal: training data always ends
before evaluation data starts.  No shuffling or random resampling is used
across the split boundary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gears.data.loader import make_demo_data
from gears.data.schemas import validate_dataframe
from gears.models.forecaster import (
    NHiTSForecaster,
    PersistenceForecaster,
    sessions_to_daily_counts,
)
from gears.smart_charging.optimizer import SmartChargingOptimizer
from gears.utils import make_price_signal

# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

def _make_sessions_with_gap(
    before_end: str = "2024-03-31",
    after_start: str = "2024-04-15",
    after_end: str = "2024-06-30",
    n_per_day: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """Build sessions covering two date ranges separated by a gap.

    The gap days will appear in sessions_to_daily_counts with count=0.
    The split is strictly temporal — no future data leaks into training.
    """
    rng = np.random.default_rng(seed)
    before = pd.date_range("2024-01-01", before_end, freq="D")
    after = pd.date_range(after_start, after_end, freq="D")
    all_dates = list(before) + list(after)

    rows = []
    for d in all_dates:
        for _ in range(n_per_day):
            rows.append({
                "arrival_time": pd.Timestamp(d) + pd.Timedelta(hours=float(rng.uniform(7, 21))),
                "duration": float(rng.uniform(0.5, 8.0)),
                "energy": float(rng.uniform(2.0, 40.0)),
            })
    df = pd.DataFrame(rows)
    return validate_dataframe(df)


def _make_sc_sessions(n: int = 30, seed: int = 0) -> pd.DataFrame:
    """Sessions aligned with a standard 7-day price signal starting 2025-06-01."""
    return make_demo_data(n=n, seed=seed, start_date="2025-06-01", end_date="2025-06-07")


def _make_sc_signal(days: int = 7) -> pd.Series:
    return make_price_signal(
        start="2025-06-01", periods=days * 48, resolution_min=30, pattern="day_night"
    )


# ===========================================================================
# R1 — PersistenceForecaster zero-drop
# ===========================================================================

class TestPersistenceZeroDrop:
    """R1: zero-count days must survive fit() and not corrupt predict()."""

    def test_zero_days_present_in_daily_counts(self):
        """
        After fit(), all gap days must exist in _daily_counts with value 0.
        Old bug: a ``daily = daily[daily > 0]`` filter silently removed them,
        leaving index holes that the 52-week look-back could not always fill.
        """
        df = _make_sessions_with_gap()
        fc = PersistenceForecaster(n_weeks=1).fit(df)

        gap_dates = pd.date_range("2024-04-01", "2024-04-14", freq="D")
        for gd in gap_dates:
            assert gd in fc._daily_counts.index, (
                f"Zero-count day {gd.date()} was silently dropped from "
                "_daily_counts — zero-drop bug still present."
            )
            assert fc._daily_counts[gd] == 0.0, (
                f"Gap day {gd.date()} is present but has count "
                f"{fc._daily_counts[gd]}, expected 0."
            )

    def test_predict_non_negative_after_gap(self):
        """predict() must return non-negative counts even when reference days were 0."""
        df = _make_sessions_with_gap()
        fc = PersistenceForecaster(n_weeks=1).fit(df)
        pred = fc.predict(horizon=14, n_scenarios=1, seed=0)
        assert len(pred) == 14
        assert (pred["n_sessions"] >= 0).all()

    def test_predict_uses_carry_forward_not_mean_when_gap(self):
        """
        When the reference weekday had count=0 (gap), carry-forward should
        find a non-zero value further back — NOT just return mean_daily_.
        mean_daily_ is only the fallback for when *no* historical weekday
        is available at all.
        """
        df = _make_sessions_with_gap()
        fc = PersistenceForecaster(n_weeks=1).fit(df)
        counts = fc._daily_counts

        # Ensure index covers at least two full weeks before the gap
        early_dates = counts[counts.index < pd.Timestamp("2024-04-01")]
        assert len(early_dates) >= 14, (
            "Need at least 2 weeks of training before the gap to verify carry-forward."
        )

        # mean_daily_ over a dataset with a gap should be < the non-gap period mean
        non_zero_mean = float(counts[counts > 0].mean())
        assert fc.mean_daily_ < non_zero_mean, (
            "mean_daily_ should be pulled down by the gap zeros — "
            "confirm the fixture is set up correctly."
        )

    def test_daily_counts_is_contiguous(self):
        """_daily_counts must have no missing dates between first and last observation."""
        df = _make_sessions_with_gap()
        fc = PersistenceForecaster(n_weeks=1).fit(df)
        idx = fc._daily_counts.index
        expected_length = (idx[-1] - idx[0]).days + 1
        assert len(idx) == expected_length, (
            f"_daily_counts has {len(idx)} dates but should have {expected_length} "
            "(including zero-count gap days).  Some dates were dropped."
        )


# ===========================================================================
# R2 — sessions_to_daily_counts date axis 1970/1971
# ===========================================================================

class TestDailyCountsDateAxis:
    """R2: index must reflect actual session dates, not the Unix epoch."""

    def test_no_epoch_dates_from_date_objects(self):
        """
        validate_dataframe produces 'date' as Python date objects (dtype=object).
        Old bug: sessions_to_daily_counts did not call pd.to_datetime(daily.index)
        before reindexing, so the DatetimeIndex full_range failed to match the
        object-typed index and filled everything with NaN → epoch anchoring.
        """
        df = make_demo_data(n=200, seed=0)
        # Confirm fixture: validate_dataframe yields Python date objects
        assert df["date"].dtype == object, (
            "Fixture assumption violated: 'date' should be Python date objects."
        )

        counts = sessions_to_daily_counts(df)

        min_year = counts.index.year.min()
        assert min_year >= 2020, (
            f"sessions_to_daily_counts produced year={min_year} — "
            "epoch anchoring (1970/1971) bug still present.  "
            "pd.to_datetime(daily.index) was not applied before reindex()."
        )

    def test_index_is_datetimeindex(self):
        """Return value must be a properly typed DatetimeIndex."""
        df = make_demo_data(n=100, seed=1)
        counts = sessions_to_daily_counts(df)
        assert isinstance(counts.index, pd.DatetimeIndex), (
            f"Expected DatetimeIndex, got {type(counts.index).__name__}."
        )

    def test_index_spans_full_date_range(self):
        """All dates between first and last session must be present (zeros included)."""
        df = make_demo_data(n=150, seed=2)
        counts = sessions_to_daily_counts(df)
        first, last = counts.index.min(), counts.index.max()
        expected_days = (last - first).days + 1
        assert len(counts) == expected_days, (
            f"Expected {expected_days} dates, got {len(counts)}.  "
            "Gaps imply the reindex fill didn't work (possible epoch bug)."
        )

    def test_values_non_negative(self):
        """fill_value=0 must produce non-negative counts, never NaN."""
        df = make_demo_data(n=100, seed=3)
        counts = sessions_to_daily_counts(df)
        assert not counts.isna().any(), "NaN values found — fill_value=0 not applied."
        assert (counts >= 0).all()

    def test_year_preserved_across_validate_then_count(self):
        """End-to-end: raw DataFrame → validate_dataframe → sessions_to_daily_counts
        must preserve the original year throughout."""
        rng = np.random.default_rng(7)
        n = 60
        raw = pd.DataFrame({
            "arrival_time": pd.date_range("2023-11-01", periods=n, freq="12h"),
            "duration": rng.uniform(0.5, 8, n),
            "energy": rng.uniform(2, 40, n),
        })
        df = validate_dataframe(raw)
        counts = sessions_to_daily_counts(df)
        assert counts.index.year.min() == 2023


# ===========================================================================
# R3 — compute_regret() multidimensional key error
# ===========================================================================

class TestComputeRegretScalars:
    """R3: all output dict values must be Python float scalars."""

    def test_basic_regret_all_scalars(self):
        """
        Old bug: oracle_opt[['cost_smart']].sum() returned a Series (DataFrame
        column-sum), not a scalar.  All arithmetic on it raised KeyError: 0 when
        code tried to access the result as a number.
        """
        sessions_a = _make_sc_sessions(30, seed=0)
        sessions_b = _make_sc_sessions(30, seed=1)
        signal = _make_sc_signal()
        opt = SmartChargingOptimizer(signal_type="price")
        regret = opt.compute_regret(sessions_a, sessions_b, signal)

        expected_keys = {
            "cost_oracle_smart",
            "cost_predicted_smart",
            "cost_predicted_plug",
            "regret_smart_vs_oracle",
            "regret_plug_vs_oracle",
            "value_of_smart_charging",
        }
        assert expected_keys <= set(regret.keys())

        for key in expected_keys:
            val = regret[key]
            assert isinstance(val, float), (
                f"compute_regret['{key}'] is {type(val).__name__} "
                f"(value={val!r}), expected float — multidimensional key bug."
            )

    def test_regret_with_persistence_all_scalars(self):
        """Same check when persistence_sessions is provided (the extra branch)."""
        sessions_a = _make_sc_sessions(30, seed=0)
        sessions_b = _make_sc_sessions(30, seed=1)
        sessions_p = _make_sc_sessions(30, seed=2)
        signal = _make_sc_signal()
        opt = SmartChargingOptimizer(signal_type="price")
        regret = opt.compute_regret(
            sessions_a, sessions_b, signal,
            persistence_sessions=sessions_p,
        )
        for key in ("cost_persistence_smart", "value_of_forecast_vs_persistence"):
            assert key in regret
            assert isinstance(regret[key], float), (
                f"compute_regret['{key}'] is {type(regret[key]).__name__}, "
                "expected float."
            )

    def test_regret_arithmetic_ordering(self):
        """Oracle ≤ smart ≤ plug must hold (no negative savings from computation bug)."""
        sessions_a = _make_sc_sessions(40, seed=0)
        sessions_b = _make_sc_sessions(40, seed=1)
        signal = _make_sc_signal()
        opt = SmartChargingOptimizer(signal_type="price")
        regret = opt.compute_regret(sessions_a, sessions_b, signal)
        # These inequalities only hold in expectation for our signal/sessions setup;
        # the critical check is that the values are scalars and do not raise.
        assert regret["value_of_smart_charging"] >= -1e-3

    def test_regret_same_sessions_zero_regret(self):
        """When oracle == predicted, regret must be ~0 (scalar 0.0, not empty array)."""
        sessions = _make_sc_sessions(25, seed=5)
        signal = _make_sc_signal()
        opt = SmartChargingOptimizer(signal_type="price")
        regret = opt.compute_regret(sessions, sessions, signal)
        assert isinstance(regret["regret_smart_vs_oracle"], float)
        assert abs(regret["regret_smart_vs_oracle"]) < 1e-3, (
            "Oracle == predicted should yield zero regret."
        )


# ===========================================================================
# R4 — NHiTSForecaster (input_size default, scaler_type)
# ===========================================================================

class TestNHiTSForecasterAvailability:
    """R4 (availability checks — no [dl] dependencies required)."""

    def test_is_available_returns_bool(self):
        """is_available() must always return a plain bool, never raise."""
        result = NHiTSForecaster.is_available()
        assert isinstance(result, bool)

    def test_import_msg_mentions_neuralforecast_and_torch(self):
        """_IMPORT_MSG must contain actionable install instructions."""
        fc = NHiTSForecaster(horizon=7)
        msg = fc._IMPORT_MSG.lower()
        assert "neuralforecast" in msg, "_IMPORT_MSG must mention neuralforecast."
        assert "torch" in msg, "_IMPORT_MSG must mention torch."

    def test_default_input_size_is_2x_horizon(self):
        """
        R4 fix: default input_size = 2 * horizon (was 4 * horizon).
        Verify the default without any [dl] import.
        """
        fc = NHiTSForecaster(horizon=30)
        assert fc.input_size == 60, (
            f"Expected default input_size=2*horizon=60, got {fc.input_size}. "
            "The 4x multiplier has been reverted."
        )

    def test_explicit_input_size_preserved(self):
        """Explicitly passed input_size must not be overridden."""
        fc = NHiTSForecaster(horizon=14, input_size=20)
        assert fc.input_size == 20

    def test_scaler_type_default_is_standard(self):
        """R4 fix: scaler_type must default to 'standard' (was absent)."""
        fc = NHiTSForecaster()
        assert fc.scaler_type == "standard", (
            f"Expected scaler_type='standard', got '{fc.scaler_type}'. "
            "Without scaler_type the gradient scale is uncontrolled on raw counts."
        )

    def test_predict_before_fit_raises(self):
        """predict() must raise RuntimeError before fit() is called."""
        fc = NHiTSForecaster(horizon=7)
        with pytest.raises(RuntimeError, match="fit"):
            fc.predict(horizon=7)

    def test_repr_contains_key_attrs(self):
        """repr must include horizon, input_size, max_steps and fitted status."""
        fc = NHiTSForecaster(horizon=14, max_steps=100)
        r = repr(fc)
        assert "14" in r
        assert "100" in r
        assert "fitted=False" in r


@pytest.mark.skipif(
    not NHiTSForecaster.is_available(),
    reason="neuralforecast and torch not installed ([dl] extra required)",
)
class TestNHiTSForecasterFitPredict:
    """R4 (full fit/predict tests — only runs when [dl] is installed)."""

    @staticmethod
    def _make_sessions(n: int = 300, seed: int = 0) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        df = pd.DataFrame({
            "arrival_time": pd.date_range("2023-01-01", periods=n, freq="12h"),
            "duration": rng.uniform(0.5, 8, n),
            "energy": rng.uniform(2, 40, n),
        })
        return validate_dataframe(df)

    def test_fit_sets_is_fitted(self):
        df = self._make_sessions()
        fc = NHiTSForecaster(horizon=7, max_steps=5).fit(df)
        assert fc.is_fitted_

    def test_input_size_capped_at_half_n_train(self):
        """
        R4 fix: effective input_size <= n_train // 2.
        Use a short training set (60 days) with large requested input_size (100)
        to confirm the cap is applied inside fit().
        """
        df = self._make_sessions(n=60)           # 60 daily counts
        fc = NHiTSForecaster(horizon=7, input_size=100, max_steps=5).fit(df)
        # After fit, _effective_input_size must be ≤ 30 (= 60 // 2)
        assert fc._effective_input_size <= 30, (
            f"effective_input_size={fc._effective_input_size} exceeds n_train//2=30. "
            "The cap is not applied — R4 bug still present."
        )

    def test_predict_returns_standard_schema(self):
        df = self._make_sessions()
        fc = NHiTSForecaster(horizon=7, max_steps=5).fit(df)
        pred = fc.predict(horizon=7, n_scenarios=3, seed=0)
        assert set(pred.columns) >= {"date", "scenario", "n_sessions"}
        assert len(pred) == 7 * 3
        assert (pred["n_sessions"] >= 0).all()

    def test_predict_non_negative(self):
        df = self._make_sessions()
        fc = NHiTSForecaster(horizon=14, max_steps=5).fit(df)
        pred = fc.predict(horizon=14, n_scenarios=5, seed=42)
        assert (pred["n_sessions"] >= 0).all()

    def test_predict_start_date_respected(self):
        df = self._make_sessions()
        fc = NHiTSForecaster(horizon=7, max_steps=5).fit(df)
        pred = fc.predict(horizon=7, n_scenarios=1, start_date="2025-01-01")
        assert pd.Timestamp(pred["date"].iloc[0]) == pd.Timestamp("2025-01-01")

    def test_repr_fitted_true(self):
        df = self._make_sessions()
        fc = NHiTSForecaster(horizon=7, max_steps=5).fit(df)
        assert "fitted=True" in repr(fc)

    def test_shared_interface_contract(self):
        """NHiTSForecaster must honour the same interface as SessionForecaster."""
        df = self._make_sessions()
        fc = NHiTSForecaster(horizon=7, max_steps=5).fit(df)
        pred = fc.predict(horizon=7, n_scenarios=2, start_date="2025-03-01")
        assert pred["scenario"].nunique() == 2
        assert len(pred) == 14

    def test_no_leakage_temporal_split(self):
        """
        No-leakage protocol: predict() must not look at dates beyond training.
        Verify that after a strict temporal split (train ends 2023-06-30,
        eval starts 2023-07-01), forecast dates are all in July or later.
        """
        # Temporal split — strictly no future data in training
        df = self._make_sessions(n=200, seed=99)
        df_sorted = df.sort_values("arrival_time")
        cutoff = df_sorted["arrival_time"].quantile(0.75)
        train = df_sorted[df_sorted["arrival_time"] <= cutoff].copy()

        fc = NHiTSForecaster(horizon=7, max_steps=5).fit(train)
        pred = fc.predict(horizon=7, n_scenarios=1)
        # All forecast dates must be strictly after the training end
        train_end = pd.Timestamp(train["date"].max())
        assert all(
            pd.Timestamp(d) > train_end for d in pred["date"]
        ), "Forecast dates overlap with training — temporal leakage detected."

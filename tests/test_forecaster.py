"""Tests for gears.models.forecaster — all three forecasters.

Includes former tests/test_regression.py content for R1 (PersistenceForecaster
zero-day-drop), R2 (sessions_to_daily_counts 1970-epoch date bug), and R4
(NHiTSForecaster input_size/scaler_type defaults), relocated here per
AUDIT.md §g so each regression guard lives next to the other tests for its
subject class instead of in a fully parallel file. The bug-guard intent and
"R<n>" labels are preserved in each test's docstring."""
import numpy as np
import pandas as pd
import pytest

from gears.data.schemas import validate_dataframe
from gears.models.forecaster import (
    NHiTSForecaster,
    PersistenceForecaster,
    SessionForecaster,
    TransformerForecaster,
    sessions_to_daily_counts,
)


def make_sessions(n=300, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "arrival_time": pd.date_range("2024-01-01", periods=n, freq="6h"),
        "duration": rng.uniform(0.5, 8, n),
        "energy": rng.uniform(2, 40, n),
    })
    return validate_dataframe(df)


# ── sessions_to_daily_counts ──────────────────────────────────────────────────

def test_daily_counts_shape():
    df = make_sessions(200)
    counts = sessions_to_daily_counts(df)
    assert isinstance(counts, pd.Series)
    assert (counts >= 0).all()
    assert counts.index.is_monotonic_increasing


def test_daily_counts_fills_missing():
    df = make_sessions(50)
    counts = sessions_to_daily_counts(df)
    expected_days = (counts.index[-1] - counts.index[0]).days + 1
    assert len(counts) == expected_days


def _make_gap_sessions(
    before_end="2024-03-31", after_start="2024-04-15", after_end="2024-06-30",
    n_per_day=3, seed=42,
):
    """Sessions covering two date ranges separated by a gap (used by R1/R2
    below). The gap days appear in sessions_to_daily_counts with count=0."""
    rng = np.random.default_rng(seed)
    before = pd.date_range("2024-01-01", before_end, freq="D")
    after = pd.date_range(after_start, after_end, freq="D")
    rows = []
    for d in list(before) + list(after):
        for _ in range(n_per_day):
            rows.append({
                "arrival_time": pd.Timestamp(d) + pd.Timedelta(hours=float(rng.uniform(7, 21))),
                "duration": float(rng.uniform(0.5, 8.0)),
                "energy": float(rng.uniform(2.0, 40.0)),
            })
    return validate_dataframe(pd.DataFrame(rows))


def test_daily_counts_no_epoch_dates_from_date_objects():
    """R2: validate_dataframe stores 'date' as Python date objects
    (dtype=object). The old sessions_to_daily_counts didn't call
    pd.to_datetime(daily.index) before reindex(full_range), so the
    DatetimeIndex full_range failed to match the object-typed index and
    silently anchored everything at the Unix epoch (1970/1971)."""
    df = make_sessions(200)
    assert df["date"].dtype == object
    counts = sessions_to_daily_counts(df)
    assert counts.index.year.min() >= 2020, (
        f"Got year={counts.index.year.min()} — epoch-anchoring bug (R2) is back."
    )


def test_daily_counts_index_is_datetimeindex():
    """R2: return value must be a properly typed DatetimeIndex."""
    counts = sessions_to_daily_counts(make_sessions(100))
    assert isinstance(counts.index, pd.DatetimeIndex)


def test_daily_counts_values_non_negative():
    """R2: fill_value=0 must produce non-negative counts, never NaN."""
    counts = sessions_to_daily_counts(make_sessions(100))
    assert not counts.isna().any()
    assert (counts >= 0).all()


def test_daily_counts_year_preserved_across_validate_then_count():
    """R2 end-to-end: raw DataFrame -> validate_dataframe ->
    sessions_to_daily_counts must preserve the original year throughout."""
    rng = np.random.default_rng(7)
    raw = pd.DataFrame({
        "arrival_time": pd.date_range("2023-11-01", periods=60, freq="12h"),
        "duration": rng.uniform(0.5, 8, 60),
        "energy": rng.uniform(2, 40, 60),
    })
    counts = sessions_to_daily_counts(validate_dataframe(raw))
    assert counts.index.year.min() == 2023


# ── SessionForecaster ─────────────────────────────────────────────────────────

def test_sarima_fit_predict():
    df = make_sessions(300)
    fc = SessionForecaster(method="sarima")
    fc.fit(df)
    pred = fc.predict(horizon=7, seed=0)
    assert set(pred.columns) >= {"date", "scenario", "n_sessions"}
    assert len(pred) == 7
    assert (pred["n_sessions"] >= 0).all()


def test_probabilistic_fit_predict():
    df = make_sessions(200)
    fc = SessionForecaster(method="probabilistic").fit(df)
    pred = fc.predict(horizon=7, n_scenarios=5, seed=0)
    assert len(pred) == 35
    assert (pred["n_sessions"] >= 0).all()


def test_sarima_scenarios():
    df = make_sessions(300)
    fc = SessionForecaster(method="sarima").fit(df)
    pred = fc.predict(horizon=5, n_scenarios=10)
    assert pred["scenario"].nunique() == 10


def test_sarima_start_date():
    df = make_sessions(300)
    fc = SessionForecaster(method="sarima").fit(df)
    pred = fc.predict(horizon=7, start_date="2025-01-01")
    assert pd.Timestamp(pred["date"].iloc[0]) == pd.Timestamp("2025-01-01")


def test_invalid_method():
    with pytest.raises(ValueError, match="method"):
        SessionForecaster(method="xgboost")


def test_predict_before_fit():
    fc = SessionForecaster()
    with pytest.raises(RuntimeError):
        fc.predict(horizon=3)


# ── PersistenceForecaster ─────────────────────────────────────────────────────

def test_persistence_fit_predict():
    df = make_sessions(200)
    fc = PersistenceForecaster(n_weeks=1).fit(df)
    pred = fc.predict(horizon=14, n_scenarios=3, seed=0)
    assert set(pred.columns) >= {"date", "scenario", "n_sessions"}
    assert len(pred) == 14 * 3
    assert (pred["n_sessions"] >= 0).all()


def test_persistence_single_scenario():
    df = make_sessions(200)
    fc = PersistenceForecaster().fit(df)
    pred = fc.predict(horizon=7, n_scenarios=1)
    assert len(pred) == 7


def test_persistence_before_fit():
    fc = PersistenceForecaster()
    with pytest.raises(RuntimeError):
        fc.predict(horizon=3)


def test_persistence_uses_historical_values():
    """Persistence should use values from 7 days ago."""
    df = make_sessions(100)
    fc = PersistenceForecaster(n_weeks=1).fit(df)
    counts = sessions_to_daily_counts(df)
    start = counts.index[-1] + pd.Timedelta(days=1)
    pred = fc.predict(horizon=1, n_scenarios=1, start_date=start)
    ref_date = start - pd.Timedelta(weeks=1)
    if ref_date in counts.index:
        expected = counts[ref_date]
        assert abs(pred["n_sessions"].iloc[0] - expected) < 5


def test_persistence_zero_days_present_in_daily_counts():
    """R1: after fit(), all gap days must exist in _daily_counts with value
    0. The old bug filtered daily counts with `daily = daily[daily > 0]`,
    silently dropping zero-session days and leaving index holes that the
    52-week look-back could not always fill."""
    fc = PersistenceForecaster(n_weeks=1).fit(_make_gap_sessions())
    for gd in pd.date_range("2024-04-01", "2024-04-14", freq="D"):
        assert gd in fc._daily_counts.index, f"Zero-count day {gd.date()} was dropped (R1)."
        assert fc._daily_counts[gd] == 0.0

def test_persistence_predict_non_negative_after_gap():
    """R1: predict() must return non-negative counts even when reference
    days were 0."""
    fc = PersistenceForecaster(n_weeks=1).fit(_make_gap_sessions())
    pred = fc.predict(horizon=14, n_scenarios=1, seed=0)
    assert len(pred) == 14
    assert (pred["n_sessions"] >= 0).all()


def test_persistence_daily_counts_is_contiguous():
    """R1: _daily_counts must have no missing dates between first and last
    observation (a gap here means the zero-drop bug returned)."""
    idx = PersistenceForecaster(n_weeks=1).fit(_make_gap_sessions())._daily_counts.index
    assert len(idx) == (idx[-1] - idx[0]).days + 1


# ── TransformerForecaster ─────────────────────────────────────────────────────

def test_transformer_is_available():
    """Just confirm the availability check works."""
    result = TransformerForecaster.is_available()
    assert isinstance(result, bool)


def test_transformer_import_error_message():
    """Check that ImportError message is informative."""
    fc = TransformerForecaster(horizon=7)
    assert "neuralforecast" in fc._IMPORT_MSG.lower()
    assert "torch" in fc._IMPORT_MSG.lower()


@pytest.mark.skipif(
    not TransformerForecaster.is_available(),
    reason="neuralforecast not installed",
)
def test_transformer_fit_predict():
    df = make_sessions(300)
    fc = TransformerForecaster(horizon=7, max_steps=5).fit(df)
    assert fc.is_fitted_
    pred = fc.predict(horizon=7, n_scenarios=3, seed=0)
    assert set(pred.columns) >= {"date", "scenario", "n_sessions"}
    assert len(pred) == 21
    assert (pred["n_sessions"] >= 0).all()


@pytest.mark.skipif(
    not TransformerForecaster.is_available(),
    reason="neuralforecast not installed",
)
def test_transformer_repr():
    df = make_sessions(200)
    fc = TransformerForecaster(horizon=7, max_steps=5).fit(df)
    r = repr(fc)
    assert "PatchTST" in r or "Transformer" in r
    assert "fitted=True" in r


# ── NHiTSForecaster (R4: input_size default and scaler_type) ──────────────────
#
# R4 bug: the old default `input_size = 4 * horizon` consumed >=80% of a
# typical 450-day training set as a single context window, leaving fewer
# than one full epoch of gradient signal -> random-walk collapse.
# `scaler_type` was absent, so raw session-count gradients (scale ~100) were
# uncontrolled. Fixed default: input_size = 2 * horizon, scaler_type="standard".

def test_nhits_is_available_returns_bool():
    assert isinstance(NHiTSForecaster.is_available(), bool)


def test_nhits_import_msg_mentions_neuralforecast_and_torch():
    msg = NHiTSForecaster(horizon=7)._IMPORT_MSG.lower()
    assert "neuralforecast" in msg
    assert "torch" in msg


def test_nhits_default_input_size_is_2x_horizon():
    """R4 fix: default input_size = 2 * horizon (was 4 * horizon)."""
    fc = NHiTSForecaster(horizon=30)
    assert fc.input_size == 60, (
        f"Expected default input_size=2*horizon=60, got {fc.input_size} — R4 regressed."
    )


def test_nhits_explicit_input_size_preserved():
    assert NHiTSForecaster(horizon=14, input_size=20).input_size == 20


def test_nhits_scaler_type_default_is_standard():
    """R4 fix: scaler_type must default to 'standard' (was absent)."""
    assert NHiTSForecaster().scaler_type == "standard"


def test_nhits_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit"):
        NHiTSForecaster(horizon=7).predict(horizon=7)


def test_nhits_repr_contains_key_attrs():
    r = repr(NHiTSForecaster(horizon=14, max_steps=100))
    assert "14" in r
    assert "100" in r
    assert "fitted=False" in r


@pytest.mark.skipif(
    not NHiTSForecaster.is_available(),
    reason="neuralforecast and torch not installed ([dl] extra required)",
)
class TestNHiTSFitPredict:
    """R4 full fit/predict tests — only run when the [dl] extra is installed."""

    @staticmethod
    def _make_sessions(n=300, seed=0):
        return make_sessions(n, seed)

    def test_fit_sets_is_fitted(self):
        fc = NHiTSForecaster(horizon=7, max_steps=5).fit(self._make_sessions())
        assert fc.is_fitted_

    def test_input_size_capped_at_half_n_train(self):
        """R4 fix: effective input_size <= n_train // 2. Uses a short
        training set (60 days) with a large requested input_size (100) to
        confirm the cap is applied inside fit()."""
        fc = NHiTSForecaster(horizon=7, input_size=100, max_steps=5).fit(
            self._make_sessions(n=60))
        assert fc._effective_input_size <= 30, (
            f"effective_input_size={fc._effective_input_size} exceeds "
            "n_train//2=30 — R4 cap regressed."
        )

    def test_predict_returns_standard_schema(self):
        fc = NHiTSForecaster(horizon=7, max_steps=5).fit(self._make_sessions())
        pred = fc.predict(horizon=7, n_scenarios=3, seed=0)
        assert set(pred.columns) >= {"date", "scenario", "n_sessions"}
        assert len(pred) == 21
        assert (pred["n_sessions"] >= 0).all()

    def test_predict_start_date_respected(self):
        fc = NHiTSForecaster(horizon=7, max_steps=5).fit(self._make_sessions())
        pred = fc.predict(horizon=7, n_scenarios=1, start_date="2025-01-01")
        assert pd.Timestamp(pred["date"].iloc[0]) == pd.Timestamp("2025-01-01")

    def test_repr_fitted_true(self):
        fc = NHiTSForecaster(horizon=7, max_steps=5).fit(self._make_sessions())
        assert "fitted=True" in repr(fc)

    def test_no_leakage_temporal_split(self):
        """No-leakage protocol: predict() must not look at dates beyond
        training. After a strict temporal split, forecast dates must all
        fall after the training cutoff."""
        df = self._make_sessions(n=200, seed=99).sort_values("arrival_time")
        cutoff = df["arrival_time"].quantile(0.75)
        train = df[df["arrival_time"] <= cutoff].copy()
        fc = NHiTSForecaster(horizon=7, max_steps=5).fit(train)
        pred = fc.predict(horizon=7, n_scenarios=1)
        train_end = pd.Timestamp(train["date"].max())
        assert all(pd.Timestamp(d) > train_end for d in pred["date"]), (
            "Forecast dates overlap with training — temporal leakage."
        )


# ── Shared interface contract ─────────────────────────────────────────────────

@pytest.mark.parametrize("fc_cls,kwargs", [
    (lambda: SessionForecaster(method="probabilistic"), {}),
    (lambda: PersistenceForecaster(n_weeks=1), {}),
])
def test_shared_interface(fc_cls, kwargs):
    """All forecasters must honour the same fit/predict interface."""
    df = make_sessions(200)
    fc = fc_cls()
    fc.fit(df)
    pred = fc.predict(horizon=7, n_scenarios=2, start_date="2025-01-01")
    assert set(pred.columns) >= {"date", "scenario", "n_sessions"}
    assert pred["scenario"].nunique() == 2
    assert len(pred) == 14

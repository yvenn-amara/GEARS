"""Tests for gears.models.forecaster — all three forecasters."""
import numpy as np
import pandas as pd
import pytest
from gears.data.schemas import validate_dataframe
from gears.models.forecaster import (
    SessionForecaster, PersistenceForecaster, TransformerForecaster,
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

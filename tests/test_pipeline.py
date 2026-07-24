"""Integration tests for GEARSModel pipeline."""
import pandas as pd
import pytest
from gears import GEARSModel
from gears.data.loader import make_demo_data


def make_df(n=800, seed=0):
    return make_demo_data(n=n, location_type="work", seed=seed)


def test_fit_and_simulate_short():
    df = make_df(800)
    model = GEARSModel(n_components=3, n_scenarios=3)
    model.fit(df, verbose=False)
    assert model.is_fitted_

    sessions = model.simulate_short_term("2025-06-01", horizon=3, n_scenarios=3, seed=0)
    assert len(sessions) > 0
    assert set(["date", "scenario", "arrival_hour", "duration", "energy"]).issubset(sessions.columns)


def test_fit_and_simulate_medium():
    df = make_df(500)
    model = GEARSModel(n_components=3, n_scenarios=3)
    model.fit(df, verbose=False)
    result = model.simulate_medium_term(years=0.1, output="daily_energy", n_scenarios=3)
    assert len(result) > 0
    assert "total_energy_kwh" in result.columns


def test_daily_energy_aggregator():
    df = make_df(500)
    model = GEARSModel(n_components=3, n_scenarios=2).fit(df, verbose=False)
    sessions = model.simulate_short_term("2025-06-01", horizon=3, n_scenarios=2)
    daily = model.daily_energy(sessions)
    assert "total_energy_kwh" in daily.columns
    assert "n_sessions" in daily.columns


def test_from_native_gmm():
    model = GEARSModel.from_native_gmm("french")
    assert model.is_fitted_
    sessions = model.simulate_short_term("2025-06-01", horizon=2, n_scenarios=2)
    assert len(sessions) > 0


def test_simulate_without_fit():
    model = GEARSModel()
    with pytest.raises(RuntimeError):
        model.simulate_short_term("2025-06-01")


def test_summary():
    model = GEARSModel(n_components=3)
    s = model.summary()
    assert "NOT FITTED" in s
    df = make_df(300)
    model.fit(df, verbose=False)
    s = model.summary()
    assert "fitted" in s


def test_smart_charge():
    from gears.utils import make_price_signal
    df = make_df(300)
    model = GEARSModel(n_components=3, n_scenarios=2).fit(df, verbose=False)
    sessions = model.simulate_short_term("2025-06-01", horizon=2, n_scenarios=1, seed=0)
    if "arrival_time" not in sessions.columns and "arrival_hour" in sessions.columns:
        sessions["arrival_time"] = (
            pd.to_datetime(sessions["date"])
            + pd.to_timedelta(sessions["arrival_hour"], unit="h")
        )
    signal = make_price_signal(start="2025-06-01", periods=96, resolution_min=30)
    result = model.smart_charge(sessions, signal)
    assert "cost_smart" in result.columns

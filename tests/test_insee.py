"""Tests for gears.data.insee."""
import numpy as np
import pandas as pd
import pytest
from gears.data.loader import make_demo_data
from gears.data.insee import (
    aggregate_by_department, build_panel, DepartmentForecaster
)


def make_dept_df(n=500, seed=0):
    rng = np.random.default_rng(seed)
    df = make_demo_data(n=n, seed=seed)
    df["department"] = rng.choice(["75", "69", "33"], n)
    return df


def test_aggregate_by_department():
    df = make_dept_df()
    agg = aggregate_by_department(df, freq="D", metric="energy_kwh")
    assert set(["date", "department", "energy_kwh"]).issubset(agg.columns)
    assert (agg["energy_kwh"] >= 0).all()


def test_build_panel():
    df = make_dept_df()
    panel = build_panel(df, freq="D", metric="energy_kwh")
    assert isinstance(panel, pd.DataFrame)
    assert panel.index.is_monotonic_increasing
    assert set(["75", "69", "33"]).issubset(panel.columns)


def test_missing_department_raises():
    df = make_demo_data(n=100)  # no department column
    with pytest.raises(ValueError, match="department"):
        aggregate_by_department(df)


def test_department_forecaster_fit_predict():
    df = make_dept_df(n=600)
    fc = DepartmentForecaster(min_obs=30)
    fc.fit(df, verbose=False)
    assert fc.is_fitted_
    pred = fc.predict(horizon=14, departments=["75", "69"])
    assert set(["date", "department", "scenario", "energy_kwh_forecast"]).issubset(pred.columns)
    assert (pred["energy_kwh_forecast"] >= 0).all()


def test_forecaster_predict_before_fit():
    fc = DepartmentForecaster()
    with pytest.raises(RuntimeError):
        fc.predict(horizon=7)

"""Tests for gears.data.insee."""
import numpy as np
import pandas as pd
import pytest

from gears.data.insee import DepartmentForecaster, aggregate_by_department, build_panel
from gears.data.loader import make_demo_data


def make_dept_df(n=500, seed=0):
    rng = np.random.default_rng(seed)
    df = make_demo_data(n=n, seed=seed)
    df["department"] = rng.choice(["75", "69", "33"], n)
    return df


def test_aggregate_by_department():
    df = make_dept_df()
    agg = aggregate_by_department(df, freq="D", metric="energy_kwh")
    assert {"date", "department", "energy_kwh"}.issubset(agg.columns)
    assert (agg["energy_kwh"] >= 0).all()


def test_build_panel():
    df = make_dept_df()
    panel = build_panel(df, freq="D", metric="energy_kwh")
    assert isinstance(panel, pd.DataFrame)
    assert panel.index.is_monotonic_increasing
    assert {"75", "69", "33"}.issubset(panel.columns)


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
    assert {"date", "department", "scenario", "energy_kwh_forecast"}.issubset(pred.columns)
    assert (pred["energy_kwh_forecast"] >= 0).all()


def test_forecaster_predict_before_fit():
    fc = DepartmentForecaster()
    with pytest.raises(RuntimeError):
        fc.predict(horizon=7)


def test_forecast_ci_width_not_artificially_pinched():
    """Session 6 regression test (AUDIT.md §e, Mechanism 1): `_forecast_dept`
    used to scale its Monte-Carlo noise by `std * 0.05`, producing an 80% CI
    band only ~2% of the median wide — visually a flat line rather than a
    genuine forecast cone, which is what made notebook 3's medium-term fan
    charts look artificially capped. After porting the `SessionForecaster`
    fix (full std, matching `gears/models/forecaster.py`), the band should be
    a clearly visible double-digit percentage of the median, not a hairline."""
    df = make_dept_df(n=600, seed=1)
    fc = DepartmentForecaster(min_obs=30)
    fc.fit(df, departments=["75"], verbose=False)
    pred = fc.predict(horizon=30, departments=["75"], n_scenarios=200, seed=0)

    pivot = pred.pivot_table(index="date", columns="scenario", values="energy_kwh_forecast")
    median = pivot.median(axis=1)
    width = pivot.quantile(0.9, axis=1) - pivot.quantile(0.1, axis=1)
    rel_width = (width / median.clip(lower=1e-6)).mean()

    # Old behaviour measured ~2% (0.02); require it to be at least an order
    # of magnitude wider now, without hard-coding an exact expected value.
    assert rel_width > 0.15, (
        f"80% CI band is only {rel_width:.1%} of the median on average — "
        "still looks like the pre-Session-6 pinched-band bug."
    )

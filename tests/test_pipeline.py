"""Tests for the GEARSModel facade: the whole chain, construction helpers and dispatch.

Saving then loading a model reproduces the simulation (invariant 8) and every entry point
refuses an unfitted model: both are checked in ``test_sanity.py``.
"""

from __future__ import annotations

import joblib
import pandas as pd
import pytest

from gears import GEARSModel
from gears.data.loader import make_demo_data
from gears.models.session_model import EVSessionModel
from gears.utils import make_price_signal


def test_a_fitted_model_simulates_aggregates_optimises_and_exports(model, tmp_path):
    sessions = model.simulate_short_term("2025-06-01", horizon=3, n_scenarios=2, seed=0)
    assert {"date", "scenario", "arrival_time", "duration", "energy"} <= set(sessions.columns)

    daily = model.daily_energy(sessions)
    assert daily["n_sessions"].sum() == len(sessions)
    assert daily["total_energy_kwh"].sum() == pytest.approx(sessions["energy"].sum())
    assert {"hour", "energy_kwh"} <= set(model.hourly_profile(sessions).columns)

    signal = make_price_signal(start="2025-06-01", periods=5 * 48, resolution_min=30)
    assert {"cost_smart", "cost_plug"} <= set(model.smart_charge(sessions, signal).columns)

    yearly = model.simulate_medium_term(years=0.1, output="daily_energy", n_scenarios=2)
    assert len(yearly) > 0 and "total_energy_kwh" in yearly.columns

    model.export(sessions, tmp_path / "sessions.csv")
    assert len(pd.read_csv(tmp_path / "sessions.csv")) == len(sessions)


def test_asking_for_several_scenarios_with_a_fixed_session_count_says_that_only_one_is_made(model):
    with pytest.warns(UserWarning, match=r"n_scenarios=3 is ignored when n_sessions is given"):
        out = model.simulate_short_term("2025-06-02", horizon=2, n_scenarios=3, n_sessions=5, seed=0)
    assert len(out) == 10 and "scenario" not in out.columns

    import warnings

    with warnings.catch_warnings():                      # one scenario asked, or none asked: no warning
        warnings.simplefilter("error")
        model.simulate_short_term("2025-06-02", horizon=2, n_scenarios=1, n_sessions=5, seed=0)
        model.simulate_short_term("2025-06-02", horizon=2, n_sessions=5, seed=0)


def test_the_summary_tells_whether_the_model_is_fitted(model):
    assert "NOT FITTED" in GEARSModel(n_components=3).summary()
    summary = model.summary()
    assert "fitted" in summary and "NOT FITTED" not in summary


def test_a_model_can_be_built_from_a_fitted_session_model():
    gmm = EVSessionModel(n_components=3).fit(make_demo_data(n=500, location_type="work", seed=0))

    model = GEARSModel.from_session_model(gmm, n_scenarios=2, forecaster_method="probabilistic")

    assert model.is_fitted_ and model.gmm_ is gmm
    assert len(model.simulate_short_term("2025-06-01", horizon=2, n_scenarios=2)) > 0


def test_gear_1_is_the_default_gears_2_to_5_are_reserved_and_other_values_are_refused():
    data = make_demo_data(n=400, location_type="work", seed=0)
    kwargs = {"n_components": 3, "n_scenarios": 2, "random_state": 0,
              "forecaster_method": "probabilistic"}
    implicit, explicit = GEARSModel(**kwargs), GEARSModel(gear=1, **kwargs)
    assert implicit.gear == explicit.gear == 1

    implicit.fit(data, verbose=False)
    explicit.fit(data, verbose=False)
    pd.testing.assert_frame_equal(
        implicit.simulate_short_term("2025-06-01", horizon=2, n_scenarios=1, seed=0),
        explicit.simulate_short_term("2025-06-01", horizon=2, n_scenarios=1, seed=0),
    )

    for gear in (2, 3, 4, 5):            # reserved for future model families
        with pytest.raises(NotImplementedError, match=rf"GEAR {gear} is reserved for a future release") as err:
            GEARSModel(gear=gear)
        assert "[1]" in str(err.value) and len(str(err.value)) < 120
    for gear in (0, 6, True, 1.5, "1", None):   # not a gear at all
        with pytest.raises(ValueError, match="gear must be"):
            GEARSModel(gear=gear)


def test_load_refuses_a_file_that_is_not_a_gears_model(tmp_path):
    joblib.dump({"not": "a model"}, tmp_path / "other.joblib")
    with pytest.raises(TypeError, match="Expected GEARSModel"):
        GEARSModel.load(tmp_path / "other.joblib")

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
    assert {"date", "scenario", "arrival_hour", "duration", "energy"}.issubset(sessions.columns)


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


# ---------------------------------------------------------------------------
# GEAR dispatch (Phase 2 / Session 3) — see PROPOSAL_GEAR_ARCHITECTURE.md
# ---------------------------------------------------------------------------

def test_gear_defaults_to_one_and_is_backward_compatible():
    """Omitting `gear` (or passing gear=1 explicitly) must be behaviorally
    identical to the pre-Session-3 GEARSModel — same attributes, same
    fit/simulate behavior."""
    df = make_df(400)
    implicit = GEARSModel(n_components=3, n_scenarios=2, random_state=0)
    explicit = GEARSModel(gear=1, n_components=3, n_scenarios=2, random_state=0)

    assert implicit.gear == 1
    assert explicit.gear == 1

    implicit.fit(df, verbose=False)
    explicit.fit(df, verbose=False)

    assert implicit.is_fitted_ and explicit.is_fitted_
    # Same seed, same data, same gear -> identical GMM repr.
    assert repr(implicit.gmm_) == repr(explicit.gmm_)


@pytest.mark.parametrize("gear", [2, 3, 4, 5])
def test_unimplemented_gears_raise_not_implemented_error(gear):
    with pytest.raises(NotImplementedError, match="GEAR 1st"):
        GEARSModel(gear=gear)


def test_gear1_supports_vae_model_type_end_to_end():
    """model_type is a first-class GEARSModel constructor param (previously
    only reachable via EVSessionModel directly) and must work end-to-end,
    including through simulate_short_term."""
    df = make_df(400)
    model = GEARSModel(
        gear=1, model_type="vae", n_scenarios=2,
        forecaster_method="probabilistic", random_state=0,
    )
    assert model.model_type == "vae"
    model.fit(df, verbose=False)
    assert model.is_fitted_
    assert model.gmm_.model_type == "vae"

    sessions = model.simulate_short_term("2025-06-01", horizon=2, n_scenarios=1, seed=0)
    assert len(sessions) > 0


def test_recency_and_half_life_days_are_constructor_params():
    """recency/half_life_days must be settable on GEARSModel directly,
    without reaching into EVSessionModel."""
    model = GEARSModel(gear=1, recency=True, half_life_days=45.0, n_components=2)
    assert model.recency is True
    assert model.half_life_days == 45.0

    df = make_df(400)
    model.fit(df, verbose=False)
    assert model.gmm_.recency is True

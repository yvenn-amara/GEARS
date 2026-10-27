"""Tests for the medium-term simulator: growth profiles and the three outputs.

That every profile starts at the base, never goes negative and grows monotonically (and
the validation of the arguments) is checked in ``test_sanity.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from gears.simulation.medium_term import (
    GROWTH_PROFILES,
    MediumTermSimulator,
    linear_growth_profile,
    s_curve_growth_profile,
)


@pytest.fixture(scope="module")
def gmm(model):
    return model.gmm_


# ── Growth profiles ───────────────────────────────────────────────────────────

def test_the_linear_profile_adds_a_constant_share_of_the_base_each_year():
    """It used to compound (base x (1 + rate) ** t) despite its name."""
    base, rate, years = 1000.0, 0.15, 15
    profile = linear_growth_profile(base, years, annual_growth_rate=rate)

    t = np.arange(len(profile)) / 365.25
    np.testing.assert_allclose(profile.to_numpy(), base * (1 + rate * t), rtol=1e-9)
    assert profile.iloc[-1] < 0.5 * base * (1 + rate) ** years     # clearly not exponential


def test_the_s_curve_spreads_its_growth_over_the_requested_horizon():
    """With fixed defaults the curve was saturated by year 8 whatever the horizon, so a
    20-year simulation was flat for its last 12 years."""
    base, saturation = 1000.0, 3.0
    share_at_year_8 = {}
    for years in (8, 15, 20):
        s = s_curve_growth_profile(base, years, saturation_factor=saturation)
        assert s.max() <= base * saturation * 1.01                       # never above the cap
        assert s.iloc[-1] / (base * saturation) > 0.9                    # ~95 % reached by the end
        share_at_year_8[years] = s.iloc[min(int(8 * 365.25), len(s) - 1)] / (base * saturation)

    assert share_at_year_8[20] < share_at_year_8[15] < share_at_year_8[8]


def test_a_profile_has_one_value_per_day_and_all_profiles_are_registered():
    assert len(linear_growth_profile(50, years=2)) == round(365.25 * 2)
    assert len(s_curve_growth_profile(50, years=3)) == round(365.25 * 3)
    assert set(GROWTH_PROFILES) == {"linear", "s_curve", "bass"}


# ── The simulator ─────────────────────────────────────────────────────────────

def test_the_three_outputs_have_their_documented_columns(gmm):
    sim = MediumTermSimulator(gmm, base_sessions_per_day=30, n_scenarios=3, seed=0)

    daily = sim.simulate(years=0.1, output="daily_energy", start_date="2025-03-01")
    assert {"date", "scenario", "n_sessions", "total_energy_kwh"} <= set(daily.columns)
    assert daily["scenario"].nunique() == 3 and (daily["total_energy_kwh"] >= 0).all()

    hourly = sim.simulate(years=0.05, output="hourly_energy", start_date="2025-03-01")
    assert {"hour", "energy_kwh"} <= set(hourly.columns)
    assert hourly["hour"].between(0, 23).all()

    sessions = sim.simulate(years=0.05, output="sessions", start_date="2025-03-01")
    assert {"arrival_hour", "duration", "energy", "power_kw"} <= set(sessions.columns)


def test_every_growth_model_drives_a_simulation_and_an_unknown_one_is_refused(gmm):
    for growth_model in GROWTH_PROFILES:
        sim = MediumTermSimulator(
            gmm, base_sessions_per_day=20, growth_model=growth_model, n_scenarios=2, seed=0
        )
        result = sim.simulate(years=0.1, output="daily_energy", start_date="2025-03-01")
        assert len(result) > 0 and (result["total_energy_kwh"] >= 0).all(), growth_model

    with pytest.raises(ValueError, match="growth_model"):
        MediumTermSimulator(gmm, growth_model="magic", n_scenarios=1).simulate(years=0.05)


def test_the_weather_factor_scales_the_number_of_sessions_of_its_season(gmm):
    sim = MediumTermSimulator(gmm, base_sessions_per_day=50, n_scenarios=2, seed=0)
    kwargs = {"years": 0.1, "output": "daily_energy", "start_date": "2025-06-10"}  # all summer

    normal = sim.simulate(**kwargs)["n_sessions"].sum()
    heatwave = sim.simulate(**kwargs, weather_factor={"summer": 0.01})["n_sessions"].sum()

    assert heatwave < 0.05 * normal


def test_a_multi_year_simulation_has_no_horizon_cap(gmm):
    sim = MediumTermSimulator(gmm, base_sessions_per_day=3, n_scenarios=1, seed=0)
    result = sim.simulate(years=6, output="daily_energy", start_date="2025-01-01")
    # one row per day with at least one session (Poisson(3) leaves 5 % of days empty)
    assert 0.9 * round(365.25 * 6) < len(result) <= round(365.25 * 6)


# ── Growth parameters, one row per day, sorted output ────────────────────────

def test_a_growth_parameter_the_model_does_not_take_is_refused_and_the_valid_ones_listed(gmm):
    sim = MediumTermSimulator(gmm, base_sessions_per_day=20, growth_model="s_curve", seed=0)
    with pytest.raises(ValueError, match=r"Unknown growth parameter\(s\) \['saturation_facter'\].*saturation_factor"):
        sim.simulate(years=0.1, start_date="2025-01-01", saturation_facter=9)

    linear = MediumTermSimulator(gmm, base_sessions_per_day=20, growth_model="linear", seed=0)
    with pytest.raises(ValueError, match=r"\['saturation_factor'\].*annual_growth_rate"):
        linear.simulate(years=0.1, start_date="2025-01-01", saturation_factor=3.0)


def test_the_growth_rate_of_the_linear_model_is_not_silently_dropped_by_the_others(gmm, recwarn):
    kwargs = dict(years=0.2, start_date="2025-01-01", n_scenarios=1)
    s_curve = MediumTermSimulator(gmm, base_sessions_per_day=20, growth_model="s_curve", seed=0)
    with pytest.warns(UserWarning, match=r"annual_growth_rate=0.5 changes nothing.*saturation_factor"):
        s_curve.simulate(annual_growth_rate=0.5, **kwargs)

    recwarn.clear()
    s_curve.simulate(**kwargs)                                  # not given: nothing to say
    MediumTermSimulator(gmm, base_sessions_per_day=20, growth_model="linear", seed=0).simulate(
        annual_growth_rate=0.5, **kwargs)
    assert not [w for w in recwarn if "changes nothing" in str(w.message)]


def test_the_shape_of_a_growth_model_can_be_chosen_through_the_facade(model):
    kwargs = dict(years=1, start_date="2025-01-01", n_scenarios=1, output="daily_energy")
    low = model.simulate_medium_term(growth_model="s_curve", saturation_factor=1.2, **kwargs)
    high = model.simulate_medium_term(growth_model="s_curve", saturation_factor=6.0, **kwargs)
    assert high["n_sessions"].sum() > 1.2 * low["n_sessions"].sum()

    slow = model.simulate_medium_term(growth_model="bass", p=0.001, q=0.1, **kwargs)
    fast = model.simulate_medium_term(growth_model="bass", p=0.05, q=0.9, **kwargs)
    assert fast["n_sessions"].sum() != slow["n_sessions"].sum()

    with pytest.raises(ValueError, match="Unknown growth parameter"):
        model.simulate_medium_term(growth_model="bass", saturation_factor=3.0, **kwargs)


def test_a_day_without_sessions_is_a_row_with_zero_and_the_rows_are_in_date_order(gmm):
    sim = MediumTermSimulator(gmm, base_sessions_per_day=0.3, n_scenarios=2, seed=1)
    daily = sim.simulate(years=0.1, start_date="2025-01-01")
    n_days = len(daily["date"].unique())
    assert len(daily) == 2 * n_days and n_days == round(365.25 * 0.1)      # one row per day and scenario
    assert (daily["n_sessions"] == 0).any()
    assert (daily.loc[daily["n_sessions"] == 0, "total_energy_kwh"] == 0).all()
    assert daily.equals(daily.sort_values(["scenario", "date"], kind="stable").reset_index(drop=True))

    hourly = MediumTermSimulator(gmm, base_sessions_per_day=30, n_scenarios=2, seed=1).simulate(
        years=0.05, start_date="2025-01-01", output="hourly_energy")
    assert hourly.equals(hourly.sort_values(["scenario", "date", "hour"], kind="stable").reset_index(drop=True))

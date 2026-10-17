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

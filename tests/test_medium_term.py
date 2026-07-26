"""Tests for medium_term simulator — growth profiles and vectorisation."""
import numpy as np
import pandas as pd
import pytest

from gears.data.loader import make_demo_data
from gears.models.gmm import EVSessionGMM
from gears.simulation.medium_term import (
    GROWTH_PROFILES,
    MediumTermSimulator,
    bass_diffusion_profile,
    double_s_curve_profile,
    linear_growth_profile,
    s_curve_growth_profile,
    s_curve_linear_tail_profile,
)


@pytest.fixture(scope="module")
def fitted_gmm():
    df = make_demo_data(n=500, seed=0)
    return EVSessionGMM(n_components=3).fit(df)


# ── Growth profiles ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("fn,extra", [
    (linear_growth_profile,            {"annual_growth_rate": 0.1}),
    (s_curve_growth_profile,           {"saturation_factor": 3.0}),
    (s_curve_linear_tail_profile,      {"saturation_factor": 3.0, "tail_rate": 0.03}),
    (bass_diffusion_profile,           {"market_potential_factor": 4.0, "p": 0.03, "q": 0.38}),
    (double_s_curve_profile,           {"saturation_factor_1": 2.0, "saturation_factor_2": 1.5}),
])
def test_growth_profile_basic(fn, extra):
    series = fn(50, 2.0, start_date="2025-01-01", **extra)
    assert isinstance(series, pd.Series)
    assert len(series) > 0
    assert (series >= 0).all()
    assert series.index.is_monotonic_increasing


def test_all_profiles_registered():
    assert set(GROWTH_PROFILES.keys()) == {
        "linear", "s_curve", "s_curve_linear_tail", "bass", "double_s_curve"
    }


def test_s_curve_tail_exceeds_s_curve():
    """S-curve+tail should be >= S-curve after saturation."""
    s   = s_curve_growth_profile(50, 10, saturation_factor=3.0, start_date="2025-01-01")
    st  = s_curve_linear_tail_profile(50, 10, saturation_factor=3.0,
                                       tail_rate=0.05, start_date="2025-01-01")
    # After midpoint, tail version should be higher
    assert (st.values[-100:] >= s.values[-100:]).all()


def test_bass_starts_near_zero():
    """Bass model cumulative adoption starts near 0."""
    b = bass_diffusion_profile(100, 5, market_potential_factor=4.0,
                               p=0.03, q=0.38, start_date="2025-01-01")
    assert b.iloc[0] < 5   # cumulative starts small


def test_double_s_curve_above_single():
    """Double S-curve (2 waves) should exceed single S-curve at long horizons."""
    s  = s_curve_growth_profile(50, 15, saturation_factor=2.0, start_date="2025-01-01")
    ds = double_s_curve_profile(50, 15, saturation_factor_1=2.0, saturation_factor_2=2.0,
                                midpoint_year_1=3.0, midpoint_year_2=9.0, start_date="2025-01-01")
    # Double S at long horizon should exceed single S
    assert ds.values[-365:].mean() > s.values[-365:].mean()


class TestGrowthProfiles:
    """Property checks specific to individual profiles, not covered by the
    generic `test_growth_profile_basic` parametrization above (moved here
    from test_simulation.py — both files were testing linear_growth_profile
    and s_curve_growth_profile; see AUDIT.md §g)."""

    def test_linear_length(self):
        profile = linear_growth_profile(50, years=2)
        assert len(profile) == 365 * 2

    def test_linear_increasing(self):
        profile = linear_growth_profile(50, years=2, annual_growth_rate=0.1)
        assert profile.iloc[-1] > profile.iloc[0]

    def test_s_curve_length(self):
        profile = s_curve_growth_profile(50, years=3)
        assert len(profile) == round(365.25 * 3)

    def test_s_curve_bounded(self):
        profile = s_curve_growth_profile(50, years=3, saturation_factor=5)
        assert profile.max() <= 50 * 5 * 1.01  # slight tolerance

    def test_linear_zero_growth(self):
        profile = linear_growth_profile(50, years=1, annual_growth_rate=0.0)
        # all values should be equal to base
        assert np.allclose(profile.values, 50.0)


# ── MediumTermSimulator ───────────────────────────────────────────────────────

@pytest.mark.parametrize("growth_model", list(GROWTH_PROFILES.keys()))
def test_simulate_all_growth_models(fitted_gmm, growth_model):
    sim = MediumTermSimulator(fitted_gmm, base_sessions_per_day=20,
                               growth_model=growth_model, n_scenarios=2, seed=0)
    result = sim.simulate(years=0.25, output="daily_energy")
    assert len(result) > 0
    assert "total_energy_kwh" in result.columns
    assert (result["total_energy_kwh"] >= 0).all()


def test_simulate_daily_energy(fitted_gmm):
    sim = MediumTermSimulator(fitted_gmm, base_sessions_per_day=30, n_scenarios=3)
    result = sim.simulate(years=0.1, output="daily_energy")
    assert set(result.columns) >= {"date", "scenario", "n_sessions", "total_energy_kwh"}
    assert result["scenario"].nunique() == 3


def test_simulate_hourly_energy(fitted_gmm):
    sim = MediumTermSimulator(fitted_gmm, base_sessions_per_day=30, n_scenarios=2)
    result = sim.simulate(years=0.05, output="hourly_energy")
    assert "hour" in result.columns
    assert "energy_kwh" in result.columns


def test_simulate_sessions(fitted_gmm):
    sim = MediumTermSimulator(fitted_gmm, base_sessions_per_day=20, n_scenarios=2)
    result = sim.simulate(years=0.05, output="sessions")
    assert "arrival_hour" in result.columns
    assert "energy" in result.columns


def test_invalid_output(fitted_gmm):
    sim = MediumTermSimulator(fitted_gmm, n_scenarios=2)
    with pytest.raises(ValueError, match="output"):
        sim.simulate(years=0.05, output="invalid")


def test_invalid_growth_model(fitted_gmm):
    sim = MediumTermSimulator(fitted_gmm, growth_model="magic", n_scenarios=2)
    with pytest.raises(ValueError, match="growth_model"):
        sim.simulate(years=0.05)


def test_scenarios_count(fitted_gmm):
    sim = MediumTermSimulator(fitted_gmm, base_sessions_per_day=20, n_scenarios=5)
    result = sim.simulate(years=0.1, output="daily_energy")
    assert result["scenario"].nunique() == 5


def test_charger_mix_normalisation(fitted_gmm):
    """charger_mix should be auto-normalised even if weights don't sum to 1."""
    sim = MediumTermSimulator(fitted_gmm, base_sessions_per_day=20,
                               charger_mix={7.4: 3, 22.0: 1}, n_scenarios=2)
    total = sum(sim.charger_mix.values())
    assert abs(total - 1.0) < 1e-9


def test_vectorised_faster_than_naive(fitted_gmm):
    """Vectorised simulation should complete within a reasonable time."""
    import time
    sim = MediumTermSimulator(fitted_gmm, base_sessions_per_day=30, n_scenarios=5, seed=0)
    t0 = time.time()
    result = sim.simulate(years=1, output="daily_energy")
    elapsed = time.time() - t0
    assert len(result) > 0
    assert elapsed < 60, f"Simulation took {elapsed:.1f}s — too slow"

"""Tests for medium_term simulator — growth profiles and vectorisation."""
import numpy as np
import pandas as pd
import pytest

from gears.data.loader import make_demo_data
from gears.models.session_model import EVSessionModel
from gears.simulation.medium_term import (
    GROWTH_PROFILES,
    MediumTermSimulator,
    bass_diffusion_profile,
    linear_growth_profile,
    s_curve_growth_profile,
)


@pytest.fixture(scope="module")
def fitted_gmm():
    df = make_demo_data(n=500, seed=0)
    return EVSessionModel(n_components=3).fit(df)


# ── Growth profiles ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("fn,extra", [
    (linear_growth_profile,            {"annual_growth_rate": 0.1}),
    (s_curve_growth_profile,           {"saturation_factor": 3.0}),
    (bass_diffusion_profile,           {"market_potential_factor": 4.0, "p": 0.03, "q": 0.38}),
])
def test_growth_profile_basic(fn, extra):
    series = fn(50, 2.0, start_date="2025-01-01", **extra)
    assert isinstance(series, pd.Series)
    assert len(series) > 0
    assert (series >= 0).all()
    assert series.index.is_monotonic_increasing


def test_all_profiles_registered():
    """Session 6: reduced from 5 to 3 profiles (`s_curve_linear_tail` and
    `double_s_curve` cut — see gears/simulation/medium_term.py's module
    docstring "Session 6 fix note" and REFACTOR_STATE.md for the
    justification)."""
    assert set(GROWTH_PROFILES.keys()) == {"linear", "s_curve", "bass"}


@pytest.mark.parametrize("fn,extra", [
    (linear_growth_profile,  {"annual_growth_rate": 0.2}),
    (s_curve_growth_profile, {"saturation_factor": 3.0}),
    (bass_diffusion_profile, {"market_potential_factor": 4.0}),
])
def test_growth_profile_anchors_at_base(fn, extra):
    """Session 6 regression test: every growth profile must start exactly at
    `base_sessions_per_day` at t=0 — no discontinuity between "current
    observed fleet" and the simulated trajectory. Before the Session 6 fix,
    this held only for `linear_growth_profile`; `s_curve_growth_profile`
    started at ~2-7% of its asymptote depending on parameters, and
    `bass_diffusion_profile` started at exactly 0 regardless of `base`."""
    base = 1234.0
    series = fn(base, years=15, start_date="2025-01-01", **extra)
    assert series.iloc[0] == pytest.approx(base, rel=1e-9)


def test_s_curve_saturation_timing_scales_with_horizon():
    """Session 6 regression test for the actual traced root cause of the
    notebook 3 "plateau" (AUDIT.md §e / REFACTOR_STATE.md): with the old
    fixed defaults (midpoint_year=2.5, steepness=1.5), the curve was ~100%
    saturated by year 8 *regardless of the requested horizon* — so a
    20-year simulation was completely flat for its last 12 years. With the
    horizon-relative defaults, the % of asymptote reached at a fixed
    absolute year (8) should clearly DECREASE as the requested horizon
    grows, showing growth is now spread across the full horizon instead of
    being front-loaded into the first ~8 years unconditionally."""
    base, saturation_factor = 1000.0, 3.0
    pct_at_year8 = {}
    for years in (8, 15, 20):
        s = s_curve_growth_profile(base, years, saturation_factor=saturation_factor)
        idx8 = min(int(8 * 365.25), len(s) - 1)
        pct_at_year8[years] = s.iloc[idx8] / (base * saturation_factor)

    # Longer horizons must show meaningfully less saturation at year 8 than
    # shorter ones — i.e. saturation timing tracks the horizon, not a fixed year.
    assert pct_at_year8[20] < pct_at_year8[15] < pct_at_year8[8]
    # And whatever the horizon, growth should still be well underway (not
    # flat/near-zero) rather than saturating implausibly early or late.
    for years in (8, 15, 20):
        idx_end = len(s_curve_growth_profile(base, years, saturation_factor=saturation_factor)) - 1
        s_full = s_curve_growth_profile(base, years, saturation_factor=saturation_factor)
        assert s_full.iloc[idx_end] / (base * saturation_factor) > 0.9  # ~95% by design


def test_linear_growth_profile_is_actually_linear():
    """Session 6 regression test: `linear_growth_profile` previously computed
    compound/exponential growth (`base * (1 + rate) ** t`) despite its name —
    flagged in AUDIT.md §c. Must now match `base * (1 + rate * t)` exactly,
    and must clearly differ from the old exponential formula at a long
    horizon (they agree only at t=0 and t=1)."""
    base, rate, years = 1000.0, 0.15, 15
    profile = linear_growth_profile(base, years, annual_growth_rate=rate)
    t = np.arange(len(profile)) / 365.25
    expected_linear = base * (1 + rate * t)
    np.testing.assert_allclose(profile.values, expected_linear, rtol=1e-9)

    old_exponential_value_at_15y = base * (1 + rate) ** 15
    assert profile.iloc[-1] < old_exponential_value_at_15y * 0.5  # clearly not exponential


def test_bass_anchors_at_base_not_zero():
    """Session 6 regression test: the raw Bass closed-form solution is 0 at
    t=0 by construction (modelling brand-new-product diffusion from zero
    adopters) — applied directly to `base_sessions_per_day`, this made the
    simulated trajectory start at a literal zero sessions/day. Must now
    start exactly at `base_sessions_per_day` instead."""
    b = bass_diffusion_profile(100, 5, market_potential_factor=4.0,
                               p=0.03, q=0.38, start_date="2025-01-01")
    assert b.iloc[0] == pytest.approx(100.0, rel=1e-9)


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

"""
Tests for gears.simulation – ShortTermSimulator and MediumTermSimulator.
"""

from __future__ import annotations

import pandas as pd
import pytest

from gears.simulation.medium_term import MediumTermSimulator
from gears.simulation.short_term import ShortTermSimulator

# ---------------------------------------------------------------------------
# ShortTermSimulator
# ---------------------------------------------------------------------------

class TestShortTermSimulator:
    @pytest.fixture
    def short_sim(self, fitted_gmm, fitted_forecaster):
        return ShortTermSimulator(
            gmm=fitted_gmm,
            forecaster=fitted_forecaster,
            charger_mix={7.4: 0.5, 22.0: 0.5},
        )

    def test_simulate_returns_dataframe(self, short_sim):
        df = short_sim.simulate("2024-06-10", horizon=3, n_scenarios=1, seed=0)
        assert isinstance(df, pd.DataFrame)

    def test_simulate_columns(self, short_sim):
        df = short_sim.simulate("2024-06-10", horizon=3, n_scenarios=1, seed=0)
        for col in ("arrival_time", "duration", "energy", "scenario", "date"):
            assert col in df.columns, f"Missing: {col}"

    def test_simulate_duration_positive(self, short_sim):
        df = short_sim.simulate("2024-06-10", horizon=3, n_scenarios=2, seed=1)
        assert (df["duration"] > 0).all()

    def test_simulate_energy_positive(self, short_sim):
        df = short_sim.simulate("2024-06-10", horizon=3, n_scenarios=2, seed=2)
        assert (df["energy"] > 0).all()

    def test_simulate_end_time_after_arrival(self, short_sim):
        df = short_sim.simulate("2024-06-10", horizon=2, n_scenarios=1, seed=3)
        if "end_time" in df.columns:
            df["arrival_time"] = pd.to_datetime(df["arrival_time"])
            df["end_time"] = pd.to_datetime(df["end_time"])
            assert (df["end_time"] >= df["arrival_time"]).all()

    def test_simulate_scenario_column(self, short_sim):
        df = short_sim.simulate("2024-06-10", horizon=3, n_scenarios=3, seed=4)
        assert df["scenario"].nunique() == 3

    def test_simulate_reproducibility(self, short_sim):
        df1 = short_sim.simulate("2024-06-10", horizon=2, n_scenarios=2, seed=99)
        df2 = short_sim.simulate("2024-06-10", horizon=2, n_scenarios=2, seed=99)
        pd.testing.assert_frame_equal(
            df1.reset_index(drop=True), df2.reset_index(drop=True)
        )

    def test_simulate_single_day_fixed_count(self, short_sim):
        df = short_sim.simulate_single_day("2024-06-10", n_sessions=50, seed=0)
        assert len(df) == 50

    def test_simulate_single_day_has_date(self, short_sim):
        df = short_sim.simulate_single_day("2024-06-10", n_sessions=20, seed=1)
        assert "date" in df.columns

    def test_charger_power_values(self, short_sim):
        df = short_sim.simulate("2024-06-10", horizon=3, n_scenarios=2, seed=5)
        assert set(df["power_kw"].unique()).issubset({7.4, 22.0})

    def test_charger_mix_respected(self, fitted_gmm, fitted_forecaster):
        """Single-charger mix should yield only that power level."""
        sim = ShortTermSimulator(
            gmm=fitted_gmm,
            forecaster=fitted_forecaster,
            charger_mix={11.0: 1.0},
        )
        df = sim.simulate("2024-06-10", horizon=2, n_scenarios=1, seed=6)
        assert (df["power_kw"] == 11.0).all()


# ---------------------------------------------------------------------------
# MediumTermSimulator
# ---------------------------------------------------------------------------

class TestMediumTermSimulator:
    @pytest.fixture
    def medium_sim(self, fitted_gmm):
        return MediumTermSimulator(
            gmm=fitted_gmm,
            base_sessions_per_day=10,
            n_scenarios=3,
            seed=42,
        )

    def test_simulate_daily_energy_returns_df(self, medium_sim):
        df = medium_sim.simulate(years=1, output="daily_energy")
        assert isinstance(df, pd.DataFrame)

    def test_simulate_daily_energy_columns(self, medium_sim):
        df = medium_sim.simulate(years=1, output="daily_energy")
        assert "date" in df.columns
        assert "total_energy_kwh" in df.columns

    def test_simulate_daily_energy_positive(self, medium_sim):
        df = medium_sim.simulate(years=1, output="daily_energy")
        assert (df["total_energy_kwh"] >= 0).all()

    def test_simulate_n_scenarios(self, medium_sim):
        df = medium_sim.simulate(years=1, output="daily_energy")
        assert df["scenario"].nunique() == 3

    def test_simulate_hourly_energy(self, medium_sim):
        df = medium_sim.simulate(years=1, output="hourly_energy")
        assert "hour" in df.columns
        assert "energy_kwh" in df.columns
        assert df["hour"].between(0, 23).all()

    def test_simulate_many_years(self, medium_sim):
        """No horizon cap in v1.0 — long simulations should work."""
        result = medium_sim.simulate(years=6, output="daily_energy")
        assert len(result) > 0

    def test_simulate_invalid_output_raises(self, medium_sim):
        with pytest.raises(ValueError, match="output"):
            medium_sim.simulate(years=1, output="unknown")

    def test_growth_increases_sessions(self, medium_sim):
        """With positive growth, total energy in year 2 > year 1."""
        df = medium_sim.simulate(
            years=2, annual_growth_rate=0.5, output="daily_energy"
        )
        df["date"] = pd.to_datetime(df["date"])
        start = df["date"].min()
        mid = start + pd.DateOffset(years=1)
        y1 = df[df["date"] < mid]["total_energy_kwh"].mean()
        y2 = df[df["date"] >= mid]["total_energy_kwh"].mean()
        assert y2 > y1

    def test_weather_factor_applied(self, fitted_gmm):
        """Extreme summer reduction should lower energy in summer months."""
        sim = MediumTermSimulator(
            gmm=fitted_gmm, base_sessions_per_day=20, n_scenarios=3, seed=0
        )
        df_no_wf = sim.simulate(years=1, output="daily_energy")
        df_with_wf = sim.simulate(
            years=1,
            output="daily_energy",
            weather_factor={"summer": 0.01},
        )
        total_no = df_no_wf["total_energy_kwh"].sum()
        total_with = df_with_wf["total_energy_kwh"].sum()
        assert total_with < total_no

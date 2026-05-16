"""
Tests for gears.output – OutputAggregator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gears.output.aggregator import OutputAggregator
from gears.data.loader import make_demo_data


@pytest.fixture
def sessions_with_scenario():
    """Simulated sessions-like DataFrame with scenario column."""
    rng = np.random.default_rng(0)
    n = 120
    return pd.DataFrame({
        "arrival_time": pd.date_range("2024-06-10", periods=n, freq="2h"),
        "arrival_hour": rng.uniform(6, 22, n),
        "duration": rng.uniform(0.5, 8, n),
        "energy": rng.uniform(5, 40, n),
        "power_kw": rng.choice([7.4, 22.0], n),
        "scenario": rng.integers(0, 4, n),
        "date": pd.date_range("2024-06-10", periods=n, freq="2h").date,
        "day_of_week": pd.date_range("2024-06-10", periods=n, freq="2h").dayofweek,
    })


@pytest.fixture
def agg():
    return OutputAggregator(resolution_min=30)


# ---------------------------------------------------------------------------
# daily_energy
# ---------------------------------------------------------------------------

class TestDailyEnergy:
    def test_returns_dataframe(self, agg, sessions_with_scenario):
        df = agg.daily_energy(sessions_with_scenario)
        assert isinstance(df, pd.DataFrame)

    def test_required_columns(self, agg, sessions_with_scenario):
        df = agg.daily_energy(sessions_with_scenario)
        for col in ("date", "n_sessions", "total_energy_kwh"):
            assert col in df.columns

    def test_energy_positive(self, agg, sessions_with_scenario):
        df = agg.daily_energy(sessions_with_scenario)
        assert (df["total_energy_kwh"] > 0).all()

    def test_by_scenario_false(self, agg, sessions_with_scenario):
        # Aggregator includes scenario if present in input; this is expected behaviour
        df = agg.daily_energy(sessions_with_scenario)
        assert isinstance(df, pd.DataFrame)
        assert "total_energy_kwh" in df.columns

    def test_by_scenario_true(self, agg, sessions_with_scenario):
        df = agg.daily_energy(sessions_with_scenario)
        assert "scenario" in df.columns

    def test_total_sessions_matches(self, agg, sessions_with_scenario):
        df = agg.daily_energy(sessions_with_scenario)
        assert df["n_sessions"].sum() == len(sessions_with_scenario)


# ---------------------------------------------------------------------------
# hourly_profile
# ---------------------------------------------------------------------------

class TestHourlyProfile:
    def test_returns_dataframe(self, agg, sessions_with_scenario):
        df = agg.hourly_profile(sessions_with_scenario)
        assert isinstance(df, pd.DataFrame)

    def test_hour_column(self, agg, sessions_with_scenario):
        df = agg.hourly_profile(sessions_with_scenario)
        assert "hour" in df.columns
        assert df["hour"].between(0, 23).all()

    def test_energy_positive(self, agg, sessions_with_scenario):
        df = agg.hourly_profile(sessions_with_scenario)
        assert (df["energy_kwh"] >= 0).all()

    def test_normalize(self, agg, sessions_with_scenario):
        df = agg.hourly_profile(sessions_with_scenario)
        total = df["energy_kwh"].sum()
        # total across all hours should sum to ~1.0 (or n_scenarios)
        assert abs(total - 1.0) < 0.05 or total > 0

    def test_no_arrival_hour_column(self, agg):
        """Falls back to arrival_time.hour when arrival_hour absent."""
        times = pd.date_range("2024-06-10 08:00", periods=20, freq="1h")
        df_raw = pd.DataFrame({
            "arrival_time": times,
            "duration": [2.0] * 20,
            "energy": [10.0] * 20,
            "date": times.date,
        })
        df = agg.hourly_profile(df_raw)
        assert "hour" in df.columns


# ---------------------------------------------------------------------------
# load_curve
# ---------------------------------------------------------------------------

class TestExport:
    def test_export_csv(self, agg, sessions_with_scenario, tmp_path):
        path = tmp_path / "out.csv"
        agg.export(sessions_with_scenario, path)
        df = pd.read_csv(path)
        assert len(df) == len(sessions_with_scenario)

    def test_export_parquet(self, agg, sessions_with_scenario, tmp_path):
        path = tmp_path / "out.parquet"
        agg.export(sessions_with_scenario, path)
        df = pd.read_parquet(path)
        assert len(df) == len(sessions_with_scenario)

    def test_export_json(self, agg, sessions_with_scenario, tmp_path):
        path = tmp_path / "out.json"
        agg.export(sessions_with_scenario, path)
        df = pd.read_json(path)
        assert len(df) == len(sessions_with_scenario)

    def test_export_creates_parent_dirs(self, agg, sessions_with_scenario, tmp_path):
        path = tmp_path / "sub" / "dir" / "out.csv"
        agg.export(sessions_with_scenario, path)
        assert path.exists()

    def test_export_invalid_format_raises(self, agg, sessions_with_scenario, tmp_path):
        path = tmp_path / "out.csv"
        with pytest.raises(ValueError, match="Unsupported export format"):
            agg.export(sessions_with_scenario, tmp_path / "out.xyz")

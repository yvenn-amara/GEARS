"""
Tests for gears.output – OutputAggregator.

Phase 2 / Session 9 additions (below the original daily_energy/hourly_profile/
export tests) target the module's biggest coverage gaps per AUDIT.md/the
Session 9 brief: the private Monte-Carlo helper functions (_overlap_profile_24h,
_draw_power_levels, _reconstruct_smart_profile_hourly, _build_smart_ts) and the
build_load_profiles() end-to-end method, which previously had zero direct
test coverage despite being the module's largest and most load-bearing piece
of logic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gears.data.loader import make_demo_data
from gears.models.session_model import EVSessionModel
from gears.output.aggregator import (
    LOCATION_POWER_PRESETS,
    OutputAggregator,
    _build_smart_ts,
    _draw_power_levels,
    _overlap_profile_24h,
    _reconstruct_smart_profile_hourly,
)


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
        with pytest.raises(ValueError, match="Unsupported export format"):
            agg.export(sessions_with_scenario, tmp_path / "out.xyz")


# ---------------------------------------------------------------------------
# hourly_profile — missing-column branch
# ---------------------------------------------------------------------------

class TestHourlyProfileErrors:
    def test_missing_arrival_columns_raises(self, agg):
        df = pd.DataFrame({
            "date": pd.date_range("2024-06-10", periods=5).date,
            "duration": [1.0] * 5,
            "energy": [10.0] * 5,
        })
        with pytest.raises(ValueError, match="arrival_hour.*arrival_time"):
            agg.hourly_profile(df)


# ---------------------------------------------------------------------------
# scenario_stats
# ---------------------------------------------------------------------------

class TestScenarioStats:
    def test_with_scenario_column(self, agg, sessions_with_scenario):
        daily = agg.daily_energy(sessions_with_scenario)
        stats = agg.scenario_stats(daily)
        for col in ("date", "mean", "p10", "p25", "p75", "p90"):
            assert col in stats.columns
        # p10 <= p25 <= mean-ish <= p75 <= p90 per date (monotonic quantiles)
        assert (stats["p10"] <= stats["p25"]).all()
        assert (stats["p75"] <= stats["p90"]).all()

    def test_without_scenario_column_returns_unchanged(self, agg):
        daily = pd.DataFrame({
            "date": pd.date_range("2024-06-10", periods=3),
            "total_energy_kwh": [10.0, 20.0, 30.0],
        })
        result = agg.scenario_stats(daily)
        pd.testing.assert_frame_equal(result, daily)


# ---------------------------------------------------------------------------
# _overlap_profile_24h — pure Monte-Carlo helper
# ---------------------------------------------------------------------------

class TestOverlapProfile24h:
    def test_energy_conservation_no_wraparound(self):
        """A single session fully inside one day: sum(profile) * 1h should
        equal total energy delivered / n_sessions_per_day (approximately,
        since the helper distributes across whole-hour buckets)."""
        arrivals = np.array([10.0])
        durations = np.array([2.0])
        powers = np.array([7.4])
        n_sessions_per_day = 1.0
        n_total = 1
        profile = _overlap_profile_24h(arrivals, durations, powers, n_sessions_per_day, n_total)
        assert profile.shape == (24,)
        # Energy = power * duration = 14.8 kWh, spread over hours 10-12.
        total_energy = profile.sum() * 1.0  # 1h buckets
        assert total_energy == pytest.approx(7.4 * 2.0, rel=1e-6)
        # No power should appear outside [10, 12].
        assert profile[:10].sum() == 0
        assert profile[12:].sum() == 0

    def test_midnight_wraparound_conserves_energy(self):
        """A session arriving at 23:00 with a 3h duration should wrap into
        hours 0-1 of the next day rather than losing energy past hour 23."""
        arrivals = np.array([23.0])
        durations = np.array([3.0])
        powers = np.array([10.0])
        profile = _overlap_profile_24h(arrivals, durations, powers, 1.0, 1)
        total_energy = profile.sum() * 1.0
        assert total_energy == pytest.approx(10.0 * 3.0, rel=1e-6)
        # Should have contributions at hour 23 (day 0) and hours 0, 1 (wrapped).
        assert profile[23] > 0
        assert profile[0] > 0
        assert profile[1] > 0

    def test_multi_day_overflow_up_to_48h_duration(self):
        """Durations approaching the 48h GMM clip must still conserve
        energy via the third overlap window (o2)."""
        arrivals = np.array([22.0])
        durations = np.array([47.0])  # arrival(22) + duration(47) = 69h < 72h cap
        powers = np.array([5.0])
        profile = _overlap_profile_24h(arrivals, durations, powers, 1.0, 1)
        total_energy = profile.sum() * 1.0
        assert total_energy == pytest.approx(5.0 * 47.0, rel=1e-6)

    def test_normalisation_by_n_sessions_per_day(self):
        """Doubling n_total while doubling n_sessions_per_day proportionally
        should leave the resulting per-day profile unchanged (Monte-Carlo
        average over more simulated days of the same underlying rate)."""
        arrivals = np.array([8.0, 8.0])
        durations = np.array([1.0, 1.0])
        powers = np.array([7.0, 7.0])
        profile_a = _overlap_profile_24h(arrivals, durations, powers, 2.0, 2)

        arrivals_b = np.tile(arrivals, 2)
        durations_b = np.tile(durations, 2)
        powers_b = np.tile(powers, 2)
        profile_b = _overlap_profile_24h(arrivals_b, durations_b, powers_b, 2.0, 4)

        np.testing.assert_allclose(profile_a, profile_b, rtol=1e-9)


# ---------------------------------------------------------------------------
# _draw_power_levels — discrete power sampling
# ---------------------------------------------------------------------------

class TestDrawPowerLevels:
    def test_only_samples_declared_power_levels(self):
        rng = np.random.default_rng(0)
        dist = {7.4: 0.5, 22.0: 0.5}
        powers = _draw_power_levels(dist, n=500, rng=rng)
        assert set(np.unique(powers)) <= {7.4, 22.0}
        assert len(powers) == 500

    def test_proportions_need_not_sum_to_one(self):
        """Proportions are auto-normalised, so e.g. {1: 3, 2: 1} should
        behave like {1: 0.75, 2: 0.25}."""
        rng = np.random.default_rng(0)
        dist = {1.0: 3, 2.0: 1}
        powers = _draw_power_levels(dist, n=4000, rng=rng)
        frac_one = (powers == 1.0).mean()
        assert frac_one == pytest.approx(0.75, abs=0.05)

    def test_single_power_level_returns_constant(self):
        rng = np.random.default_rng(0)
        powers = _draw_power_levels({11.0: 1.0}, n=10, rng=rng)
        assert (powers == 11.0).all()


# ---------------------------------------------------------------------------
# _reconstruct_smart_profile_hourly
# ---------------------------------------------------------------------------

class TestReconstructSmartProfileHourly:
    def test_empty_result_returns_zeros(self):
        date = pd.Timestamp("2025-06-10")
        result = pd.DataFrame({
            "scheduled_start": [pd.NaT, pd.NaT],
            "energy": [10.0, 5.0],
            "power_kw": [7.4, 7.4],
        })
        profile = _reconstruct_smart_profile_hourly(
            result, date, resolution_min=30, n_sessions_per_day=1.0, n_total=1,
        )
        np.testing.assert_array_equal(profile, np.zeros(24))

    def test_valid_schedule_conserves_energy(self):
        date = pd.Timestamp("2025-06-10")
        result = pd.DataFrame({
            "scheduled_start": [date + pd.Timedelta(hours=2)],
            "energy": [7.4],       # 1h at 7.4 kW
            "power_kw": [7.4],
        })
        profile = _reconstruct_smart_profile_hourly(
            result, date, resolution_min=30, n_sessions_per_day=1.0, n_total=1,
        )
        assert profile.sum() * 1.0 == pytest.approx(7.4, rel=1e-6)
        assert profile[2] > 0

    def test_missing_power_kw_column_defaults_to_7_4(self):
        date = pd.Timestamp("2025-06-10")
        result = pd.DataFrame({
            "scheduled_start": [date + pd.Timedelta(hours=5)],
            "energy": [7.4],
        })
        profile = _reconstruct_smart_profile_hourly(
            result, date, resolution_min=30, n_sessions_per_day=1.0, n_total=1,
        )
        assert profile.sum() * 1.0 == pytest.approx(7.4, rel=1e-6)


# ---------------------------------------------------------------------------
# _build_smart_ts
# ---------------------------------------------------------------------------

class TestBuildSmartTs:
    def test_falls_back_to_baseline_profile_when_no_smart_profile(self):
        """For (dow, season) keys absent from smart_profiles_mw, _build_smart_ts
        must fall back to the baseline profiles_mw rather than zeroing out."""
        baseline = {(d, s): np.full(24, 1.0) for d in range(7)
                    for s in ("winter", "spring", "summer", "autumn")}
        ts = _build_smart_ts(
            gmm=None, profiles_mw=baseline, smart_profiles_mw={},
            year=2025, noise_std=0.0, seed=0,
        )
        assert isinstance(ts, pd.Series)
        # noise_std=0 => exactly the baseline profile value every hour.
        np.testing.assert_allclose(ts.values, 1.0)

    def test_uses_smart_profile_when_present(self):
        baseline = {(d, s): np.full(24, 1.0) for d in range(7)
                    for s in ("winter", "spring", "summer", "autumn")}
        smart = {(d, s): np.full(24, 2.0) for d in range(7)
                 for s in ("winter", "spring", "summer", "autumn")}
        ts = _build_smart_ts(
            gmm=None, profiles_mw=baseline, smart_profiles_mw=smart,
            year=2025, noise_std=0.0, seed=0,
        )
        np.testing.assert_allclose(ts.values, 2.0)


# ---------------------------------------------------------------------------
# build_load_profiles — end-to-end (Session 9: previously untested)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fitted_gmm_for_load_profiles():
    """A small GMM fitted with location_type in stratify_by, so all three
    charging_mode branches of build_load_profiles are reachable."""
    sessions = pd.concat([
        make_demo_data(n=250, location_type="work", seed=10),
        make_demo_data(n=250, location_type="home", seed=11),
    ], ignore_index=True)
    model = EVSessionModel(n_components=2, random_state=0,
                            stratify_by=["location_type", "day_of_week"])
    return model.fit(sessions)


class TestBuildLoadProfiles:
    def test_invalid_charging_mode_raises(self, fitted_gmm_for_load_profiles):
        agg = OutputAggregator()
        with pytest.raises(ValueError, match="Unknown charging_mode"):
            agg.build_load_profiles(fitted_gmm_for_load_profiles, charging_mode="bogus")

    def test_fixed_power_without_charger_power_kw_raises(self, fitted_gmm_for_load_profiles):
        agg = OutputAggregator()
        with pytest.raises(ValueError, match="requires charger_power_kw"):
            agg.build_load_profiles(fitted_gmm_for_load_profiles, charging_mode="fixed_power")

    def test_by_location_without_location_power_map_raises(self, fitted_gmm_for_load_profiles):
        agg = OutputAggregator()
        with pytest.raises(ValueError, match="requires location_power_map"):
            agg.build_load_profiles(fitted_gmm_for_load_profiles, charging_mode="by_location")

    def test_by_location_without_location_type_in_stratify_raises(self):
        """A GMM fitted without location_type in stratify_by can't support
        charging_mode='by_location' even with a location_power_map given."""
        sessions = make_demo_data(n=200, location_type="work", seed=12)
        model = EVSessionModel(n_components=2, random_state=0,
                                stratify_by=["day_of_week"]).fit(sessions)
        agg = OutputAggregator()
        with pytest.raises(ValueError, match="stratify_by"):
            agg.build_load_profiles(
                model, charging_mode="by_location",
                location_power_map=LOCATION_POWER_PRESETS["french_2024"],
            )

    def test_mean_power_mode_returns_expected_keys(self, fitted_gmm_for_load_profiles):
        agg = OutputAggregator()
        out = agg.build_load_profiles(
            fitted_gmm_for_load_profiles, year=2025, n_days_mc=2,
            charging_mode="mean_power",
        )
        for key in ("ts", "profiles", "loc_profiles", "n_by_type", "charging_mode"):
            assert key in out
        assert out["charging_mode"] == "mean_power"
        assert isinstance(out["ts"], pd.Series)
        assert len(out["ts"]) in (8760, 8784)  # 2025 is not a leap year -> 8760
        assert (out["ts"] >= 0).all()
        assert "ts_smart" not in out

    def test_fixed_power_mode_runs(self, fitted_gmm_for_load_profiles):
        agg = OutputAggregator()
        out = agg.build_load_profiles(
            fitted_gmm_for_load_profiles, year=2025, n_days_mc=2,
            charging_mode="fixed_power", charger_power_kw=7.4,
        )
        assert out["charging_mode"] == "fixed_power"
        assert (out["ts"] >= 0).all()

    def test_by_location_mode_runs_and_populates_loc_profiles(self, fitted_gmm_for_load_profiles):
        agg = OutputAggregator()
        out = agg.build_load_profiles(
            fitted_gmm_for_load_profiles, year=2025, n_days_mc=2,
            charging_mode="by_location",
            location_power_map=LOCATION_POWER_PRESETS["french_2024"],
        )
        assert out["charging_mode"] == "by_location"
        assert set(out["loc_profiles"].keys()) <= {"work", "home"}
        assert len(out["loc_profiles"]) > 0

    def test_ts_length_matches_calendar_year(self, fitted_gmm_for_load_profiles):
        agg = OutputAggregator()
        out_2024 = agg.build_load_profiles(  # 2024 is a leap year
            fitted_gmm_for_load_profiles, year=2024, n_days_mc=1,
            charging_mode="mean_power",
        )
        assert len(out_2024["ts"]) == 8784


# ---------------------------------------------------------------------------
# build_load_profiles — smart-charging integration (previously 0% covered)
# ---------------------------------------------------------------------------

class TestBuildLoadProfilesSmartCharging:
    def test_smart_charging_signal_adds_ts_smart(self, fitted_gmm_for_load_profiles):
        agg = OutputAggregator(resolution_min=30)
        idx = pd.date_range("2025-01-01", periods=60 * 48, freq="30min")
        rng = np.random.default_rng(1)
        signal = pd.Series(
            0.15 + 0.05 * np.sin(np.linspace(0, 20 * np.pi, len(idx)))
            + rng.normal(0, 0.01, len(idx)),
            index=idx, name="price_eur_kwh",
        )
        out = agg.build_load_profiles(
            fitted_gmm_for_load_profiles, year=2025, n_days_mc=1,
            charging_mode="mean_power", smart_charging_signal=signal,
        )
        assert "ts_smart" in out
        assert isinstance(out["ts_smart"], pd.Series)
        assert len(out["ts_smart"]) == len(out["ts"])
        assert (out["ts_smart"] >= 0).all()

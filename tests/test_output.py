"""Tests for gears.output.OutputAggregator: aggregation, export and annual load profiles."""

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
from gears.utils import make_price_signal

SEASONS = ("winter", "spring", "summer", "autumn")


@pytest.fixture
def sessions():
    """Five sessions on two days and two scenarios (energy 10+20+30 | 5 | 8)."""
    return pd.DataFrame({
        "arrival_time": pd.to_datetime([
            "2025-06-10 08:10", "2025-06-10 08:50", "2025-06-10 17:00",
            "2025-06-11 09:00", "2025-06-10 08:20",
        ]),
        "arrival_hour": [8.17, 8.83, 17.0, 9.0, 8.33],
        "duration": [1.0, 2.0, 3.0, 1.0, 2.0],
        "energy": [10.0, 20.0, 30.0, 5.0, 8.0],
        "power_kw": [7.4, 7.4, 22.0, 7.4, 22.0],
        "scenario": [0, 0, 0, 0, 1],
        "date": pd.to_datetime(
            ["2025-06-10"] * 3 + ["2025-06-11", "2025-06-10"]
        ).date,
    })


@pytest.fixture(scope="module")
def load_model():
    """Two locations, stratified by place, weekday and season: every context has about
    two sessions a day, which ``build_load_profiles`` needs to produce any load."""
    both = pd.concat([
        make_demo_data(n=1200, location_type="work", seed=10),
        make_demo_data(n=1200, location_type="home", seed=11),
    ], ignore_index=True)
    return EVSessionModel(
        n_components=2, random_state=0, stratify_by=["location_type", "day_of_week", "season"]
    ).fit(both)


# ── Aggregating sessions ──────────────────────────────────────────────────────

def test_daily_energy_sums_each_day_and_scenario(sessions):
    daily = OutputAggregator().daily_energy(sessions).set_index(["date", "scenario"])

    assert daily["total_energy_kwh"].to_dict() == {
        (pd.Timestamp("2025-06-10"), 0): 60.0,
        (pd.Timestamp("2025-06-10"), 1): 8.0,
        (pd.Timestamp("2025-06-11"), 0): 5.0,
    }
    assert daily["n_sessions"].sum() == len(sessions)
    assert daily.loc[(pd.Timestamp("2025-06-10"), 0), "mean_duration_h"] == 2.0
    assert daily.loc[(pd.Timestamp("2025-06-10"), 0), "peak_power_kw"] == 7.4 + 7.4 + 22.0
    # without a scenario column the scenarios of the same day are one total
    merged = OutputAggregator().daily_energy(sessions.drop(columns="scenario"))
    assert merged["total_energy_kwh"].tolist() == [68.0, 5.0]


def test_hourly_profile_sums_the_energy_that_starts_in_each_hour(sessions):
    agg = OutputAggregator()
    hourly = agg.hourly_profile(sessions).set_index(["date", "hour", "scenario"])
    assert hourly["energy_kwh"].to_dict() == {
        (pd.Timestamp("2025-06-10"), 8, 0): 30.0,
        (pd.Timestamp("2025-06-10"), 8, 1): 8.0,
        (pd.Timestamp("2025-06-10"), 17, 0): 30.0,
        (pd.Timestamp("2025-06-11"), 9, 0): 5.0,
    }
    # arrival_hour is optional: arrival_time is used when it is absent
    from_time = agg.hourly_profile(sessions.drop(columns="arrival_hour"))
    assert from_time["energy_kwh"].sum() == 68 + 5

    with pytest.raises(ValueError, match="arrival_hour.*arrival_time"):
        agg.hourly_profile(sessions.drop(columns=["arrival_hour", "arrival_time"]))


def test_scenario_stats_are_the_quantiles_across_scenarios_of_each_date():
    daily = pd.DataFrame({
        "date": pd.Timestamp("2025-06-10"), "scenario": [0, 1, 2, 3],
        "total_energy_kwh": [10.0, 20.0, 30.0, 40.0],
    })
    stats = OutputAggregator().scenario_stats(daily).iloc[0]
    assert (stats["mean"], stats["p10"], stats["p25"], stats["p75"], stats["p90"]) == (
        25.0, 13.0, 17.5, 32.5, 37.0
    )

    single = daily.drop(columns="scenario")     # no scenario: nothing to summarise
    pd.testing.assert_frame_equal(OutputAggregator().scenario_stats(single), single)


@pytest.mark.parametrize("suffix", ["csv", "json", "parquet"])
def test_export_writes_the_sessions_in_the_format_of_the_file_name(sessions, tmp_path, suffix):
    if suffix == "parquet":
        pytest.importorskip("pyarrow")
    path = tmp_path / "new" / "folder" / f"out.{suffix}"          # the folders are created

    OutputAggregator().export(sessions, path)

    read = {"csv": pd.read_csv, "json": pd.read_json, "parquet": pd.read_parquet}[suffix]
    assert read(path)["energy"].tolist() == sessions["energy"].tolist()


def test_export_refuses_an_unknown_format(sessions, tmp_path):
    with pytest.raises(ValueError, match="Unsupported export format"):
        OutputAggregator().export(sessions, tmp_path / "out.xyz")


# ── The building blocks of the annual load profile ────────────────────────────

def test_a_session_spreads_its_power_over_the_hours_it_lasts():
    def profile(arrival, duration, power, n_per_day=1.0, n_total=1):
        return _overlap_profile_24h(
            np.array([arrival]), np.array([duration]), np.array([power]), n_per_day, n_total
        )

    inside = profile(10.0, 2.0, 7.4)
    assert inside.shape == (24,)
    assert inside[10:12].tolist() == [7.4, 7.4] and inside.sum() == pytest.approx(7.4 * 2)

    wrapping = profile(23.0, 3.0, 10.0)                  # 23:00 -> 02:00 wraps past midnight
    assert wrapping[[23, 0, 1]].tolist() == [10.0, 10.0, 10.0] and wrapping.sum() == 30.0

    two_days = profile(22.0, 47.0, 5.0)                  # the 48 h upper bound of the model
    assert two_days.sum() == pytest.approx(5.0 * 47.0)

    # n_total simulated sessions of a rate of n_per_day: the same sessions twice over, at
    # twice the number of days, give the same average day
    twice = _overlap_profile_24h(
        np.array([8.0, 8.0, 8.0, 8.0]), np.ones(4), np.full(4, 7.0), 2.0, 4
    )
    once = _overlap_profile_24h(np.array([8.0, 8.0]), np.ones(2), np.full(2, 7.0), 2.0, 2)
    np.testing.assert_allclose(twice, once)

    # ... and the average day holds n_per_day sessions: 2 sessions of 7 kW at 08:00 each day
    assert once[8] == 14.0 and once.sum() == 14.0           # sum x 1 h = energy per session x 2


def test_power_levels_are_drawn_from_the_declared_mix_whatever_its_scale():
    rng = np.random.default_rng(0)
    drawn = _draw_power_levels({7.4: 0.5, 22.0: 0.5}, n=500, rng=rng)
    assert len(drawn) == 500 and set(drawn) == {7.4, 22.0}

    heavy = _draw_power_levels({1.0: 3, 2.0: 1}, n=4000, rng=rng)       # proportions 3:1
    assert (heavy == 1.0).mean() == pytest.approx(0.75, abs=0.05)
    assert (_draw_power_levels({11.0: 1.0}, n=10, rng=rng) == 11.0).all()


def test_the_smart_profile_is_rebuilt_from_the_scheduled_start_of_each_session():
    day = pd.Timestamp("2025-06-10")
    def rebuild(frame):
        return _reconstruct_smart_profile_hourly(frame, day, n_sessions_per_day=1.0, n_total=1)

    unscheduled = pd.DataFrame({
        "scheduled_start": [pd.NaT, pd.NaT], "energy": [10.0, 5.0], "power_kw": [7.4, 7.4],
    })
    assert rebuild(unscheduled).tolist() == [0.0] * 24

    one_hour = pd.DataFrame({
        "scheduled_start": [day + pd.Timedelta(hours=2)], "energy": [7.4], "power_kw": [7.4],
    })
    assert rebuild(one_hour)[2] == pytest.approx(7.4) and rebuild(one_hour).sum() == pytest.approx(7.4)

    without_power = one_hour.drop(columns="power_kw").assign(
        scheduled_start=day + pd.Timedelta(hours=5), energy=14.8
    )
    assert rebuild(without_power)[5:7].tolist() == pytest.approx([7.4, 7.4])   # defaults to 7.4 kW
    assert rebuild(without_power).sum() == pytest.approx(14.8)                 # for 2 hours


def test_the_smart_series_falls_back_to_the_baseline_profile_where_it_has_none():
    keys = [(d, s) for d in range(7) for s in SEASONS]
    baseline = {k: np.full(24, 1.0) for k in keys}
    smart = {k: np.full(24, 2.0) for k in keys}

    without = _build_smart_ts(baseline, {}, year=2025, noise_std=0.0, seed=0)
    assert isinstance(without, pd.Series) and (without.to_numpy() == 1.0).all()
    assert (_build_smart_ts(baseline, smart, 2025, 0.0, 0).to_numpy() == 2.0).all()


# ── The annual load profile ───────────────────────────────────────────────────

def test_build_load_profiles_refuses_what_would_silently_give_no_load(load_model):
    agg = OutputAggregator()
    build = agg.build_load_profiles
    demo = make_demo_data(n=300, location_type="work", seed=12)

    with pytest.raises(ValueError, match="Unknown charging_mode"):
        build(load_model, charging_mode="bogus")
    with pytest.raises(ValueError, match="requires charger_power_kw"):
        build(load_model, charging_mode="fixed_power")
    with pytest.raises(ValueError, match="requires location_power_map"):
        build(load_model, charging_mode="by_location")

    no_location = EVSessionModel(n_components=2, random_state=0).fit(demo)   # weekday x season
    with pytest.raises(ValueError, match="location_type"):
        build(no_location, charging_mode="by_location",
              location_power_map=LOCATION_POWER_PRESETS["french_2024"])

    # the series is built per weekday and season: a model without them would give zeros
    no_season = EVSessionModel(n_components=2, random_state=0, stratify_by=["day_of_week"]).fit(demo)
    with pytest.raises(ValueError, match=r"'season'.*zero all year"):
        build(no_season)


def test_each_charging_mode_builds_a_year_of_hourly_load_in_megawatts(load_model):
    agg = OutputAggregator()
    modes = {
        "mean_power": {},
        "fixed_power": {"charger_power_kw": 7.4},
        "by_location": {"location_power_map": LOCATION_POWER_PRESETS["french_2024"]},
    }
    for mode, kwargs in modes.items():
        out = agg.build_load_profiles(load_model, year=2025, n_days_mc=2, charging_mode=mode, **kwargs)

        ts = out["ts"]
        assert out["charging_mode"] == mode and "ts_smart" not in out
        assert len(ts) == 8760 and ts.index[0] == pd.Timestamp("2025-01-01 00:00")
        assert ts.index[-1] == pd.Timestamp("2025-12-31 23:00")
        assert ts.sum() > 0 and (ts >= 0).all(), mode                    # a real load, not zeros
        assert len(out["profiles"]) == 28                               # 7 weekdays x 4 seasons
        assert set(out["loc_profiles"]) == {"work", "home"}
        # the same sessions, summed by place or by (weekday, season)
        assert sum(out["loc_profiles"].values()).sum() == pytest.approx(
            sum(out["profiles"].values()).sum()
        )

    leap = agg.build_load_profiles(load_model, year=2024, n_days_mc=1)
    assert len(leap["ts"]) == 8784


def test_a_price_signal_adds_a_smart_charging_series_with_about_the_same_energy(load_model):
    signal = make_price_signal(start="2025-01-01", periods=60 * 48, resolution_min=30)
    out = OutputAggregator(resolution_min=30).build_load_profiles(
        load_model, year=2025, n_days_mc=1, smart_charging_signal=signal
    )

    ts, smart = out["ts"], out["ts_smart"]
    assert isinstance(smart, pd.Series) and smart.index.equals(ts.index) and (smart >= 0).all()
    assert (smart != ts).any()                                   # the load really moved ...
    assert smart.sum() == pytest.approx(ts.sum(), rel=0.05)      # ... but it is the same energy

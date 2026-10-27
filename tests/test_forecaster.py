"""Tests for the daily session-count forecasters (gears.models.forecaster).

SARIMA's policy (try it, announce the switch to 'probabilistic' when it fails) and the
errors for bad input or an unfitted forecaster are checked in ``test_sanity.py``; the
deep-learning forecasters that need neuralforecast are in ``test_forecaster_dl.py``.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from gears.data.schemas import validate_dataframe
from gears.models.forecaster import (
    PersistenceForecaster,
    SessionForecaster,
    sessions_to_daily_counts,
)
from gears.models.forecaster_dl import NHiTSForecaster, TransformerForecaster

FORECAST_COLUMNS = {"date", "scenario", "n_sessions"}


def make_sessions(n=300, seed=0):
    rng = np.random.default_rng(seed)
    return validate_dataframe(pd.DataFrame({
        "arrival_time": pd.date_range("2024-01-01", periods=n, freq="6h"),
        "duration": rng.uniform(0.5, 8, n),
        "energy": rng.uniform(2, 40, n),
    }))


def make_weekly_sessions(weeks=12, weekday=40, weekend=3, seed=0):
    """From Monday 2024-01-01 to a Sunday: about 40 sessions on a weekday, 3 on a weekend day."""
    rng = np.random.default_rng(seed)
    rows = [
        {
            "arrival_time": day + pd.Timedelta(hours=float(rng.uniform(7, 21))),
            "duration": 2.0,
            "energy": 10.0,
        }
        for day in pd.date_range("2024-01-01", periods=7 * weeks)
        for _ in range(max(int(rng.normal(weekend if day.day_of_week >= 5 else weekday, 2)), 1))
    ]
    return validate_dataframe(pd.DataFrame(rows))


def make_gap_sessions(n_per_day=3, seed=42):
    """Three sessions a day from 2024-01-01 to 03-31, nothing until 04-15, then up to 06-30."""
    rng = np.random.default_rng(seed)
    days = list(pd.date_range("2024-01-01", "2024-03-31")) + list(
        pd.date_range("2024-04-15", "2024-06-30")
    )
    rows = [
        {
            "arrival_time": day + pd.Timedelta(hours=float(rng.uniform(7, 21))),
            "duration": float(rng.uniform(0.5, 8.0)),
            "energy": float(rng.uniform(2.0, 40.0)),
        }
        for day in days for _ in range(n_per_day)
    ]
    return validate_dataframe(pd.DataFrame(rows))


def make_uneven_sessions(n_days=80):
    """1 to 6 sessions a day, in a cycle of six days: no two weeks look alike."""
    rows = [
        {"arrival_time": day + pd.Timedelta(hours=8 + k), "duration": 2.0, "energy": 10.0}
        for i, day in enumerate(pd.date_range("2024-01-01", periods=n_days))
        for k in range(1 + i % 6)
    ]
    return validate_dataframe(pd.DataFrame(rows))


def make_spiky_sessions():
    """One session on the first and last of 60 days and 40 on a day in between: a mean of
    0.7 sessions a day for a standard deviation of 5, so a normal draw is often negative."""
    days = [pd.Timestamp("2024-01-01")] * 1 + [pd.Timestamp("2024-01-31")] * 40 + [
        pd.Timestamp("2024-02-29")
    ]
    rows = [
        {"arrival_time": day + pd.Timedelta(hours=8 + k % 8), "duration": 2.0, "energy": 10.0}
        for k, day in enumerate(days)
    ]
    return validate_dataframe(pd.DataFrame(rows))


# ── sessions_to_daily_counts ──────────────────────────────────────────────────

def test_daily_counts_are_a_contiguous_zero_filled_series_on_the_real_dates():
    """``validate_dataframe`` stores ``date`` as Python date objects: the counts used to be
    anchored at the Unix epoch (1970) because the index was not converted first."""
    df = make_gap_sessions()
    assert df["date"].dtype == object

    counts = sessions_to_daily_counts(df)

    assert isinstance(counts.index, pd.DatetimeIndex)
    assert counts.index[0] == pd.Timestamp("2024-01-01")
    assert counts.index[-1] == pd.Timestamp("2024-06-30")
    assert len(counts) == (counts.index[-1] - counts.index[0]).days + 1       # no missing day
    assert not counts.isna().any()
    assert (counts[pd.date_range("2024-04-01", "2024-04-14")] == 0).all()     # the gap is zeros
    assert counts.sum() == len(df)
    assert (counts[pd.date_range("2024-05-01", "2024-05-31")] == 3).all()


# ── SARIMA ────────────────────────────────────────────────────────────────────

def test_sarima_gives_one_non_negative_trajectory_per_scenario_from_the_start_date():
    fc = SessionForecaster(method="sarima", use_holidays=False).fit(make_sessions(300))
    assert fc.method_ == "sarima"           # it really was SARIMA, not the 'probabilistic' switch

    pred = fc.predict(horizon=5, n_scenarios=10, start_date="2025-01-01", seed=0)

    assert FORECAST_COLUMNS <= set(pred.columns)
    assert len(pred) == 50 and pred["scenario"].nunique() == 10
    assert pd.Timestamp(pred["date"].min()) == pd.Timestamp("2025-01-01")
    assert (pred["n_sessions"] >= 0).all()


# ── The interface shared by the forecasters ───────────────────────────────────

@pytest.mark.parametrize("make_forecaster", [
    lambda: SessionForecaster(method="probabilistic"),
    lambda: PersistenceForecaster(n_weeks=1),
], ids=["probabilistic", "persistence"])
def test_every_forecaster_returns_scenarios_in_the_same_format(make_forecaster):
    fc = make_forecaster().fit(make_sessions(200))

    pred = fc.predict(horizon=7, n_scenarios=3, start_date="2025-01-01", seed=0)

    assert FORECAST_COLUMNS <= set(pred.columns)
    assert len(pred) == 21 and pred["scenario"].nunique() == 3
    assert pd.Timestamp(pred["date"].min()) == pd.Timestamp("2025-01-01")
    assert (pred["n_sessions"] >= 0).all()
    pd.testing.assert_frame_equal(
        pred, fc.predict(horizon=7, n_scenarios=3, start_date="2025-01-01", seed=0)
    )


def test_a_forecast_is_clipped_at_zero_whatever_the_noise():
    df = make_spiky_sessions()
    counts = sessions_to_daily_counts(df)
    assert counts.std() > 5 * counts.mean()                       # a normal draw is often < 0

    probabilistic = SessionForecaster(method="probabilistic").fit(df)
    pred = probabilistic.predict(horizon=30, n_scenarios=20, seed=0)
    assert (pred["n_sessions"] >= 0).all()
    assert (pred["n_sessions"] == 0).any()                        # and it was clipped, not shifted

    sarima = SessionForecaster(method="sarima", use_holidays=False).fit(make_sessions(300))
    assert sarima.method_ == "sarima"
    sarima._point_forecast = lambda horizon, future_exog: np.full(horizon, -50.0)   # a bad fit
    pred = sarima.predict(horizon=5, n_scenarios=5, seed=0)
    assert (pred["n_sessions"] == 0).all()


# ── Persistence ───────────────────────────────────────────────────────────────

def test_persistence_repeats_the_count_of_the_same_weekday_n_weeks_ago():
    df = make_uneven_sessions()
    counts = sessions_to_daily_counts(df)
    start = counts.index[-1] + pd.Timedelta(days=1)
    forecasts = {}
    for n_weeks in (1, 2):
        pred = PersistenceForecaster(n_weeks=n_weeks).fit(df).predict(
            horizon=7, n_scenarios=1, start_date=start
        )
        forecasts[n_weeks] = pred["n_sessions"].tolist()
        assert forecasts[n_weeks] == [counts[d - pd.Timedelta(weeks=n_weeks)] for d in pred["date"]]
    assert forecasts[1] != forecasts[2]               # the two look-backs are told apart


def test_persistence_remembers_the_days_without_sessions():
    """Days with no session are real zeros: they used to be dropped, so the forecast for
    the days after a gap picked an older, non-zero value instead of 0."""
    fc = PersistenceForecaster(n_weeks=1).fit(make_gap_sessions())

    pred = fc.predict(horizon=7, n_scenarios=1, start_date="2024-04-08")   # a week after the gap

    assert pred["n_sessions"].tolist() == [0.0] * 7
    assert (fc.predict(horizon=14, n_scenarios=3, seed=0)["n_sessions"] >= 0).all()


# ── Deep-learning forecasters: what needs neither torch nor neuralforecast ────

@pytest.mark.parametrize("cls", [NHiTSForecaster, TransformerForecaster])
def test_a_dl_forecaster_without_neuralforecast_says_how_to_install_it(cls, monkeypatch):
    monkeypatch.setitem(sys.modules, "neuralforecast", None)   # simulate "not installed"
    assert cls.is_available() is False
    with pytest.raises(ImportError, match=r'pip install "gears-ev\[dl\]"'):
        cls(horizon=7).fit(make_sessions(100))
    with pytest.raises(NotFittedError, match="fit"):
        cls(horizon=7).predict(horizon=7)


def test_nhits_defaults_keep_the_context_window_short_and_the_series_scaled():
    """The old default (input_size = 4 x horizon) used most of a 450-day training set as
    a single window and the counts were not scaled: the forecast collapsed to a random walk."""
    assert NHiTSForecaster(horizon=30).input_size == 60
    assert NHiTSForecaster(horizon=14, input_size=20).input_size == 20
    assert NHiTSForecaster().scaler_type == "standard"


# ---------------------------------------------------------------------------
# SARIMA: a forecast that starts later than the day after the data keeps the weekdays
# ---------------------------------------------------------------------------

def mean_per_date(forecast):
    return forecast.groupby("date")["n_sessions"].mean()


@pytest.fixture(scope="module")
def weekly_sarima():
    """SARIMA fitted on 12 weeks that end on Sunday 2024-03-24: weekends are almost empty."""
    forecaster = SessionForecaster(method="sarima").fit(make_weekly_sessions())
    assert forecaster.method_ == "sarima", "the test needs SARIMA to fit; it would prove nothing"
    return forecaster


def test_sarima_gives_a_late_start_date_the_forecast_of_its_own_weekdays(weekly_sarima):
    # starts on a Thursday, three days after the day that follows the data
    late = mean_per_date(weekly_sarima.predict(horizon=7, n_scenarios=300, seed=0,
                                               start_date="2024-03-28"))
    weekend = late[late.index.day_of_week >= 5]
    weekdays = late[late.index.day_of_week < 5]
    assert list(late.index.day_name()[:3]) == ["Thursday", "Friday", "Saturday"]
    assert weekend.max() < 0.25 * weekdays.min(), (
        "Saturday and Sunday are the quiet days of the training data; a forecast shifted "
        f"by the gap would put its quiet days elsewhere:\n{late}"
    )
    # same dates as seen from the day after the data: the same forecast, up to the noise
    early = mean_per_date(weekly_sarima.predict(horizon=14, n_scenarios=300, seed=0))
    both = late.index.intersection(early.index)
    assert len(both) == 7
    assert (late[both] - early[both]).abs().max() < 4


def test_sarima_forecast_far_after_the_data_warns_once_and_keeps_the_weekdays(weekly_sarima):
    forecaster = SessionForecaster(method="sarima").fit(make_weekly_sessions())
    with pytest.warns(UserWarning, match=r"starts 98 days after the last day of the training data"):
        far = forecaster.predict(horizon=7, n_scenarios=300, seed=1, start_date="2024-07-01")
    assert far["date"].iloc[0].day_name() == "Monday"
    means = mean_per_date(far)
    # three months out the level has drifted a little: the weekend is still the quiet part
    assert means[means.index.day_of_week >= 5].max() < 0.4 * means[means.index.day_of_week < 5].min()
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        forecaster.predict(horizon=7, n_scenarios=2, seed=1, start_date="2024-07-01")  # once


def test_sarima_cannot_start_in_the_past_but_keeps_the_weekdays_and_says_so(weekly_sarima):
    forecaster = SessionForecaster(method="sarima").fit(make_weekly_sessions())
    with pytest.warns(UserWarning, match=r"not after the last day of the training data"):
        past = forecaster.predict(horizon=7, n_scenarios=300, seed=2, start_date="2024-02-03")
    means = mean_per_date(past)
    assert past["date"].iloc[0].day_name() == "Saturday"
    assert means[means.index.day_of_week >= 5].max() < 0.25 * means[means.index.day_of_week < 5].min()


def test_the_default_start_is_the_day_after_the_data_and_triggers_no_warning(weekly_sarima):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        forecast = weekly_sarima.predict(horizon=7, n_scenarios=2, seed=0)
    assert forecast["date"].iloc[0] == pd.Timestamp("2024-03-25")

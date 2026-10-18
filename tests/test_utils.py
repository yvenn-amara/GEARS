"""Tests for gears.utils: example signals and forecast metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gears.utils import (
    forecast_metrics,
    mae,
    make_price_signal,
    make_res_signal,
    mape,
    rmse,
)


def test_a_price_signal_starts_where_asked_with_the_requested_step_and_stays_positive():
    for pattern in ("flat", "day_night", "spot_like"):
        signal = make_price_signal("2025-06-01", periods=96, resolution_min=30, pattern=pattern)
        assert len(signal) == 96 and (signal > 0).all(), pattern
        assert signal.index[0] == pd.Timestamp("2025-06-01")
        assert signal.index[1] - signal.index[0] == pd.Timedelta(minutes=30)

    day_night = make_price_signal("2025-06-01", periods=48, pattern="day_night", noise_std=0.0)
    assert day_night["2025-06-01 03:00"] < day_night["2025-06-01 12:00"] < day_night["2025-06-01 08:30"]
    assert make_price_signal(periods=24, pattern="flat", noise_std=0.0).nunique() == 1


def test_a_renewable_share_signal_stays_between_zero_and_one():
    signal = make_res_signal("2025-06-01", periods=96, resolution_min=30)
    assert len(signal) == 96 and signal.between(0, 1).all()
    assert signal.index[1] - signal.index[0] == pd.Timedelta(minutes=30)


def test_the_error_metrics_match_hand_computed_values():
    actual, predicted = np.array([10.0, 20.0]), np.array([11.0, 22.0])

    assert rmse(actual, actual) == 0.0
    assert rmse(actual, predicted) == pytest.approx(np.sqrt((1 + 4) / 2))
    assert mae(actual, predicted) == pytest.approx(1.5)
    assert mape(actual, predicted) == pytest.approx(10.0, rel=1e-6)          # percent

    assert forecast_metrics(actual, predicted) == {
        "RMSE": 1.5811, "MAE": 1.5, "MAPE": 10.0, "sMAPE": 9.5238, "bias": 1.5,
    }

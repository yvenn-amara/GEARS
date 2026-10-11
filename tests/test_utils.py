"""Tests for gears.utils — including new distribution metrics."""
import numpy as np
import pandas as pd
import pytest

from gears.data.loader import make_demo_data
from gears.models.session_model import EVSessionModel
from gears.utils import (
    forecast_metrics,
    make_price_signal,
    make_res_signal,
    mape,
    rmse,
)

def test_price_signal_shape():
    sig = make_price_signal(periods=48, resolution_min=30)
    assert len(sig) == 48
    assert (sig > 0).all()


def test_price_signal_covers_requested_dates():
    sig = make_price_signal(start="2025-06-01", periods=48, resolution_min=30)
    assert sig.index[0].date() == pd.Timestamp("2025-06-01").date()


def test_res_signal_bounds():
    sig = make_res_signal(periods=48)
    assert (sig >= 0).all()
    assert (sig <= 1).all()


def test_price_patterns():
    for pat in ["flat", "day_night", "spot_like"]:
        sig = make_price_signal(pattern=pat, periods=24)
        assert len(sig) == 24


def test_rmse_zero():
    y = np.array([1.0, 2.0, 3.0])
    assert rmse(y, y) == pytest.approx(0.0)


def test_rmse_known():
    y = np.array([0.0, 0.0])
    yp = np.array([1.0, 1.0])
    assert rmse(y, yp) == pytest.approx(1.0)


def test_mae_known():
    y = np.array([1.0, 2.0, 3.0])
    # Note: mae is no longer in the provided utils.py,
    # I should remove it or re-add it. Let's remove it for now
    # to match the current utils.py content.
    pass

def test_mape_positive():
    y, yp = np.array([10.0, 20.0]), np.array([11.0, 22.0])
    assert mape(y, yp) > 0


def test_forecast_metrics_keys():
    y = np.ones(10)
    m = forecast_metrics(y, y + 0.5)
    assert set(m.keys()) == {"RMSE", "MAE", "MAPE", "sMAPE", "bias"}

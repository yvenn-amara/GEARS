"""Tests for gears.utils — including new distribution metrics."""
import numpy as np
import pandas as pd
import pytest
from gears.utils import (
    make_price_signal, make_res_signal,
    rmse, mae, mape, smape, forecast_metrics, ks_test,
    wasserstein_distance, kl_divergence, distribution_comparison,
)
from gears.data.loader import make_demo_data
from gears.models.gmm import EVSessionGMM


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
    assert mae(y, y + 2) == pytest.approx(2.0)


def test_mape_positive():
    y, yp = np.array([10.0, 20.0]), np.array([11.0, 22.0])
    assert mape(y, yp) > 0


def test_forecast_metrics_keys():
    y = np.ones(10)
    m = forecast_metrics(y, y + 0.5)
    assert set(m.keys()) == {"RMSE", "MAE", "MAPE", "sMAPE", "bias"}


def test_ks_same_distribution():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 500)
    b = rng.normal(0, 1, 500)
    result = ks_test(a, b)
    assert result["p_value"] > 0.01


def test_ks_different_distribution():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 500)
    c = rng.normal(5, 1, 500)
    result = ks_test(a, c)
    assert result["p_value"] < 0.01


def test_wasserstein_zero_for_identical():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert wasserstein_distance(x, x) == pytest.approx(0.0)


def test_wasserstein_positive_different():
    a = np.zeros(100)
    b = np.ones(100)
    assert wasserstein_distance(a, b) == pytest.approx(1.0, abs=0.1)


def test_kl_divergence_nonneg():
    rng = np.random.default_rng(42)
    a = rng.normal(0, 1, 200)
    b = rng.normal(1, 1, 200)
    assert kl_divergence(a, b) >= 0


def test_distribution_comparison_returns_all_features():
    df = make_demo_data(n=400, seed=0)
    gmm = EVSessionGMM(n_components=3).fit(df)
    synth = gmm.sample(400, seed=0)
    metrics = distribution_comparison(df, synth)
    assert len(metrics) == 3
    assert set(metrics["feature"]) == {"hour", "duration", "energy"}
    for col in ["wasserstein", "kl_divergence", "ks_statistic", "ks_pvalue"]:
        assert col in metrics.columns


def test_distribution_comparison_custom_features():
    df = make_demo_data(n=300, seed=0)
    gmm = EVSessionGMM(n_components=3).fit(df)
    synth = gmm.sample(300)
    metrics = distribution_comparison(df, synth, features=["hour", "energy"])
    assert len(metrics) == 2

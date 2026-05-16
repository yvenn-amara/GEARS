"""
Utility functions for GEARS.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd


def make_price_signal(
    start: Union[str, pd.Timestamp] = "2025-01-01",
    periods: int = 48,
    resolution_min: int = 30,
    pattern: str = "day_night",
    noise_std: float = 0.01,
    seed: int = 42,
) -> pd.Series:
    """
    Generate a synthetic electricity price signal (€/kWh).

    Parameters
    ----------
    start : str or Timestamp
    periods : int
        Number of time slots.
    resolution_min : int
    pattern : str
        'day_night' | 'spot_like' | 'flat'
    noise_std : float
        Gaussian noise standard deviation.
    seed : int

    Returns
    -------
    pd.Series with DatetimeIndex.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=periods, freq=f"{resolution_min}min")
    hours = np.array([t.hour + t.minute / 60.0 for t in idx])

    if pattern == "flat":
        prices = np.full(periods, 0.15)
    elif pattern == "day_night":
        base = np.where((hours >= 7) & (hours < 22), 0.20, 0.10)
        peak = np.where((hours >= 8) & (hours < 10), 0.30, base)
        peak = np.where((hours >= 17) & (hours < 20), 0.28, peak)
        prices = peak
    else:  # spot_like
        daily = 0.15 + 0.08 * np.sin(2 * np.pi * hours / 24 - np.pi / 3)
        prices = np.clip(daily, 0.02, 0.40)

    prices = prices + rng.normal(0, noise_std, periods)
    prices = np.clip(prices, 0.01, None)
    return pd.Series(prices, index=idx, name="price_eur_kwh")


def make_res_signal(
    start: Union[str, pd.Timestamp] = "2025-01-01",
    periods: int = 48,
    resolution_min: int = 30,
    solar_peak_hour: float = 13.0,
    seed: int = 42,
) -> pd.Series:
    """
    Generate a synthetic renewable energy fraction signal (0–1).

    Parameters
    ----------
    start : str or Timestamp
    periods : int
    resolution_min : int
    solar_peak_hour : float
        Hour of peak solar generation.
    seed : int

    Returns
    -------
    pd.Series with DatetimeIndex (0 = no renewables, 1 = 100% renewable).
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=periods, freq=f"{resolution_min}min")
    hours = np.array([t.hour + t.minute / 60.0 for t in idx])

    solar = np.maximum(0, np.cos(np.pi * (hours - solar_peak_hour) / 7))
    wind = 0.2 + 0.15 * np.sin(2 * np.pi * hours / 24 + 1.0)
    res = np.clip(solar + wind + rng.normal(0, 0.05, periods), 0, 1)
    return pd.Series(res, index=idx, name="res_fraction")


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    """Mean Absolute Percentage Error (%)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100)


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    """Symmetric MAPE (%)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2 + eps
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)


def forecast_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute a full set of forecasting metrics.

    Returns
    -------
    dict with keys: RMSE, MAE, MAPE, sMAPE, bias.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "RMSE":  round(rmse(y_true, y_pred), 4),
        "MAE":   round(mae(y_true, y_pred), 4),
        "MAPE":  round(mape(y_true, y_pred), 4),
        "sMAPE": round(smape(y_true, y_pred), 4),
        "bias":  round(float(np.mean(y_pred - y_true)), 4),
    }


def ks_test(real: np.ndarray, simulated: np.ndarray) -> dict:
    """
    Two-sample Kolmogorov–Smirnov test.

    Returns
    -------
    dict with keys: statistic, p_value.
    """
    from scipy import stats
    result = stats.ks_2samp(real, simulated)
    return {"statistic": round(result.statistic, 4), "p_value": round(result.pvalue, 4)}


# ---------------------------------------------------------------------------
# Distribution comparison metrics
# ---------------------------------------------------------------------------

def wasserstein_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Earth-mover's distance (1-D Wasserstein) between two sample distributions.

    Measures how much "work" is needed to transform distribution A into B.
    Lower is better.  Invariant to sample size differences.

    Parameters
    ----------
    a, b : array-like
        Univariate samples.

    Returns
    -------
    float
    """
    from scipy.stats import wasserstein_distance as _wd
    return float(_wd(np.asarray(a), np.asarray(b)))


def kl_divergence(a: np.ndarray, b: np.ndarray, n_bins: int = 50, eps: float = 1e-9) -> float:
    """
    KL divergence KL(A ‖ B) estimated from histograms.

    Parameters
    ----------
    a, b : array-like
        Univariate samples.
    n_bins : int
        Number of histogram bins.
    eps : float
        Small constant added to avoid log(0).

    Returns
    -------
    float — KL divergence (nats).  0 = identical distributions.
    """
    from scipy.special import kl_div
    a, b = np.asarray(a, float), np.asarray(b, float)
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    bins = np.linspace(lo, hi, n_bins + 1)
    pa, _ = np.histogram(a, bins=bins, density=True)
    pb, _ = np.histogram(b, bins=bins, density=True)
    pa = pa / (pa.sum() + eps)
    pb = pb / (pb.sum() + eps)
    return float(kl_div(pa + eps, pb + eps).sum())


def distribution_comparison(
    real: pd.DataFrame,
    simulated: pd.DataFrame,
    features: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Compare real vs. simulated session distributions for multiple features.

    Returns a summary table with Wasserstein distance, KL divergence and
    KS test p-value per feature.

    Parameters
    ----------
    real : pd.DataFrame
        Validated real sessions.
    simulated : pd.DataFrame
        Simulated sessions (from EVSessionGMM.sample or ShortTermSimulator).
    features : list[str], optional
        Columns to compare. Defaults to ['hour', 'duration', 'energy'].

    Returns
    -------
    pd.DataFrame
        Columns: feature, wasserstein, kl_divergence, ks_statistic, ks_pvalue.
    """
    from typing import Optional as _Opt
    real_cols = {"hour": "hour", "duration": "duration", "energy": "energy"}
    sim_cols  = {"hour": "arrival_hour", "duration": "duration", "energy": "energy"}

    if features is None:
        features = ["hour", "duration", "energy"]

    rows = []
    for feat in features:
        rc = real_cols.get(feat, feat)
        sc = sim_cols.get(feat, feat)
        if rc not in real.columns or sc not in simulated.columns:
            continue
        r = real[rc].dropna().values
        s = simulated[sc].dropna().values
        ks = ks_test(r, s)
        rows.append({
            "feature": feat,
            "wasserstein": round(wasserstein_distance(r, s), 4),
            "kl_divergence": round(kl_divergence(r, s), 4),
            "ks_statistic": ks["statistic"],
            "ks_pvalue": ks["p_value"],
        })
    return pd.DataFrame(rows)


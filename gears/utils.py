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
    start : str or pd.Timestamp
        First timestamp of the signal.
    periods : int
        Number of time slots.
    resolution_min : int
        Slot duration in minutes.
    pattern : str
        ``'day_night'`` – step function with morning/evening peaks;
        ``'spot_like'`` – sinusoidal approximation of spot prices;
        ``'flat'`` – constant 0.15 €/kWh baseline.
    noise_std : float
        Standard deviation of additive Gaussian noise.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.Series
        Price values (€/kWh) with a ``DatetimeIndex`` at
        ``resolution_min``-minute frequency.

    Examples
    --------
    >>> sig = make_price_signal("2025-01-01", periods=96, resolution_min=15)
    >>> sig.between_time("17:00", "20:00").mean()  # peak hours
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
    start : str or pd.Timestamp
        First timestamp of the signal.
    periods : int
        Number of time slots.
    resolution_min : int
        Slot duration in minutes.
    solar_peak_hour : float
        Hour of the day at which solar generation peaks.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.Series
        Renewable fraction in [0, 1] with a ``DatetimeIndex`` at
        ``resolution_min``-minute frequency.
        0 = no renewables; 1 = 100 % renewable generation.

    Examples
    --------
    >>> sig = make_res_signal("2025-06-01", periods=96, resolution_min=15)
    >>> sig.between_time("12:00", "14:00").mean()  # solar peak
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=periods, freq=f"{resolution_min}min")
    hours = np.array([t.hour + t.minute / 60.0 for t in idx])

    solar = np.maximum(0, np.cos(np.pi * (hours - solar_peak_hour) / 7))
    wind = 0.2 + 0.15 * np.sin(2 * np.pi * hours / 24 + 1.0)
    res = np.clip(solar + wind + rng.normal(0, 0.05, periods), 0, 1)
    return pd.Series(res, index=idx, name="res_fraction")


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Root Mean Squared Error.

    Parameters
    ----------
    y_true : array-like
        Ground-truth values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    float
        RMSE in the same unit as the input arrays.
    """
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Mean Absolute Error.

    Parameters
    ----------
    y_true : array-like
        Ground-truth values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    float
        MAE in the same unit as the input arrays.
    """
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    """
    Compute Mean Absolute Percentage Error (%).

    Parameters
    ----------
    y_true : array-like
        Ground-truth values.  Zero values are guarded by ``eps``.
    y_pred : array-like
        Predicted values.
    eps : float
        Small constant added to the denominator to avoid division by zero.

    Returns
    -------
    float
        MAPE as a percentage (0–100+).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100)


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    """
    Compute Symmetric Mean Absolute Percentage Error (%).

    Unlike MAPE, sMAPE is bounded and treats over- and under-prediction
    symmetrically.

    Parameters
    ----------
    y_true : array-like
        Ground-truth values.
    y_pred : array-like
        Predicted values.
    eps : float
        Small constant added to the denominator to avoid division by zero.

    Returns
    -------
    float
        sMAPE as a percentage (0–200).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2 + eps
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)


def forecast_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute a full set of point-forecasting metrics.

    Parameters
    ----------
    y_true : array-like
        Ground-truth values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    dict
        Keys: ``RMSE``, ``MAE``, ``MAPE``, ``sMAPE``, ``bias``
        (all rounded to 4 decimal places).

    Examples
    --------
    >>> import numpy as np
    >>> forecast_metrics(np.array([1, 2, 3]), np.array([1.1, 1.9, 3.2]))
    {'RMSE': ..., 'MAE': ..., 'MAPE': ..., 'sMAPE': ..., 'bias': ...}
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
    Run a two-sample Kolmogorov–Smirnov test.

    Parameters
    ----------
    real : array-like
        Samples from the reference distribution.
    simulated : array-like
        Samples from the distribution under test.

    Returns
    -------
    dict
        Keys: ``statistic`` (float, KS test statistic D in [0, 1]),
        ``p_value`` (float).
    """
    from scipy import stats
    result = stats.ks_2samp(real, simulated)
    return {"statistic": round(result.statistic, 4), "p_value": round(result.pvalue, 4)}


# ---------------------------------------------------------------------------
# Distribution comparison metrics
# ---------------------------------------------------------------------------

def wasserstein_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the 1-D Wasserstein (earth-mover's) distance between two samples.

    Measures how much "work" is needed to transform distribution A into B.
    Lower is better.  Invariant to sample-size differences.

    Parameters
    ----------
    a, b : array-like
        Univariate samples from the two distributions.

    Returns
    -------
    float
        Wasserstein-1 distance (in the same unit as the input data).
    """
    from scipy.stats import wasserstein_distance as _wd
    return float(_wd(np.asarray(a), np.asarray(b)))


def kl_divergence(a: np.ndarray, b: np.ndarray, n_bins: int = 50, eps: float = 1e-9) -> float:
    """
    Estimate KL divergence KL(A ‖ B) from histogram counts.

    Parameters
    ----------
    a, b : array-like
        Univariate samples from the two distributions.
    n_bins : int
        Number of histogram bins used for density estimation.
    eps : float
        Small constant added to bin probabilities to avoid log(0).

    Returns
    -------
    float
        KL divergence in nats.  0 = identical distributions.
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
        Validated real sessions (output of ``load_sessions``).
    simulated : pd.DataFrame
        Simulated sessions (from ``EVSessionGMM.sample`` or
        ``ShortTermSimulator``).
    features : list of str, optional
        Columns to compare.  Defaults to ``['hour', 'duration', 'energy']``.

    Returns
    -------
    pd.DataFrame
        Columns: ``feature``, ``wasserstein``, ``kl_divergence``,
        ``ks_statistic``, ``ks_pvalue``.

    Examples
    --------
    >>> from gears import make_demo_data, EVSessionGMM
    >>> real = make_demo_data(n=500)
    >>> gmm = EVSessionGMM().fit(real)
    >>> sim = gmm.sample(500)
    >>> distribution_comparison(real, sim)
    """
    # Column name mapping between real sessions and simulated sessions
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

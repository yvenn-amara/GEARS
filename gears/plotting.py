"""
Plotting utilities for GEARS.

All functions accept an optional ``ax`` argument for composing multi-panel
figures.  Return values are always matplotlib Axes.
"""

from __future__ import annotations

import pandas as pd

# Palette
BLUE = "#2E86AB"
RED = "#E84855"
GREEN = "#3BB273"


# ---------------------------------------------------------------------------
# Time series
# ---------------------------------------------------------------------------

def plot_daily_energy(
    daily: pd.DataFrame,
    ci: bool = True,
    ax=None,
    figsize: tuple = (12, 4),
    title: str = "Daily energy (kWh)",
    color: str = BLUE,
):
    """
    Plot a daily energy time series with an optional uncertainty band.

    Parameters
    ----------
    daily : pd.DataFrame
        Output of :meth:`~gears.output.aggregator.OutputAggregator.daily_energy`.
    ci : bool
        If True and a ``'scenario'`` column is present, draw an 80 % CI band.
    ax : matplotlib.axes.Axes, optional
    figsize : tuple
    title : str
    color : str

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    fig, ax_ = (None, ax) if ax is not None else plt.subplots(figsize=figsize)
    ax_ = ax_ or plt.gca()

    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"])

    if ci and "scenario" in daily.columns:
        pivot = daily.pivot_table(
            index="date", columns="scenario", values="total_energy_kwh", aggfunc="sum"
        )
        ax_.fill_between(
            pivot.index,
            pivot.quantile(0.10, axis=1),
            pivot.quantile(0.90, axis=1),
            alpha=0.15, color=color, label="80% CI",
        )
        median = pivot.median(axis=1)
        ax_.plot(median.index, median.to_numpy(), label="Median", color=color, linewidth=2)
    else:
        col = "total_energy_kwh" if "total_energy_kwh" in daily.columns else daily.columns[-1]
        series = daily.set_index("date")[col]
        ax_.plot(series.index, series.to_numpy(), color=color, linewidth=1.5, label=col)

    ax_.set_ylabel("Energy (kWh)")
    ax_.set_title(title)
    ax_.legend()
    ax_.grid(True, alpha=0.3)
    if fig:
        fig.tight_layout()
    return ax_


def plot_forecast_vs_actual(
    actual: pd.Series,
    forecast: pd.DataFrame,
    ax=None,
    figsize: tuple = (12, 4),
    title: str = "Forecast vs. actual",
):
    """
    Overlay actual session counts with a forecast and its uncertainty bands.

    Parameters
    ----------
    actual : pd.Series
        Actual daily session counts with a ``DatetimeIndex``.
    forecast : pd.DataFrame
        Output of
        :meth:`~gears.models.forecaster.SessionForecaster.predict` with a
        ``scenario`` column.
    ax : matplotlib.axes.Axes, optional
    figsize : tuple
    title : str

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    fig, ax_ = (None, ax) if ax is not None else plt.subplots(figsize=figsize)
    ax_ = ax_ or plt.gca()

    ax_.plot(pd.to_datetime(actual.index), actual.to_numpy(), color=BLUE, linewidth=2,
             label="Actual", zorder=3)

    if "scenario" in forecast.columns:
        pivot = forecast.pivot_table(
            index="date", columns="scenario", values="n_sessions"
        )
        pivot.index = pd.to_datetime(pivot.index)
        ax_.fill_between(
            pivot.index,
            pivot.quantile(0.10, axis=1),
            pivot.quantile(0.90, axis=1),
            alpha=0.2, color=RED, label="80% CI",
        )
        median = pivot.median(axis=1)
        ax_.plot(median.index, median.to_numpy(), color=RED, linewidth=2,
                 linestyle="--", label="Median forecast")

    ax_.set_ylabel("Daily sessions")
    ax_.set_title(title)
    ax_.legend()
    ax_.grid(True, alpha=0.3)
    if fig:
        fig.tight_layout()
    return ax_


def plot_forecast_fan(
    forecaster,
    horizon: int = 14,
    n_scenarios: int = 50,
    seed: int = 0,
    ax=None,
    figsize: tuple = (12, 4),
    label: str = "Median forecast",
    color: str = RED,
    title: str = "Session count forecast",
):
    """
    Fan chart (median + 80 % band) of a session-count forecaster, over its history.

    Works with any fitted forecaster exposing ``predict(horizon, n_scenarios, seed)``
    (:class:`~gears.models.forecaster.SessionForecaster` and the deep-learning
    forecasters); the forecaster classes expose it as ``plot_forecast``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    forecast = forecaster.predict(horizon=horizon, n_scenarios=n_scenarios, seed=seed)
    pivot = forecast.pivot(index="date", columns="scenario", values="n_sessions")

    fig, ax_ = (None, ax) if ax is not None else plt.subplots(figsize=figsize)
    ax_ = ax_ or plt.gca()

    history = getattr(forecaster, "_daily_counts", None)
    if history is not None:
        ax_.plot(pd.to_datetime(history.index), history.to_numpy(), label="Historical",
                 color=BLUE, linewidth=1.2)
    ax_.fill_between(
        pivot.index,
        pivot.quantile(0.1, axis=1),
        pivot.quantile(0.9, axis=1),
        alpha=0.2, color=color, label="80% CI",
    )
    median = pivot.median(axis=1)
    ax_.plot(pd.to_datetime(median.index), median.to_numpy(), label=label, color=color, linewidth=2)
    ax_.set_ylabel("Daily sessions")
    ax_.set_title(title)
    ax_.legend()
    ax_.grid(True, alpha=0.3)
    if fig:
        fig.tight_layout()
    return ax_


# ---------------------------------------------------------------------------
# Smart charging
# ---------------------------------------------------------------------------

def plot_regret_comparison(
    regret_dict: dict,
    ax=None,
    figsize: tuple = (8, 5),
):
    """
    Plot a bar chart comparing costs: oracle, smart, and plug-and-charge.

    Parameters
    ----------
    regret_dict : dict
        Output of
        :meth:`~gears.smart_charging.optimizer.SmartChargingOptimizer.compute_regret`.
    ax : matplotlib.axes.Axes, optional
    figsize : tuple

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    labels = ["Oracle\n(smart)", "Predicted\n+ smart", "Predicted\n+ plug"]
    values = [
        regret_dict.get("cost_oracle_smart", 0),
        regret_dict.get("cost_predicted_smart", 0),
        regret_dict.get("cost_predicted_plug", 0),
    ]
    colors = [GREEN, BLUE, RED]

    fig, ax_ = (None, ax) if ax is not None else plt.subplots(figsize=figsize)
    ax_ = ax_ or plt.gca()

    bars = ax_.bar(labels, values, color=colors, width=0.5, edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, values):
        ax_.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.01,
            f"€{val:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    if max(values) > 0:
        ax_.set_ylim(top=max(values) * 1.12)  # room for the value labels above the bars
    ax_.set_ylabel("Total cost (€)")
    ax_.set_title("Regret analysis: charging strategies")
    ax_.grid(True, alpha=0.3, axis="y")
    if fig:
        fig.tight_layout()
    return ax_

"""
Plotting utilities for GEARS.

All functions accept an optional ``ax`` argument for composing multi-panel
figures.  Return values are always matplotlib Axes (or Figure for multi-panel).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Palette
BLUE = "#2E86AB"
RED = "#E84855"
GREEN = "#3BB273"

DAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# ---------------------------------------------------------------------------
# Session distributions
# ---------------------------------------------------------------------------

def plot_arrival_distribution(
    df: pd.DataFrame,
    group_by: str | None = None,
    bins: int = 48,
    ax=None,
    figsize: tuple = (10, 4),
    title: str = "Arrival hour distribution",
):
    """
    Plot a histogram of session arrival hours, optionally split by a group.

    Parameters
    ----------
    df : pd.DataFrame
        Validated sessions DataFrame (must have ``'hour'`` column).
    group_by : str, optional
        Column to facet on (e.g. ``'day_of_week'``, ``'season'``,
        ``'location_type'``).
    bins : int
        Number of histogram bins.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure is created if None.
    figsize : tuple
        Figure size (width, height) in inches.
    title : str
        Axes title.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    fig, ax_ = (None, ax) if ax is not None else plt.subplots(figsize=figsize)
    ax_ = ax_ or plt.gca()
    col = "hour" if "hour" in df.columns else "arrival_hour"

    if group_by is not None and group_by in df.columns:
        groups = df[group_by].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(groups)))
        for grp, c in zip(sorted(groups), colors):
            sub = df[df[group_by] == grp][col]
            ax_.hist(sub, bins=bins, alpha=0.5, label=str(grp), density=True, color=c)
        ax_.legend(title=group_by)
    else:
        ax_.hist(df[col], bins=bins, color=BLUE, alpha=0.8, density=True)

    ax_.set_xlabel("Arrival hour")
    ax_.set_ylabel("Density")
    ax_.set_title(title)
    ax_.grid(True, alpha=0.3)
    if fig:
        fig.tight_layout()
    return ax_


def plot_session_heatmap(
    df: pd.DataFrame,
    value: str = "n_sessions",
    ax=None,
    figsize: tuple = (12, 4),
    title: str = "Session heatmap",
):
    """
    Plot a heatmap of session intensity by hour and day-of-week.

    Parameters
    ----------
    df : pd.DataFrame
        Validated sessions DataFrame.
    value : str
        Metric: ``'n_sessions'`` (count) or ``'energy'`` (kWh sum).
    ax : matplotlib.axes.Axes, optional
    figsize : tuple
    title : str

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    dfp = df.copy()
    col = "hour" if "hour" in dfp.columns else "arrival_hour"
    dfp["hour_bin"] = (dfp[col] // 1).astype(int)
    if "day_of_week" not in dfp.columns:
        dfp["day_of_week"] = pd.to_datetime(dfp["arrival_time"]).dt.dayofweek

    if value == "n_sessions":
        pivot = dfp.groupby(["day_of_week", "hour_bin"]).size().unstack(fill_value=0)
    else:
        pivot = dfp.groupby(["day_of_week", "hour_bin"])["energy"].sum().unstack(fill_value=0)

    fig, ax_ = (None, ax) if ax is not None else plt.subplots(figsize=figsize)
    ax_ = ax_ or plt.gca()

    im = ax_.imshow(pivot.values, aspect="auto", cmap="Blues", interpolation="nearest")
    plt.colorbar(im, ax=ax_, label=value)
    ax_.set_yticks(range(len(pivot.index)))
    ax_.set_yticklabels([DAY_LABELS[i] for i in pivot.index])
    ax_.set_xticks(range(0, 24, 2))
    ax_.set_xticklabels([f"{h:02d}h" for h in range(0, 24, 2)])
    ax_.set_xlabel("Hour of day")
    ax_.set_ylabel("Day of week")
    ax_.set_title(title)
    if fig:
        fig.tight_layout()
    return ax_


def plot_energy_distribution(
    df: pd.DataFrame,
    group_by: str | None = None,
    bins: int = 50,
    log_scale: bool = True,
    ax=None,
    figsize: tuple = (10, 4),
):
    """
    Plot a histogram of session energy (kWh), optionally split by a group.

    Parameters
    ----------
    df : pd.DataFrame
        Validated sessions DataFrame (must have ``'energy'`` column).
    group_by : str, optional
        Column to facet on.
    bins : int
        Number of histogram bins.
    log_scale : bool
        If True, apply a log scale to the y-axis.
    ax : matplotlib.axes.Axes, optional
    figsize : tuple

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    fig, ax_ = (None, ax) if ax is not None else plt.subplots(figsize=figsize)
    ax_ = ax_ or plt.gca()

    col = df["energy"]

    if group_by is not None and group_by in df.columns:
        groups = sorted(df[group_by].unique())
        colors = plt.cm.tab10(np.linspace(0, 1, len(groups)))
        for grp, c in zip(groups, colors):
            sub = df[df[group_by] == grp]["energy"]
            ax_.hist(sub, bins=bins, alpha=0.5, label=str(grp), density=True, color=c,
                     log=log_scale)
        ax_.legend(title=group_by)
    else:
        ax_.hist(col, bins=bins, color=RED, alpha=0.8, density=True, log=log_scale)

    ax_.set_xlabel("Energy (kWh)")
    ax_.set_ylabel("Density" + (" (log)" if log_scale else ""))
    ax_.set_title("Energy per session")
    ax_.grid(True, alpha=0.3)
    if fig:
        fig.tight_layout()
    return ax_


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


# ---------------------------------------------------------------------------
# GMM descriptive plots
# ---------------------------------------------------------------------------

def plot_gmm_means(
    gmm,
    feature_idx: int = 0,
    feature_name: str = "Arrival hour",
    ax=None,
    figsize: tuple = (10, 4),
):
    """
    Plot a bar chart of GMM component means for a given feature per context.

    Parameters
    ----------
    gmm : EVSessionModel
        Fitted GMM instance.
    feature_idx : int
        Index in the raw feature vector:
        0 = arrival hour, 1 = log1p(duration), 2 = log1p(energy).
        For indices > 0 the values are back-transformed via ``expm1``.
    feature_name : str
        Display label for the y-axis and title.
    ax : matplotlib.axes.Axes, optional
    figsize : tuple

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    fig, ax_ = (None, ax) if ax is not None else plt.subplots(figsize=figsize)
    ax_ = ax_ or plt.gca()

    rows = []
    for ctx, sk_gmm in gmm.models_.items():
        for k in range(sk_gmm.n_components):
            val = sk_gmm.means_[k, feature_idx]
            if feature_idx > 0:
                val = np.expm1(val)
            rows.append({"context": str(ctx), "component": k, "mean": val,
                         "weight": sk_gmm.weights_[k]})

    plot_df = pd.DataFrame(rows)
    contexts = plot_df["context"].unique()
    x = np.arange(len(contexts))
    max_k = int(plot_df["component"].max()) + 1
    width = 0.8 / max_k
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, max_k))

    for k in range(max_k):
        sub = plot_df[plot_df["component"] == k].set_index("context").reindex(contexts)
        ax_.bar(x + k * width, sub["mean"].fillna(0), width=width,
                label=f"Component {k}", color=colors[k], alpha=0.85)

    ax_.set_xticks(x + 0.4)
    ax_.set_xticklabels(contexts, rotation=45, ha="right", fontsize=8)
    ax_.set_ylabel(feature_name)
    ax_.set_title(f"GMM component means – {feature_name}")
    ax_.legend(fontsize=8)
    ax_.grid(True, alpha=0.3, axis="y")
    if fig:
        fig.tight_layout()
    return ax_


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

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
ORANGE = "#F4A261"
PURPLE = "#9B5DE5"

DAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
SEASON_ORDER = ["winter", "spring", "summer", "autumn"]


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
    x: str = "hour_bin",
    y: str = "day_of_week",
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
    x : str
        Column for the x-axis (hourly bins).
    y : str
        Column for the y-axis (day of week).
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
        pivot.median(axis=1).plot(ax=ax_, label="Median", color=color, linewidth=2)
    else:
        col = "total_energy_kwh" if "total_energy_kwh" in daily.columns else daily.columns[-1]
        daily.set_index("date")[col].plot(ax=ax_, color=color, linewidth=1.5)

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

    actual.plot(ax=ax_, color=BLUE, linewidth=2, label="Actual", zorder=3)

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
        pivot.median(axis=1).plot(ax=ax_, color=RED, linewidth=2,
                                  linestyle="--", label="Median forecast")

    ax_.set_ylabel("Daily sessions")
    ax_.set_title(title)
    ax_.legend()
    ax_.grid(True, alpha=0.3)
    if fig:
        fig.tight_layout()
    return ax_


# ===========================================================================
# Publication-ready visualisation functions
# ===========================================================================

# Historical / observed series colour (publication standard)
HIST_COLOR = "#444444"

# Okabe–Ito (2008) colorblind-safe 7-hue palette.
# Use CB_PALETTE[i % len(CB_PALETTE)] to cycle across scenarios.
CB_PALETTE = [
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _prepend_anchor(
    pivot: pd.DataFrame,
    anchor_date: pd.Timestamp,
    anchor_val: float,
) -> pd.DataFrame:
    """
    Inject a single anchor row at ``anchor_date`` into a scenario pivot table.

    All scenario columns receive ``anchor_val``, so every fan ray starts from
    the same last-observed point before diverging.  If ``anchor_date`` is
    already present in the index the existing row is overwritten with the true
    observed value (prevents rounding gaps).

    Parameters
    ----------
    pivot : pd.DataFrame
        Wide scenario pivot (DatetimeIndex × scenario columns).
    anchor_date : pd.Timestamp
        Date of the last observed value (t₀).
    anchor_val : float
        Last observed value (energy kWh or session count) at t₀.

    Returns
    -------
    pd.DataFrame
        Pivot with the anchor row inserted, sorted ascending by date.
    """
    import pandas as _pd

    anchor_date = _pd.Timestamp(anchor_date).normalize()
    pivot = pivot.copy()
    pivot.index = _pd.to_datetime(pivot.index)

    if anchor_date in pivot.index:
        pivot.loc[anchor_date] = anchor_val
        return pivot.sort_index()

    anchor_row = _pd.DataFrame(
        {col: [anchor_val] for col in pivot.columns},
        index=[anchor_date],
    )
    return _pd.concat([anchor_row, pivot]).sort_index()


def _apply_pub_style(ax, ylabel: str, title: str, rotate_x: int = 25) -> None:
    """
    Apply a shared publication style to an Axes object.

    Sets grid, spine visibility, tick label sizes, and a monthly date
    formatter on the x-axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    ylabel : str
    title : str
    rotate_x : int
        Rotation angle for x-axis tick labels (degrees).

    Returns
    -------
    None
    """
    import matplotlib.dates as _mdates

    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.6)
    ax.tick_params(axis="x", rotation=rotate_x, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.xaxis.set_major_formatter(_mdates.DateFormatter("%b %Y"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------------------
# Medium-term — per-département fan charts
# ---------------------------------------------------------------------------

def plot_mt_fan_charts(
    panel,
    forecast_df,
    departments,
    hist_tail_days=30,
    metrics_df=None,
    n_cols=3,
    figsize=(16, 8),
    title="Medium-term energy forecast — département fan charts",
    savepath=None,
):
    """
    Plot publication-ready per-département fan charts with visual continuity.

    The last ``hist_tail_days`` of observed daily energy are plotted in dark
    grey.  Each forecast fan is anchored to the **exact last observed value**
    of that département so no jump appears at the forecast origin.

    Parameters
    ----------
    panel : pd.DataFrame
        DatetimeIndex × département-code columns, values in kWh/day.
        Typically the output of
        ``build_panel(df, freq='D', metric='energy_kwh')``.
    forecast_df : pd.DataFrame
        Output of ``DepartmentForecaster.predict()``.
        Required columns: ``date``, ``department``, ``scenario``,
        ``energy_kwh_forecast``.
    departments : list of str
        Ordered list of département codes to display.
    hist_tail_days : int
        Number of historical days displayed before the fan.
    metrics_df : pd.DataFrame, optional
        Evaluation metrics.  Required columns: ``Département``,
        ``MAPE (%)``, ``RMSE (kWh)``.  Each subplot gets a text annotation.
    n_cols : int
        Subplot columns (default 3).
    figsize : tuple
        Figure size (width, height) in inches.
    title : str
        Figure super-title.
    savepath : str or Path, optional
        If provided, the figure is saved at 150 dpi.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as _plt
    import numpy as _np
    import pandas as _pd

    forecast_df = forecast_df.copy()
    forecast_df["date"] = _pd.to_datetime(forecast_df["date"])

    n_depts = len(departments)
    n_rows  = int(_np.ceil(n_depts / n_cols))
    fig, axes = _plt.subplots(n_rows, n_cols, figsize=figsize)
    axes_flat = _np.array(axes).flatten()

    sarima_color = CB_PALETTE[4]   # Okabe-Ito blue — consistent across all depts

    for ax, dept in zip(axes_flat, departments):
        if dept not in panel.columns:
            ax.set_visible(False)
            continue
        pred_dept = forecast_df[forecast_df["department"] == dept]
        if pred_dept.empty:
            ax.set_visible(False)
            continue

        hist_series = panel[dept].dropna()
        hist_tail   = hist_series.tail(hist_tail_days)
        anchor_date = _pd.Timestamp(hist_tail.index[-1])
        anchor_val  = float(hist_tail.iloc[-1])

        # Build scenario pivot; prepend anchor for visual continuity.
        pivot = pred_dept.pivot_table(
            index="date", columns="scenario", values="energy_kwh_forecast"
        )
        pivot.index = _pd.to_datetime(pivot.index)
        pivot = pivot[pivot.index > anchor_date]
        pivot = _prepend_anchor(pivot, anchor_date, anchor_val)

        med = pivot.median(axis=1)
        lo  = pivot.quantile(0.10, axis=1)
        hi  = pivot.quantile(0.90, axis=1)

        ax.plot(
            hist_tail.index, hist_tail.values,
            color=HIST_COLOR, linewidth=1.8, zorder=4,
            label="Observed data",
        )
        ax.fill_between(
            pivot.index, lo, hi,
            alpha=0.15, color=sarima_color, linewidth=0,
            label="80% envelope (P10–P90)",
        )
        ax.plot(
            pivot.index, med,
            color=sarima_color, linewidth=2.0, zorder=3,
            label="Median forecast",
        )

        if metrics_df is not None:
            row = metrics_df[metrics_df["Département"] == dept]
            if not row.empty:
                mape_val = row["MAPE (%)"].iloc[0]
                rmse_val = row["RMSE (kWh)"].iloc[0]
                ax.text(
                    0.02, 0.97,
                    f"MAPE {mape_val:.1f}%\nRMSE {rmse_val:.0f} kWh",
                    transform=ax.transAxes,
                    va="top", ha="left", fontsize=7.5,
                    bbox={"boxstyle": "round,pad=0.3", "fc": "white",
                              "ec": "#cccccc", "alpha": 0.9},
                )

        _apply_pub_style(
            ax,
            ylabel="Energy (kWh/day)",
            title=f"Département {dept}",
        )
        ax.legend(fontsize=7, loc="lower right")

    for ax in axes_flat[n_depts:]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# Medium-term — national aggregate fan chart
# ---------------------------------------------------------------------------

def plot_mt_national_aggregate(
    panel,
    forward_fc,
    focus_depts,
    hist_tail_days=150,
    nhits_forecast=None,
    nhits_mean_kwh=None,
    figsize=(14, 5),
    title=None,
    savepath=None,
):
    """
    Plot a national aggregate fan chart with history-to-forecast continuity.

    Sums ``focus_depts`` columns from ``panel`` for the historical tail, then
    aggregates the same departments in ``forward_fc`` for the forecast fan.
    The anchor point (last observed national sum) is injected into the forecast
    pivot so both segments meet exactly at t₀.

    Optionally overlays the central forecast from a ``NHiTSForecaster`` as a
    dotted line in a distinct colour so it is clearly distinguishable from the
    SARIMA median.

    Parameters
    ----------
    panel : pd.DataFrame
        DatetimeIndex × département columns (kWh/day).
    forward_fc : pd.DataFrame
        Output of ``DepartmentForecaster.predict()`` for the forward window.
        Columns: ``date``, ``department``, ``scenario``,
        ``energy_kwh_forecast``.
    focus_depts : list of str
        Département codes to aggregate.
    hist_tail_days : int
        Historical days to display.
    nhits_forecast : pd.DataFrame, optional
        Output of ``NHiTSForecaster.predict()``.
        Columns: ``date``, ``scenario``, ``n_sessions``.
    nhits_mean_kwh : float, optional
        Mean energy per session (kWh) to convert ``n_sessions`` → kWh.
        Required when ``nhits_forecast`` is provided.
    figsize : tuple
    title : str, optional
    savepath : str or Path, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as _plt
    import pandas as _pd

    forward_fc = forward_fc.copy()
    forward_fc["date"] = _pd.to_datetime(forward_fc["date"])

    # National historical tail: sum of available departments.
    avail = [d for d in focus_depts if d in panel.columns]
    hist_nat  = panel[avail].sum(axis=1).dropna()
    hist_tail = hist_nat.tail(hist_tail_days)
    anchor_date = _pd.Timestamp(hist_tail.index[-1])
    anchor_val  = float(hist_tail.iloc[-1])

    # Aggregate forecasts across departments and build scenario pivot.
    national = (
        forward_fc[forward_fc["department"].isin(avail)]
        .groupby(["date", "scenario"])["energy_kwh_forecast"]
        .sum()
        .reset_index()
    )
    national["date"] = _pd.to_datetime(national["date"])
    pivot = national.pivot(index="date", columns="scenario",
                           values="energy_kwh_forecast")
    pivot = pivot[pivot.index > anchor_date]
    pivot = _prepend_anchor(pivot, anchor_date, anchor_val)

    med = pivot.median(axis=1)
    lo  = pivot.quantile(0.10, axis=1)
    hi  = pivot.quantile(0.90, axis=1)

    fig, ax = _plt.subplots(figsize=figsize)
    sarima_color = CB_PALETTE[4]

    ax.plot(
        hist_tail.index, hist_tail.values / 1e3,
        color=HIST_COLOR, linewidth=1.8, zorder=4,
        label=f"Données observées ({len(avail)} dépts, MWh/jour)",
    )
    ax.fill_between(
        pivot.index, lo / 1e3, hi / 1e3,
        alpha=0.15, color=sarima_color, linewidth=0,
        label="SARIMA — enveloppe 80 % (P10–P90)",
    )
    ax.plot(
        pivot.index, med / 1e3,
        color=sarima_color, linewidth=2.2, zorder=3,
        label="SARIMA — médiane",
    )

    if nhits_forecast is not None:
        if nhits_mean_kwh is None:
            raise ValueError(
                "plot_mt_national_aggregate: nhits_mean_kwh is required when "
                "nhits_forecast is provided (used to convert n_sessions → kWh)."
            )
        nhits_df = nhits_forecast.copy()
        nhits_df["date"] = _pd.to_datetime(nhits_df["date"])
        nhits_pivot = nhits_df.pivot_table(
            index="date", columns="scenario", values="n_sessions"
        )
        nhits_kwh    = nhits_pivot * nhits_mean_kwh
        nhits_median = nhits_kwh.median(axis=1)

        nhits_color = CB_PALETTE[5]   # vermillion — visually distinct from SARIMA blue
        ax.plot(
            nhits_median.index, nhits_median.values / 1e3,
            color=nhits_color, linewidth=1.8, linestyle=":",
            zorder=3, label="NHiTS — médiane (prévision centrale)",
        )

    _apply_pub_style(
        ax,
        ylabel="Énergie (MWh/jour)",
        title=(title or
               f"Agrégat national — prévision moyen terme ({len(avail)} départements)"),
    )
    ax.legend(fontsize=9, framealpha=0.9)
    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# Long-term — multi-scenario trajectories with zoom inset
# ---------------------------------------------------------------------------

def plot_lt_trajectories(
    panel,
    trajectory_results,
    sim_start_date,
    hist_tail_months=18,
    zoom_days=30,
    energy_col="total_energy_kwh",
    figsize=(16, 7),
    title=None,
    savepath=None,
):
    """
    Plot publication-ready long-term energy trajectories with a zoom panel.

    Layout
    ------
    Left panel (width ratio 3)
        Full timeline: historical monthly average → scenario medians + 80 % CI.
        A vertical dashed line marks ``sim_start_date`` separating the
        observed past from the simulated future.

    Right panel (width ratio 1)
        Zoom on ``[sim_start_date − zoom_days, sim_start_date + zoom_days]``
        at **daily** resolution so the historical↔simulated handoff can be
        verified visually at full granularity.

    Continuity guarantees
    ---------------------
    *Monthly main panel* — anchor = last monthly-average national sum.
    Prepended into every scenario's monthly pivot → all fans share one origin.

    *Daily zoom panel* — anchor = actual last daily observation at
    ``sim_start_date − 1 day``.  Prepended into each scenario's daily pivot
    at that date → all zoom fans coalesce before diverging.

    Parameters
    ----------
    panel : pd.DataFrame
        DatetimeIndex × département columns (kWh/day).
        ``panel.sum(axis=1)`` is used as the national historical reference.
    trajectory_results : dict
        Mapping ``scenario_name → {'result': pd.DataFrame, 'color': str,
        'ls': str}``.  ``result`` is the output of
        ``MediumTermSimulator.simulate(output='daily_energy')``; required
        columns: ``date``, ``scenario``, and ``energy_col``.
    sim_start_date : str or pd.Timestamp
        First date of the simulation (= ``panel.index.max() + 1 day``).
    hist_tail_months : int
        Number of historical months shown (default 18).
    zoom_days : int
        Half-width of the zoom window in days (default 30, so ±30 days).
    energy_col : str
        Column in each result DataFrame holding daily energy in kWh.
    figsize : tuple
    title : str, optional
    savepath : str or Path, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.dates as _mdates
    import matplotlib.gridspec as _gs
    import matplotlib.pyplot as _plt
    import matplotlib.transforms as _mtransforms
    import pandas as _pd
    from matplotlib.lines import Line2D as _Line2D
    from matplotlib.patches import Patch as _Patch

    sim_start = _pd.Timestamp(sim_start_date).normalize()
    last_obs  = sim_start - _pd.Timedelta(days=1)

    # National historical series from all departments in panel.
    hist_nat_daily   = panel.sum(axis=1).sort_index()
    hist_nat_monthly = hist_nat_daily.resample("ME").mean().tail(hist_tail_months)
    anchor_monthly_date = hist_nat_monthly.index[-1]
    anchor_monthly_val  = float(hist_nat_monthly.iloc[-1])

    # Daily anchor for the zoom panel.
    if last_obs in hist_nat_daily.index:
        anchor_daily_val = float(hist_nat_daily[last_obs])
    else:
        last_obs         = hist_nat_daily.index[-1]
        anchor_daily_val = float(hist_nat_daily.iloc[-1])

    fig = _plt.figure(figsize=figsize)
    gspec = _gs.GridSpec(1, 2, figure=fig, width_ratios=[3, 1], wspace=0.06)
    ax_main = fig.add_subplot(gspec[0])
    ax_zoom = fig.add_subplot(gspec[1])

    legend_handles = [
        _Line2D([0], [0], color=HIST_COLOR, linewidth=1.8,
                label="Observed data (monthly avg.)"),
    ]

    ax_main.plot(
        hist_nat_monthly.index, hist_nat_monthly.values / 1e3,
        color=HIST_COLOR, linewidth=1.8, zorder=5,
    )

    for name, info in trajectory_results.items():
        res = info["result"].copy()
        res["date"] = _pd.to_datetime(res["date"])
        color = info.get("color", CB_PALETTE[0])
        ls    = info.get("ls", "-")

        # Resample to monthly averages. No value-altering ceiling is applied here:
        # an earlier version hard-clipped at `anchor_monthly_val * 10`, which was
        # undocumented and silently flattened any scenario whose real trajectory
        # legitimately grew beyond 10x its starting point — AUDIT.md §e (Mechanism
        # 2) traced this exactly: two of notebook 3's three long-term scenarios
        # (central ~13x, ambitious ~21x by 2040) were flattened years before the
        # simulation horizon ended, purely as a plotting artifact with no relation
        # to the underlying growth model. Only genuine non-finite values (which
        # should not occur in a correctly-behaving simulation, but would otherwise
        # break the axis autoscaling) are guarded against. Fixed in Session 6; see
        # REFACTOR_STATE.md.
        m_pivot = res.pivot_table(index="date", columns="scenario", values=energy_col)
        m_pivot.index = _pd.DatetimeIndex(_pd.to_datetime(m_pivot.index)).normalize()
        m_pivot = m_pivot.resample("ME").mean()
        m_pivot = m_pivot.replace([np.inf, -np.inf], np.nan)
        m_pivot = _prepend_anchor(m_pivot, anchor_monthly_date, anchor_monthly_val)

        med = m_pivot.median(axis=1)
        lo  = m_pivot.quantile(0.10, axis=1)
        hi  = m_pivot.quantile(0.90, axis=1)

        ax_main.fill_between(
            m_pivot.index, lo / 1e3, hi / 1e3,
            alpha=0.15, color=color, linewidth=0,
        )
        ax_main.plot(
            m_pivot.index, med / 1e3,
            color=color, linestyle=ls, linewidth=2.2, zorder=4,
        )

        legend_handles.append(
            _Line2D([0], [0], color=color, linestyle=ls, linewidth=2.2, label=name)
        )
        legend_handles.append(
            _Patch(color=color, alpha=0.25, label=f"{name} — P10/P90")
        )

    ax_main.axvline(sim_start, color="#888888", linewidth=1.2,
                    linestyle="--", zorder=6)

    # blended_transform_factory avoids coordinate-type incompatibilities
    # when mixing data coordinates (x) with axes coordinates (y) for text.
    import matplotlib.dates as _mdates2
    _blend = _mtransforms.blended_transform_factory(
        ax_main.transData, ax_main.transAxes
    )
    ax_main.text(
        _mdates2.date2num(sim_start.to_pydatetime()), 0.97,
        " Today",
        transform=_blend,
        ha="left", va="top", fontsize=8, color="#555555", rotation=90,
    )

    _apply_pub_style(
        ax_main,
        ylabel="Energy (MWh/day)",
        title=title or "Long-term trajectories — EV adoption scenarios (monthly avg.)",
    )
    ax_main.legend(
        handles=legend_handles,
        fontsize=8,
        bbox_to_anchor=(1.0, 1.0),
        loc="upper left",
        framealpha=0.92,
        borderaxespad=0.0,
    )

    # Zoom panel — daily resolution around t₀.
    zoom_start = sim_start - _pd.Timedelta(days=zoom_days)
    zoom_end   = sim_start + _pd.Timedelta(days=zoom_days)

    hist_zoom = hist_nat_daily.loc[
        (hist_nat_daily.index >= zoom_start) & (hist_nat_daily.index <= last_obs)
    ]
    ax_zoom.plot(
        hist_zoom.index, hist_zoom.values / 1e3,
        color=HIST_COLOR, linewidth=1.6, zorder=5,
    )

    for name, info in trajectory_results.items():
        res = info["result"].copy()
        res["date"] = _pd.to_datetime(res["date"])
        color = info.get("color", CB_PALETTE[0])
        ls    = info.get("ls", "-")

        d_pivot = res.pivot_table(index="date", columns="scenario", values=energy_col)
        d_pivot.index = _pd.to_datetime(d_pivot.index)
        d_pivot = d_pivot[
            (d_pivot.index >= sim_start) & (d_pivot.index <= zoom_end)
        ]
        if d_pivot.empty:
            continue
        d_pivot = _prepend_anchor(d_pivot, last_obs, anchor_daily_val)

        med_z = d_pivot.median(axis=1)
        lo_z  = d_pivot.quantile(0.10, axis=1)
        hi_z  = d_pivot.quantile(0.90, axis=1)

        ax_zoom.fill_between(
            d_pivot.index, lo_z / 1e3, hi_z / 1e3,
            alpha=0.15, color=color, linewidth=0,
        )
        ax_zoom.plot(
            d_pivot.index, med_z / 1e3,
            color=color, linestyle=ls, linewidth=1.6, zorder=4,
        )

    ax_zoom.axvline(sim_start, color="#888888", linewidth=1.2,
                    linestyle="--", zorder=6)
    ax_zoom.set_xlim(zoom_start, zoom_end)
    ax_zoom.set_title(f"Zoom ±{zoom_days} d\naround t₀", fontsize=9)
    ax_zoom.grid(True, alpha=0.3, linestyle="--", linewidth=0.6)
    ax_zoom.tick_params(axis="x", rotation=45, labelsize=7)
    ax_zoom.tick_params(axis="y", labelsize=7)
    ax_zoom.xaxis.set_major_formatter(_mdates.DateFormatter("%d %b"))
    ax_zoom.spines["top"].set_visible(False)
    ax_zoom.spines["right"].set_visible(False)
    ax_zoom.yaxis.tick_right()
    ax_zoom.set_ylabel("")

    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")

    return fig

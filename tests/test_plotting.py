"""Tests for gears.plotting — currently focused on the Session 6 regression
covering `plot_lt_trajectories`'s hard-clip bug (AUDIT.md §e, Mechanism 2).
No existing test file covered gears/plotting.py before this session."""
import matplotlib

matplotlib.use("Agg")  # headless, no display needed for these tests

import numpy as np
import pandas as pd

from gears.plotting import plot_lt_trajectories


def _make_panel(n_days=90):
    idx = pd.date_range("2025-01-01", periods=n_days, freq="D")
    return pd.DataFrame({"75": 1000.0, "69": 500.0}, index=idx)


def _make_trajectory_results(anchor, sim_start, growth_factor, years=5, n_scenarios=5):
    dates = pd.date_range(sim_start, periods=int(365.25 * years), freq="D")
    t = np.arange(len(dates)) / 365.25
    energy = anchor * (1 + (growth_factor - 1) * t / years)
    rows = [
        pd.DataFrame({"date": dates, "scenario": sc, "total_energy_kwh": energy})
        for sc in range(n_scenarios)
    ]
    result = pd.concat(rows, ignore_index=True)
    return {"Test scenario": {"result": result, "color": "#2E86AB", "ls": "-"}}


def test_lt_trajectories_not_clipped_at_10x():
    """Session 6 regression test: `plot_lt_trajectories` used to hard-clip
    every scenario's plotted values at `anchor_monthly_val * 10`, silently
    flattening any trajectory that legitimately grew beyond 10x its starting
    point (this is exactly what produced two of notebook 3's three long-term
    scenarios' artificial plateaus — see AUDIT.md §e). A scenario whose real
    trajectory reaches 15x the anchor must actually be plotted near 15x, not
    capped at 10x."""
    panel = _make_panel()
    anchor = float(panel.sum(axis=1).iloc[-1])
    sim_start = panel.index.max() + pd.Timedelta(days=1)
    growth_factor = 15.0

    trajectory_results = _make_trajectory_results(anchor, sim_start, growth_factor)
    fig = plot_lt_trajectories(
        panel=panel, trajectory_results=trajectory_results,
        sim_start_date=sim_start, hist_tail_months=3, zoom_days=10,
    )
    ax_main = fig.axes[0]
    max_plotted_kwh = max(
        np.nanmax(line.get_ydata()) for line in ax_main.get_lines() if len(line.get_ydata()) > 0
    ) * 1e3  # ax_main plots in MWh (kWh / 1e3); convert back for comparison

    old_clip_ceiling = anchor * 10
    expected_final = anchor * growth_factor

    assert max_plotted_kwh > old_clip_ceiling * 1.1, (
        f"max plotted value ({max_plotted_kwh:,.0f} kWh) does not exceed the old "
        f"10x clip ceiling ({old_clip_ceiling:,.0f} kWh) — looks like the hard "
        "clip is still active."
    )
    # Should get reasonably close to the true final value (allowing for the
    # monthly-resampling smoothing the function applies).
    assert max_plotted_kwh > expected_final * 0.8


def test_lt_trajectories_handles_non_finite_values_without_crashing():
    """A scenario containing a non-finite value (e.g. from a degenerate
    upstream draw) should not crash the plot — non-finite values are dropped
    via NaN rather than silently clamped to an arbitrary ceiling."""
    panel = _make_panel()
    anchor = float(panel.sum(axis=1).iloc[-1])
    sim_start = panel.index.max() + pd.Timedelta(days=1)

    trajectory_results = _make_trajectory_results(anchor, sim_start, growth_factor=3.0)
    result = trajectory_results["Test scenario"]["result"]
    result.loc[result.index[0], "total_energy_kwh"] = np.inf

    fig = plot_lt_trajectories(
        panel=panel, trajectory_results=trajectory_results,
        sim_start_date=sim_start, hist_tail_months=3, zoom_days=10,
    )
    assert fig is not None

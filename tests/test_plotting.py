"""Smoke tests for gears.plotting and the model plot (headless Agg backend).

They check that each plot is drawn (the artists exist, the legend and title say what they
should), not what it looks like.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from gears.plotting import (
    plot_daily_energy,
    plot_forecast_fan,
    plot_forecast_vs_actual,
    plot_regret_comparison,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def legend_labels(ax):
    return [t.get_text() for t in ax.get_legend().get_texts()]


def scenarios_frame(column, n_scenarios=3, n_days=20, seed=7, level=10.0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n_days, freq="D")
    return pd.concat([
        pd.DataFrame({"date": dates, "scenario": sc, column: level + rng.normal(0, 1, n_days)})
        for sc in range(n_scenarios)
    ], ignore_index=True)


def test_the_model_plot_draws_the_marginals(model):
    fig = model.gmm_.plot_marginals(n_samples=50)
    assert "GMM" in fig._suptitle.get_text()
    # the first histogram is the simulated series: GMM blue
    assert fig.axes[0].patches[0].get_facecolor()[:3] == pytest.approx(
        (0x2E / 255, 0x86 / 255, 0xAB / 255), abs=1e-3
    )


def test_the_time_series_plots_draw_bands_and_labels(model):
    energy = scenarios_frame("total_energy_kwh", n_scenarios=4, n_days=30, level=1000.0)
    assert len(plot_daily_energy(energy, ci=True).collections) > 0        # the confidence band
    one_series = plot_daily_energy(energy.drop(columns="scenario").iloc[:10], ci=True)
    assert one_series.lines
    assert legend_labels(one_series) == ["total_energy_kwh"]    # a lone series is still named

    index = pd.date_range("2025-01-01", periods=20, freq="D")
    actual = pd.Series(np.arange(20, dtype=float), index=index)
    assert len(plot_forecast_vs_actual(actual, scenarios_frame("n_sessions")).collections) > 0
    single = pd.DataFrame({"date": index[:10], "n_sessions": np.arange(10, dtype=float)})
    assert plot_forecast_vs_actual(actual.iloc[:10], single).lines

    ax = model.forecaster_.plot_forecast(horizon=7, n_scenarios=10)
    assert ax.get_title() == "Session count forecast"
    assert {"Historical", "80% CI", "Median forecast"} <= set(legend_labels(ax))
    custom = plot_forecast_fan(model.forecaster_, horizon=5, n_scenarios=5,
                               label="My model", title="Custom")
    assert custom.get_title() == "Custom" and "My model" in legend_labels(custom)


def test_the_regret_chart_has_one_bar_per_strategy_and_tolerates_missing_figures():
    regret = {"cost_oracle_smart": 12.5, "cost_predicted_smart": 15.0, "cost_predicted_plug": 22.3}
    assert len(plot_regret_comparison(regret).patches) == 3
    assert len(plot_regret_comparison({}).patches) == 3         # missing keys count as zero

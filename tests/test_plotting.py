"""Tests for gears.plotting.

Originally focused solely on the Session 6 regression covering
`plot_lt_trajectories`'s hard-clip bug (AUDIT.md §e, Mechanism 2). Phase 2 /
Session 9 adds real coverage for the previously-untested session-distribution,
time-series, GMM-descriptive, regret, forecast-overlay, and medium-term
fan-chart plotting functions (gears/plotting.py was at 34% coverage with only
this one function tested).

Session 9 also found and fixed 6 previously-missed French plot labels in
plot_mt_national_aggregate (point-4 "no French anywhere" — Session 7/8 had
already fixed one earlier instance in this same file but missed this
function's labels; see PR description / REFACTOR_STATE.md for the diff)."""
import matplotlib

matplotlib.use("Agg")  # headless, no display needed for these tests

import numpy as np
import pandas as pd
import pytest

from gears.plotting import (
    plot_arrival_distribution,
    plot_daily_energy,
    plot_energy_distribution,
    plot_forecast_vs_actual,
    plot_gmm_means,
    plot_regret_comparison,
    plot_session_heatmap,
)


@pytest.fixture(autouse=True)
def _close_figures_after_test():
    """Every plotting function here opens a new figure; close them all after
    each test so the suite doesn't trip matplotlib's open-figure warning."""
    yield
    import matplotlib.pyplot as plt
    plt.close("all")


def _make_panel(n_days=90):
    idx = pd.date_range("2025-01-01", periods=n_days, freq="D")
    return pd.DataFrame({"75": 1000.0, "69": 500.0}, index=idx)


# ---------------------------------------------------------------------------
# Shared fixtures for the newly-covered functions
# ---------------------------------------------------------------------------

@pytest.fixture
def sessions_df():
    from gears.data.loader import make_demo_data
    return make_demo_data(n=200, location_type="work", seed=3)


@pytest.fixture
def fitted_gmm_for_plots():
    from gears.data.loader import make_demo_data
    from gears.models.session_model import EVSessionModel
    sessions = make_demo_data(n=300, location_type="work", seed=4)
    return EVSessionModel(n_components=2, random_state=0,
                           stratify_by=["day_of_week"]).fit(sessions)


@pytest.fixture
def daily_energy_df():
    rng = np.random.default_rng(5)
    dates = pd.date_range("2025-01-01", periods=30, freq="D")
    rows = []
    for sc in range(4):
        rows.append(pd.DataFrame({
            "date": dates,
            "scenario": sc,
            "total_energy_kwh": 1000 + rng.normal(0, 50, len(dates)),
        }))
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# plot_arrival_distribution
# ---------------------------------------------------------------------------

class TestPlotArrivalDistribution:
    def test_basic_returns_axes(self, sessions_df):
        ax = plot_arrival_distribution(sessions_df)
        assert ax is not None

    def test_group_by_creates_legend(self, sessions_df):
        sessions_df = sessions_df.copy()
        sessions_df["day_of_week"] = sessions_df["arrival_time"].dt.dayofweek % 3
        ax = plot_arrival_distribution(sessions_df, group_by="day_of_week")
        assert ax.get_legend() is not None

    def test_accepts_existing_ax(self, sessions_df):
        import matplotlib.pyplot as plt
        _fig, ax = plt.subplots()
        returned = plot_arrival_distribution(sessions_df, ax=ax)
        assert returned is ax


# ---------------------------------------------------------------------------
# plot_session_heatmap
# ---------------------------------------------------------------------------

class TestPlotSessionHeatmap:
    def test_n_sessions_mode(self, sessions_df):
        ax = plot_session_heatmap(sessions_df, value="n_sessions")
        assert ax is not None

    def test_energy_mode(self, sessions_df):
        ax = plot_session_heatmap(sessions_df, value="energy")
        assert ax is not None

    def test_derives_day_of_week_when_absent(self, sessions_df):
        df = sessions_df.drop(columns=["day_of_week"], errors="ignore")
        ax = plot_session_heatmap(df)
        assert ax is not None


# ---------------------------------------------------------------------------
# plot_energy_distribution
# ---------------------------------------------------------------------------

class TestPlotEnergyDistribution:
    def test_basic_returns_axes(self, sessions_df):
        ax = plot_energy_distribution(sessions_df)
        assert ax is not None

    def test_log_scale_false(self, sessions_df):
        ax = plot_energy_distribution(sessions_df, log_scale=False)
        assert "log" not in ax.get_ylabel()

    def test_group_by(self, sessions_df):
        sessions_df = sessions_df.copy()
        sessions_df["bucket"] = sessions_df["energy"] > sessions_df["energy"].median()
        ax = plot_energy_distribution(sessions_df, group_by="bucket")
        assert ax.get_legend() is not None


# ---------------------------------------------------------------------------
# plot_daily_energy
# ---------------------------------------------------------------------------

class TestPlotDailyEnergy:
    def test_with_ci_band(self, daily_energy_df):
        ax = plot_daily_energy(daily_energy_df, ci=True)
        assert ax is not None
        assert len(ax.collections) > 0  # fill_between produces a PolyCollection

    def test_without_scenario_column(self):
        df = pd.DataFrame({
            "date": pd.date_range("2025-01-01", periods=10),
            "total_energy_kwh": np.arange(10, dtype=float),
        })
        ax = plot_daily_energy(df, ci=True)  # no scenario col -> falls back to line
        assert ax is not None


# ---------------------------------------------------------------------------
# plot_gmm_means
# ---------------------------------------------------------------------------

class TestPlotGmmMeans:
    def test_arrival_hour_feature(self, fitted_gmm_for_plots):
        ax = plot_gmm_means(fitted_gmm_for_plots, feature_idx=0, feature_name="Arrival hour")
        assert ax is not None
        assert len(ax.patches) > 0  # bars drawn

    def test_energy_feature_backtransformed(self, fitted_gmm_for_plots):
        # feature_idx=2 triggers the expm1 back-transform branch.
        ax = plot_gmm_means(fitted_gmm_for_plots, feature_idx=2, feature_name="Energy (kWh)")
        assert ax is not None


# ---------------------------------------------------------------------------
# plot_regret_comparison
# ---------------------------------------------------------------------------

class TestPlotRegretComparison:
    def test_basic_bar_chart(self):
        regret = {
            "cost_oracle_smart": 12.5,
            "cost_predicted_smart": 15.0,
            "cost_predicted_plug": 22.3,
        }
        ax = plot_regret_comparison(regret)
        assert ax is not None
        assert len(ax.patches) == 3

    def test_missing_keys_default_to_zero(self):
        ax = plot_regret_comparison({})
        assert ax is not None


# ---------------------------------------------------------------------------
# plot_forecast_vs_actual
# ---------------------------------------------------------------------------

class TestPlotForecastVsActual:
    def test_with_scenario_forecast(self):
        idx = pd.date_range("2025-01-01", periods=20, freq="D")
        actual = pd.Series(np.arange(20, dtype=float), index=idx, name="n_sessions")

        rng = np.random.default_rng(7)
        rows = []
        for sc in range(3):
            rows.append(pd.DataFrame({
                "date": idx,
                "scenario": sc,
                "n_sessions": 10 + rng.normal(0, 1, len(idx)),
            }))
        forecast = pd.concat(rows, ignore_index=True)

        ax = plot_forecast_vs_actual(actual, forecast)
        assert ax is not None
        assert len(ax.collections) > 0

    def test_without_scenario_column(self):
        idx = pd.date_range("2025-01-01", periods=10, freq="D")
        actual = pd.Series(np.arange(10, dtype=float), index=idx)
        forecast = pd.DataFrame({"date": idx, "n_sessions": np.arange(10, dtype=float)})
        ax = plot_forecast_vs_actual(actual, forecast)
        assert ax is not None

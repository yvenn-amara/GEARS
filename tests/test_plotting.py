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
    plot_lt_trajectories,
    plot_mt_fan_charts,
    plot_mt_national_aggregate,
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


@pytest.fixture
def mt_panel_and_forecast():
    """Small synthetic (panel, forecast_df) pair matching what
    DepartmentForecaster.predict() / build_panel() would produce."""
    hist_dates = pd.date_range("2024-01-01", periods=120, freq="D")
    rng = np.random.default_rng(6)
    panel = pd.DataFrame(
        {"92": 1000 + rng.normal(0, 20, len(hist_dates)),
         "69": 500 + rng.normal(0, 15, len(hist_dates))},
        index=hist_dates,
    )
    fc_dates = pd.date_range(hist_dates[-1] + pd.Timedelta(days=1), periods=30, freq="D")
    rows = []
    for dept, base in (("92", 1000.0), ("69", 500.0)):
        for sc in range(5):
            rows.append(pd.DataFrame({
                "date": fc_dates,
                "department": dept,
                "scenario": sc,
                "energy_kwh_forecast": base + rng.normal(0, 30, len(fc_dates)),
            }))
    forecast_df = pd.concat(rows, ignore_index=True)
    return panel, forecast_df


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


# ---------------------------------------------------------------------------
# plot_mt_fan_charts
# ---------------------------------------------------------------------------

class TestPlotMtFanCharts:
    def test_basic_two_departments(self, mt_panel_and_forecast):
        panel, forecast_df = mt_panel_and_forecast
        fig = plot_mt_fan_charts(panel, forecast_df, departments=["92", "69"])
        assert fig is not None
        assert len(fig.axes) >= 2

    def test_missing_department_hides_axis(self, mt_panel_and_forecast):
        panel, forecast_df = mt_panel_and_forecast
        fig = plot_mt_fan_charts(panel, forecast_df, departments=["92", "999"])
        assert fig is not None

    def test_with_metrics_df(self, mt_panel_and_forecast):
        panel, forecast_df = mt_panel_and_forecast
        metrics_df = pd.DataFrame({
            "Département": ["92", "69"],
            "MAPE (%)": [5.2, 7.8],
            "RMSE (kWh)": [120.0, 80.0],
        })
        fig = plot_mt_fan_charts(panel, forecast_df, departments=["92", "69"],
                                  metrics_df=metrics_df)
        assert fig is not None


# ---------------------------------------------------------------------------
# plot_mt_national_aggregate (also exercises the French-label fix, Session 9)
# ---------------------------------------------------------------------------

class TestPlotMtNationalAggregate:
    def test_basic_aggregate(self, mt_panel_and_forecast):
        panel, forward_fc = mt_panel_and_forecast
        fig = plot_mt_national_aggregate(panel, forward_fc, focus_depts=["92", "69"])
        assert fig is not None
        ax = fig.axes[0]
        # English labels only — regression guard for the Session 9 translation fix.
        legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
        joined = " ".join(legend_texts + [ax.get_ylabel(), ax.get_title()])
        for french_word in ("Données", "Énergie", "Agrégat", "prévision", "médiane", "dépts"):
            assert french_word not in joined

    def test_with_nhits_overlay(self, mt_panel_and_forecast):
        panel, forward_fc = mt_panel_and_forecast
        nhits_dates = forward_fc["date"].unique()
        rng = np.random.default_rng(8)
        rows = []
        for sc in range(3):
            rows.append(pd.DataFrame({
                "date": nhits_dates,
                "scenario": sc,
                "n_sessions": 100 + rng.normal(0, 5, len(nhits_dates)),
            }))
        nhits_forecast = pd.concat(rows, ignore_index=True)
        fig = plot_mt_national_aggregate(
            panel, forward_fc, focus_depts=["92", "69"],
            nhits_forecast=nhits_forecast, nhits_mean_kwh=10.0,
        )
        assert fig is not None

    def test_nhits_without_mean_kwh_raises(self, mt_panel_and_forecast):
        panel, forward_fc = mt_panel_and_forecast
        nhits_forecast = pd.DataFrame({
            "date": forward_fc["date"].unique(),
            "scenario": 0,
            "n_sessions": 100.0,
        })
        with pytest.raises(ValueError, match="nhits_mean_kwh"):
            plot_mt_national_aggregate(
                panel, forward_fc, focus_depts=["92", "69"],
                nhits_forecast=nhits_forecast,
            )

"""Forecast the number of sessions per day with three methods, then schedule charging on a price.

    python examples/02_forecast_and_smart_charging.py

Part 1 holds out the last two weeks of the demo data and compares what each forecaster predicts
with what happened. Part 2 shifts the charging of simulated sessions to the cheapest hours of a
day/night tariff (V1G) and measures what the forecast costs compared with perfect knowledge.
Takes a few seconds; no data file, no internet connection. The figure is saved to
``$GEARS_EXAMPLES_OUT`` (default: ``./example_outputs``) as ``02_forecast_and_smart_charging.png``.
"""

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")  # no window: the figure goes to a PNG file

import matplotlib.pyplot as plt
import pandas as pd

from gears import (
    GEARSModel,
    PersistenceForecaster,
    SessionForecaster,
    SmartChargingOptimizer,
    make_demo_data,
    sessions_to_daily_counts,
)
from gears.plotting import plot_forecast_vs_actual, plot_regret_comparison
from gears.utils import forecast_metrics, make_price_signal

OUT = Path(os.environ.get("GEARS_EXAMPLES_OUT", "example_outputs"))
OUT.mkdir(parents=True, exist_ok=True)

sessions = make_demo_data(n=3000, location_type="work", seed=0)
dates = pd.to_datetime(sessions["date"])
train, test = sessions[dates < "2023-12-18"], sessions[dates >= "2023-12-18"]

# --- Part 1: three forecasters of the daily number of sessions ---------------------------------
actual = sessions_to_daily_counts(test)  # a Series: sessions per day, zero-filled, dated

forecasters = {
    "probabilistic": SessionForecaster(method="probabilistic"),  # mean and spread of past days
    "persistence": PersistenceForecaster(n_weeks=1),  # the same weekday, one week earlier
    # SARIMA is always tried; if it cannot be fitted a warning says which method replaces it.
    # use_holidays=False skips the public-holiday regressor, which makes the fit slower.
    "SARIMA": SessionForecaster(method="sarima", use_holidays=False),
}
forecasts = {}
for name, forecaster in forecasters.items():
    forecaster.fit(train)
    forecasts[name] = forecaster.predict(
        horizon=len(actual), n_scenarios=100, start_date=actual.index[0], seed=0
    )
    median = forecasts[name].groupby("date")["n_sessions"].median()
    scores = forecast_metrics(actual.to_numpy(), median.to_numpy())
    print(f"{name:14s} RMSE {scores['RMSE']:.2f}  MAE {scores['MAE']:.2f}  bias {scores['bias']:+.2f}")

# --- Part 2: V1G smart charging against a day/night price ---------------------------------------
# Every charger delivers 7.4 kW, so that real and simulated sessions are comparable.
model = GEARSModel(
    n_components=3, forecaster_method="probabilistic", n_scenarios=1,
    charger_mix={7.4: 1.0}, random_state=0,
).fit(train, verbose=False)

price = make_price_signal(start="2023-12-18", periods=14 * 48, resolution_min=30, pattern="day_night")
predicted = model.simulate_short_term("2023-12-18", horizon=14, n_scenarios=1, seed=0)

optimiser = SmartChargingOptimizer(signal_type="price", resolution_min=30)
scheduled = optimiser.optimise(predicted, price)  # the input plus cost_smart, cost_plug, scheduled_start...
summary = optimiser.savings_summary(scheduled)
print(
    f"\nV1G on {summary['n_sessions_optimised']} simulated sessions: "
    f"{summary['total_cost_plug_eur']:.0f} EUR plug-and-charge -> "
    f"{summary['total_cost_smart_eur']:.0f} EUR smart ({summary['mean_savings_pct']:.1f} % saved per session)"
)

# Regret: what the forecast costs compared with knowing the real sessions in advance ("oracle").
# A charger cannot deliver more than power x duration; the simulator respects that, so cap the
# real sessions too (the synthetic demo data does not).
real = test.assign(power_kw=7.4)
real = real.assign(energy=real["energy"].clip(upper=real["power_kw"] * real["duration"]))
# Persistence guess: the sessions of the last week of the training data, repeated twice.
last_week = train[pd.to_datetime(train["date"]) >= "2023-12-11"]
persistence = pd.concat(
    [last_week.assign(arrival_time=last_week["arrival_time"] + pd.Timedelta(days=d)) for d in (7, 14)]
)
persistence = persistence.assign(date=persistence["arrival_time"].dt.date, power_kw=7.4)
regret = optimiser.compute_regret(real, predicted, price, persistence_sessions=persistence)
print(f"{len(real)} real, {len(predicted)} simulated and {len(persistence)} persistence sessions")
print("costs and regret (EUR):", regret)
# Costs add up over sessions: when a scenario holds fewer sessions than the real period, its
# regret is negative without the forecast being "better". Compare scenarios of similar size.

# --- Figure ---------------------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(13, 7))
for ax, (name, forecast) in zip(axes.flat, forecasts.items()):
    plot_forecast_vs_actual(actual, forecast, ax=ax, title=f"{name}: forecast vs actual")
    ax.tick_params(axis="x", rotation=30)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.2)  # room above the curves for the legend
    ax.legend(loc="upper left", ncol=3, fontsize=8)
plot_regret_comparison(regret, ax=axes[1, 1])
fig.tight_layout()
fig.savefig(OUT / "02_forecast_and_smart_charging.png", dpi=120)
print(f"figure saved to {OUT / '02_forecast_and_smart_charging.png'}")

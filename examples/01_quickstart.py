"""Quickstart: fit GEARS on demo data, simulate a week of charging, draw the load curve.

    python examples/01_quickstart.py

Takes a few seconds and needs no data file and no internet connection. The figure is saved
to ``$GEARS_EXAMPLES_OUT`` (default: ``./example_outputs``) as ``01_quickstart.png``.
"""

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")  # no window: the figure goes to a PNG file

import matplotlib.pyplot as plt
import pandas as pd

from gears import GEARSModel, make_demo_data
from gears.plotting import plot_daily_energy

OUT = Path(os.environ.get("GEARS_EXAMPLES_OUT", "example_outputs"))
OUT.mkdir(parents=True, exist_ok=True)

# 1. Data: 3 000 synthetic sessions of a workplace car park, over the year 2023.
#    Your own file works the same way: see 04_bring_your_own_data.py.
sessions = make_demo_data(n=3000, location_type="work", seed=0)
print(sessions[["arrival_time", "duration", "energy"]].head(3), end="\n\n")

# 2. Fit: one Gaussian mixture of (arrival hour, duration, energy) per weekday and season,
#    and a forecaster of the number of sessions per day.
model = GEARSModel(n_components=3, forecaster_method="probabilistic", n_scenarios=10, random_state=0)
model.fit(sessions, verbose=False)
print(model.summary(), end="\n\n")

# 3. Simulate the first week of June 2024: 10 scenarios, one row per charging session.
simulated = model.simulate_short_term("2024-06-03", horizon=7, seed=0)
print(f"{len(simulated)} simulated sessions in {simulated['scenario'].nunique()} scenarios")

# 4. Aggregate: energy per day, and energy of the sessions that arrive in each hour.
daily = model.daily_energy(simulated)
hourly = model.hourly_profile(simulated)
hourly["time"] = pd.to_datetime(hourly["date"]) + pd.to_timedelta(hourly["hour"], unit="h")
curve = hourly.pivot_table(index="time", columns="scenario", values="energy_kwh", aggfunc="sum")
curve = curve.reindex(pd.date_range("2024-06-03", periods=7 * 24, freq="h")).fillna(0.0)

# 5. Draw: the daily energy with its range over the scenarios, and the hourly curve.
fig, (ax_daily, ax_hourly) = plt.subplots(1, 2, figsize=(13, 4))
plot_daily_energy(daily, ax=ax_daily, title="Daily energy, 10 scenarios")
ax_hourly.fill_between(curve.index, curve.quantile(0.1, axis=1), curve.quantile(0.9, axis=1),
                       alpha=0.25, label="10-90 % of the scenarios")
ax_hourly.plot(curve.index, curve.median(axis=1), label="median")
ax_hourly.set_title("Energy of the sessions arriving each hour")
ax_hourly.set_ylabel("kWh")
ax_hourly.legend(loc="upper left", fontsize=8)
ax_hourly.tick_params(axis="x", rotation=30)
ax_daily.tick_params(axis="x", rotation=30)
ax_daily.set_xlabel("")
fig.tight_layout()
fig.savefig(OUT / "01_quickstart.png", dpi=120)
print(f"figure saved to {OUT / '01_quickstart.png'}")

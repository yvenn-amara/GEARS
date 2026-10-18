"""Medium-term scenarios: how demand grows over the years, and what the chargers change.

    python examples/03_medium_term_scenarios.py

Part 1 compares the three growth curves (linear, S-curve, Bass diffusion) and the daily energy
they give over a year. Part 2 changes the mix of charger powers with ``CHARGER_PRESETS``.
Takes a few seconds; no data file, no internet connection. The figure is saved to
``$GEARS_EXAMPLES_OUT`` (default: ``./example_outputs``) as ``03_medium_term_scenarios.png``.
"""

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")  # no window: the figure goes to a PNG file

import matplotlib.pyplot as plt

from gears import (
    CHARGER_PRESETS,
    GROWTH_PROFILES,
    GEARSModel,
    MediumTermSimulator,
    make_demo_data,
)
from gears.plotting import plot_daily_energy

OUT = Path(os.environ.get("GEARS_EXAMPLES_OUT", "example_outputs"))
OUT.mkdir(parents=True, exist_ok=True)

sessions = make_demo_data(n=3000, location_type="work", seed=0)
model = GEARSModel(n_components=3, forecaster_method="probabilistic", random_state=0)
model.fit(sessions, verbose=False)
base = len(sessions) / sessions["date"].nunique()  # sessions per day today
print(f"today: {base:.1f} sessions per day\n")

# --- Part 1: three ways for the number of sessions to grow ---------------------------------------
# Each profile is a Series of expected sessions per day, from today's level.
profiles = {
    "linear": GROWTH_PROFILES["linear"](base, years=5, annual_growth_rate=0.15, start_date="2024-01-01"),
    "s_curve": GROWTH_PROFILES["s_curve"](base, years=5, saturation_factor=3.0, start_date="2024-01-01"),
    "bass": GROWTH_PROFILES["bass"](base, years=5, market_potential_factor=4.0, start_date="2024-01-01"),
}
for name, profile in profiles.items():
    print(f"{name:8s} sessions per day: {profile.iloc[0]:5.1f} today -> {profile.iloc[-1]:5.1f} after 5 years")

# The same growth models drive the simulator: one year, 3 scenarios, daily energy.
daily = {}
for name in profiles:
    simulator = MediumTermSimulator(
        model.gmm_, charger_mix=CHARGER_PRESETS["workplace"], growth_model=name, n_scenarios=3, seed=0
    )
    daily[name] = simulator.simulate(years=1, start_date="2024-01-01", output="daily_energy")
    mwh = daily[name]["total_energy_kwh"].sum() / daily[name]["scenario"].nunique() / 1000
    print(f"{name:8s} energy over 2024, mean of the scenarios: {mwh:5.1f} MWh")

# --- Part 2: the same demand with other chargers -----------------------------------------------
# The charger mix sets the power of each session, and a session cannot take more energy than
# power x duration. A busy hour then asks for very different installed power.
print("\ncharger preset      mean charger power   power connected in the busiest hour")
for preset, mix in CHARGER_PRESETS.items():
    simulator = MediumTermSimulator(model.gmm_, charger_mix=mix, growth_model="linear", n_scenarios=1, seed=0)
    year = simulator.simulate(years=0.25, start_date="2024-01-01", output="sessions")
    hourly = model.hourly_profile(year)  # per hour: energy, number of sessions, sum of charger powers
    mean_power = sum(power * share for power, share in simulator.charger_mix.items())
    busiest = hourly.groupby("date")["power_kw"].max().mean()
    print(f"{preset:18s} {mean_power:12.1f} kW {busiest:24.0f} kW")

# --- Figure ----------------------------------------------------------------------------------------
fig, (ax_growth, ax_energy) = plt.subplots(1, 2, figsize=(13, 4))
for name, profile in profiles.items():
    ax_growth.plot(profile.index, profile.to_numpy(), label=name)
ax_growth.set_title("Sessions per day under three growth models")
ax_growth.set_ylabel("sessions per day")
ax_growth.legend()
ax_growth.grid(True, alpha=0.3)
plot_daily_energy(daily["s_curve"], ax=ax_energy, title="Daily energy, S-curve, 3 scenarios")
for ax in (ax_growth, ax_energy):
    ax.tick_params(axis="x", rotation=30)
ax_energy.set_xlabel("")
fig.tight_layout()
fig.savefig(OUT / "03_medium_term_scenarios.png", dpi=120)
print(f"\nfigure saved to {OUT / '03_medium_term_scenarios.png'}")

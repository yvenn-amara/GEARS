# GEARS — Generating Electric vehicle recharging Sessions

[![CI](https://github.com/EDF-Lab/GEARS/actions/workflows/ci.yml/badge.svg)](https://github.com/EDF-Lab/GEARS/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## What GEARS does

GEARS learns from charging-session records what a session looks like (when it starts, how long
it lasts, how much energy it takes) and how many sessions a day to expect. It then simulates
new sessions, from a week to several years ahead, and tests smart-charging strategies on them.

```
sessions ─► load_sessions ─► EVSessionModel ─┐   what one session looks like
(CSV, ...)   (checks)        (a mixture per  │   (arrival, duration, energy)
                              context)       ├─► ShortTermSimulator ─► simulated sessions
daily counts ─► SessionForecaster ───────────┘   (days to weeks)         │
                (SARIMA, probabilistic,                                  ├─► OutputAggregator: energy, load
                 persistence)                                            └─► SmartChargingOptimizer: V1G cost
growth curves + chargers ─► MediumTermSimulator ─► years of demand
```

## Installation

Python 3.10 or later.

```bash
pip install "gears-ev @ git+https://github.com/EDF-Lab/GEARS"
```

| Extra | Adds | For |
|---|---|---|
| `[vae]` | `torch` | the VAE session model (`model_type="vae"`) |
| `[dl]` | `torch`, `neuralforecast` | the NHiTS and PatchTST forecasters |
| `[dev]` | `pytest`, `ruff`, ... | working on GEARS itself (see [CONTRIBUTING.md](CONTRIBUTING.md)) |

Install an extra with `pip install "gears-ev[vae] @ git+https://github.com/EDF-Lab/GEARS"`.
`import gears` is instant and loads none of them.

## Start in 60 seconds

No data needed: `make_demo_data` generates synthetic sessions.

<!-- test -->
```python
from gears import GEARSModel, make_demo_data

sessions = make_demo_data(n=3000, location_type="work", seed=0)
model = GEARSModel(n_components=3, forecaster_method="probabilistic", random_state=0)
model.fit(sessions, verbose=False)

simulated = model.simulate_short_term("2024-06-03", horizon=7, n_scenarios=5, seed=0)
print(simulated.head())  # one row per session: arrival_time, duration, energy, power_kw, scenario...
```

With your own data, `load_sessions` reads a CSV, Parquet, Excel, JSON or pickle file, renames the
usual column names, drops impossible rows (and counts them in one warning), then you fit as above:

<!-- test -->
```python
from gears import GEARSModel, load_sessions

sessions = load_sessions("data/sessions.csv")
model = GEARSModel(n_components=3, forecaster_method="probabilistic").fit(sessions)
```

## Your data

One row per charging session. Column names are matched loosely (`start_time`, `kwh`, ...).

| Column | | Meaning |
|---|---|---|
| `arrival_time` | required | date and time of plug-in, local wall-clock time |
| `duration` | required | hours, > 0 |
| `energy` | required | kWh, ≥ 0 |
| `location_type` | optional | `work`, `home`, `public` or `heavy`: one model per type |
| `department` | optional | INSEE département code (French data) |
| `power`, `user_id` | optional | charger power in kW, anonymised user |

Units, time zones, filters and the minimum size are in [data/README.md](data/README.md).

## Three levels of use

<!-- test -->
```python
# 1. Only the session generator (a Gaussian mixture per context)
from gears.models import EVSessionModel

gmm = EVSessionModel(n_components=3).fit(sessions)
new_sessions = gmm.sample(1000, seed=0)  # arrival_hour, duration, energy

# 2. Independent building blocks
from gears.models import SessionForecaster
from gears.simulation import ShortTermSimulator

forecaster = SessionForecaster(method="probabilistic").fit(sessions)
week = ShortTermSimulator(gmm, forecaster).simulate("2024-06-03", horizon=7, n_scenarios=5, seed=0)

# 3. The whole pipeline: session model + forecaster + simulation + smart charging
from gears import GEARSModel

model = GEARSModel(n_components=3, forecaster_method="probabilistic").fit(sessions)
```

## The pre-trained French model

GEARS ships no data and no model. The official French model is downloaded on first use,
checked against its SHA-256 before it is opened, and cached in `~/.cache/gears`
(`GEARS_HOME` changes the folder). It is fitted per site type, département, weekday and season,
so name the context you want:

<!-- test -->
```python
from gears import GEARSModel, ShortTermSimulator

french = GEARSModel.from_pretrained("french")
paris_monday = {"location_type": "work", "department": "75", "day_of_week": 0, "season": "summer"}
day = ShortTermSimulator(french.gmm_).simulate_single_day(
    "2025-06-02", n_sessions=40, context=paris_monday, seed=0, strict=True
)
```

- **Without internet**: download the file on another machine, then
  `load_pretrained("french", path="/where/gmm_french.joblib")`, or copy it into `GEARS_HOME`.
  `GEARS_OFFLINE=1` forbids any download.
- **Security**: model files are `joblib` pickles, which can run code when opened. GEARS checks the
  checksum of the official file; never open a model file whose source you do not trust.
- `from_pretrained("french")` warns if your scikit-learn differs from the one the model was
  trained with (`pip install scikit-learn==<that version>` reproduces it exactly).

## Forecasting, medium-term scenarios, smart charging

**Forecast the number of sessions per day** (`"sarima"`, `"probabilistic"`, or `PersistenceForecaster`):

<!-- test -->
```python
from gears import SessionForecaster, sessions_to_daily_counts
from gears.utils import forecast_metrics

history = sessions[sessions["arrival_time"] < "2023-12-18"]
actual = sessions_to_daily_counts(sessions[sessions["arrival_time"] >= "2023-12-18"])
forecast = SessionForecaster(method="probabilistic").fit(history).predict(
    horizon=len(actual), n_scenarios=100, start_date=actual.index[0], seed=0
)
median = forecast.groupby("date")["n_sessions"].median()
print(forecast_metrics(actual.to_numpy(), median.to_numpy()))  # RMSE, MAE, bias...
```

**Medium term**: years of demand under a growth model (`linear`, `s_curve`, `bass`) and a mix of chargers:

<!-- test -->
```python
from gears import CHARGER_PRESETS, MediumTermSimulator

simulator = MediumTermSimulator(
    model.gmm_, charger_mix=CHARGER_PRESETS["workplace"], growth_model="s_curve", n_scenarios=3, seed=0
)
daily = simulator.simulate(years=1, start_date="2024-01-01", output="daily_energy")
```

**Smart charging (V1G)**: charge each session in the cheapest slots of its plug-in window, and compare
with plug-and-charge:

<!-- test -->
```python
from gears import SmartChargingOptimizer
from gears.utils import make_price_signal

price = make_price_signal(start="2024-06-03", periods=7 * 48, resolution_min=30, pattern="day_night")
optimiser = SmartChargingOptimizer(signal_type="price", resolution_min=30)
scheduled = optimiser.optimise(simulated[simulated["scenario"] == 0], price)
print(optimiser.savings_summary(scheduled))  # plug-and-charge cost vs smart cost
```

## Command line

<!-- test -->
```bash
gears fit data/sessions.csv --output my_model.joblib --forecaster probabilistic --n-components 3
gears simulate --model my_model.joblib --start 2025-10-01 --horizon 7 --output simulated.csv
```

`gears --help` lists the options (`--stratify-by`, `--scenarios`, `--seed`, `--pretrained french`...).

## Examples

Each script runs in well under a minute, on simulated data, and writes its figures and files to
`example_outputs/` (or to `$GEARS_EXAMPLES_OUT`).

| Script | Shows |
|---|---|
| `examples/01_quickstart.py` | demo data → fit → a week of sessions → daily energy and hourly load |
| `examples/02_forecast_and_smart_charging.py` | three forecasters compared, V1G savings, regret |
| `examples/03_medium_term_scenarios.py` | linear, S-curve and Bass growth; charger presets |
| `examples/04_bring_your_own_data.py` | your file → fit → save → load → simulate (`--pretrained`: the French model) |

## Known limits

- A session is described by its arrival, duration and energy. GEARS does not follow vehicles,
  drivers or batteries, and it does not model the grid.
- Smart charging is V1G: charging is moved in time within the plug-in window; nothing is fed back to
  the grid (no V2G).
- Three session models are available (Gaussian mixture, recency-weighted mixture, VAE). This README
  does not rank them: no comparison is claimed here.
- `GEARSModel.simulate_short_term` cannot yet choose the site type or département of a model fitted per
  département (the French model): use `ShortTermSimulator.simulate_single_day(context=...)` as above.
- Timestamps are naive local times; a time zone in your file is dropped, not converted.

## Contributing, tests, citation, licence

- **Contributing**: see [CONTRIBUTING.md](CONTRIBUTING.md) (setup, tests, style, pull-request checklist).
- **Tests**: `pip install -e ".[dev]"`, then `pytest`. The Python blocks marked in this README run
  in `tests/test_readme.py`, so they cannot drift from the code.
- **Citation**: if you use GEARS in a publication, please cite the software: Amara-Ouali, Y., *GEARS:
  Generating Electric vehicle recharging Sessions*, https://github.com/EDF-Lab/GEARS.
- **Licence**: MIT, see [LICENSE](LICENSE). Changes: [CHANGELOG.md](CHANGELOG.md).

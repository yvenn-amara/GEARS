# GEARS — EV Charging Simulator

**GEARS** (Generating Electric Vehicle Recharging Sessions) is an open-source Python package for
simulating, forecasting, and optimising electric-vehicle charging demand at any scale —
from a single site to a national fleet.

```
[Short-term]   GMM session sampling  +  SARIMA / NHiTS forecast  →  V1G optimiser
[Medium-term]  DepartmentForecaster (SARIMA per département)      →  fan-chart projections
[Long-term]    EV adoption curves (Bass / S-curve / linear)       →  energy trajectory scenarios
```

---

## Installation (source — no PyPI release yet)

Requires **Python ≥ 3.10** and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/your-org/gears.git
cd gears
uv pip install -e .
```

All core dependencies — including `pmdarima` (SARIMA order selection) and `holidays`
(French bank-holiday covariates) — are installed automatically.

**With deep-learning forecasters (NHiTS / PatchTST):**
```bash
uv pip install -e ".[dl]"
```

**Full development environment (tests, notebooks, linting):**
```bash
uv pip install -e ".[dev]"
```

---

## Quick Start

```python
import gears
from gears import load_sessions, GEARSModel

# 1. Load and validate sessions (any format — see Data Formats below)
df = load_sessions("data/sessions.pkl")

# 2. Fit the full pipeline (GMM + SARIMA forecaster)
model = GEARSModel(forecaster_method="sarima")
model.fit(df)

# 3. Simulate 14 days of individual sessions (50 stochastic scenarios)
simulated = model.simulate_short_term(
    start_date="2025-10-01", horizon=14, n_scenarios=50, seed=0
)
print(simulated.head())
```

---

## Running the Notebooks

```bash
cd notebooks
jupyter lab
```

| # | Notebook | Contents |
|---|---|---|
| 1 | `1_gmm_descriptive.ipynb` | Registry API, sklearn access, component weights & means, descriptive stats per stratum, arrival/energy distributions, seasonal & DoW heatmaps, département coverage map |
| 2 | `2_gmm_forecasting.ipynb` | D−1 split, SARIMA/NHiTS/Persistence evaluation, energy forecast, V1G smart charging & regret analysis |
| 3 | `3_gmm_scenarios.ipynb` | **A:** département fan charts (1–3 yr); **B:** EV adoption curves → energy trajectories 2025–2040 |

Notebooks 2 & 3 expect a session dataset at `../data/sample_df.pkl`. Any file format
supported by `load_sessions()` works — update the `DATA_PATH` constant at the top of
each notebook.

---

## Registry API — Pre-fitted GMMs

GEARS ships a **single unified GMM** (`'french'`) fitted on the French national IRVE
dataset, stratified by **`location_type × département × day_of_week × season`**.

### Load the bundle

```python
from gears import NativeGMMRegistry

registry = NativeGMMRegistry()
print(registry.list())        # one entry: 'french'
gmm = registry.load("french")
```

### Retrieve a specific stratum

```python
import gears

# Primary retrieval API — returns the full EVSessionGMM wrapper
gmm = gears.get_gmm(
    location_type="work",    # "work" | "home" | "public" | "heavy"
    departement="75",        # INSEE département code
    saison="winter",         # "winter" | "spring" | "summer" | "autumn"
    day_of_week=0,           # Monday=0, …, Sunday=6
)

# Underlying sklearn GaussianMixture for this stratum
sk_gmm = gmm.get_sklearn_gmm(
    context={"location_type": "work", "department": "75",
             "season": "winter", "day_of_week": 0}
)
print(sk_gmm.means_)   # columns: [arrival_hour, log1p(duration_h), log1p(energy_kWh)]

# Sample synthetic sessions
sessions = gmm.sample(
    100,
    context={"location_type": "work", "department": "75",
             "season": "winter", "day_of_week": 0},
    seed=42,
)
```

---

## Fitting GMMs with `fit_gmm.py`

Re-fit on your own data using the provided script. All supported formats work.

### Usage

```bash
# Quickstart — use all defaults
python scripts/fit_gmm.py --input /path/to/sessions.pkl

# CSV with custom BIC range
python scripts/fit_gmm.py \
    --input /data/france_ev.csv \
    --year 2025 \
    --max-samples 5000 \
    --n-components auto \
    --max-components 10 \
    --output-dir gears/data/gmm

# Full option list
python scripts/fit_gmm.py --help
```

### Supported data formats

| Extension | Notes |
|---|---|
| `.pkl` / `.pickle` | Pandas DataFrame serialised with `joblib` or `pickle` |
| `.csv` / `.tsv` | Auto-detects `,` vs `;` separator |
| `.parquet` | Apache Parquet |
| `.xlsx` / `.xls` | Excel workbooks |
| `.json` / `.jsonl` | JSON / newline-delimited JSON |

### Required columns (after loading)

After `load_sessions()` processes the file, these columns must be present:

| Column | Type | Description |
|---|---|---|
| `arrival_time` | `datetime` | Plug-in timestamp |
| `duration` | `float` | Session duration in **hours** |
| `energy` | `float` | Energy delivered in **kWh** |

Optional but used for stratification:

| Column | Description |
|---|---|
| `location_type` | `work` / `home` / `public` / `heavy` |
| `department` | INSEE département code (e.g. `'75'`) |

Common column name aliases (e.g. `start_time`, `energy_kwh`, `duration_h`) are
auto-detected and renamed by `load_sessions()`. For non-standard names, rename
your columns before running the script.

### Verify the result

```python
import gears
reg = gears.NativeGMMRegistry()
print(reg.list())            # should show 'french' as available=True
gmm = reg.load("french")
print(repr(gmm))             # EVSessionGMM(n_contexts=..., stratify_by=...)
```

---

## Forecasters

All share the same `fit(df)` / `predict(horizon, n_scenarios, ...)` interface.

| Class | Description | Requires |
|---|---|---|
| `SessionForecaster(method='sarima')` | Auto-ARIMA (BIC) + bank-holiday exog | base |
| `SessionForecaster(method='probabilistic')` | Normal distribution baseline | base |
| `PersistenceForecaster` | Same weekday, walks back indefinitely | base |
| `TransformerForecaster` | PatchTST via neuralforecast | `[dl]` |
| `NHiTSForecaster` | NHiTS via neuralforecast | `[dl]` |

> **D−1 rule:** train on `df[df["arrival_time"] < eval_date]` — data strictly before
> evaluation day. See Notebook 2 for the canonical evaluation loop.

---

## Smart Charging (V1G)

```python
from gears.smart_charging.optimizer import SmartChargingOptimizer
from gears.utils import make_price_signal

signal = make_price_signal(start="2025-10-01", periods=48*14, resolution_min=30)
opt    = SmartChargingOptimizer(signal_type="price")
result = opt.optimise(sessions_df, signal)
regret = opt.compute_regret(oracle_sessions, predicted_sessions, signal)
```

---

## Tests

```bash
python -m pytest tests/ -q
# Expected: 142 passed, 2 skipped
```

---

## Package Structure

```
gears/
├── data/           loader, schemas, INSEE helpers, pre-fitted GMMs
├── models/         EVSessionGMM, forecasters, registry, get_gmm()
├── simulation/     ShortTermSimulator, MediumTermSimulator
├── smart_charging/ SmartChargingOptimizer (V1G)
├── output/         aggregators, exporters
├── plotting.py
└── utils.py
scripts/
└── fit_gmm.py      — fit GMMs on your own data
notebooks/
├── 1_gmm_descriptive.ipynb
├── 2_gmm_forecasting.ipynb
└── 3_gmm_scenarios.ipynb
tests/              142 tests
```

---

## Licence

MIT — see `LICENSE`.

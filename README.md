# GEARS — EV Charging Simulator

**GEARS** (Generating Electric Vehicle Recharging Sessions) is an open-source Python package for
simulating, forecasting, and optimising electric-vehicle charging demand at any scale —
from a single site to a national fleet.

```
[Short-term]   GMM / VAE session sampling  +  SARIMA / NHiTS forecast  →  V1G optimiser
[Medium-term]  DepartmentForecaster (SARIMA per département)            →  fan-chart projections
[Long-term]    EV adoption curves (linear / S-curve / Bass)             →  energy trajectory scenarios
```

Two interchangeable session-generation models are available: a Gaussian-Mixture model
(`EVSessionGMM`, the default — optionally recency-weighted) and a Conditional VAE
(`model_type="vae"`). Both share the same fit/sample interface and are compared head-to-head
by the built-in persistence/GMM/VAE benchmark harness (see below).

---

## Installation (source — no PyPI release yet)

Requires **Python ≥ 3.10**.

```bash
git clone https://github.com/yvenn-amara/GEARS.git
cd GEARS
pip install -e .
```

All core dependencies — including `pmdarima` (SARIMA order selection) and `holidays`
(French bank-holiday covariates) — are installed automatically.

**With the VAE session model:**
```bash
pip install -e ".[vae]"
```

**With deep-learning forecasters (NHiTS / PatchTST):**
```bash
pip install -e ".[dl]"
```

**Full development environment (tests, notebooks, linting, VAE, coverage):**
```bash
pip install -e ".[dev]"
```

---

## Quick Start

```python
from gears import load_sessions, GEARSModel

# 1. Load and validate sessions (any format — see data/README.md)
df = load_sessions("data/sessions.csv")

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

## Command line

`pip install -e .` also registers a `gears` command covering the same pipeline
(`gears --help` for the full list):

```bash
# Fit a model on your own data and save it
gears fit data/sessions.csv --output my_model.joblib --forecaster probabilistic

# Simulate sessions from a saved model
gears simulate --model my_model.joblib --start 2025-10-01 --horizon 7 --output sessions.csv

# Medium-term energy projection (SARIMA-driven, growth-adjusted)
gears medium-term --model my_model.joblib --years 2 --growth 0.15 --growth-model linear

# Apply V1G smart charging to a previously-simulated sessions file
gears smart-charge --model my_model.joblib --sessions sessions.csv --signal price.csv

# List pre-trained model bundles (downloaded from Hugging Face Hub on first use —
# requires the `hub` extra and network access to huggingface.co)
gears list-models
```

`fit`, `simulate --model ...`, and `medium-term` were each run end-to-end against real data
while writing this section. `simulate --pretrained ...` / `list-models`'s underlying download
step needs Hugging Face Hub network access this environment didn't have when this README was
verified — the command itself is wired up correctly (`gears list-models` prints the catalogue
below without downloading anything), just not exercised end-to-end here.

```
$ gears list-models
   model_id  ...  country                                       stratify_by  n_sessions
french_demo  ...       FR  [location_type, department, day_of_week, season]       15000
```

---

## Running the Notebooks

```bash
cd notebooks
jupyter lab
```

| # | Notebook | Contents | Runtime |
|---|---|---|---|
| 1 | `1_gmm_descriptive.ipynb` | Registry API, sklearn access, component weights & means, descriptive stats per stratum, arrival/energy distributions, seasonal & day-of-week breakdowns | 28s |
| 2 | `2_gmm_forecasting.ipynb` | D−1 split, SARIMA/Persistence evaluation, GMM vs. VAE session simulation, energy forecast, V1G smart charging & regret analysis | 92s |
| 3 | `3_gmm_scenarios.ipynb` | **A:** département-level SARIMA fan charts (1–3 yr); **B:** EV adoption curves (linear/S-curve/Bass) → national energy trajectories 2025–2040; **C:** plug-and-charge vs. smart-charging load profiles, GMM vs. VAE | 107s |
| 4 | `4_persistence_vs_gmm_benchmark.ipynb` | 4-arm rolling-origin benchmark (persistence / GMM / recency-GMM / VAE) across 8 public datasets, win-rates, CRPS, cached results | 7s (cached) |
| 5 | `5_generic_dataset_example.ipynb` | Bring-your-own-data walkthrough: load, fit, simulate, forecast, smart-charge on a non-French dataset, plus a schema-portability check on a second dataset | 41s |

All five execute end-to-end via `nbconvert` with zero errors (measured this session, see
`REFACTOR_STATE.md`); each is comfortably under 5 minutes. Notebooks 1–3 expect a session
dataset at `../data/sample_df.pkl` (notebook 2 subsamples it explicitly, see its own markdown);
notebook 4 reads the 11 public CSVs under `../data/preprocessed_data/`; notebook 5 reads
`../data/custom/sap.csv`. See `data/README.md` for how to obtain all of these.

---

## Registry API — Pre-fitted GMM / VAE bundles

GEARS ships two pre-fitted native bundles, both stratified by
**`location_type × département × day_of_week × season`**:

| `gmm_id` | Model | Notes |
|---|---|---|
| `"french"` | GMM | 8,008 contexts, fitted on the full French national IRVE dataset. |
| `"french_vae_sample"` | VAE | Curated bundle not committed to this repo (see caveat below) — falls back to a small synthetic demo (109 contexts, no département dimension) when unavailable. |

### Load a bundle

```python
from gears import NativeGMMRegistry

registry = NativeGMMRegistry()
print(registry.list())        # 'french' and 'french_vae_sample'
gmm = registry.load("french")
```

> **Known gap** (flagged, not silently papered over): `gmm_vae_french_sample.joblib`, the
> curated real VAE bundle, isn't git-tracked in this repo. `registry.load("french_vae_sample")`
> transparently substitutes a small synthetic fallback instead (`gmm.metadata_["synthetic_fallback"]
> == True`) — useful for exercising the API, not for real GMM-vs-VAE quality comparisons.
> Notebooks 2 and 3 print this caveat explicitly whenever it applies. Fitting and committing the
> real bundle is flagged as future work, not attempted in this refactor (modeling work needing
> separate review — see `REFACTOR_STATE.md`).

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

## Recency-weighted GMM and the VAE session model

`EVSessionGMM` supports two opt-in alternatives to the default single-component-per-context fit:

```python
from gears import EVSessionGMM

# Recency-weighted: half-life exponential decay + weighted bootstrap resample,
# since sklearn's GaussianMixture has no sample_weight.
gmm_recency = EVSessionGMM(stratify_by=["day_of_week"], recency=True, half_life_days=21)
gmm_recency.fit(df)

# Conditional VAE instead of a GMM, same interface.
vae = EVSessionGMM(stratify_by=["day_of_week"], model_type="vae")
vae.fit(df)
```

`half_life_days` defaults to an auto-scaled value per context (`span_days / 3.5`) if omitted.
Validated on `sample_df.pkl`: recency weighting did **not** reduce the energy bias it was
designed to fix in the currently-relevant history window (see `REFACTOR_STATE.md`, Session 2,
for the full honest write-up) — it's available and tested, not defaulted-on, and not
recommended without re-validating on the specific window you care about.

The VAE (`gears/models/vae.py`) fixed a real variance-collapse bug (Session 3) and is
competitive with the GMM on every configuration tested, but neither consistently beats the
persistence-bootstrap baseline — see the 4-arm benchmark below for real numbers, not just this
section's summary.

---

## Fitting your own GMM / VAE bundle with `fit_gmm.py`

```bash
# Quickstart — GMM, all defaults
python scripts/fit_gmm.py --input /path/to/sessions.pkl

# CSV with custom BIC range
python scripts/fit_gmm.py \
    --input /data/france_ev.csv \
    --year 2025 \
    --max-samples 5000 \
    --n-components auto \
    --max-components 10 \
    --output-dir gears/data/gmm

# Recency-weighted GMM
python scripts/fit_gmm.py --input /path/to/sessions.pkl --recency --half-life-days 21

# VAE instead of GMM
python scripts/fit_gmm.py --input /path/to/sessions.pkl --model-type vae

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

| Column | Type | Description |
|---|---|---|
| `arrival_time` | `datetime` | Plug-in timestamp |
| `duration` | `float` | Session duration in **hours** |
| `energy` | `float` | Energy delivered in **kWh** |

Optional but used for stratification: `location_type` (`work`/`home`/`public`/`heavy`),
`department` (INSEE code, e.g. `'75'`). Common column name aliases are auto-detected by
`load_sessions()` — see `gears/data/schemas.py`'s `COLUMN_ALIASES` for the full list.

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
from gears import load_sessions, GEARSModel
from gears.smart_charging.optimizer import SmartChargingOptimizer
from gears.utils import make_price_signal

df = load_sessions("data/sessions.csv")
model = GEARSModel(forecaster_method="probabilistic").fit(df, verbose=False)
sessions_df = model.simulate_short_term(start_date="2025-10-01", horizon=14, n_scenarios=1, seed=0)

signal = make_price_signal(start="2025-10-01", periods=48 * 14, resolution_min=30)
opt = SmartChargingOptimizer(signal_type="price")
result = opt.optimise(sessions_df, signal)
print(result[["arrival_time", "energy", "scheduled_end", "savings_pct"]].head())
```

---

## Benchmark: persistence vs. GMM vs. recency-GMM vs. VAE

`gears/evaluation/benchmark.py` runs a rolling-origin evaluation (Wasserstein distance + CRPS
+ load-profile NRMSE) across all four session-generation arms. `run_benchmark.py` is the CLI:

```bash
# All 4 arms, all 11 public datasets except the acn union (avoids triple-counting
# caltech+jpl+office — see data/README.md)
python run_benchmark.py --arms persistence,gmm,gmm_recency,vae

# Single dataset, quick grid
python run_benchmark.py --dataset office --quick --arms persistence,gmm
```

Results are cached by a config hash (`results/benchmark_cache/`, see `gears/evaluation/cache.py`)
so notebook 4 loads instantly by default; set `RERUN_BENCHMARK = True` at the top of the
notebook to force a fresh run. Headline result from the last real 8-dataset run (see
`REFACTOR_STATE.md`, Session 4, for the full numbers and caveats): **persistence wins the large
majority of paired cells** (~70% win-rate); the gap to persistence narrows as the history
window `X` grows for all model-based arms, but none overtakes it into a majority-win position
in this benchmark's design. Reported as measured, not tuned to a preferred conclusion.

---

## Tests

```bash
python -m pytest tests/ -v --cov=gears
# 291 passed, 10 skipped, 301 collected (0 failed)
```

`ruff check gears/ tests/` returns 0 errors. The 10 skips are pre-existing
`neuralforecast`/`dl`-extra skips, unrelated to VAE/GMM/CLI (see `REFACTOR_STATE.md`).

---

## Data

See [`data/README.md`](data/README.md) for the 11 public benchmark datasets, the
`data/custom/` bring-your-own-data path, and the two GEARS reference archives
(`preprocessed_data.zip`, `sample_df.pkl`) used throughout this README and the notebooks.

---

## Package Structure

```
gears/
├── data/           loader, schemas, INSEE helpers, pre-fitted GMM/VAE bundles
├── models/         EVSessionGMM (GMM + VAE + recency), forecasters, registry, get_gmm()
├── evaluation/      benchmark harness (4 arms), results cache, rolling-origin windowing
├── simulation/     ShortTermSimulator, MediumTermSimulator (linear/s_curve/bass)
├── smart_charging/ SmartChargingOptimizer (V1G)
├── output/         aggregators, exporters
├── cli.py          `gears` command (fit / simulate / medium-term / smart-charge / list-models)
├── plotting.py
└── utils.py
scripts/
└── fit_gmm.py      — fit GMM/VAE bundles on your own data
run_benchmark.py    — CLI for the persistence/GMM/recency-GMM/VAE benchmark
notebooks/
├── 1_gmm_descriptive.ipynb
├── 2_gmm_forecasting.ipynb
├── 3_gmm_scenarios.ipynb
├── 4_persistence_vs_gmm_benchmark.ipynb
└── 5_generic_dataset_example.ipynb
tests/              301 tests (291 passed, 10 skipped)
```

---

## Licence

MIT — see `LICENSE`.

# `data/` — raw datasets for GEARS

Nothing under `data/` is committed to the repo (see the root `.gitignore`
rule: only this README, the folder tree, and the `.gitkeep` placeholders are
tracked). This keeps the repo small; none of the datasets referenced below
are proprietary or confidential — they are simply left out to avoid bloat.

Every subfolder expects data already in (or convertible to) the columns
`gears.data.schemas` validates: `arrival_time`, `duration` (hours), `energy`
(kWh), plus the optional columns (`power`, `location_type`, `user_id`,
`department`). See the module docstring and `COLUMN_ALIASES` in
[`gears/data/schemas.py`](../gears/data/schemas.py) for the full canonical
schema and the raw-column-name aliases `load_sessions`/`validate_dataframe`
already recognise — this README intentionally does not duplicate that list,
to keep a single source of truth. In practice, the 11 public CSVs below
already load through `load_sessions()` without manual renaming, because
their native columns (`Start`, `Arrival`, `Energy`, ...) match those aliases.

## Public benchmark datasets (11)

Used by `run_benchmark.py` / `gears.evaluation.benchmark` (Session 3–4's
persistence/GMM/VAE rolling-origin benchmark). Download each raw source
below, preprocess it into the canonical schema (or reuse
`data/preprocessed_data/<name>.csv`, see "GEARS reference datasets" below —
those are the exact already-preprocessed versions this benchmark runs
against by default), and drop the resulting CSV at `data/<name>/<name>.csv`
(the harness expects one CSV per dataset name under
`data/preprocessed_data/`, see `run_benchmark.py`'s `--data-dir`).

Sources below were individually verified (not assumed) against each
provider's current public data portal.

### `acn/` — ACN-Data (Caltech + JPL + Office), union

`acn.csv` is the row-wise union of `caltech.csv` + `jpl.csv` + `office.csv`
below (see `gears/data/schemas.py`'s combined-site handling) — load it
*instead of*, never *alongside*, those three, or sessions get triple-counted.
Source: [ACN-Data, Caltech](https://ev.caltech.edu/dataset) (Caltech Adaptive
Charging Network — covers the Caltech, JPL, and one Office site).

### `caltech/`, `jpl/`, `office/` — ACN-Data (individual sites)

Same source as `acn/` above: [ACN-Data](https://ev.caltech.edu/dataset).
These three are the individual per-site breakdown of the same underlying
dataset (Caltech campus, NASA JPL, and one office/workplace site).

### `boulder/`

Source: [City of Boulder, CO — Electric Vehicle Charging Station Data](https://open-data.bouldercolorado.gov/maps/95992b3938be4622b07f0b05eba95d4c_0/explore)
(Boulder Open Data portal). A companion data dictionary is published
alongside it on the same portal.

### `domestics/`

Source: [Electric Chargepoint Analysis 2017: Domestics](https://www.gov.uk/government/statistics/electric-chargepoint-analysis-2017-domestics)
— UK Department for Transport (published via the Office for Low Emission
Vehicles' Local Authority Grant Fund), also mirrored on
[data.gov.uk](https://www.data.gov.uk/dataset/5438d88d-695b-4381-a5f2-6ea03bf3dcf0/electric-chargepoint-analysis-2017-domestics).
~3.2M domestic (residential) charging events across ~25,000 UK chargepoints,
2017.

### `dundee/`

Source: [Dundee City Council Open Data — EV Charging Data](https://data.dundeecity.gov.uk/dataset/activity/ev-charging-data)
(CKAN portal, published quarterly).

### `palo_alto/`

Source: [City of Palo Alto — Electric Vehicle Charging Station Usage (Jul 2011 – Dec 2020)](https://data.paloalto.gov/datasets/194693/electric-vehicle-charging-station-usage-july-2011-dec-2020/)
(Palo Alto Open Data portal). Note the portal's built-in export caps CSV
downloads at 10,000 rows — the portal's own "download full dataset" flow
(linked from the same page) is needed to get the complete history.

### `paris/`

Source: [Belib' — réseau parisien de recharge, sessions de charge](https://www.data.gouv.fr/datasets/belib-reseau-parisien-de-bornes-de-recharges-accelerees-22-kw-ac-dc-pour-vehicules-electriques)
(data.gouv.fr, Ville de Paris / Total Marketing France). Session-level data:
location, start/end time, energy delivered, connector type. Note
`EVAL_WINDOW_OVERRIDES` in `gears/evaluation/benchmark.py` shortens this
dataset's benchmark evaluation window (14d instead of the 30d default) — its
usable span is short.

### `perth/`

Source: [Perth & Kinross Council Open Data — EV Charging Data](https://data.pkc.gov.uk/dataset/ev-charging-data)
(ChargePlace Scotland scheme).

### `sap/`

Source: **no standalone public open-data portal was found for this one** —
flagging honestly rather than inventing a link. It was contributed directly
by SAP Labs France (workplace + home charging transactions for their EV
fleet, since June 2017) to the benchmark this package's 11-dataset harness
is built on. The closest available public copies are the raw mirror in
[`yvenn-amara/ev-load-open-data`](https://github.com/yvenn-amara/ev-load-open-data/tree/master/1.%20Input%20Data/8.%20SAP%20Labs%20France)
(same author as this repo) and general background on the programme at
[SAP Labs France](https://www.sap.com/france/about.sap-labs-france.html).
If a proper public portal for this dataset surfaces later, update this
section rather than leaving the GitHub mirror as the canonical pointer.

---

## Your own data (`custom/`)

Drop your own CSV (or Parquet — anything `pandas.read_csv`/`read_parquet` can open) under
`data/custom/` and point `load_sessions()` at it directly; no code changes needed as long
as the file's columns match (or alias to) the same canonical schema every dataset on this
page already goes through — the one `gears.data.schemas` validates:

| Column | Required? | Meaning |
|---|---|---|
| `arrival_time` | **required** | datetime of plug-in |
| `duration` | **required** | session duration in hours (> 0) |
| `energy` | **required** | energy delivered in kWh (≥ 0) |
| `power` | optional | charger power in kW |
| `location_type` | optional | one of `work`, `home`, `public`, `heavy` |
| `user_id` | optional | anonymised user identifier |
| `department` | optional | INSEE department code (French-specific; irrelevant for non-French data) |

You don't need your raw column names to match exactly — `load_sessions()` resolves common
aliases automatically (e.g. a raw `Start`/`Energy`/`Arrival` column layout, like the 11
public benchmark CSVs above, loads unmodified). See `COLUMN_ALIASES` and `REQUIRED_COLS` in
[`gears/data/schemas.py`](../gears/data/schemas.py) for the exact, authoritative list of
recognised aliases and required columns — this table intentionally doesn't duplicate it in
full, to keep one source of truth.

Once loaded, a `GEARSModel` fits on it exactly like any dataset on this page:

```python
from gears.data.loader import load_sessions
from gears.pipeline import GEARSModel

df = load_sessions("data/custom/my_sessions.csv")
model = GEARSModel().fit(df)
```

[`notebooks/5_generic_dataset_example.ipynb`](../notebooks/5_generic_dataset_example.ipynb)
demonstrates this end-to-end (data loading through V1G smart-charging optimisation) on
`sap.csv`, staged at `data/custom/` for the walkthrough — a single-site dataset with none
of the French-specific columns (`department`, `location_type`) sample_df.pkl has, closer to
what a first-time user's own file typically looks like. It also includes a quick
load-and-validate check on `domestics.csv` (UK residential charging — a genuinely different
usage profile from `sap.csv`'s workplace+home fleet) to show the same schema handles both
without any code changes.

## GEARS reference datasets

Two archives hosted for this package specifically (not part of the 11-dataset
public benchmark above) — both open-source, not proprietary; they're kept out
of the repo purely to avoid bloating it, not because of any access
restriction:

- **`preprocessed_data.zip`** —
  <https://www.yvenn-amara.com/wp-content/uploads/2026/07/preprocessed_data.zip>.
  Contains the already-preprocessed versions of the 11 public benchmark
  datasets above (one CSV per dataset, canonical-schema-ready), light and
  varied across usage types. **Default choice** for fast iteration: dev
  loops, exploratory diagnostics, and most of the benchmark/VAE work
  (Sessions 3–4) — including this benchmark's default `--data-dir`.
- **`sample_df.zip`** (contains `sample_df.pkl`) —
  <https://www.yvenn-amara.com/wp-content/uploads/2026/07/sample_df.zip>.
  ~3,000,000 real EV charging sessions, France only, all usage types
  combined (residential, workplace, public). Heavier, more complete.
  **Reserved for official measurements** explicitly listed in a session's
  acceptance criteria (e.g. the X=52 energy-bias check, final VAE win-rate,
  the 4-arm benchmark's real-data validation, notebook 1/2/3 runs, README
  examples) — not for routine iteration.

Download both with:

```bash
mkdir -p data
curl -L -o data/preprocessed_data.zip https://www.yvenn-amara.com/wp-content/uploads/2026/07/preprocessed_data.zip
unzip -oq data/preprocessed_data.zip -d data/preprocessed/
curl -L -o data/sample_df.zip https://www.yvenn-amara.com/wp-content/uploads/2026/07/sample_df.zip
unzip -oq data/sample_df.zip -d data/
```

(Adjust the extraction paths if the real zip layout nests files in an
unexpected subfolder — both land under `data/`, already covered by the
`.gitignore` rule above.)

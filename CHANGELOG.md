# Changelog

All notable changes to GEARS are documented here. Dates are when each change was made,
not necessarily when it lands on `main` (this repo uses a branch-and-PR workflow).

Entries up to 2.0.0 mention design documents (`REFACTOR_STATE.md`, `AUDIT.md`,
`PROPOSAL_*.md`, `INVESTIGATION_*.md`), scripts and notebooks that were removed from the
repository during the clean-up; they are still in the git history at the `pre-cleanup` tag.

## [Unreleased]

### Breaking — removed from the public API (S2–S4)
Experiment code moved to the private `GEARS_sandbox` repository: `DepartmentForecaster`,
`aggregate_by_department`, `build_panel`, `crps_ensemble`, `PersistenceSessionSampler`,
`run_benchmark_for_datasets`, `run_rolling_origin_benchmark`, `run_sarima_sanity_check`,
`summarize_sarima_sanity_check`, `sessions_in_last_n_occurrences`.
Pre-trained models: `ModelRegistry`, `NativeSessionModelRegistry`, `get_session_model`,
`GEARSModel.from_native_gmm` and the Hugging Face hub are replaced by
`gears.load_pretrained("french")` / `GEARSModel.from_pretrained("french")` /
`GEARSModel.from_session_model(...)`; nothing is downloaded at `import gears` any more.
`NHiTSForecaster` / `TransformerForecaster` now live in `gears.models.forecaster_dl`
(import them from `gears.models`).

### Changed (S5)
- **CLI**: only `gears fit` and `gears simulate` remain. Removed `medium-term`,
  `smart-charge`, `list-models` and `fit --gear`. Added `fit --stratify-by`,
  `--max-components`, `--max-samples-per-context`, `--seed`. `simulate` requires exactly
  one of `--model` / `--pretrained` (previously `--pretrained` silently won).
- `GEARSModel` / `Gear1Backend` accept `max_components`; `GEARSModel(gear=n)` with
  `n != 1` raises a one-line `NotImplementedError`.
- **Logging**: the library no longer calls `print`. `verbose=True` now shows `INFO` messages
  on stderr (previously stdout) through a temporary console handler; the per-context
  "Fitted GMM" line is `DEBUG`, with one `INFO` summary.
- `OutputAggregator` (`output/aggregator.py`): 982 → 586 lines by sharing the sampling and
  annual-series code; `build_load_profiles` output is unchanged (bit-identical on fixed seeds).
- `plot_forecast` of `SessionForecaster`, `TransformerForecaster` and `NHiTSForecaster`
  delegates to the new `gears.plotting.plot_forecast_fan`.

### Removed (S5)
- Unused methods: `EVSessionModel.plot_components`, `MediumTermSimulator.plot_energy_trajectory`,
  `ShortTermSimulator.compute_load_curve`, `SmartChargingOptimizer.plot_load_curve`.
- `plot_session_heatmap`: the `x` and `y` parameters were never used.

### Added (S2–S4 fixes)
- `load_pretrained(model_id, path=..., cache_dir=...)`, `GEARSModel.from_pretrained(..., path=...)`:
  load the official file from a local path on a machine without internet access.
- `GEARS_HOME` (cache folder, default `~/.cache/gears`) and `GEARS_OFFLINE=1` (never use the
  network; a cached model is still used, otherwise `PretrainedModelError` says what to do).
- `load_pretrained` warns, once and with the fix, when the installed scikit-learn differs from
  the one the model was trained with. `EVSessionModel.fit` now records `gears_version` and
  `sklearn_version` in `metadata_`.
- `gears._optional.require`: one `ImportError` for every optional dependency
  (`pip install "gears-ev[vae]"`, `pip install "gears-ev[dl]"`).

### Changed (S2–S4 fixes)
- **`import gears` is lazy** (PEP 562): a few milliseconds instead of ~1.2 s, and it no longer imports
  pandas, scikit-learn or any submodule. `gears.<name>`, `from gears import <name>` and
  `dir(gears)` behave as before; `gears.models` is lazy too. `gears.__all__` is unchanged
  (20 names).
- `load_pretrained`: the SHA-256 of `gmm_french.joblib` is now the real one (it was the hash of an
  empty file, so every call failed the integrity check); a download is verified **before** it is
  moved into the cache, so a corrupt or truncated file never lands there; the checksum also
  applies to `path=`; the loaded object must be an `EVSessionModel`; the repository and release
  tag live in `gears.pretrained.REPO` / `RELEASE_TAG`.
- `data/README.md` reduced to the expected format and "put your files here".

### Fixed
- `pretrained._download_file` leaves no temporary file behind when a download fails, including
  truncated transfers (`http.client.IncompleteRead`).
- Install hints named an extra that does not exist (`gears-ev[transformer]`); they now say `[dl]`.
- Tests: `tests/test_forecaster.py` imports again (DL forecasters moved in S3);
  `tests/test_pretrained.py` mocked a function the code does not call.
- 12 lines of double-encoded UTF-8 in `gears/models/forecaster_dl.py` docstrings.
- Comments and docstrings no longer point to the design documents deleted in S2.

### Added (S6 — sanity checks and error messages)
- **Data checks** (`gears.data.schemas.validate_dataframe`, so also `load_sessions` and every
  `fit`): an empty dataset or fewer than 10 sessions raises a `ValueError` naming the minimum
  (it used to end in `RuntimeError: No GMM was fitted`); rows with `duration <= 0`, a negative or
  missing energy, `duration >= 168 h`, `energy >= 400 kWh` or an exact duplicate are dropped
  and **counted by reason in one warning**; if nothing is left, a `ValueError` says what to
  check; `strict=True` raises instead of dropping.
- **Units**: a median energy above 200 or a median duration above 48 triggers a warning that
  states the heuristic ("probably Wh" / "probably minutes") and the fix.
- **Time zones**: the convention is documented (`gears.data.schemas`, `gears._validation`,
  `data/README.md`) and tested: GEARS works in local wall-clock time; a zone or UTC offset in
  `arrival_time` or in `start_date` is **dropped, not converted**, and `arrival_time` is announced
  by a warning. Simulated `arrival_time` values are naive local timestamps.
- **Contexts with few sessions**: contexts with fewer than 10 sessions get no model of their own;
  one summary warning at `fit` gives their number and share of the data, they are listed in
  `EVSessionModel.skipped_contexts_` and in `GEARSModel.summary()`.
- **Unknown context at simulation time**: `EVSessionModel.sample(..., strict=False)` warns **once
  per model**, naming the requested context and the one used, and records every substitution in
  `fallback_contexts_`; `strict=True` raises a `ValueError`. `strict` is also accepted by
  `ShortTermSimulator.simulate / simulate_single_day` and `GEARSModel.simulate_short_term`.
- **Known limitation, now visible**: a date alone only sets `day_of_week` / `season` / `is_weekend`.
  On a model that is also stratified by `location_type` / `department` (the pre-trained French
  model: `location_type`, `department`, `day_of_week`, `season`), the requested context does not
  exist and the **most populous context overall** is used whatever the date. This was silent;
  it now triggers the fallback warning (so `gears simulate --pretrained french` prints it once).
  The selection rule itself is unchanged and is left for a decision.
- `tests/test_sanity.py`: one test per case of the plan's table and the ten output invariants.

### Changed (S6)
- **Argument errors are `ValueError`s that say what was received and what is valid**:
  `horizon`, `n_scenarios`, `n_sessions` (integers >= 1; `2.5` and `True` are refused),
  `start_date` / `date` (invalid or missing: a `KeyError: 'date'` or a pandas message before),
  `years` (> 0), `annual_growth_rate` (finite, > -1), `output`, `forecaster_method`
  (now checked in `GEARSModel(...)`, not after a long fit), `context` (must be a dict) and the
  `SmartChargingOptimizer` constructor arguments.
- **Using an unfitted object raises `sklearn.exceptions.NotFittedError`** (a `ValueError` and an
  `AttributeError`, not a `RuntimeError`) with the way forward, for `GEARSModel`, `EVSessionModel`
  and the four forecasters.
- **SARIMA is always tried, and never degrades silently**: whatever the length of the history.
  If it cannot be fitted (pmdarima refuses histories below about two weeks) or the fit diverges
  (the forecast over one season exceeds 3x the largest observed daily count), a warning gives the
  reason and the forecaster switches to `'probabilistic'` (`forecaster.method_` tells which is
  used; `method` stays what was asked). If `pmdarima` cannot be imported (any error, not only
  `ImportError`) it warns and fits the fixed statsmodels SARIMAX(1,1,1)x(1,1,0,7); a failing
  `predict` warns once before falling back to the historical mean. A history of a single day no
  longer gives `NaN` counts.
- `simulate_short_term(n_sessions=...)`: the per-day seeds are now drawn from `seed` instead of
  computed as `seed + day`, which made day *d* of seed *s* identical to day 0 of seed *s + d*
  whenever the weekday and season matched, and `seed=None` is now random (it was silently 0).
  **Simulated values differ from earlier versions for the same seed in this path.**
- **`SmartChargingOptimizer` is consistent with the simulator** (**costs and savings differ from
  earlier versions**). It assumed a charging efficiency of 0.9 on top of the session's `energy`,
  while GEARS caps simulated energy at `power_kw x duration`: sessions close to the cap could not
  be completed (on simulated `public` data, 24 % of them) and were truncated silently. Now:
  `efficiency` defaults to 1.0 (a session's `energy` is the energy delivered by the charger, losses
  included; pass e.g. 0.9 to add losses on top) and the energy is allocated slot by slot: a slot only
  partly inside the plug-in window offers only the time inside it, the last slot used is only partly
  filled, and a session is billed for the energy it receives and nothing else (before: whole slots,
  energy / 0.9). Absolute costs are 20-25 % lower on simulated data, savings percentages
  move by a few points (work 24.8 -> 24.5 %, home 23.2 -> 27.5 %, public 6.0 -> 7.0 %), and the
  optimiser is about 7x faster. `scheduled_start` / `scheduled_end` are now the first and last
  instants of actual charging, always inside the plug-in window. New column `energy_scheduled`
  (kWh that reach the vehicle: equals `energy` unless the session cannot be completed).
  `OutputAggregator.build_load_profiles(..., smart_charging_signal=...)` uses the optimiser, so its
  `ts_smart` series changes slightly.
- `SmartChargingOptimizer.optimise` warns about sessions it could not schedule (no usable arrival,
  energy or duration; no signal value during the plug-in window - their costs stay `NaN`), about
  sessions that cannot be completed, and about a signal whose step is not `resolution_min`; it
  raises a `TypeError` for a signal without a timestamp index and a `ValueError` for `power_kw <= 0`.
- `linear_growth_profile` is clamped at 0: a negative growth rate used to produce negative
  expected sessions.
- `n_scenarios=0` is no longer replaced by the default (`n_scenarios or default`).

### Added (S7 — a short, useful test suite)
- **pytest markers** `vae` (needs torch), `dl` (needs torch and neuralforecast) and `slow`, declared in
  `pyproject.toml` with `--strict-markers`. A plain `pytest` runs `-m "not dl and not slow"`;
  `pytest -m vae` and `pytest -m dl` select one family. The modules that need an extra start with
  `pytest.importorskip(...)`: without torch they are skipped instead of failing (14 failures before).
- `tests/conftest.py`: any internet connection or DNS lookup raises `NetworkAccessError`; every test
  runs with a private `GEARS_HOME` and without `GEARS_OFFLINE` (the real `~/.cache/gears` is never
  touched, and an exported `GEARS_OFFLINE=1` does not change the results); two read-only
  session fixtures (`demo`, `model`) replace the per-file fixtures.
- `tests/test_vae.py` (8 tests, `-m vae`) and `tests/test_forecaster_dl.py` (2 tests, `-m dl`, run in a
  temporary folder: no more `lightning_logs/` in the repository).
- Coverage: `[tool.coverage]` (branch coverage; `models/vae.py` and `models/forecaster_dl.py` are left
  to the extras jobs) and a floor `fail_under = 89`, applied by `pytest --cov`. Measured: 91 % without
  torch, 93 % with torch.

### Changed (S7)
- **361 test functions (416 collected) → 127 functions (131 collected)**; without torch, 122 tests run in
  about 45 s (before: 14 failures, 381 passed, 105 s; with torch 397 passed, 19 skipped, 209 s). Each
  deleted test is accounted for in the pull request (merged, redundant, moved to the sandbox).
- Tests check hand-computed values or independent recomputations (a scikit-learn BIC search, a
  re-derived optimiser cost, the calendar of twelve months...) instead of "it returned something".
  A mutation campaign (90 changes to the code, each of which must make a test fail) checked that:
  85 are caught, the 5 others are redundant protections or only visible from a developer's shell.
- `tests/test_sanity.py` is now 31 tests: the plan's table and the ten output invariants, regrouped by
  behaviour.

### Fixed (S7)
- `OutputAggregator.build_load_profiles` raises a `ValueError` when `gmm.stratify_by` lacks
  `day_of_week` or `season`. It used to return a whole year of **zero** load without a word (no context
  was ever selected); the tests of this function were passing on that all-zero series.
- The unseeded CLI test `test_simulate_cli_writes_sessions` failed intermittently; it is seeded.

### Added (S8 — toy examples)
- `examples/`: four scripts that run on a laptop, each in under 20 s, without any external data and
  without a network (`MPLBACKEND=Agg` is enough; figures and files go to `example_outputs/`, or to
  `$GEARS_EXAMPLES_OUT`):
  `01_quickstart.py` (demo data → `fit` → 7-day simulation → daily energy and hourly load → PNG),
  `02_forecast_and_smart_charging.py` (probabilistic, persistence and SARIMA forecasters scored with
  `forecast_metrics`; V1G on a `make_price_signal` tariff; `compute_regret`),
  `03_medium_term_scenarios.py` (`MediumTermSimulator` with the linear, S-curve and Bass growth models;
  the `CHARGER_PRESETS`), `04_bring_your_own_data.py` (CSV → `load_sessions` → `fit` → `save` → `load` →
  `simulate` → `export`; `--pretrained` runs the same with `GEARSModel.from_pretrained("french")`, which
  needs the network or a local file).
- `tests/test_examples.py` (5 tests, marker `slow`: `pytest -m slow tests/test_examples.py`, about 30 s):
  runs every example with `runpy`, checks that each writes its figure or files, and that none raises a
  `UserWarning`. A broken example fails the tests. `--pretrained` is tested on a fake local release.
- `.gitignore`: `example_outputs/`.

### Changed (S8)
- `gears.plotting` draws with matplotlib directly instead of pandas' `.plot(ax=...)`: pandas registered
  its own date converter and every plot of a date axis emitted "This axis already has a converter set
  and is updating to a potentially incompatible converter" (a `UserWarning` on a standard use of the
  library).
- `plot_regret_comparison` leaves room above the tallest bar for the value labels.
- `tests/test_plotting.py`: 3 tests for the 4 remaining functions.

### Removed (S8)
- Plotting functions that no example uses: `plot_arrival_distribution`, `plot_session_heatmap`,
  `plot_energy_distribution`, `plot_gmm_means`, and the `DAY_LABELS` constant. `plot_daily_energy`,
  `plot_forecast_vs_actual`, `plot_forecast_fan` (behind `plot_forecast`) and `plot_regret_comparison`
  stay.

## [2.0.0] — 2026-08 Phase 2 complete: GEAR architecture, naming, VAE integration, notebook overhaul, CI health

Release-note-level summary of the full Phase 2 arc (11 sessions, 2026-08-02 → 2026-08-07).
Per-session detail with real measured numbers is in the dated sub-entries below and in
`REFACTOR_STATE.md`; this section is the rollup a new user or reviewer should read first.

### Architecture and naming
- `GEARSModel(gear=1..5)`: explicit gear-dispatch facade over a new `Gear1Backend`, with
  GEAR 2nd–5th reserved (raise `NotImplementedError` by name, not silently wrong). See the
  `[2.0.0] GEAR-level architecture` entry below for the full rename list.
- Clean-break naming pass across the public API (`EVSessionGMM`→`EVSessionModel`,
  `NativeGMMRegistry`→`NativeSessionModelRegistry`, `get_gmm()`→`get_session_model()` — the
  latter also fixed a real bug where the old function silently ignored all four of its
  arguments). Full list in the `[2.0.0] naming consistency pass` entry below.

### Model integration
- The real, curated VAE bundle (`vae_french_sample.joblib`, 691,157 sessions, 516 contexts,
  departments 59/69/78/92/93) is confirmed live on the `models-v1` GitHub Release and used
  by default — closes the `[1.1.0]` release's last known gap (previously a synthetic
  fallback with no département dimension).
- Recency-weighted and holdout-variant GMM fitting scripts (`--recency`, `--half-life-days`,
  `--exclude-last-n-days`) wired into a worked notebook example. One caveat carried
  forward, not silently hidden: the specific `gmm_french_recency.joblib` artifact currently
  on the release doesn't actually have recency weighting applied (see Session 11 below) —
  the recency *feature* itself is implemented and tested, but this one fitted artifact
  needs re-running.
- Recency weighting, once validated on real data, did not reduce the targeted energy bias
  in the tested history window — implemented, tested, and available, not defaulted on.
- 4-arm benchmark (persistence / GMM / recency-GMM / VAE): persistence remains the
  strongest arm on the primary Wasserstein metric; a follow-up investigation into whether a
  shared (not per-cell) VAE closes the gap found a partial result — real wins on
  profile-NRMSE, not on the primary metric, and at higher compute cost, not lower. Full
  numbers in `INVESTIGATION_PERSISTENCE_GAP.md`.

### Notebooks and documentation
- All 5 notebooks rebuilt or overhauled to run end-to-end in well under 5 minutes each
  against real data, with consistent GMM/VAE color-coding, department-matched comparisons
  where population scale matters, and neutral (non-comparative) language.
- Full French-to-English translation pass across notebooks and the workshop-adjacent
  scripts.
- README rewritten against the actual current package, every snippet executed before being
  written down, with a new "GEAR levels" section.

### CI and quality
- `ruff check gears/ tests/`: 0 errors (from 274 at the pre-refactor baseline).
- Test suite: 377 collected, 369 passing, 8 skipped (optional `[dl]` extra only) against
  real data and the real release-fetched model bundles. Coverage materially improved on
  previously weak modules (`cli.py`, `aggregator.py`, `plotting.py`).

### Known, deliberately-not-resolved-here
- The persistence-vs-model-family investigation above is informative, not conclusive;
  formal harness integration of a shared-VAE arm is a future session's call.
- `gmm_french_recency.joblib` needs re-fitting with `--recency` actually passed to be a
  correct illustration of the feature it's named for.
- Whether this repo should be public remains an open decision, carried since Phase 1.

---

## [2.0.0] — 2026-08 real bundle integration + Phase 2 close-out (Session 11)

### Added
- `notebooks/2_gmm_forecasting.ipynb` §7.3: wires `gmm_french_recency.joblib` and
  `gmm_french_holdout.joblib` (previously unreferenced anywhere in the codebase) into a
  worked example, loaded via `EVSessionModel.load(path)` and scored against `eval_df`
  alongside the production `french` GMM.

### Verified (no code change, confirmed by direct inspection)
- The real, curated `vae_french_sample.joblib` (691,157 sessions, 516 contexts, departments
  59/69/78/92/93) is live on the `models-v1` GitHub Release and used by default —
  `NativeSessionModelRegistry` no longer falls back to the synthetic demo bundle. This
  closes the last "known gap" carried in the `[1.1.0]` entry below.
- Full test suite (369 passed / 8 skipped, `[dl]`-extra only / 0 failed) and all 5 notebooks
  (96s/191s/198s/40s/41s, 0 errors) re-run end-to-end against real data.

### Found, documented, not fixed this session
- `gmm_french_recency.joblib` is not actually recency-weighted (`.recency=False`,
  `.half_life_days=None` on the loaded object) despite its name — the local fit that
  produced this specific file appears not to have had `--recency` passed. See
  `REFACTOR_STATE.md`, Session 11.
- A newer `ruff` flags a pre-existing notebook-schema issue in
  `scripts/compare_external.ipynb`; outside CI's linted path (`gears/ tests/`), not
  introduced this session.

## [2.0.0] — 2026-08 GEAR-level architecture (Phase 2 / Session 3)

Implements the gear-dispatch design from `PROPOSAL_GEAR_ARCHITECTURE.md`.

### Added
- `GEARSModel(gear=1, ...)`: `gear` is now an explicit constructor parameter. `1` (the
  default) is the only implemented GEAR today — the current GMM/VAE session modeling +
  SARIMA/probabilistic forecasting + simulation + smart-charging pipeline, unchanged in
  behavior. `gear=2..5` raise `NotImplementedError` naming GEAR 1st as the working
  alternative, rather than silently doing the wrong thing.
- `model_type`, `recency`, and `half_life_days` are now first-class `GEARSModel`
  constructor parameters, forwarded straight through to `EVSessionModel`. Previously these
  were only reachable by constructing `EVSessionModel` directly, bypassing the unified
  facade entirely.
- CLI: `gears fit` gains `--gear`, `--model-type` (`gmm`/`vae`), `--recency/--no-recency`,
  and `--half-life-days`, for parity with the Python API.

### Internal
- `GEARSModel` is now a thin, gear-dispatching facade (`gears/pipeline.py`) over
  `Gear1Backend` (`gears/pipeline_gears/gear1.py`), which is the pre-2.0.0 `GEARSModel`
  class body moved essentially as-is. No behavior change for any GEAR-1 caller — every
  pre-existing test in `test_pipeline.py`/`test_registry.py` passes unmodified against the
  new facade.

## [2.0.0] — 2026-08 naming consistency pass (Phase 2 / Session 2)

Clean-break rename — no deprecated aliases. See `PROPOSAL_NAMING.md` for the full
rationale behind every renamed symbol below.

### Breaking changes
- `EVSessionGMM` → `EVSessionModel` (`gears/models/gmm.py` → `gears/models/session_model.py`)
- `NativeGMMRegistry` → `NativeSessionModelRegistry`
- `gmm_id` → `session_model_id` (param, used throughout the registry, CLI, and tests)
- `get_gmm(location_type, departement, saison, day_of_week)` → `get_session_model(bundle_id="french")`.
  This is more than a rename: the old signature accepted four arguments and silently ignored
  all of them, always returning the `"french"` bundle regardless — a real bug, not just a
  naming issue. The new signature matches what the function actually does; stratum-level
  lookups now happen afterward via `.get_sklearn_component(context=...)` on the returned object.
  As a side effect, `get_session_model("french_vae_sample")` now works as a direct shortcut for
  the VAE bundle too, which the old `get_gmm()` could not do.
- `get_sklearn_gmm()` → `get_sklearn_component()` (on `EVSessionModel` and on the registry's
  stratum-level convenience method)
- `departement`, `saison` → `department`, `season` (params on `get_sklearn_component()`)
- `gears/data/gmm/` → `gears/data/session_models/` (package data directory; the two shipped
  `.joblib` bundles were re-pickled under the new class path — the old bundles would not have
  unpickled under the renamed class)
- `scripts/fit_gmm.py` → `scripts/fit_session_model.py`
- `gears-fit-gmm` console script → `gears-fit-session-model`
- `list_gmms()` / `args.list_gmms` → `list_session_models()` / `args.list_models`
- `notebooks/4_persistence_vs_gmm_benchmark.ipynb` → `notebooks/4_persistence_vs_session_model_benchmark.ipynb`
  (notebooks 1–3 keep their filenames — see `PROPOSAL_NAMING.md` for why)

### Fixed
- `NativeSessionModelRegistry`'s internal `_GMM_DIR` constant and `gmm_dir` constructor
  parameter (and the same on `GEARSModel.from_native_gmm()`) now consistently point at and
  are named after the renamed `session_models/` directory.
- `scripts/fit_session_model.py`'s `--output-dir` default still pointed at the old
  `gears/data/gmm` path — updated to `gears/data/session_models`.

### Not changed (deliberately, see `PROPOSAL_NAMING.md`)
- `ModelRegistry` and its `model_id` (the separate, already-generic HF-Hub-backed registry)
- `GEARSModel.from_native_gmm()`'s own method name and `self.gmm_` attribute — part of the
  GEAR-level facade, in scope for a later session
- `n_components` / `--n-components` (GMM-specific by design)

## [1.1.0] — 2026-07 refactor (Sessions 1–7)

### Added
- **VAE session model** (`EVSessionGMM(model_type="vae")`): a Conditional VAE alternative
  to the default GMM, sharing the same fit/sample interface. Fixed a real variance-collapse
  bug (sampling now includes the learned observation noise, not just the decoder mean).
  Competitive with the GMM on every tested configuration; does not consistently beat the
  persistence-bootstrap baseline — see the 4-arm benchmark.
- **Recency-weighted GMM** (`EVSessionGMM(recency=True, half_life_days=...)`): half-life
  exponential-decay weighted bootstrap resampling, opt-in, `recency=None`/omitted is
  byte-identical to the pre-refactor fit. Validated on real data; did not reduce the
  targeted energy bias in the currently-relevant history window — available and tested,
  not defaulted on.
- **4-arm benchmark harness**: `persistence`, `gmm`, `gmm_recency`, and `vae` are now
  first-class arms in `gears/evaluation/benchmark.py`, selectable via `run_benchmark.py
  --arms`, with a config-hashed results cache (`gears/evaluation/cache.py`) so repeated
  notebook runs load instantly.
- **`data/` directory** with the 11 public benchmark datasets + `data/custom/` (bring
  your own data), documented in `data/README.md` with individually-verified source links.
- **`gears` CLI** (`gears fit / simulate / medium-term / smart-charge / list-models`):
  the command was fully implemented in `gears/cli.py` but never registered as a console
  script — fixed in `pyproject.toml`; each subcommand verified end-to-end against real
  data while writing this release's README.
- Notebook 5's bring-your-own-data path completed (`data/custom/`, schema-portability
  check against a second real dataset).

### Fixed
- CI: `ruff check` 274 → 0 errors; `torch` moved into the `dev` extra so VAE tests run
  for real in CI instead of failing with `ModuleNotFoundError`.
- `EVSessionGMM` failed to unpickle bundles fitted before the recency feature was added
  (`AttributeError` in `__repr__`); added backward-compatible `__setstate__`.
- Medium-term growth models (`gears/simulation/medium_term.py`): fixed a `t=0`
  discontinuity in `s_curve`/`bass` profiles, saturation timing that didn't scale with
  the requested horizon, and a genuinely-exponential `linear_growth_profile` that wasn't
  linear. Reduced the exposed surface from 5 profiles to 3 (`linear`, `s_curve`, `bass`),
  each justified and each now mapping 1:1 to notebook 3's three scenarios.
- `DepartmentForecaster._forecast_dept`'s forecast-interval noise scale was pinned to a
  near-zero floor (~2% CI width regardless of actual data variance); ported the correct
  scale already used elsewhere in the codebase.
- `plot_lt_trajectories` silently hard-clipped any trajectory growing past 10x its
  anchor value — this, not the growth-model bugs alone, was the direct cause of the
  "plateau" notebook 3 was rebuilt to fix. Clip removed.
- Notebook 3's GMM-vs-VAE smart-charging comparison crashed (`KeyError: 'department'`)
  against the VAE's synthetic fallback bundle, which has no département stratification —
  first surfaced this session once real data + torch were both available. Fixed with an
  explicit fallback-aware filter and a printed caveat, consistent with the same known gap
  already documented in notebooks 1 and 2.
- `data/README.md`'s own download snippet extracted `preprocessed_data.zip` to a path
  (`data/preprocessed/`) that didn't match what `run_benchmark.py` actually reads
  (`data/preprocessed_data/`) — fixed and re-verified against a real extraction.
- Stale `work_fr_demo` example/docstring references updated to the real catalogue entry,
  `french_demo`.

### Changed
- All 5 notebooks rebuilt or cleaned up to run end-to-end in under 5 minutes each, with
  explicit, documented subsampling of the ~3M-row national dataset where used, and
  neutral (non-comparative) language between GMM/VAE/persistence in markdown cells.
- README rewritten from scratch against the actual current package (every code snippet
  in it was executed, not just read, while writing this release).

### Known gaps (not resolved this refactor — see `REFACTOR_STATE.md`'s release checklist)
- The real curated VAE bundle (`gmm_vae_french_sample.joblib`) is not committed; the
  registry transparently falls back to a small synthetic demo with no département
  dimension.
- The 4-arm benchmark's public-dataset run excludes `domestics.csv`/`palo_alto.csv`
  (VAE fit time not tractable in the sandbox this was developed in) and uses a reduced
  grid, not the full literal spec.
- Whether this repo should be public, and which version of notebook 4's language
  (explicit-comparison vs. neutral) should ship, are both open decisions for the
  maintainer — not decided by this refactor.

## [1.0.0] — pre-refactor baseline

Initial state audited at the start of this refactor (see `AUDIT.md`, Session 0):
GMM-based session generation, SARIMA/NHiTS/persistence forecasting, medium/long-term
growth simulation, V1G smart charging, 3 notebooks, broken CI (274 lint errors, 12
failing tests).

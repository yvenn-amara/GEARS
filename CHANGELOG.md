# Changelog

All notable changes to GEARS are documented here. Dates are when each change was made,
not necessarily when it lands on `main` (this repo uses a branch-and-PR workflow; see
`REFACTOR_STATE.md` for full session-by-session detail, real measured numbers, and
honestly-reported gaps behind every line below).

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

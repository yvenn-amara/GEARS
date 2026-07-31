# Changelog

All notable changes to GEARS are documented here. Dates are when each change was made,
not necessarily when it lands on `main` (this repo uses a branch-and-PR workflow; see
`REFACTOR_STATE.md` for full session-by-session detail, real measured numbers, and
honestly-reported gaps behind every line below).

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

# Changelog

[Keep a Changelog](https://keepachangelog.com) format, [SemVer](https://semver.org) versions. 2.0.0 is the
first public release. The detailed notes of the clean-up (with measurements) and of the earlier internal
versions are in `docs/archive/CHANGELOG_detailed.md` of the private `GEARS_sandbox` repository.

## [Unreleased] — becomes 2.0.0

### Breaking
- Experiment code moved to the sandbox: `DepartmentForecaster`, `aggregate_by_department`, `build_panel`,
  `crps_ensemble`, `PersistenceSessionSampler`, the benchmark and rolling-origin functions,
  `sessions_in_last_n_occurrences`.
- Pre-trained models: `ModelRegistry`, `NativeSessionModelRegistry`, `get_session_model`,
  `GEARSModel.from_native_gmm` and the Hugging Face hub are replaced by `load_pretrained("french")`,
  `GEARSModel.from_pretrained("french")` and `GEARSModel.from_session_model(...)`.
- Command line reduced to `gears fit` and `gears simulate` (removed: `medium-term`, `smart-charge`,
  `list-models`, `fit --gear`); `simulate` needs exactly one of `--model` / `--pretrained`.
- Removed because unused: `EVSessionModel.plot_components`, `MediumTermSimulator.plot_energy_trajectory`,
  `ShortTermSimulator.compute_load_curve`, `SmartChargingOptimizer.plot_load_curve`, and the plotting
  functions `plot_arrival_distribution`, `plot_session_heatmap`, `plot_energy_distribution`, `plot_gmm_means`.
- `NHiTSForecaster` and `TransformerForecaster` live in `gears.models.forecaster_dl` (import them from
  `gears.models`).

### Changed
- **`import gears` is lazy**: no `torch`, `neuralforecast`, `pmdarima`, `matplotlib` or `requests`, no
  network, no disk access. The library logs instead of printing; `verbose=True` shows it on stderr.
- Bad arguments raise a `ValueError` saying what was received and what is valid; unfitted objects raise
  `NotFittedError`.
- SARIMA is always tried; if it cannot be fitted or diverges, a warning gives the reason and the forecaster
  switches to `"probabilistic"` (`forecaster.method_` tells which one ran).
- **Results differ from earlier versions**: the smart-charging optimiser no longer assumes a hidden 0.9
  efficiency, allocates energy slot by slot and bills only the energy delivered (costs 20–25 % lower on
  simulated data); `simulate_short_term(n_sessions=...)` seeds its days from `seed`, not `seed + day`.
- `OutputAggregator` is about 40 % shorter with identical output; `gears.plotting` no longer emits a
  converter `UserWarning`.

### Added
- `load_pretrained(model_id, path=..., cache_dir=...)`: SHA-256 checked before the file is opened,
  `GEARS_HOME` and `GEARS_OFFLINE=1`, a local `path` for machines without internet, a warning when
  scikit-learn differs from the training version.
- Data checks in `load_sessions` and `fit`: fewer than 10 sessions refused; impossible rows dropped and
  counted in one warning (`strict=True` raises); Wh / minutes heuristics; time zones dropped with a warning;
  sparse contexts listed and served by the closest one; an unknown context warns once (`strict=True` raises).
- `gears fit --stratify-by --max-components --max-samples-per-context --seed`; `gears._optional.require`
  (a missing extra gives one `ImportError` naming `gears-ev[vae]` or `[dl]`).
- `examples/` (four toy scripts), a README whose code blocks are tested, and a short test suite
  (under a minute by default; markers `vae`, `dl`, `slow`; no network; a coverage floor).

### Fixed
- The checksum of the French model was that of an empty file, so every `load_pretrained` failed; a download
  is verified before it enters the cache and leaves no temporary file when it fails.
- `OutputAggregator.build_load_profiles` returned a year of zero load when `stratify_by` lacked
  `day_of_week` or `season`; it now raises.
- `linear_growth_profile` gave negative sessions for a negative rate; `n_scenarios=0` was replaced by the
  default; a one-day history gave `NaN` counts.

### Known limits
- `GEARSModel.simulate_short_term` cannot choose the site type or département of a model fitted per
  département (the French model): it uses the most populous context and warns. Use
  `ShortTermSimulator.simulate_single_day(context=...)`.

# Changelog

[Keep a Changelog](https://keepachangelog.com) format, [SemVer](https://semver.org) versions.

## [Unreleased]

## [2.0.0] — first public release

GEARS simulates the charging sessions of electric vehicles, forecasts how many there will be per day, builds
multi-year demand scenarios and schedules smart charging.

### Simulation and forecasting
- `EVSessionModel`: one Gaussian mixture (or conditional VAE, `[vae]` extra) per context (site type,
  département, weekday, season) over arrival time, duration and energy; sparse contexts are listed and served by
  the closest one; `fit` accepts `n_components="auto"` (BIC) or an integer, optional recency weighting.
- `GEARSModel`: `fit`, `simulate_short_term`, `simulate_medium_term`, `smart_charge`, `save` / `load`;
  `context=` chooses the site type and département of a model fitted per site (the French one), the dates
  giving weekday and season; a key the model is not stratified by raises instead of being ignored.
- Session counts: SARIMA with a public-holiday regressor (`use_holidays`, on by default: a holiday is forecast like
  the holidays of the history; falls back to a normal distribution with a warning that gives the reason;
  `forecaster.method_` tells which one ran), `PersistenceForecaster`, and `NHiTSForecaster` /
  `TransformerForecaster` (`[dl]` extra). A simulation without a fitted forecaster (any pre-trained model) warns
  that each day gets 20 sessions.
- Medium term: linear, S-curve and Bass growth profiles, all starting from the observed fleet;
  `annual_growth_rate` for the linear one.
- Smart charging: V1G price and renewable-share optimisation per slot, `savings_summary`.

### Pre-trained French model
- `load_pretrained("french")` / `GEARSModel.from_pretrained("french")` download the model from the
  GitHub release, check its SHA-256 before opening the file and cache it (`GEARS_HOME`, `GEARS_OFFLINE=1`; a local
  `path=` serves machines without internet). It needs NumPy 2 and warns when the scikit-learn major version differs
  from the one it was trained with. The README gives the source and licence (Licence Ouverte 2.0) of the data.

### Data
- `load_sessions` and `fit` read CSV, Excel, Parquet, JSON and pickle, recognise common column names and units
  (Wh / minutes), and refuse fewer than 10 sessions. Impossible rows are dropped and counted in one warning
  (`strict=True` raises); time zones are dropped with a warning.
- Dates written as numbers are read day-first or month-first from their values (`13/01/2024` can only be the
  13th); a file where no value settles it warns, and `dayfirst=True/False` decides.

### Command line and API conventions
- `gears fit` and `gears simulate` (`--model` or `--pretrained`, `-c KEY=VALUE`, `--strict`).
- `import gears` is lazy: no `torch`, `neuralforecast`, `pmdarima`, `matplotlib` or `requests`, no network, no
  disk access. A missing extra raises one `ImportError` naming `gears-ev[vae]` or `[dl]`.
- The library logs instead of printing (`verbose=True` shows it on stderr). Bad arguments raise a `ValueError`
  saying what was received and what is valid; unfitted objects raise `NotFittedError`.

### Project
- MIT licence, `SECURITY.md`, `CITATION.cff`, issue and pull-request templates, four example scripts and a notebook,
  a README whose code blocks are tested, a test suite with a coverage floor, CI on Python 3.10 to 3.13 including a
  run with every dependency at its minimum version.

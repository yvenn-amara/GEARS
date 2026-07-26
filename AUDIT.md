# GEARS (gears-ev) — Pre-Refactor Audit

Session 0. Analysis only — no code under `gears/`, `tests/`, or `notebooks/` was modified.
All findings below were verified by actually reading the cited file/line or by running the
command shown; none are assumptions. Environment: fresh venv, `pip install -e ".[dev]"`,
Python 3.12.3, ruff 0.16.0.

---

## a. CI root cause

**Your two numbers are both confirmed exactly, but the failure story is different from
what they suggest.**

- `ruff check gears/ tests/` → **274 errors**, exit code 1. Confirmed by running it fresh.
  Breakdown (`ruff check gears/ tests/ --statistics`):
  - 211 are pure style/modernization, auto-fixable: `UP045`/`UP037`/`UP007` (104+41+33 —
    `Optional[X]`/quoted annotations → `X | None`) and `I001` (33, import sorting).
  - **18 `F821` undefined-name** — all in `gears/models/vae.py` (lines 199, 208, 213, 217,
    226‑229, 251‑254, 299‑301). Cause: `torch` is only imported lazily inside methods, but
    type hints like `x: "torch.Tensor"` reference it at module scope with no
    `TYPE_CHECKING` import — real lint errors, but not runtime bugs (the file has
    `from __future__ import annotations` at the top, so annotations are never evaluated).
  - **10 `BLE001` blind `except Exception`** at `gears/data/insee.py:261,429`,
    `gears/evaluation/benchmark.py:533,536,546,549,564`, `gears/models/forecaster.py:263,388`,
    `gears/models/gmm.py:358`, `gears/models/registry.py:515`.
  - Remainder: 6 `C405`, 5 `C408`, 5 `RUF022`, 4 `F841`, 3 `RUF046`, 3 `RUF059`, 3 `UP035`,
    1 each of `C401`/`F401`/`PIE790`/`RUF012`/`RUF100`/`UP006`.

- **Correction to the framing:** `.github/workflows/ci.yml`'s lint step runs
  `ruff check gears/ tests/` with no `continue-on-error` and no `|| true` (unlike the mypy
  step three lines later, which does have `|| true`). GitHub Actions stops a job at the
  first failing step by default. **With 274 ruff errors, the CI job fails at Lint and the
  `pytest` step never runs today** — the 12 test failures below are a real, second,
  independent blocker, not the one currently surfacing in CI. `gitlab-ci.yml` has the
  identical pattern (`script:` line with no error suppression), so the same is true there.
  The `build` job also `needs: test`, so it never runs either.

- `pytest tests/` under `.[dev]` → **250 passed, 10 skipped, 12 failed** — confirmed exactly.
  `pytest tests/ --collect-only -q` shows **272 tests collected**, not 262 — a further
  correction to the "actually 262" figure. All 12 failures are in tests that exercise
  `gears/models/vae.py`. **Correction on the exception type:** it is not literally
  `ModuleNotFoundError` — `vae.py` wraps the torch import in a `_require_torch()` helper
  that catches the import failure and re-raises a custom `ImportError` with install
  instructions. Root cause confirmed in `pyproject.toml:37‑40`: `torch` is declared under
  the `vae` extra, not `dev`.
  - **Verified this is purely a packaging gap, not a real failure**: after installing torch
    into the same venv (see §d for how, given a disk-space constraint), re-running the
    identical suite gives **262 passed, 10 skipped, 0 failed** — every one of the 12
    "failures" passes once torch is present. The 10 skips are unrelated (`neuralforecast`/
    `torch` for the `dl` extra, in `test_forecaster.py` and `test_regression.py`).

- **CI blind spot, discovered while tracing a different issue (see §c):** `ruff check`
  only targets `gears/` and `tests/` — `scripts/` is never linted, and nothing in `tests/`
  imports it either. `scripts/fit_gmm.py` currently has a genuine `SyntaxError` (`return`
  outside function, line 495) that makes the file **fail to import at all** — confirmed by
  running the installed `gears-fit-gmm` console script, which crashes immediately. This is
  invisible to both CI jobs and to `pytest`. Full diagnosis in §c.

- `CONTRIBUTING.md`'s own documented dev workflow doesn't fully work either: `pip install
  -e ".[dev,notebooks]"` triggers `WARNING: gears-ev 1.0.0 does not provide the extra
  'notebooks'` (verified via `pip install ... --dry-run`) — there is no `notebooks` extra
  in `pyproject.toml` (only `dl`, `vae`, `hub`, `dev`, `all`). And `black gears/ tests/`
  (the documented formatting step) fails with "command not found" — `black` is not in the
  `dev` extra's dependency list (`pyproject.toml:44‑53`).

---

## b. README drift

Verified line-by-line against `README.md` (400 lines) and the current code:

1. **Notebooks** — README documents 3 notebooks; there are 5:
   `1_gmm_descriptive.ipynb`, `2_gmm_forecasting.ipynb`, `3_gmm_scenarios.ipynb`,
   `4_persistence_vs_gmm_benchmark.ipynb`, `5_generic_dataset_example.ipynb`. Notebooks 4
   and 5 appear nowhere in the README (not in the notebook table, not in the package-tree
   listing).
2. **Test count** — README's "Tests" section says `python -m pytest tests/ -q` → expects
   "142 passed, 2 skipped", and the bottom summary table repeats "tests/ 142 tests". Actual:
   272 collected, 250 passed / 10 skipped / 12 failed under `.[dev]` alone (see §a).
3. **Registry listing is wrong** — README line 92: `print(registry.list())  # one entry:
   'french'`. Actual: `NativeGMMRegistry._CATALOGUE` (`gears/models/registry.py:87‑109`) has
   **two** entries, `"french"` and `"french_vae_sample"` — confirmed both by reading the
   dict and by `tests/test_registry.py::test_vae_sample_entry_present`,
   `test_list_returns_at_least_two_rows`, which require both to be present and passing.
4. **No mention of the VAE at all** — no `model_type="vae"`, no `ConditionalVAE`, no
   `[vae]` extra, nowhere in the README, despite `gears/models/vae.py` (594 lines),
   `french_vae_sample` in the registry, an entire VAE track in notebooks 1 and 2, and a
   `--model-type vae` flag in `scripts/fit_gmm.py`.
5. **No mention of the persistence-vs-GMM benchmark** — `run_benchmark.py` (194 lines),
   `gears/evaluation/` (two modules, 594+76 lines), and notebook 4 are entirely undocumented
   in the README.
6. **Package-structure tree omits `gears/evaluation/`, `gears/cli.py`, and `gears/pipeline.py`**
   entirely, and lists only `scripts/fit_gmm.py` under `scripts/`, omitting
   `scripts/prepare_hf_bundles.py` and `scripts/compare_external.ipynb`.
7. `requires-python = ">=3.10"` (`pyproject.toml:6`) does match the README's stated minimum —
   this one line is accurate.

---

## c. Dead / low-quality code inventory

### Most severe: `scripts/fit_gmm.py` is completely broken
Running the package's own installed console script reproduces this directly:
```
$ gears-fit-gmm --help
  File ".../scripts/fit_gmm.py", line 495
    return gmm
SyntaxError: 'return' outside function
```
The file (578 lines) contains **two full, overlapping copies** of its tail, apparently
left over from an incomplete refactor that added VAE support:
- Lines 1‑447: current, correct version — `fit_and_save()` (270‑353) handles both GMM and
  VAE, `main()` (358‑444) has `--model-type` support, ends cleanly at
  `if __name__ == "__main__": main()` (446‑447).
- **Lines 449‑577: an orphaned leftover**, indented as if still inside the
  `if __name__ == "__main__":` block, containing the tail of an *old* `fit_and_save` body
  (imports, GMM-only fitting logic, ending in a bare `return gmm` at line 495 — invalid
  because it's not inside a `def`), followed by an entire second, GMM-only
  `def main() -> None:` (500‑577) that duplicates the first one with no VAE support.

  `ast.parse()` on the file succeeds (the grammar is technically valid — the orphaned code
  is syntactically part of the `if` block), but `compile()`/`import` fails, because "return
  outside a function" is a compile-time semantic check, not a parse-time one. This is why
  the bug is invisible to `ast.parse`-based tools but fatal to a real import — and, per §a,
  invisible to `ruff` (doesn't scan `scripts/`) and to `pytest` (nothing imports this file:
  there is no `test_fit_gmm.py` or `test_cli.py` anywhere in `tests/`).

  By contrast, `scripts/prepare_hf_bundles.py` — the repo's *other* utility script — was
  run end-to-end (`python scripts/prepare_hf_bundles.py --demo`) and works correctly.

### `gears/models/registry.py` — comment directly contradicts the code six lines below it
Lines 71‑85 document a "KEY INVARIANT" that the catalogue has exactly one entry
(`"french"`), citing `test_registry.py::TestCatalogue expects exactly ["french"]"` as the
reason, and stating the sample bundle "is NOT exposed in the public catalogue." The actual
`_CATALOGUE` dict two lines later (87‑109) has **two** entries, `"french"` and
`"french_vae_sample"` — and `tests/test_registry.py` (viewed directly) has no
`TestCatalogue` class at all; instead `test_vae_sample_entry_present` and
`test_vae_sample_meta` require the second entry to exist. The comment is simply stale.

### `gears/models/registry.py:593‑647` — `get_gmm()` ignores its own parameters
```python
def get_gmm(location_type, departement, saison, day_of_week) -> "EVSessionGMM":
    ...
    registry = _get_default_registry()
    return registry.load("french")
```
All four parameters are validated nowhere and used nowhere in the body — the function
always returns the same full "french" bundle regardless of what's passed. Calling
`get_gmm("work", "999", "not-a-season", 99)` returns exactly the same object as
`get_gmm("work", "75", "winter", 0)`. (Confirmed in notebook 1, cell 10 — the returned
object is then filtered again with a `context=` dict, which is where the real stratum
selection happens; the four top-level parameters are decorative.)

### `gears/models/registry.py:354` — docstring example references a nonexistent model ID
`ModelRegistry`'s class docstring: `>>> bundle = registry.load("work_fr_demo")`. The actual
`_CATALOGUE` for this class (330‑341) has exactly one key, `"french_demo"`. Calling the
documented example raises `ValueError: Unknown model ID 'work_fr_demo'`.

### `run_benchmark.py` lives at the repo root, but every reference to it says `scripts/`
The file's own docstring (`run_benchmark.py:26,29,36,38`) and notebook 4
(`4_persistence_vs_gmm_benchmark.ipynb`, 3 separate markdown cells) all instruct
`python scripts/run_benchmark.py ...`. The actual file is `./run_benchmark.py`, not under
`scripts/`. `ls scripts/` confirms it only contains `fit_gmm.py`, `prepare_hf_bundles.py`,
`compare_external.ipynb`.

### `gears/cli.py:99` — help text contradicts `pipeline.py`'s own docstring
`--years` is documented as `"Horizon in years (max 5)."`, but no validation anywhere
clamps or rejects `years > 5` — it's passed straight through to
`GEARSModel.simulate_medium_term`, whose own docstring in `gears/pipeline.py` states
explicitly "No upper limit." Separately: the `gears` command itself isn't installable —
`pyproject.toml`'s `[project.scripts]` (line 56‑57) only registers `gears-fit-gmm =
"scripts.fit_gmm:main"`. There is no entry point for `gears.cli:main`. Confirmed by
installing the package and running `gears --help` → `not found` (exit 127). `cli.py`'s own
module-level docstring examples (`gears fit ...`, `gears simulate ...`) do not work as
written; only `python -m gears.cli ...` does.

### `gears/simulation/medium_term.py:61` — misleading name
`linear_growth_profile()` is not linear: it computes
`base_sessions_per_day * (1 + annual_growth_rate) ** t`, i.e. compound/exponential growth.
Relevant to §e below.

### Notebook 3 imports a whole simulation subsystem it never calls
`3_gmm_scenarios.ipynb`, cell 1, imports `MediumTermSimulator`, `linear_growth_profile`,
`s_curve_growth_profile`, `s_curve_linear_tail_profile`, `bass_diffusion_profile`, and
`double_s_curve_profile` from `gears.simulation.medium_term`. Grepping the notebook's own
source for `linear_growth_profile(`, `s_curve_growth_profile(`, `MediumTermSimulator(`, or
`.simulate(` returns **zero matches** — none of these are ever called. Part B ("Long term")
instead defines and calls a bespoke `build_lt_scenario_analytical()` function (cell ~19),
with its own explicit comment: *"Pas de simulation GMM : evite la divergence numerique sur
15 ans"* ("No GMM simulation: avoids numerical divergence over 15 years"). The notebook's
own markdown (cell 15, "Approach" step 4) still claims "Run MediumTermSimulator with ≥ 30
scenarios" — that's not what the code below it does. Full implications in §e.

### `gears/plotting.py:928‑929` — undocumented hard clip
```python
_y_ceiling = anchor_monthly_val * 10
m_pivot = m_pivot.clip(upper=_y_ceiling)
```
Inside `plot_lt_trajectories`. Not mentioned in the function's docstring. Traced in full
in §e — this silently flattens two of notebook 3's three long-term scenarios well before
2040.

### Notebook 1 vs. notebook 2 — an unverified claim contradicted by a later, more rigorous check
Notebook 1's summary (final markdown cell) states: *"The VAE reproduces the same marginal
shapes as the GMM ... despite having no discrete mixture structure to introspect."* Every
quantitative VAE-vs-GMM comparison in notebook 1 (cells 15/17, 19/22, 32/33/35) compares
**means only** (bar charts of averages) — never spread/shape. Notebook 2 (cells 29 and 31)
runs the package's own `distribution_comparison()` (Wasserstein/KL/KS) and its *stored,
already-executed output* tells a different story — see §d. This is a real, checkable
inconsistency between two notebooks' narratives, not a matter of interpretation.

### Minor / lower-priority
- `gears/simulation/short_term.py`, `compute_load_curve()` (line 215): applies each
  session's nameplate `power_kw` for its *entire* duration to build the load curve, which
  is inconsistent with `sessions_batch["energy"]` (in `gears/simulation/medium_term.py:536‑539`)
  being capped at `power_kw * duration` — i.e. most sessions' true average power is below
  nameplate, but the load-curve reconstruction doesn't account for that. A modeling
  simplification, not a crash bug.
- `gears/smart_charging/optimizer.py:143,398`: core optimisation loop uses
  `sessions.iterrows()` — O(n) Python-level iteration with per-row boolean masking against
  the full price signal; a real scalability concern given the ~3M-row production dataset
  mentioned in your brief, though not incorrect.
- `notebooks/lightning_logs/` (4 version dirs) contains leftover PyTorch-Lightning/
  `neuralforecast` NHiTS training logs. These are `.gitignore`'d (`git status --ignored`
  confirms), so not part of the tracked repo — but they are present in the delivered
  archive. Their hyperparameters (`h=14, input_size=28, mlp_units=[[512,512]]×3`) are
  consistent with `NHiTSForecaster`'s actual defaults in `gears/models/forecaster.py`, so
  this looks like real leftover output from running notebook 2 or 5's optional NHiTS cell,
  not an orphaned unrelated experiment.

---

## d. VAE current state

**Architecture** (`gears/models/vae.py`, 594 lines): a single shared `ConditionalVAE`
(encoder/decoder MLPs) is trained once across *all* stratification contexts at once
(`gmm.py`, `_fit_vae`), conditioned on a learned per-context embedding
(`_embed_context`). Each context is then wrapped in a lightweight `VAEContextSlice` that
duck-types the `sklearn.mixture.GaussianMixture` interface (`.sample()`, `.score_samples()`,
`.weights_`, `.n_components` — always `1`), so `EVSessionGMM(model_type="vae")` is a
drop-in for the GMM path everywhere downstream (confirmed: it's exercised, unmodified,
through `GEARSModel`, `OutputAggregator`, `SmartChargingOptimizer`, and `plotting.py` in
notebooks 1 and 2). Training: Adam, a KL term with a `beta` coefficient, and a **learned,
global, context-invariant observation variance** (`log_recon_var`, clamped to
`[1e‑4, 10.0]`) shared across every feature and every context. Scoring uses an IWAE
(importance-weighted) log-likelihood estimator (`score_samples`, K samples).

**Is it benchmarked against persistence anywhere? No — confirmed, not inferred.**
`gears/evaluation/benchmark.py` (the rolling-origin harness backing `run_benchmark.py` and
notebook 4) is explicitly scoped as "persistence-bootstrap vs. windowed GMM" — grepping the
whole repo for "vae" turns up matches only in `gmm.py`, `vae.py`, `registry.py`, and
`scripts/fit_gmm.py`; nothing under `gears/evaluation/`. Notebook 4
(`4_persistence_vs_gmm_benchmark.ipynb`, 10 code cells) has zero VAE mentions. The results
file it reads, `results/benchmark/all_results.parquet` (31,578 rows), has a `method` column
with exactly two values: `persistence` and `gmm`.

**Why persistence wins the *actual* benchmark that exists (GMM, not VAE) — verified from
the data, not just quoted from the notebook:**
```
Overall GMM win rate: 9.9%   (paired cells, independently recomputed from the parquet)
```
matching the notebook's own stated ~10%. The benchmark's own code comment
(`gears/evaluation/benchmark.py:268‑269`) and notebook 4's conclusion agree on why: the
harness forces `n_components=1` for GMM (windowed training pools inside a rolling-origin
cell are usually too small to support a >1-component BIC search), so the GMM degenerates to
a single Gaussian while bootstrap-persistence reproduces the empirical (often multi-modal,
right-skewed) shape exactly by construction — a single Gaussian systematically loses
Wasserstein/KL/KS distance to a skewed real distribution regardless of how well its mean is
placed.

**VAE quality — actually fitted and tested end-to-end** (torch isn't in `.[dev]`; I
downloaded the wheel directly with `pip download torch --no-deps` and installed it with
`--no-deps` after a disk-space error blocked the normal `pip install ".[vae]"` — this avoids
pulling several GB of CUDA runtime packages that aren't needed for CPU execution, verified
working with a basic tensor op). Result: **a real, reproducible variance-collapse bug**.
Fitting a `ConditionalVAE` on synthetic multi-context data (default hyperparameters:
epochs=50, hidden_dim=256, latent_dim=16) and sampling from a held-in context vs. the true
generating distribution:
```
                  arrival_hour std   duration std   energy std
Real data              1.12            2.42           7.76
VAE samples            0.087 (13x too narrow)  0.825 (2.9x)   5.65 (1.4x)
```
Root cause, traced to code: `ConditionalVAE.sample_prior()` (`vae.py:296‑310`) draws
`z ~ N(0, I)`, decodes it, and **returns the decoder's mean output directly** — it never
adds the learned observation noise (`log_recon_var`) that the model estimated during
training. The only source of sample-to-sample variety is however much the decoder's output
happens to vary across different `z` draws; if the decoder has learned a fairly smooth
mapping (encouraged by the KL term, especially since the global `log_recon_var` can absorb
reconstruction error up to its cap without needing informative latents), samples end up
under-dispersed. Scoring (`score_samples`, IWAE) and generating (`sample_prior`) use
different parts of the model — this is why the bug is invisible to log-likelihood-based
evaluation but visible immediately in generated-sample statistics.

**This is independently confirmed by numbers already computed and stored in the repo's own
notebook 2** (cells 22, 29, 31 — I did not need to recompute these, they are stored
execution outputs):
```
Mean log-likelihood on held-out eval_df (higher = better):
  GMM (french)            : -4.439
  VAE (french_vae_sample) : -3.464   ← VAE looks BETTER by this metric

Distribution distances vs. the SAME held-out real data (lower = better):
              GMM wasserstein   VAE wasserstein    GMM KL    VAE KL
  hour             0.2458            2.6934  (11x)   0.0231    0.3437 (15x)
  duration         0.3240            0.7206  (2.2x)   0.0310    0.0736 (2.4x)
  energy           7.0915           10.8862  (1.5x)   0.2022    2.4753 (12x)
```
The VAE scores *better* on log-likelihood but is worse on every single distributional
distance metric, for every feature, all statistically significant (KS p-value = 0.0) — the
exact signature you'd expect from a model that places density accurately at the mean but
under-samples the true spread. Notebook 2, cell 18, also shows the VAE-simulated aggregate
daily energy (1,717 kWh) is roughly half the real value (3,579 kWh) and well below the
GMM's (3,038 kWh) for the same département and window — consistent with the same root
cause.

---

## e. Medium-term "capping" diagnosis

Two distinct, independently-verified mechanisms — not one bug, and neither is a guess:

### Mechanism 1 — the Part A "fan chart" is real, but its width is artificially pinched
`DepartmentForecaster._forecast_dept` (`gears/data/insee.py`, ~line 426) generates each
Monte-Carlo scenario as `model.forecast(horizon)` (the fitted SARIMA's point forecast) plus
independent Gaussian noise with
```python
noise_scale = max(stats["std"] * 0.05, 0.1)
```
i.e. 5% of the department's raw historical std, floored at 0.1 kWh, applied identically at
every horizon step. Verified by fitting this exact class on synthetic trending+seasonal
daily data (std=32.4 kWh) and measuring the resulting 80% CI band from `predict()`:
```
day +1 : median=205.7  80%CI=[203.9, 208.1]  width=4.1 kWh (2.0% of median)
day +30: median=223.1  80%CI=[221.4, 225.1]  width=3.7 kWh (1.7% of median)
day +120:median=226.6  80%CI=[224.6, 228.7]  width=4.1 kWh (1.8% of median)
```
The band width stays ~2% of the median at *every* horizon — it does not widen as a genuine
forecast cone should (real SARIMA forecast-error variance grows with horizon). This is
exactly the mechanism behind `plot_mt_fan_charts` (`gears/plotting.py:539‑680`), which draws
its band from `pivot.quantile(0.1/0.9, axis=1)` over these scenarios.

**This is a known, already-fixed-elsewhere bug that was never ported.**
`gears/models/forecaster.py`'s `SessionForecaster._forecast_one_scenario` (~line 391‑396)
uses the *full* std instead, with an explicit code comment: a smaller fraction (their
comment cites the same 0.1× pattern) "was used historically and produced only ~33%
coverage — a serious underestimate of real forecast uncertainty." I re-ran the identical
synthetic-data test through `SessionForecaster` instead of `DepartmentForecaster`:
```
[SessionForecaster, full-std noise] day+1:  80%CI width=83.1 kWh (40.3% of median)
[SessionForecaster, full-std noise] day+120:80%CI width=82.1 kWh (39.3% of median)
```
Same underlying data, ~20x wider band. `DepartmentForecaster` (insee.py) is structurally
almost identical to the pre-fix `SessionForecaster` — it looks like a fork that predates
(or never received) that fix.

### Mechanism 2 — the Part B long-term scenarios are real, but the chart hard-clips them
`plot_lt_trajectories` (`gears/plotting.py:928‑929`) clips the plotted monthly series at
`anchor_monthly_val * 10`. Notebook 3's own scenario constants
(`EV_SHARE_NOW = 0.031`, cell with `N_LONG_SCENARIOS`/`LONG_YEARS = 15`) put the ceiling at
a 10x share multiple, i.e. `share = 0.31`. I reconstructed the notebook's exact Scenario
B (logistic, target 40% by 2040) and Scenario C (Bass diffusion, target 65% by 2040)
formulas and ran them:
```
Ceiling: share = 0.3100 (10x anchor)

  year  share_B  mul_B  clipped?    share_C  mul_C  clipped?
  2033   0.2159   6.97     no        0.1039   3.35     no
  2034   0.2674   8.62     no        0.1315   4.24     no
  2035   0.3120  10.06    YES        0.1663   5.37     no
  2038   ...                        crosses ceiling ~2038.0
  2040   0.4000  12.90    YES        0.4195  13.53    YES

Scenario B crosses the 10x ceiling around year 2034.9
Scenario C crosses the 10x ceiling around year 2038.0
```
Both scenarios' *true modeled* trajectories keep climbing to ~12.9x and ~13.5x their 2025
baseline by 2040 — but the chart visually flattens them at 10x starting in 2035 and 2038
respectively, years before the simulation horizon ends. This produces a hard, visible
plateau that has nothing to do with the underlying growth model and everything to do with
an undocumented plotting-side y-axis safeguard.

### A third, related but currently-inert mechanism
`gears/simulation/medium_term.py`'s growth-profile functions (`s_curve_growth_profile`
etc.) are genuinely self-saturating by design (that's the point of an S-curve/Bass model).
Traced directly by calling them:
```
s_curve_growth_profile (defaults: saturation_factor=3.0, midpoint_year=2.5, steepness=1.5)
  year 3 :  67.9% of the way to its ceiling — already visibly decelerating
  year 4 :  90.5% of ceiling
  year 8 : 100.00% of ceiling (flat to 2 decimals for the rest of any longer horizon)
```
This *would* independently explain a hard plateau in a `MediumTermSimulator.simulate(...,
growth_model="s_curve")` run over a multi-year horizon with default parameters. But per §c,
**notebook 3 never actually calls this class or these functions** — Part B's chart is
generated entirely by the separate analytical shortcut, so this mechanism is not what's
currently producing notebook 3's plateau; it is a latent property of the (currently
unexercised) "real" simulator path that would matter if/when that path is restored.

### What I tested and could not confirm (reported honestly rather than forced)
I hypothesized the SARIMA models themselves (no explicit drift/trend term on a
differenced series) might independently flatten toward a constant regardless of horizon.
I built synthetic data with a strong, clean linear trend, fit `DepartmentForecaster` on it,
and inspected the raw point-forecast (no scenario noise) out to 400 days — it tracked
(and in this specific configuration, slightly exceeded) the true trend rather than
flattening. **I could not reproduce a SARIMA-driven plateau** with synthetic data in this
sandbox; I can't rule it out for the real ~3M-row dataset (not available here), but it is
not a confirmed contributor the way Mechanisms 1 and 2 are.

---

## f. Notebook complexity

| Notebook | Total cells | Code cells | Markdown |
|---|---|---|---|
| `1_gmm_descriptive.ipynb` | 47 | 27 | 20 |
| `2_gmm_forecasting.ipynb` | 43 | 21 | 22 |
| `3_gmm_scenarios.ipynb` | 31 | 23 | 8 |
| `4_persistence_vs_gmm_benchmark.ipynb` | 23 | 10 | 13 |
| `5_generic_dataset_example.ipynb` | 28 | 14 | 14 |
| `scripts/compare_external.ipynb` (not one of the 5) | 30 | 16 | 14 |

**Notebook 1** (27 code cells): every GMM section is mirrored by a near-identical VAE
section (weights/means, distributions, seasonal, day-of-week, département coverage, BIC —
6 pairs). This is deliberate (demonstrating API parity) and each pair is short, but three
different chart types (bar chart cells 14‑15, heatmap cell 20, line chart cells 32‑33)
all show essentially the same "weighted means by stratum" data — one of the three could be
cut without losing signal. The hand-rolled France outline + bubble map (cell 39, ~35 lines
of hardcoded coordinates) is decorative for a "descriptive" notebook; cell 38's bar chart
already conveys département coverage numerically.

**Notebook 2** (21 code cells): well-scoped for its 11 stated sections. The one clear cut:
the "live VAE fit" demo (cell 20, §7.2) is explicitly labeled in its own markdown as "not
meant to reproduce the quality of the shipped bundle... purely to demonstrate the API," and
nothing downstream (regret analysis, distribution comparison) depends on its output — it
could move to an appendix or be dropped without affecting any of the notebook's conclusions.

**Notebook 3** (23 code cells): the biggest complexity-vs-signal mismatch of the five. Six
imports (`MediumTermSimulator` + 5 growth-profile functions, cell 1) are never used (§c) —
removing the dead imports, or alternatively actually wiring Part B to the real simulator
they were presumably meant to demonstrate, would both reduce confusion and cut the ~60-line
bespoke analytical code (`build_lt_scenario_analytical`) that currently duplicates what the
simulator is supposed to do.

**Notebook 4** (10 code cells): already lean for a full benchmark load-and-analyze pass;
nothing obvious to cut.

**Notebook 5** (14 code cells): explicitly the "generic/minimal" variant (no VAE, no
department split, per its own intro) — already the trimmed version of notebook 2's
structure; nothing to cut further.

---

## g. Proposed test suite trim

Current: 272 collected tests across 14 files (13 `test_*.py` + fixtures), counted with
`grep -c "^    def test_\|^def test_"` per file then cross-checked against
`pytest --collect-only`.

**Real, confirmed duplication:**
- `tests/test_simulation.py` (26 tests) and `tests/test_medium_term.py` (14 tests) both
  import and test the *same two functions* from `gears.simulation.medium_term` —
  `linear_growth_profile` and `s_curve_growth_profile` (confirmed via each file's own
  import list). `test_simulation.py` has `test_linear_length`, `test_linear_increasing`,
  `test_s_curve_length`, `test_s_curve_bounded`, `test_linear_zero_growth`,
  `test_s_curve_growth`; `test_medium_term.py` has the more complete
  `test_growth_profile_basic`, `test_all_profiles_registered` (covering all 5 profiles, not
  just 2). Recommend consolidating the growth-profile-only tests into `test_medium_term.py`
  (which already covers all 5 profiles) and leaving `test_simulation.py` focused on
  `ShortTermSimulator`/`MediumTermSimulator.simulate()` behavior, which is its other,
  non-overlapping half.

**Deliberate, not redundant, but worth restructuring:**
- `tests/test_regression.py` (28 tests, the largest file) is explicitly documented (its own
  module docstring) as "one test per bug corrected in audit sessions 1 and 2" — 4 named
  historical bugs (R1: `PersistenceForecaster` zero-day-drop, R2: 1970-epoch date bug in
  `sessions_to_daily_counts`, R3: `compute_regret()` double-bracket/DataFrame-vs-Series bug,
  R4: `NHiTSForecaster` input_size/scaler defaults). This is good practice and shouldn't be
  deleted, but it structurally overlaps with `test_forecaster.py` (which separately tests
  `NHiTSForecaster`/`TransformerForecaster`: `test_transformer_is_available`,
  `test_shared_interface`, near-duplicates of `test_regression.py`'s
  `test_is_available_returns_bool`, `test_shared_interface_contract`) and with
  `test_smart_charging.py` (`test_compute_regret_basic`/`test_compute_regret_with_persistence`
  vs. `test_regression.py`'s `test_basic_regret_all_scalars`/`test_regret_with_persistence_all_scalars`).
  Recommend: keep the regression *intent* (a comment referencing "R1"–"R4" and the bug it
  guards against) but move each test into its subject module's file, next to the other
  tests for that class, rather than maintaining a fully parallel file with duplicated
  fixtures.

**Gap, not redundancy — the more important finding:**
- `tests/test_insee.py` has only 5 tests for a 514-line module, and **none of them assert
  anything about forecast uncertainty** — `test_department_forecaster_fit_predict` only
  checks column presence and non-negativity. This directly explains why the noise_scale
  bug in §e went uncaught: there is no test resembling "CI band width should scale with
  historical std" or "CI band should widen with horizon," of the kind that a hypothetical
  R5 regression test would look like. Recommend adding 1‑2 tests here (band-width vs. std,
  band-width vs. horizon) before any refactor — cheap insurance against reintroducing the
  same bug.
- There is no `test_cli.py` and no test that imports `scripts/fit_gmm.py` at all — the
  broken-script bug in §c would have been caught by a single `import scripts.fit_gmm` smoke
  test.

**Well-targeted, no changes suggested:** `test_benchmark.py` (18 tests: no-leakage,
insufficient-history/-volume, degenerate-ensemble edge cases) and
`test_persistence_sampler.py` (23 tests, similarly edge-case-focused) are good examples of
lean-but-meaningful coverage — nothing to trim there.

---

## h. Session plan sign-off

| Item | Validated by actually running/reading | Result |
|---|---|---|
| Ruff error count | `ruff check gears/ tests/` + `--statistics` + `--select F821`/others | 274, breakdown confirmed |
| CI blocks at Lint, not Test | Read `.github/workflows/ci.yml` and `gitlab-ci.yml` line-by-line | confirmed, no `continue-on-error` |
| Pytest count/outcome | `pytest tests/ -q` and `--collect-only -q` | 250/10/12, 272 collected |
| 12 failures are torch-only | Installed torch (worked around a disk-space error via `pip download --no-deps`), re-ran suite | 262 passed/10 skipped/0 failed |
| README notebook/test claims | Read `README.md` in full, `cat -n` for line citations | 3 documented vs. 5 real; stale test count |
| Registry catalogue contents | Read `gears/models/registry.py` in full (647 lines, in 3 passes) | 2 entries, comment says 1 |
| `get_gmm()` ignores its args | Read `registry.py:593‑647`, traced call sites in notebook 1 | confirmed |
| `run_benchmark.py` path drift | `grep -rn "scripts/run_benchmark"` across repo | confirmed, file is at root |
| `gears` CLI not installed | Installed package, ran `gears --help` and `which gears` | not found, only `gears-fit-gmm` exists |
| `fit_gmm.py` syntax error | Ran `gears-fit-gmm --help`, then `ast.parse` vs `compile()` to isolate why, read full 578-line file | duplicate `main()`, dangling `return`, fully diagnosed |
| `prepare_hf_bundles.py` works | Ran `python scripts/prepare_hf_bundles.py --demo` end-to-end | succeeded |
| `CONTRIBUTING.md` install/format steps | `pip install -e ".[dev,notebooks]" --dry-run`; `black --version` | "notebooks" extra warning; black not found |
| VAE architecture & sampling bug | Installed torch, fit `ConditionalVAE` on synthetic data, compared sampled vs. true std; read `sample_prior()` | 3–13x under-dispersion, traced to missing noise injection |
| VAE vs. persistence benchmark scope | Grepped repo for "vae" outside `models/`; read notebook 4 in full; inspected `results/benchmark/all_results.parquet` schema | VAE never benchmarked; GMM-only |
| GMM win-rate vs. persistence | Recomputed independently from the raw parquet | 9.9%, matches notebook's stated ~10% |
| Fan-chart width bug (Mechanism 1) | Read `insee.py`/`forecaster.py`, built synthetic trending data, fit both classes, compared CI widths side-by-side | ~2% vs. ~40% of median, same data |
| Long-term clip bug (Mechanism 2) | Read `plotting.py:928‑929`; reconstructed notebook 3's exact scenario formulas and computed crossing years | B/C cross the 10x ceiling in 2035/2038 |
| S-curve saturation math | Called `s_curve_growth_profile`/`bass_diffusion_profile`/etc. directly with default params | traced year-by-year |
| SARIMA-drift hypothesis | Built synthetic trending data, fit `DepartmentForecaster`, inspected raw forecast | could not reproduce a plateau — reported as inconclusive, not asserted |
| Notebook cell counts | `json.load` each `.ipynb`, counted cell types | table in §f |
| Test suite overlap | `grep` every test file's imports and test names | duplication in growth-profile tests, confirmed |
| `test_insee.py` coverage gap | Read the file in full (46 lines, 5 tests) | no uncertainty-related assertions |
| `git diff --stat` clean | `git init` + baseline commit before any analysis; re-checked after every exploratory step | clean throughout; only this file + `REFACTOR_STATE.md` are new |

All 8 sections above are present under their exact headings. No file under `gears/`,
`tests/`, or `notebooks/` was modified in this session.

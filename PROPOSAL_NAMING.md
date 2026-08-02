# Naming Consistency Proposal — Phase 2 / Session 1

**Status:** proposal for review — no renames implemented yet (that's Session 2).
**Deprecation policy (decided 2026-08-02):** clean break, version bump to **2.0.0**. No
deprecated aliases. Session 2 implements the renames directly; `CHANGELOG.md` documents every
break under a `## [2.0.0] — Breaking changes` heading.

## Why this is needed

Several names in the public API are GMM-specific even though the thing they name now also
covers VAE (and, from Session 4 onward, a plain GMM / recency-GMM / VAE trio under one registry
entry point). This was fine when there was exactly one model family; it stopped being accurate
the moment `NativeGMMRegistry`'s catalogue grew a `"french_vae_sample"` entry. Left alone, every
new model family added later inherits the same mislabeling.

## Rename map

| # | Current | Where | New | Why | Risk |
|---|---|---|---|---|---|
| 1 | `EVSessionGMM` (class) | `gears/models/gmm.py`, re-exported from `gears` | `EVSessionModel` | Handles GMM *and* VAE via `model_type=` | Low — mechanical rename, `ruff`/tests will catch missed references |
| 2 | `gears/models/gmm.py` (module file) | path | `gears/models/session_model.py` | Module name should match the class it defines | Medium — changes the import path `gears.models.gmm`; anyone importing the submodule directly (not just `from gears import EVSessionGMM`) breaks. Fine given clean-break policy; call out explicitly in CHANGELOG |
| 3 | `NativeGMMRegistry` (class) | `gears/models/registry.py`, re-exported from `gears` | `NativeSessionModelRegistry` | Catalogue already holds a VAE entry | Low |
| 4 | `gmm_id` (param, used in `registry.py`, `fit_gmm.py`, tests) | throughout | `session_model_id` | Same reason — see collision note below | Low-medium — widest blast radius by occurrence count, but purely mechanical |
| 5 | `get_gmm()` (module function) | `gears/models/registry.py`, re-exported from `gears` | `get_session_model()` | Same reason, **plus a real bug**: ignores all four of its arguments and always returns the `"french"` bundle | Medium — see behavior decision below |
| 6 | `get_sklearn_gmm()` (method on `EVSessionGMM`/`EVSessionModel`) | `gears/models/gmm.py` | `get_sklearn_component()` | Returns a raw sklearn `GaussianMixture` when `model_type="gmm"`, but the wrapper itself isn't GMM-only; name should describe "the underlying fitted component," not assume GMM | Low |
| 7 | `departement`, `saison` (params on `get_gmm`/`get_sklearn_gmm`) | `gears/models/registry.py` | `department`, `season` | French identifiers in the public API, inconsistent with `department`/`season` already used in `stratify_by` and context dicts everywhere else; also a point-4 (no-French) violation | Low — two params, straightforward |
| 8 | `gears/data/gmm/` (directory) | package data | `gears/data/session_models/` | Stores VAE bundles too | Medium — affects `_GMM_DIR`, packaging config (`MANIFEST.in`/`pyproject.toml` data-files entries if any), and any user relying on the on-disk path |
| 9 | `scripts/fit_gmm.py` | scripts | `scripts/fit_session_model.py` | Already fits VAE via `--model-type vae` | Low |
| 10 | `gears-fit-gmm` (console script) | `pyproject.toml [project.scripts]` | `gears-fit-session-model` | Same reason | Low — one line in `pyproject.toml` |
| 11 | `list_gmms()` (function) + `args.list_gmms` (argparse dest, `--list`) | `scripts/fit_gmm.py` | `list_session_models()` / `args.list_models` | Same reason | Low |
| 12 | `notebooks/4_persistence_vs_gmm_benchmark.ipynb` | notebooks | `notebooks/4_persistence_vs_session_model_benchmark.ipynb` | It's a 4-arm benchmark (persistence / GMM / recency-GMM / VAE), not GMM-only | Low, but see below |
| 13 | `notebooks/1_gmm_descriptive.ipynb`, `2_gmm_forecasting.ipynb`, `3_gmm_scenarios.ipynb` | notebooks | **not renamed** (recommendation) | Same underlying issue, lower priority | N/A — see recommendation |

### Notebook filenames — recommendation

Renaming notebook *files* breaks external bookmarks/cross-links (README links, any blog posts,
anyone's saved links into the public repo) for less benefit than fixing the *content*, since the
content is what a reader actually reads. Recommendation: rename notebook 4's file (its scope is
actively wrong, not just imprecisely named — it benchmarks four arms and calls itself GMM-only)
but leave 1/2/3's filenames as-is; fix any GMM-only *language* inside their cells/headers instead.
This is a judgment call, not a mechanical one — flagging for explicit sign-off along with
everything else here.

## The `gmm_id` / `model_id` collision

`ModelRegistry` (the *other* registry class, for full HF-Hub `GEARSModel` bundles) already uses
`model_id` for a different concept: the identifier of a full `{gmm, forecaster, metadata}` bundle
on Hugging Face Hub. If `NativeGMMRegistry`'s `gmm_id` simply became `model_id` too, the same name
would mean two different things depending on which registry you're looking at.

**Resolution:** keep the new name more specific — `session_model_id` for
`NativeSessionModelRegistry`, leaving `ModelRegistry.model_id` untouched. This is more consistent
with how "session model" is used elsewhere in this proposal (item 1) and avoids the larger,
unnecessary churn of renaming `ModelRegistry`'s concept too.

## `get_gmm()`'s dead parameters — a real bug, not just a naming issue

Confirmed by reading the current source: `get_gmm(location_type, departement, saison,
day_of_week)` accepts four arguments and uses none of them — it always calls
`registry.load("french")`. Two fixes are possible:

- **(a) Make the parameters real**: the function would need to translate
  `(location_type, department, season, day_of_week)` into a specific stratum lookup. But
  `NativeSessionModelRegistry.load()` operates at the *bundle* level (`"french"` vs
  `"french_vae_sample"`), not the *stratum* level — strata are resolved later, inside
  `EVSessionModel.get_sklearn_component(context=...)`. Making the four arguments "real" here would
  mean either duplicating that resolution logic in two places, or changing what "loading a
  registry entry" means.
- **(b) Narrow the signature to match reality**: `get_session_model(bundle_id: str = "french") ->
  EVSessionModel`, returning the whole fitted wrapper; stratum-level lookups stay where they
  already correctly live, on the returned object via `.get_sklearn_component(context=...)`.

**Recommendation: (b).** It matches what the function actually does today, doesn't invent new
resolution logic, and makes the one-argument reality visible in the signature instead of hidden
behind four arguments that silently do nothing. Flagging as a judgment call for sign-off since (a)
is defensible if there's a reason to want stratum-level retrieval at this exact call site that
isn't visible from the code alone.

## Stale, low-risk fixes bundled into this session's PR (doc/comment-only, no behavior change)

These were confirmed against the current file contents this session, not assumed from the plan:

- `gears/models/registry.py`'s `NativeGMMRegistry._CATALOGUE`'s own "KEY INVARIANT" comment
  claimed the catalogue has exactly one entry and cited a `TestCatalogue` class that doesn't
  exist in the current test suite — the dict six lines below it has had two entries since
  Session 4. Comment rewritten to describe the current two-entry catalogue accurately.
- `run_benchmark.py`'s own module docstring still told users to run
  `python scripts/run_benchmark.py ...` — the file lives at the repo root, not `scripts/`.
  `README.md` was fixed for this in Session 7; the script's own internal docstring wasn't.
  Fixed to `python run_benchmark.py ...`.
- `CONTRIBUTING.md` documented `pip install -e ".[dev,notebooks]"` (no `notebooks` extra exists in
  `pyproject.toml` — only `dl`) and `black gears/ tests/` (black isn't declared as a dependency
  anywhere). Both confirmed broken by reading the current `pyproject.toml`. Fixed to
  `pip install -e ".[dev]"` and removed the `black` line/checklist item (ruff already covers
  formatting via `ruff format`, so no replacement command is needed).
- `GEARSModel.simulate_medium_term`'s docstring and the CLI's `--growth-model` choices only
  offered `linear`/`s_curve`, but `gears/simulation/medium_term.py` fully supports `bass` as a
  third `growth_model` (confirmed: `GROWTH_PROFILES` dict has all three, `bass_diffusion_profile`
  is exported from `gears` top-level). It was simply never surfaced above that layer. Exposed
  `bass` in both the CLI's `click.Choice([...])` and the docstring's enumeration.

## Not part of this rename map (confirmed fine as-is)

- `n_components` / `--n-components` (GMM-specific by design — only meaningful when
  `model_type="gmm"`; the help text correctly says "Number of GMM components").
- `--vae-*` CLI flags in `scripts/fit_gmm.py` — already correctly VAE-scoped, not misnamed.
- `ModelRegistry` and its `model_id` — a genuinely different, already-generic concept; see
  collision note above for why it stays untouched.

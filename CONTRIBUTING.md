# Contributing to GEARS

Thank you for helping. This page is the short version: how to set up, how to check your work, what a
pull request must contain, and how a finding from the (private) `GEARS_sandbox` repository comes back
here.

## Set up

```bash
git clone https://github.com/EDF-Lab/GEARS.git
cd GEARS
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Python 3.10 or later. `[dev]` holds the test and lint tools, not `torch`: to run the tests of an extra, install
it too (`pip install -e ".[dev,vae]"`, or `[dev,dl]`; see the README).

## Run the tests

```bash
pytest                                   # the default run: everything except the markers below
pytest --cov                             # same, with branch coverage (fails below the floor in pyproject.toml)
pytest -m slow                           # the few tests that need data files not in the repository
pytest -m vae                            # needs torch
pytest -m dl                             # needs torch and neuralforecast; heavy, run on demand
pytest tests/test_sanity.py -k sarima    # one file, or the tests matching a word
```

- A test file that needs an extra starts with `pytest.importorskip(...)`: without the extra it is
  skipped, never failing.
- Tests never use the network (any connection raises `NetworkAccessError`) and never touch your real
  `~/.cache/gears` (each test has a private `GEARS_HOME`).
- Prefer a test that checks a value you can compute by hand, or against an independent recomputation,
  to one that only checks that something was returned.
- `tests/test_examples.py` runs every script of `examples/`, and `tests/test_readme.py` runs every code
  block of the README marked `<!-- test -->`. If you change an API those use, they fail: fix the
  example or the README in the same pull request. Mark every new `python` block of the README.
- `examples/quickstart.ipynb` is `examples/01_quickstart.py` (its `# %%` lines are the cells). After
  editing the script, rebuild the notebook, which stays without outputs (the test checks both):
  `pip install jupytext`, then
  `jupytext --set-kernel python3 --to ipynb examples/01_quickstart.py -o examples/quickstart.ipynb`.

## Style and design rules

```bash
ruff check gears tests examples          # the rules are in pyproject.toml
vulture                                  # unused code (also in pyproject.toml)
```

- `import gears` must stay instant: no import of `torch`, `neuralforecast`, `pmdarima`, `matplotlib` or
  `requests` at module level, and no network or disk access (`tests/test_imports.py` checks it). Import
  an optional dependency inside the function that needs it, through `gears._optional.require`, so that
  the error says which extra to install.
- The library does not `print`: use `logging`. Warnings are `UserWarning`s that say what happened and what
  to do; errors are one line, name the argument, and say the expected value.
- No silent fallback to a substitute model or to made-up data: raise, or warn with the consequence.
- New public names go in `gears/__init__.py` (`_EXPORTS` and `__all__`) only if a user needs them at top
  level; keep that list short.

## Continuous integration

`.github/workflows/ci.yml` runs on every pull request: `core` (lint and the suite with coverage, Python 3.10 to
3.13, no torch), `extras` (`pytest -m vae` with torch), `lowest` (the suite with every dependency at the minimum
version in `pyproject.toml`) and `build` (the wheel is built, checked and installed in a clean environment).
`weekly.yml` runs the `dl` tests and the download of the French model.

A lower bound in `pyproject.toml` is a promise, and `lowest` keeps it. To raise or check one (the test tools go
in a second install: they must not be lowered too):

```bash
uv venv --python 3.10 /tmp/lowest && . /tmp/lowest/bin/activate
uv pip install --resolution lowest-direct -e . && uv pip install pytest pyarrow
pytest
```

## Adding a gear

`GEARSModel(gear=N)` hands every call to a backend class listed in `_GEAR_BACKENDS` (`gears/pipeline.py`).
GEAR 1st is `Gear1Backend` (`gears/pipeline_gears/gear1.py`); 2nd to 5th are reserved, and a number that is not
in the dict raises `NotImplementedError`. To add one, write `gears/pipeline_gears/gearN.py`, register the class
in the dict and test it. The facade forwards `*args, **kwargs`, so a gear is free to define its own `simulate_*`
signatures; only `fit(data, ...)` is expected to take a DataFrame or a path first. No abstract base class is
imposed, on purpose: the internals of the next gear are not known. The reasoning is in
`docs/archive/PROPOSAL_GEAR_ARCHITECTURE.md` of the private `GEARS_sandbox`.

## Pre-trained models

The official models are release files listed in `_CATALOGUE` (`gears/pretrained.py`) with their SHA-256.
A new or changed model means a new release asset and a new checksum: open an issue first. Model files
are pickles, so a pull request never contains one.

## Pull request checklist

- [ ] The change does one thing; the pull request says what was added, removed, moved or changed.
- [ ] Tests added or updated; `pytest` passes (it includes the examples and the README blocks).
- [ ] `ruff check gears tests examples` and `vulture` are clean.
- [ ] Docstrings, the README and `examples/` still describe what the code does.
- [ ] `CHANGELOG.md` has a line under `[Unreleased]`.
- [ ] No data, no `.joblib` file, no notebook output, no personal path or secret is committed.

## Reporting a bug

Open an issue with the GEARS version (`python -c "import gears; print(gears.__version__)"`), your Python
version and OS, a minimal example that fails (`make_demo_data()` gives sessions to build it on), and the
full traceback.

## Reporting a bug found in the sandbox

Experiments live in the private `GEARS_sandbox` repository, which depends on a pinned version of GEARS.
When one of them finds a bug:

1. **Reproduce it here, as a failing test**: the smallest test on `make_demo_data()` or a few lines of
   data that fails (`@pytest.mark.xfail(strict=True)` is fine while the fix is pending). Do not paste
   real data.
2. **Fix it** on a branch `fix/<short-name>` and open a pull request that links the sandbox entry
   (`docs/FINDINGS.md`) and restates the finding in a few lines, since reviewers may not see the
   sandbox. Add the line to `CHANGELOG.md`.
3. **After the release** that contains the fix, bump the pinned GEARS version in the sandbox and mark the
   entry as fixed.

An idea for an improvement is a GitHub issue labelled `enhancement`, with a link to the sandbox result.
It enters GEARS only once it is shown to work in the sandbox, and with tests.

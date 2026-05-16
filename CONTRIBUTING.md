# Contributing to GEARS

Thank you for your interest in contributing! This document explains the process.

---

## Development setup

```bash
git clone https://github.com/yvenn-amara/GEARS.git
cd GEARS
pip install -e ".[dev,notebooks]"
```

## Running tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=gears --cov-report=term-missing

# One module only
pytest tests/test_gmm.py -v
```

## Code style

```bash
ruff check gears/ tests/          # lint
black gears/ tests/               # format
mypy gears/ --ignore-missing-imports  # type hints (non-blocking)
```

## Adding a new pre-trained model

1. Prepare a sessions DataFrame (CSV/Parquet).
2. Fit a `GEARSModel` on it.
3. Add an entry to `_CATALOGUE` in `gears/models/registry.py`.
4. Call `registry.save_bundle(...)` then `registry.upload_to_hub(...)`.
5. Update `README.md` model table.

## Pull Request checklist

- [ ] Tests added / updated for new functionality
- [ ] All 260+ existing tests still pass
- [ ] Docstrings updated
- [ ] `CHANGELOG.md` entry added under `[Unreleased]`
- [ ] `ruff` and `black` pass with no errors

## Reporting bugs

Open a GitHub issue with:
- GEARS version (`python -c "import gears; print(gears.__version__)"`)
- Python version and OS
- Minimal reproducible example
- Full traceback

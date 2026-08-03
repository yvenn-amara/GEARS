"""
Config-hashed caching for gears.evaluation.benchmark runs (Session 4).

notebooks/4_persistence_vs_session_model_benchmark.ipynb is the primary consumer: by
default it loads the last cached run matching its exact config instantly
(``RERUN_BENCHMARK = False``); flipping that one flag re-runs the full
4-arm harness for real and refreshes the cache. This is what keeps the
notebook fast by default (Session 4 task 3) while staying reproducible --
the cache key is derived from the run config itself, so any change to
``x_grid``, ``arms``, ``n_scenarios``, etc. is a cache miss, never a stale
silent reuse of a different config's results.

Cache layout: ``results/benchmark_cache/<config_hash>.parquet`` (the results
table) plus a sibling ``<config_hash>.config.json`` (the exact config that
produced it, including the resolved dataset roster -- see
:func:`resolve_config` -- for human inspection, not read back
programmatically).
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

#: Default cache location, alongside the other committed benchmark evidence
#: under results/ (results/benchmark/all_results.parquet,
#: results/recency/recency_validation.csv -- Sessions 2-3's precedent).
DEFAULT_CACHE_DIR = Path("results/benchmark_cache")


def config_hash(config: dict) -> str:
    """
    Stable, order-independent hash of a run config dict.

    Parameters
    ----------
    config : dict
        JSON-serialisable run config (e.g. ``arms``, ``x_grid``,
        ``horizons``, ``n_scenarios``, ``step_days``, ``eval_window_days``,
        ``min_sessions_for_fit``, ``n_components``, ``random_state``, and
        (via :func:`resolve_config`) which datasets are included). Key
        order and list-vs-tuple don't affect the hash (JSON round-trips
        lists either way); the *values* fully determine it.

    Returns
    -------
    str
        16 hex characters (first 64 bits of a sha256 of the canonical JSON
        encoding) -- short enough for a readable filename, long enough that
        an accidental collision between two genuinely different configs is
        not a practical concern for this use case.
    """
    canonical = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def resolve_config(config: dict, dataset_names) -> dict:
    """
    Merge a run config with the dataset roster it applies to.

    The dataset roster is part of a cache entry's identity just as much as
    the numeric settings (``arms``, ``x_grid``, ...) -- hashing the bare
    settings dict alone would let a changed dataset list silently reuse a
    stale cache entry from a *different* set of datasets. Always resolve
    before hashing/loading/saving; ``load_or_run_benchmark`` does this
    internally, but callers building their own cache key by hand (e.g. to
    check a hit before deciding whether to load real data) should call this
    explicitly too, rather than hashing ``config`` on its own.

    Parameters
    ----------
    config : dict
        The numeric/arm run config (forwarded as-is to
        ``run_benchmark_for_datasets`` -- this function does not mutate it).
    dataset_names : iterable of str
        The dataset names the run does or would cover.

    Returns
    -------
    dict
        A new dict, ``config`` plus a ``"_dataset_names"`` key (sorted, for
        order-independence) -- this is what gets hashed/cached, never the
        bare ``config``.
    """
    return {**config, "_dataset_names": sorted(dataset_names)}


def cache_paths(config: dict, cache_dir: str | Path = DEFAULT_CACHE_DIR) -> tuple[Path, Path]:
    """Return ``(results_path, config_path)`` for this config's cache entry."""
    cache_dir = Path(cache_dir)
    h = config_hash(config)
    return cache_dir / f"{h}.parquet", cache_dir / f"{h}.config.json"


def load_cached(config: dict, cache_dir: str | Path = DEFAULT_CACHE_DIR) -> pd.DataFrame | None:
    """Return the cached results for ``config``, or ``None`` on a cache miss."""
    results_path, _ = cache_paths(config, cache_dir)
    if not results_path.exists():
        return None
    return pd.read_parquet(results_path)


def save_cache(
    results: pd.DataFrame, config: dict, cache_dir: str | Path = DEFAULT_CACHE_DIR,
) -> Path:
    """Write ``results`` plus the ``config`` that produced them to the cache dir."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    results_path, config_path = cache_paths(config, cache_dir)
    results.to_parquet(results_path, index=False)
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True, default=str))
    logger.info("Cached benchmark results (config hash %s, %d rows) -> %s.",
                config_hash(config), len(results), results_path)
    return results_path


def load_or_run_benchmark(
    datasets: dict[str, pd.DataFrame],
    config: dict,
    force_rerun: bool = False,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    exclude: tuple[str, ...] = ("acn",),
) -> tuple[pd.DataFrame, bool]:
    """
    Load cached rolling-origin benchmark results for ``config`` if present
    (and ``force_rerun`` is falsy); otherwise run the harness for real and
    cache the result.

    Parameters
    ----------
    datasets : dict[str, pd.DataFrame]
        Mapping dataset name -> validated sessions DataFrame. Determines
        the dataset roster half of the cache key (via
        :func:`resolve_config`) *and* is what actually gets run on a cache
        miss -- so, unlike the config, this must be the real dict even to
        check for a hit (its keys, not its values, are what's hashed).
    config : dict
        The numeric/arm run config -- forwarded as ``**config`` to
        :func:`gears.evaluation.benchmark.run_benchmark_for_datasets` when a
        fresh run is needed, and (merged with the dataset roster) hashed to
        form the cache key. Keep values JSON-serialisable (e.g. lists, not
        tuples, for ``arms``/``x_grid``).
    force_rerun : bool
        The notebook's ``RERUN_BENCHMARK`` flag -- bypass the cache and run
        for real regardless of whether a cached entry exists.
    cache_dir : str or Path
    exclude : sequence of str
        Forwarded to ``run_benchmark_for_datasets`` (default: ``("acn",)``,
        matching the harness's own default -- acn.csv is the union of
        caltech+jpl+office, Section 1.7).

    Returns
    -------
    (pd.DataFrame, bool)
        The results, and whether they came from the cache (``True``) or a
        fresh run (``False``) -- callers should report this explicitly
        rather than let a cache hit look like a fresh measurement.
    """
    from gears.evaluation.benchmark import run_benchmark_for_datasets

    full_config = resolve_config(config, datasets.keys())

    if not force_rerun:
        cached = load_cached(full_config, cache_dir)
        if cached is not None:
            logger.info("Cache hit (config hash %s) -- skipping the real run.",
                        config_hash(full_config))
            return cached, True
        logger.info("Cache miss (config hash %s) -- running fresh.", config_hash(full_config))
    else:
        logger.info("force_rerun=True -- bypassing the cache regardless of hit/miss.")

    results = run_benchmark_for_datasets(datasets, exclude=list(exclude), **config)
    save_cache(results, full_config, cache_dir)
    return results, False

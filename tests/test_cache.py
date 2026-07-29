"""Tests for gears.evaluation.cache -- the config-hashed benchmark results
cache that lets notebook 4 default to an instant cached load (Session 4
task 3), with a single flag (RERUN_BENCHMARK) to force a genuine fresh run.
"""
from __future__ import annotations

import pandas as pd

from gears.evaluation.cache import (
    cache_paths,
    config_hash,
    load_cached,
    load_or_run_benchmark,
    resolve_config,
    save_cache,
)


def test_resolve_config_adds_sorted_dataset_names():
    resolved = resolve_config({"arms": ["gmm"]}, ["boulder", "office"])
    assert resolved["_dataset_names"] == ["boulder", "office"]
    assert resolved["arms"] == ["gmm"]


def test_resolve_config_order_independent_in_dataset_names():
    a = resolve_config({"arms": ["gmm"]}, ["boulder", "office"])
    b = resolve_config({"arms": ["gmm"]}, ["office", "boulder"])
    assert config_hash(a) == config_hash(b)


def test_different_dataset_roster_is_a_cache_miss():
    """The bug this guards against: changing which datasets a run covers,
    with every other setting held fixed, must never silently reuse another
    roster's cached results."""
    config = {"arms": ["gmm"], "x_grid": [1]}
    a = resolve_config(config, ["boulder", "office"])
    b = resolve_config(config, ["boulder", "office", "paris"])
    assert config_hash(a) != config_hash(b)


def test_config_hash_stable_regardless_of_key_order():
    a = {"arms": ["persistence", "gmm"], "x_grid": [1, 4], "n_scenarios": 10}
    b = {"n_scenarios": 10, "x_grid": [1, 4], "arms": ["persistence", "gmm"]}
    assert config_hash(a) == config_hash(b)


def test_config_hash_differs_for_different_config():
    a = {"arms": ["persistence", "gmm"], "n_scenarios": 10}
    b = {"arms": ["persistence", "gmm"], "n_scenarios": 20}
    assert config_hash(a) != config_hash(b)


def test_config_hash_is_16_hex_chars():
    h = config_hash({"a": 1})
    assert len(h) == 16
    int(h, 16)  # raises ValueError if not valid hex


def test_save_and_load_cache_roundtrip(tmp_path):
    config = {"arms": ["gmm"], "x_grid": [1, 2]}
    results = pd.DataFrame({"dataset": ["office"], "status": ["ok"]})
    save_cache(results, config, cache_dir=tmp_path)
    loaded = load_cached(config, cache_dir=tmp_path)
    assert loaded is not None
    pd.testing.assert_frame_equal(loaded, results)


def test_load_cached_returns_none_when_missing(tmp_path):
    assert load_cached({"arms": ["gmm"]}, cache_dir=tmp_path) is None


def test_cache_paths_share_hash_stem(tmp_path):
    config = {"arms": ["vae"]}
    results_path, config_path = cache_paths(config, cache_dir=tmp_path)
    assert results_path.stem == config_path.stem.replace(".config", "")
    assert results_path.suffix == ".parquet"
    assert config_path.name.endswith(".config.json")


def test_load_or_run_benchmark_cache_hit_skips_rerun(tmp_path, monkeypatch):
    """On a cache hit, run_benchmark_for_datasets must never be called --
    the whole point of the cache is to avoid re-running the harness."""
    config = {"arms": ["gmm"], "x_grid": [2], "horizons": [1],
              "eval_window_days": 5, "n_scenarios": 1, "verbose": False}
    dataset_names = ["office"]
    cached_results = pd.DataFrame({"dataset": ["office"], "status": ["ok"]})
    save_cache(cached_results, resolve_config(config, dataset_names), cache_dir=tmp_path)

    def _boom(*args, **kwargs):
        raise AssertionError("run_benchmark_for_datasets should not be called on a cache hit")

    monkeypatch.setattr(
        "gears.evaluation.benchmark.run_benchmark_for_datasets", _boom,
    )
    results, from_cache = load_or_run_benchmark(
        datasets={"office": pd.DataFrame()}, config=config, force_rerun=False,
        cache_dir=tmp_path,
    )
    assert from_cache is True
    pd.testing.assert_frame_equal(results, cached_results)


def test_load_or_run_benchmark_force_rerun_bypasses_cache(tmp_path, monkeypatch):
    config = {"arms": ["gmm"], "x_grid": [2], "horizons": [1],
              "eval_window_days": 5, "n_scenarios": 1, "verbose": False}
    dataset_names = ["office"]
    stale_cached = pd.DataFrame({"dataset": ["office"], "status": ["ok"]})
    save_cache(stale_cached, resolve_config(config, dataset_names), cache_dir=tmp_path)

    fresh_results = pd.DataFrame({"dataset": ["office"], "status": ["fresh"]})
    called = {}

    def _fake_run(datasets, exclude, **kwargs):
        called["exclude"] = exclude
        called["kwargs"] = kwargs
        return fresh_results

    monkeypatch.setattr(
        "gears.evaluation.benchmark.run_benchmark_for_datasets", _fake_run,
    )
    results, from_cache = load_or_run_benchmark(
        datasets={"office": pd.DataFrame()}, config=config,
        force_rerun=True, cache_dir=tmp_path,
    )
    assert from_cache is False
    pd.testing.assert_frame_equal(results, fresh_results)
    assert called["kwargs"]["x_grid"] == [2]
    # A fresh run must also refresh the cache on disk (not just return
    # in-memory), so a later cache_hit test with the same config+datasets
    # sees it.
    reloaded = load_cached(resolve_config(config, dataset_names), cache_dir=tmp_path)
    pd.testing.assert_frame_equal(reloaded, fresh_results)


def test_load_or_run_benchmark_cache_miss_runs_and_caches(tmp_path, monkeypatch):
    config = {"arms": ["persistence"], "x_grid": [1], "horizons": [1],
              "eval_window_days": 5, "n_scenarios": 1, "verbose": False}
    dataset_names = ["office"]
    fresh_results = pd.DataFrame({"dataset": ["office"], "status": ["ok"]})

    def _fake_run(datasets, exclude, **kwargs):
        return fresh_results

    monkeypatch.setattr(
        "gears.evaluation.benchmark.run_benchmark_for_datasets", _fake_run,
    )
    # sanity: real miss before anything has been cached
    assert load_cached(resolve_config(config, dataset_names), cache_dir=tmp_path) is None
    results, from_cache = load_or_run_benchmark(
        datasets={"office": pd.DataFrame()}, config=config,
        force_rerun=False, cache_dir=tmp_path,
    )
    assert from_cache is False
    pd.testing.assert_frame_equal(results, fresh_results)
    # Second call with the identical config *and* dataset roster must now
    # hit the cache.
    results2, from_cache2 = load_or_run_benchmark(
        datasets={"office": pd.DataFrame()}, config=config,
        force_rerun=False, cache_dir=tmp_path,
    )
    assert from_cache2 is True
    pd.testing.assert_frame_equal(results2, fresh_results)

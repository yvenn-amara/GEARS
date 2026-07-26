"""Tests for gears.models.persistence_sampler and gears.evaluation.windowing."""
import numpy as np
import pandas as pd
import pytest

from gears.data.schemas import validate_dataframe
from gears.evaluation.windowing import sessions_in_last_n_occurrences
from gears.models.gmm import EVSessionGMM
from gears.models.persistence_sampler import PersistenceSessionSampler
from gears.utils import distribution_comparison


def make_pool(n=30, seed=0):
    """A tiny already-windowed pool in the arrival_hour/duration/energy shape
    PersistenceSessionSampler.fit() expects (mirrors what a caller would hand
    it after `sessions_in_last_n_occurrences` + a hour-column rename)."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "arrival_hour": rng.uniform(0, 24, n),
        "duration": rng.uniform(0.5, 8, n),
        "energy": rng.uniform(2, 40, n),
    })


def make_real_df(n=300, seed=0):
    """A validated 'real' sessions dataframe, for distribution_comparison."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "arrival_time": pd.date_range("2025-01-01", periods=n, freq="1h"),
        "duration": rng.uniform(0.5, 8, n),
        "energy": rng.uniform(2, 40, n),
    })
    return validate_dataframe(df)


# ---------------------------------------------------------------------------
# PersistenceSessionSampler.fit()
# ---------------------------------------------------------------------------

def test_fit_basic():
    pool = make_pool(30)
    sampler = PersistenceSessionSampler(random_state=42).fit(pool)
    assert sampler.is_fitted_
    assert len(sampler.pool_) == 30
    assert set(sampler.pool_.columns) == {"arrival_hour", "duration", "energy"}


def test_fit_drops_extra_columns():
    pool = make_pool(10)
    pool["day_of_week"] = 2  # extra column that should be dropped by fit()
    sampler = PersistenceSessionSampler().fit(pool)
    assert set(sampler.pool_.columns) == {"arrival_hour", "duration", "energy"}


def test_fit_missing_columns_raises():
    df = pd.DataFrame({"arrival_hour": [1, 2, 3], "duration": [1, 2, 3]})  # no energy
    with pytest.raises(ValueError, match="Missing columns"):
        PersistenceSessionSampler().fit(df)


def test_fit_empty_pool_raises():
    df = pd.DataFrame({"arrival_hour": [], "duration": [], "energy": []})
    with pytest.raises(RuntimeError, match="Empty pool"):
        PersistenceSessionSampler().fit(df)


def test_sample_before_fit_raises():
    sampler = PersistenceSessionSampler()
    with pytest.raises(RuntimeError, match="Call fit"):
        sampler.sample(10)


# ---------------------------------------------------------------------------
# PersistenceSessionSampler.sample() -- bootstrap behaviour
# ---------------------------------------------------------------------------

def test_sample_basic_shape():
    pool = make_pool(20)
    sampler = PersistenceSessionSampler(random_state=42).fit(pool)
    synth = sampler.sample(50, seed=0)
    assert len(synth) == 50
    assert {"arrival_hour", "duration", "energy"}.issubset(synth.columns)


def test_sample_more_than_pool_size():
    """Bootstrap with replacement must handle n_sessions > pool size fine."""
    pool = make_pool(5)
    sampler = PersistenceSessionSampler(random_state=42).fit(pool)
    synth = sampler.sample(500, seed=0)
    assert len(synth) == 500
    # every sampled row must be one of the (only 5) pool rows
    assert set(synth["arrival_hour"].round(8)).issubset(
        set(pool["arrival_hour"].round(8))
    )


def test_sample_values_come_from_pool():
    """Every resampled row must be an exact record from the fitted pool
    (this is a bootstrap of real records, not a generative model)."""
    pool = make_pool(15)
    sampler = PersistenceSessionSampler(random_state=1).fit(pool)
    synth = sampler.sample(200, seed=1)
    cols = ["arrival_hour", "duration", "energy"]  # fixed order for a fair comparison
    pool_tuples = set(pool[cols].itertuples(index=False, name=None))
    synth_tuples = set(synth[cols].itertuples(index=False, name=None))
    assert synth_tuples.issubset(pool_tuples)


def test_sample_determinism_same_seed():
    pool = make_pool(20)
    sampler = PersistenceSessionSampler(random_state=42).fit(pool)
    s1 = sampler.sample(30, seed=7)
    s2 = sampler.sample(30, seed=7)
    pd.testing.assert_frame_equal(s1, s2)


def test_sample_different_seeds_differ():
    pool = make_pool(200)
    sampler = PersistenceSessionSampler(random_state=42).fit(pool)
    s1 = sampler.sample(50, seed=1)
    s2 = sampler.sample(50, seed=2)
    assert not s1["arrival_hour"].equals(s2["arrival_hour"])


def test_sample_default_seed_uses_random_state():
    """Omitting `seed` should fall back to self.random_state (deterministic)."""
    pool = make_pool(20)
    sampler_a = PersistenceSessionSampler(random_state=42).fit(pool)
    sampler_b = PersistenceSessionSampler(random_state=42).fit(pool)
    pd.testing.assert_frame_equal(sampler_a.sample(10), sampler_b.sample(10))


def test_context_argument_is_ignored():
    pool = make_pool(20)
    sampler = PersistenceSessionSampler(random_state=42).fit(pool)
    s1 = sampler.sample(10, seed=5, context=None)
    s2 = sampler.sample(10, seed=5, context={"day_of_week": 3})
    pd.testing.assert_frame_equal(s1, s2)


# ---------------------------------------------------------------------------
# arrival_time reconstruction from `date`
# ---------------------------------------------------------------------------

def test_sample_with_date_adds_arrival_time():
    pool = make_pool(20)
    sampler = PersistenceSessionSampler(random_state=42).fit(pool)
    synth = sampler.sample(20, date="2025-06-15", seed=1)
    assert "arrival_time" in synth.columns
    # arrival_time must fall on the requested calendar day
    assert (synth["arrival_time"].dt.normalize() == pd.Timestamp("2025-06-15")).all()


def test_sample_without_date_has_no_arrival_time():
    pool = make_pool(20)
    sampler = PersistenceSessionSampler(random_state=42).fit(pool)
    synth = sampler.sample(20, seed=1)
    assert "arrival_time" not in synth.columns


def test_arrival_time_matches_arrival_hour():
    pool = make_pool(20)
    sampler = PersistenceSessionSampler(random_state=42).fit(pool)
    synth = sampler.sample(20, date="2025-03-10", seed=2)
    reconstructed_hour = (
        synth["arrival_time"] - synth["arrival_time"].dt.normalize()
    ).dt.total_seconds() / 3600.0
    np.testing.assert_allclose(reconstructed_hour.values, synth["arrival_hour"].values, atol=1e-6)


# ---------------------------------------------------------------------------
# Interface parity with EVSessionGMM.sample() (Session 2 acceptance criteria)
# ---------------------------------------------------------------------------

def test_column_parity_with_gmm_no_date():
    pool_gmm = pd.DataFrame({
        "arrival_time": pd.date_range("2025-01-01", periods=200, freq="2h"),
        "duration": np.random.default_rng(0).uniform(0.5, 8, 200),
        "energy": np.random.default_rng(1).uniform(2, 40, 200),
    })
    pool_gmm = validate_dataframe(pool_gmm)
    gmm = EVSessionGMM(n_components=1, stratify_by=["day_of_week"]).fit(pool_gmm)
    gmm_out = gmm.sample(25, seed=0)

    pool_persist = make_pool(30)
    sampler = PersistenceSessionSampler(random_state=42).fit(pool_persist)
    persist_out = sampler.sample(25, seed=0)

    assert set(persist_out.columns) == set(gmm_out.columns)


def test_column_parity_with_gmm_with_date():
    pool_gmm = pd.DataFrame({
        "arrival_time": pd.date_range("2025-01-01", periods=200, freq="2h"),
        "duration": np.random.default_rng(0).uniform(0.5, 8, 200),
        "energy": np.random.default_rng(1).uniform(2, 40, 200),
    })
    pool_gmm = validate_dataframe(pool_gmm)
    gmm = EVSessionGMM(n_components=1, stratify_by=["day_of_week"]).fit(pool_gmm)
    gmm_out = gmm.sample(25, date="2025-06-15", seed=0)

    pool_persist = make_pool(30)
    sampler = PersistenceSessionSampler(random_state=42).fit(pool_persist)
    persist_out = sampler.sample(25, date="2025-06-15", seed=0)

    assert set(persist_out.columns) == set(gmm_out.columns)


def test_distribution_comparison_runs_unmodified():
    """gears.utils.distribution_comparison must accept the sampler's output
    exactly like it accepts EVSessionGMM's output, with no modification."""
    real = make_real_df(300)
    pool = pd.DataFrame({
        "arrival_hour": real["hour"].values,
        "duration": real["duration"].values,
        "energy": real["energy"].values,
    })
    sampler = PersistenceSessionSampler(random_state=42).fit(pool)
    synth = sampler.sample(300, seed=0)

    metrics = distribution_comparison(real, synth)
    assert len(metrics) == 3
    assert set(metrics["feature"]) == {"hour", "duration", "energy"}
    for col in ["wasserstein", "kl_divergence", "ks_statistic", "ks_pvalue"]:
        assert col in metrics.columns
    # bootstrapping from the real pool itself should match near-perfectly
    assert metrics.loc[metrics["feature"] == "hour", "wasserstein"].item() < 1.0


# ---------------------------------------------------------------------------
# sessions_in_last_n_occurrences (shared windowing utility)
# ---------------------------------------------------------------------------

def _make_daily_history(start="2024-01-01", n_days=120, sessions_per_day=3, seed=0):
    """One synthetic dataset spanning n_days, sessions_per_day sessions/day."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n_days, freq="D")
    rows = []
    for d in dates:
        for _ in range(sessions_per_day):
            hour = rng.uniform(0, 24)
            rows.append({"arrival_time": d + pd.Timedelta(hours=hour)})
    return pd.DataFrame(rows)


def test_windowing_no_leakage_past_target():
    df = _make_daily_history(n_days=60)
    target = pd.Timestamp("2024-02-01")  # Thursday
    pool, _info = sessions_in_last_n_occurrences(df, target, n=4)
    assert (pool["arrival_time"] < target).all()


def test_windowing_only_same_weekday():
    df = _make_daily_history(n_days=60)
    target = pd.Timestamp("2024-02-01")
    pool, _info = sessions_in_last_n_occurrences(df, target, n=4)
    assert (pool["arrival_time"].dt.dayofweek == target.dayofweek).all()


def test_windowing_respects_n_occurrences():
    df = _make_daily_history(n_days=120, sessions_per_day=3)
    target = pd.Timestamp("2024-04-15")
    _pool, info = sessions_in_last_n_occurrences(df, target, n=4)
    assert info["n_available_occurrences"] == 4
    assert info["n_sessions"] == 4 * 3
    assert info["insufficient_history"] is False


def test_windowing_insufficient_history():
    """Requesting more occurrences than exist must be flagged, not silently
    degraded to a smaller X."""
    df = _make_daily_history(n_days=20, sessions_per_day=2)  # < 3 weeks of history
    target = pd.Timestamp("2024-01-20")
    pool, info = sessions_in_last_n_occurrences(df, target, n=52)
    assert info["insufficient_history"] is True
    assert info["n_available_occurrences"] < 52
    # the pool itself must still only contain what *is* available -- never
    # mislabeled as if the full X=52 had been satisfied.
    assert info["n_available_occurrences"] == len(pool["arrival_time"].dt.normalize().unique())


def test_windowing_empty_history_before_target():
    df = _make_daily_history(start="2024-06-01", n_days=5, sessions_per_day=2)
    target = pd.Timestamp("2024-06-01")  # nothing strictly before this
    pool, info = sessions_in_last_n_occurrences(df, target, n=1)
    assert len(pool) == 0
    assert info["insufficient_history"] is True
    assert info["n_sessions"] == 0

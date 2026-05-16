"""Tests for gears.models.gmm."""
import numpy as np
import pandas as pd
import pytest
import tempfile
from pathlib import Path
from gears.data.schemas import validate_dataframe
from gears.models.gmm import EVSessionGMM


def make_df(n=500, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "arrival_time": pd.date_range("2025-01-01", periods=n, freq="1h"),
        "duration": rng.uniform(0.5, 8, n),
        "energy": rng.uniform(2, 40, n),
        "location_type": rng.choice(["work", "home", "public"], n),
    })
    return validate_dataframe(df)


def test_fit_basic():
    df = make_df(300)
    gmm = EVSessionGMM(n_components=3, stratify_by=["day_of_week", "season"])
    gmm.fit(df)
    assert gmm.is_fitted_
    assert len(gmm.models_) > 0


def test_fit_auto():
    df = make_df(500)
    gmm = EVSessionGMM(n_components="auto", max_components=5)
    gmm.fit(df)
    bic = gmm.bic_summary()
    assert all(bic["n_components"] >= 2)


def test_sample_basic():
    df = make_df(500)
    gmm = EVSessionGMM(n_components=3).fit(df)
    synth = gmm.sample(50, seed=0)
    assert len(synth) == 50
    assert set(["arrival_hour", "duration", "energy"]).issubset(synth.columns)
    assert synth["duration"].min() >= 0
    assert synth["energy"].min() >= 0


def test_sample_with_date():
    df = make_df(500)
    gmm = EVSessionGMM(n_components=3).fit(df)
    synth = gmm.sample(20, date="2025-06-15", seed=1)
    assert "arrival_time" in synth.columns


def test_context_fallback():
    """Unknown context should fall back gracefully."""
    df = make_df(500)
    gmm = EVSessionGMM(n_components=3).fit(df)
    # Non-existent context
    synth = gmm.sample(10, context={"day_of_week": 99, "season": "unknown"})
    assert len(synth) == 10


def test_get_sklearn_gmm():
    df = make_df(500)
    gmm = EVSessionGMM(n_components=3).fit(df)
    from sklearn.mixture import GaussianMixture
    sk = gmm.get_sklearn_gmm()
    assert isinstance(sk, GaussianMixture)
    assert hasattr(sk, "means_")
    assert hasattr(sk, "weights_")


def test_list_contexts():
    df = make_df(500)
    gmm = EVSessionGMM(n_components=3).fit(df)
    ctxs = gmm.list_contexts()
    assert len(ctxs) > 0
    assert all(isinstance(c, tuple) for c in ctxs)


def test_score():
    df = make_df(500)
    gmm = EVSessionGMM(n_components=3).fit(df)
    score = gmm.score(df)
    assert np.isfinite(score)


def test_is_sample_flag():
    df = make_df(200)
    gmm = EVSessionGMM(n_components=2).fit(df, is_sample=True)
    assert gmm.is_sample_ is True


def test_save_load():
    df = make_df(300)
    gmm = EVSessionGMM(n_components=3).fit(df)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "gmm_test.joblib"
        gmm.save(path)
        loaded = EVSessionGMM.load(path)
    assert loaded.is_fitted_
    assert len(loaded.models_) == len(gmm.models_)


def test_max_samples_per_context():
    df = make_df(2000)
    gmm = EVSessionGMM(n_components=3, max_samples_per_context=50).fit(df)
    assert gmm.is_fitted_


def test_repr_unfitted():
    gmm = EVSessionGMM()
    r = repr(gmm)
    assert "fitted=False" in r


def test_repr_fitted():
    df = make_df(300)
    gmm = EVSessionGMM(n_components=3).fit(df)
    r = repr(gmm)
    assert "fitted=True" in r

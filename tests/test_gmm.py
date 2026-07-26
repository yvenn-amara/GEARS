"""Tests for gears.models.gmm."""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

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
    assert {"arrival_hour", "duration", "energy"}.issubset(synth.columns)
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


# ── VAE tests (fast — tiny synthetic data) ───────────────────────────────────

def make_df_vae(n=300, seed=0):
    """Synthetic dataset with department column for VAE stratification."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "arrival_time": pd.date_range("2025-01-01", periods=n, freq="2h"),
        "duration": rng.uniform(0.5, 8, n),
        "energy": rng.uniform(2, 40, n),
        "location_type": rng.choice(["work", "home"], n),
        "department": rng.choice(["75", "69"], n),
    })
    return validate_dataframe(df)


def _fit_tiny_vae(n=300, seed=0):
    """Helper: fit a minimal VAE for use in multiple tests."""
    df = make_df_vae(n=n, seed=seed)
    model = EVSessionGMM(
        model_type="vae",
        stratify_by=["location_type", "department", "day_of_week", "season"],
        max_samples_per_context=30,
        vae_epochs=3,
        vae_hidden_dim=32,
        vae_latent_dim=4,
        vae_batch_size=32,
        random_state=seed,
    )
    model.fit(df)
    return model, df


def test_vae_fit_basic():
    model, _ = _fit_tiny_vae()
    assert model.is_fitted_
    assert model.model_type == "vae"
    assert len(model.models_) > 0


def test_vae_models_are_context_slices():
    """Each value in models_ must be a VAEContextSlice (not sklearn GMM)."""
    from gears.models.vae import VAEContextSlice
    model, _ = _fit_tiny_vae()
    for slice_ in model.models_.values():
        assert isinstance(slice_, VAEContextSlice)


def test_vae_sample_shape():
    model, _ = _fit_tiny_vae()
    synth = model.sample(20, seed=0)
    assert len(synth) == 20
    assert {"arrival_hour", "duration", "energy"}.issubset(synth.columns)
    assert synth["duration"].min() >= 0
    assert synth["energy"].min() >= 0


def test_vae_score_finite():
    model, df = _fit_tiny_vae()
    s = model.score(df)
    assert np.isfinite(s)


def test_vae_save_load():
    model, df = _fit_tiny_vae()
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "vae_test.joblib"
        model.save(p)
        loaded = EVSessionGMM.load(p)
    assert loaded.is_fitted_
    assert loaded.model_type == "vae"
    assert len(loaded.models_) == len(model.models_)
    synth = loaded.sample(10, seed=42)
    assert len(synth) == 10
    assert np.isfinite(loaded.score(df))


def test_vae_list_contexts():
    model, _ = _fit_tiny_vae()
    ctxs = model.list_contexts()
    assert len(ctxs) > 0
    assert all(isinstance(c, tuple) for c in ctxs)


def test_vae_bic_summary():
    """bic_summary() must work for VAE — n_components=1 for all slices."""
    model, _ = _fit_tiny_vae()
    bic = model.bic_summary()
    assert len(bic) > 0
    assert "n_components" in bic.columns
    assert (bic["n_components"] == 1).all()


def test_vae_context_slice_sklearn_api():
    """VAEContextSlice must satisfy the duck-type contract used by aggregator.py
    and plotting.py: random_state, sample, score_samples, n_components,
    means_, weights_."""
    model, _ = _fit_tiny_vae()
    slice_ = next(iter(model.models_.values()))

    # random_state settable (aggregator / medium_term pattern)
    slice_.random_state = 7
    assert slice_.random_state == 7

    # sample returns (ndarray, None)
    raw, none_val = slice_.sample(5)
    assert isinstance(raw, np.ndarray)
    assert raw.shape == (5, 3)
    assert none_val is None

    # score_samples returns 1-D array
    lp = slice_.score_samples(raw)
    assert isinstance(lp, np.ndarray)
    assert lp.shape == (5,)
    assert np.all(np.isfinite(lp))

    # plotting attributes
    assert slice_.n_components == 1
    assert np.allclose(slice_.weights_, [1.0])
    assert slice_.means_.shape == (1, 3)


def test_vae_n_sessions_per_day_populated():
    """n_sessions_per_day_ must be populated identically to GMM path."""
    model, _ = _fit_tiny_vae()
    assert len(model.n_sessions_per_day_) == len(model.models_)
    assert all(v > 0 for v in model.n_sessions_per_day_.values())


def test_vae_context_counts_populated():
    model, _ = _fit_tiny_vae()
    assert len(model.context_counts_) == len(model.models_)
    assert all(v > 0 for v in model.context_counts_.values())


def test_vae_is_sample_flag():
    df = make_df_vae(200)
    model = EVSessionGMM(
        model_type="vae", vae_epochs=2, vae_hidden_dim=16,
        vae_latent_dim=4, vae_batch_size=32,
        stratify_by=["location_type", "day_of_week", "season"],
        max_samples_per_context=20,
    ).fit(df, is_sample=True)
    assert model.is_sample_ is True

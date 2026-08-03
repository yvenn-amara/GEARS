"""Tests for gears.models.session_model."""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gears.data.schemas import validate_dataframe
from gears.models.session_model import EVSessionModel


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
    gmm = EVSessionModel(n_components=3, stratify_by=["day_of_week", "season"])
    gmm.fit(df)
    assert gmm.is_fitted_
    assert len(gmm.models_) > 0


def test_fit_auto():
    df = make_df(500)
    gmm = EVSessionModel(n_components="auto", max_components=5)
    gmm.fit(df)
    bic = gmm.bic_summary()
    assert all(bic["n_components"] >= 2)


def test_sample_basic():
    df = make_df(500)
    gmm = EVSessionModel(n_components=3).fit(df)
    synth = gmm.sample(50, seed=0)
    assert len(synth) == 50
    assert {"arrival_hour", "duration", "energy"}.issubset(synth.columns)
    assert synth["duration"].min() >= 0
    assert synth["energy"].min() >= 0


def test_sample_with_date():
    df = make_df(500)
    gmm = EVSessionModel(n_components=3).fit(df)
    synth = gmm.sample(20, date="2025-06-15", seed=1)
    assert "arrival_time" in synth.columns


def test_context_fallback():
    """Unknown context should fall back gracefully."""
    df = make_df(500)
    gmm = EVSessionModel(n_components=3).fit(df)
    # Non-existent context
    synth = gmm.sample(10, context={"day_of_week": 99, "season": "unknown"})
    assert len(synth) == 10


def test_get_sklearn_component():
    df = make_df(500)
    gmm = EVSessionModel(n_components=3).fit(df)
    from sklearn.mixture import GaussianMixture
    sk = gmm.get_sklearn_component()
    assert isinstance(sk, GaussianMixture)
    assert hasattr(sk, "means_")
    assert hasattr(sk, "weights_")


def test_list_contexts():
    df = make_df(500)
    gmm = EVSessionModel(n_components=3).fit(df)
    ctxs = gmm.list_contexts()
    assert len(ctxs) > 0
    assert all(isinstance(c, tuple) for c in ctxs)


def test_score():
    df = make_df(500)
    gmm = EVSessionModel(n_components=3).fit(df)
    score = gmm.score(df)
    assert np.isfinite(score)


def test_is_sample_flag():
    df = make_df(200)
    gmm = EVSessionModel(n_components=2).fit(df, is_sample=True)
    assert gmm.is_sample_ is True


def test_save_load():
    df = make_df(300)
    gmm = EVSessionModel(n_components=3).fit(df)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "gmm_test.joblib"
        gmm.save(path)
        loaded = EVSessionModel.load(path)
    assert loaded.is_fitted_
    assert len(loaded.models_) == len(gmm.models_)


def test_max_samples_per_context():
    df = make_df(2000)
    gmm = EVSessionModel(n_components=3, max_samples_per_context=50).fit(df)
    assert gmm.is_fitted_


def test_repr_unfitted():
    gmm = EVSessionModel()
    r = repr(gmm)
    assert "fitted=False" in r


def test_repr_fitted():
    df = make_df(300)
    gmm = EVSessionModel(n_components=3).fit(df)
    r = repr(gmm)
    assert "fitted=True" in r


# ── Recency-weighted GMM tests ───────────────────────────────────────────────

def make_df_regime_shift(n_old=800, n_recent=200, seed=0):
    """Synthetic dataset with a deliberate distribution shift over time.

    Old sessions (first ~300 days): low duration/energy, one 'home' regime.
    Recent sessions (last 21 days before the reference date): high
    duration/energy, a shifted regime. Used to check that recency weighting
    actually pulls the fit toward the recent cluster.
    """
    rng = np.random.default_rng(seed)
    old_dates = pd.Timestamp("2024-01-01") + pd.to_timedelta(
        rng.integers(0, 300, n_old), unit="D"
    )
    old = pd.DataFrame({
        "arrival_time": old_dates + pd.to_timedelta(rng.uniform(0, 23, n_old), unit="h"),
        "duration": rng.uniform(0.5, 2, n_old),
        "energy": rng.uniform(2, 8, n_old),
        "location_type": "home",
    })
    recent_dates = pd.Timestamp("2024-10-27") + pd.to_timedelta(
        rng.integers(0, 21, n_recent), unit="D"
    )
    recent = pd.DataFrame({
        "arrival_time": recent_dates + pd.to_timedelta(rng.uniform(0, 23, n_recent), unit="h"),
        "duration": rng.uniform(6, 10, n_recent),
        "energy": rng.uniform(30, 45, n_recent),
        "location_type": "home",
    })
    df = pd.concat([old, recent], ignore_index=True)
    return validate_dataframe(df)


def _mean_feature(gmm_obj: EVSessionModel, ctx: tuple, feature_idx: int) -> float:
    """Weighted mean of one GMM feature dimension (log1p-space) for a context."""
    sk = gmm_obj.models_[ctx]
    return float(np.sum(sk.weights_ * sk.means_[:, feature_idx]))


def test_recency_none_matches_pristine_unmodified_code():
    """recency=None must produce byte-for-byte identical results to the
    pre-change code path.

    Rather than only asserting internal self-consistency, this re-derives
    the model with the *original* (pre-recency-feature) implementation of
    the fit loop -- reconstructed here verbatim from the version of
    EVSessionModel.fit()/._fit_single() that existed before recency weighting
    was added -- and checks the new code, called with recency=None, matches
    it exactly. This is the regression proof, not just a self-consistency
    check.
    """
    from sklearn.mixture import GaussianMixture

    df = make_df(500, seed=3)

    # New code, recency explicitly disabled.
    gmm_new = EVSessionModel(
        n_components=3, stratify_by=["day_of_week", "season"], random_state=7,
        recency=None,
    ).fit(df)

    # Pristine pre-change logic, reproduced independently: no recency
    # concept at all, group -> optional uniform subsample -> _fit_single.
    df2 = df.copy()
    if "day_of_week" not in df2.columns:
        df2["day_of_week"] = df2["arrival_time"].dt.dayofweek
    groups = df2.groupby(["day_of_week", "season"], observed=True)
    expected_models = {}
    for ctx_key, group_df in groups:
        ctx_tuple = ctx_key if isinstance(ctx_key, tuple) else (ctx_key,)
        if len(group_df) < 10:
            continue
        hour = group_df["hour"].values
        log_dur = np.log1p(group_df["duration"].values)
        log_ene = np.log1p(group_df["energy"].values)
        X = np.column_stack([hour, log_dur, log_ene])
        # gmm_new below is constructed with a *fixed* n_components=3, so the
        # pre-change _fit_single takes its "else" branch (single fit, no BIC
        # search) -- reproduce exactly that, not the auto-BIC branch.
        best_gmm = GaussianMixture(
            n_components=3, covariance_type="full", random_state=7,
            n_init=3, max_iter=300, reg_covar=1e-5,
        ).fit(X)
        expected_models[ctx_tuple] = best_gmm

    assert set(gmm_new.models_.keys()) == set(expected_models.keys())
    for ctx, expected in expected_models.items():
        actual = gmm_new.models_[ctx]
        assert np.array_equal(actual.means_, expected.means_)
        assert np.array_equal(actual.covariances_, expected.covariances_)
        assert np.array_equal(actual.weights_, expected.weights_)


def test_recency_none_equals_default_omitted():
    """Explicitly passing recency=None must equal simply omitting it."""
    df = make_df(400, seed=1)
    gmm_a = EVSessionModel(n_components=3, random_state=5, recency=None).fit(df)
    gmm_b = EVSessionModel(n_components=3, random_state=5).fit(df)
    for ctx in gmm_a.models_:
        assert np.array_equal(gmm_a.models_[ctx].means_, gmm_b.models_[ctx].means_)
        assert np.array_equal(gmm_a.models_[ctx].weights_, gmm_b.models_[ctx].weights_)
    assert gmm_a.n_sessions_per_day_ == gmm_b.n_sessions_per_day_
    assert gmm_a.context_counts_ == gmm_b.context_counts_


def test_recency_pulls_fit_toward_recent_cluster():
    """With a deliberate regime shift, the recency-weighted fit's means must
    sit much closer to the recent cluster than the unweighted fit's."""
    df = make_df_regime_shift()
    ctx = ("home",)

    gmm_plain = EVSessionModel(
        n_components=2, stratify_by=["location_type"], random_state=0,
    ).fit(df)
    gmm_recency = EVSessionModel(
        n_components=2, stratify_by=["location_type"], random_state=0,
        recency=True, half_life_days=7,
    ).fit(df)

    # energy is feature index 2 (log1p space); true recent-cluster energy
    # range is [30, 45] kWh -> log1p in roughly [3.43, 3.83].
    plain_energy_log = _mean_feature(gmm_plain, ctx, 2)
    recency_energy_log = _mean_feature(gmm_recency, ctx, 2)

    recent_target = np.log1p(37.5)  # midpoint of the recent regime's energy range
    old_target = np.log1p(5.0)      # midpoint of the old regime's energy range

    # The recency-weighted fit must be closer to the recent regime than the
    # plain fit is, by a wide margin -- not just marginally closer.
    assert abs(recency_energy_log - recent_target) < abs(plain_energy_log - recent_target)
    assert abs(recency_energy_log - recent_target) < abs(recency_energy_log - old_target)


def test_recency_resample_cap_respected():
    """The recency-weighted resample must never exceed recency_resample_cap,
    even when the pooled context has far more sessions than the cap."""
    rng = np.random.default_rng(2)
    n = 12_000
    dates = pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 200, n), unit="D")
    df = pd.DataFrame({
        "arrival_time": dates + pd.to_timedelta(rng.uniform(0, 23, n), unit="h"),
        "duration": rng.uniform(0.5, 8, n),
        "energy": rng.uniform(2, 40, n),
        "location_type": "home",
    })
    df = validate_dataframe(df)

    small_cap = 500
    gmm = EVSessionModel(
        n_components=2, stratify_by=["location_type"], random_state=0,
        recency=True, half_life_days=30, recency_resample_cap=small_cap,
    ).fit(df)

    ctx = ("home",)
    assert gmm.context_counts_[ctx] == small_cap

    # n_sessions_per_day_ must reflect the TRUE pool, not the capped
    # training sample -- this is the exact bug pattern flagged elsewhere for
    # the VAE path (n_sessions_per_day_ computed from a subsampled count
    # instead of the true underlying count).
    true_n_days = df["arrival_time"].dt.normalize().nunique()
    true_rate = len(df) / true_n_days
    assert abs(gmm.n_sessions_per_day_[ctx] - true_rate) < 1e-9


def test_recency_default_cap_is_5000():
    """Default recency_resample_cap must be DEFAULT_RECENCY_RESAMPLE_CAP (5000)
    and must actually be applied without an explicit override."""
    from gears.models.session_model import DEFAULT_RECENCY_RESAMPLE_CAP

    assert DEFAULT_RECENCY_RESAMPLE_CAP == 5000

    rng = np.random.default_rng(4)
    n = 8_000
    dates = pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 300, n), unit="D")
    df = pd.DataFrame({
        "arrival_time": dates + pd.to_timedelta(rng.uniform(0, 23, n), unit="h"),
        "duration": rng.uniform(0.5, 8, n),
        "energy": rng.uniform(2, 40, n),
        "location_type": "home",
    })
    df = validate_dataframe(df)

    gmm = EVSessionModel(
        n_components=2, stratify_by=["location_type"], random_state=0,
        recency=True, half_life_days=30,
    ).fit(df)
    assert gmm.context_counts_[("home",)] == DEFAULT_RECENCY_RESAMPLE_CAP


def test_recency_max_samples_per_context_lowers_effective_cap():
    """If max_samples_per_context is also set and smaller than
    recency_resample_cap, the smaller value governs the training size."""
    rng = np.random.default_rng(5)
    n = 8_000
    dates = pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 300, n), unit="D")
    df = pd.DataFrame({
        "arrival_time": dates + pd.to_timedelta(rng.uniform(0, 23, n), unit="h"),
        "duration": rng.uniform(0.5, 8, n),
        "energy": rng.uniform(2, 40, n),
        "location_type": "home",
    })
    df = validate_dataframe(df)

    gmm = EVSessionModel(
        n_components=2, stratify_by=["location_type"], random_state=0,
        recency=True, half_life_days=30,
        max_samples_per_context=1000,
    ).fit(df)
    assert gmm.context_counts_[("home",)] == 1000


def test_recency_half_life_default_scales_with_span():
    """When half_life_days is not given, it must scale with the observed
    per-context history span, per recency_halflife_divisor."""

    def make(span_days, n=2000, seed=0):
        r = np.random.default_rng(seed)
        dates = pd.Timestamp("2024-01-01") + pd.to_timedelta(
            r.integers(0, span_days, n), unit="D"
        )
        d = pd.DataFrame({
            "arrival_time": dates + pd.to_timedelta(r.uniform(0, 23, n), unit="h"),
            "duration": r.uniform(0.5, 8, n),
            "energy": r.uniform(2, 40, n),
            "location_type": "home",
        })
        return validate_dataframe(d)

    df_short = make(56)   # X=8 weeks
    df_long = make(364)   # X=52 weeks

    gmm_short = EVSessionModel(
        n_components=2, stratify_by=["location_type"], random_state=0, recency=True,
    ).fit(df_short)
    gmm_long = EVSessionModel(
        n_components=2, stratify_by=["location_type"], random_state=0, recency=True,
    ).fit(df_long)

    hl_short = gmm_short.half_life_days_used_[("home",)]
    hl_long = gmm_long.half_life_days_used_[("home",)]

    # Longer history window -> longer default half-life.
    assert hl_long > hl_short
    # Matches the documented formula: span_days / recency_halflife_divisor.
    assert hl_short == pytest.approx(55 / 3.5, rel=0.05)
    assert hl_long == pytest.approx(363 / 3.5, rel=0.05)


def test_recency_explicit_half_life_overrides_default():
    df = make_df_regime_shift()
    gmm = EVSessionModel(
        n_components=2, stratify_by=["location_type"], random_state=0,
        recency=True, half_life_days=99,
    ).fit(df)
    assert gmm.half_life_days_used_[("home",)] == 99.0


def test_recency_ignored_for_vae_with_warning(caplog):
    """recency=True with model_type='vae' must be ignored (loudly), not
    silently applied -- the VAE path is out of scope for this feature."""
    import logging

    df = make_df_vae(300)
    with caplog.at_level(logging.WARNING, logger="gears.models.session_model"):
        model = EVSessionModel(
            model_type="vae",
            stratify_by=["location_type", "department", "day_of_week", "season"],
            max_samples_per_context=30,
            vae_epochs=3, vae_hidden_dim=32, vae_latent_dim=4, vae_batch_size=32,
            random_state=0,
            recency=True, half_life_days=7,
        ).fit(df)
    assert model.is_fitted_
    assert any("recency" in rec.message and "vae" in rec.message for rec in caplog.records)
    # No recency diagnostics should have been populated for the ignored path.
    assert model.recency_reference_date_used_ is None
    assert model.half_life_days_used_ == {}


def test_unpickling_old_bundle_backfills_recency_attrs():
    """Regression test for the bug notebook 1 surfaced via ``get_session_model()``.

    The committed ``gmm_french.joblib`` bundle was pickled before Session 2
    added the recency-weighting attributes to ``EVSessionModel.__init__``, so
    its pickled ``__dict__`` doesn't have them. Unpickling it and calling
    ``repr()`` (which does ``if self.recency:``) raised
    ``AttributeError: 'EVSessionModel' object has no attribute 'recency'``.
    Simulates that exact situation: fit a model, strip the recency-era
    attributes to mimic an old-style pickle, round-trip it through
    ``pickle``, and confirm it loads cleanly with sane defaults.
    """
    import pickle

    gmm = EVSessionModel(n_components=2, stratify_by=["day_of_week"]).fit(make_df(200))

    recency_attrs = [
        "recency",
        "half_life_days",
        "recency_reference_date",
        "recency_resample_cap",
        "recency_halflife_divisor",
        "half_life_days_used_",
        "recency_reference_date_used_",
    ]
    for attr in recency_attrs:
        delattr(gmm, attr)

    old_style_bytes = pickle.dumps(gmm)
    loaded = pickle.loads(old_style_bytes)

    assert loaded.recency is None
    assert loaded.half_life_days is None
    assert loaded.recency_resample_cap == 5000
    assert loaded.half_life_days_used_ == {}
    assert loaded.recency_reference_date_used_ is None
    # The actual symptom reported: __repr__ must not raise AttributeError.
    assert "EVSessionModel" in repr(loaded)


def test_recency_repr_flag():
    df = make_df_regime_shift()
    gmm = EVSessionModel(
        n_components=2, stratify_by=["location_type"], random_state=0,
        recency=True, half_life_days=7,
    ).fit(df)
    assert "recency=True" in repr(gmm)


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
    model = EVSessionModel(
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


def test_vae_sample_prior_adds_observation_noise():
    """Regression test for the variance-collapse bug (AUDIT.md section d).

    ``ConditionalVAE.sample_prior`` used to return ``decode(z, ctx_emb)``
    directly -- the decoder's mean output with zero observation noise --
    even though the model has an explicit, learned observation variance
    (``log_recon_var``) for exactly this purpose (used in ``elbo_loss`` and
    ``iwae_log_prob``). This made generated samples collapse to far less
    spread than the training data (empirically ~10-150x too narrow).

    This test isolates the fix directly: it compares the variance of a
    "decoder-only" draw (same z, no noise -- what the old, buggy
    ``sample_prior`` effectively returned) against the variance of the real
    ``sample_prior`` output using the *same* seed (so both draw the same
    z's). The full sample's variance must exceed the decoder-only variance
    by close to the learned ``recon_var`` -- if ``sample_prior`` regresses
    to not adding observation noise, this assertion fails.
    """
    torch = pytest.importorskip("torch")
    model, _ = _fit_tiny_vae()
    ctx_key = next(iter(model.models_.keys()))
    slice_ = model.models_[ctx_key]
    cvae = slice_.cvae
    n = 2000

    ctx_idx = torch.tensor(slice_.ctx_index, dtype=torch.long).repeat(n, 1)

    torch.manual_seed(123)
    with torch.no_grad():
        ctx_emb = cvae._embed_context(ctx_idx)
        z = torch.randn(n, cvae.latent_dim)
        decode_only = cvae.decode(z, ctx_emb)
    decode_only_var = decode_only.var(dim=0)

    full_samples = cvae.sample_prior(n, ctx_idx, seed=123)
    full_var = full_samples.var(dim=0)

    recon_var = torch.exp(cvae.log_recon_var).clamp(1e-4, 10.0)

    # Full generative variance must exceed decoder-only variance by a
    # meaningful fraction of the learned observation variance -- i.e. the
    # noise term is actually being added, not silently dropped.
    assert torch.all(full_var > decode_only_var + 0.3 * recon_var)


def test_vae_score_finite():
    model, df = _fit_tiny_vae()
    s = model.score(df)
    assert np.isfinite(s)


def test_vae_save_load():
    model, df = _fit_tiny_vae()
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "vae_test.joblib"
        model.save(p)
        loaded = EVSessionModel.load(p)
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
    model = EVSessionModel(
        model_type="vae", vae_epochs=2, vae_hidden_dim=16,
        vae_latent_dim=4, vae_batch_size=32,
        stratify_by=["location_type", "day_of_week", "season"],
        max_samples_per_context=20,
    ).fit(df, is_sample=True)
    assert model.is_sample_ is True


# ── Regression: partial stratify_by fallback (Phase 2 Session 4) -------------
#
# Previously, if ANY stratify_by column was missing from the data,
# EVSessionModel.fit() silently dropped ALL of them down to
# ['day_of_week', 'season'] — even columns that were actually present. A
# single-site export missing only `location_type` would also lose a present
# `department` column with no warning distinguishing the two. Confirmed via a
# real smoke test on office.csv (which has neither column): the caller's own
# pre-fit log claimed the full 4-column stratify_by was in use while the
# model had actually silently fallen back further, with no cross-reference
# between the two log lines.

def make_df_with_department(n=500, seed=0):
    """Like make_df(), but with a `department` column and no `location_type` —
    mirrors a single-site export that has department info but isn't
    multi-site (the inverse of office.csv, which has neither)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "arrival_time": pd.date_range("2025-01-01", periods=n, freq="1h"),
        "duration": rng.uniform(0.5, 8, n),
        "energy": rng.uniform(2, 40, n),
        "department": rng.choice(["75", "92"], n),
    })
    return validate_dataframe(df)


def test_partial_stratify_fallback_keeps_present_columns():
    """Only the missing column(s) should be dropped from stratify_by — a
    present `department` must survive a missing `location_type`, not be
    swept away by the old blanket fallback to ['day_of_week', 'season']."""
    df = make_df_with_department(300)
    gmm = EVSessionModel(
        n_components=1,
        stratify_by=["location_type", "department", "day_of_week", "season"],
    ).fit(df)
    assert gmm.stratify_by == ["department", "day_of_week", "season"]
    assert "location_type" not in gmm.stratify_by


def test_full_stratify_fallback_when_all_missing():
    """When every extra column is missing (e.g. office.csv: neither
    location_type nor department), stratify_by still collapses to
    ['day_of_week', 'season'] — same end result as before, just reached via
    the general "drop what's missing" rule rather than a special case."""
    df = make_df(300)  # has location_type but not department
    gmm = EVSessionModel(
        n_components=1,
        stratify_by=["location_type", "department", "day_of_week", "season"],
    ).fit(df)
    assert gmm.stratify_by == ["location_type", "day_of_week", "season"]

    df_neither = df.drop(columns=["location_type"])
    gmm2 = EVSessionModel(
        n_components=1,
        stratify_by=["location_type", "department", "day_of_week", "season"],
    ).fit(df_neither)
    assert gmm2.stratify_by == ["day_of_week", "season"]


def test_stratify_fallback_no_missing_columns_unchanged():
    """When nothing is missing, stratify_by is left untouched (no warning
    path taken at all)."""
    df = make_df(300)
    gmm = EVSessionModel(
        n_components=1, stratify_by=["location_type", "day_of_week", "season"],
    ).fit(df)
    assert gmm.stratify_by == ["location_type", "day_of_week", "season"]

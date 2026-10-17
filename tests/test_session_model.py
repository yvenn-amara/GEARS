"""Tests for the GMM session model (gears.models.session_model).

The VAE variant lives in ``test_vae.py``; the messages and fallbacks for unknown or sparse
contexts, and for too little data, are checked in ``test_sanity.py``.
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.mixture import GaussianMixture

from gears.data.schemas import validate_dataframe
from gears.models.session_model import DEFAULT_RECENCY_RESAMPLE_CAP, EVSessionModel


def make_df(n=500, seed=0, location_types=("work", "home", "public")):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "arrival_time": pd.date_range("2025-01-01", periods=n, freq="1h"),
        "duration": rng.uniform(0.5, 8, n),
        "energy": rng.uniform(2, 40, n),
        "location_type": rng.choice(list(location_types), n),
    })
    return validate_dataframe(df)


def make_long_history(n, span_days, seed=0):
    """One location, ``n`` sessions spread over ``span_days`` days."""
    rng = np.random.default_rng(seed)
    dates = pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, span_days, n), unit="D")
    return validate_dataframe(pd.DataFrame({
        "arrival_time": dates + pd.to_timedelta(rng.uniform(0, 23, n), unit="h"),
        "duration": rng.uniform(0.5, 8, n),
        "energy": rng.uniform(2, 40, n),
        "location_type": "home",
    }))


def make_regime_shift(n_old=800, n_recent=200, seed=0):
    """Old sessions are short and small, the last three weeks are long and large."""
    rng = np.random.default_rng(seed)

    def block(n, first_day, n_days, duration, energy):
        days = pd.Timestamp(first_day) + pd.to_timedelta(rng.integers(0, n_days, n), unit="D")
        return pd.DataFrame({
            "arrival_time": days + pd.to_timedelta(rng.uniform(0, 23, n), unit="h"),
            "duration": rng.uniform(*duration, n),
            "energy": rng.uniform(*energy, n),
            "location_type": "home",
        })

    old = block(n_old, "2024-01-01", 300, (0.5, 2), (2, 8))
    recent = block(n_recent, "2024-10-27", 21, (6, 10), (30, 45))
    return validate_dataframe(pd.concat([old, recent], ignore_index=True))


def _mean_log_energy(gmm: EVSessionModel, ctx: tuple) -> float:
    """Weighted mean of the energy dimension (log1p space) of one context's mixture."""
    sk = gmm.models_[ctx]
    return float(np.sum(sk.weights_ * sk.means_[:, 2]))


# ── Fit, sample, score, save ──────────────────────────────────────────────────

def test_a_fitted_model_samples_scores_and_describes_itself():
    df = make_df(500)
    gmm = EVSessionModel(n_components=3)
    assert "fitted=False" in repr(gmm)

    gmm.fit(df)

    assert gmm.is_fitted_ and "fitted=True" in repr(gmm)
    contexts = gmm.list_contexts()
    assert contexts and all(isinstance(c, tuple) for c in contexts)
    assert isinstance(gmm.get_sklearn_component(), GaussianMixture)
    assert np.isfinite(gmm.score(df))

    synth = gmm.sample(50, seed=0)
    assert len(synth) == 50
    assert {"arrival_hour", "duration", "energy"} <= set(synth.columns)
    assert synth["arrival_hour"].between(0, 24).all()
    assert (synth["duration"] >= 0).all() and (synth["energy"] >= 0).all()
    pd.testing.assert_frame_equal(synth, gmm.sample(50, seed=0))     # same seed, same sample

    dated = gmm.sample(20, date="2025-06-15", seed=1)                # the date picks the context
    assert (dated["arrival_time"].dt.date == pd.Timestamp("2025-06-15").date()).all()


def test_each_mixture_is_the_best_of_three_initialisations():
    df = make_df(500, location_types=("home",))
    X = np.column_stack([
        df["hour"].values, np.log1p(df["duration"].values), np.log1p(df["energy"].values),
    ])

    def reference(n_init):
        return GaussianMixture(
            n_components=4, covariance_type="full", random_state=0,
            n_init=n_init, max_iter=300, reg_covar=1e-5,
        ).fit(X)

    fitted = EVSessionModel(
        n_components=4, stratify_by=["location_type"], random_state=0
    ).fit(df).models_[("home",)]

    assert reference(1).score(X) < reference(3).score(X)         # the data tell the two apart
    np.testing.assert_allclose(fitted.means_, reference(3).means_)
    np.testing.assert_allclose(fitted.covariances_, reference(3).covariances_)


def test_the_number_of_components_can_be_chosen_by_bic():
    df = make_df(500, location_types=("home",))
    gmm = EVSessionModel(
        n_components="auto", max_components=5, stratify_by=["location_type"], random_state=0
    ).fit(df)
    assert gmm.bic_summary()["n_components"].between(2, 5).all()

    # the model kept is the one with the lowest BIC, found here with scikit-learn alone
    X = np.column_stack([
        df["hour"].values, np.log1p(df["duration"].values), np.log1p(df["energy"].values),
    ])
    candidates = {
        k: GaussianMixture(
            n_components=k, covariance_type="full", random_state=0,
            n_init=3, max_iter=300, reg_covar=1e-5,
        ).fit(X)
        for k in range(2, 6)
    }
    bic = {k: candidate.bic(X) for k, candidate in candidates.items()}
    assert len(set(bic.values())) == 4                       # the choice is not a tie
    best = candidates[min(bic, key=bic.get)]
    kept = gmm.models_[("home",)]
    assert kept.n_components == best.n_components
    np.testing.assert_allclose(kept.means_, best.means_)


def test_training_data_per_context_can_be_capped():
    gmm = EVSessionModel(n_components=3, max_samples_per_context=50).fit(make_df(2000))
    assert max(gmm.context_counts_.values()) == 50
    # ... but the daily rate still describes the whole history, not the capped sample.
    assert all(rate > 0 for rate in gmm.n_sessions_per_day_.values())


def test_save_then_load_gives_the_same_model(tmp_path):
    gmm = EVSessionModel(n_components=3).fit(make_df(300), is_sample=True)
    gmm.save(tmp_path / "gmm.joblib")
    loaded = EVSessionModel.load(tmp_path / "gmm.joblib")

    assert loaded.is_fitted_ and loaded.is_sample_ is True
    assert set(loaded.models_) == set(gmm.models_)
    pd.testing.assert_frame_equal(loaded.sample(20, seed=4), gmm.sample(20, seed=4))


# ── Stratification ────────────────────────────────────────────────────────────

def test_stratify_by_drops_only_the_columns_that_are_missing():
    full = ["location_type", "department", "day_of_week", "season"]
    with_department = validate_dataframe(pd.DataFrame({
        "arrival_time": pd.date_range("2025-01-01", periods=300, freq="1h"),
        "duration": np.random.default_rng(0).uniform(0.5, 8, 300),
        "energy": np.random.default_rng(1).uniform(2, 40, 300),
        "department": np.random.default_rng(2).choice(["75", "92"], 300),
    }))
    no_department = make_df(300)

    def stratified_by(df, columns):
        return EVSessionModel(n_components=1, stratify_by=columns).fit(df).stratify_by

    # a present `department` must survive a missing `location_type`
    assert stratified_by(with_department, full) == ["department", "day_of_week", "season"]
    assert stratified_by(no_department, full) == ["location_type", "day_of_week", "season"]
    assert stratified_by(no_department.drop(columns="location_type"), full) == [
        "day_of_week", "season"
    ]
    # nothing missing: left untouched
    assert stratified_by(no_department, ["location_type", "day_of_week"]) == [
        "location_type", "day_of_week"
    ]


# ── Recency weighting ─────────────────────────────────────────────────────────

def test_without_recency_the_fit_is_exactly_the_plain_gmm_fit():
    """Re-derive the fit with scikit-learn alone and compare: this is the regression
    proof that adding recency weighting changed nothing for ``recency=None``."""
    df = make_df(500, seed=3)
    gmm = EVSessionModel(
        n_components=3, stratify_by=["day_of_week", "season"], random_state=7, recency=None
    ).fit(df)

    expected = {}
    for ctx, group in df.groupby(["day_of_week", "season"], observed=True):
        if len(group) < 10:
            continue
        X = np.column_stack([
            group["hour"].values, np.log1p(group["duration"].values),
            np.log1p(group["energy"].values),
        ])
        expected[ctx] = GaussianMixture(
            n_components=3, covariance_type="full", random_state=7,
            n_init=3, max_iter=300, reg_covar=1e-5,
        ).fit(X)

    assert set(gmm.models_) == set(expected)
    for ctx, reference in expected.items():
        assert np.array_equal(gmm.models_[ctx].means_, reference.means_)
        assert np.array_equal(gmm.models_[ctx].covariances_, reference.covariances_)
        assert np.array_equal(gmm.models_[ctx].weights_, reference.weights_)

    # and `recency=None` is the default
    default = EVSessionModel(
        n_components=3, stratify_by=["day_of_week", "season"], random_state=7
    ).fit(df)
    assert all(np.array_equal(default.models_[c].means_, gmm.models_[c].means_) for c in expected)


def test_recency_pulls_the_fit_towards_the_recent_regime():
    df = make_regime_shift()
    kwargs = {"n_components": 2, "stratify_by": ["location_type"], "random_state": 0}
    plain = EVSessionModel(**kwargs).fit(df)
    weighted = EVSessionModel(**kwargs, recency=True, half_life_days=7).fit(df)
    ctx = ("home",)

    recent, old = np.log1p(37.5), np.log1p(5.0)          # mid-points of the two regimes
    plain_energy, weighted_energy = _mean_log_energy(plain, ctx), _mean_log_energy(weighted, ctx)
    assert abs(weighted_energy - recent) < abs(plain_energy - recent)
    assert abs(weighted_energy - recent) < abs(weighted_energy - old)
    assert "recency=True" in repr(weighted)


def test_recency_resampling_is_capped_but_the_daily_rate_is_not():
    df = make_long_history(n=5200, span_days=200)
    home = ("home",)

    def fit(**kwargs):
        return EVSessionModel(
            n_components=2, stratify_by=["location_type"], random_state=0,
            recency=True, half_life_days=30, **kwargs,
        ).fit(df)

    capped = fit(recency_resample_cap=500)
    assert capped.context_counts_[home] == 500
    true_rate = len(df) / df["arrival_time"].dt.normalize().nunique()
    assert capped.n_sessions_per_day_[home] == pytest.approx(true_rate)   # not the capped sample

    assert DEFAULT_RECENCY_RESAMPLE_CAP == 5000
    assert fit().context_counts_[home] == DEFAULT_RECENCY_RESAMPLE_CAP
    assert fit(max_samples_per_context=1000).context_counts_[home] == 1000   # the smaller wins


def test_the_default_half_life_scales_with_the_history_and_can_be_overridden():
    home = ("home",)

    def half_life(df, **kwargs):
        gmm = EVSessionModel(
            n_components=2, stratify_by=["location_type"], random_state=0, recency=True, **kwargs
        ).fit(df)
        return gmm.half_life_days_used_[home]

    short, long = half_life(make_long_history(2000, 56)), half_life(make_long_history(2000, 364))
    assert short == pytest.approx(55 / 3.5, rel=0.05)       # span / recency_halflife_divisor
    assert long == pytest.approx(363 / 3.5, rel=0.05)
    assert half_life(make_long_history(500, 56), half_life_days=99) == 99.0


# ── Old pickles ───────────────────────────────────────────────────────────────

def test_a_model_pickled_by_an_older_version_is_backfilled_on_load():
    """The shipped ``gmm_french.joblib`` predates the ``recency*`` and ``model_type``
    attributes: unpickling it and calling ``repr`` or ``plot_marginals`` used to raise
    ``AttributeError``.  Strip those attributes to mimic that pickle."""
    gmm = EVSessionModel(n_components=2, stratify_by=["day_of_week"]).fit(make_df(200))
    for attr in ("recency", "half_life_days", "recency_reference_date", "recency_resample_cap",
                 "recency_halflife_divisor", "half_life_days_used_",
                 "recency_reference_date_used_", "model_type"):
        delattr(gmm, attr)

    loaded = pickle.loads(pickle.dumps(gmm))

    assert loaded.recency is None and loaded.half_life_days is None
    assert loaded.recency_resample_cap == DEFAULT_RECENCY_RESAMPLE_CAP
    assert loaded.half_life_days_used_ == {} and loaded.recency_reference_date_used_ is None
    assert loaded.model_type == "gmm"
    assert "EVSessionModel" in repr(loaded)
    assert "GMM" in loaded.plot_marginals(n_samples=20)._suptitle.get_text()

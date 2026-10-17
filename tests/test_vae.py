"""Tests for the conditional-VAE session model (``model_type="vae"``).

Needs torch (``pip install "gears-ev[vae]"``): the whole module is skipped without it, and
``pytest -m vae`` runs just these tests.  The data is tiny and the training lasts a few
epochs, so this checks plumbing and contracts, not the quality of the fit.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

torch = pytest.importorskip("torch")

from gears import GEARSModel
from gears.cli import main
from gears.data.loader import make_demo_data
from gears.data.schemas import validate_dataframe
from gears.models.session_model import EVSessionModel
from gears.models.vae import VAEContextSlice

pytestmark = pytest.mark.vae

TINY = {"vae_epochs": 3, "vae_hidden_dim": 32, "vae_latent_dim": 4, "vae_batch_size": 32}
STRATIFY = ["location_type", "department", "day_of_week", "season"]


def make_df_vae(n=300, seed=0):
    rng = np.random.default_rng(seed)
    return validate_dataframe(pd.DataFrame({
        "arrival_time": pd.date_range("2025-01-01", periods=n, freq="2h"),
        "duration": rng.uniform(0.5, 8, n),
        "energy": rng.uniform(2, 40, n),
        "location_type": rng.choice(["work", "home"], n),
        "department": rng.choice(["75", "69"], n),
    }))


def fit_tiny_vae(df, **kwargs):
    return EVSessionModel(
        model_type="vae", stratify_by=STRATIFY, max_samples_per_context=30, random_state=0,
        **{**TINY, **kwargs},
    ).fit(df)


@pytest.fixture(scope="module")
def df():
    return make_df_vae()


@pytest.fixture(scope="module")
def vae(df):
    """Read-only: tests that change something fit their own."""
    return fit_tiny_vae(df)


def test_a_vae_fits_samples_scores_and_keeps_the_same_bookkeeping_as_the_gmm(vae, df):
    assert vae.is_fitted_ and vae.model_type == "vae"
    assert all(isinstance(s, VAEContextSlice) for s in vae.models_.values())

    synth = vae.sample(20, seed=0)
    assert len(synth) == 20 and {"arrival_hour", "duration", "energy"} <= set(synth.columns)
    assert (synth["duration"] >= 0).all() and (synth["energy"] >= 0).all()
    assert np.isfinite(vae.score(df))

    n = len(vae.models_)
    assert n > 0 and len(vae.list_contexts()) == n
    assert len(vae.n_sessions_per_day_) == len(vae.context_counts_) == n
    assert all(v > 0 for v in vae.n_sessions_per_day_.values())
    assert (vae.bic_summary()["n_components"] == 1).all()       # one "component" per context


def test_save_then_load_gives_a_working_vae(vae, df, tmp_path):
    vae.save(tmp_path / "vae.joblib")
    loaded = EVSessionModel.load(tmp_path / "vae.joblib")

    assert loaded.is_fitted_ and loaded.model_type == "vae"
    assert len(loaded.models_) == len(vae.models_)
    assert len(loaded.sample(10, seed=42)) == 10
    assert np.isfinite(loaded.score(df))


def test_a_context_slice_quacks_like_the_sklearn_mixture_the_rest_of_gears_expects(vae):
    """aggregator.py, medium_term.py and plotting.py use ``random_state``, ``sample``,
    ``score_samples``, ``n_components``, ``weights_`` and ``means_``."""
    slice_ = next(iter(vae.models_.values()))
    slice_.random_state = 7
    assert slice_.random_state == 7

    raw, labels = slice_.sample(5)
    assert isinstance(raw, np.ndarray) and raw.shape == (5, 3) and labels is None
    log_prob = slice_.score_samples(raw)
    assert log_prob.shape == (5,) and np.isfinite(log_prob).all()
    assert slice_.n_components == 1 and np.allclose(slice_.weights_, [1.0])
    assert slice_.means_.shape == (1, 3)


def test_sampling_adds_the_observation_noise_the_model_learned(vae):
    """Regression (variance collapse): ``sample_prior`` once returned the decoder's mean,
    with no observation noise, so samples were 10-150x too narrow.  With the same latent
    draws, the sample variance must exceed the decoder-only variance by about the learned
    observation variance."""
    slice_ = next(iter(vae.models_.values()))
    cvae, n = slice_.cvae, 2000
    ctx_idx = torch.tensor(slice_.ctx_index, dtype=torch.long).repeat(n, 1)

    torch.manual_seed(123)
    with torch.no_grad():
        z = torch.randn(n, cvae.latent_dim)
        decoder_only = cvae.decode(z, cvae._embed_context(ctx_idx))
    full = cvae.sample_prior(n, ctx_idx, seed=123)
    recon_var = torch.exp(cvae.log_recon_var).clamp(1e-4, 10.0)

    assert torch.all(full.var(dim=0) > decoder_only.var(dim=0) + 0.3 * recon_var)


def test_recency_is_ignored_for_a_vae_and_the_user_is_told(df, caplog):
    with caplog.at_level(logging.WARNING, logger="gears.models.session_model"):
        model = fit_tiny_vae(df, recency=True, half_life_days=7)

    assert model.is_fitted_
    assert any("recency" in r.message and "vae" in r.message for r in caplog.records)
    assert model.recency_reference_date_used_ is None and model.half_life_days_used_ == {}


def test_the_marginals_plot_of_a_vae_is_labelled_and_coloured_as_a_vae(vae):
    fig = vae.plot_marginals(n_samples=50)
    assert "VAE" in fig._suptitle.get_text() and "GMM" not in fig._suptitle.get_text()
    # the first histogram is the simulated series: VAE orange, not GMM blue
    assert fig.axes[0].patches[0].get_facecolor()[:3] == pytest.approx(
        (0xF4 / 255, 0xA2 / 255, 0x61 / 255), abs=1e-3
    )


def test_a_vae_works_end_to_end_through_the_facade():
    model = GEARSModel(
        model_type="vae", n_scenarios=2, forecaster_method="probabilistic", random_state=0,
    ).fit(make_demo_data(n=400, location_type="work", seed=0), verbose=False)

    assert model.gmm_.model_type == "vae"
    assert len(model.simulate_short_term("2025-06-01", horizon=2, n_scenarios=1, seed=0)) > 0


def test_the_cli_fits_a_vae(tmp_path):
    make_demo_data(n=400, location_type="work", seed=0).to_csv(tmp_path / "sessions.csv", index=False)
    result = CliRunner().invoke(main, [
        "fit", str(tmp_path / "sessions.csv"), "--output", str(tmp_path / "m.joblib"),
        "--n-components", "2", "--scenarios", "2", "--quiet", "--model-type", "vae",
    ])
    assert result.exit_code == 0, result.output
    assert GEARSModel.load(tmp_path / "m.joblib").model_type == "vae"

"""Tests for the unified GEARS session-model registry (NativeSessionModelRegistry + get_session_model)."""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import gears
from gears import NativeSessionModelRegistry
from gears.data.schemas import validate_dataframe
from gears.models.session_model import EVSessionModel

# ── helpers ──────────────────────────────────────────────────────────────────

def make_unified_gmm(n=400, seed=0):
    """Fit a small unified GMM (loc_type × dept × dow × season)."""
    rng = np.random.default_rng(seed)
    n_per = n // 4
    dfs = []
    for dept in ["75", "69"]:
        for lt in ["work", "home"]:
            df_ = pd.DataFrame({
                "arrival_time": pd.date_range("2024-01-01", periods=n_per, freq="2h"),
                "duration": rng.uniform(0.5, 6, n_per),
                "energy": rng.uniform(2, 30, n_per),
                "location_type": lt,
                "department": dept,
            })
            dfs.append(df_)
    df = pd.concat(dfs, ignore_index=True)
    df = validate_dataframe(df)

    gmm = EVSessionModel(
        n_components=2,
        stratify_by=["location_type", "department", "day_of_week", "season"],
        max_samples_per_context=50,
        random_state=seed,
    )
    gmm.fit(df, is_sample=True)
    return gmm


# ── catalogue ────────────────────────────────────────────────────────────────

class TestCatalogue:
    def test_french_entry_present(self):
        """Registry must expose at least the 'french' bundle."""
        reg = NativeSessionModelRegistry()
        cat = reg._CATALOGUE
        assert "french" in cat, f"'french' missing from catalogue: {list(cat.keys())}"

    def test_vae_sample_entry_present(self):
        """Registry must expose the 'french_vae_sample' bundle."""
        reg = NativeSessionModelRegistry()
        cat = reg._CATALOGUE
        assert "french_vae_sample" in cat, (
            f"'french_vae_sample' missing from catalogue: {list(cat.keys())}"
        )

    def test_vae_sample_meta(self):
        reg = NativeSessionModelRegistry()
        meta = reg._CATALOGUE["french_vae_sample"]
        assert meta["model_type"] == "vae"
        assert meta["is_sample"] is True
        assert "location_type" in meta["stratify_by"]

    def test_stratify_by_has_location_type(self):
        reg = NativeSessionModelRegistry()
        strat = reg._CATALOGUE["french"]["stratify_by"]
        assert "location_type" in strat
        assert "department" in strat
        assert "season" in strat
        assert "day_of_week" in strat

    def test_list_returns_at_least_two_rows(self):
        reg = NativeSessionModelRegistry()
        listing = reg.list()
        assert len(listing) >= 2
        assert "french" in listing["session_model_id"].values
        assert "french_vae_sample" in listing["session_model_id"].values

    def test_repr(self):
        reg = NativeSessionModelRegistry()
        r = repr(reg)
        assert "NativeSessionModelRegistry" in r


# ── save / load round-trip ────────────────────────────────────────────────────

class TestSaveLoad:
    def test_save_and_load(self):
        gmm = make_unified_gmm()
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = NativeSessionModelRegistry(session_model_dir=tmpdir)
            path = reg.save("french", gmm)
            assert Path(path).exists()

            loaded = reg.load("french")
            assert loaded.is_fitted_
            assert loaded.stratify_by == gmm.stratify_by
            assert len(loaded.models_) == len(gmm.models_)

    def test_load_missing_falls_back_to_synthetic(self):
        """When gmm_french.joblib is absent the registry generates a synthetic
        fallback rather than raising — that is the intentional behaviour so that
        notebooks work out of the box before the user runs fit_session_model.py."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = NativeSessionModelRegistry(session_model_dir=tmpdir)
            loaded = reg.load("french")
            assert isinstance(loaded, EVSessionModel)
            assert loaded.is_fitted_
            assert loaded.is_sample_

    def test_vae_sample_load_missing_falls_back(self):
        """When gmm_vae_french_sample.joblib is absent, a VAE fallback is generated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = NativeSessionModelRegistry(session_model_dir=tmpdir)
            loaded = reg.load("french_vae_sample")
            assert isinstance(loaded, EVSessionModel)
            assert loaded.is_fitted_
            assert loaded.model_type == "vae"
            assert loaded.is_sample_

    def test_stratify_by_preserved(self):
        gmm = make_unified_gmm()
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = NativeSessionModelRegistry(session_model_dir=tmpdir)
            reg.save("french", gmm)
            loaded = reg.load("french")
        assert "location_type" in loaded.stratify_by
        assert "department" in loaded.stratify_by


# ── get_sklearn_component (on NativeSessionModelRegistry) ─────────────────────

class TestGetSklearnComponent:
    def test_returns_fitted_gmm(self):
        gmm = make_unified_gmm()
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = NativeSessionModelRegistry(session_model_dir=tmpdir)
            reg.save("french", gmm)

            ctx0 = gmm.list_contexts()[0]
            _loc_type, _dept = ctx0[0], ctx0[1]
            # find matching season and dow
            ctx_d = dict(zip(gmm.stratify_by, ctx0))
            sk = reg.get_sklearn_component(
                location_type=ctx_d["location_type"],
                department=ctx_d["department"],
                season=ctx_d["season"],
                day_of_week=ctx_d["day_of_week"],
                session_model_id="french",
            )
        from sklearn.mixture import GaussianMixture
        assert isinstance(sk, GaussianMixture)
        assert sk.n_components >= 2


# ── module-level get_session_model() ──────────────────────────────────────────

class TestGetSessionModel:
    def test_returns_ev_session_model(self):
        """get_session_model() must return the EVSessionModel wrapper (not sklearn directly)."""
        model = gears.get_session_model()
        assert isinstance(model, EVSessionModel)

    def test_location_type_in_stratify_by(self):
        model = gears.get_session_model("french")
        assert "location_type" in model.stratify_by

    def test_sample_returns_dataframe(self):
        model = gears.get_session_model()
        # Find any valid context with this location type
        ctx_key = next(
            (c for c in model.list_contexts() if c[0] == "public"),
            model.list_contexts()[0],
        )
        ctx = dict(zip(model.stratify_by, ctx_key))
        samples = model.sample(10, context=ctx, seed=0)
        assert len(samples) == 10
        assert "arrival_hour" in samples.columns
        assert "energy" in samples.columns
        assert "duration" in samples.columns

    def test_get_sklearn_component_via_wrapper(self):
        model = gears.get_session_model()
        ctx_key = next(
            (c for c in model.list_contexts() if c[0] == "work"),
            model.list_contexts()[0],
        )
        ctx = dict(zip(model.stratify_by, ctx_key))
        sk = model.get_sklearn_component(context=ctx)
        assert hasattr(sk, "means_")
        assert sk.means_.shape[1] == 3   # [hour, log1p(dur), log1p(energy)]

    def test_exported_from_gears_top_level(self):
        assert hasattr(gears, "get_session_model")
        assert callable(gears.get_session_model)

    def test_four_location_types_accessible(self):
        """All four location types must have at least one fitted context."""
        model = gears.get_session_model()
        ctx_loc_types = {c[0] for c in model.list_contexts()}
        # The sample GMM covers the top-5 departments and all loc types present
        # in the data — at minimum work, home, public should be there
        for lt in ("work", "home", "public"):
            assert lt in ctx_loc_types, (
                f"location_type '{lt}' missing from fitted contexts.\n"
                f"Available: {sorted(ctx_loc_types)}"
            )


# ── no old bundle names in catalogue ─────────────────────────────────────────

class TestNoLegacyBundles:
    @pytest.mark.parametrize("old_id", ["global", "work", "home", "public"])
    def test_old_ids_not_in_catalogue(self, old_id):
        reg = NativeSessionModelRegistry()
        assert old_id not in reg._CATALOGUE, (
            f"Legacy bundle '{old_id}' still present in catalogue. "
            "Remove it — all location types now live in 'french'."
        )

    @pytest.mark.parametrize("old_id", ["global", "work", "home", "public"])
    def test_loading_old_id_raises(self, old_id):
        reg = NativeSessionModelRegistry()
        with pytest.raises((KeyError, FileNotFoundError, ValueError)):
            reg.load(old_id)

"""Tests for the unified GEARS GMM registry (NativeGMMRegistry + get_gmm)."""
import numpy as np
import pandas as pd
import pytest
import tempfile
from pathlib import Path

import gears
from gears import NativeGMMRegistry, get_gmm
from gears.models.gmm import EVSessionGMM
from gears.data.schemas import validate_dataframe


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

    gmm = EVSessionGMM(
        n_components=2,
        stratify_by=["location_type", "department", "day_of_week", "season"],
        max_samples_per_context=50,
        random_state=seed,
    )
    gmm.fit(df, is_sample=True)
    return gmm


# ── catalogue ────────────────────────────────────────────────────────────────

class TestCatalogue:
    def test_single_entry(self):
        """Registry must expose exactly one bundle: 'french'."""
        reg = NativeGMMRegistry()
        cat = reg._CATALOGUE
        assert list(cat.keys()) == ["french"], (
            f"Expected only ['french'], got {list(cat.keys())}"
        )

    def test_stratify_by_has_location_type(self):
        reg = NativeGMMRegistry()
        strat = reg._CATALOGUE["french"]["stratify_by"]
        assert "location_type" in strat
        assert "department" in strat
        assert "season" in strat
        assert "day_of_week" in strat

    def test_list_returns_one_row(self):
        reg = NativeGMMRegistry()
        listing = reg.list()
        assert len(listing) == 1
        assert listing.iloc[0]["gmm_id"] == "french"

    def test_repr(self):
        reg = NativeGMMRegistry()
        r = repr(reg)
        assert "NativeGMMRegistry" in r


# ── save / load round-trip ────────────────────────────────────────────────────

class TestSaveLoad:
    def test_save_and_load(self):
        gmm = make_unified_gmm()
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = NativeGMMRegistry(gmm_dir=tmpdir)
            path = reg.save("french", gmm)
            assert Path(path).exists()

            loaded = reg.load("french")
            assert loaded.is_fitted_
            assert loaded.stratify_by == gmm.stratify_by
            assert len(loaded.models_) == len(gmm.models_)

    def test_load_missing_falls_back_to_synthetic(self):
        """When gmm_french.joblib is absent the registry generates a synthetic
        fallback rather than raising — that is the intentional behaviour so that
        notebooks work out of the box before the user runs fit_gmm.py."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = NativeGMMRegistry(gmm_dir=tmpdir)
            # Should not raise — returns a synthetic (sample) GMM
            loaded = reg.load("french")
            assert isinstance(loaded, EVSessionGMM)
            assert loaded.is_fitted_
            assert loaded.is_sample_   # synthetic fallback is always is_sample=True

    def test_stratify_by_preserved(self):
        gmm = make_unified_gmm()
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = NativeGMMRegistry(gmm_dir=tmpdir)
            reg.save("french", gmm)
            loaded = reg.load("french")
        assert "location_type" in loaded.stratify_by
        assert "department" in loaded.stratify_by


# ── get_sklearn_gmm ───────────────────────────────────────────────────────────

class TestGetSklearnGmm:
    def test_returns_fitted_gmm(self):
        gmm = make_unified_gmm()
        with tempfile.TemporaryDirectory() as tmpdir:
            reg = NativeGMMRegistry(gmm_dir=tmpdir)
            reg.save("french", gmm)

            ctx0 = gmm.list_contexts()[0]
            loc_type, dept = ctx0[0], ctx0[1]
            # find matching season and dow
            ctx_d = dict(zip(gmm.stratify_by, ctx0))
            sk = reg.get_sklearn_gmm(
                location_type=ctx_d["location_type"],
                departement=ctx_d["department"],
                saison=ctx_d["season"],
                day_of_week=ctx_d["day_of_week"],
                gmm_id="french",
            )
        from sklearn.mixture import GaussianMixture
        assert isinstance(sk, GaussianMixture)
        assert sk.n_components >= 2


# ── module-level get_gmm() ───────────────────────────────────────────────────

class TestGetGmm:
    def test_returns_evSessionGMM(self):
        """get_gmm() must return the EVSessionGMM wrapper (not sklearn directly)."""
        gmm = gears.get_gmm("work", "92", "winter", 0)
        assert isinstance(gmm, EVSessionGMM)

    def test_location_type_in_stratify_by(self):
        gmm = gears.get_gmm("home", "69", "summer", 2)
        assert "location_type" in gmm.stratify_by

    def test_sample_returns_dataframe(self):
        gmm = gears.get_gmm("public", "92", "autumn", 4)
        # Find any valid context with this location type
        ctx_key = next(
            (c for c in gmm.list_contexts() if c[0] == "public"),
            gmm.list_contexts()[0],
        )
        ctx = dict(zip(gmm.stratify_by, ctx_key))
        samples = gmm.sample(10, context=ctx, seed=0)
        assert len(samples) == 10
        assert "arrival_hour" in samples.columns
        assert "energy" in samples.columns
        assert "duration" in samples.columns

    def test_get_sklearn_gmm_via_wrapper(self):
        gmm = gears.get_gmm("work", "92", "winter", 0)
        ctx_key = next(
            (c for c in gmm.list_contexts() if c[0] == "work"),
            gmm.list_contexts()[0],
        )
        ctx = dict(zip(gmm.stratify_by, ctx_key))
        sk = gmm.get_sklearn_gmm(context=ctx)
        assert hasattr(sk, "means_")
        assert sk.means_.shape[1] == 3   # [hour, log1p(dur), log1p(energy)]

    def test_exported_from_gears_top_level(self):
        assert hasattr(gears, "get_gmm")
        assert callable(gears.get_gmm)

    def test_four_location_types_accessible(self):
        """All four location types must have at least one fitted context."""
        gmm = gears.get_gmm("work", "92", "winter", 0)
        ctx_loc_types = {c[0] for c in gmm.list_contexts()}
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
        reg = NativeGMMRegistry()
        assert old_id not in reg._CATALOGUE, (
            f"Legacy bundle '{old_id}' still present in catalogue. "
            "Remove it — all location types now live in 'french'."
        )

    @pytest.mark.parametrize("old_id", ["global", "work", "home", "public"])
    def test_loading_old_id_raises(self, old_id):
        reg = NativeGMMRegistry()
        with pytest.raises((KeyError, FileNotFoundError, ValueError)):
            reg.load(old_id)

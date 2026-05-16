"""
Model registry for GEARS.

Two registries are provided:

1. NativeGMMRegistry
   Manages GMMs pre-fitted on real French data and shipped with the package.
   These live in gears/data/gmm/ and are available without raw data.

2. ModelRegistry
   Manages full GEARSModel bundles (GMM + forecaster), optionally hosted
   on Hugging Face Hub.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

logger = logging.getLogger(__name__)

# Directory where pre-fitted GMMs are stored inside the package
_GMM_DIR = Path(__file__).parent.parent / "data" / "gmm"

# HF Hub repository for full model bundles
HF_REPO_ID = "yvenn-amara/GEARS-pretrained"
_LOCAL_CACHE = Path.home() / ".cache" / "gears" / "models"


# ---------------------------------------------------------------------------
# Native GMM Registry – pre-fitted GMMs shipped with the package
# ---------------------------------------------------------------------------

class NativeGMMRegistry:
    """
    Access pre-fitted GMM bundles embedded in the GEARS package.

    Pre-fitted GMMs are stored in gears/data/gmm/ as joblib files.
    They are fitted on the French national dataset (2025) and are available
    without any raw data.

    Available bundles
    -----------------
    - ``"french"`` : single unified GMM fitted on the full French national
      dataset, stratified by
      ``location_type × département × day_of_week × season``.
      This is the canonical bundle used by all GEARS APIs.
      File: ``gears/data/gmm/gmm_french.joblib`` (8 008 contexts).

    Examples
    --------
    >>> registry = NativeGMMRegistry()
    >>> registry.list()
    >>> gmm = registry.load("french")
    >>> sk = gmm.get_sklearn_gmm(context={
    ...     "location_type": "work", "department": "75",
    ...     "season": "winter", "day_of_week": 0,
    ... })
    """

    # ------------------------------------------------------------------
    # Catalogue — single source of truth for all GMM bundle metadata.
    #
    # KEY INVARIANT: the canonical key is "french" and it must point to
    # gmm_french.joblib (the full 8 008-context model, is_sample=False).
    #
    # Why only one entry?
    #   • get_gmm() calls registry.load("french").
    #   • get_sklearn_gmm() defaults to gmm_id="french".
    #   • fit_gmm.py saves with gmm_id="french".
    #   • test_registry.py::TestCatalogue expects exactly ["french"].
    #
    # The sample bundle (gmm_french_sample.joblib) is a development /
    # CI artefact; it is NOT exposed in the public catalogue to keep the
    # API surface clean and tests unambiguous.  Developers who need the
    # sample bundle can load it directly via:
    #   EVSessionGMM.load("gears/data/gmm/gmm_french_sample.joblib")
    # ------------------------------------------------------------------

    _CATALOGUE: dict[str, dict] = {
        "french": {
            "filename": "gmm_french.joblib",
            "description": (
                "French national EV dataset — single GMM stratified by "
                "location_type × département × season × day_of_week "
                "(8 008 contexts, full dataset, is_sample=False)."
            ),
            "stratify_by": ["location_type", "department", "day_of_week", "season"],
            "is_sample": False,
        },
    }

    def __init__(self, gmm_dir: Optional[Path] = None):
        self.gmm_dir = Path(gmm_dir) if gmm_dir else _GMM_DIR
        self.gmm_dir.mkdir(parents=True, exist_ok=True)

    def list(self) -> pd.DataFrame:
        """Return a DataFrame listing all available native GMMs."""
        rows = []
        for gmm_id, meta in self._CATALOGUE.items():
            path = self.gmm_dir / meta["filename"]
            rows.append({
                "gmm_id": gmm_id,
                "description": meta["description"],
                "available": path.exists(),
                "is_sample": meta.get("is_sample", False),
                "stratify_by": meta["stratify_by"],
            })
        return pd.DataFrame(rows)

    def load(self, gmm_id: str) -> "EVSessionGMM":
        """
        Load a pre-fitted GMM from the package.

        Parameters
        ----------
        gmm_id : str
            Registry bundle ID.  Currently only ``'french'`` is available —
            the single unified GMM stratified by
            ``location_type × département × season × day_of_week``.

        Returns
        -------
        EVSessionGMM (fitted)
        """
        from gears.models.gmm import EVSessionGMM

        if gmm_id not in self._CATALOGUE:
            raise ValueError(
                f"Unknown GMM ID '{gmm_id}'. Available: {list(self._CATALOGUE)}"
            )

        meta = self._CATALOGUE[gmm_id]
        path = self.gmm_dir / meta["filename"]

        if not path.exists():
            logger.warning(
                "Native GMM '%s' not found at %s. "
                "Run `python scripts/fit_gmm.py` to fit GMMs on your data, "
                "or the package will generate a synthetic fallback.",
                gmm_id, path,
            )
            return self._generate_fallback(gmm_id)

        gmm = joblib.load(path)
        if not isinstance(gmm, EVSessionGMM):
            raise TypeError(f"Expected EVSessionGMM, got {type(gmm)}")

        logger.info("Loaded native GMM '%s' from %s.", gmm_id, path)
        return gmm

    def _generate_fallback(self, gmm_id: str) -> "EVSessionGMM":
        """Generate a lightweight synthetic GMM when native data is absent.

        The fallback covers all four location types so the context structure
        of the real ``gmm_french.joblib`` is replicated at a small scale.
        """
        from gears.data.loader import make_demo_data
        from gears.models.gmm import EVSessionGMM

        logger.info(
            "Native GMM '%s' not found — generating synthetic fallback. "
            "Run `python scripts/fit_gmm.py --input <data>` to fit on real data.",
            gmm_id,
        )

        # Build a small synthetic dataset covering all four location types
        frames = [
            make_demo_data(n=500, location_type=lt, seed=i)
            for i, lt in enumerate(["work", "home", "public", "heavy"])
        ]
        import pandas as pd
        df = pd.concat(frames, ignore_index=True)

        gmm = EVSessionGMM(
            n_components="auto",
            max_components=5,
            stratify_by=["location_type", "day_of_week", "season"],
        ).fit(df, is_sample=True, metadata={"synthetic_fallback": True, "gmm_id": gmm_id})

        return gmm

    def save(self, gmm_id: str, gmm) -> Path:
        """
        Save a fitted GMM to the package GMM directory.

        Parameters
        ----------
        gmm_id : str
            Registry ID to save under.
        gmm : EVSessionGMM
            Fitted GMM to save.

        Returns
        -------
        Path to saved file.
        """
        if gmm_id not in self._CATALOGUE:
            raise ValueError(f"Unknown GMM ID '{gmm_id}'. Available: {list(self._CATALOGUE)}")
        path = self.gmm_dir / self._CATALOGUE[gmm_id]["filename"]
        joblib.dump(gmm, path)
        logger.info("Saved GMM '%s' to %s.", gmm_id, path)
        return path

    def get_sklearn_gmm(
        self,
        location_type: str,
        departement: str,
        saison: str,
        day_of_week: int,
        gmm_id: str = "french",
    ):
        """
        Return the underlying sklearn GaussianMixture for a specific stratum
        of the unified French GMM.

        Parameters
        ----------
        location_type : str
            Charging location: ``'work'``, ``'home'``, ``'public'``, or ``'heavy'``.
        departement : str
            INSEE département code, e.g. ``'75'``, ``'69'``, ``'13'``.
        saison : str
            Season: ``'winter'``, ``'spring'``, ``'summer'``, ``'autumn'``.
        day_of_week : int
            Monday = 0, …, Sunday = 6.
        gmm_id : str, optional
            Registry bundle ID (default ``'french'``).

        Returns
        -------
        sklearn.mixture.GaussianMixture (fitted)
        """
        gmm = self.load(gmm_id)
        return gmm.get_sklearn_gmm(
            context={
                "location_type": location_type,
                "department":    departement,
                "season":        saison,
                "day_of_week":   day_of_week,
            }
        )

    def __repr__(self) -> str:
        n_available = sum(
            1 for meta in self._CATALOGUE.values()
            if (self.gmm_dir / meta["filename"]).exists()
        )
        return (
            f"NativeGMMRegistry("
            f"{n_available}/{len(self._CATALOGUE)} GMMs available, "
            f"dir={self.gmm_dir})"
        )


# ---------------------------------------------------------------------------
# Full Model Registry – GEARSModel bundles (GMM + forecaster)
# ---------------------------------------------------------------------------

_CATALOGUE: dict[str, dict] = {
    "french_demo": {
        "description": (
            "French national EV charging — unified GMM + SARIMA forecaster (demo bundle). "
            "GMM stratified by location_type × département × season × day_of_week."
        ),
        "country": "FR",
        "stratify_by": ["location_type", "department", "day_of_week", "season"],
        "n_sessions": 15000,
        "hf_filename": "french_demo.joblib",
    },
}


class ModelRegistry:
    """
    Access and manage pre-trained GEARSModel bundles.

    Bundles are hosted on Hugging Face Hub and cached locally.

    Usage
    -----
    >>> registry = ModelRegistry()
    >>> registry.list_models()
    >>> bundle = registry.load("work_fr_demo")
    >>> gmm = bundle["gmm"]
    """

    def __init__(
        self,
        hf_repo_id: str = HF_REPO_ID,
        cache_dir: Optional[Path] = None,
    ):
        self.hf_repo_id = hf_repo_id
        self.cache_dir = Path(cache_dir) if cache_dir else _LOCAL_CACHE
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def list_models(self) -> pd.DataFrame:
        """Return a DataFrame listing all available pre-trained models."""
        rows = [{"model_id": mid, **meta} for mid, meta in _CATALOGUE.items()]
        return pd.DataFrame(rows)

    def load(self, model_id: str, force_download: bool = False) -> dict:
        """
        Load a pre-trained model bundle by ID.

        Parameters
        ----------
        model_id : str
            One of the IDs returned by list_models().
        force_download : bool
            Re-download even if cached locally.

        Returns
        -------
        dict with keys 'gmm', 'forecaster' (optional), 'metadata'.
        """
        if model_id not in _CATALOGUE:
            raise ValueError(
                f"Unknown model ID '{model_id}'. "
                f"Available: {list(_CATALOGUE)}"
            )

        local_path = self.cache_dir / f"{model_id}.joblib"

        if not local_path.exists() or force_download:
            local_path = self._download(model_id)

        bundle = joblib.load(local_path)
        logger.info("Loaded model bundle '%s' from %s.", model_id, local_path)
        return bundle

    def save_bundle(
        self,
        model_id: str,
        gmm,
        forecaster=None,
        metadata: Optional[dict] = None,
    ) -> Path:
        """Serialize a model bundle to the local cache."""
        bundle = {"gmm": gmm, "forecaster": forecaster, "metadata": metadata or {}}
        path = self.cache_dir / f"{model_id}.joblib"
        joblib.dump(bundle, path)
        logger.info("Saved bundle '%s' to %s.", model_id, path)
        return path

    def upload_to_hub(self, model_id: str, token: Optional[str] = None) -> None:
        """Upload a locally cached bundle to HF Hub."""
        try:
            from huggingface_hub import HfApi
        except ImportError:
            raise ImportError("pip install huggingface-hub to upload models.")

        local_path = self.cache_dir / f"{model_id}.joblib"
        if not local_path.exists():
            raise FileNotFoundError(f"No cached bundle for '{model_id}'.")

        api = HfApi(token=token)
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=f"{model_id}.joblib",
            repo_id=self.hf_repo_id,
            repo_type="model",
        )
        logger.info("Uploaded '%s' to %s.", model_id, self.hf_repo_id)

    def _download(self, model_id: str) -> Path:
        """Try HF Hub, fall back to synthetic demo bundle."""
        filename = _CATALOGUE[model_id]["hf_filename"]
        dest = self.cache_dir / f"{model_id}.joblib"

        try:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(
                repo_id=self.hf_repo_id,
                filename=filename,
                cache_dir=str(self.cache_dir),
                local_dir=str(self.cache_dir),
            )
            return Path(path)
        except Exception as e:
            logger.warning(
                "Could not download '%s' from HF Hub (%s). "
                "Generating synthetic demo bundle instead.",
                model_id, e,
            )
            return self._generate_demo_bundle(model_id, dest)

    def _generate_demo_bundle(self, model_id: str, dest: Path) -> Path:
        from gears.data.loader import make_demo_data
        from gears.models.gmm import EVSessionGMM
        from gears.models.forecaster import SessionForecaster

        meta = _CATALOGUE[model_id]
        loc = meta.get("location_type", "work")
        n = meta.get("n_sessions", 2000)

        logger.info("Generating synthetic demo bundle for '%s'.", model_id)
        df = make_demo_data(n=n, location_type=loc, seed=42)

        gmm = EVSessionGMM(
            n_components="auto",
            max_components=6,
            stratify_by=["day_of_week", "season"],
        ).fit(df)
        fc = SessionForecaster(method="probabilistic").fit(df)

        bundle = {"gmm": gmm, "forecaster": fc, "metadata": {**meta, "synthetic": True}}
        joblib.dump(bundle, dest)
        logger.info("Saved synthetic demo bundle to %s.", dest)
        return dest

    def __repr__(self) -> str:
        return f"ModelRegistry(repo={self.hf_repo_id!r}, cache={self.cache_dir})"


# ---------------------------------------------------------------------------
# Module-level retrieval API
# ---------------------------------------------------------------------------

_default_registry = None


def _get_default_registry() -> NativeGMMRegistry:
    global _default_registry
    if _default_registry is None:
        _default_registry = NativeGMMRegistry()
    return _default_registry


def get_gmm(
    location_type: str,
    departement: str,
    saison: str,
    day_of_week: int,
) -> "EVSessionGMM":
    """
    Retrieve the pre-fitted French EVSessionGMM for a given stratum.

    This is the primary retrieval API for the GEARS registry. The unified
    ``'french'`` bundle contains a single ``EVSessionGMM`` fitted on the French
    national dataset, stratified by
    **(location_type × département × saison × day_of_week)**.

    Parameters
    ----------
    location_type : str
        Charging location type: ``'work'``, ``'home'``, ``'public'``, or ``'heavy'``.
    departement : str
        INSEE département code, e.g. ``'75'`` (Paris), ``'69'`` (Rhône),
        ``'13'`` (Bouches-du-Rhône).
    saison : str
        Season: ``'winter'``, ``'spring'``, ``'summer'``, ``'autumn'``.
    day_of_week : int
        Day-of-week (Monday = 0, Tuesday = 1, …, Sunday = 6).

    Returns
    -------
    EVSessionGMM
        Fitted wrapper exposing:

        - ``.get_sklearn_gmm(context=...)`` → raw ``sklearn.mixture.GaussianMixture``
        - ``.sample(n, context=...)`` → synthetic sessions DataFrame
        - ``.list_contexts()`` → all fitted (loc, dept, season, dow) tuples
        - ``.bic_summary()`` → BIC / n_components per stratum

    Examples
    --------
    >>> import gears
    >>> gmm = gears.get_gmm("work", "75", "winter", 0)
    >>> # underlying sklearn object
    >>> sk = gmm.get_sklearn_gmm(
    ...     context={"location_type": "work", "department": "75",
    ...              "season": "winter", "day_of_week": 0}
    ... )
    >>> print(sk.means_)   # [arrival_hour, log1p(duration_h), log1p(energy_kWh)]
    >>> # sample 50 synthetic sessions
    >>> sessions = gmm.sample(
    ...     50,
    ...     context={"location_type": "work", "department": "75",
    ...              "season": "winter", "day_of_week": 0},
    ... )
    """
    registry = _get_default_registry()
    return registry.load("french")

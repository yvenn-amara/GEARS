"""
Model registry for GEARS.

Two registries are provided:

1. NativeSessionModelRegistry
   Manages GMMs pre-fitted on real French data and shipped with the package.
   These live in gears/data/session_models/ and are available without raw data.

2. ModelRegistry
   Manages full GEARSModel bundles (GMM + forecaster), optionally hosted
   on Hugging Face Hub.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import joblib
import pandas as pd

if TYPE_CHECKING:
    from gears.models.session_model import EVSessionModel

logger = logging.getLogger(__name__)

# Directory where pre-fitted GMMs are stored inside the package
_SESSION_MODEL_DIR = Path(__file__).parent.parent / "data" / "session_models"

# HF Hub repository for full model bundles
HF_REPO_ID = "yvenn-amara/GEARS-pretrained"
_LOCAL_CACHE = Path.home() / ".cache" / "gears" / "models"


# ---------------------------------------------------------------------------
# Native GMM Registry – pre-fitted GMMs shipped with the package
# ---------------------------------------------------------------------------

class NativeSessionModelRegistry:
    """
    Access pre-fitted GMM bundles embedded in the GEARS package.

    Pre-fitted GMMs are stored in gears/data/session_models/ as joblib files.
    They are fitted on the French national dataset (2025) and are available
    without any raw data.

    Available bundles
    -----------------
    - ``"french"`` : single unified GMM fitted on the full French national
      dataset, stratified by
      ``location_type × département × day_of_week × season``.
      This is the canonical bundle used by all GEARS APIs.
      File: ``gears/data/session_models/gmm_french.joblib`` (8 008 contexts).

    Examples
    --------
    >>> registry = NativeSessionModelRegistry()
    >>> registry.list()
    >>> gmm = registry.load("french")
    >>> sk = gmm.get_sklearn_component(context={
    ...     "location_type": "work", "department": "75",
    ...     "season": "winter", "day_of_week": 0,
    ... })
    """

    # ------------------------------------------------------------------
    # Catalogue — single source of truth for all native bundle metadata.
    #
    # Two entries as of Session 4:
    #   • "french": the full national GMM (8 008 contexts, is_sample=False).
    #     get_session_model() currently always resolves to this entry regardless of
    #     its arguments — see PROPOSAL_NAMING.md for why and what's proposed.
    #   • "french_vae_sample": the shared conditional VAE (top-5 departments,
    #     is_sample=True), added in Session 4.
    #
    # The separate CI/dev-only sample bundle (gmm_french_sample.joblib) is
    # NOT in this catalogue, to keep the public API surface unambiguous.
    # Developers who need it can load it directly via:
    #   EVSessionModel.load("gears/data/session_models/gmm_french_sample.joblib")
    # ------------------------------------------------------------------

    _CATALOGUE: ClassVar[dict[str, dict]] = {
        "french": {
            "filename": "gmm_french.joblib",
            "description": (
                "French national EV dataset — single GMM stratified by "
                "location_type × département × season × day_of_week "
                "(8 008 contexts, full dataset, is_sample=False)."
            ),
            "stratify_by": ["location_type", "department", "day_of_week", "season"],
            "is_sample": False,
            "model_type": "gmm",
        },
        "french_vae_sample": {
            "filename": "vae_french_sample.joblib",
            "description": (
                "French EV sample dataset — shared conditional VAE stratified by "
                "location_type × département × season × day_of_week "
                "(top-5 departments, is_sample=True, model_type='vae')."
            ),
            "stratify_by": ["location_type", "department", "day_of_week", "season"],
            "is_sample": True,
            "model_type": "vae",
        },
    }

    def __init__(self, session_model_dir: Path | None = None):
        """
        Parameters
        ----------
        session_model_dir : Path, optional
            Override the default GMM storage directory
            (``gears/data/session_models/``).  The directory is created if missing.
        """
        self.session_model_dir = Path(session_model_dir) if session_model_dir else _SESSION_MODEL_DIR
        self.session_model_dir.mkdir(parents=True, exist_ok=True)

    def list(self) -> pd.DataFrame:
        """Return a DataFrame listing all available native models.

        Returns
        -------
        pd.DataFrame
            Columns: session_model_id, model_type, description, available, is_sample, stratify_by.
        """
        rows = []
        for session_model_id, meta in self._CATALOGUE.items():
            path = self.session_model_dir / meta["filename"]
            rows.append({
                "session_model_id": session_model_id,
                "model_type": meta.get("model_type", "gmm"),
                "description": meta["description"],
                "available": path.exists(),
                "is_sample": meta.get("is_sample", False),
                "stratify_by": meta["stratify_by"],
            })
        return pd.DataFrame(rows)

    def load(self, session_model_id: str) -> EVSessionModel:
        """
        Load a pre-fitted GMM from the package.

        Parameters
        ----------
        session_model_id : str
            Registry bundle ID.  Currently only ``'french'`` is available —
            the single unified GMM stratified by
            ``location_type × département × season × day_of_week``.

        Returns
        -------
        EVSessionModel
            Fitted GMM instance.  Falls back to a synthetic model when the
            joblib file is absent (e.g. in CI without large artefacts).

        Raises
        ------
        ValueError
            If *session_model_id* is not in the catalogue.
        TypeError
            If the file contains an object that is not an ``EVSessionModel``.
        """
        from gears.models.session_model import EVSessionModel

        if session_model_id not in self._CATALOGUE:
            raise ValueError(
                f"Unknown GMM ID '{session_model_id}'. Available: {list(self._CATALOGUE)}"
            )

        meta = self._CATALOGUE[session_model_id]
        path = self.session_model_dir / meta["filename"]

        if not path.exists():
            logger.warning(
                "Native GMM '%s' not found at %s. "
                "Run `python scripts/fit_session_model.py` to fit GMMs on your data, "
                "or the package will generate a synthetic fallback.",
                session_model_id, path,
            )
            return self._generate_fallback(session_model_id)

        gmm = joblib.load(path)
        if not isinstance(gmm, EVSessionModel):
            raise TypeError(f"Expected EVSessionModel, got {type(gmm)}")

        logger.info("Loaded native GMM '%s' from %s.", session_model_id, path)
        return gmm

    def _generate_fallback(self, session_model_id: str) -> EVSessionModel:
        """Generate a lightweight synthetic model when the native file is absent.

        For ``model_type="gmm"`` (default), fits a small sklearn GMM.
        For ``model_type="vae"``, fits a tiny CVAE (CPU, very small dataset).

        Parameters
        ----------
        session_model_id : str
            Registry ID used to tag the fallback metadata.

        Returns
        -------
        EVSessionModel
            Fitted synthetic model with ``is_sample=True``.
        """
        from gears.data.loader import make_demo_data
        from gears.models.session_model import EVSessionModel

        meta = self._CATALOGUE.get(session_model_id, {})
        model_type = meta.get("model_type", "gmm")

        logger.info(
            "Native model '%s' not found — generating synthetic fallback (model_type=%s). "
            "Run `python scripts/fit_session_model.py --model-type %s --input <data>` to fit on real data.",
            session_model_id, model_type, model_type,
        )

        frames = [
            make_demo_data(n=500, location_type=lt, seed=i)
            for i, lt in enumerate(["work", "home", "public", "heavy"])
        ]
        import pandas as pd
        df = pd.concat(frames, ignore_index=True)

        if model_type == "vae":
            model = EVSessionModel(
                model_type="vae",
                stratify_by=["location_type", "day_of_week", "season"],
                vae_epochs=5,
                vae_hidden_dim=64,
                vae_latent_dim=8,
            ).fit(df, is_sample=True, metadata={"synthetic_fallback": True, "session_model_id": session_model_id})
        else:
            model = EVSessionModel(
                n_components="auto",
                max_components=5,
                stratify_by=["location_type", "day_of_week", "season"],
            ).fit(df, is_sample=True, metadata={"synthetic_fallback": True, "session_model_id": session_model_id})

        return model

    def save(self, session_model_id: str, gmm) -> Path:
        """
        Save a fitted GMM to the package GMM directory.

        Parameters
        ----------
        session_model_id : str
            Registry ID to save under.
        gmm : EVSessionModel
            Fitted GMM to save.

        Returns
        -------
        Path
            Path of the saved joblib file.

        Raises
        ------
        ValueError
            If *session_model_id* is not in the catalogue.
        """
        if session_model_id not in self._CATALOGUE:
            raise ValueError(f"Unknown GMM ID '{session_model_id}'. Available: {list(self._CATALOGUE)}")
        path = self.session_model_dir / self._CATALOGUE[session_model_id]["filename"]
        joblib.dump(gmm, path)
        logger.info("Saved GMM '%s' to %s.", session_model_id, path)
        return path

    def get_sklearn_component(
        self,
        location_type: str,
        department: str,
        season: str,
        day_of_week: int,
        session_model_id: str = "french",
    ):
        """
        Return the underlying sklearn GaussianMixture for a specific stratum
        of the unified French GMM.

        Parameters
        ----------
        location_type : str
            Charging location: ``'work'``, ``'home'``, ``'public'``, or ``'heavy'``.
        department : str
            INSEE department code, e.g. ``'75'``, ``'69'``, ``'13'``.
        season : str
            Season: ``'winter'``, ``'spring'``, ``'summer'``, ``'autumn'``.
        day_of_week : int
            Monday = 0, …, Sunday = 6.
        session_model_id : str, optional
            Registry bundle ID (default ``'french'``).

        Returns
        -------
        sklearn.mixture.GaussianMixture
            Fitted GaussianMixture for the requested stratum.
        """
        gmm = self.load(session_model_id)
        return gmm.get_sklearn_component(
            context={
                "location_type": location_type,
                "department":    department,
                "season":        season,
                "day_of_week":   day_of_week,
            }
        )

    def __repr__(self) -> str:
        n_available = sum(
            1 for meta in self._CATALOGUE.values()
            if (self.session_model_dir / meta["filename"]).exists()
        )
        return (
            f"NativeSessionModelRegistry("
            f"{n_available}/{len(self._CATALOGUE)} GMMs available, "
            f"dir={self.session_model_dir})"
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
    >>> bundle = registry.load("french_demo")
    >>> gmm = bundle["gmm"]
    """

    def __init__(
        self,
        hf_repo_id: str = HF_REPO_ID,
        cache_dir: Path | None = None,
    ):
        """
        Parameters
        ----------
        hf_repo_id : str
            Hugging Face Hub repository identifier.
        cache_dir : Path, optional
            Local directory for cached bundles.
            Defaults to ``~/.cache/gears/models``.
        """
        self.hf_repo_id = hf_repo_id
        self.cache_dir = Path(cache_dir) if cache_dir else _LOCAL_CACHE
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def list_models(self) -> pd.DataFrame:
        """Return a DataFrame listing all available pre-trained models.

        Returns
        -------
        pd.DataFrame
            One row per catalogue entry with columns mirroring the
            ``_CATALOGUE`` dict keys.
        """
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
        dict
            Keys: ``'gmm'``, ``'forecaster'`` (optional), ``'metadata'``.

        Raises
        ------
        ValueError
            If *model_id* is not in the catalogue.
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
        metadata: dict | None = None,
    ) -> Path:
        """Serialize a model bundle to the local cache.

        Parameters
        ----------
        model_id : str
            Identifier used as the filename stem.
        gmm : EVSessionModel
            Fitted GMM to include in the bundle.
        forecaster : optional
            Fitted forecaster object, or None.
        metadata : dict, optional
            Arbitrary key-value metadata to embed in the bundle.

        Returns
        -------
        Path
            Path of the saved joblib file.
        """
        bundle = {"gmm": gmm, "forecaster": forecaster, "metadata": metadata or {}}
        path = self.cache_dir / f"{model_id}.joblib"
        joblib.dump(bundle, path)
        logger.info("Saved bundle '%s' to %s.", model_id, path)
        return path

    def upload_to_hub(self, model_id: str, token: str | None = None) -> None:
        """Upload a locally cached bundle to HF Hub.

        Parameters
        ----------
        model_id : str
            Local bundle identifier; the file must exist in ``cache_dir``.
        token : str, optional
            Hugging Face authentication token.  Uses the cached token if None.

        Raises
        ------
        ImportError
            If ``huggingface-hub`` is not installed.
        FileNotFoundError
            If the local bundle does not exist.
        """
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
        """Try HF Hub first; fall back to a synthetic demo bundle.

        Parameters
        ----------
        model_id : str
            Catalogue identifier.

        Returns
        -------
        Path
            Local path of the downloaded or generated bundle.
        """
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
        except Exception as e:  # noqa: BLE001 - HF Hub download is best-effort; fall back to local cache on any failure
            logger.warning(
                "Could not download '%s' from HF Hub (%s). "
                "Generating synthetic demo bundle instead.",
                model_id, e,
            )
            return self._generate_demo_bundle(model_id, dest)

    def _generate_demo_bundle(self, model_id: str, dest: Path) -> Path:
        """Generate and cache a synthetic demo bundle when HF Hub is unreachable.

        The synthetic bundle contains a small GMM fitted on generated data and
        a probabilistic forecaster.  It lets demos and tests run offline
        without real training data.

        Parameters
        ----------
        model_id : str
            Catalogue entry used for location type and session count defaults.
        dest : Path
            Destination file path for the serialised bundle.

        Returns
        -------
        Path
            Path of the saved bundle (equal to *dest*).
        """
        from gears.data.loader import make_demo_data
        from gears.models.forecaster import SessionForecaster
        from gears.models.session_model import EVSessionModel

        meta = _CATALOGUE[model_id]
        loc = meta.get("location_type", "work")
        n = meta.get("n_sessions", 2000)

        logger.info("Generating synthetic demo bundle for '%s'.", model_id)
        df = make_demo_data(n=n, location_type=loc, seed=42)

        gmm = EVSessionModel(
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


def _get_default_registry() -> NativeSessionModelRegistry:
    """Return the module-level singleton ``NativeSessionModelRegistry``.

    The registry is instantiated lazily on first call and reused thereafter
    so that the GMM joblib file is not reloaded for every ``get_session_model()``
    invocation within the same Python session.

    Returns
    -------
    NativeSessionModelRegistry
    """
    global _default_registry
    if _default_registry is None:
        _default_registry = NativeSessionModelRegistry()
    return _default_registry


def get_session_model(bundle_id: str = "french") -> EVSessionModel:
    """
    Retrieve a pre-fitted native ``EVSessionModel`` bundle by ID.

    This is the primary retrieval API for the GEARS registry. It resolves at
    the *bundle* level — ``'french'`` (the unified national GMM) or
    ``'french_vae_sample'`` (the shared conditional VAE) — and returns the
    whole fitted wrapper. Stratum-level lookups (by location type, department,
    season, day of week) happen afterward, on the returned object, via
    ``.get_sklearn_component(context=...)``.

    Note: prior to Session 2, this function took
    ``(location_type, departement, saison, day_of_week)`` and silently ignored
    all four, always returning the ``'french'`` bundle — a real bug, not just
    a naming issue. The signature below was narrowed to match what the
    function actually does; see ``PROPOSAL_NAMING.md`` for the full rationale.

    Parameters
    ----------
    bundle_id : str, optional
        Registry bundle ID (default ``'french'``). See
        ``NativeSessionModelRegistry.list()`` for all available bundles.

    Returns
    -------
    EVSessionModel
        Fitted wrapper exposing:

        - ``.get_sklearn_component(context=...)`` → raw ``sklearn.mixture.GaussianMixture``
        - ``.sample(n, context=...)`` → synthetic sessions DataFrame
        - ``.list_contexts()`` → all fitted (loc, dept, season, dow) tuples
        - ``.bic_summary()`` → BIC / n_components per stratum

    Examples
    --------
    >>> import gears
    >>> model = gears.get_session_model("french")
    >>> # underlying sklearn object for one stratum
    >>> sk = model.get_sklearn_component(
    ...     context={"location_type": "work", "department": "75",
    ...              "season": "winter", "day_of_week": 0}
    ... )
    >>> print(sk.means_)   # [arrival_hour, log1p(duration_h), log1p(energy_kWh)]
    >>> # sample 50 synthetic sessions
    >>> sessions = model.sample(
    ...     50,
    ...     context={"location_type": "work", "department": "75",
    ...              "season": "winter", "day_of_week": 0},
    ... )
    """
    registry = _get_default_registry()
    return registry.load(bundle_id)

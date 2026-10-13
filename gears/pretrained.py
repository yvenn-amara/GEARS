"""
Pre-trained session models.

``load_pretrained("french")`` returns a fitted
:class:`~gears.models.session_model.EVSessionModel`. The model file is a GitHub
Release asset: it is downloaded once, checked against the SHA-256 stored in
``_CATALOGUE``, and cached on disk. Nothing is downloaded or written when you
``import gears``.

Environment variables
---------------------
``GEARS_HOME``
    Cache directory. Default: ``~/.cache/gears``. GEARS never writes inside
    its own package folder.
``GEARS_OFFLINE``
    Set to ``1`` to forbid any network access. A model already in the cache is
    still used; otherwise a :class:`PretrainedModelError` explains what to do.

Machine without internet access
-------------------------------
Download the file listed under ``url`` in ``_CATALOGUE`` on another machine,
then either copy it into the cache directory (``$GEARS_HOME``), or pass its
location with ``load_pretrained("french", path="/where/it/is/gmm_french.joblib")``.

There is no fallback: when the model cannot be loaded, a
:class:`PretrainedModelError` is raised, never a substitute model.
"""

from __future__ import annotations

import hashlib
import http.client
import logging
import os
import shutil
import tempfile
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Final

import joblib

if TYPE_CHECKING:
    from gears.models.session_model import EVSessionModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Where the model files are published. These two constants are the only place
# that names the repository; they move to EDF-Lab/GEARS in step S12.
REPO: Final = "yvenn-amara/GEARS"
RELEASE_TAG: Final = "models-v1"
_RELEASE_URL: Final = f"https://github.com/{REPO}/releases/download/{RELEASE_TAG}"

# SHA-256 of a release asset: ``sha256sum gmm_french.joblib``. GitHub also shows
# it as the asset's "digest" on the release page.
_CATALOGUE: Final[dict[str, dict[str, str]]] = {
    "french": {
        "file": "gmm_french.joblib",
        "sha256": "ea8a65da5718c6a13139d48e5e0171796d3ab630dc89c2d275d63b22feabff3d",
        "url": f"{_RELEASE_URL}/gmm_french.joblib",
    },
}

_DOWNLOAD_TIMEOUT_S: Final = 30
_DOWNLOAD_ATTEMPTS: Final = 2

# URLError, HTTPError and timeouts are OSErrors; a transfer cut short raises
# http.client.IncompleteRead, which is not.
_NETWORK_ERRORS: Final = (OSError, http.client.HTTPException)


class PretrainedModelError(Exception):
    """A pre-trained model could not be found, downloaded, verified or loaded."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_cache_dir() -> Path:
    """Cache directory: ``$GEARS_HOME`` if set, otherwise ``~/.cache/gears``."""
    env_home = os.environ.get("GEARS_HOME")
    if env_home:
        return Path(env_home).expanduser()
    return Path.home() / ".cache" / "gears"


def _is_offline() -> bool:
    return os.environ.get("GEARS_OFFLINE") == "1"


def _sha256_of(file_path: Path) -> str:
    digest = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_sha256(file_path: Path, expected_sha256: str) -> bool:
    """True if the SHA-256 of ``file_path`` equals ``expected_sha256``."""
    return _sha256_of(file_path) == expected_sha256.lower()


def _manual_install_hint(model_id: str, local_path: Path) -> str:
    entry = _CATALOGUE[model_id]
    return (
        f"To install the model by hand: download {entry['url']} on a machine with "
        f"internet access, then copy it to {local_path} (or set GEARS_HOME to the "
        f"folder that contains it), or call load_pretrained('{model_id}', path=...)."
    )


def _download_file(
    url: str,
    dest_path: Path,
    expected_sha256: str,
    timeout: float = _DOWNLOAD_TIMEOUT_S,
    retries: int = _DOWNLOAD_ATTEMPTS,
) -> None:
    """
    Download ``url`` to ``dest_path`` and verify it **before** it becomes visible.

    The data goes to a temporary file in the destination folder. Only a file
    whose SHA-256 matches ``expected_sha256`` is renamed to ``dest_path``, so a
    corrupt or truncated download never lands in the cache.

    Raises
    ------
    PretrainedModelError
        If the downloaded bytes do not match ``expected_sha256``.
    OSError, http.client.HTTPException
        If the transfer fails on every attempt (URLError, HTTPError and
        timeouts are OSErrors).
    """
    import urllib.request

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=dest_path.parent, prefix=f"{dest_path.name}.", suffix=".part"
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as tmp:
            for attempt in range(1, retries + 1):
                try:
                    logger.info(
                        "Downloading %s (attempt %d/%d)...", url, attempt, retries
                    )
                    with urllib.request.urlopen(url, timeout=timeout) as response:
                        shutil.copyfileobj(response, tmp)
                    break
                except _NETWORK_ERRORS as e:
                    if attempt == retries:
                        raise
                    logger.warning("Download attempt %d failed (%s); retrying.", attempt, e)
                    tmp.seek(0)
                    tmp.truncate()

        actual = _sha256_of(tmp_path)
        if actual != expected_sha256.lower():
            raise PretrainedModelError(
                f"The file downloaded from {url} does not match the expected SHA-256 "
                f"(got {actual}, expected {expected_sha256}). It was discarded; "
                "nothing was written to the cache."
            )
        tmp_path.replace(dest_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _read_session_model(local_path: Path, model_id: str) -> tuple[EVSessionModel, str | None]:
    """
    Unpickle ``local_path``; return the model and the scikit-learn version it
    was trained with (``None`` if unknown).

    scikit-learn warns by itself, once per estimator, when it unpickles an
    estimator saved by another version. That warning is captured here and
    replaced by the single, actionable one emitted by :func:`load_pretrained`;
    every other warning raised while loading is passed on unchanged.
    """
    from sklearn.exceptions import InconsistentVersionWarning

    from gears.models.session_model import EVSessionModel

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", InconsistentVersionWarning)
        try:
            model = joblib.load(local_path)
        except Exception as e:  # the file passed the checksum: this is an environment problem
            raise PretrainedModelError(
                f"Could not load model '{model_id}' from {local_path}: "
                f"{type(e).__name__}: {e}. The file is the official one, so the installed "
                "scikit-learn / numpy / pandas versions are probably incompatible with the "
                "ones used to train it."
            ) from e

    trained_with: str | None = None
    for w in caught:
        if issubclass(w.category, InconsistentVersionWarning):
            trained_with = trained_with or w.message.original_sklearn_version
        else:
            warnings.warn_explicit(w.message, w.category, w.filename, w.lineno, source=w.source)

    if not isinstance(model, EVSessionModel):
        raise PretrainedModelError(
            f"{local_path} does not contain an EVSessionModel (got {type(model).__name__})."
        )
    metadata = getattr(model, "metadata_", None) or {}
    return model, trained_with or metadata.get("sklearn_version")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_pretrained(
    model_id: str,
    path: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> EVSessionModel:
    """
    Load a pre-trained EV session model.

    The file is looked up in the cache (``cache_dir``, else ``$GEARS_HOME``,
    else ``~/.cache/gears``) and downloaded when missing, unless
    ``GEARS_OFFLINE=1``. Its SHA-256 is always checked before it is unpickled
    (a pickle can run arbitrary code), whether it comes from the cache, from
    ``path`` or from a download.

    Parameters
    ----------
    model_id : str
        Model identifier, e.g. ``"french"``.
    path : str or Path, optional
        Location of the official model file, for machines without internet
        access. Bypasses the cache and the download; the checksum still applies,
        so this is not a way to load your own model (use ``GEARSModel.load`` or
        ``joblib.load`` for that).
    cache_dir : str or Path, optional
        Cache directory, overriding ``$GEARS_HOME``.

    Returns
    -------
    EVSessionModel
        The fitted session model.

    Raises
    ------
    PretrainedModelError
        Unknown ``model_id``; file missing in offline mode; download failure;
        SHA-256 mismatch; unreadable file; file that is not an ``EVSessionModel``.
        The message says what to do next.

    Warns
    -----
    UserWarning
        If the installed scikit-learn differs from the one used for training.

    Examples
    --------
    >>> from gears import load_pretrained
    >>> gmm = load_pretrained("french")          # doctest: +SKIP
    """
    if model_id not in _CATALOGUE:
        raise PretrainedModelError(
            f"Unknown model ID '{model_id}'. Available models: {sorted(_CATALOGUE)}."
        )
    entry = _CATALOGUE[model_id]

    # ``checked`` is True once the checksum has been verified (right after a download).
    checked = False
    if path is not None:
        local_path = Path(path).expanduser()
        if not local_path.is_file():
            raise PretrainedModelError(f"No model file at {local_path} (path=...).")
    else:
        active_cache = Path(cache_dir).expanduser() if cache_dir else _get_cache_dir()
        local_path = active_cache / entry["file"]
        if not local_path.is_file():
            if _is_offline():
                raise PretrainedModelError(
                    f"GEARS_OFFLINE=1 and model '{model_id}' is not in the cache "
                    f"({local_path}). " + _manual_install_hint(model_id, local_path)
                )
            try:
                _download_file(entry["url"], local_path, entry["sha256"])
            except _NETWORK_ERRORS as e:
                raise PretrainedModelError(
                    f"Could not download model '{model_id}' from {entry['url']}: "
                    f"{type(e).__name__}: {e}. " + _manual_install_hint(model_id, local_path)
                ) from e
            checked = True

    if not checked and not _verify_sha256(local_path, entry["sha256"]):
        raise PretrainedModelError(
            f"Integrity check failed for model '{model_id}': {local_path} does not match "
            f"the expected SHA-256 ({entry['sha256']}). The file is corrupted or is not the "
            "official one. Delete it and call load_pretrained again to download it anew."
        )

    model, trained_with = _read_session_model(local_path, model_id)

    import sklearn

    if trained_with and trained_with != sklearn.__version__:
        warnings.warn(
            f"Model '{model_id}' was trained with scikit-learn {trained_with}, but "
            f"scikit-learn {sklearn.__version__} is installed. It may load with different "
            f"results or fail later; to get exactly the original behaviour, install "
            f"scikit-learn=={trained_with}.",
            UserWarning,
            stacklevel=2,
        )

    logger.info("Loaded pre-trained model '%s' from %s.", model_id, local_path)
    return model

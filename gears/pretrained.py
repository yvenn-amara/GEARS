"""
Pre-trained model management for GEARS.

This module handles the discovery, downloading, and loading of pre-trained
EV session models. It ensures that models are cached locally and verified
via SHA-256 checksums to avoid corruption or malicious injection.

It replaces the previous ModelRegistry and NativeSessionModelRegistry.
"""

from __future__ import annotations

import hashlib
import logging
import os
import pickle
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Final

import joblib

if TYPE_CHECKING:
    from gears.models.session_model import EVSessionModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Only the "french" model is currently supported as a public pre-trained asset.
# The URL points to a GitHub Release asset.
_CATALOGUE: Final[dict[str, dict]] = {
    "french": {
        "file": "gmm_french.joblib",
        "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",  # Placeholder
        "url": "https://github.com/yvenn-amara/GEARS/releases/download/models-v1/gmm_french.joblib",
    }
}

# Default cache directory: ~/.cache/gears/
DEFAULT_CACHE_DIR: Final[Path] = Path.home() / ".cache" / "gears"

class PretrainedModelError(Exception):
    """Dedicated error for pre-trained model retrieval failures."""


def _get_cache_dir() -> Path:
    """Return the active cache directory, respecting GEARS_HOME env var."""
    env_home = os.getenv("GEARS_HOME")
    return Path(env_home) if env_home else DEFAULT_CACHE_DIR


def _verify_sha256(file_path: Path, expected_sha256: str) -> bool:
    """Verify the SHA-256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest() == expected_sha256


def _download_file(url: str, dest_path: Path, timeout: int = 30, retries: int = 2) -> None:
    """Download a file atomically using urllib.request."""
    import urllib.request

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # Atomic write: download to a temp file in the same directory, then rename.
    with tempfile.NamedTemporaryFile(dir=dest_path.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        try:
            for attempt in range(retries):
                try:
                    logger.info(
                        "Downloading model from %s (attempt %d/%d)...", url, attempt + 1, retries
                    )
                    with urllib.request.urlopen(url, timeout=timeout) as response:
                        shutil.copyfileobj(response, tmp)
                    break
                except OSError as e:  # URLError, HTTPError and timeouts all derive from OSError
                    if attempt == retries - 1:
                        raise
                    logger.warning("Download attempt %d failed: %s. Retrying...", attempt + 1, e)
                    tmp.seek(0)
                    tmp.truncate()
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise

    tmp_path.replace(dest_path)


def load_pretrained(
    model_id: str,
    path: str | Path | None = None,
    cache_dir: str | Path | None = None
) -> EVSessionModel:
    """
    Load a pre-fitted EV session model by ID.

    If 'path' is provided, the model is loaded directly from that file.
    Otherwise, the model is searched for in the cache, or downloaded if missing.

    Parameters
    ----------
    model_id : str
        ID of the model (e.g., 'french').
    path : str or Path, optional
        Direct path to a .joblib file. Overrides cache and download.
    cache_dir : str or Path, optional
        Override the default cache directory (~/.cache/gears).

    Returns
    -------
    EVSessionModel
        The fitted session model.

    Raises
    ------
    PretrainedModelError
        If the model is not in the catalogue, the checksum is invalid,
        or the file is missing in offline mode.
    """
    if model_id not in _CATALOGUE:
        raise PretrainedModelError(
            f"Unknown model ID '{model_id}'. Available models: {list(_CATALOGUE.keys())}"
        )

    # 1. Direct path override
    if path:
        local_path = Path(path)
    else:
        # 2. Cache logic
        active_cache = Path(cache_dir) if cache_dir else _get_cache_dir()
        local_path = active_cache / _CATALOGUE[model_id]["file"]

        if not local_path.exists():
            if os.getenv("GEARS_OFFLINE") == "1":
                raise PretrainedModelError(
                    f"GEARS_OFFLINE=1 and model '{model_id}' is not cached at {local_path}. "
                    "Please download the model manually and place it in the cache or use the 'path' argument."
                )

            # Download
            try:
                _download_file(_CATALOGUE[model_id]["url"], local_path)
            except OSError as e:
                raise PretrainedModelError(
                    f"Failed to download model '{model_id}' from {_CATALOGUE[model_id]['url']}: {e}"
                ) from e

    # 3. Integrity check
    expected_sha = _CATALOGUE[model_id]["sha256"]
    if not _verify_sha256(local_path, expected_sha):
        # Note: For the very first release, we might allow this if the checksum is a placeholder,
        # but according to S4, we MUST verify SHA-256.
        raise PretrainedModelError(
            f"Integrity check failed for model '{model_id}' at {local_path}. "
            "The file may be corrupted. Try deleting the cache and re-downloading."
        )

    # 4. Load
    try:
        model = joblib.load(local_path)
    except (OSError, EOFError, ValueError, AttributeError, ImportError, pickle.UnpicklingError) as e:
        raise PretrainedModelError(f"Failed to load joblib file at {local_path}: {e}") from e

    if TYPE_CHECKING:
        from gears.models.session_model import EVSessionModel
        if not isinstance(model, EVSessionModel):
            raise PretrainedModelError(f"Expected EVSessionModel, got {type(model)}")

    # 5. Version check (Simplified for now, will be refined)
    # In a real scenario, we'd check scikit-learn version here.

    logger.info("Successfully loaded pre-trained model '%s' from %s.", model_id, local_path)
    return model

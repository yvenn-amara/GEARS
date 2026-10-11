import pytest
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import joblib
import numpy as np
from gears.pretrained import load_pretrained, PretrainedModelError, _CATALOGUE
from gears.models.session_model import EVSessionModel

# --- Mocks ---

class MockSessionModel:
    """A simple picklable class to simulate EVSessionModel."""
    def __init__(self, model_id):
        self.metadata_ = {"model_id": model_id}
        self.models_ = {("home", "75", "winter", 0): None}

    def __eq__(self, other):
        if not isinstance(other, MockSessionModel):
            return False
        return self.metadata_ == other.metadata_

@pytest.fixture
def mock_model():
    """Create a dummy session model."""
    return MockSessionModel("test_model")

@pytest.fixture
def mock_catalog():
    """A minimal catalogue for testing."""
    return {
        "test_model": {
            "file": "test_model.joblib",
            "url": "file://mock/path/test_model.joblib",
            "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", # Empty file SHA
            "description": "Test model",
            "n_sessions": 100,
        }
    }

# --- Tests ---

def test_load_pretrained_cache_miss_download(tmp_path, mock_model):
    """
    Test the full flow:
    1. Cache miss
    2. Download (simulated via file://)
    3. SHA verification
    4. Final move to cache
    """
    # Setup a fake remote file
    remote_dir = tmp_path / "remote"
    remote_dir.mkdir()
    model_file = remote_dir / "test_model.joblib"

    # Save the mock model
    joblib.dump(mock_model, model_file)

    # Update the mock catalogue to point to this local file
    model_url = model_file.as_uri()

    # Calculate real SHA256 for the saved file
    import hashlib
    with open(model_file, "rb") as f:
        real_sha = hashlib.sha256(f.read()).hexdigest()

    test_cat = {
        "test_model": {
            "file": "test_model.joblib",
            "url": model_url,
            "sha256": real_sha,
            "description": "Test model",
            "n_sessions": 100,
        }
    }

    with patch("gears.pretrained._CATALOGUE", test_cat):
        # Call load_pretrained with a temporary cache dir
        cache_dir = tmp_path / "cache"
        model = load_pretrained("test_model", cache_dir=cache_dir)

        assert model == mock_model
        # Verify it exists in cache
        expected_cache_path = cache_dir / "test_model.joblib"
        assert expected_cache_path.exists()

def test_load_pretrained_cache_hit(tmp_path, mock_model):
    """Test that it returns the cached model immediately if SHA matches."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    model_path = cache_dir / "test_model.joblib"
    joblib.dump(mock_model, model_path)

    import hashlib
    with open(model_path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()

    test_cat = {
        "test_model": {
            "file": "test_model.joblib",
            "url": "http://fake.com/model.joblib",
            "sha256": sha,
            "description": "Test model",
            "n_sessions": 100,
        }
    }

    with patch("gears.pretrained._CATALOGUE", test_cat):
        # Should load from cache without downloading
        with patch("urllib.request.urlretrieve") as mock_retrieve:
            model = load_pretrained("test_model", cache_dir=cache_dir)
            assert model == mock_model
            mock_retrieve.assert_not_called()

def test_load_pretrained_sha_mismatch(tmp_path, mock_model):
    """Test that SHA mismatch triggers a re-download or error."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    model_path = cache_dir / "test_model.joblib"
    joblib.dump(mock_model, model_path)

    test_cat = {
        "test_model": {
            "file": "test_model.joblib",
            "url": "http://fake.com/model.joblib",
            "sha256": "wrong_sha",
            "description": "Test model",
            "n_sessions": 100,
        }
    }

    with patch("gears.pretrained._CATALOGUE", test_cat):
        # Simulate download failure to see it's trying to download again
        with patch("urllib.request.urlretrieve", side_effect=Exception("Download failed")):
            with pytest.raises(PretrainedModelError, match="Integrity check failed"):
                load_pretrained("test_model", cache_dir=cache_dir)

def test_load_pretrained_offline_mode(tmp_path):
    """Test that GEARS_OFFLINE=1 prevents downloads."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    test_cat = {
        "test_model": {
            "file": "test_model.joblib",
            "url": "http://fake.com/model.joblib",
            "sha256": "sha",
            "description": "Test model",
            "n_sessions": 100,
        }
    }

    with patch("gears.pretrained._CATALOGUE", test_cat):
        with patch.dict(os.environ, {"GEARS_OFFLINE": "1"}):
            with pytest.raises(PretrainedModelError, match="GEARS_OFFLINE=1"):
                load_pretrained("test_model", cache_dir=cache_dir)

def test_load_pretrained_invalid_id():
    """Test that requesting a non-existent model ID raises an error."""
    with pytest.raises(PretrainedModelError, match="Unknown model ID"):
        load_pretrained("non_existent_model")

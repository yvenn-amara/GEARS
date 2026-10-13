"""
Tests for ``gears.pretrained``.

No test touches the network or the real cache: the "release" is a ``file://``
URL and ``GEARS_HOME`` points to a temporary folder.
"""

from __future__ import annotations

import hashlib
import http.client
import re
import urllib.error
import urllib.request
import warnings
from pathlib import Path

import joblib
import pytest
import sklearn.base
from sklearn.exceptions import InconsistentVersionWarning

import gears
from gears import pretrained
from gears.models.session_model import EVSessionModel
from gears.pretrained import PretrainedModelError, load_pretrained

EMPTY_FILE_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_model(work_sessions):
    """A real, small, fitted EVSessionModel."""
    return EVSessionModel(n_components=2, stratify_by=["season"]).fit(work_sessions)


@pytest.fixture(autouse=True)
def isolated_environment(tmp_path, monkeypatch):
    """Keep every test away from the real cache and out of offline mode."""
    monkeypatch.setenv("GEARS_HOME", str(tmp_path / "gears_home"))
    monkeypatch.delenv("GEARS_OFFLINE", raising=False)


@pytest.fixture
def release(tmp_path, tiny_model, monkeypatch):
    """A fake release: one model file served through ``file://``."""
    remote = tmp_path / "remote" / "test_model.joblib"
    remote.parent.mkdir()
    joblib.dump(tiny_model, remote)
    catalogue = {
        "test": {
            "file": "test_model.joblib",
            "sha256": _sha256(remote),
            "url": remote.as_uri(),
        }
    }
    monkeypatch.setattr(pretrained, "_CATALOGUE", catalogue)
    return remote


@pytest.fixture
def cache(tmp_path) -> Path:
    """Where ``GEARS_HOME`` points to (created lazily by the code under test)."""
    return tmp_path / "gears_home"


@pytest.fixture
def no_network(monkeypatch):
    """Any attempt to open a URL fails the test."""
    def _forbidden(*args, **kwargs):
        raise AssertionError("the network must not be used here")

    monkeypatch.setattr(urllib.request, "urlopen", _forbidden)


# ---------------------------------------------------------------------------
# The shipped catalogue
# ---------------------------------------------------------------------------

def test_catalogue_entry_is_well_formed():
    entry = pretrained._CATALOGUE["french"]
    assert re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])
    assert entry["sha256"] != EMPTY_FILE_SHA256  # the placeholder used before the first release
    assert entry["url"] == f"https://github.com/{pretrained.REPO}/releases/download/" \
                           f"{pretrained.RELEASE_TAG}/{entry['file']}"


# ---------------------------------------------------------------------------
# Success paths
# ---------------------------------------------------------------------------

def test_download_verify_cache_and_load(release, cache):
    model = load_pretrained("test")

    assert isinstance(model, EVSessionModel)
    assert model.is_fitted_
    assert (cache / "test_model.joblib").read_bytes() == release.read_bytes()
    assert list(cache.iterdir()) == [cache / "test_model.joblib"]  # no temporary file left


def test_second_call_uses_the_cache_without_network(release, cache, monkeypatch):
    load_pretrained("test")
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: pytest.fail("network used"))

    assert isinstance(load_pretrained("test"), EVSessionModel)


def test_cache_dir_argument_overrides_gears_home(release, tmp_path, cache):
    other = tmp_path / "other_cache"

    load_pretrained("test", cache_dir=other)

    assert (other / "test_model.joblib").exists()
    assert not cache.exists()


def test_default_cache_dir_is_under_the_user_home(monkeypatch, tmp_path):
    monkeypatch.delenv("GEARS_HOME")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    assert pretrained._get_cache_dir() == tmp_path / ".cache" / "gears"


def test_path_argument_loads_a_local_file_without_cache_or_network(release, cache, no_network):
    model = load_pretrained("test", path=release)

    assert isinstance(model, EVSessionModel)
    assert not cache.exists()


# ---------------------------------------------------------------------------
# Integrity
# ---------------------------------------------------------------------------

def test_wrong_checksum_after_download_is_an_error_and_nothing_is_cached(
    release, cache, monkeypatch
):
    monkeypatch.setitem(pretrained._CATALOGUE["test"], "sha256", "0" * 64)

    with pytest.raises(PretrainedModelError, match="does not match the expected SHA-256"):
        load_pretrained("test")

    assert not cache.exists() or list(cache.iterdir()) == []  # not even a .part file


def test_corrupted_cache_file_is_an_error_and_is_left_untouched(release, cache, no_network):
    cache.mkdir()
    corrupted = cache / "test_model.joblib"
    corrupted.write_bytes(b"not the model")

    with pytest.raises(PretrainedModelError, match="Integrity check failed"):
        load_pretrained("test")

    assert corrupted.read_bytes() == b"not the model"


def test_path_argument_is_checksum_verified(release, tmp_path, no_network):
    impostor = tmp_path / "impostor.joblib"
    impostor.write_bytes(b"not the model")

    with pytest.raises(PretrainedModelError, match="Integrity check failed"):
        load_pretrained("test", path=impostor)


def test_path_argument_missing_file(tmp_path, release):
    with pytest.raises(PretrainedModelError, match="No model file"):
        load_pretrained("test", path=tmp_path / "nope.joblib")


def test_file_that_is_not_a_session_model_is_rejected(tmp_path, monkeypatch, cache):
    remote = tmp_path / "wrong_type.joblib"
    joblib.dump({"not": "a model"}, remote)
    monkeypatch.setattr(pretrained, "_CATALOGUE", {
        "test": {"file": remote.name, "sha256": _sha256(remote), "url": remote.as_uri()}
    })

    with pytest.raises(PretrainedModelError, match="does not contain an EVSessionModel"):
        load_pretrained("test")


def test_unreadable_file_with_a_valid_checksum_is_a_clean_error(tmp_path, monkeypatch):
    remote = tmp_path / "garbage.joblib"
    remote.write_bytes(b"\x00garbage, but with a known checksum")
    monkeypatch.setattr(pretrained, "_CATALOGUE", {
        "test": {"file": remote.name, "sha256": _sha256(remote), "url": remote.as_uri()}
    })

    with pytest.raises(PretrainedModelError, match="Could not load model 'test'"):
        load_pretrained("test")


def test_unknown_model_id():
    with pytest.raises(PretrainedModelError, match="Unknown model ID 'atlantis'.*french"):
        load_pretrained("atlantis")


# ---------------------------------------------------------------------------
# Offline mode
# ---------------------------------------------------------------------------

def test_offline_without_cache_is_a_clear_error(release, cache, monkeypatch, no_network):
    monkeypatch.setenv("GEARS_OFFLINE", "1")

    with pytest.raises(PretrainedModelError) as excinfo:
        load_pretrained("test")

    message = str(excinfo.value)
    assert "GEARS_OFFLINE=1" in message
    assert str(cache / "test_model.joblib") in message  # where to put the file
    assert release.as_uri() in message                  # where to get it
    assert "path=" in message
    assert not cache.exists()


def test_offline_with_cache_is_fine(release, cache, monkeypatch):
    load_pretrained("test")  # populate the cache while "online"
    monkeypatch.setenv("GEARS_OFFLINE", "1")
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: pytest.fail("network used"))

    assert isinstance(load_pretrained("test"), EVSessionModel)


@pytest.mark.parametrize("value", ["0", "", "true"])
def test_only_offline_equal_to_1_forbids_the_network(release, monkeypatch, value):
    monkeypatch.setenv("GEARS_OFFLINE", value)

    assert isinstance(load_pretrained("test"), EVSessionModel)


# ---------------------------------------------------------------------------
# Download failures
# ---------------------------------------------------------------------------

def _urlopen_failing_first(monkeypatch, n_failures, error):
    """urlopen that raises ``error`` ``n_failures`` times, then really opens the URL."""
    real_urlopen = urllib.request.urlopen
    calls = []

    def flaky(url, *args, **kwargs):
        calls.append(url)
        if len(calls) <= n_failures:
            raise error
        return real_urlopen(url, *args, **kwargs)

    monkeypatch.setattr(urllib.request, "urlopen", flaky)
    return calls


def test_one_failed_attempt_is_retried(release, monkeypatch):
    calls = _urlopen_failing_first(monkeypatch, 1, urllib.error.URLError("flaky wifi"))

    assert isinstance(load_pretrained("test"), EVSessionModel)
    assert len(calls) == 2


def test_two_failed_attempts_give_an_actionable_error(release, cache, monkeypatch):
    calls = _urlopen_failing_first(monkeypatch, 99, urllib.error.URLError("no route to host"))

    with pytest.raises(PretrainedModelError) as excinfo:
        load_pretrained("test")

    message = str(excinfo.value)
    assert len(calls) == 2  # 2 attempts, as in the plan
    assert "no route to host" in message
    assert release.as_uri() in message
    assert "path=" in message
    assert not cache.exists() or list(cache.iterdir()) == []


def test_truncated_transfer_is_a_clean_error(release, cache, monkeypatch):
    _urlopen_failing_first(monkeypatch, 99, http.client.IncompleteRead(b"abc", 100))

    with pytest.raises(PretrainedModelError, match="IncompleteRead"):
        load_pretrained("test")

    assert not cache.exists() or list(cache.iterdir()) == []


# ---------------------------------------------------------------------------
# Library versions
# ---------------------------------------------------------------------------

def test_fit_records_library_versions(tiny_model):
    assert tiny_model.metadata_["sklearn_version"] == sklearn.__version__
    assert tiny_model.metadata_["gears_version"] == gears.__version__


def test_matching_scikit_learn_version_gives_no_warning(release):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        load_pretrained("test")

    assert not [w for w in caught if "scikit-learn" in str(w.message)]


def test_other_scikit_learn_version_gives_one_clear_warning(tmp_path, tiny_model, monkeypatch):
    # Pickle the estimators as if they came from scikit-learn 1.0.0.
    old_bundle = tmp_path / "old.joblib"
    with monkeypatch.context() as m:
        m.setattr(sklearn.base, "__version__", "1.0.0")
        joblib.dump(tiny_model, old_bundle)
    monkeypatch.setattr(pretrained, "_CATALOGUE", {
        "test": {"file": old_bundle.name, "sha256": _sha256(old_bundle), "url": old_bundle.as_uri()}
    })

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = load_pretrained("test")

    assert isinstance(model, EVSessionModel)
    ours = [w for w in caught if "was trained with scikit-learn" in str(w.message)]
    assert len(ours) == 1
    assert issubclass(ours[0].category, UserWarning)
    assert f"scikit-learn {sklearn.__version__} is installed" in str(ours[0].message)
    assert "scikit-learn==1.0.0" in str(ours[0].message)
    # scikit-learn's own per-estimator warnings are replaced by that single one
    assert not [w for w in caught if issubclass(w.category, InconsistentVersionWarning)]


def test_version_recorded_in_metadata_also_triggers_the_warning(tmp_path, tiny_model, monkeypatch):
    bundle = tmp_path / "meta.joblib"
    tiny_model.metadata_["sklearn_version"] = "0.99.0"
    try:
        joblib.dump(tiny_model, bundle)
    finally:
        tiny_model.metadata_["sklearn_version"] = sklearn.__version__
    monkeypatch.setattr(pretrained, "_CATALOGUE", {
        "test": {"file": bundle.name, "sha256": _sha256(bundle), "url": bundle.as_uri()}
    })

    with pytest.warns(UserWarning, match=r"trained with scikit-learn 0\.99\.0"):
        load_pretrained("test")


# ---------------------------------------------------------------------------
# No side effects on the package
# ---------------------------------------------------------------------------

def _package_tree() -> dict[str, int]:
    root = Path(gears.__file__).parent
    return {
        str(p.relative_to(root)): p.stat().st_mtime_ns
        for p in root.rglob("*")
        if "__pycache__" not in p.parts
    }


def test_nothing_is_written_inside_the_package_folder(release):
    before = _package_tree()

    load_pretrained("test")

    assert _package_tree() == before

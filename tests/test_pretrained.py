"""
Tests for ``gears.pretrained``.

No test touches the network or the real cache: the "release" is a ``file://`` URL and
``GEARS_HOME`` points to a temporary folder (see ``conftest.py``, which also makes any
internet connection fail).
"""

from __future__ import annotations

import hashlib
import http.client
import os
import re
import socket
import urllib.error
import urllib.request
import warnings
from pathlib import Path

import joblib
import pytest
import sklearn.base
from sklearn.exceptions import InconsistentVersionWarning

import gears
from gears import GEARSModel, pretrained
from gears.models.session_model import EVSessionModel
from gears.pretrained import PretrainedModelError, load_pretrained

EMPTY_FILE_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _catalogue_for(file: Path) -> dict:
    return {"test": {"file": file.name, "sha256": _sha256(file), "url": file.as_uri()}}


def _package_tree() -> dict[str, int]:
    root = Path(gears.__file__).parent
    return {
        str(p.relative_to(root)): p.stat().st_mtime_ns
        for p in root.rglob("*")
        if "__pycache__" not in p.parts
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_model(demo):
    """A real, small, fitted EVSessionModel."""
    return EVSessionModel(n_components=2, stratify_by=["season"]).fit(demo)


@pytest.fixture
def release(tmp_path, tiny_model, monkeypatch):
    """A fake release: one model file served through ``file://``."""
    remote = tmp_path / "remote" / "test_model.joblib"
    remote.parent.mkdir()
    joblib.dump(tiny_model, remote)
    monkeypatch.setattr(pretrained, "_CATALOGUE", _catalogue_for(remote))
    return remote


@pytest.fixture
def cache(tmp_path) -> Path:
    """Where ``GEARS_HOME`` points to (created lazily by the code under test)."""
    return tmp_path / "gears_home"


@pytest.fixture
def no_download(monkeypatch):
    """Any attempt to open a URL, even a file:// one, fails the test."""
    def _forbidden(*args, **kwargs):
        raise AssertionError("no download was expected here")

    monkeypatch.setattr(urllib.request, "urlopen", _forbidden)


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


def _urlopen_cut_midway_first(monkeypatch, first_bytes):
    """urlopen whose first response delivers ``first_bytes`` and then drops the connection;
    the following calls really open the URL."""
    real_urlopen = urllib.request.urlopen
    calls = []

    class CutResponse:
        def __init__(self):
            self.chunks = [first_bytes]

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def read(self, size=-1):
            if self.chunks:
                return self.chunks.pop()
            raise http.client.IncompleteRead(first_bytes, 1000)

    def cut(url, *args, **kwargs):
        calls.append(url)
        return CutResponse() if len(calls) == 1 else real_urlopen(url, *args, **kwargs)

    monkeypatch.setattr(urllib.request, "urlopen", cut)
    return calls


# ---------------------------------------------------------------------------
# The shipped catalogue and the safety net of the test suite itself
# ---------------------------------------------------------------------------

def test_the_shipped_catalogue_points_to_a_real_checksummed_release_and_unknown_ids_list_it():
    entry = pretrained._CATALOGUE["french"]
    assert re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])
    assert entry["sha256"] != EMPTY_FILE_SHA256          # the placeholder before the first release
    assert entry["url"] == (
        f"https://github.com/{pretrained.REPO}/releases/download/"
        f"{pretrained.RELEASE_TAG}/{entry['file']}"
    )
    with pytest.raises(PretrainedModelError, match="Unknown model ID 'atlantis'.*french"):
        load_pretrained("atlantis")                          # the message lists what exists


def test_the_tests_cannot_reach_the_internet_nor_the_real_cache(tmp_path):
    with pytest.raises(RuntimeError, match="network"):
        socket.create_connection(("example.com", 80))
    with pytest.raises(RuntimeError, match="network"):
        socket.getaddrinfo("example.com", 80)                    # not even a DNS lookup
    with socket.socket(socket.AF_INET) as sock:
        with pytest.raises(RuntimeError, match="network"):
            sock.connect(("93.184.216.34", 80))
        with pytest.raises(RuntimeError, match="network"):
            sock.connect_ex(("93.184.216.34", 80))

    left, right = socket.socketpair()                            # local sockets keep working
    with left, right:
        left.sendall(b"x")
        assert right.recv(1) == b"x"

    assert Path(os.environ["GEARS_HOME"]).parent == tmp_path     # never ~/.cache/gears
    assert "GEARS_OFFLINE" not in os.environ


# ---------------------------------------------------------------------------
# Download, cache, local file
# ---------------------------------------------------------------------------

def test_a_model_is_downloaded_verified_cached_and_then_reused(release, cache, tmp_path, monkeypatch):
    package_before = _package_tree()

    model = load_pretrained("test")

    assert isinstance(model, EVSessionModel) and model.is_fitted_
    assert (cache / "test_model.joblib").read_bytes() == release.read_bytes()
    assert list(cache.iterdir()) == [cache / "test_model.joblib"]       # no temporary file left
    assert _package_tree() == package_before                            # nothing in the package

    with monkeypatch.context() as m:
        m.setattr(urllib.request, "urlopen", lambda *a, **k: pytest.fail("network used"))
        assert isinstance(load_pretrained("test"), EVSessionModel)      # second call: the cache

    elsewhere = tmp_path / "other_cache"                                # cache_dir beats GEARS_HOME
    load_pretrained("test", cache_dir=elsewhere)
    assert (elsewhere / "test_model.joblib").exists()


def test_the_default_cache_is_under_the_user_home(monkeypatch, tmp_path):
    monkeypatch.delenv("GEARS_HOME")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    assert pretrained._get_cache_dir() == tmp_path / ".cache" / "gears"


@pytest.mark.usefixtures("no_download")
def test_a_local_file_is_loaded_without_cache_or_download_but_is_still_verified(
    release, cache, tmp_path
):
    assert isinstance(load_pretrained("test", path=release), EVSessionModel)
    assert not cache.exists()

    impostor = tmp_path / "impostor.joblib"
    impostor.write_bytes(b"not the model")
    with pytest.raises(PretrainedModelError, match="Integrity check failed"):
        load_pretrained("test", path=impostor)
    with pytest.raises(PretrainedModelError, match="No model file"):
        load_pretrained("test", path=tmp_path / "nope.joblib")


def test_from_pretrained_gives_a_facade_that_is_ready_to_simulate(release):
    model = GEARSModel.from_pretrained("test", forecaster_method="probabilistic")

    assert model.is_fitted_
    assert len(model.simulate_short_term("2025-06-10", horizon=1, n_sessions=4, seed=0)) == 4


# ---------------------------------------------------------------------------
# Integrity
# ---------------------------------------------------------------------------

def test_a_file_that_cannot_be_trusted_is_a_clean_error_and_leaves_nothing_behind(
    release, cache, tmp_path, monkeypatch
):
    # downloaded, but not the expected bytes: refused, not even a .part file kept
    monkeypatch.setitem(pretrained._CATALOGUE["test"], "sha256", "0" * 64)
    with pytest.raises(PretrainedModelError, match="does not match the expected SHA-256"):
        load_pretrained("test")
    assert not cache.exists() or list(cache.iterdir()) == []
    monkeypatch.setitem(pretrained._CATALOGUE["test"], "sha256", _sha256(release))

    # already cached but corrupted: refused, and the file is left for the user to inspect
    cache.mkdir(exist_ok=True)
    (cache / "test_model.joblib").write_bytes(b"not the model")
    with monkeypatch.context() as m:
        m.setattr(urllib.request, "urlopen", lambda *a, **k: pytest.fail("network used"))
        with pytest.raises(PretrainedModelError, match="Integrity check failed"):
            load_pretrained("test")
    assert (cache / "test_model.joblib").read_bytes() == b"not the model"

    # the right checksum, but not an EVSessionModel / not even a readable file
    for content, message in (
        ({"not": "a model"}, "does not contain an EVSessionModel"),
        (None, "Could not load model 'test'"),
    ):
        bad = tmp_path / "bad.joblib"
        if content is None:
            bad.write_bytes(b"\x00garbage, but with a known checksum")
        else:
            joblib.dump(content, bad)
        monkeypatch.setattr(pretrained, "_CATALOGUE", _catalogue_for(bad))
        with pytest.raises(PretrainedModelError, match=message):
            load_pretrained("test", cache_dir=tmp_path / f"cache_{message[:4]}")


# ---------------------------------------------------------------------------
# Offline mode and download failures
# ---------------------------------------------------------------------------

def test_offline_mode_uses_the_cache_and_otherwise_says_where_to_put_the_file(
    release, cache, monkeypatch
):
    monkeypatch.setenv("GEARS_OFFLINE", "1")
    with monkeypatch.context() as m:
        m.setattr(urllib.request, "urlopen", lambda *a, **k: pytest.fail("network used"))
        with pytest.raises(PretrainedModelError) as excinfo:
            load_pretrained("test")
    message = str(excinfo.value)
    assert "GEARS_OFFLINE=1" in message
    assert str(cache / "test_model.joblib") in message           # where to put the file
    assert release.as_uri() in message and "path=" in message    # where to get it, how to use it
    assert not cache.exists()

    for value in ("0", "", "true"):                              # only "1" forbids the download
        monkeypatch.setenv("GEARS_OFFLINE", value)
        assert isinstance(load_pretrained("test"), EVSessionModel), value

    monkeypatch.setenv("GEARS_OFFLINE", "1")                     # now it is cached
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: pytest.fail("network used"))
    assert isinstance(load_pretrained("test"), EVSessionModel)


def test_a_failed_download_is_retried_once_then_reported_with_the_way_out(
    release, cache, monkeypatch
):
    calls = _urlopen_failing_first(monkeypatch, 1, urllib.error.URLError("flaky wifi"))
    assert isinstance(load_pretrained("test"), EVSessionModel)
    assert len(calls) == 2                                       # one failure, one success
    (cache / "test_model.joblib").unlink()

    # a transfer cut midway: what was received is thrown away, the retry starts from scratch
    calls = _urlopen_cut_midway_first(monkeypatch, release.read_bytes()[:50])
    assert isinstance(load_pretrained("test"), EVSessionModel)
    assert len(calls) == 2
    assert (cache / "test_model.joblib").read_bytes() == release.read_bytes()
    (cache / "test_model.joblib").unlink()

    calls = _urlopen_failing_first(monkeypatch, 99, urllib.error.URLError("no route to host"))
    with pytest.raises(PretrainedModelError) as excinfo:
        load_pretrained("test")
    message = str(excinfo.value)
    assert len(calls) == 2                                       # two attempts, as in the plan
    assert "no route to host" in message and release.as_uri() in message and "path=" in message
    assert list(cache.iterdir()) == []

    _urlopen_failing_first(monkeypatch, 99, http.client.IncompleteRead(b"abc", 100))
    with pytest.raises(PretrainedModelError, match="IncompleteRead"):
        load_pretrained("test")
    assert list(cache.iterdir()) == []


# ---------------------------------------------------------------------------
# Library versions
# ---------------------------------------------------------------------------

def test_a_model_remembers_the_versions_it_was_fitted_with_and_loads_silently_with_them(
    tiny_model, release
):
    assert tiny_model.metadata_["sklearn_version"] == sklearn.__version__
    assert tiny_model.metadata_["gears_version"] == gears.__version__

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        load_pretrained("test")
    assert not [w for w in caught if "scikit-learn" in str(w.message)]


def test_a_model_trained_with_another_scikit_learn_gives_one_clear_warning(
    tmp_path, tiny_model, monkeypatch
):
    # pickled as if by scikit-learn 1.0.0 ...
    old_bundle = tmp_path / "old.joblib"
    with monkeypatch.context() as m:
        m.setattr(sklearn.base, "__version__", "1.0.0")
        joblib.dump(tiny_model, old_bundle)
    monkeypatch.setattr(pretrained, "_CATALOGUE", _catalogue_for(old_bundle))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert isinstance(load_pretrained("test"), EVSessionModel)

    ours = [w for w in caught if "was trained with scikit-learn" in str(w.message)]
    assert len(ours) == 1 and issubclass(ours[0].category, UserWarning)
    assert f"scikit-learn {sklearn.__version__} is installed" in str(ours[0].message)
    assert "scikit-learn==1.0.0" in str(ours[0].message)
    # scikit-learn's own per-estimator warnings are replaced by that single one
    assert not [w for w in caught if issubclass(w.category, InconsistentVersionWarning)]

    # ... or recorded as such in the model's metadata
    recorded = tmp_path / "meta.joblib"
    tiny_model.metadata_["sklearn_version"] = "0.99.0"
    try:
        joblib.dump(tiny_model, recorded)
    finally:
        tiny_model.metadata_["sklearn_version"] = sklearn.__version__
    monkeypatch.setattr(pretrained, "_CATALOGUE", _catalogue_for(recorded))
    with pytest.warns(UserWarning, match=r"trained with scikit-learn 0\.99\.0"):
        load_pretrained("test", cache_dir=tmp_path / "cache_meta")

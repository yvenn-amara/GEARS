"""
Shared pytest configuration and fixtures for the GEARS tests.

Rules every test file relies on:

* **No network.**  Any attempt to open an internet socket raises
  :class:`NetworkAccessError` (the ``_no_network`` fixture below, applied to every test).
  ``file://`` URLs, pipes and unix sockets keep working.
* **No writes outside ``tmp_path``.**  ``GEARS_HOME`` points to a throw-away folder, so
  ``load_pretrained`` can never touch ``~/.cache/gears``.
* **Headless matplotlib** (``MPLBACKEND=Agg``).
* **Shared fixtures are read-only.**  The session-scoped ``demo`` data and ``model`` are
  built once; a test that needs to modify a model builds its own.
"""

from __future__ import annotations

import hashlib
import os
import socket
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")  # before anything imports matplotlib

import joblib
import pandas as pd
import pytest

from gears import GEARSModel, pretrained
from gears.data.loader import make_demo_data


class NetworkAccessError(RuntimeError):
    """Raised when a test tries to reach the internet."""


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Make any internet connection, and any DNS lookup, fail loudly."""

    def blocked(*args, **kwargs):
        raise NetworkAccessError("tests must not use the network")

    def guard(real_method):
        def guarded(self, *args, **kwargs):
            if self.family in (socket.AF_INET, socket.AF_INET6):
                blocked()
            return real_method(self, *args, **kwargs)

        return guarded

    monkeypatch.setattr(socket.socket, "connect", guard(socket.socket.connect))
    monkeypatch.setattr(socket.socket, "connect_ex", guard(socket.socket.connect_ex))
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)


@pytest.fixture(autouse=True)
def _private_gears_home(tmp_path, monkeypatch):
    """Keep every test away from the real model cache and out of offline mode."""
    monkeypatch.setenv("GEARS_HOME", str(tmp_path / "gears_home"))
    monkeypatch.delenv("GEARS_OFFLINE", raising=False)


# ---------------------------------------------------------------------------
# Shared data and model (read-only)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def demo() -> pd.DataFrame:
    """1 500 demo sessions of a workplace: enough for every (weekday, season) context."""
    return make_demo_data(n=1500, location_type="work", seed=0)


@pytest.fixture(scope="session")
def model(demo) -> GEARSModel:
    """A fitted model.  'probabilistic' session counts keep the fit fast."""
    return GEARSModel(
        n_components=3, forecaster_method="probabilistic", n_scenarios=3, random_state=0
    ).fit(demo, verbose=False)


# ---------------------------------------------------------------------------
# A stand-in for the official French model (no download)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def french_like_model_file(tmp_path_factory) -> Path:
    """A small session model stratified like the French one (site type, département, weekday,
    season), with Paris workplaces only: the same code runs on it as on the real model."""
    data = make_demo_data(n=600, location_type="work", seed=0).assign(department="75")
    stand_in = GEARSModel(
        n_components=2, forecaster_method="probabilistic", random_state=0,
        stratify_by=["location_type", "department", "day_of_week", "season"],
    ).fit(data, verbose=False)
    path = tmp_path_factory.mktemp("release") / "french.joblib"
    joblib.dump(stand_in.gmm_, path)
    return path


@pytest.fixture
def french_release(french_like_model_file, monkeypatch) -> Path:
    """``load_pretrained("french")`` is served from ``french_like_model_file`` (a ``file://``
    URL with its real SHA-256), the way a download would be."""
    monkeypatch.setattr(pretrained, "_CATALOGUE", {"french": {
        "file": french_like_model_file.name,
        "url": french_like_model_file.as_uri(),
        "sha256": hashlib.sha256(french_like_model_file.read_bytes()).hexdigest(),
    }})
    return french_like_model_file

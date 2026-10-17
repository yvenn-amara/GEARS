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

import os
import socket

os.environ.setdefault("MPLBACKEND", "Agg")  # before anything imports matplotlib

import pandas as pd
import pytest

from gears import GEARSModel
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

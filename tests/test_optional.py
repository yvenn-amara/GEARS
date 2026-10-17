"""The "optional dependency is missing" error is the same everywhere."""

from __future__ import annotations

import json
import sys

import pytest

from gears._optional import require


def test_require_returns_an_installed_module():
    assert require("json") is json


def test_a_missing_dependency_says_which_extra_installs_it(monkeypatch):
    monkeypatch.setitem(sys.modules, "some_optional_pkg", None)  # simulate "not installed"

    with pytest.raises(ImportError) as excinfo:
        require("some_optional_pkg", extra="dl", feature="the demo feature")
    message = str(excinfo.value)
    assert "'some_optional_pkg'" in message and "the demo feature" in message
    assert 'pip install "gears-ev[dl]"' in message
    assert isinstance(excinfo.value.__cause__, ModuleNotFoundError)   # original error kept

    with pytest.raises(ImportError, match="pip install some_optional_pkg"):
        require("some_optional_pkg")                                   # no extra: the package

    from gears.models.vae import _require_torch  # the VAE has its own entry

    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ImportError, match=r'pip install "gears-ev\[vae\]"'):
        _require_torch()

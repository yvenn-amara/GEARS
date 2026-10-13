"""The "optional dependency is missing" error is the same everywhere."""
import sys

import pytest

from gears._optional import require


def test_require_returns_an_installed_module():
    import json

    assert require("json") is json


def test_require_missing_module_with_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "some_optional_pkg", None)  # simulate "not installed"

    with pytest.raises(ImportError) as excinfo:
        require("some_optional_pkg", extra="dl", feature="the demo feature")

    message = str(excinfo.value)
    assert "'some_optional_pkg'" in message
    assert "the demo feature" in message
    assert 'pip install "gears-ev[dl]"' in message
    assert isinstance(excinfo.value.__cause__, ModuleNotFoundError)  # original error kept


def test_require_missing_module_without_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "some_optional_pkg", None)

    with pytest.raises(ImportError, match="pip install some_optional_pkg"):
        require("some_optional_pkg")


def test_vae_without_torch_points_to_the_vae_extra(monkeypatch):
    from gears.models.vae import _require_torch

    monkeypatch.setitem(sys.modules, "torch", None)

    with pytest.raises(ImportError, match=r'pip install "gears-ev\[vae\]"'):
        _require_torch()

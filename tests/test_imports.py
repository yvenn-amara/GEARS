"""Importing GEARS must stay light: no heavy optional dependency, no side effects."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import gears
import gears.models

HEAVY = ["torch", "neuralforecast", "pmdarima", "statsmodels", "matplotlib", "requests"]
REPO_ROOT = Path(gears.__file__).resolve().parent.parent


def _run(statement: str, *, cwd: Path | None = None, home: Path | None = None) -> str:
    """Run ``statement`` in a fresh interpreter; return the last line it printed."""
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT), PYTHONDONTWRITEBYTECODE="1")
    env.pop("GEARS_HOME", None)
    env.pop("GEARS_OFFLINE", None)
    if home is not None:
        env["HOME"] = env["USERPROFILE"] = str(home)
    out = subprocess.run(
        [sys.executable, "-c", statement],
        capture_output=True, text=True, check=True, cwd=cwd, env=env,
    )
    return out.stdout.strip().splitlines()[-1]


def _loaded_heavy_modules(statement: str) -> list[str]:
    """Which of ``HEAVY`` does ``statement`` load, in a fresh interpreter?"""
    code = (
        f"import sys, json\n{statement}\n"
        f"print(json.dumps([m for m in {HEAVY!r} if m in sys.modules]))"
    )
    return json.loads(_run(code))


def test_import_gears_is_lightweight():
    assert _loaded_heavy_modules("import gears") == []


def test_import_session_model_layer_is_lightweight():
    assert _loaded_heavy_modules("from gears.models import EVSessionModel") == []


def test_importing_the_deep_learning_forecasters_does_not_load_torch():
    # They are importable without the [dl] extra; torch only matters at fit() time.
    assert _loaded_heavy_modules(
        "from gears.models import NHiTSForecaster, TransformerForecaster"
    ) == []


def test_import_gears_does_not_import_the_submodules():
    loaded = json.loads(_run(
        "import sys, json; import gears; "
        "print(json.dumps(sorted(m for m in sys.modules if m.startswith('gears.'))))"
    ))
    assert loaded == []


def _package_tree() -> dict[str, int]:
    root = Path(gears.__file__).parent
    return {
        str(p.relative_to(root)): p.stat().st_mtime_ns
        for p in root.rglob("*")
        if "__pycache__" not in p.parts
    }


def test_import_touches_neither_the_network_nor_the_disk(tmp_path):
    """Sockets are disabled, HOME and the working directory are empty folders."""
    block_network = (
        "import socket\n"
        "def _no(*a, **k):\n"
        "    raise RuntimeError('network access during import')\n"
        "socket.socket.connect = socket.socket.connect_ex = _no\n"
        "socket.create_connection = socket.getaddrinfo = _no\n"
    )
    imports = (
        "import gears\n"
        "from gears import GEARSModel, load_pretrained, EVSessionModel, plotting, utils\n"
        "from gears.models import EVSessionModel, SessionForecaster, NHiTSForecaster\n"
        "from gears.simulation import ShortTermSimulator, MediumTermSimulator\n"
        "from gears.smart_charging import SmartChargingOptimizer\n"
        "import gears.pretrained, gears.cli\n"
        "print('imported')\n"
    )
    before = _package_tree()

    assert _run(block_network + imports, cwd=tmp_path, home=tmp_path) == "imported"

    assert list(tmp_path.iterdir()) == []  # nothing in HOME (no ~/.cache/gears) or in the cwd
    assert _package_tree() == before        # nothing inside the package folder


def test_all_matches_the_lazy_export_tables():
    assert set(gears.__all__) == set(gears._EXPORTS) | set(gears._SUBMODULES)
    assert set(gears.models.__all__) == set(gears.models._EXPORTS)


def test_all_is_short_and_every_name_resolves():
    assert len(gears.__all__) <= 25
    assert len(set(gears.__all__)) == len(gears.__all__)
    for name in gears.__all__:
        assert getattr(gears, name) is not None, name
    assert set(gears.__all__) <= set(dir(gears))


def test_unknown_attribute_raises_attribute_error():
    with pytest.raises(AttributeError, match="no attribute 'nope'"):
        gears.nope  # noqa: B018
    with pytest.raises(AttributeError, match="no attribute 'nope'"):
        gears.models.nope  # noqa: B018


def test_models_exports_resolve_and_are_listed():
    expected = {
        "EVSessionModel", "SessionForecaster", "PersistenceForecaster",
        "NHiTSForecaster", "TransformerForecaster", "sessions_to_daily_counts",
    }
    assert set(gears.models.__all__) == expected
    assert expected <= set(dir(gears.models))
    for name in expected:
        assert getattr(gears.models, name) is not None, name


def test_lazy_names_are_the_real_objects():
    from gears.models.session_model import EVSessionModel
    from gears.pipeline import GEARSModel

    assert gears.GEARSModel is GEARSModel
    assert gears.EVSessionModel is EVSessionModel
    assert gears.models.EVSessionModel is EVSessionModel

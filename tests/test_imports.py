"""Importing GEARS must stay light: no heavy optional dependency, no side effects.

Each check runs in a fresh interpreter, because what matters is what *importing* loads.
"""

from __future__ import annotations

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


def _package_tree() -> dict[str, int]:
    root = Path(gears.__file__).parent
    return {
        str(p.relative_to(root)): p.stat().st_mtime_ns
        for p in root.rglob("*")
        if "__pycache__" not in p.parts
    }


def test_importing_gears_loads_no_heavy_dependency_and_no_submodule():
    assert _loaded_heavy_modules("import gears") == []
    assert _loaded_heavy_modules("from gears.models import EVSessionModel") == []
    # the deep-learning forecasters are importable without the [dl] extra: torch only
    # matters when fit() is called
    assert _loaded_heavy_modules(
        "from gears.models import NHiTSForecaster, TransformerForecaster"
    ) == []
    loaded = json.loads(_run(
        "import sys, json; import gears; "
        "print(json.dumps(sorted(m for m in sys.modules if m.startswith('gears.'))))"
    ))
    assert loaded == []


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

    assert list(tmp_path.iterdir()) == []   # nothing in HOME (no ~/.cache/gears) or the cwd
    assert _package_tree() == before        # nothing inside the package folder


def test_the_core_fits_and_simulates_without_torch_or_neuralforecast():
    """Not only importing: the whole fit -> simulate chain never needs the extras."""
    # An import hook, not ``sys.modules['torch'] = None``: scipy probes ``sys.modules`` for
    # torch and would crash on the placeholder before reaching any GEARS code.
    core_only = (
        "import sys\n"
        "from importlib.abc import MetaPathFinder\n"
        "class NotInstalled(MetaPathFinder):\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in ('torch', 'neuralforecast', 'lightning', 'pytorch_lightning'):\n"
        "            raise ModuleNotFoundError(f'No module named {name!r}', name=name)\n"
        "sys.meta_path.insert(0, NotInstalled())\n"
        "from gears import GEARSModel\n"
        "from gears.data.loader import make_demo_data\n"
        "model = GEARSModel(n_components=2, forecaster_method='probabilistic', n_scenarios=2)\n"
        "model.fit(make_demo_data(n=600, seed=0), verbose=False)\n"
        "print(len(model.simulate_short_term('2025-06-10', horizon=2, n_scenarios=2, seed=0)))\n"
    )
    assert int(_run(core_only)) > 0


def test_the_public_names_are_listed_resolvable_and_the_real_objects():
    assert set(gears.__all__) == set(gears._EXPORTS) | set(gears._SUBMODULES)
    assert len(gears.__all__) <= 25 and len(set(gears.__all__)) == len(gears.__all__)
    assert set(gears.__all__) <= set(dir(gears))
    for name in gears.__all__:
        assert getattr(gears, name) is not None, name

    models = {
        "EVSessionModel", "SessionForecaster", "PersistenceForecaster",
        "NHiTSForecaster", "TransformerForecaster", "sessions_to_daily_counts",
    }
    assert set(gears.models.__all__) == set(gears.models._EXPORTS) == models
    assert models <= set(dir(gears.models))

    from gears.models.session_model import EVSessionModel
    from gears.pipeline import GEARSModel

    assert gears.GEARSModel is GEARSModel
    assert gears.EVSessionModel is gears.models.EVSessionModel is EVSessionModel

    for module in (gears, gears.models):
        with pytest.raises(AttributeError, match="no attribute 'nope'"):
            module.nope  # noqa: B018

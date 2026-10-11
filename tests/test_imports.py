"""Importing GEARS must stay light: no heavy optional dependency, no side effects."""

import json
import subprocess
import sys

HEAVY = ["torch", "neuralforecast", "pmdarima", "statsmodels", "matplotlib", "requests"]


def _loaded_heavy_modules(statement: str) -> list[str]:
    """Run ``statement`` in a fresh interpreter and report which heavy modules got loaded."""
    code = (
        f"import sys, json\n{statement}\n"
        f"print(json.dumps([m for m in {HEAVY!r} if m in sys.modules]))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_import_gears_is_lightweight():
    assert _loaded_heavy_modules("import gears") == []


def test_import_session_model_layer_is_lightweight():
    assert _loaded_heavy_modules("from gears.models import EVSessionModel") == []

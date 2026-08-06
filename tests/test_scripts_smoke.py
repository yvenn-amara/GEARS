"""Import/argument-parsing smoke tests for every file directly under scripts/.

Phase 2 / Session 9: closes the exact blind spot AUDIT.md §c/§h and the
Session 9 brief both flag — no test previously imported anything under
scripts/ except fit_session_model.py (covered separately in
test_fit_session_model_script.py, which predates this file and already goes
further than a plain smoke test for that one script). A syntax or top-level
import error in any of these files would previously be invisible to both
ruff and pytest, and only discovered by a user actually running the script
(this exact failure mode already happened once for fit_gmm.py before its
Session 2 rename — see AUDIT.md).

These run each script as a real subprocess with --help, which is a stronger
check than a bare `import` would be: --help exercises the full module body
(including argparse construction) without requiring real data files or
network access, so a broken top-level import, a malformed argparse spec, or
a bad default value all fail loudly here instead of at first real use.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"

# fit_session_model.py already has its own dedicated, more thorough test file
# (test_fit_session_model_script.py) — no need to duplicate its --help check
# here, but it's included in the parametrization for completeness/symmetry
# so that "every scripts/*.py file has at least one smoke test in this file
# OR a more specific one elsewhere" is true by inspection of this one list.
ALL_SCRIPT_FILES = sorted(SCRIPTS_DIR.glob("*.py"))


def _run_help(script_path: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(script_path), "--help"],
        capture_output=True, text=True, cwd=REPO_ROOT, check=False, timeout=60,
    )


def test_scripts_directory_is_not_empty():
    """Guard against this test file silently covering nothing if scripts/
    is ever restructured (e.g. moved to a package)."""
    assert len(ALL_SCRIPT_FILES) >= 3


@pytest.mark.parametrize(
    "script_path", ALL_SCRIPT_FILES, ids=[p.name for p in ALL_SCRIPT_FILES]
)
def test_script_help_parses_without_import_or_syntax_error(script_path):
    result = _run_help(script_path)
    assert result.returncode == 0, (
        f"{script_path.name} --help failed (syntax/import error not caught "
        f"by ruff or any other test):\n{result.stderr}"
    )
    assert "usage:" in result.stdout.lower()


def test_prepare_hf_bundles_help_lists_demo_flag():
    result = _run_help(SCRIPTS_DIR / "prepare_hf_bundles.py")
    assert "--demo" in result.stdout
    assert "--upload" in result.stdout


def test_validate_recency_bias_help_lists_required_data_flag():
    result = _run_help(SCRIPTS_DIR / "validate_recency_bias.py")
    assert "--data" in result.stdout


def test_validate_vae_competitiveness_help_lists_required_flags():
    result = _run_help(SCRIPTS_DIR / "validate_vae_competitiveness.py")
    assert "--data" in result.stdout
    assert "--dataset-name" in result.stdout

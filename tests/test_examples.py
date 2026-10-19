"""Every script in ``examples/`` runs end to end, offline, without a warning, and writes the
files it announces. A broken example breaks this test.

The scripts run in this process with ``runpy``, in a temporary folder (``GEARS_EXAMPLES_OUT``).
They take a few seconds each, so the module is marked ``slow``: ``pytest -m slow`` runs it.
"""

from __future__ import annotations

import runpy
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from gears import GEARSModel

pytestmark = pytest.mark.slow

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@pytest.fixture
def out(tmp_path, monkeypatch) -> Path:
    """The folder the examples write to; the working directory is empty and elsewhere."""
    folder = tmp_path / "out"
    monkeypatch.setenv("GEARS_EXAMPLES_OUT", str(folder))
    monkeypatch.chdir(tmp_path)
    yield folder
    plt.close("all")


@pytest.fixture
def run_example(out, monkeypatch):
    """``run_example(name, *args)`` runs ``examples/<name>`` as a script with these command-line
    arguments and returns the user-facing warnings it raised."""

    def run(name: str, *args: str) -> list[warnings.WarningMessage]:
        monkeypatch.setattr(sys, "argv", [name, *args])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            runpy.run_path(str(EXAMPLES / name), run_name="__main__")
        return [w for w in caught if issubclass(w.category, UserWarning)]

    return run


def assert_is_a_figure(path: Path) -> None:
    assert path.read_bytes()[:8] == PNG_SIGNATURE and path.stat().st_size > 20_000


def test_the_quickstart_simulates_a_week_and_draws_it(run_example, out, capsys):
    assert run_example("01_quickstart.py") == []

    assert_is_a_figure(out / "01_quickstart.png")
    printed = capsys.readouterr().out
    assert "fitted" in printed and "simulated sessions in 10 scenarios" in printed


def test_forecasters_are_compared_and_the_regret_of_a_forecast_is_computed(run_example, out, capsys):
    assert run_example("02_forecast_and_smart_charging.py") == []

    assert_is_a_figure(out / "02_forecast_and_smart_charging.png")
    printed = capsys.readouterr().out
    for name in ("probabilistic", "persistence", "SARIMA"):
        assert name in printed
    assert printed.count("RMSE") == 3
    assert "V1G on" in printed and "'regret_smart_vs_oracle'" in printed


def test_the_three_growth_models_and_the_charger_presets_are_compared(run_example, out, capsys):
    assert run_example("03_medium_term_scenarios.py") == []

    assert_is_a_figure(out / "03_medium_term_scenarios.png")
    printed = capsys.readouterr().out
    for growth_model in ("linear", "s_curve", "bass"):
        assert printed.count(growth_model) >= 2          # the growth curve and the simulation
    for preset in ("current_fr", "fast_dominant", "slow_residential", "workplace", "public_mixed"):
        assert preset in printed


def test_a_file_goes_through_load_fit_save_load_and_simulate(run_example, out, capsys):
    assert run_example("04_bring_your_own_data.py") == []

    written = pd.read_csv(out / "my_sessions.csv")
    assert list(written.columns) == ["start_time", "duration_h", "energy_kwh", "site_type"]
    assert GEARSModel.load(out / "my_model.joblib").is_fitted_
    simulated = pd.read_csv(out / "my_simulated_sessions.csv")
    assert len(simulated) > 0 and {"arrival_time", "duration", "energy"} <= set(simulated.columns)
    assert "from 2024-01-01" in capsys.readouterr().out      # the day after the last session

    # your own file: the same script, a path on the command line (here the file it just wrote)
    assert run_example("04_bring_your_own_data.py", str(out / "my_sessions.csv")) == []
    assert f"sessions file: {out / 'my_sessions.csv'}" in capsys.readouterr().out


@pytest.mark.usefixtures("french_release")
def test_the_french_model_is_used_on_request_and_only_then(run_example, capsys):
    run_example("04_bring_your_own_data.py")
    assert "French model" not in capsys.readouterr().out      # no download without the flag

    run_example("04_bring_your_own_data.py", "--pretrained")
    assert "French model, workplaces of Paris: 40 sessions on 2025-06-02" in capsys.readouterr().out

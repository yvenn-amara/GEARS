"""Tests for the command line: ``gears fit`` and ``gears simulate``."""

from __future__ import annotations

import pandas as pd
import pytest
from click.testing import CliRunner

from gears.cli import main
from gears.data.loader import make_demo_data
from gears.pipeline import GEARSModel

SMALL_FIT = ["--n-components", "2", "--scenarios", "2", "--quiet"]


@pytest.fixture(scope="module")
def sessions_csv(tmp_path_factory):
    path = tmp_path_factory.mktemp("cli") / "sessions.csv"
    make_demo_data(n=400, location_type="work", seed=0).to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def model_path(sessions_csv, tmp_path_factory):
    path = tmp_path_factory.mktemp("cli_model") / "model.joblib"
    result = CliRunner().invoke(main, ["fit", str(sessions_csv), "-o", str(path), *SMALL_FIT])
    assert result.exit_code == 0, result.output
    return path


def test_fit_saves_a_model_and_forwards_its_options(sessions_csv, model_path, tmp_path):
    default = GEARSModel.load(model_path)
    assert default.is_fitted_ and default.gear == 1 and default.model_type == "gmm"
    assert default.recency is False and default.random_state == 42
    assert default.stratify_by == ["day_of_week", "season"]      # no option: the library default

    out = tmp_path / "custom.joblib"
    result = CliRunner().invoke(main, [
        "fit", str(sessions_csv), "-o", str(out), *SMALL_FIT,
        "--stratify-by", "day_of_week, season,", "--max-components", "4",
        "--max-samples-per-context", "150", "--seed", "7",
        "--recency", "--half-life-days", "30",
    ])
    assert result.exit_code == 0, result.output

    custom = GEARSModel.load(out)
    assert custom.stratify_by == ["day_of_week", "season"]   # spaces and a trailing comma are fine
    assert (custom.max_components, custom.max_samples_per_context) == (4, 150)
    assert custom.random_state == 7 and custom.recency is True and custom.half_life_days == 30.0
    assert custom.gmm_.recency is True and custom.gmm_.half_life_days == 30.0   # down to the GMM

    # the same seed on the same data gives the same fitted model, hence the same simulation
    again = tmp_path / "again.joblib"
    result = CliRunner().invoke(main, [
        "fit", str(sessions_csv), "-o", str(again), *SMALL_FIT, "--seed", "7",
        "--stratify-by", "day_of_week, season,", "--max-components", "4",
        "--max-samples-per-context", "150", "--recency", "--half-life-days", "30",
    ])
    assert result.exit_code == 0, result.output
    kwargs = {"horizon": 2, "n_scenarios": 1, "n_sessions": 5, "seed": 0}
    pd.testing.assert_frame_equal(
        custom.simulate_short_term("2024-06-01", **kwargs),
        GEARSModel.load(again).simulate_short_term("2024-06-01", **kwargs),
    )

    # an option made of separators only names no column: refused before anything is fitted
    nothing = tmp_path / "nothing.joblib"
    result = CliRunner().invoke(
        main, ["fit", str(sessions_csv), "-o", str(nothing), *SMALL_FIT, "--stratify-by", " , "]
    )
    assert result.exit_code == 2 and "expected at least one column name" in result.output
    assert not nothing.exists()


def test_removed_options_and_commands_are_really_gone(sessions_csv):
    runner = CliRunner()
    result = runner.invoke(main, ["fit", str(sessions_csv), "--gear", "2"])
    assert result.exit_code == 2 and "No such option" in result.output

    for command in ("medium-term", "smart-charge", "list-models"):
        result = runner.invoke(main, [command])
        assert result.exit_code == 2 and "No such command" in result.output, command


def test_simulate_writes_the_sessions_of_the_requested_days(model_path, tmp_path):
    runner = CliRunner()
    forecast = tmp_path / "forecast.csv"
    result = runner.invoke(main, [
        "simulate", "--model", str(model_path), "--start", "2024-06-01",
        "--horizon", "3", "--scenarios", "1", "--output", str(forecast), "--seed", "0",
    ])
    assert result.exit_code == 0, result.output
    assert len(pd.read_csv(forecast)) > 0

    fixed = tmp_path / "fixed.csv"                       # --n-sessions bypasses the forecaster
    result = runner.invoke(main, [
        "simulate", "--model", str(model_path), "--start", "2024-06-01",
        "--horizon", "2", "--n-sessions", "5", "--output", str(fixed),
    ])
    assert result.exit_code == 0, result.output
    assert len(pd.read_csv(fixed)) == 2 * 5


def test_simulate_needs_exactly_one_model_and_a_known_pretrained_id(model_path):
    runner = CliRunner()
    for args in (
        ["--start", "2024-06-01"],
        ["--model", str(model_path), "--pretrained", "french", "--start", "2024-06-01"],
    ):
        result = runner.invoke(main, ["simulate", *args])
        assert result.exit_code == 2
        assert "provide exactly one of --model or --pretrained" in result.output

    result = runner.invoke(main, ["simulate", "--pretrained", "atlantis", "--start", "2024-06-01"])
    assert result.exit_code == 1 and "Unknown model ID 'atlantis'" in result.output

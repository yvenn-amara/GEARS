"""Tests for the gears CLI (``gears fit`` and ``gears simulate``)."""
import pandas as pd
import pytest
from click.testing import CliRunner

from gears.cli import _split_columns, main
from gears.data.loader import make_demo_data
from gears.pipeline import GEARSModel


def _write_demo_csv(path, n=400, seed=0):
    df = make_demo_data(n=n, location_type="work", seed=seed)
    df.to_csv(path, index=False)
    return path


def _fit_model_via_cli(tmp_path, runner, *extra_args):
    """Fit a small model through the CLI and return its path."""
    data_path = _write_demo_csv(tmp_path / "sessions.csv")
    model_path = tmp_path / "model.joblib"
    result = runner.invoke(
        main,
        ["fit", str(data_path), "--output", str(model_path),
         "--n-components", "2", "--scenarios", "2", "--quiet", *extra_args],
    )
    assert result.exit_code == 0, result.output
    return model_path


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------

def test_fit_cli_default_roundtrip(tmp_path):
    model_path = _fit_model_via_cli(tmp_path, CliRunner())

    model = GEARSModel.load(model_path)
    assert model.gear == 1
    assert model.is_fitted_
    assert model.model_type == "gmm"
    assert model.recency is False
    assert model.random_state == 42


def test_fit_cli_forwards_options(tmp_path):
    model_path = _fit_model_via_cli(
        tmp_path, CliRunner(),
        "--stratify-by", "day_of_week, season",
        "--max-components", "4",
        "--max-samples-per-context", "150",
        "--seed", "7",
        "--recency", "--half-life-days", "30",
    )

    model = GEARSModel.load(model_path)
    assert model.stratify_by == ["day_of_week", "season"]
    assert model.max_components == 4
    assert model.max_samples_per_context == 150
    assert model.random_state == 7
    assert model.recency is True
    assert model.half_life_days == 30.0


def test_fit_cli_vae_model_type_roundtrip(tmp_path):
    pytest.importorskip("torch")
    model_path = _fit_model_via_cli(tmp_path, CliRunner(), "--model-type", "vae")
    assert GEARSModel.load(model_path).model_type == "vae"


def test_fit_cli_same_seed_gives_same_simulation(tmp_path):
    runner = CliRunner()
    out = []
    for name in ("a", "b"):
        folder = tmp_path / name
        folder.mkdir()
        model_path = _fit_model_via_cli(folder, runner, "--seed", "3")
        out.append(GEARSModel.load(model_path).simulate_short_term(
            "2024-06-01", horizon=2, n_scenarios=1, n_sessions=5, seed=0))
    pd.testing.assert_frame_equal(out[0], out[1])


def test_fit_cli_gear_option_is_gone(tmp_path):
    data_path = _write_demo_csv(tmp_path / "sessions.csv")
    result = CliRunner().invoke(main, ["fit", str(data_path), "--gear", "2"])
    assert result.exit_code == 2
    assert "No such option" in result.output


def test_removed_commands_are_gone():
    runner = CliRunner()
    for command in ("medium-term", "smart-charge", "list-models"):
        result = runner.invoke(main, [command])
        assert result.exit_code == 2, command
        assert "No such command" in result.output


def test_split_columns():
    assert _split_columns(None) is None
    assert _split_columns("a,b") == ["a", "b"]
    assert _split_columns(" a , b ,") == ["a", "b"]


# ---------------------------------------------------------------------------
# simulate
# ---------------------------------------------------------------------------

def test_simulate_cli_writes_sessions(tmp_path):
    runner = CliRunner()
    model_path = _fit_model_via_cli(tmp_path, runner)
    output_path = tmp_path / "sessions_out.csv"

    result = runner.invoke(
        main,
        ["simulate", "--model", str(model_path), "--start", "2024-06-01",
         "--horizon", "3", "--scenarios", "1", "--output", str(output_path)],
    )

    assert result.exit_code == 0, result.output
    assert len(pd.read_csv(output_path)) > 0


def test_simulate_cli_n_sessions_flag_bypasses_forecaster(tmp_path):
    runner = CliRunner()
    model_path = _fit_model_via_cli(tmp_path, runner)
    output_path = tmp_path / "sessions_fixed.csv"

    result = runner.invoke(
        main,
        ["simulate", "--model", str(model_path), "--start", "2024-06-01",
         "--horizon", "2", "--n-sessions", "5", "--output", str(output_path)],
    )

    assert result.exit_code == 0, result.output
    # 2 days * 5 sessions/day = 10 rows.
    assert len(pd.read_csv(output_path)) == 10


def test_simulate_cli_requires_model_or_pretrained():
    result = CliRunner().invoke(main, ["simulate", "--start", "2024-06-01"])
    assert result.exit_code == 2
    assert "provide exactly one of --model or --pretrained" in result.output


def test_simulate_cli_rejects_model_and_pretrained_together(tmp_path):
    runner = CliRunner()
    model_path = _fit_model_via_cli(tmp_path, runner)
    result = runner.invoke(
        main,
        ["simulate", "--model", str(model_path), "--pretrained", "french",
         "--start", "2024-06-01"],
    )
    assert result.exit_code == 2
    assert "provide exactly one of --model or --pretrained" in result.output


def test_simulate_cli_unknown_pretrained_id_is_a_clean_error():
    result = CliRunner().invoke(
        main, ["simulate", "--pretrained", "atlantis", "--start", "2024-06-01"]
    )
    assert result.exit_code == 1
    assert "Unknown model ID 'atlantis'" in result.output

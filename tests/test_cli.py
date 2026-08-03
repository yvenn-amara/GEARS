"""Tests for the gears CLI (gears.cli), focused on the Phase 2 / Session 3
--gear / --model-type / --recency / --half-life-days additions and the
pre-existing --growth-model 'bass' choice."""
import pandas as pd
from click.testing import CliRunner

from gears.cli import main
from gears.data.loader import make_demo_data
from gears.pipeline import GEARSModel


def _write_demo_csv(path, n=400, seed=0):
    df = make_demo_data(n=n, location_type="work", seed=seed)
    df.to_csv(path, index=False)
    return path


def test_fit_cli_default_gear_roundtrip(tmp_path):
    data_path = _write_demo_csv(tmp_path / "sessions.csv")
    output_path = tmp_path / "model.joblib"

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["fit", str(data_path), "--output", str(output_path),
         "--n-components", "2", "--scenarios", "2", "--quiet"],
    )

    assert result.exit_code == 0, result.output
    assert output_path.exists()

    model = GEARSModel.load(output_path)
    assert model.gear == 1
    assert model.is_fitted_
    assert model.model_type == "gmm"
    assert model.recency is False


def test_fit_cli_model_type_recency_half_life_days_roundtrip(tmp_path):
    data_path = _write_demo_csv(tmp_path / "sessions.csv")
    output_path = tmp_path / "model.joblib"

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["fit", str(data_path), "--output", str(output_path),
         "--n-components", "2", "--scenarios", "2",
         "--model-type", "vae", "--recency", "--half-life-days", "30",
         "--quiet"],
    )

    assert result.exit_code == 0, result.output

    model = GEARSModel.load(output_path)
    assert model.model_type == "vae"
    assert model.recency is True
    assert model.half_life_days == 30.0


def test_fit_cli_unimplemented_gear_errors(tmp_path):
    data_path = _write_demo_csv(tmp_path / "sessions.csv")
    output_path = tmp_path / "model.joblib"

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["fit", str(data_path), "--output", str(output_path), "--gear", "2"],
    )

    assert result.exit_code != 0
    assert isinstance(result.exception, NotImplementedError)
    assert "GEAR 1st" in str(result.exception)
    assert not output_path.exists()


def test_medium_term_cli_growth_model_bass_choice(tmp_path):
    data_path = _write_demo_csv(tmp_path / "sessions.csv")
    model_path = tmp_path / "model.joblib"
    output_path = tmp_path / "medium_term.csv"

    runner = CliRunner()
    fit_result = runner.invoke(
        main,
        ["fit", str(data_path), "--output", str(model_path),
         "--n-components", "2", "--scenarios", "2", "--quiet"],
    )
    assert fit_result.exit_code == 0, fit_result.output

    mt_result = runner.invoke(
        main,
        ["medium-term", "--model", str(model_path), "--years", "1",
         "--growth-model", "bass", "--scenarios", "2",
         "--output", str(output_path)],
    )

    assert mt_result.exit_code == 0, mt_result.output
    assert output_path.exists()
    result_df = pd.read_csv(output_path)
    assert len(result_df) > 0

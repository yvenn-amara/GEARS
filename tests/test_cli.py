"""Tests for the gears CLI (gears.cli).

Originally focused solely on the Phase 2 / Session 3 --gear / --model-type /
--recency / --half-life-days additions and the pre-existing --growth-model
'bass' choice. Phase 2 / Session 9 fills the remaining gap flagged in its own
acceptance criteria: `simulate`, `smart-charge`, and `list-models` had no
end-to-end CLI coverage at all before this session."""
import pandas as pd
from click.testing import CliRunner

from gears.cli import main
from gears.data.loader import make_demo_data
from gears.pipeline import GEARSModel
from gears.utils import make_price_signal


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


# ---------------------------------------------------------------------------
# simulate (Session 9: previously untested)
# ---------------------------------------------------------------------------

def _fit_model_via_cli(tmp_path, runner, **fit_kwargs):
    """Shared helper: fit a small model via the CLI and return its path."""
    data_path = _write_demo_csv(tmp_path / "sessions.csv")
    model_path = tmp_path / "model.joblib"
    args = ["fit", str(data_path), "--output", str(model_path),
            "--n-components", "2", "--scenarios", "2", "--quiet"]
    for k, v in fit_kwargs.items():
        args += [k, str(v)]
    result = runner.invoke(main, args)
    assert result.exit_code == 0, result.output
    return model_path


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
    assert output_path.exists()
    df = pd.read_csv(output_path)
    assert len(df) > 0


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
    df = pd.read_csv(output_path)
    # 2 days * 5 sessions/day = 10 rows.
    assert len(df) == 10


def test_simulate_cli_requires_model_or_pretrained(tmp_path):
    runner = CliRunner()
    result = runner.invoke(main, ["simulate", "--start", "2024-06-01"])
    assert result.exit_code != 0
    assert "provide --model or --pretrained" in result.output


# ---------------------------------------------------------------------------
# smart-charge (Session 9: previously untested)
# ---------------------------------------------------------------------------

def test_smart_charge_cli_writes_optimised_schedule(tmp_path):
    runner = CliRunner()
    model_path = _fit_model_via_cli(tmp_path, runner)

    # Sessions aligned with the signal's date range, matching the pattern
    # already established in tests/test_smart_charging.py.
    sessions = make_demo_data(n=15, seed=0, start_date="2025-06-01", end_date="2025-06-07")
    sessions_path = tmp_path / "sessions_for_smart_charge.csv"
    sessions.to_csv(sessions_path, index=False)

    signal = make_price_signal(start="2025-06-01", periods=7 * 48, resolution_min=30,
                                pattern="day_night")
    signal_path = tmp_path / "price_signal.csv"
    signal.to_csv(signal_path, header=True)

    output_path = tmp_path / "optimised.csv"
    result = runner.invoke(
        main,
        ["smart-charge", "--model", str(model_path),
         "--sessions", str(sessions_path), "--signal", str(signal_path),
         "--signal-type", "price", "--output", str(output_path)],
    )

    assert result.exit_code == 0, result.output
    assert output_path.exists()
    df = pd.read_csv(output_path)
    for col in ("cost_smart", "cost_plug", "savings_pct"):
        assert col in df.columns


# ---------------------------------------------------------------------------
# list-models (Session 9: previously untested)
# ---------------------------------------------------------------------------

def test_list_models_cli_prints_catalogue():
    runner = CliRunner()
    result = runner.invoke(main, ["list-models"])
    assert result.exit_code == 0, result.output
    # The catalogue is keyed by model_id; the printed table should at least
    # contain that column header and be non-empty.
    assert "model_id" in result.output
    assert len(result.output.strip().splitlines()) > 1

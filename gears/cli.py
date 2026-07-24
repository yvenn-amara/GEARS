"""
GEARS command-line interface.

Usage examples
--------------
    gears fit data/sessions.csv --output models/my_model.joblib
    gears simulate --model models/my_model.joblib --start 2024-06-01 --horizon 7
    gears simulate --pretrained work_fr_demo --start 2024-06-01 --output out/sessions.csv
    gears medium-term --model models/my_model.joblib --years 3 --growth 0.15
    gears list-models
"""

from __future__ import annotations

import sys
import logging

import click
import pandas as pd

logging.basicConfig(level=logging.WARNING)


@click.group()
@click.version_option(package_name="gears-ev")
def main():
    """GEARS – Generating Electric Vehicle Recharging Sessions."""
    pass


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------

@main.command()
@click.argument("data_path", type=click.Path(exists=True))
@click.option("--output", "-o", default="gears_model.joblib", show_default=True,
              help="Path to save the fitted model.")
@click.option("--n-components", default="auto", show_default=True,
              help="Number of GMM components ('auto' or integer).")
@click.option("--forecaster", default="probabilistic", show_default=True,
              type=click.Choice(["sarima", "probabilistic"]),
              help="Session-count forecasting method.")
@click.option("--scenarios", default=10, show_default=True,
              help="Default number of stochastic scenarios.")
@click.option("--verbose/--quiet", default=True)
def fit(data_path, output, n_components, forecaster, scenarios, verbose):
    """Fit a GEARSModel on DATA_PATH (CSV/Excel/Parquet)."""
    from gears.pipeline import GEARSModel

    n_comp = int(n_components) if n_components.isdigit() else n_components
    model = GEARSModel(
        n_components=n_comp,
        forecaster_method=forecaster,
        n_scenarios=scenarios,
    )
    model.fit(data_path, verbose=verbose)
    model.save(output)
    click.echo(f"✓ Model saved to {output}")


# ---------------------------------------------------------------------------
# simulate (short-term)
# ---------------------------------------------------------------------------

@main.command()
@click.option("--model", "-m", default=None, help="Path to a saved .joblib model.")
@click.option("--pretrained", default=None, help="Pre-trained model ID (e.g. work_fr_demo).")
@click.option("--start", required=True, help="Start date (YYYY-MM-DD).")
@click.option("--horizon", default=7, show_default=True, help="Number of days.")
@click.option("--scenarios", default=1, show_default=True, help="Number of scenarios.")
@click.option("--n-sessions", default=None, type=int,
              help="Fixed session count per day (bypasses forecaster).")
@click.option("--output", "-o", default="sessions.csv", show_default=True,
              help="Output file path (CSV/Parquet/JSON).")
@click.option("--seed", default=None, type=int)
def simulate(model, pretrained, start, horizon, scenarios, n_sessions, output, seed):
    """Run short-term session simulation."""

    m = _load_model(model, pretrained)
    sessions = m.simulate_short_term(
        start_date=start,
        horizon=horizon,
        n_scenarios=scenarios,
        n_sessions=n_sessions,
        seed=seed,
    )
    m.export(sessions, output)
    click.echo(f"✓ {len(sessions):,} sessions written to {output}")


# ---------------------------------------------------------------------------
# medium-term
# ---------------------------------------------------------------------------

@main.command("medium-term")
@click.option("--model", "-m", default=None)
@click.option("--pretrained", default=None)
@click.option("--years", default=3, show_default=True, help="Horizon in years (max 5).")
@click.option("--growth", default=0.15, show_default=True,
              help="Annual growth rate (0.15 = +15%/yr).")
@click.option("--growth-model", default="linear", show_default=True,
              type=click.Choice(["linear", "s_curve"]))
@click.option("--output-type", default="daily_energy", show_default=True,
              type=click.Choice(["daily_energy", "hourly_energy", "sessions"]))
@click.option("--scenarios", default=10, show_default=True)
@click.option("--output", "-o", default="medium_term.csv", show_default=True)
def medium_term(model, pretrained, years, growth, growth_model, output_type, scenarios, output):
    """Run medium-term energy simulation."""

    m = _load_model(model, pretrained)
    result = m.simulate_medium_term(
        years=years,
        annual_growth_rate=growth,
        output=output_type,
        growth_model=growth_model,
        n_scenarios=scenarios,
    )
    m.export(result, output)
    click.echo(f"✓ Medium-term output ({len(result):,} rows) written to {output}")


# ---------------------------------------------------------------------------
# smart-charge
# ---------------------------------------------------------------------------

@main.command("smart-charge")
@click.option("--model", "-m", default=None)
@click.option("--pretrained", default=None)
@click.option("--sessions", "sessions_path", required=True,
              type=click.Path(exists=True), help="Sessions CSV from a previous simulate run.")
@click.option("--signal", "signal_path", required=True,
              type=click.Path(exists=True),
              help="CSV with columns: datetime, value (price or RES fraction).")
@click.option("--signal-type", default="price", type=click.Choice(["price", "res"]),
              show_default=True)
@click.option("--output", "-o", default="optimised.csv", show_default=True)
def smart_charge(model, pretrained, sessions_path, signal_path, signal_type, output):
    """Apply smart charging optimisation."""

    m = _load_model(model, pretrained)

    sessions_df = pd.read_csv(sessions_path, parse_dates=["arrival_time"])
    signal_df = pd.read_csv(signal_path, parse_dates=[0], index_col=0).squeeze()
    signal_df.index = pd.to_datetime(signal_df.index)

    optimised = m.smart_charge(sessions_df, signal_df, signal_type=signal_type)
    m.export(optimised, output)
    click.echo(f"✓ Optimised schedule written to {output}")


# ---------------------------------------------------------------------------
# list-models
# ---------------------------------------------------------------------------

@main.command("list-models")
def list_models():
    """List available pre-trained model IDs."""
    from gears.models.registry import ModelRegistry

    registry = ModelRegistry()
    df = registry.list_models()
    click.echo(df.to_string(index=False))


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _load_model(model_path, pretrained_id):
    from gears.pipeline import GEARSModel

    if pretrained_id:
        return GEARSModel.from_pretrained(pretrained_id)
    if model_path:
        return GEARSModel.load(model_path)
    click.echo("Error: provide --model or --pretrained.", err=True)
    sys.exit(1)


if __name__ == "__main__":
    main()

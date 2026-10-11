"""
GEARS command-line interface.

Usage examples
--------------
    gears fit data/sessions.csv --output models/my_model.joblib
    gears simulate --model models/my_model.joblib --start 2024-06-01 --horizon 7
    gears simulate --pretrained french --start 2024-06-01 --output out/sessions.csv

Medium-term scenarios and smart charging are available from Python
(``GEARSModel.simulate_medium_term`` / ``GEARSModel.smart_charge``).
"""

from __future__ import annotations

import click


@click.group()
@click.version_option(package_name="gears-ev")
def main():
    """GEARS – Generating Electric Vehicle Recharging Sessions."""


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------

@main.command()
@click.argument("data_path", type=click.Path(exists=True))
@click.option("--output", "-o", default="gears_model.joblib", show_default=True,
              help="Path to save the fitted model.")
@click.option("--n-components", default="auto", show_default=True,
              help="Number of mixture components ('auto' or an integer).")
@click.option("--max-components", default=10, show_default=True, type=int,
              help="Upper bound of the BIC search when --n-components is 'auto'.")
@click.option("--stratify-by", default=None,
              help="Comma-separated context columns, e.g. 'day_of_week,season' "
                   "(default) or 'location_type,department,day_of_week,season'.")
@click.option("--max-samples-per-context", default=None, type=int,
              help="Cap on training sessions per context (speeds up large datasets).")
@click.option("--model-type", default="gmm", show_default=True,
              type=click.Choice(["gmm", "vae"]),
              help="Session-model family: Gaussian mixture or conditional VAE "
                   "(vae needs the [vae] extra).")
@click.option("--recency/--no-recency", default=False, show_default=True,
              help="Fit each context on a recency-weighted bootstrap "
                   "resample instead of a uniform one (GMM only).")
@click.option("--half-life-days", default=None, type=float,
              help="Recency half-life in days (only used with --recency; "
                   "auto-derived per context group if omitted).")
@click.option("--forecaster", default="probabilistic", show_default=True,
              type=click.Choice(["sarima", "probabilistic"]),
              help="Session-count forecasting method.")
@click.option("--scenarios", default=10, show_default=True,
              help="Default number of stochastic scenarios.")
@click.option("--seed", default=42, show_default=True, type=int,
              help="Random seed used for fitting.")
@click.option("--verbose/--quiet", default=True)
def fit(data_path, output, n_components, max_components, stratify_by,
        max_samples_per_context, model_type, recency, half_life_days,
        forecaster, scenarios, seed, verbose):
    """Fit a GEARSModel on DATA_PATH (CSV/Excel/Parquet/JSON/Pickle)."""
    from gears.pipeline import GEARSModel

    model = GEARSModel(
        n_components=int(n_components) if n_components.isdigit() else n_components,
        max_components=max_components,
        stratify_by=_split_columns(stratify_by),
        max_samples_per_context=max_samples_per_context,
        model_type=model_type,
        recency=recency,
        half_life_days=half_life_days,
        forecaster_method=forecaster,
        n_scenarios=scenarios,
        random_state=seed,
    )
    model.fit(data_path, verbose=verbose)
    model.save(output)
    click.echo(f"✓ Model saved to {output}")


# ---------------------------------------------------------------------------
# simulate (short-term)
# ---------------------------------------------------------------------------

@main.command()
@click.option("--model", "-m", default=None, type=click.Path(exists=True),
              help="Path to a model saved by 'gears fit'.")
@click.option("--pretrained", default=None,
              help="Pre-trained model ID (e.g. 'french'); downloaded on first use.")
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
    if bool(model) == bool(pretrained):
        raise click.UsageError("provide exactly one of --model or --pretrained.")

    from gears.pipeline import GEARSModel
    from gears.pretrained import PretrainedModelError

    try:
        m = GEARSModel.from_pretrained(pretrained) if pretrained else GEARSModel.load(model)
    except PretrainedModelError as e:
        raise click.ClickException(str(e)) from e
    sessions = m.simulate_short_term(
        start_date=start,
        horizon=horizon,
        n_scenarios=scenarios,
        n_sessions=n_sessions,
        seed=seed,
    )
    m.export(sessions, output)
    click.echo(f"✓ {len(sessions):,} sessions written to {output}")


def _split_columns(value: str | None) -> list[str] | None:
    """Turn 'a, b,c' into ['a', 'b', 'c'] (None stays None)."""
    if value is None:
        return None
    columns = [c.strip() for c in value.split(",") if c.strip()]
    if not columns:
        raise click.BadParameter("expected at least one column name.", param_hint="--stratify-by")
    return columns


if __name__ == "__main__":
    main()

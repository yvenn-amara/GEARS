#!/usr/bin/env python3
"""
fit_session_model.py – Fit the unified GEARS GMM on EV session data.

Fits a single EVSessionModel stratified by
    location_type × département × day_of_week × season
and saves it to gears/data/session_models/gmm_french.joblib.

Supported input formats
-----------------------
    .pkl / .pickle   Pandas DataFrame (joblib or pickle protocol)
    .csv / .tsv      Auto-detects separator (comma vs. semicolon)
    .parquet         Apache Parquet
    .xlsx / .xls     Excel workbooks
    .json / .jsonl   JSON / newline-delimited JSON

Usage
-----
    # List existing GMMs in the registry
    python scripts/fit_session_model.py --list

    # Quickstart (full dataset, all defaults)
    python scripts/fit_session_model.py --input /path/to/sessions.pkl

    # CSV with custom year + BIC component range
    python scripts/fit_session_model.py \\
        --input /data/france_ev.csv \\
        --year 2025 \\
        --max-samples 5000 \\
        --n-components auto \\
        --max-components 10 \\
        --output-dir gears/data/session_models

    # Overwrite existing file without confirmation
    python scripts/fit_session_model.py --input /data/france_ev.csv --overwrite

    python scripts/fit_session_model.py --help
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Allow running from any working directory without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fit_session_model")

_SUPPORTED_EXTENSIONS = {
    ".pkl", ".pickle",
    ".csv", ".tsv",
    ".parquet",
    ".xlsx", ".xls",
    ".json", ".jsonl",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fit GEARS native GMMs or VAE on raw EV session data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input", default=None,
        help=(
            f"Path to raw session data. Supported: {sorted(_SUPPORTED_EXTENSIONS)}. "
            "Required unless --list is used."
        ),
    )
    p.add_argument(
        "--output-dir", default="gears/data/session_models",
        help="Directory where the fitted model is saved.",
    )
    p.add_argument(
        "--year", type=int, default=None,
        help="If set, filter data to sessions in this calendar year only.",
    )
    p.add_argument(
        "--max-samples", type=int, default=5000,
        help="Max training samples per (location_type, dept, dow, season) context.",
    )
    p.add_argument(
        "--n-components", default="auto",
        help="[GMM only] Components: 'auto' (BIC) or a fixed integer.",
    )
    p.add_argument(
        "--max-components", type=int, default=10,
        help="[GMM only] Upper bound on BIC search when --n-components=auto.",
    )
    p.add_argument(
        "--min-components", type=int, default=2,
        help="[GMM only] Lower bound on BIC search when --n-components=auto.",
    )
    p.add_argument(
        "--model-type", default="gmm", choices=["gmm", "vae"],
        help="Model type to fit: 'gmm' (default) or 'vae' (conditional VAE).",
    )
    p.add_argument(
        "--recency", action="store_true",
        help=(
            "[GMM only] Fit each context on a recency-weighted bootstrap resample "
            "(half-life exponential decay) instead of a uniform one. See "
            "--half-life-days. Off by default (unweighted, current behavior)."
        ),
    )
    p.add_argument(
        "--half-life-days", type=float, default=None,
        help=(
            "[GMM only, requires --recency] Half-life (days) of the recency decay: "
            "a session this many days old counts half as much as the most recent "
            "one. If omitted, it is derived per context from that context's own "
            "history span (span_days / 3.5)."
        ),
    )
    # VAE hyperparameters
    p.add_argument("--vae-latent-dim", type=int, default=16, help="[VAE] Latent space dimension.")
    p.add_argument("--vae-hidden-dim", type=int, default=256, help="[VAE] MLP hidden layer width.")
    p.add_argument("--vae-n-layers", type=int, default=2, help="[VAE] Number of hidden layers.")
    p.add_argument("--vae-epochs", type=int, default=50, help="[VAE] Training epochs.")
    p.add_argument("--vae-batch-size", type=int, default=512, help="[VAE] Mini-batch size.")
    p.add_argument("--vae-lr", type=float, default=3e-3, help="[VAE] Adam learning rate.")
    p.add_argument("--vae-beta", type=float, default=1.0, help="[VAE] Beta coefficient for KL term.")
    p.add_argument("--vae-score-n-samples", type=int, default=20, help="[VAE] IWAE K samples for scoring.")
    p.add_argument(
        "--no-filter-failed", dest="filter_failed", action="store_false",
        help=(
            "Do NOT drop failed sessions "
            "(French data: succes_session != 't'). "
            "Failures are dropped by default."
        ),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    p.add_argument(
        "--list", dest="list_session_models", action="store_true",
        help=(
            "List all models currently available in the registry "
            "(uses --output-dir as the model directory) and exit. "
            "Does not require --input."
        ),
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help=(
            "Overwrite an existing model file without interactive confirmation. "
            "Without this flag, the script aborts if the target file already exists."
        ),
    )
    return p.parse_args()


# ── Listing -------------------------------------------------------------------

def list_session_models(output_dir: Path) -> None:
    """Print a table of all GMMs registered in NativeSessionModelRegistry."""
    from gears.models.registry import NativeSessionModelRegistry

    reg = NativeSessionModelRegistry(session_model_dir=output_dir)
    df = reg.list()

    if df.empty:
        logger.info("No GMMs registered in the catalogue.")
        return

    # Pretty-print
    print("\n" + "=" * 72)
    print(f"  GEARS GMM Registry — dir: {output_dir}")
    print("=" * 72)
    for _, row in df.iterrows():
        status = "✓ AVAILABLE" if row["available"] else "✗ MISSING"
        sample_tag = " [sample]" if row["is_sample"] else " [full]"
        print(f"  {row['session_model_id']:<20} {status}{sample_tag}")
        print(f"    {row['description']}")
        print(f"    stratify_by: {row['stratify_by']}")
        print()
    print("=" * 72 + "\n")


# ── Input validation ----------------------------------------------------------

def _validate_input_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}\n"
            f"  Check the path and try again."
        )
    ext = path.suffix.lower()
    if ext not in _SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file extension '{ext}'.\n"
            f"  Supported formats: {sorted(_SUPPORTED_EXTENSIONS)}\n"
            f"  Tip: convert to CSV or Parquet if your format is not listed."
        )
    return path


def _validate_dataframe(df: pd.DataFrame) -> None:
    required = {"arrival_time", "duration", "energy"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"DataFrame is missing required columns: {missing}\n"
            f"  Available columns: {df.columns.tolist()}\n"
            f"  gears.load_sessions() auto-renames common aliases (e.g. 'start_time',\n"
            f"  'energy_kwh', 'duration_h'). Rename non-standard columns manually."
        )


# ── Overwrite guard -----------------------------------------------------------

def _check_overwrite(output_dir: Path, session_model_id: str, overwrite: bool) -> None:
    """
    Abort if the target file already exists and --overwrite was not set.

    Raises SystemExit so the caller never needs to re-check.
    """
    from gears.models.registry import NativeSessionModelRegistry

    reg = NativeSessionModelRegistry(session_model_dir=output_dir)
    catalogue = reg._CATALOGUE
    if session_model_id not in catalogue:
        return  # unknown id — let fit_and_save raise the proper error later

    target = output_dir / catalogue[session_model_id]["filename"]
    if target.exists() and not overwrite:
        logger.error(
            "Target file already exists: %s\n"
            "  Use --overwrite to replace it, or choose a different --output-dir.",
            target,
        )
        sys.exit(1)


# ── Data loading --------------------------------------------------------------

def load_data(path: Path, year: int | None, filter_failed: bool) -> pd.DataFrame:
    from gears.data.loader import load_sessions

    logger.info("Loading data from %s …", path)
    t0 = time.time()

    try:
        df = load_sessions(str(path), verbose=True, filter_failed=filter_failed)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load '{path}': {exc}\n"
            f"  Make sure the file is not corrupted and matches a supported format."
        ) from exc

    _validate_dataframe(df)
    logger.info("Loaded %d sessions in %.1fs.", len(df), time.time() - t0)

    if year is not None:
        years_present = sorted(df["arrival_time"].dt.year.unique().tolist())
        df["_year"] = df["arrival_time"].dt.year
        before = len(df)
        df = df[df["_year"] == year].drop(columns=["_year"]).reset_index(drop=True)
        logger.info("Filtered to year %d: %d / %d sessions retained.", year, len(df), before)
        if len(df) == 0:
            raise ValueError(
                f"No sessions remain after filtering to year {year}.\n"
                f"  Available years in the dataset: {years_present}"
            )

    return df


# ── Fitting -------------------------------------------------------------------

def fit_and_save(
    df: pd.DataFrame,
    session_model_id: str,
    stratify_by: list[str],
    output_dir: Path,
    n_components: str | int,
    min_components: int,
    max_components: int,
    max_samples: int,
    seed: int,
    model_type: str = "gmm",
    recency: bool = False,
    half_life_days: float | None = None,
    vae_latent_dim: int = 16,
    vae_hidden_dim: int = 256,
    vae_n_layers: int = 2,
    vae_epochs: int = 50,
    vae_batch_size: int = 512,
    vae_lr: float = 3e-3,
    vae_beta: float = 1.0,
    vae_score_n_samples: int = 20,
) -> "EVSessionModel":
    from gears.models.registry import NativeSessionModelRegistry
    from gears.models.session_model import EVSessionModel

    n_comp_arg = "auto" if n_components == "auto" else int(n_components)

    recency_note = ""
    if recency:
        hl = half_life_days if half_life_days is not None else "auto"
        recency_note = f" | recency=True half_life_days={hl}"
    logger.info(
        "Fitting %s '%s' | sessions=%d | stratify_by=%s | max_samples_per_ctx=%d%s",
        model_type.upper(), session_model_id, len(df), stratify_by, max_samples, recency_note,
    )
    t0 = time.time()

    if model_type == "vae":
        model = EVSessionModel(
            model_type="vae",
            stratify_by=stratify_by,
            max_samples_per_context=max_samples,
            random_state=seed,
            vae_latent_dim=vae_latent_dim,
            vae_hidden_dim=vae_hidden_dim,
            vae_n_layers=vae_n_layers,
            vae_epochs=vae_epochs,
            vae_batch_size=vae_batch_size,
            vae_lr=vae_lr,
            vae_beta=vae_beta,
            vae_score_n_samples=vae_score_n_samples,
        )
    else:
        model = EVSessionModel(
            n_components=n_comp_arg,
            min_components=min_components,
            max_components=max_components,
            covariance_type="full",
            stratify_by=stratify_by,
            max_samples_per_context=max_samples,
            random_state=seed,
            recency=recency,
            half_life_days=half_life_days,
        )

    year_val = int(df["arrival_time"].dt.year.mode()[0]) if len(df) > 0 else None
    model.fit(
        df,
        is_sample=False,
        metadata={
            "session_model_id": session_model_id,
            "model_type": model_type,
            "n_training_sessions": len(df),
            "stratify_by": stratify_by,
            "year": year_val,
        },
    )
    elapsed = time.time() - t0
    logger.info("Fitted '%s': %d contexts | %.1fs.", session_model_id, len(model.models_), elapsed)

    bic_df = model.bic_summary()
    logger.info(
        "  n_components — mean=%.1f | min=%d | max=%d",
        bic_df["n_components"].mean(),
        bic_df["n_components"].min(),
        bic_df["n_components"].max(),
    )

    registry   = NativeSessionModelRegistry(session_model_dir=output_dir)
    saved_path = registry.save(session_model_id, model)
    logger.info("  Saved → %s", saved_path)
    return model


# ── Entry point ---------------------------------------------------------------

def main() -> None:
    args = parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --list: display registry and exit immediately
    if args.list_session_models:
        list_session_models(output_dir)
        sys.exit(0)

    # --input is required for all other operations
    if args.input is None:
        logger.error(
            "--input is required unless --list is used.\n"
            "  Example: python scripts/fit_session_model.py --input /path/to/sessions.csv\n"
            "  Run with --list to see existing models."
        )
        sys.exit(1)

    try:
        input_path = _validate_input_path(args.input)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        sys.exit(1)

    n_components = "auto" if args.n_components == "auto" else int(args.n_components)

    # Determine session_model_id from model_type
    session_model_id = "french_vae_sample" if args.model_type == "vae" else "french"

    # Overwrite guard — abort early before spending time on fitting
    _check_overwrite(output_dir, session_model_id, args.overwrite)

    try:
        df = load_data(input_path, year=args.year, filter_failed=args.filter_failed)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.error("%s", exc)
        sys.exit(1)

    if len(df) == 0:
        logger.error("No sessions found after loading/filtering. Exiting.")
        sys.exit(1)

    # Omit 'department' from stratification when only one is present
    has_dept = "department" in df.columns and df["department"].nunique() > 1
    if not has_dept:
        logger.warning(
            "Column 'department' is missing or has a single unique value. "
            "Stratifying by ['location_type', 'day_of_week', 'season'] only."
        )
        stratify = ["location_type", "day_of_week", "season"]
    else:
        stratify = ["location_type", "department", "day_of_week", "season"]

    try:
        model = fit_and_save(
            df=df,
            session_model_id=session_model_id,
            stratify_by=stratify,
            output_dir=output_dir,
            n_components=n_components,
            min_components=args.min_components,
            max_components=args.max_components,
            max_samples=args.max_samples,
            seed=args.seed,
            model_type=args.model_type,
            vae_latent_dim=args.vae_latent_dim,
            vae_hidden_dim=args.vae_hidden_dim,
            vae_n_layers=args.vae_n_layers,
            vae_epochs=args.vae_epochs,
            vae_batch_size=args.vae_batch_size,
            vae_lr=args.vae_lr,
            vae_beta=args.vae_beta,
            vae_score_n_samples=args.vae_score_n_samples,
        )
    except Exception as exc:
        logger.error("Fitting failed: %s", exc, exc_info=True)
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Done — %d contexts fitted.", len(model.models_))
    logger.info(
        "Verify: python -c \"import gears; print(gears.NativeSessionModelRegistry().list())\""
    )


if __name__ == "__main__":
    main()

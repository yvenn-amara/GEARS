#!/usr/bin/env python3
"""
fit_gmm.py – Fit the unified GEARS GMM on EV session data.

Fits a single EVSessionGMM stratified by
    location_type × département × day_of_week × season
and saves it to gears/data/gmm/gmm_french.joblib.

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
    python scripts/fit_gmm.py --list

    # Quickstart (full dataset, all defaults)
    python scripts/fit_gmm.py --input /path/to/sessions.pkl

    # CSV with custom year + BIC component range
    python scripts/fit_gmm.py \\
        --input /data/france_ev.csv \\
        --year 2025 \\
        --max-samples 5000 \\
        --n-components auto \\
        --max-components 10 \\
        --output-dir gears/data/gmm

    # Overwrite existing file without confirmation
    python scripts/fit_gmm.py --input /data/france_ev.csv --overwrite

    python scripts/fit_gmm.py --help
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Allow running from any working directory without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import joblib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fit_gmm")

_SUPPORTED_EXTENSIONS = {
    ".pkl", ".pickle",
    ".csv", ".tsv",
    ".parquet",
    ".xlsx", ".xls",
    ".json", ".jsonl",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fit GEARS native GMMs on raw EV session data.",
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
        "--output-dir", default="gears/data/gmm",
        help="Directory where the fitted GMM is saved.",
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
        help="GMM components: 'auto' (BIC) or a fixed integer.",
    )
    p.add_argument(
        "--max-components", type=int, default=10,
        help="Upper bound on BIC search when --n-components=auto.",
    )
    p.add_argument(
        "--min-components", type=int, default=2,
        help="Lower bound on BIC search when --n-components=auto.",
    )
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
    # ------------------------------------------------------------------
    # New options
    # ------------------------------------------------------------------
    p.add_argument(
        "--list", dest="list_gmms", action="store_true",
        help=(
            "List all GMMs currently available in the registry "
            "(uses --output-dir as the GMM directory) and exit. "
            "Does not require --input."
        ),
    )
    p.add_argument(
        "--overwrite", action="store_true",
        help=(
            "Overwrite an existing GMM file without interactive confirmation. "
            "Without this flag, the script aborts if the target file already exists."
        ),
    )
    return p.parse_args()


# ── Listing -------------------------------------------------------------------

def list_gmms(output_dir: Path) -> None:
    """Print a table of all GMMs registered in NativeGMMRegistry."""
    from gears.models.registry import NativeGMMRegistry

    reg = NativeGMMRegistry(gmm_dir=output_dir)
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
        print(f"  {row['gmm_id']:<20} {status}{sample_tag}")
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

def _check_overwrite(output_dir: Path, gmm_id: str, overwrite: bool) -> None:
    """
    Abort if the target file already exists and --overwrite was not set.

    Raises SystemExit so the caller never needs to re-check.
    """
    from gears.models.registry import NativeGMMRegistry

    reg = NativeGMMRegistry(gmm_dir=output_dir)
    catalogue = reg._CATALOGUE
    if gmm_id not in catalogue:
        return  # unknown id — let fit_and_save raise the proper error later

    target = output_dir / catalogue[gmm_id]["filename"]
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
    gmm_id: str,
    stratify_by: list[str],
    output_dir: Path,
    n_components: str | int,
    min_components: int,
    max_components: int,
    max_samples: int,
    seed: int,
) -> "EVSessionGMM":
    from gears.models.gmm import EVSessionGMM
    from gears.models.registry import NativeGMMRegistry

    n_comp_arg = "auto" if n_components == "auto" else int(n_components)

    logger.info(
        "Fitting GMM '%s' | sessions=%d | stratify_by=%s | max_samples_per_ctx=%d",
        gmm_id, len(df), stratify_by, max_samples,
    )
    t0 = time.time()

    gmm = EVSessionGMM(
        n_components=n_comp_arg,
        min_components=min_components,
        max_components=max_components,
        covariance_type="full",
        stratify_by=stratify_by,
        max_samples_per_context=max_samples,
        random_state=seed,
    )

    year_val = int(df["arrival_time"].dt.year.mode()[0]) if len(df) > 0 else None
    gmm.fit(
        df,
        is_sample=False,
        metadata={
            "gmm_id": gmm_id,
            "n_training_sessions": len(df),
            "stratify_by": stratify_by,
            "year": year_val,
        },
    )
    elapsed = time.time() - t0
    logger.info("Fitted '%s': %d contexts | %.1fs.", gmm_id, len(gmm.models_), elapsed)

    bic_df = gmm.bic_summary()
    logger.info(
        "  BIC — mean_k=%.1f | min=%d | max=%d",
        bic_df["n_components"].mean(),
        bic_df["n_components"].min(),
        bic_df["n_components"].max(),
    )

    registry   = NativeGMMRegistry(gmm_dir=output_dir)
    saved_path = registry.save(gmm_id, gmm)
    logger.info("  Saved → %s", saved_path)
    return gmm


# ── Entry point ---------------------------------------------------------------

def main() -> None:
    args = parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --list: display registry and exit immediately
    if args.list_gmms:
        list_gmms(output_dir)
        sys.exit(0)

    # --input is required for all other operations
    if args.input is None:
        logger.error(
            "--input is required unless --list is used.\n"
            "  Example: python scripts/fit_gmm.py --input /path/to/sessions.csv\n"
            "  Run with --list to see existing GMMs."
        )
        sys.exit(1)

    try:
        input_path = _validate_input_path(args.input)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        sys.exit(1)

    n_components = "auto" if args.n_components == "auto" else int(args.n_components)

    # Overwrite guard — abort early before spending time on fitting
    _check_overwrite(output_dir, "french", args.overwrite)

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
        gmm = fit_and_save(
            df=df,
            gmm_id="french",
            stratify_by=stratify,
            output_dir=output_dir,
            n_components=n_components,
            min_components=args.min_components,
            max_components=args.max_components,
            max_samples=args.max_samples,
            seed=args.seed,
        )
    except Exception as exc:
        logger.error("Fitting failed: %s", exc, exc_info=True)
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Done — %d contexts fitted.", len(gmm.models_))
    logger.info(
        "Verify: python -c \"import gears; print(gears.NativeGMMRegistry().list())\""
    )


if __name__ == "__main__":
    main()

"""
Data loader for GEARS – handles CSV, Excel, Parquet, Pickle, JSON,
and in-memory DataFrames.  Auto-detects the French national dataset format.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from gears.data.schemas import validate_dataframe, summary_stats

logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS = {
    ".csv": "_load_csv",
    ".tsv": "_load_csv",
    ".xlsx": "_load_excel",
    ".xls": "_load_excel",
    ".parquet": "_load_parquet",
    ".json": "_load_json",
    ".jsonl": "_load_jsonl",
    ".pkl": "_load_pickle",
    ".pickle": "_load_pickle",
}


def load_sessions(
    source: Union[str, Path, pd.DataFrame],
    *,
    strict: bool = False,
    filter_failed: bool = True,
    verbose: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """
    Load and validate EV charging sessions from various sources.

    Automatically detects the French national dataset format and applies
    the appropriate preprocessing, including unit conversions (Wh -> kWh,
    min -> hours) and filtering of failed sessions.

    Parameters
    ----------
    source : str | Path | pd.DataFrame
        Path to file (CSV / Excel / Parquet / JSON / Pickle) or a DataFrame.
    strict : bool
        If True, raise on data quality issues instead of dropping rows.
    filter_failed : bool
        French data only: drop sessions with succes_session != 't'.
    verbose : bool
        Print a short summary after loading.
    **kwargs
        Extra keyword arguments forwarded to the underlying reader.

    Returns
    -------
    pd.DataFrame
        Validated DataFrame with canonical columns and derived features.
    """
    if isinstance(source, pd.DataFrame):
        raw = source.copy()
    else:
        raw = _dispatch_file(Path(source), **kwargs)

    df = validate_dataframe(raw, strict=strict, filter_failed=filter_failed)

    if verbose:
        logger.info("Loaded %d sessions.", len(df))
        print(f"[GEARS] Loaded {len(df):,} sessions.")
        if "department" in df.columns:
            print(f"[GEARS] Departments: {df['department'].nunique()}")
        if "location_type" in df.columns:
            print(f"[GEARS] Location types: {dict(df['location_type'].value_counts())}")
        print(summary_stats(df).to_string())

    return df


def _dispatch_file(path: Path, **kwargs) -> pd.DataFrame:
    ext = path.suffix.lower()
    method_name = _SUPPORTED_EXTENSIONS.get(ext)
    if method_name is None:
        raise ValueError(
            f"Unsupported file extension '{ext}'. "
            f"Supported: {list(_SUPPORTED_EXTENSIONS)}"
        )
    return globals()[method_name](path, **kwargs)


def _load_csv(path: Path, **kwargs) -> pd.DataFrame:
    sep = kwargs.pop("sep", None)
    if sep is None:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            sample = f.read(2048)
        sep = ";" if sample.count(";") > sample.count(",") else ","
    return pd.read_csv(path, sep=sep, low_memory=False, **kwargs)


def _load_excel(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_excel(path, **kwargs)


def _load_parquet(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_parquet(path, **kwargs)


def _load_json(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_json(path, **kwargs)


def _load_jsonl(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_json(path, lines=True, **kwargs)


def _load_pickle(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_pickle(path, **kwargs)


def make_demo_data(
    n: int = 1_000,
    location_type: str = "work",
    seed: int = 42,
    start_date: str = "2023-01-01",
    end_date: str = "2023-12-31",
) -> pd.DataFrame:
    """
    Generate a realistic synthetic EV sessions dataset for testing and demos.

    Parameters
    ----------
    n : int
        Number of sessions to generate.
    location_type : str
        'work', 'home', or 'public'.
    seed : int
        Random seed for reproducibility.
    start_date, end_date : str
        Date range for arrival times.

    Returns
    -------
    pd.DataFrame
        Validated sessions DataFrame.
    """
    rng = np.random.default_rng(seed)

    dates = pd.date_range(start_date, end_date, freq="D")
    arrival_dates = rng.choice(dates, size=n)

    params: dict[str, dict] = {
        "work":   {"hour_mu": [8.5, 9.5], "hour_sigma": [0.8, 1.2],
                   "hour_weights": [0.6, 0.4],
                   "dur_mu": 7.0, "dur_sigma": 2.5,
                   "energy_mu": 15.0, "energy_sigma": 8.0},
        "home":   {"hour_mu": [18.0, 22.0], "hour_sigma": [1.5, 1.0],
                   "hour_weights": [0.7, 0.3],
                   "dur_mu": 10.0, "dur_sigma": 3.0,
                   "energy_mu": 20.0, "energy_sigma": 10.0},
        "public": {"hour_mu": [11.0, 14.5, 17.0], "hour_sigma": [1.5, 1.0, 1.5],
                   "hour_weights": [0.3, 0.4, 0.3],
                   "dur_mu": 1.5, "dur_sigma": 1.0,
                   "energy_mu": 12.0, "energy_sigma": 7.0},
    }
    p = params.get(location_type, params["work"])

    weights = p["hour_weights"]
    components = rng.choice(len(weights), size=n, p=weights)
    hours_float = np.array([
        rng.normal(p["hour_mu"][c], p["hour_sigma"][c])
        for c in components
    ])
    hours_float = np.clip(hours_float, 0, 23.99)

    arrival_times = pd.to_datetime(arrival_dates) + pd.to_timedelta(hours_float, unit="h")

    base_dur = rng.normal(p["dur_mu"], p["dur_sigma"], n)
    duration = np.clip(base_dur, 0.25, 24.0)
    energy = np.clip(
        rng.normal(p["energy_mu"], p["energy_sigma"], n) * (duration / p["dur_mu"]) ** 0.3,
        0.1, 150.0,
    )

    df = pd.DataFrame({
        "arrival_time": arrival_times,
        "duration": duration,
        "energy": energy,
        "location_type": location_type,
    })
    return load_sessions(df, verbose=False)

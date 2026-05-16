"""
Data schemas and validation for GEARS input data.

Supports both generic EV session datasets and the French national dataset
(with INSEE department codes, Wh energy values, minute durations, etc.).

Expected canonical columns after validation:
    arrival_time  : datetime of plug-in
    duration      : session duration in hours  (>0)
    energy        : energy delivered in kWh    (>=0)
    power         : (optional) charger power in kW
    location_type : (optional) 'work' | 'home' | 'public' | 'heavy'
    user_id       : (optional) anonymised user identifier
    department    : (optional) INSEE department code

Derived columns added automatically:
    hour          : arrival hour (float, 0-24)
    day_of_week   : 0=Monday, 6=Sunday
    month         : 1-12
    season        : 'winter' | 'spring' | 'summer' | 'autumn'
    is_weekend    : 0/1
    date          : date object
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd


def _season(month: int) -> str:
    """Northern-hemisphere meteorological season."""
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "autumn"


# French national dataset: domaine_subvention code -> location_type
FRENCH_DOMAINE_MAP: dict[str, str] = {
    "en_opu": "public",   # entreprise, ouvert au public
    "en_pri": "work",     # entreprise, prive
    "coll":   "public",   # collectivite
    "re_co":  "home",     # residentiel collectif
    "po_lo":  "heavy",    # poids lourds
}

FRENCH_COLUMN_MAP: dict[str, str] = {
    "debut_session_timestamp":  "arrival_time",
    "fin_session_timestamp":    "end_time",
    "duree_min":                "_duration_min",
    "energie_delivree_wh":      "_energy_wh",
    "succes_session":           "_succes",
    "id_domaine_subvention":    "_domaine_id",
    "nom_domaine_subvention":   "_domaine_nom",
    "insee_code_departement":   "department",
    "insee_nom_departement":    "department_name",
    "insee_code_region":        "region_code",
    "insee_nom_region":         "region_name",
    "datasource":               "datasource",
}

COLUMN_ALIASES: dict[str, list[str]] = {
    "arrival_time": [
        "arrival_time", "start_time", "start", "plug_in_time",
        "connection_time", "timestamp", "date_start", "t_start",
    ],
    "duration": [
        "duration", "duration_h", "duration_hours", "session_duration",
        "connected_time", "charging_duration",
    ],
    "end_time": [
        "end_time", "stop_time", "end", "plug_out_time",
        "disconnection_time", "t_end",
    ],
    "energy": [
        "energy", "energy_kwh", "kwh", "total_energy", "energy_delivered",
        "charged_energy",
    ],
    "power": [
        "power", "power_kw", "max_power", "rated_power", "charger_power",
    ],
    "location_type": [
        "location_type", "site_type", "charging_type", "type",
        "location", "site",
    ],
    "user_id": ["user_id", "ev_id", "vehicle_id", "driver_id", "session_id"],
    "department": [
        "department", "dept", "departement", "code_dept", "dept_code",
    ],
}

REQUIRED_COLS = {"arrival_time", "duration", "energy"}

LOCATION_NORMALISATION: dict[str, str] = {
    "work": "work", "workplace": "work", "bureau": "work", "office": "work",
    "en_pri": "work",
    "home": "home", "residential": "home", "domicile": "home",
    "re_co": "home",
    "public": "public", "street": "public", "commercial": "public",
    "en_opu": "public", "coll": "public",
    "heavy": "heavy", "po_lo": "heavy", "poids_lourds": "heavy",
}


def _is_french_format(df: pd.DataFrame) -> bool:
    return "debut_session_timestamp" in df.columns or "energie_delivree_wh" in df.columns


def _preprocess_french(df: pd.DataFrame, *, filter_failed: bool = True) -> pd.DataFrame:
    df = df.copy()
    rename = {k: v for k, v in FRENCH_COLUMN_MAP.items() if k in df.columns}
    df.rename(columns=rename, inplace=True)

    if filter_failed and "_succes" in df.columns:
        before = len(df)
        df = df[df["_succes"] == "t"].reset_index(drop=True)
        dropped = before - len(df)
        if dropped > 0:
            warnings.warn(
                f"Dropped {dropped:,}/{before:,} failed sessions (succes_session != 't').",
                UserWarning, stacklevel=4,
            )
        df.drop(columns=["_succes"], inplace=True)

    # French IRVE: energie_delivree_wh is in Wh — convert to kWh.
    # This conversion only applies when the raw column is detected as French format
    # (debut_session_timestamp or energie_delivree_wh present). If data is already
    # in kWh, no conversion occurs. See _is_french_format() above.
    if "_energy_wh" in df.columns:
        df["energy"] = df["_energy_wh"] / 1000.0
        df.drop(columns=["_energy_wh"], inplace=True)

    if "_duration_min" in df.columns:
        df["duration"] = df["_duration_min"] / 60.0
        df.drop(columns=["_duration_min"], inplace=True)

    if "_domaine_id" in df.columns:
        df["location_type"] = df["_domaine_id"].map(FRENCH_DOMAINE_MAP).fillna("unknown")
        df.drop(columns=["_domaine_id"], inplace=True)
    if "_domaine_nom" in df.columns:
        df.drop(columns=["_domaine_nom"], inplace=True)

    if "department" in df.columns:
        df["department"] = df["department"].astype(str).str.strip()

    return df


def _find_column(df: pd.DataFrame, canonical: str) -> Optional[str]:
    for alias in COLUMN_ALIASES.get(canonical, [canonical]):
        if alias in df.columns:
            return alias
    return None


def _resolve_aliases(df: pd.DataFrame) -> pd.DataFrame:
    rename_map: dict[str, str] = {}
    for canonical in COLUMN_ALIASES:
        col = _find_column(df, canonical)
        if col and col != canonical:
            rename_map[col] = canonical
    df.rename(columns=rename_map, inplace=True)
    return df


def validate_dataframe(
    df: pd.DataFrame,
    strict: bool = False,
    filter_failed: bool = True,
) -> pd.DataFrame:
    """
    Validate and normalise a raw EV sessions DataFrame.

    Automatically detects the French national format and applies the
    appropriate preprocessing. Generic datasets are supported via
    column-alias resolution.

    Parameters
    ----------
    df : pd.DataFrame
        Raw data with arbitrary column names.
    strict : bool
        If True, raise on any anomaly instead of dropping/warning.
    filter_failed : bool
        French data only: if True, drop sessions with succes_session != 't'.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with canonical columns and derived calendar features.
    """
    df = df.copy()

    if _is_french_format(df):
        df = _preprocess_french(df, filter_failed=filter_failed)

    df = _resolve_aliases(df)

    if "duration" not in df.columns and "end_time" in df.columns:
        at = pd.to_datetime(df["arrival_time"], format="mixed", dayfirst=False)
        et = pd.to_datetime(df["end_time"], format="mixed", dayfirst=False)
        df["duration"] = (et - at).dt.total_seconds() / 3600.0

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns after aliasing: {missing}")

    df["arrival_time"] = pd.to_datetime(df["arrival_time"], format="mixed", dayfirst=False)
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce")
    df["energy"] = pd.to_numeric(df["energy"], errors="coerce")
    if "power" in df.columns:
        df["power"] = pd.to_numeric(df["power"], errors="coerce")

    n_before = len(df)
    mask = (
        df["duration"].notna() & (df["duration"] > 0) & (df["duration"] < 168)
        & df["energy"].notna() & (df["energy"] >= 0) & (df["energy"] < 400)
        & df["arrival_time"].notna()
    )
    bad = (~mask).sum()
    if bad > 0:
        msg = f"Dropping {bad:,}/{n_before:,} rows failing quality filters."
        if strict:
            raise ValueError(msg)
        warnings.warn(msg, UserWarning, stacklevel=3)
    df = df[mask].reset_index(drop=True)

    df["hour"] = df["arrival_time"].dt.hour + df["arrival_time"].dt.minute / 60.0
    df["day_of_week"] = df["arrival_time"].dt.dayofweek   # 0=Monday, 6=Sunday
    df["month"] = df["arrival_time"].dt.month
    df["season"] = df["month"].apply(_season)
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["date"] = df["arrival_time"].dt.date

    if "location_type" in df.columns:
        df["location_type"] = (
            df["location_type"]
            .astype(str)
            .str.lower()
            .str.strip()
            .map(LOCATION_NORMALISATION)
            .fillna("unknown")
        )

    if "department" in df.columns:
        df["department"] = df["department"].astype(str).str.strip()

    return df


def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return a quick descriptive summary of a validated sessions DataFrame."""
    cols = [c for c in ["duration", "energy", "power"] if c in df.columns]
    stats = df[cols].describe().T
    stats["missing_%"] = df[cols].isna().mean() * 100
    return stats

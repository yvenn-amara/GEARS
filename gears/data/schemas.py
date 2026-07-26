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

import re
import warnings

import pandas as pd


def _season(month: int) -> str:
    """
    Return the northern-hemisphere meteorological season for a given month.

    Parameters
    ----------
    month : int
        Calendar month (1 = January … 12 = December).

    Returns
    -------
    str
        One of ``'winter'``, ``'spring'``, ``'summer'``, ``'autumn'``.

    Examples
    --------
    >>> _season(1)
    'winter'
    >>> _season(7)
    'summer'
    >>> _season(10)
    'autumn'
    """
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

# Raw column names that must NEVER resolve to a given canonical column, even
# though a case/punctuation-normalized comparison would otherwise make them
# look like a match. This is a *meaning* problem, not a normalization
# problem — e.g. the EVSE-style datasets' ``Arrival`` column is a float
# hour-of-day (e.g. 4.13), not a timestamp, even though its name looks like
# a near-miss alias for ``arrival_time``. No amount of fuzzy matching should
# paper over this; it is handled as an explicit, permanent exclusion.
EXCLUDED_RAW_ALIASES: dict[str, set[str]] = {
    "arrival_time": {"arrival"},
}

# Raw columns known to carry a *duration in minutes* rather than the
# canonical hours, matched by normalized name. Handled as an explicit unit
# conversion (see ``_convert_minute_duration_columns``) rather than a plain
# alias rename, mirroring the existing French ``duree_min`` -> ``duration``
# (/ 60) conversion in ``_preprocess_french`` — the same kind of fix, just
# triggered generically instead of by a France-specific literal.
MINUTE_DURATION_COLUMNS: set[str] = {"park_duration", "duree_min"}

# Raw columns that are redundant with a value GEARS derives itself (e.g.
# ``Day``/``Weekend`` duplicate the ``day_of_week``/``is_weekend`` computed
# from ``arrival_time``) or are pandas/export artefacts (an unnamed index
# column, a date-only column duplicating a full timestamp). None of these
# are required or consumed downstream, so they are dropped as noise rather
# than left to accumulate as confusing passthrough columns.
NOISE_COLUMNS: set[str] = {"day", "weekend", "unnamed_0", "start_date"}

REQUIRED_COLS = {"arrival_time", "duration", "energy"}

# Data-quality note: acn.csv (from the 11-dataset EVSE benchmark) is the
# row-wise union of caltech.csv + jpl.csv + office.csv, distinguishable via
# its extra ``data`` site-indicator column. Loading acn.csv alongside any of
# those three individually and combining them will triple-count the
# overlapping sessions — treat acn as a deliberate "combined-site" scenario,
# not as a fourth independent dataset, unless that duplication is intended.
# See ``_warn_if_combined_site_dataset`` for the runtime version of this note.
_KNOWN_COMBINED_SITE_VALUES = {"caltech", "jpl", "office"}


def _warn_if_combined_site_dataset(df: pd.DataFrame) -> None:
    """
    Emit a ``UserWarning`` if ``df`` looks like a combined multi-site export
    (e.g. ``acn.csv`` = ``caltech.csv`` + ``jpl.csv`` + ``office.csv``
    concatenated, distinguished by a ``data`` site-indicator column), so
    nobody accidentally loads it alongside the individual site files and
    double/triple-counts the overlapping sessions.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    None
    """
    normalized = {_normalize_key(c): c for c in df.columns}
    site_col = normalized.get("data")
    if site_col is None:
        return
    sites = set(df[site_col].dropna().astype(str).str.lower().unique())
    if sites and sites.issubset(_KNOWN_COMBINED_SITE_VALUES) and len(sites) > 1:
        warnings.warn(
            f"This dataset looks like a combined multi-site export (site values: "
            f"{sorted(sites)}) — e.g. acn.csv = caltech.csv + jpl.csv + office.csv "
            "concatenated. Loading it alongside those individual site files in the "
            "same analysis will double/triple-count overlapping sessions.",
            UserWarning, stacklevel=3,
        )

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
    """
    Return True if ``df`` looks like the French national IRVE dataset.

    Detection relies on the presence of ``debut_session_timestamp`` or
    ``energie_delivree_wh`` — columns unique to that format.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    bool
    """
    return "debut_session_timestamp" in df.columns or "energie_delivree_wh" in df.columns


def _preprocess_french(df: pd.DataFrame, *, filter_failed: bool = True) -> pd.DataFrame:
    """
    Preprocess a French national IRVE dataset into canonical GEARS columns.

    Renames columns, converts units (Wh → kWh, minutes → hours), maps
    domaine codes to location types, and optionally drops failed sessions.

    Parameters
    ----------
    df : pd.DataFrame
        Raw French national dataset.
    filter_failed : bool
        If True, drop rows where ``succes_session != 't'``.

    Returns
    -------
    pd.DataFrame
        Partially normalised DataFrame ready for ``validate_dataframe``.
    """
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


def _normalize_key(s: str) -> str:
    """
    Normalize a column name for case/punctuation-insensitive alias matching.

    Lower-cases the string and collapses any run of non-alphanumeric
    characters into a single underscore, so ``'Park.Duration'``,
    ``'park duration'``, and ``'park_duration'`` all normalize identically.

    Parameters
    ----------
    s : str

    Returns
    -------
    str

    Examples
    --------
    >>> _normalize_key("Park.Duration")
    'park_duration'
    >>> _normalize_key("Start Time")
    'start_time'
    """
    return re.sub(r"[^a-z0-9]+", "_", s.strip().lower()).strip("_")


def _find_column(df: pd.DataFrame, canonical: str) -> str | None:
    """
    Return the first alias of ``canonical`` present in ``df``, matched in a
    case/punctuation-insensitive way (e.g. ``'Start'`` matches the
    ``'start'`` alias, ``'Park.Duration'`` matches ``'park_duration'``).

    Aliases listed in :data:`EXCLUDED_RAW_ALIASES` for ``canonical`` are
    never matched, even if their normalized form would otherwise line up —
    this is a semantic exclusion (e.g. ``Arrival`` vs. ``arrival_time``),
    not something normalization should ever paper over.

    Parameters
    ----------
    df : pd.DataFrame
    canonical : str
        Canonical column name (key in :data:`COLUMN_ALIASES`).

    Returns
    -------
    str or None
        Matching (raw, un-normalized) column name, or ``None`` if none is
        found.
    """
    normalized = {_normalize_key(c): c for c in df.columns}
    excluded = EXCLUDED_RAW_ALIASES.get(canonical, set())
    for alias in COLUMN_ALIASES.get(canonical, [canonical]):
        key = _normalize_key(alias)
        if key in excluded:
            continue
        hit = normalized.get(key)
        if hit is not None:
            return hit
    return None


def _resolve_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename aliased columns to their canonical GEARS names.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        DataFrame with columns renamed in-place.
    """
    rename_map: dict[str, str] = {}
    for canonical in COLUMN_ALIASES:
        col = _find_column(df, canonical)
        if col and col != canonical:
            rename_map[col] = canonical
    df.rename(columns=rename_map, inplace=True)
    return df


def _convert_minute_duration_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect a raw duration column expressed in **minutes** (matched by
    normalized name against :data:`MINUTE_DURATION_COLUMNS`, e.g. the
    EVSE-style ``Park.Duration`` column) and convert it into the canonical
    ``duration`` column in hours.

    This mirrors the unit conversion ``_preprocess_french`` already applies
    to its own French-specific ``_duration_min`` column, but is triggered
    generically by normalized column name rather than a France-only
    literal, so it also covers non-French minute-based datasets.

    No-ops if ``duration`` is already present (already resolved by a
    direct/hour-based alias) or no known minute-duration column is found.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    if "duration" in df.columns:
        return df
    normalized = {_normalize_key(c): c for c in df.columns}
    for key in MINUTE_DURATION_COLUMNS:
        raw_col = normalized.get(key)
        if raw_col is not None:
            df = df.copy()
            df["duration"] = pd.to_numeric(df[raw_col], errors="coerce") / 60.0
            return df
    return df


def _drop_noise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop raw columns matched (by normalized name) against
    :data:`NOISE_COLUMNS` — values GEARS derives itself (``Day``,
    ``Weekend``) or pandas/export artefacts (an unnamed index column, a
    redundant date-only column). Safe no-op if none are present.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    to_drop = [c for c in df.columns if _normalize_key(c) in NOISE_COLUMNS]
    if to_drop:
        df = df.drop(columns=to_drop)
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
        French data only: if True, drop sessions with ``succes_session != 't'``.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with canonical columns and derived calendar
        features: ``hour``, ``day_of_week``, ``month``, ``season``,
        ``is_weekend``, ``date``.

    Raises
    ------
    ValueError
        If required columns (``arrival_time``, ``duration``, ``energy``) are
        still missing after alias resolution, or if ``strict=True`` and any
        rows fail quality filters.

    Examples
    --------
    >>> import pandas as pd
    >>> raw = pd.DataFrame({
    ...     "start_time": ["2025-01-15 08:30", "2025-06-01 18:00"],
    ...     "duration_h": [7.5, 2.0],
    ...     "energy_kwh": [15.0, 8.0],
    ... })
    >>> df = validate_dataframe(raw)
    >>> df.columns.tolist()  # doctest: +NORMALIZE_WHITESPACE
    ['arrival_time', 'duration', 'energy', 'hour', 'day_of_week', ...]
    """
    df = df.copy()

    if _is_french_format(df):
        df = _preprocess_french(df, filter_failed=filter_failed)

    _warn_if_combined_site_dataset(df)

    df = _drop_noise_columns(df)
    df = _resolve_aliases(df)
    df = _convert_minute_duration_columns(df)

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
    """
    Return a quick descriptive summary of a validated sessions DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Validated sessions DataFrame (output of :func:`validate_dataframe`).

    Returns
    -------
    pd.DataFrame
        Descriptive statistics (count, mean, std, min, quartiles, max) for
        ``duration``, ``energy``, and ``power`` (if present), with an extra
        ``missing_%`` column.
    """
    cols = [c for c in ["duration", "energy", "power"] if c in df.columns]
    stats = df[cols].describe().T
    stats["missing_%"] = df[cols].isna().mean() * 100
    return stats

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

Conventions
-----------
Time: ``arrival_time`` is **local wall-clock time** at the charging site, and is
stored without a time zone. A time zone or UTC offset in the input is dropped and
the clock reading kept, not converted (a ``UserWarning`` says so): if your
timestamps are in UTC, convert them to local time first. Simulated
``arrival_time`` values are naive local times as well.
Units: ``duration`` in hours, ``energy`` in kWh. A median that is implausible for
charging sessions triggers a ``UserWarning`` suggesting Wh or minutes.
Quality filters: rows with a missing or unreadable ``arrival_time``, a missing or
non-positive ``duration``, a missing or negative ``energy``, an implausible value
(``duration >= 168`` h, ``energy >= 400`` kWh) or that duplicate an earlier row are
dropped and counted in a single ``UserWarning``; if nothing is left, a
``ValueError`` is raised.
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
# problem — e.g. some public datasets' ``Arrival`` column is a float
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

# Plausibility bounds: rows at or beyond them are dropped (and reported).
MAX_DURATION_H = 168.0   # one week
MAX_ENERGY_KWH = 400.0
# Unit heuristics, applied to the median of the raw column. A typical charging
# session lasts 1-24 h and delivers 5-60 kWh, so a median far above these values
# means the column is not in the expected unit.
SUSPECT_ENERGY_MEDIAN_KWH = 200.0   # above: probably Wh
SUSPECT_DURATION_MEDIAN_H = 48.0    # above: probably minutes

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
    df = df.rename(columns=rename)

    if filter_failed and "_succes" in df.columns:
        before = len(df)
        df = df[df["_succes"] == "t"].reset_index(drop=True)
        dropped = before - len(df)
        if dropped > 0:
            warnings.warn(
                f"Dropped {dropped:,}/{before:,} failed sessions (succes_session != 't').",
                UserWarning, stacklevel=4,
            )
        df = df.drop(columns=["_succes"])

    # French IRVE: energie_delivree_wh is in Wh — convert to kWh.
    # This conversion only applies when the raw column is detected as French format
    # (debut_session_timestamp or energie_delivree_wh present). If data is already
    # in kWh, no conversion occurs. See _is_french_format() above.
    if "_energy_wh" in df.columns:
        df["energy"] = df["_energy_wh"] / 1000.0
        df = df.drop(columns=["_energy_wh"])

    if "_duration_min" in df.columns:
        df["duration"] = df["_duration_min"] / 60.0
        df = df.drop(columns=["_duration_min"])

    if "_domaine_id" in df.columns:
        df["location_type"] = df["_domaine_id"].map(FRENCH_DOMAINE_MAP).fillna("unknown")
        df = df.drop(columns=["_domaine_id"])
    if "_domaine_nom" in df.columns:
        df = df.drop(columns=["_domaine_nom"])

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
    df = df.rename(columns=rename_map)
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


def _missing_columns_error(df: pd.DataFrame, missing: set[str]) -> ValueError:
    """Build the error for missing required columns: what is missing, what is accepted."""
    lines = []
    for canonical in sorted(missing):
        accepted = ", ".join(COLUMN_ALIASES.get(canonical, [canonical]))
        line = f"  - {canonical}: accepted names are {accepted}"
        if canonical == "duration":
            line += (
                f"; or give end_time (duration = end_time - arrival_time); or a duration in "
                f"minutes named one of {', '.join(sorted(MINUTE_DURATION_COLUMNS))}"
            )
        lines.append(line)
    return ValueError(
        f"Missing required column(s): {sorted(missing)}.\n"
        + "\n".join(lines)
        + f"\nColumns found in your data: {list(df.columns)}.\n"
        "Rename your columns to one of the accepted names (case and punctuation are ignored)."
    )


def _wall_clock(value):
    """Timestamp of ``value`` without its time zone (clock reading kept)."""
    ts = pd.Timestamp(value)
    return ts.tz_localize(None) if ts.tzinfo is not None else ts


# A date written as numbers with a separator: 05/01/2024, 5.1.24, 13-01-2024 ... (not ISO 2024-01-05)
_NUMERIC_DATE = re.compile(r"^\s*(\d{1,2})[/.\-](\d{1,2})[/.\-]\d{2,4}(?!\d)")


def _date_order(*columns: pd.Series) -> bool | None:
    """Day first (``True``) or month first (``False``) in numeric dates such as 05/01/2024.

    The order is read from the values themselves: a first number above 12 can only be a
    day, a second one only a day in the other order. ``None`` when the columns hold no such
    dates (ISO dates, real timestamps) or when they cannot settle it (every number is 12
    or less).

    Raises
    ------
    ValueError
        If the dates use both orders (some first numbers and some second numbers above 12).
    """
    firsts, seconds = [], []
    for column in columns:
        if column.dtype.kind in "Mm" or pd.api.types.is_numeric_dtype(column):
            continue
        found = column.dropna().astype(str).str.extract(_NUMERIC_DATE)
        found = found.dropna()
        firsts.append(found[0].astype(int))
        seconds.append(found[1].astype(int))
    if not firsts:
        return None
    day_first = bool((pd.concat(firsts) > 12).any())
    month_first = bool((pd.concat(seconds) > 12).any())
    if day_first and month_first:
        raise ValueError(
            "The dates mix two orders: some read day/month/year (a first number above 12, "
            "e.g. 13/01/2024) and others month/day/year (a second number above 12, e.g. "
            "01/13/2024). Write them all in one order, preferably as 2024-01-13."
        )
    if day_first or month_first:
        return day_first
    return None


def _has_ambiguous_dates(column: pd.Series) -> bool:
    """True when the column holds numeric dates like 05/01/2024 (order unknown)."""
    if column.dtype.kind in "Mm" or pd.api.types.is_numeric_dtype(column):
        return False
    return bool(column.dropna().astype(str).str.match(_NUMERIC_DATE).any())


def _parse_arrival_time(
    column: pd.Series, dayfirst: bool | None = None
) -> tuple[pd.Series, bool]:
    """
    Parse ``arrival_time`` into naive local timestamps.

    Returns the parsed column and whether a time zone / UTC offset was dropped.
    Unreadable values become ``NaT`` (the quality filter then drops and counts them).
    ``dayfirst`` says how numeric dates such as 05/01/2024 are read (``True``: 5 January);
    ``None`` reads it from the values (see :func:`_date_order`), month first when they
    cannot settle it.

    Raises
    ------
    ValueError
        If the column holds numbers instead of dates.
    """
    if pd.api.types.is_numeric_dtype(column) and not pd.api.types.is_bool_dtype(column):
        raise ValueError(
            f"arrival_time holds numbers (first value: {column.iloc[0]!r}), not dates. "
            "GEARS needs the date and time of plug-in, e.g. '2025-06-10 08:30'. "
            "A number such as 4.13 is an hour of the day: use the column that holds the "
            "full timestamp instead."
        )
    if isinstance(column.dtype, pd.DatetimeTZDtype):
        return column.dt.tz_localize(None), True
    try:
        parsed = pd.to_datetime(column, format="mixed", dayfirst=bool(dayfirst), errors="coerce")
    except ValueError:
        # Several UTC offsets in one column (e.g. +01:00 and +02:00 across a DST change):
        # keep each clock reading, as the convention says.
        parsed = pd.to_datetime(
            column.map(lambda v: _wall_clock(v) if pd.notna(v) else pd.NaT),
            errors="coerce",
        )
        return parsed, True
    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        return parsed.dt.tz_localize(None), True
    return parsed, False


def _warn_if_suspicious_units(df: pd.DataFrame) -> None:
    """Warn when the median energy / duration is implausible for charging sessions."""
    energy = df["energy"].dropna()
    energy = energy[energy > 0]
    if len(energy) and energy.median() > SUSPECT_ENERGY_MEDIAN_KWH:
        warnings.warn(
            f"The median energy is {energy.median():,.0f}, above {SUSPECT_ENERGY_MEDIAN_KWH:.0f}, "
            "which is implausible for charging sessions in kWh: the column is probably in Wh. "
            "GEARS expects kWh; divide it by 1000 before loading (the French national format "
            f"is converted automatically). Values >= {MAX_ENERGY_KWH:.0f} are dropped as outliers.",
            UserWarning, stacklevel=3,
        )
    duration = df["duration"].dropna()
    duration = duration[duration > 0]
    if len(duration) and duration.median() > SUSPECT_DURATION_MEDIAN_H:
        warnings.warn(
            f"The median duration is {duration.median():,.0f}, above "
            f"{SUSPECT_DURATION_MEDIAN_H:.0f}, which is implausible for charging sessions in "
            "hours: the column is probably in minutes (or seconds). GEARS expects hours; "
            f"divide it by 60 before loading. Values >= {MAX_DURATION_H:.0f} are dropped "
            "as outliers.",
            UserWarning, stacklevel=3,
        )


def _describe_dropped(counts: dict[str, int], n_dropped: int, n_total: int) -> str:
    """One sentence listing why rows were dropped (only the non-zero reasons)."""
    reasons = ", ".join(f"{n:,} {label}" for label, n in counts.items() if n)
    return (
        f"Dropped {n_dropped:,}/{n_total:,} rows failing the quality filters: {reasons}. "
        "A row can fail several checks, so the counts may add up to more than the total."
    )


def validate_dataframe(
    df: pd.DataFrame,
    strict: bool = False,
    filter_failed: bool = True,
    dayfirst: bool | None = None,
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
        still missing after alias resolution (the message lists the accepted
        names), if the dataset is empty or every row is dropped by the quality
        filters, if ``arrival_time`` holds numbers instead of dates, or if
        ``strict=True`` and any rows fail the quality filters.

    Warns
    -----
    UserWarning
        Rows dropped (with the count per reason); a time zone dropped from
        ``arrival_time``; a median energy or duration that suggests Wh or minutes.

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


    df = _drop_noise_columns(df)
    df = _resolve_aliases(df)
    df = _convert_minute_duration_columns(df)

    if dayfirst is None and "arrival_time" in df.columns:
        order_columns = [df["arrival_time"]] + ([df["end_time"]] if "end_time" in df.columns else [])
        dayfirst = _date_order(*order_columns)
        if dayfirst is None and any(_has_ambiguous_dates(c) for c in order_columns):
            warnings.warn(
                "The dates are written as numbers in which no value settles whether the "
                "day or the month comes first (e.g. 05/01/2024): they are read month first. "
                "If they are day/month/year, pass dayfirst=True; to silence this, pass "
                "dayfirst=False or write dates as 2024-01-05.",
                UserWarning, stacklevel=3,
            )

    if "duration" not in df.columns and "end_time" in df.columns and "arrival_time" in df.columns:
        at = pd.to_datetime(df["arrival_time"], format="mixed", dayfirst=bool(dayfirst))
        et = pd.to_datetime(df["end_time"], format="mixed", dayfirst=bool(dayfirst))
        df["duration"] = (et - at).dt.total_seconds() / 3600.0

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise _missing_columns_error(df, missing)

    if len(df) == 0:
        raise ValueError(
            "The dataset has no rows (0 sessions). Check the file you loaded and any "
            "filter applied before this call."
        )

    df["arrival_time"], tz_dropped = _parse_arrival_time(df["arrival_time"], dayfirst)
    if tz_dropped:
        warnings.warn(
            "arrival_time carries a time zone or UTC offset. GEARS works with local "
            "wall-clock time: the zone was dropped and the clock reading kept, not "
            "converted. If these timestamps are in UTC, convert them to the site's local "
            "time first (e.g. df['arrival_time'].dt.tz_convert('Europe/Paris')).",
            UserWarning, stacklevel=3,
        )
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce")
    df["energy"] = pd.to_numeric(df["energy"], errors="coerce")
    if "power" in df.columns:
        df["power"] = pd.to_numeric(df["power"], errors="coerce")

    _warn_if_suspicious_units(df)

    n_before = len(df)
    time_ok = df["arrival_time"].notna()
    dur, ene = df["duration"], df["energy"]
    ok = (
        time_ok & dur.notna() & (dur > 0) & (dur < MAX_DURATION_H)
        & ene.notna() & (ene >= 0) & (ene < MAX_ENERGY_KWH)
    )
    n_duplicates = int(df[ok].duplicated().sum())
    counts = {
        "with a missing or unreadable arrival_time": int((~time_ok).sum()),
        "with a missing duration": int(dur.isna().sum()),
        "with duration <= 0": int((dur <= 0).sum()),
        f"with duration >= {MAX_DURATION_H:.0f} h": int((dur >= MAX_DURATION_H).sum()),
        "with a missing energy": int(ene.isna().sum()),
        "with negative energy": int((ene < 0).sum()),
        f"with energy >= {MAX_ENERGY_KWH:.0f} kWh": int((ene >= MAX_ENERGY_KWH).sum()),
        "duplicating an earlier row": n_duplicates,
    }
    df = df[ok].drop_duplicates().reset_index(drop=True)
    n_dropped = n_before - len(df)
    if len(df) == 0:
        raise ValueError(
            _describe_dropped(counts, n_dropped, n_before)
            + " Nothing is left. Check that duration is in hours, energy in kWh and "
            "arrival_time holds timestamps (see the unit warnings above, if any)."
        )
    if n_dropped > 0:
        msg = _describe_dropped(counts, n_dropped, n_before)
        if strict:
            raise ValueError(msg)
        warnings.warn(msg, UserWarning, stacklevel=3)

    df["hour"] = df["arrival_time"].dt.hour + df["arrival_time"].dt.minute / 60.0
    df["day_of_week"] = df["arrival_time"].dt.day_of_week   # 0=Monday, 6=Sunday
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

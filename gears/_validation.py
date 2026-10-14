"""
Argument checks shared by the simulators and the forecasters.

Every message says what is wrong, what was received and what to do, so a bad
call fails here with a clear ``ValueError`` instead of deeper in pandas / numpy.

Time convention
---------------
GEARS works with **local wall-clock time**: ``arrival_time`` is the clock reading
at the charging site, and simulated ``arrival_time`` values are naive timestamps
in that same local time. A time zone or UTC offset carried by an input is dropped
and the clock reading kept (``2025-06-10 08:00+02:00`` becomes
``2025-06-10 08:00``); it is **not** converted. If your timestamps are in UTC,
convert them to the site's local time first (``ts.dt.tz_convert("Europe/Paris")``).
"""

from __future__ import annotations

import numbers

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError


def check_int(value, name: str, *, minimum: int = 1, meaning: str = "") -> int:
    """
    Return ``value`` as an ``int`` if it is an integer ``>= minimum``.

    Raises
    ------
    ValueError
        If ``value`` is not an integer (floats such as ``2.5`` and booleans are
        rejected) or is smaller than ``minimum``.
    """
    what = f" ({meaning})" if meaning else ""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        # ValueError (not TypeError) on purpose: 2.5 or True is a bad *value* for a
        # count, and callers then need to catch a single exception type.
        raise ValueError(  # noqa: TRY004
            f"{name} must be an integer >= {minimum}{what}, "
            f"got {value!r} of type {type(value).__name__}."
        )
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}{what}, got {value}.")
    return int(value)


def check_float(
    value,
    name: str,
    *,
    gt: float | None = None,
    ge: float | None = None,
    meaning: str = "",
) -> float:
    """
    Return ``value`` as a finite ``float`` satisfying ``> gt`` (or ``>= ge``).

    Raises
    ------
    ValueError
        If ``value`` is not a real number, is NaN or infinite, or violates the bound.
    """
    what = f" ({meaning})" if meaning else ""
    bound = f"> {gt}" if gt is not None else (f">= {ge}" if ge is not None else "")
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not np.isfinite(value):
        raise ValueError(
            f"{name} must be a finite number{' ' + bound if bound else ''}{what}, "
            f"got {value!r}."
        )
    if (gt is not None and value <= gt) or (ge is not None and value < ge):
        raise ValueError(f"{name} must be {bound}{what}, got {value}.")
    return float(value)


def as_naive_timestamp(value, name: str = "start_date") -> pd.Timestamp:
    """
    Parse a date or timestamp into a naive :class:`pandas.Timestamp`.

    A time zone is dropped and the clock reading kept (see the module docstring).

    Raises
    ------
    ValueError
        If ``value`` is missing or cannot be read as a date.
    """
    try:
        ts = pd.Timestamp(value)
    except (ValueError, TypeError, OverflowError) as e:
        raise ValueError(
            f"{name}={value!r} is not a valid date or timestamp ({e}). "
            "Use an ISO date such as '2025-06-10' or a pandas Timestamp."
        ) from e
    if pd.isna(ts):
        raise ValueError(
            f"{name} is missing (got {value!r}). "
            "Pass an ISO date such as '2025-06-10' or a pandas Timestamp."
        )
    return ts.tz_localize(None) if ts.tzinfo is not None else ts


def not_fitted_error(obj_name: str, how_to_fit: str) -> NotFittedError:
    """Build the ``NotFittedError`` raised when an unfitted object is used."""
    return NotFittedError(
        f"This {obj_name} is not fitted yet, so it cannot be used. {how_to_fit}"
    )

"""
Shared windowing utility for the persistence-bootstrap vs. GMM benchmark.

Used by both :class:`gears.models.persistence_sampler.PersistenceSessionSampler`
and a windowed :class:`gears.models.gmm.EVSessionGMM` fit, so that the exact
same historical pool feeds both arms of the comparison (see Section 1,
assumption 2, of the benchmark prompt: "Pour le GMM il faudra prendre les
mêmes historiques.").
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd


def sessions_in_last_n_occurrences(
    df: pd.DataFrame,
    target_date: pd.Timestamp,
    n: int,
) -> Tuple[pd.DataFrame, dict]:
    """
    Pool sessions from the ``n`` most recent occurrences of ``target_date``'s
    weekday, strictly before ``target_date`` (no leakage).

    Parameters
    ----------
    df : pd.DataFrame
        Sessions data, already restricted to what should be "visible" at the
        evaluation origin (i.e. the caller is responsible for passing
        ``train_df`` -- data with ``arrival_time <= origin`` -- not the full
        dataset). Must contain an ``arrival_time`` column.
    target_date : pd.Timestamp
        The day being forecast. Only occurrences of the same weekday
        strictly before this date are pooled.
    n : int
        Requested history depth (number of most recent same-weekday
        occurrences to pool).

    Returns
    -------
    pool : pd.DataFrame
        Subset of ``df`` whose ``arrival_time`` falls on one of the ``n``
        (or fewer, if not enough exist) most recent qualifying dates.
    info : dict
        Diagnostics distinguishing the two possible skip reasons, so a
        benchmark harness can log them separately rather than merging or
        silently degrading:

        - ``n_requested`` : the requested ``n``.
        - ``n_available_occurrences`` : how many same-weekday dates before
          ``target_date`` actually exist in ``df``.
        - ``n_sessions`` : total sessions pooled across those dates.
        - ``insufficient_history`` : True if fewer than ``n`` qualifying
          calendar occurrences exist at all (a *calendar* problem -- distinct
          from ``n_sessions`` being too thin, which is a *volume* problem the
          caller should check separately against its own minimum-sample
          gate).
    """
    target_date = pd.Timestamp(target_date)
    weekday = target_date.dayofweek
    normalized = df["arrival_time"].dt.normalize()
    candidate_dates = sorted(
        d for d in normalized.unique()
        if pd.Timestamp(d) < target_date.normalize() and pd.Timestamp(d).dayofweek == weekday
    )
    kept_dates = candidate_dates[-n:] if n > 0 else []
    pool = df[normalized.isin(kept_dates)]
    info = {
        "n_requested": n,
        "n_available_occurrences": len(kept_dates),
        "n_sessions": len(pool),
        "insufficient_history": len(kept_dates) < n,
    }
    return pool, info

"""
Non-parametric persistence-bootstrap session generator.

Alternative to :class:`gears.models.session_model.EVSessionModel`: instead of fitting a
parametric Gaussian mixture on a pool of historical sessions, this simply
bootstrap-resamples (with replacement) actual session records from that pool.

This is deliberately a *different* abstraction from the existing count-level
:class:`gears.models.forecaster.PersistenceForecaster`, which forecasts the
*number* of sessions on a target day from a single same-weekday reference
point (with fallback search), not a pooled window. That class is untouched
and keeps doing its own job elsewhere in the package (e.g. the SARIMA
end-to-end comparison). ``PersistenceSessionSampler`` instead generates
*synthetic session records* (arrival hour, duration, energy) by resampling
from an already-windowed pool -- it does no date filtering or count
forecasting of its own.

Mirrors :meth:`EVSessionModel.fit`/:meth:`EVSessionModel.sample`'s public surface
closely enough to be a drop-in alternative wherever an ``EVSessionModel``
instance is currently expected (``distribution_comparison``,
``ShortTermSimulator``, etc.).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class PersistenceSessionSampler:
    """
    Non-parametric alternative to EVSessionModel: generates synthetic sessions by
    bootstrap-resampling (with replacement) from a pool of historical sessions,
    instead of fitting a parametric mixture model on the same pool.

    Mirrors EVSessionModel's fit()/sample() surface so it can be swapped in wherever
    an EVSessionModel is expected. Does NOT do any date filtering itself -- the caller
    (the benchmark harness, via the shared windowing utility in
    ``gears.evaluation.windowing``) is responsible for handing it an
    already-windowed pool, so the exact same pool can be reused for both this
    class and a windowed GMM fit.

    Parameters
    ----------
    random_state : int
        Default random seed used when ``sample()`` is called without an
        explicit ``seed``.

    Attributes
    ----------
    pool_ : pd.DataFrame or None
        The historical pool this sampler was fitted on (``arrival_hour``,
        ``duration``, ``energy`` columns only).
    is_fitted_ : bool
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.pool_: pd.DataFrame | None = None
        self.is_fitted_: bool = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> PersistenceSessionSampler:
        """
        Store the historical pool to bootstrap-resample from.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain ``arrival_hour``, ``duration``, and ``energy``
            columns (extra columns are dropped). This is the same
            already-windowed pool that would be handed to a windowed
            ``EVSessionModel.fit()`` call for a controlled comparison.

        Returns
        -------
        PersistenceSessionSampler
            ``self``, allowing method chaining.

        Raises
        ------
        ValueError
            If required columns are missing.
        RuntimeError
            If the pool is empty.
        """
        required = {"arrival_hour", "duration", "energy"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns for PersistenceSessionSampler: {missing}")
        if len(df) == 0:
            raise RuntimeError("Empty pool: cannot fit PersistenceSessionSampler.")
        self.pool_ = df[list(required)].reset_index(drop=True)
        self.is_fitted_ = True
        return self

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(
        self,
        n_sessions: int,
        context: dict | None = None,
        date: str | pd.Timestamp | None = None,
        seed: int | None = None,
    ) -> pd.DataFrame:
        """
        Generate synthetic EV sessions by bootstrap-resampling the fitted pool.

        Signature matches :meth:`EVSessionModel.sample` for interface parity.

        Parameters
        ----------
        n_sessions : int
            Number of sessions to generate (may exceed the pool size --
            resampling is with replacement).
        context : dict, optional
            Accepted but ignored -- the pool is already context-specific by
            construction (decided at ``fit()`` time by the caller).
        date : str or Timestamp, optional
            If given, an ``arrival_time`` column is added anchored to this
            date, exactly as ``EVSessionModel.sample`` does.
        seed : int, optional
            Random seed. Falls back to ``self.random_state`` if not given.

        Returns
        -------
        pd.DataFrame
            Columns: ``arrival_hour``, ``duration``, ``energy``,
            [``arrival_time``] -- the same column set as
            ``EVSessionModel.sample()`` for the same call shape.

        Raises
        ------
        RuntimeError
            If called before ``fit()``.
        """
        if not self.is_fitted_:
            raise RuntimeError("Call fit() before sample().")
        rng = np.random.default_rng(seed if seed is not None else self.random_state)
        idx = rng.integers(0, len(self.pool_), size=n_sessions)
        sessions = self.pool_.iloc[idx].reset_index(drop=True)
        if date is not None:
            ref = pd.Timestamp(date)
            td = pd.to_timedelta(sessions["arrival_hour"], unit="h")
            sessions = sessions.assign(arrival_time=ref.normalize() + td)
        return sessions

    def __repr__(self) -> str:
        if not self.is_fitted_:
            return f"PersistenceSessionSampler(random_state={self.random_state}, fitted=False)"
        return (
            f"PersistenceSessionSampler(random_state={self.random_state}, "
            f"pool_size={len(self.pool_)}, fitted=True)"
        )

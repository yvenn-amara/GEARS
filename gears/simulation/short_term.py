"""
Short-term EV session simulator.

Generates individual charging sessions for a horizon of 1–N days,
using the fitted GMM for session properties (arrival_hour, duration, energy)
and the SessionForecaster for daily session counts.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from gears._validation import as_naive_timestamp, check_int
from gears.models.forecaster import SessionForecaster
from gears.models.session_model import EVSessionModel

logger = logging.getLogger(__name__)


class ShortTermSimulator:
    """
    Day-level EV session simulator.

    Parameters
    ----------
    gmm : EVSessionModel
        Fitted GMM for sampling session properties.
    forecaster : SessionForecaster, optional
        Fitted count forecaster.  When None, each day falls back to a
        hard-coded default of 20 sessions.
    charger_mix : dict, optional
        Power distribution ``{power_kw: proportion}``.  Proportions are
        normalised automatically.  Default: ``{7.4: 0.5, 22.0: 0.5}``.
    resolution_min : int
        Time resolution in minutes for load-curve output.

    Examples
    --------
    >>> sim = ShortTermSimulator(gmm, forecaster)
    >>> sessions = sim.simulate(start_date="2025-06-10", horizon=7)
    """

    def __init__(
        self,
        gmm: EVSessionModel,
        forecaster: SessionForecaster | None = None,
        charger_mix: dict[float, float] | None = None,
        resolution_min: int = 30,
    ):
        self.gmm = gmm
        self.forecaster = forecaster
        self.resolution_min = resolution_min

        self.charger_mix = dict(charger_mix) if charger_mix else {7.4: 0.5, 22.0: 0.5}
        total = sum(self.charger_mix.values())
        self.charger_mix = {k: v / total for k, v in self.charger_mix.items()}

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate(
        self,
        start_date: str | pd.Timestamp,
        horizon: int = 7,
        n_scenarios: int = 10,
        seed: int | None = None,
        strict: bool = False,
    ) -> pd.DataFrame:
        """
        Simulate sessions over a multi-day horizon across N scenarios.

        Parameters
        ----------
        start_date : str or Timestamp
            First day of the simulation, read as local wall-clock time (a time
            zone is dropped, not converted).
        horizon : int
            Number of days (an integer >= 1).
        n_scenarios : int
            Number of stochastic scenarios (an integer >= 1).
        seed : int, optional
            Master random seed.
        strict : bool, default False
            Raise ``ValueError`` instead of falling back to the closest fitted
            context when the session model has no mixture for a day's context.

        Returns
        -------
        pd.DataFrame
            One row per session with columns:
            date, scenario, arrival_time, arrival_hour,
            duration, energy, power_kw.

        Raises
        ------
        ValueError
            If ``horizon`` or ``n_scenarios`` is not an integer >= 1, or
            ``start_date`` is not a valid date.
        """
        horizon = check_int(horizon, "horizon", minimum=1, meaning="number of days to simulate")
        n_scenarios = check_int(
            n_scenarios, "n_scenarios", minimum=1, meaning="number of stochastic scenarios"
        )
        start = as_naive_timestamp(start_date, "start_date")
        rng = np.random.default_rng(seed)
        dates = pd.date_range(start, periods=horizon, freq="D")

        # Forecast daily counts for all dates
        if self.forecaster is not None and self.forecaster.is_fitted_:
            fc_df = self.forecaster.predict(
                horizon=horizon,
                n_scenarios=n_scenarios,
                start_date=start,
                seed=int(rng.integers(0, 2**31)),
            )
            counts_pivot = fc_df.pivot(index="date", columns="scenario", values="n_sessions")
            # Normalise the index to pd.Timestamp so that date membership checks
            # work regardless of whether the forecaster returned datetime.date,
            # datetime64[s], or datetime64[ns] objects.
            counts_pivot.index = pd.to_datetime(counts_pivot.index).normalize()
        else:
            counts_pivot = None

        frames = []
        for sc in range(n_scenarios):
            sc_seed = int(rng.integers(0, 2**31))
            sc_rng = np.random.default_rng(sc_seed)

            for date in dates:
                # Compare normalised Timestamps to ensure consistent matching
                date_key = date.normalize()
                if counts_pivot is not None and date_key in counts_pivot.index:
                    n_today = max(0, int(counts_pivot.loc[date_key, sc]))
                else:
                    # Forecaster absent or date outside the forecast window;
                    # use a conservative default so the simulation can proceed.
                    n_today = 20

                if n_today == 0:
                    continue

                day_sessions = self.simulate_single_day(
                    date=date,
                    n_sessions=n_today,
                    seed=int(sc_rng.integers(0, 2**31)),
                    strict=strict,
                )
                day_sessions["scenario"] = sc
                frames.append(day_sessions)

        if not frames:
            return pd.DataFrame()

        result = pd.concat(frames, ignore_index=True)
        return result

    def simulate_single_day(
        self,
        date: str | pd.Timestamp,
        n_sessions: int = 20,
        seed: int | None = None,
        context: dict | None = None,
        strict: bool = False,
    ) -> pd.DataFrame:
        """
        Simulate sessions for a single day.

        Parameters
        ----------
        date : str or Timestamp
            Target date; used to infer the GMM context and to anchor
            ``arrival_time`` values.
        n_sessions : int
            Number of sessions to generate.
        seed : int, optional
            Random seed passed to the GMM sampler.
        context : dict, optional
            Explicit GMM context (overrides date-derived context).
        strict : bool, default False
            Raise ``ValueError`` if the session model has no mixture for the
            context, instead of using the closest fitted one.

        Returns
        -------
        pd.DataFrame
            Columns: arrival_hour, duration, energy, arrival_time, date, power_kw.

        Raises
        ------
        ValueError
            If ``n_sessions`` is not an integer >= 1 or ``date`` is not valid.
        """
        n_sessions = check_int(n_sessions, "n_sessions", minimum=1)
        date = as_naive_timestamp(date, "date")
        rng = np.random.default_rng(seed)
        sessions = self.gmm.sample(
            n_sessions=n_sessions,
            context=context,
            date=date,
            seed=int(rng.integers(0, 2**31)),
            strict=strict,
        )
        sessions["date"] = date.date()
        sessions["power_kw"] = self._assign_power(n_sessions, rng)

        # Cap energy by power × duration to enforce the physical constraint
        # that a charger cannot deliver more energy than its rated power allows
        # over the session duration.
        sessions["energy"] = np.minimum(
            sessions["energy"],
            sessions["power_kw"] * sessions["duration"],
        )
        return sessions

    def _assign_power(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Sample a charger power level for each session from the charger mix.

        Parameters
        ----------
        n : int
            Number of sessions to assign power levels to.
        rng : np.random.Generator
            Seeded random generator for reproducibility.

        Returns
        -------
        np.ndarray
            Array of shape ``(n,)`` with power values in kW drawn according
            to the proportions defined in ``self.charger_mix``.
        """
        powers = list(self.charger_mix.keys())
        weights = np.array(list(self.charger_mix.values()), dtype=float)
        return rng.choice(powers, size=n, p=weights)

    def __repr__(self) -> str:
        return (
            f"ShortTermSimulator("
            f"charger_mix={self.charger_mix}, "
            f"has_forecaster={self.forecaster is not None})"
        )

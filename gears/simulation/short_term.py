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
    ) -> pd.DataFrame:
        """
        Simulate sessions over a multi-day horizon across N scenarios.

        Parameters
        ----------
        start_date : str or Timestamp
            First day of the simulation.
        horizon : int
            Number of days.
        n_scenarios : int
            Number of stochastic scenarios.
        seed : int, optional
            Master random seed.

        Returns
        -------
        pd.DataFrame
            One row per session with columns:
            date, scenario, arrival_time, arrival_hour,
            duration, energy, power_kw.
        """
        rng = np.random.default_rng(seed)
        start = pd.Timestamp(start_date)
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

        Returns
        -------
        pd.DataFrame
            Columns: arrival_hour, duration, energy, arrival_time, date, power_kw.
        """
        rng = np.random.default_rng(seed)
        sessions = self.gmm.sample(
            n_sessions=n_sessions,
            context=context,
            date=date,
            seed=int(rng.integers(0, 2**31)),
        )
        sessions["date"] = pd.Timestamp(date).date()
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

    # ------------------------------------------------------------------
    # Load-curve computation
    # ------------------------------------------------------------------

    def compute_load_curve(
        self,
        sessions: pd.DataFrame,
        date: str | pd.Timestamp | None = None,
        scenario: int = 0,
    ) -> pd.Series:
        """
        Compute a time-step load curve (kW) from simulated sessions.

        Parameters
        ----------
        sessions : pd.DataFrame
            Output of :meth:`simulate` or :meth:`simulate_single_day`.
        date : str or Timestamp, optional
            Filter to a specific date.
        scenario : int
            Scenario index to select when *sessions* contains multiple scenarios.

        Returns
        -------
        pd.Series
            Instantaneous aggregate load in kW, indexed by a
            ``DatetimeIndex`` at ``resolution_min`` intervals.
            Returns an empty Series when no sessions match the filters.
        """
        df = sessions.copy()
        if "scenario" in df.columns:
            df = df[df["scenario"] == scenario]
        if date is not None:
            target = pd.Timestamp(date).date()
            df = df[df["date"] == target]

        if df.empty:
            return pd.Series(dtype=float)

        ref_date = pd.Timestamp(df["date"].iloc[0])
        n_slots = 24 * 60 // self.resolution_min
        load = np.zeros(n_slots)

        for _, row in df.iterrows():
            if "arrival_time" in df.columns:
                start_h = pd.Timestamp(row["arrival_time"]).hour + \
                          pd.Timestamp(row["arrival_time"]).minute / 60.0
            else:
                start_h = float(row.get("arrival_hour", 8.0))

            end_h = start_h + float(row["duration"])
            start_slot = int(start_h * 60 / self.resolution_min)
            end_slot = int(min(end_h * 60 / self.resolution_min, n_slots))
            if start_slot < n_slots:
                load[start_slot:end_slot] += float(row["power_kw"])

        times = pd.date_range(ref_date, periods=n_slots, freq=f"{self.resolution_min}min")
        return pd.Series(load, index=times, name="load_kw")

    def __repr__(self) -> str:
        return (
            f"ShortTermSimulator("
            f"charger_mix={self.charger_mix}, "
            f"has_forecaster={self.forecaster is not None})"
        )

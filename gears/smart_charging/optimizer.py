"""
Smart charging optimiser for GEARS (V1G strategy).

Implements a time-shiftable greedy scheduler: given a price (or RES)
signal, each session is scheduled to charge at the cheapest available
time slots within its connection window, subject to power constraints.

This is a **V1G** (Vehicle-to-Grid, unidirectional) strategy — the
charger can shift load in time but **cannot** feed energy back to the
grid (no V2G).  The optimiser therefore only moves charging demand
forward or backward within each vehicle's connection window.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from gears._validation import check_float, check_int

logger = logging.getLogger(__name__)


def _as_ns(ts: pd.Timestamp) -> np.datetime64:
    """Naive ``datetime64[ns]`` of a Timestamp, keeping nanoseconds (``np.datetime64(ts)`` drops them)."""
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts.to_datetime64().astype("datetime64[ns]")


class SmartChargingOptimizer:
    """
    Greedy V1G smart charging scheduler.

    The signal is cut into slots of ``resolution_min`` minutes. For each session
    the algorithm:

    1. Takes the slots that overlap the connection window ``[arrival, departure]``;
       a slot that is only partly inside the window offers only the time inside it.
    2. Sorts them by signal (ascending for price, descending for RES).
    3. Charges at the charger's power in the best slots first, until the session's
       energy is delivered; the last slot used may be only partly filled.

    The baseline "plug-and-charge" fills the same slots in chronological order,
    starting at arrival. Costs are signal x energy, slot by slot, so a session is
    never billed for energy it does not receive.

    This is a **V1G** strategy: load can only be *shifted*, not injected
    back to the grid.

    Parameters
    ----------
    signal_type : str
        ``'price'`` – minimise cost (€); ``'res'`` – maximise renewable use.
    resolution_min : int
        Signal time resolution in minutes (must match the signal's step).
    efficiency : float
        Share of the energy drawn from the charger that reaches the battery,
        in ]0, 1]. Default 1.0: a session's ``energy`` is the energy delivered
        by the charger (as metered, and as simulated by GEARS), which already
        includes the losses. Use e.g. 0.9 to add losses on top: the charger then
        has to draw ``energy / efficiency``, and sessions for which that exceeds
        ``power_kw x duration`` cannot be completed (they are reported).
    default_power_kw : float
        Fallback charger power if ``power_kw`` column is absent in the
        sessions DataFrame. Default 7.4 kW.

    Examples
    --------
    >>> opt = SmartChargingOptimizer(signal_type='price')
    >>> result = opt.optimise(sessions, price_signal)
    >>> summary = opt.savings_summary(result)
    """

    def __init__(
        self,
        signal_type: str = "price",
        resolution_min: int = 30,
        efficiency: float = 1.0,
        default_power_kw: float = 7.4,
    ):
        if signal_type not in ("price", "res"):
            raise ValueError(f"signal_type must be 'price' or 'res', got '{signal_type}'")
        self.signal_type = signal_type
        self.resolution_min = check_int(
            resolution_min, "resolution_min", minimum=1, meaning="minutes per signal slot"
        )
        self.efficiency = check_float(
            efficiency, "efficiency", gt=0, meaning="charging efficiency, in ]0, 1]"
        )
        if self.efficiency > 1:
            raise ValueError(
                f"efficiency must be <= 1 (charging efficiency, in ]0, 1]), got {efficiency}."
            )
        self.default_power_kw = check_float(
            default_power_kw, "default_power_kw", gt=0, meaning="charger power in kW"
        )

    # ------------------------------------------------------------------
    # Main optimisation
    # ------------------------------------------------------------------

    def optimise(
        self,
        sessions: pd.DataFrame,
        signal: pd.Series,
    ) -> pd.DataFrame:
        """
        Schedule sessions against a price/RES signal.

        Parameters
        ----------
        sessions : pd.DataFrame
            Sessions with columns:

            - ``arrival_time`` (``datetime.date``, ``str``, or
              ``pd.Timestamp``) **or** ``date`` + ``arrival_hour`` (float).
            - ``duration`` (hours, float).
            - ``energy`` (kWh, float): energy delivered by the charger.
            - ``power_kw`` (kW, float, optional – defaults to
              ``self.default_power_kw``).

            The ``date`` column (when present) may be a ``datetime.date``
            object, a date string, or a ``pd.Timestamp``; all are coerced
            to ``datetime.date`` internally.
        signal : pd.Series
            Indexed by datetime (slot start) at ``resolution_min``-minute intervals.
            Values: €/kWh (for ``signal_type='price'``) or fraction 0–1
            (for ``signal_type='res'``).

        Returns
        -------
        pd.DataFrame
            Original sessions with additional columns:

            - ``cost_smart`` (€): cost when charging in the best slots (for a
              RES signal: green-weighted energy, higher is better).
            - ``cost_plug`` (€): cost under plug-and-charge (first slots).
            - ``savings_pct`` (%): relative saving vs. plug-and-charge.
            - ``scheduled_start`` (``pd.Timestamp``): start of the first
              charging interval of the optimised plan (never before arrival).
            - ``scheduled_end`` (``pd.Timestamp``): end of the last charging
              interval (never after departure).
            - ``energy_scheduled`` (kWh): energy that reaches the vehicle in the
              plan. Equal to ``energy`` unless the session cannot be completed
              (see the warning), then smaller.

        Raises
        ------
        TypeError
            If ``signal`` is not indexed by timestamps.
        ValueError
            If ``power_kw`` holds a value <= 0.

        Warns
        -----
        UserWarning
            For sessions that cannot be scheduled (no usable arrival, energy or
            duration; no signal during the window) and for sessions that cannot
            be completed; ``resolution_min`` not matching the signal's step.
        """
        sessions = sessions.copy().reset_index(drop=True)
        # Remove any output columns left over from a prior optimise() call.
        _output_cols = ["cost_smart", "cost_plug", "savings_pct",
                        "scheduled_start", "scheduled_end", "energy_scheduled"]
        sessions.drop(columns=[c for c in _output_cols if c in sessions.columns],
                      inplace=True)

        if not isinstance(signal.index, pd.DatetimeIndex):
            raise TypeError(
                "signal must be indexed by timestamps (one value per slot start), got an "
                f"index of type {type(signal.index).__name__}. For example "
                "gears.utils.make_price_signal(...) builds one."
            )
        signal = signal.sort_index()
        if "power_kw" in sessions.columns and (sessions["power_kw"] <= 0).any():
            raise ValueError(
                f"power_kw must be > 0, but {int((sessions['power_kw'] <= 0).sum())} session(s) "
                "have 0 or a negative charger power."
            )

        step = np.timedelta64(self.resolution_min, "m")
        sig_index = signal.index.tz_localize(None) if signal.index.tz is not None else signal.index
        sig_t = sig_index.to_numpy("datetime64[ns]")
        sig_v = signal.to_numpy(dtype=float)
        self._warn_if_step_differs(sig_t)

        if "date" in sessions.columns:
            sessions["date"] = pd.to_datetime(sessions["date"]).dt.date

        # Build arrival_time from (date, arrival_hour) if not already present.
        if "arrival_time" not in sessions.columns and "arrival_hour" in sessions.columns:
            sessions["arrival_time"] = (
                pd.to_datetime(sessions["date"].astype(str))
                + pd.to_timedelta(sessions["arrival_hour"].astype(float), unit="h")
            )
        elif "arrival_time" in sessions.columns:
            sessions["arrival_time"] = pd.to_datetime(sessions["arrival_time"])

        n = len(sessions)
        cost_smart = np.full(n, np.nan)
        cost_plug = np.full(n, np.nan)
        delivered = np.full(n, np.nan)
        sched_start = np.full(n, np.datetime64("NaT", "ns"))
        sched_end = np.full(n, np.datetime64("NaT", "ns"))

        n_unusable = n_no_signal = n_truncated = 0
        for idx, row in sessions.iterrows():
            arrival_dt, departure_dt = self._get_window(row)
            energy_kwh = float(row["energy"]) if pd.notna(row.get("energy")) else np.nan
            if (
                arrival_dt is None or pd.isna(arrival_dt) or pd.isna(departure_dt)
                or departure_dt <= arrival_dt or not np.isfinite(energy_kwh) or energy_kwh < 0
            ):
                n_unusable += 1
                continue

            arrival = _as_ns(arrival_dt)
            departure = _as_ns(departure_dt)
            # Slots overlapping [arrival, departure): start < departure and start + step > arrival.
            i0 = int(np.searchsorted(sig_t, arrival - step, side="right"))
            i1 = int(np.searchsorted(sig_t, departure, side="left"))
            if i1 <= i0:
                n_no_signal += 1
                continue

            starts = sig_t[i0:i1]
            prices = sig_v[i0:i1]
            lo = np.maximum(starts, arrival)
            hours = (np.minimum(starts + step, departure) - lo) / np.timedelta64(1, "h")
            power_kw = float(row["power_kw"]) if "power_kw" in row.index and pd.notna(row.get("power_kw")) \
                else self.default_power_kw
            capacity = power_kw * hours                      # kWh the charger can give in each slot
            needed = energy_kwh / self.efficiency            # kWh the charger must draw
            if needed > capacity.sum() * (1 + 1e-9):
                n_truncated += 1

            plug = self._fill(np.arange(len(starts)), capacity, needed)
            best_first = np.argsort(prices if self.signal_type == "price" else -prices, kind="stable")
            smart = self._fill(best_first, capacity, needed)

            cost_plug[idx] = float((prices * plug).sum())
            cost_smart[idx] = float((prices * smart).sum())
            delivered[idx] = float(smart.sum() * self.efficiency)

            used = smart > 0
            if used.any():
                busy = (smart[used] / power_kw * 3.6e12).astype("int64").astype("timedelta64[ns]")
                sched_start[idx] = lo[used].min()
                sched_end[idx] = (lo[used] + busy).max()
            else:                                            # nothing to charge
                sched_start[idx] = sched_end[idx] = arrival

        sessions["cost_smart"] = cost_smart
        sessions["cost_plug"] = cost_plug
        sessions["scheduled_start"] = sched_start
        sessions["scheduled_end"] = sched_end
        sessions["energy_scheduled"] = delivered

        self._warn_unscheduled(n, n_unusable, n_no_signal, n_truncated, signal)

        mask = sessions["cost_plug"].notna() & (sessions["cost_plug"] > 0)
        sessions["savings_pct"] = np.nan
        sessions.loc[mask, "savings_pct"] = (
            (sessions.loc[mask, "cost_plug"] - sessions.loc[mask, "cost_smart"])
            / sessions.loc[mask, "cost_plug"] * 100
        )
        return sessions

    @staticmethod
    def _fill(order: np.ndarray, capacity: np.ndarray, needed: float) -> np.ndarray:
        """Energy per slot when slots are filled, in ``order``, up to their capacity."""
        cap = capacity[order]
        before = np.cumsum(cap) - cap
        filled = np.zeros_like(capacity)
        filled[order] = np.clip(needed - before, 0.0, cap)
        return filled

    def _warn_if_step_differs(self, sig_t: np.ndarray) -> None:
        """Warn when the signal's step is not ``resolution_min``: slots would be wrong."""
        if len(sig_t) < 2:
            return
        step_min = float(np.median(np.diff(sig_t)) / np.timedelta64(1, "m"))
        if not np.isclose(step_min, self.resolution_min):
            warnings.warn(
                f"The signal has a {step_min:g}-minute step but the optimiser was built with "
                f"resolution_min={self.resolution_min}, so slots are mis-sized and the costs "
                f"are wrong. Build SmartChargingOptimizer(resolution_min={step_min:g}) or "
                "resample the signal.",
                UserWarning,
                stacklevel=3,
            )

    def _warn_unscheduled(
        self, n_total: int, n_unusable: int, n_no_signal: int, n_truncated: int,
        signal: pd.Series,
    ) -> None:
        """Announce sessions that were not scheduled, or not fully, instead of hiding them."""
        problems = []
        if n_unusable:
            problems.append(
                f"{n_unusable} have no usable arrival time, energy >= 0 or duration > 0"
            )
        if n_no_signal:
            span = (
                f"{signal.index.min()} to {signal.index.max()}" if len(signal) else "empty"
            )
            problems.append(
                f"{n_no_signal} have no signal value during their plug-in window "
                f"(the signal covers {span})"
            )
        if problems:
            warnings.warn(
                f"{n_unusable + n_no_signal} of {n_total} sessions were not optimised: "
                + "; ".join(problems) + ". Their costs are NaN. Pass a signal that covers "
                "all the sessions' dates and check the sessions' columns.",
                UserWarning,
                stacklevel=3,
            )
        if n_truncated:
            warnings.warn(
                f"{n_truncated} of {n_total} sessions cannot be completed: their energy"
                + (f" / efficiency ({self.efficiency})" if self.efficiency != 1 else "")
                + " exceeds power_kw x the time available in their plug-in window (or in "
                "the part of it that the signal covers). Only the energy that fits is "
                "scheduled (see the energy_scheduled column). Check the 'power_kw' and "
                "'duration' of these sessions.",
                UserWarning,
                stacklevel=3,
            )

    def _get_window(self, row: pd.Series):
        """
        Extract the connection window from a session row.

        Parameters
        ----------
        row : pd.Series
            A single session row, either with ``arrival_time`` or with
            ``date`` + ``arrival_hour`` columns.

        Returns
        -------
        tuple[pd.Timestamp, pd.Timestamp] or tuple[None, None]
            ``(arrival_datetime, departure_datetime)``.  Returns
            ``(None, None)`` if the row does not contain enough
            information to determine the window.
        """
        if "arrival_time" in row.index and pd.notna(row.get("arrival_time")):
            arrival_dt = pd.Timestamp(row["arrival_time"])
        elif "date" in row.index and "arrival_hour" in row.index:
            arrival_dt = pd.Timestamp(str(row["date"])) + pd.to_timedelta(
                float(row["arrival_hour"]), unit="h"
            )
        else:
            return None, None

        duration_h = float(row.get("duration", 8.0))
        departure_dt = arrival_dt + pd.Timedelta(hours=duration_h)
        return arrival_dt, departure_dt

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def savings_summary(self, result: pd.DataFrame) -> dict:
        """
        Compute aggregate cost and savings statistics.

        Parameters
        ----------
        result : pd.DataFrame
            Output of :meth:`optimise`.

        Returns
        -------
        dict
            Keys:

            - ``total_cost_smart_eur`` (float): total charging cost with V1G.
            - ``total_cost_plug_eur`` (float): total cost under plug-and-charge.
            - ``total_savings_eur`` (float): absolute saving.
            - ``mean_savings_pct`` (float): average percentage saving per session.
            - ``n_sessions_optimised`` (int): sessions for which optimisation
              was possible (i.e. at least one feasible slot existed).

        Examples
        --------
        >>> opt = SmartChargingOptimizer()
        >>> result = opt.optimise(sessions, signal)
        >>> opt.savings_summary(result)
        {'total_cost_smart_eur': ..., 'total_savings_eur': ..., ...}
        """
        valid = result[result["cost_plug"].notna()].copy()
        if valid.empty:
            return {}
        return {
            "total_cost_smart_eur": round(valid["cost_smart"].sum(), 2),
            "total_cost_plug_eur": round(valid["cost_plug"].sum(), 2),
            "total_savings_eur": round(
                (valid["cost_plug"] - valid["cost_smart"]).sum(), 2
            ),
            "mean_savings_pct": round(valid["savings_pct"].mean(), 2),
            "n_sessions_optimised": len(valid),
        }

    # ------------------------------------------------------------------
    # Regret analysis
    # ------------------------------------------------------------------

    def compute_regret(
        self,
        oracle_sessions: pd.DataFrame,
        predicted_sessions: pd.DataFrame,
        signal: pd.Series,
        persistence_sessions: pd.DataFrame | None = None,
    ) -> dict:
        """
        Compute a four-way regret analysis across charging strategies.

        Computes costs for:

        - **Oracle**: real sessions + smart charging (best achievable).
        - **Predicted + Smart**: GEARS sessions + V1G optimisation.
        - **Predicted + Plug**: GEARS sessions + plug-and-charge (no V1G).
        - **Persistence + Smart** (optional): naive baseline sessions + V1G.

        Parameters
        ----------
        oracle_sessions : pd.DataFrame
            Real observed sessions (ground truth).
        predicted_sessions : pd.DataFrame
            Forecasted/simulated sessions from GEARS.
        signal : pd.Series
            Price or RES signal (must cover the sessions' date range).
        persistence_sessions : pd.DataFrame, optional
            Sessions from the persistence baseline.

        Returns
        -------
        dict
            Keys: ``cost_oracle_smart``, ``cost_predicted_smart``,
            ``cost_predicted_plug``, ``regret_smart_vs_oracle``,
            ``regret_plug_vs_oracle``, ``value_of_smart_charging``,
            and optionally ``cost_persistence_smart`` and
            ``value_of_forecast_vs_persistence`` (all in €, rounded to
            2 decimal places).
        """
        oracle_opt = self.optimise(oracle_sessions, signal)
        pred_opt = self.optimise(predicted_sessions, signal)

        cost_oracle = oracle_opt["cost_smart"].sum()
        cost_pred_smart = pred_opt["cost_smart"].sum()
        cost_pred_plug = pred_opt["cost_plug"].sum()

        result = {
            "cost_oracle_smart":     round(float(cost_oracle), 2),
            "cost_predicted_smart":  round(float(cost_pred_smart), 2),
            "cost_predicted_plug":   round(float(cost_pred_plug), 2),
            "regret_smart_vs_oracle": round(float(cost_pred_smart - cost_oracle), 2),
            "regret_plug_vs_oracle":  round(float(cost_pred_plug - cost_oracle), 2),
            "value_of_smart_charging": round(float(cost_pred_plug - cost_pred_smart), 2),
        }

        if persistence_sessions is not None:
            pers_opt = self.optimise(persistence_sessions, signal)
            cost_pers = pers_opt["cost_smart"].sum()
            result["cost_persistence_smart"] = round(float(cost_pers), 2)
            result["value_of_forecast_vs_persistence"] = round(
                float(cost_pers - cost_pred_smart), 2
            )

        return result

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
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SmartChargingOptimizer:
    """
    Greedy V1G smart charging scheduler.

    For each session the algorithm:

    1. Identifies the feasible charging window ``[arrival, departure]``.
    2. Sorts slots by signal (ascending for price, descending for RES).
    3. Greedily fills cheapest slots until the energy demand is met.

    This is a **V1G** strategy: load can only be *shifted*, not injected
    back to the grid.

    Parameters
    ----------
    signal_type : str
        ``'price'`` – minimise cost (€); ``'res'`` – maximise renewable use.
    resolution_min : int
        Signal time resolution in minutes.
    efficiency : float
        Charging efficiency (0–1). Default 0.9.
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
        efficiency: float = 0.9,
        default_power_kw: float = 7.4,
    ):
        if signal_type not in ("price", "res"):
            raise ValueError(f"signal_type must be 'price' or 'res', got '{signal_type}'")
        self.signal_type = signal_type
        self.resolution_min = resolution_min
        self.efficiency = efficiency
        self.default_power_kw = default_power_kw

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
            - ``energy`` (kWh, float).
            - ``power_kw`` (kW, float, optional – defaults to
              ``self.default_power_kw``).

            The ``date`` column (when present) may be a ``datetime.date``
            object, a date string, or a ``pd.Timestamp``; all are coerced
            to ``datetime.date`` internally.
        signal : pd.Series
            Indexed by datetime at ``resolution_min``-minute intervals.
            Values: €/kWh (for ``signal_type='price'``) or fraction 0–1
            (for ``signal_type='res'``).

        Returns
        -------
        pd.DataFrame
            Original sessions with four additional columns:

            - ``cost_smart`` (€): cost when charging at optimal slots.
            - ``cost_plug`` (€): cost under plug-and-charge (first slots).
            - ``savings_pct`` (%): relative saving vs. plug-and-charge.
            - ``scheduled_start`` (``pd.Timestamp``): start of optimised
              charging window.
            - ``scheduled_end`` (``pd.Timestamp``): end of optimised
              charging window.
        """
        sessions = sessions.copy().reset_index(drop=True)
        # Remove any output columns left over from a prior optimise() call.
        _output_cols = ["cost_smart", "cost_plug", "savings_pct",
                        "scheduled_start", "scheduled_end"]
        sessions.drop(columns=[c for c in _output_cols if c in sessions.columns],
                      inplace=True)
        signal = signal.sort_index()

        res_h = self.resolution_min / 60.0

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

        sessions["_cost_smart"] = np.nan
        sessions["_cost_plug"] = np.nan
        sessions["_scheduled_start"] = pd.NaT
        sessions["_scheduled_end"] = pd.NaT

        for idx, row in sessions.iterrows():
            arrival_dt, departure_dt = self._get_window(row)
            if arrival_dt is None:
                continue

            window_sig = signal[
                (signal.index >= arrival_dt) & (signal.index < departure_dt)
            ]
            if window_sig.empty:
                continue

            energy_kwh = float(row["energy"])
            power_kw = float(row["power_kw"]) if "power_kw" in row.index and pd.notna(row.get("power_kw")) \
                else self.default_power_kw
            energy_needed = energy_kwh / self.efficiency
            slots_needed = int(np.ceil(energy_needed / (power_kw * res_h)))
            slots_needed = min(slots_needed, len(window_sig))

            # Plug-and-charge baseline: charge at arrival, first available slots.
            plug_slots = window_sig.iloc[:slots_needed]
            cost_plug = float((plug_slots * power_kw * res_h).sum())

            # Smart charging: choose the cheapest (or greenest) slots first.
            if self.signal_type == "price":
                ordered = window_sig.nsmallest(slots_needed)
            else:
                ordered = window_sig.nlargest(slots_needed)

            cost_smart = float((ordered * power_kw * res_h).sum())
            scheduled_start = ordered.index.min() if not ordered.empty else arrival_dt
            scheduled_end = scheduled_start + pd.Timedelta(
                minutes=self.resolution_min * slots_needed
            )

            sessions.at[idx, "_cost_smart"] = cost_smart
            sessions.at[idx, "_cost_plug"] = cost_plug
            sessions.at[idx, "_scheduled_start"] = scheduled_start
            sessions.at[idx, "_scheduled_end"] = scheduled_end

        sessions.rename(columns={
            "_cost_smart": "cost_smart",
            "_cost_plug": "cost_plug",
            "_scheduled_start": "scheduled_start",
            "_scheduled_end": "scheduled_end",
        }, inplace=True)

        mask = sessions["cost_plug"].notna() & (sessions["cost_plug"] > 0)
        sessions.loc[mask, "savings_pct"] = (
            (sessions.loc[mask, "cost_plug"] - sessions.loc[mask, "cost_smart"])
            / sessions.loc[mask, "cost_plug"] * 100
        )
        return sessions

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
        persistence_sessions: Optional[pd.DataFrame] = None,
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

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_load_curve(
        self,
        sessions: pd.DataFrame,
        signal: Optional[pd.Series] = None,
        date: Optional[str] = None,
        ax=None,
        figsize: tuple = (14, 5),
    ):
        """
        Plot plug-and-charge vs. smart-charging load curves for a single day.

        Parameters
        ----------
        sessions : pd.DataFrame
            Output of :meth:`optimise` (must include ``scheduled_start``).
        signal : pd.Series, optional
            Price or RES signal. When provided, plotted on a secondary
            y-axis for visual correlation.
        date : str, optional
            Filter sessions to this date (``'YYYY-MM-DD'``). If None,
            all sessions are used (they should cover only one day).
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. A new figure is created if None.
        figsize : tuple
            Figure size (width, height) in inches.

        Returns
        -------
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt

        if date is not None:
            target = pd.Timestamp(date).date()
            sessions = sessions[
                pd.to_datetime(sessions["date"]).dt.date == target
            ].copy()

        if sessions.empty:
            raise ValueError("No sessions found for the given date.")

        # Ensure arrival_time exists for index arithmetic.
        if "arrival_time" not in sessions.columns and "arrival_hour" in sessions.columns:
            sessions["arrival_time"] = (
                pd.to_datetime(sessions["date"].astype(str))
                + pd.to_timedelta(sessions["arrival_hour"].astype(float), unit="h")
            )

        ref_date = pd.Timestamp(pd.to_datetime(sessions["date"].iloc[0]).date())
        n_slots = 24 * 60 // self.resolution_min
        res_h = self.resolution_min / 60.0

        plug_load = np.zeros(n_slots)
        smart_load = np.zeros(n_slots)

        for _, row in sessions.iterrows():
            if pd.isna(row.get("scheduled_start")):
                continue
            power = float(row.get("power_kw", self.default_power_kw))
            energy_needed = float(row["energy"]) / self.efficiency
            slots_needed = max(1, int(np.ceil(energy_needed / (power * res_h))))

            arrival_dt = pd.Timestamp(row["arrival_time"])
            p_start = int((arrival_dt - ref_date).total_seconds() / 60 / self.resolution_min)
            p_end = min(p_start + slots_needed, n_slots)
            if 0 <= p_start < n_slots:
                plug_load[p_start:p_end] += power

            s_start = int(
                (pd.Timestamp(row["scheduled_start"]) - ref_date).total_seconds()
                / 60 / self.resolution_min
            )
            s_end = min(s_start + slots_needed, n_slots)
            if 0 <= s_start < n_slots:
                smart_load[s_start:s_end] += power

        times = pd.date_range(ref_date, periods=n_slots, freq=f"{self.resolution_min}min")

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = None

        ax.step(times, plug_load, label="Plug-and-charge", color="#E84855",
                linewidth=1.8, where="post")
        ax.step(times, smart_load, label="Smart charging (V1G)", color="#2E86AB",
                linewidth=1.8, where="post", linestyle="--")

        if signal is not None:
            ax2 = ax.twinx()
            day_signal = signal[
                (signal.index >= ref_date) &
                (signal.index < ref_date + pd.Timedelta(days=1))
            ]
            if not day_signal.empty:
                day_signal.plot(ax=ax2, color="#F4A261", alpha=0.5, linewidth=1,
                                label="Price (€/kWh)")
                ax2.set_ylabel("Price (€/kWh)", color="#F4A261")
                ax2.tick_params(axis="y", labelcolor="#F4A261")

        ax.set_ylabel("Total load (kW)")
        ax.set_xlabel("Time")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"V1G load curve – {sessions['date'].iloc[0]}")

        if fig:
            fig.tight_layout()
        return ax

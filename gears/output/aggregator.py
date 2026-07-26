"""
Output aggregation utilities for GEARS.

Provides helpers for:
- Daily energy totals
- Hourly load profiles
- Annual load-profile reconstruction from a fitted EVSessionGMM
- Exporting to CSV / Parquet / Excel
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from gears.models.gmm import EVSessionGMM

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Charger-mix presets by location type
# ---------------------------------------------------------------------------

#: Location-stratified charger-power presets for ``build_load_profiles()``
#: in ``"by_location"`` mode.
#:
#: Each preset maps a ``location_type`` string to a dict
#: ``{power_kw: proportion}`` (proportions are auto-normalised).
#:
#: ``"french_2024"``
#:     Representative mix for the French charging network in 2024.
#:     Residential: mostly slow AC; work: AC 7–22 kW;
#:     public: mixed AC/DC including high-power DC.
#:
#: ``"french_2030_target"``
#:     Scenario aligned with French NECP targets for 2030:
#:     faster residential chargers, more 50–150 kW public stations.
LOCATION_POWER_PRESETS: dict[str, dict[str, dict[float, float]]] = {
    "french_2024": {
        "residential": {3.7: 0.30, 7.4: 0.60, 11.0: 0.10},
        "work":        {7.4: 0.55, 11.0: 0.25, 22.0: 0.20},
        "public":      {22.0: 0.40, 50.0: 0.35, 150.0: 0.20, 350.0: 0.05},
    },
    "french_2030_target": {
        "residential": {7.4: 0.50, 11.0: 0.35, 22.0: 0.15},
        "work":        {11.0: 0.30, 22.0: 0.50, 50.0: 0.20},
        "public":      {22.0: 0.20, 50.0: 0.35, 150.0: 0.30, 350.0: 0.15},
    },
}


# ---------------------------------------------------------------------------
# Dispatch table for export formats
# ---------------------------------------------------------------------------

_EXPORT_DISPATCH = {
    ".csv":     "to_csv",
    ".parquet": "to_parquet",
    ".xlsx":    "to_excel",
    ".json":    "to_json",
}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _overlap_profile_24h(
    arrivals: np.ndarray,
    active_durations: np.ndarray,
    powers: np.ndarray,
    n_sessions_per_day: float,
    n_total: int,
) -> np.ndarray:
    """
    Compute a 24-hour average power profile (kW) for a batch of sessions.

    Each session is characterised by its arrival hour, the duration of the
    *active charging window* (which may differ from the connection window in
    fixed-power mode), and the charger power.  Sessions that overflow
    midnight are handled correctly: their energy contribution wraps into the
    corresponding hours of the following day (modulo 24), preserving energy
    conservation across the 24-slot profile.

    The function supports durations up to 48 h (``EVSessionGMM`` hard clip),
    which means a session can span at most three calendar days.  Contributions
    from each day are folded back onto the 24-slot array.

    Parameters
    ----------
    arrivals : np.ndarray, shape (N,)
        Session arrival hours in [0, 24).
    active_durations : np.ndarray, shape (N,)
        Duration of the active charging window (hours).  For ``mean_power``
        mode this equals the connection duration; for ``fixed_power`` mode
        it is ``energy / charger_power_kw``, capped at the connection window.
    powers : np.ndarray, shape (N,)
        Charger power per session (kW).
    n_sessions_per_day : float
        Expected number of sessions per day for this context.  Used to
        normalise the Monte-Carlo average.
    n_total : int
        Total number of simulated sessions (= n_sessions_per_day × n_days_mc).

    Returns
    -------
    np.ndarray, shape (24,)
        Average power load per hour (kW), normalised to one representative day.

    Notes
    -----
    Energy conservation check:
    ``sum(profile) * 1 h ≈ mean_energy_per_session * n_sessions_per_day``

    The three-day overlap (o0, o1, o2) handles arrivals in [0, 24) and
    durations in [0, 48]: arrival + duration ≤ 72, so at most three
    calendar-day slots are touched.
    """
    hourly = np.zeros(24, dtype=float)
    # N sessions represent n_days_mc days, so dividing by (N / n_sessions_per_day)
    # yields the expected profile for one representative day.
    norm = n_total / n_sessions_per_day

    for h in range(24):
        h_f = float(h)
        # Overlap with the day-0 window [h, h+1].
        o0 = np.clip(
            np.minimum(arrivals + active_durations, h_f + 1.0)
            - np.maximum(arrivals, h_f),
            0.0, 1.0,
        )
        # Overlap with the day-1 window [h+24, h+25] for sessions crossing midnight.
        o1 = np.clip(
            np.minimum(arrivals + active_durations, h_f + 25.0)
            - np.maximum(arrivals, h_f + 24.0),
            0.0, 1.0,
        )
        # Overlap with the day-2 window [h+48, h+49] for sessions crossing two midnights
        # (possible when duration > 48 - arrival_hour, up to the 48 h GMM clip).
        o2 = np.clip(
            np.minimum(arrivals + active_durations, h_f + 49.0)
            - np.maximum(arrivals, h_f + 48.0),
            0.0, 1.0,
        )
        hourly[h] = np.sum(powers * (o0 + o1 + o2)) / norm

    return hourly  # kW


def _draw_power_levels(
    power_dist: dict[float, float],
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample charger power levels from a discrete probability distribution.

    Parameters
    ----------
    power_dist : dict[float, float]
        Mapping ``{power_kw: proportion}``.  Proportions are auto-normalised
        so they need not sum to exactly 1.
    n : int
        Number of samples to draw.
    rng : np.random.Generator
        NumPy random generator (for reproducibility).

    Returns
    -------
    np.ndarray, shape (n,)
        Sampled power values in kW.
    """
    powers = np.array(list(power_dist.keys()), dtype=float)
    probs  = np.array(list(power_dist.values()), dtype=float)
    probs /= probs.sum()  # normalise in case proportions don't sum to 1
    indices = rng.choice(len(powers), size=n, p=probs)
    return powers[indices]


def _build_smart_ts(
    gmm: EVSessionGMM,
    profiles_mw: dict,
    smart_profiles_mw: dict,
    year: int,
    noise_std: float,
    seed: int,
) -> pd.Series:
    """
    Assemble an annual hourly smart-charging time series (MW).

    Uses the same daily-noise draw as the baseline ``ts`` so that
    day-to-day variability is consistent between ``ts`` and ``ts_smart``.
    For (day_of_week, season) keys without a smart-charging profile,
    falls back to the corresponding baseline profile.

    Parameters
    ----------
    gmm : EVSessionGMM
        Fitted GMM (used for its ``stratify_by`` metadata).
    profiles_mw : dict
        Baseline (plug-and-charge) profiles
        ``{(dow, season): np.ndarray(24)}``.
    smart_profiles_mw : dict
        Smart-charging profiles ``{(dow, season): np.ndarray(24)}``.
    year : int
        Calendar year for the annual time series.
    noise_std : float
        Standard deviation of the multiplicative daily noise term.
    seed : int
        Master random seed (must match the seed used for ``ts``).

    Returns
    -------
    pd.Series
        Hourly MW time series with ``DatetimeIndex`` for the full
        calendar year (8 760 or 8 784 values).
    """
    from gears.data.schemas import _season

    dates_daily = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    rng_noise   = np.random.default_rng(seed)
    daily_noise = rng_noise.normal(1.0, noise_std, size=len(dates_daily))

    n_hours  = len(dates_daily) * 24
    ts_values = np.zeros(n_hours)

    for i, day in enumerate(dates_daily):
        key     = (day.dayofweek, _season(day.month))
        profile = smart_profiles_mw.get(key, profiles_mw.get(key, np.zeros(24)))
        ts_values[i * 24: (i + 1) * 24] = profile * daily_noise[i]

    ts_index = pd.date_range(f"{year}-01-01", periods=n_hours, freq="h")
    return pd.Series(ts_values, index=ts_index, name="GEARS_smart_MW")


def _reconstruct_smart_profile_hourly(
    result: pd.DataFrame,
    date: pd.Timestamp,
    resolution_min: int,
    n_sessions_per_day: float,
    n_total: int,
) -> np.ndarray:
    """
    Reconstruct a 24-hour smart-charging power profile (kW) from optimizer output.

    The optimizer marks each session with a ``scheduled_start`` Timestamp.
    The active charging window is taken as
    ``[scheduled_start, scheduled_start + charge_duration]``
    where ``charge_duration = energy / power_kw``.  This is consistent with
    the contiguous-slots approximation used in
    :meth:`~gears.smart_charging.optimizer.SmartChargingOptimizer.plot_load_curve`.

    Parameters
    ----------
    result : pd.DataFrame
        Output of
        :meth:`~gears.smart_charging.optimizer.SmartChargingOptimizer.optimise`.
    date : pd.Timestamp
        Reference date (midnight) used to convert Timestamps to fractional
        hours.
    resolution_min : int
        Optimizer resolution (minutes); used to derive ``power_kw`` from
        ``slots_needed`` if the column is absent.
    n_sessions_per_day : float
        Expected number of sessions per day for this context.
    n_total : int
        Total number of simulated sessions.

    Returns
    -------
    np.ndarray, shape (24,)
        Average hourly power (kW) for one representative day.
    """
    valid = result[result["scheduled_start"].notna()].copy()
    if valid.empty:
        return np.zeros(24)

    power_kw    = valid["power_kw"].values if "power_kw" in valid.columns else np.full(len(valid), 7.4)
    energies    = valid["energy"].values
    sched_starts = pd.to_datetime(valid["scheduled_start"])

    arrivals        = (sched_starts - date).dt.total_seconds().values / 3600.0
    charge_durations = np.clip(energies / power_kw, 0.0, 48.0)

    return _overlap_profile_24h(arrivals, charge_durations, power_kw, n_sessions_per_day, n_total)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class OutputAggregator:
    """
    Aggregate and export GEARS simulation outputs.

    Parameters
    ----------
    resolution_min : int
        Time resolution for smart-charging operations (minutes).
        Does not affect the hourly load profiles returned by
        :meth:`build_load_profiles`, which are always at 1-hour resolution.

    Examples
    --------
    >>> import gears
    >>> agg = OutputAggregator(resolution_min=60)
    >>> gmm = gears.get_gmm()
    >>> result = agg.build_load_profiles(gmm, year=2025, n_days_mc=10)
    >>> result["ts"].plot()
    """

    def __init__(self, resolution_min: int = 30):
        self.resolution_min = resolution_min

    # ------------------------------------------------------------------
    # Aggregation helpers
    # ------------------------------------------------------------------

    def daily_energy(
        self,
        sessions: pd.DataFrame,
        groupby: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Aggregate sessions to daily energy totals.

        Parameters
        ----------
        sessions : pd.DataFrame
            Sessions output from any GEARS simulator.
        groupby : list of str, optional
            Extra grouping columns
            (e.g. ``['scenario', 'location_type']``).

        Returns
        -------
        pd.DataFrame
            Columns: ``date``, ``[groupby…]``, ``total_energy_kwh``,
            ``n_sessions``, ``mean_duration_h``, ``peak_power_kw``.
        """
        df = sessions.copy()
        df["date"] = pd.to_datetime(df["date"])
        base_groupby = ["date"] + (groupby or [])
        if "scenario" in df.columns and "scenario" not in base_groupby:
            base_groupby.append("scenario")

        agg = df.groupby(base_groupby, observed=True).agg(
            total_energy_kwh=("energy", "sum"),
            n_sessions=("energy", "count"),
            mean_duration_h=("duration", "mean"),
            peak_power_kw=("power_kw", "sum") if "power_kw" in df.columns
                          else ("energy", "count"),
        ).reset_index()
        return agg

    def hourly_profile(
        self,
        sessions: pd.DataFrame,
        groupby: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Aggregate sessions to hourly load profiles.

        Parameters
        ----------
        sessions : pd.DataFrame
            Sessions output from any GEARS simulator.
        groupby : list of str, optional
            Extra grouping columns.

        Returns
        -------
        pd.DataFrame
            Columns: ``date``, ``hour``, ``[groupby…]``, ``power_kw``
            (the **sum** of the active charger powers across all sessions
            whose arrival hour falls in that slot — not a per-session
            value), ``energy_kwh``, ``n_sessions``.

        Notes
        -----
        ``power_kw`` represents the **aggregated** instantaneous power
        (sum of all active charger powers) for that hour bucket, not the
        power of an individual session.
        """
        df = sessions.copy()
        df["date"] = pd.to_datetime(df["date"])

        if "arrival_hour" in df.columns:
            df["hour"] = df["arrival_hour"].astype(int)
        elif "arrival_time" in df.columns:
            df["hour"] = pd.to_datetime(df["arrival_time"]).dt.hour
        else:
            raise ValueError(
                "Sessions must have 'arrival_hour' or 'arrival_time' for hourly profiles."
            )

        base_groupby = ["date", "hour"] + (groupby or [])
        if "scenario" in df.columns and "scenario" not in base_groupby:
            base_groupby.append("scenario")

        agg_dict: dict = {
            "energy_kwh": ("energy", "sum"),
            "n_sessions": ("energy", "count"),
        }
        if "power_kw" in df.columns:
            agg_dict["power_kw"] = ("power_kw", "sum")

        agg = df.groupby(base_groupby, observed=True).agg(**agg_dict).reset_index()
        return agg

    def scenario_stats(self, daily: pd.DataFrame) -> pd.DataFrame:
        """
        Compute cross-scenario statistics from daily energy output.

        Parameters
        ----------
        daily : pd.DataFrame
            Output of :meth:`daily_energy` with a ``scenario`` column.

        Returns
        -------
        pd.DataFrame
            Columns: ``date``, ``mean``, ``p10``, ``p25``, ``p75``, ``p90``.
            Returns ``daily`` unchanged if no ``scenario`` column is present.
        """
        if "scenario" not in daily.columns:
            return daily

        return (
            daily.groupby("date")["total_energy_kwh"]
            .agg(
                mean="mean",
                p10=lambda x: x.quantile(0.10),
                p25=lambda x: x.quantile(0.25),
                p75=lambda x: x.quantile(0.75),
                p90=lambda x: x.quantile(0.90),
            )
            .reset_index()
        )

    def export(
        self,
        df: pd.DataFrame,
        path: str | Path,
        index: bool = False,
        **kwargs,
    ) -> Path:
        """
        Export a DataFrame to file (CSV, Parquet, Excel, JSON).

        Parameters
        ----------
        df : pd.DataFrame
            Data to export.
        path : str or Path
            Destination file path.  The extension determines the format.
        index : bool
            Whether to include the DataFrame index in the output.
        **kwargs
            Forwarded to the underlying pandas writer method.

        Returns
        -------
        Path
            Resolved output path.

        Raises
        ------
        ValueError
            If the file extension is not supported.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        ext = path.suffix.lower()

        method_name = _EXPORT_DISPATCH.get(ext)
        if method_name is None:
            raise ValueError(
                f"Unsupported export format '{ext}'. "
                f"Supported: {list(_EXPORT_DISPATCH)}"
            )

        writer = getattr(df, method_name)
        if ext == ".json":
            writer(path, index=index, date_format="iso", **kwargs)
        elif ext in (".csv", ".xlsx"):
            writer(path, index=index, **kwargs)
        else:
            writer(path, **kwargs)

        logger.info("Exported %d rows to %s.", len(df), path)
        return path

    # ------------------------------------------------------------------
    # Load-profile reconstruction from a fitted GMM
    # ------------------------------------------------------------------

    def build_load_profiles(
        self,
        gmm: EVSessionGMM,
        year: int = 2025,
        n_days_mc: int = 10,
        charging_mode: str = "mean_power",
        charger_power_kw: float | None = None,
        location_power_map: dict | None = None,
        noise_std: float = 0.04,
        seed: int = 42,
        smart_charging_signal: pd.Series | None = None,
    ) -> dict:
        """
        Reconstruct annual hourly load profiles from a fitted EVSessionGMM.

        Integrates the Monte-Carlo profile-building logic from the
        ``scripts/compare.ipynb`` scratch notebook into the package as a
        first-class method.  Three charging-behaviour modes are supported,
        and sessions that overflow midnight are handled correctly so that
        energy is conserved across the 24-slot profile.

        Parameters
        ----------
        gmm : EVSessionGMM
            Fitted GMM (e.g. loaded via ``gears.get_gmm()``).
        year : int
            Calendar year for the annual time series.
        n_days_mc : int
            Monte-Carlo days per context stratum.  Higher values reduce
            variance at the cost of computation time.  Typically 10 is
            sufficient; use 30+ for publication-quality figures.
        charging_mode : str
            One of:

            ``"mean_power"``
                Each session charges at its average power
                ``energy / duration`` throughout its connection window.
                Sessions crossing midnight are handled correctly via
                modulo-24 folding in :func:`_overlap_profile_24h`.

            ``"fixed_power"``
                Every charger delivers a constant ``charger_power_kw`` kW.
                The active charging window is
                ``[arrival, arrival + energy / charger_power_kw]``;
                no power flows after the battery is full even if the
                vehicle remains plugged in.

            ``"by_location"``
                Per-session charger power is drawn randomly from
                ``location_power_map[location_type]``, then the
                ``"fixed_power"`` logic is applied.  Falls back to
                ``"mean_power"`` for location types absent from the map.

        charger_power_kw : float, optional
            Required when ``charging_mode="fixed_power"``.
        location_power_map : dict, optional
            Required when ``charging_mode="by_location"``.
            Format: ``{location_type: {power_kw: proportion}}``.
            See :data:`LOCATION_POWER_PRESETS` for ready-made examples.
        noise_std : float
            Standard deviation of the multiplicative daily noise term
            applied to each day's profile (Gaussian, mean=1).
            Controls realistic day-to-day variability.
        seed : int
            Master random seed.  Each context gets a deterministic
            child seed derived from this value.
        smart_charging_signal : pd.Series, optional
            If provided, a V1G smart-charging optimisation is run on
            internally generated sessions using
            :class:`~gears.smart_charging.optimizer.SmartChargingOptimizer`.
            The series must have a ``DatetimeIndex`` at
            ``self.resolution_min``-minute resolution.  When this
            parameter is set, the returned dict contains an additional
            ``"ts_smart"`` key.

        Returns
        -------
        dict
            Keys:

            ``"ts"`` : pd.Series
                Hourly annual load time series in MW,
                ``DatetimeIndex`` from ``{year}-01-01`` to
                ``{year}-12-31 23:00``.
            ``"profiles"`` : dict
                ``{(day_of_week, season): np.ndarray(24)}`` in MW.
                Day-of-week is an integer (0=Monday … 6=Sunday),
                season is a string (``"winter"``, ``"spring"``,
                ``"summer"``, ``"autumn"``).
            ``"loc_profiles"`` : dict
                ``{location_type: np.ndarray(24)}`` in MW.
                Sum of profiles across all strata sharing that
                location type.
            ``"n_by_type"`` : dict
                ``{(day_of_week, season): float}`` total expected
                sessions per day (summed over all location types and
                departments).
            ``"charging_mode"`` : str
                The mode used, for traceability.
            ``"ts_smart"`` : pd.Series (only when smart_charging_signal given)
                Same shape as ``"ts"`` but with V1G-optimised scheduling.

        Raises
        ------
        ValueError
            If ``charging_mode="fixed_power"`` without ``charger_power_kw``,
            or ``charging_mode="by_location"`` without ``location_power_map``,
            or ``charging_mode`` is unrecognised.

        Notes
        -----
        **Energy conservation** The sum of the profile values (× 1 h) equals
        the expected total energy per day for each (dow, season) stratum,
        to within Monte-Carlo variance.

        **Multi-day overflow** Sessions arriving late in the evening with
        long durations extend into the next calendar day.  The
        :func:`_overlap_profile_24h` helper folds those contributions
        back into the 24-slot array (modulo 24) so that energy is not lost.

        Examples
        --------
        >>> import gears
        >>> agg = gears.OutputAggregator()
        >>> gmm = gears.get_gmm()
        >>> out = agg.build_load_profiles(gmm, year=2025, charging_mode="mean_power")
        >>> out["ts"].resample("W").mean().plot(title="Weekly avg load (MW)")

        >>> out_fp = agg.build_load_profiles(
        ...     gmm, charging_mode="fixed_power", charger_power_kw=7.4
        ... )

        >>> out_bl = agg.build_load_profiles(
        ...     gmm,
        ...     charging_mode="by_location",
        ...     location_power_map=LOCATION_POWER_PRESETS["french_2024"],
        ... )
        """
        from gears.data.schemas import _season

        # --- Input validation --------------------------------------------
        if charging_mode not in ("mean_power", "fixed_power", "by_location"):
            raise ValueError(
                f"Unknown charging_mode '{charging_mode}'. "
                "Choose 'mean_power', 'fixed_power', or 'by_location'."
            )
        if charging_mode == "fixed_power" and charger_power_kw is None:
            raise ValueError(
                "charging_mode='fixed_power' requires charger_power_kw."
            )
        if charging_mode == "by_location" and location_power_map is None:
            raise ValueError(
                "charging_mode='by_location' requires location_power_map."
            )

        gmm._check_fitted()

        # Locate context-key positions within stratify_by.
        stratify    = gmm.stratify_by
        loc_idx     = stratify.index("location_type") if "location_type" in stratify else None
        dow_idx     = stratify.index("day_of_week")   if "day_of_week"   in stratify else None
        season_idx  = stratify.index("season")        if "season"        in stratify else None

        if charging_mode == "by_location" and loc_idx is None:
            raise ValueError(
                "charging_mode='by_location' requires 'location_type' in "
                "gmm.stratify_by, but it is not present."
            )

        # --- Accumulation containers -------------------------------------
        profiles_kw:    dict[tuple, np.ndarray] = defaultdict(lambda: np.zeros(24))
        loc_profiles_kw: dict[str, np.ndarray]  = defaultdict(lambda: np.zeros(24))
        n_by_type:       dict[tuple, float]      = defaultdict(float)

        n_contexts = len(gmm.models_)
        logger.info(
            "build_load_profiles: mode=%s, year=%d, n_days_mc=%d, %d contexts.",
            charging_mode, year, n_days_mc, n_contexts,
        )

        for ctx_idx, (ctx, sk_gmm) in enumerate(gmm.models_.items()):
            ctx = ctx if isinstance(ctx, tuple) else (ctx,)

            loc    = ctx[loc_idx]    if loc_idx    is not None else None
            dow    = ctx[dow_idx]    if dow_idx    is not None else None
            season = ctx[season_idx] if season_idx is not None else None

            n_per_day = gmm.n_sessions_per_day_.get(ctx, 0.0)
            if n_per_day < 1.0:
                logger.debug("Skipping context %s: n_per_day=%.2f", ctx, n_per_day)
                continue

            N = max(int(n_per_day * n_days_mc), 100)

            # Derive a deterministic per-context seed from the master seed so
            # different contexts are independent but results are reproducible.
            ctx_seed = int(seed + ctx_idx * 1_000_003 % (2**31))
            rng = np.random.default_rng(ctx_seed)

            # Sample raw GMM features: [hour, log1p(duration), log1p(energy)].
            sk_gmm.random_state = int(rng.integers(0, 2**31))
            raw, _ = sk_gmm.sample(N)

            arrivals  = np.clip(raw[:, 0], 0.0, 23.99)
            durations = np.clip(np.expm1(raw[:, 1]), 0.08, 48.0)
            energies  = np.clip(np.expm1(raw[:, 2]), 0.01, 350.0)

            # --- Compute per-session power and active charging window ----
            if charging_mode == "mean_power":
                powers           = energies / durations
                active_durations = durations

            elif charging_mode == "fixed_power":
                assert charger_power_kw is not None  # validated above
                powers           = np.full(N, charger_power_kw)
                # The vehicle stops drawing power as soon as its energy need is met.
                charge_times     = energies / charger_power_kw
                active_durations = np.minimum(charge_times, durations)

            else:  # "by_location"
                assert location_power_map is not None  # validated above
                if loc is not None and loc in location_power_map:
                    powers       = _draw_power_levels(location_power_map[loc], N, rng)
                    charge_times = energies / powers
                    active_durations = np.minimum(charge_times, durations)
                else:
                    # Location type unknown or not in map: fall back to mean_power.
                    logger.debug(
                        "Location type '%s' not in location_power_map; "
                        "falling back to mean_power for context %s.",
                        loc, ctx,
                    )
                    powers           = energies / durations
                    active_durations = durations

            # --- Build 24-hour profile with multi-day overflow -----------
            hp = _overlap_profile_24h(arrivals, active_durations, powers, n_per_day, N)

            profile_key = (dow, season)
            profiles_kw[profile_key]  += hp
            if loc is not None:
                loc_profiles_kw[loc]  += hp
            n_by_type[profile_key]    += n_per_day

        # --- Convert kW → MW ---------------------------------------------
        profiles_mw:     dict[tuple, np.ndarray] = {
            k: v / 1_000.0 for k, v in profiles_kw.items()
        }
        loc_profiles_mw: dict[str, np.ndarray] = {
            k: v / 1_000.0 for k, v in loc_profiles_kw.items()
        }

        # --- Build annual hourly time series with multiplicative noise ---
        dates_daily = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
        rng_noise   = np.random.default_rng(seed)
        daily_noise = rng_noise.normal(1.0, noise_std, size=len(dates_daily))

        n_hours  = len(dates_daily) * 24
        ts_values = np.zeros(n_hours)

        for i, day in enumerate(dates_daily):
            key     = (day.dayofweek, _season(day.month))
            profile = profiles_mw.get(key, np.zeros(24))
            ts_values[i * 24: (i + 1) * 24] = profile * daily_noise[i]

        ts_index = pd.date_range(f"{year}-01-01", periods=n_hours, freq="h")
        ts = pd.Series(ts_values, index=ts_index, name="GEARS_MW")

        logger.info(
            "Annual ts built: mean=%.3f MW, peak=%.3f MW.",
            ts.mean(), ts.max(),
        )

        result: dict = {
            "ts":           ts,
            "profiles":     profiles_mw,
            "loc_profiles": loc_profiles_mw,
            "n_by_type":    dict(n_by_type),
            "charging_mode": charging_mode,
        }

        # --- Optional smart-charging time series -------------------------
        if smart_charging_signal is not None:
            result["ts_smart"] = self._build_smart_charging_ts(
                gmm=gmm,
                sk_models=gmm.models_,
                stratify=stratify,
                loc_idx=loc_idx,
                dow_idx=dow_idx,
                season_idx=season_idx,
                charging_mode=charging_mode,
                charger_power_kw=charger_power_kw,
                location_power_map=location_power_map,
                n_days_mc=n_days_mc,
                profiles_mw=profiles_mw,
                year=year,
                noise_std=noise_std,
                seed=seed,
                signal=smart_charging_signal,
            )

        return result

    # ------------------------------------------------------------------
    # Private: smart-charging annual ts construction
    # ------------------------------------------------------------------

    def _build_smart_charging_ts(
        self,
        gmm: EVSessionGMM,
        sk_models: dict,
        stratify: list,
        loc_idx: int | None,
        dow_idx: int | None,
        season_idx: int | None,
        charging_mode: str,
        charger_power_kw: float | None,
        location_power_map: dict | None,
        n_days_mc: int,
        profiles_mw: dict,
        year: int,
        noise_std: float,
        seed: int,
        signal: pd.Series,
    ) -> pd.Series:
        """
        Build an annual hourly smart-charging time series (MW).

        For each (day_of_week, season) context that appears in the signal's
        date range, this method:

        1. Picks one representative date from the signal that matches
           the context.
        2. Generates ``n_sessions_per_day_mean`` sessions with explicit
           timestamps.
        3. Assigns charger power according to ``charging_mode``.
        4. Runs
           :class:`~gears.smart_charging.optimizer.SmartChargingOptimizer`
           on those sessions.
        5. Reconstructs a 24-hour smart-charging profile from the
           scheduled windows.
        6. Uses that profile (with daily noise) to assemble the full
           annual series.

        Parameters
        ----------
        gmm : EVSessionGMM
        sk_models : dict
        stratify : list
        loc_idx, dow_idx, season_idx : int or None
        charging_mode : str
        charger_power_kw : float or None
        location_power_map : dict or None
        n_days_mc : int
        profiles_mw : dict
            Baseline profiles used as fallback for days without a smart profile.
        year : int
        noise_std : float
        seed : int
        signal : pd.Series
            Price or RES signal at ``self.resolution_min`` resolution.

        Returns
        -------
        pd.Series
            Hourly MW time series with ``DatetimeIndex`` for the full year.
        """
        from gears.data.schemas import _season as _seas
        from gears.smart_charging.optimizer import SmartChargingOptimizer

        opt = SmartChargingOptimizer(
            resolution_min=self.resolution_min,
            signal_type="price",
        )

        # Build a lookup: (dow, season) → one representative date in signal.
        signal_dates = signal.index.normalize().unique()
        representative_dates: dict[tuple, pd.Timestamp] = {}
        for sig_dt in signal_dates:
            key = (sig_dt.dayofweek, _seas(sig_dt.month))
            if key not in representative_dates:
                representative_dates[key] = sig_dt

        # Per-context smart-charging 24 h profiles (kW).
        smart_profiles_kw: dict[tuple, np.ndarray] = defaultdict(lambda: np.zeros(24))

        for ctx_idx, (ctx, sk_gmm) in enumerate(sk_models.items()):
            ctx = ctx if isinstance(ctx, tuple) else (ctx,)

            loc    = ctx[loc_idx]    if loc_idx    is not None else None
            dow    = ctx[dow_idx]    if dow_idx    is not None else None
            season = ctx[season_idx] if season_idx is not None else None

            profile_key = (dow, season)
            rep_date    = representative_dates.get(profile_key)
            if rep_date is None:
                # No matching date in signal — skip this context.
                continue

            n_per_day = gmm.n_sessions_per_day_.get(ctx, 0.0)
            if n_per_day < 1.0:
                continue

            N = max(int(n_per_day * n_days_mc), 100)

            # Use a seed offset so the smart-charging samples differ from the
            # baseline profile samples while remaining deterministic.
            ctx_seed = int(seed + ctx_idx * 1_000_003 % (2**31))
            rng      = np.random.default_rng(ctx_seed + 7)

            sk_gmm.random_state = int(rng.integers(0, 2**31))
            raw, _ = sk_gmm.sample(N)

            arrivals  = np.clip(raw[:, 0], 0.0, 23.99)
            durations = np.clip(np.expm1(raw[:, 1]), 0.08, 48.0)
            energies  = np.clip(np.expm1(raw[:, 2]), 0.01, 350.0)

            if charging_mode == "mean_power":
                powers = energies / durations
            elif charging_mode == "fixed_power":
                assert charger_power_kw is not None
                powers = np.full(N, charger_power_kw)
            else:  # by_location
                assert location_power_map is not None
                if loc is not None and loc in location_power_map:
                    powers = _draw_power_levels(location_power_map[loc], N, rng)
                else:
                    powers = energies / durations

            # Build a sessions DataFrame compatible with the optimizer API.
            # Arrival times are anchored to the representative date for this context.
            arrival_times = rep_date + pd.to_timedelta(arrivals, unit="h")
            sessions_df   = pd.DataFrame({
                "arrival_time": arrival_times,
                "duration":     durations,
                "energy":       energies,
                "power_kw":     powers,
            })

            try:
                optimised = opt.optimise(sessions_df, signal)
            except Exception as exc:  # noqa: BLE001 - aggregation step is best-effort; log and continue on any failure
                logger.warning(
                    "SmartChargingOptimizer failed for context %s on date %s: %s. "
                    "Falling back to baseline profile.",
                    ctx, rep_date.date(), exc,
                )
                continue

            hp = _reconstruct_smart_profile_hourly(
                result=optimised,
                date=rep_date,
                resolution_min=self.resolution_min,
                n_sessions_per_day=n_per_day,
                n_total=N,
            )

            smart_profiles_kw[profile_key] += hp

        # Convert to MW; baseline profiles cover (dow, season) without smart data.
        smart_profiles_mw: dict[tuple, np.ndarray] = {
            key: arr_kw / 1_000.0 for key, arr_kw in smart_profiles_kw.items()
        }

        ts_smart = _build_smart_ts(
            gmm=gmm,
            profiles_mw=profiles_mw,
            smart_profiles_mw=smart_profiles_mw,
            year=year,
            noise_std=noise_std,
            seed=seed,
        )

        logger.info(
            "Smart-charging ts built: mean=%.3f MW, peak=%.3f MW.",
            ts_smart.mean(), ts_smart.max(),
        )
        return ts_smart

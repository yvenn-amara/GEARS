"""
Output aggregation for GEARS.

- daily energy totals and hourly load profiles of simulated sessions
- annual hourly load profiles reconstructed from a fitted EVSessionModel
  (optionally with V1G smart charging)
- export to CSV / Parquet / Excel / JSON
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from gears.data.schemas import _season

if TYPE_CHECKING:
    from gears.models.session_model import EVSessionModel

logger = logging.getLogger(__name__)

_CHARGING_MODES = ("mean_power", "fixed_power", "by_location")


# ---------------------------------------------------------------------------
# Charger-mix presets by location type
# ---------------------------------------------------------------------------

#: Charger-power presets for ``build_load_profiles(charging_mode="by_location")``:
#: ``{preset: {location_type: {power_kw: proportion}}}`` (proportions are normalised).
#: ``"french_2024"`` is a representative mix of the French network in 2024,
#: ``"french_2030_target"`` a scenario aligned with the 2030 national targets.
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


# --- private helpers --------------------------------------------------------

def _overlap_profile_24h(
    arrivals: np.ndarray,
    active_durations: np.ndarray,
    powers: np.ndarray,
    n_sessions_per_day: float,
    n_total: int,
) -> np.ndarray:
    """
    Average 24-hour power profile (kW) of a batch of Monte-Carlo sessions.

    Each session draws ``powers`` kW during ``[arrival, arrival + active_duration]``
    (hours). Sessions running past midnight wrap onto the following days (modulo 24;
    durations up to 48 h touch at most three calendar days), so energy is conserved:
    ``sum(profile) * 1 h == mean energy per session * n_sessions_per_day``.

    ``n_total`` sessions stand for ``n_total / n_sessions_per_day`` days; the result
    is normalised to one representative day.
    """
    # N sessions represent n_days_mc days, so dividing by (N / n_sessions_per_day)
    # yields the expected profile for one representative day.
    norm = n_total / n_sessions_per_day
    ends = arrivals + active_durations

    hourly = np.zeros(24, dtype=float)
    for h in range(24):
        # Overlap with the hour [h, h+1] on day 0, 1 and 2 (shifts of 0, 24, 48 h).
        overlap = sum(
            np.clip(np.minimum(ends, h + shift + 1.0) - np.maximum(arrivals, h + shift), 0.0, 1.0)
            for shift in (0.0, 24.0, 48.0)
        )
        hourly[h] = np.sum(powers * overlap) / norm
    return hourly  # kW


def _draw_power_levels(
    power_dist: dict[float, float],
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample ``n`` charger powers (kW) from ``{power_kw: proportion}`` (auto-normalised)."""
    powers = np.array(list(power_dist.keys()), dtype=float)
    probs  = np.array(list(power_dist.values()), dtype=float)
    probs /= probs.sum()  # normalise in case proportions don't sum to 1
    indices = rng.choice(len(powers), size=n, p=probs)
    return powers[indices]


def _context_indices(stratify: list[str]) -> tuple[int | None, int | None, int | None]:
    """Positions of ``location_type``, ``day_of_week`` and ``season`` in ``stratify_by``."""
    return tuple(stratify.index(c) if c in stratify else None
                 for c in ("location_type", "day_of_week", "season"))


def _unpack_context(ctx, indices) -> tuple[tuple, tuple]:
    """Normalise a context key to a tuple and extract ``(loc, dow, season)`` from it."""
    ctx = ctx if isinstance(ctx, tuple) else (ctx,)
    return ctx, tuple(ctx[i] if i is not None else None for i in indices)


def _context_seed(seed: int, ctx_idx: int) -> int:
    """Deterministic per-context seed: contexts are independent, results reproducible."""
    return int(seed + ctx_idx * 1_000_003 % (2**31))


def _sample_context(sk_gmm, n: int, rng: np.random.Generator):
    """Draw ``n`` sessions from a fitted context model: (arrivals, durations, energies)."""
    sk_gmm.random_state = int(rng.integers(0, 2**31))
    raw, _ = sk_gmm.sample(n)  # features: [hour, log1p(duration), log1p(energy)]
    arrivals = np.clip(raw[:, 0], 0.0, 23.99)
    durations = np.clip(np.expm1(raw[:, 1]), 0.08, 48.0)
    energies = np.clip(np.expm1(raw[:, 2]), 0.01, 350.0)
    return arrivals, durations, energies


def _session_powers(
    charging_mode: str,
    energies: np.ndarray,
    durations: np.ndarray,
    rng: np.random.Generator,
    charger_power_kw: float | None,
    location_power_map: dict | None,
    loc,
) -> tuple[np.ndarray, np.ndarray]:
    """Charger power and active-charging duration of each session, per ``charging_mode``."""
    if charging_mode == "fixed_power":
        powers = np.full(len(energies), charger_power_kw)
    elif charging_mode == "by_location" and loc is not None and loc in location_power_map:
        powers = _draw_power_levels(location_power_map[loc], len(energies), rng)
    else:
        # "mean_power" (also the fallback for a location type missing from the map):
        # constant power over the whole connection window.
        return energies / durations, durations
    # The vehicle stops drawing power as soon as its energy need is met.
    return powers, np.minimum(energies / powers, durations)


def _annual_series(
    profiles_mw: dict,
    year: int,
    noise_std: float,
    seed: int,
    name: str,
    fallback: dict | None = None,
) -> pd.Series:
    """
    Hourly MW series for a calendar year from ``{(day_of_week, season): profile(24)}``.

    Each day's profile is scaled by multiplicative Gaussian noise (mean 1,
    ``noise_std``); the same ``seed`` always yields the same daily noise, so series
    built from different profile sets stay comparable. Days whose key is missing
    use ``fallback`` and, failing that, zeros.
    """
    days = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    daily_noise = np.random.default_rng(seed).normal(1.0, noise_std, size=len(days))

    values = np.zeros(len(days) * 24)
    for i, day in enumerate(days):
        key = (day.dayofweek, _season(day.month))
        profile = profiles_mw.get(key)
        if profile is None:
            profile = (fallback or {}).get(key, np.zeros(24))
        values[i * 24:(i + 1) * 24] = profile * daily_noise[i]

    index = pd.date_range(f"{year}-01-01", periods=len(values), freq="h")
    return pd.Series(values, index=index, name=name)


def _build_smart_ts(
    profiles_mw: dict,
    smart_profiles_mw: dict,
    year: int,
    noise_std: float,
    seed: int,
) -> pd.Series:
    """Annual smart-charging series; keys without a smart profile use the baseline one."""
    return _annual_series(smart_profiles_mw, year, noise_std, seed,
                          name="GEARS_smart_MW", fallback=profiles_mw)


def _reconstruct_smart_profile_hourly(
    result: pd.DataFrame,
    date: pd.Timestamp,
    n_sessions_per_day: float,
    n_total: int,
) -> np.ndarray:
    """
    24-hour smart-charging power profile (kW) from the optimiser's output.

    Each session is charging during ``[scheduled_start, scheduled_start + energy / power_kw]``
    (``power_kw`` defaults to 7.4 kW when the column is absent). ``date`` is the
    midnight reference used to convert timestamps to fractional hours.
    """
    valid = result[result["scheduled_start"].notna()]
    if valid.empty:
        return np.zeros(24)

    power_kw = valid["power_kw"].values if "power_kw" in valid.columns else np.full(len(valid), 7.4)
    arrivals = (pd.to_datetime(valid["scheduled_start"]) - date).dt.total_seconds().values / 3600.0
    charge_durations = np.clip(valid["energy"].values / power_kw, 0.0, 48.0)

    return _overlap_profile_24h(arrivals, charge_durations, power_kw, n_sessions_per_day, n_total)


class OutputAggregator:
    """
    Aggregate and export GEARS simulation outputs.

    Parameters
    ----------
    resolution_min : int
        Time resolution of the smart-charging optimisation (minutes). The hourly
        load profiles of :meth:`build_load_profiles` are always hourly.

    Examples
    --------
    >>> import gears
    >>> agg = gears.OutputAggregator(resolution_min=60)
    >>> gmm = gears.load_pretrained("french")
    >>> result = agg.build_load_profiles(gmm, year=2025, n_days_mc=10)
    >>> result["ts"].plot()
    """

    def __init__(self, resolution_min: int = 30):
        self.resolution_min = resolution_min

    def daily_energy(
        self,
        sessions: pd.DataFrame,
        groupby: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Aggregate sessions to daily totals.

        ``groupby`` adds grouping columns (``scenario`` is always kept when present).
        Returns ``date``, ``[groupby…]``, ``total_energy_kwh``, ``n_sessions``,
        ``mean_duration_h`` and ``peak_power_kw`` (sum of charger powers).
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
            Sessions output from any GEARS simulator (needs ``arrival_hour`` or
            ``arrival_time``).
        groupby : list of str, optional
            Extra grouping columns.

        Returns
        -------
        pd.DataFrame
            Columns: ``date``, ``hour``, ``[groupby…]``, ``power_kw`` (the **sum** of
            the charger powers of the sessions arriving in that hour, when the input
            has a ``power_kw`` column), ``energy_kwh``, ``n_sessions``.
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
        Cross-scenario statistics of :meth:`daily_energy` output.

        Returns ``date``, ``mean``, ``p10``, ``p25``, ``p75``, ``p90`` of
        ``total_energy_kwh``; ``daily`` is returned unchanged without a ``scenario`` column.
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
        Export a DataFrame to CSV, Parquet, Excel or JSON (chosen by the extension).

        Parent directories are created; ``index`` and ``**kwargs`` go to the pandas
        writer. Returns the output path; raises ``ValueError`` for an unsupported extension.
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

    def build_load_profiles(
        self,
        gmm: EVSessionModel,
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
        Reconstruct annual hourly load profiles from a fitted EVSessionModel.

        Sessions are sampled from every context (Monte-Carlo), turned into 24-hour
        power profiles (sessions running past midnight wrap around, so energy is
        conserved) and assembled into a year of hourly values.

        Parameters
        ----------
        gmm : EVSessionModel
            Fitted session model (e.g. from :func:`gears.load_pretrained`).
        year : int
            Calendar year of the output series.
        n_days_mc : int
            Monte-Carlo days per context; 10 is usually enough, use 30+ for figures.
        charging_mode : {"mean_power", "fixed_power", "by_location"}
            ``"mean_power"``: each session charges at ``energy / duration`` over its
            connection window. ``"fixed_power"``: every charger delivers
            ``charger_power_kw`` until the energy need is met. ``"by_location"``: the
            power is drawn from ``location_power_map[location_type]`` (then as
            ``"fixed_power"``); location types missing from the map use ``"mean_power"``.
        charger_power_kw : float, optional
            Required for ``"fixed_power"``.
        location_power_map : dict, optional
            Required for ``"by_location"``: ``{location_type: {power_kw: proportion}}``
            (see :data:`LOCATION_POWER_PRESETS`).
        noise_std : float
            Standard deviation of the multiplicative daily noise (Gaussian, mean 1).
        seed : int
            Master seed; each context derives its own child seed.
        smart_charging_signal : pd.Series, optional
            Price signal on a ``DatetimeIndex`` at ``resolution_min``. When given, V1G
            smart charging is applied to generated sessions and the result gains ``"ts_smart"``.

        Returns
        -------
        dict
            ``"ts"`` (hourly MW series), ``"profiles"`` (``{(day_of_week, season): ndarray(24)}``
            in MW), ``"loc_profiles"`` (``{location_type: ndarray(24)}`` in MW),
            ``"n_by_type"`` (expected sessions per day per ``(day_of_week, season)``),
            ``"charging_mode"`` and, with a signal, ``"ts_smart"``.

        Raises
        ------
        ValueError
            Unknown ``charging_mode`` or missing required argument (``"by_location"``
            also needs ``location_type`` in ``gmm.stratify_by``).
        """
        if charging_mode not in _CHARGING_MODES:
            raise ValueError(
                f"Unknown charging_mode '{charging_mode}'. "
                "Choose 'mean_power', 'fixed_power', or 'by_location'."
            )
        if charging_mode == "fixed_power" and charger_power_kw is None:
            raise ValueError("charging_mode='fixed_power' requires charger_power_kw.")
        if charging_mode == "by_location" and location_power_map is None:
            raise ValueError("charging_mode='by_location' requires location_power_map.")

        gmm._check_fitted()

        indices = _context_indices(gmm.stratify_by)
        if charging_mode == "by_location" and indices[0] is None:
            raise ValueError(
                "charging_mode='by_location' requires 'location_type' in "
                "gmm.stratify_by, but it is not present."
            )

        profiles_kw: dict[tuple, np.ndarray] = defaultdict(lambda: np.zeros(24))
        loc_profiles_kw: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(24))
        n_by_type: dict[tuple, float] = defaultdict(float)

        logger.info(
            "build_load_profiles: mode=%s, year=%d, n_days_mc=%d, %d contexts.",
            charging_mode, year, n_days_mc, len(gmm.models_),
        )

        for ctx_idx, (ctx, sk_gmm) in enumerate(gmm.models_.items()):
            ctx, (loc, dow, season) = _unpack_context(ctx, indices)

            n_per_day = gmm.n_sessions_per_day_.get(ctx, 0.0)
            if n_per_day < 1.0:
                logger.debug("Skipping context %s: n_per_day=%.2f", ctx, n_per_day)
                continue

            n = max(int(n_per_day * n_days_mc), 100)
            rng = np.random.default_rng(_context_seed(seed, ctx_idx))
            arrivals, durations, energies = _sample_context(sk_gmm, n, rng)
            powers, active_durations = _session_powers(
                charging_mode, energies, durations, rng,
                charger_power_kw, location_power_map, loc,
            )

            hourly_kw = _overlap_profile_24h(arrivals, active_durations, powers, n_per_day, n)
            profiles_kw[(dow, season)] += hourly_kw
            if loc is not None:
                loc_profiles_kw[loc] += hourly_kw
            n_by_type[(dow, season)] += n_per_day

        profiles_mw = {k: v / 1_000.0 for k, v in profiles_kw.items()}
        ts = _annual_series(profiles_mw, year, noise_std, seed, name="GEARS_MW")
        logger.info("Annual ts built: mean=%.3f MW, peak=%.3f MW.", ts.mean(), ts.max())

        result: dict = {
            "ts": ts,
            "profiles": profiles_mw,
            "loc_profiles": {k: v / 1_000.0 for k, v in loc_profiles_kw.items()},
            "n_by_type": dict(n_by_type),
            "charging_mode": charging_mode,
        }
        if smart_charging_signal is not None:
            result["ts_smart"] = self._build_smart_charging_ts(
                gmm, charging_mode, charger_power_kw, location_power_map,
                n_days_mc, profiles_mw, year, noise_std, seed, smart_charging_signal,
            )
        return result

    def _build_smart_charging_ts(
        self,
        gmm: EVSessionModel,
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
        Annual hourly smart-charging series (MW).

        For each context whose ``(day_of_week, season)`` occurs in the signal's date
        range, sessions are generated on one representative date, scheduled with
        :class:`~gears.smart_charging.optimizer.SmartChargingOptimizer` and turned into
        a 24-hour profile; days without a smart profile use the baseline ``profiles_mw``.
        """
        from gears.smart_charging.optimizer import SmartChargingOptimizer

        opt = SmartChargingOptimizer(resolution_min=self.resolution_min, signal_type="price")
        indices = _context_indices(gmm.stratify_by)

        # One representative date per (day_of_week, season) present in the signal.
        representative_dates: dict[tuple, pd.Timestamp] = {}
        for day in signal.index.normalize().unique():
            representative_dates.setdefault((day.dayofweek, _season(day.month)), day)

        smart_profiles_kw: dict[tuple, np.ndarray] = defaultdict(lambda: np.zeros(24))

        for ctx_idx, (ctx, sk_gmm) in enumerate(gmm.models_.items()):
            ctx, (loc, dow, season) = _unpack_context(ctx, indices)

            rep_date = representative_dates.get((dow, season))
            n_per_day = gmm.n_sessions_per_day_.get(ctx, 0.0)
            if rep_date is None or n_per_day < 1.0:
                continue

            n = max(int(n_per_day * n_days_mc), 100)
            # Seed offset: smart-charging samples differ from the baseline ones, deterministically.
            rng = np.random.default_rng(_context_seed(seed, ctx_idx) + 7)
            arrivals, durations, energies = _sample_context(sk_gmm, n, rng)
            powers, _ = _session_powers(
                charging_mode, energies, durations, rng,
                charger_power_kw, location_power_map, loc,
            )

            sessions = pd.DataFrame({
                "arrival_time": rep_date + pd.to_timedelta(arrivals, unit="h"),
                "duration": durations,
                "energy": energies,
                "power_kw": powers,
            })
            try:
                optimised = opt.optimise(sessions, signal)
            except Exception as exc:  # noqa: BLE001 - best-effort per context: warn and keep the baseline
                logger.warning(
                    "SmartChargingOptimizer failed for context %s on date %s: %s. "
                    "Falling back to baseline profile.",
                    ctx, rep_date.date(), exc,
                )
                continue

            smart_profiles_kw[(dow, season)] += _reconstruct_smart_profile_hourly(
                optimised, rep_date, n_per_day, n,
            )

        smart_profiles_mw = {k: v / 1_000.0 for k, v in smart_profiles_kw.items()}
        ts_smart = _build_smart_ts(profiles_mw, smart_profiles_mw, year, noise_std, seed)
        logger.info("Smart-charging ts built: mean=%.3f MW, peak=%.3f MW.",
                    ts_smart.mean(), ts_smart.max())
        return ts_smart

"""
Medium-term EV demand simulator.

Generates aggregated daily energy demand over a configurable multi-year horizon
(no fixed limit) using GMM-based session sampling with EV adoption growth.

Growth profiles
---------------
Three growth models are available (see module-level functions). All three
are anchored so that ``growth(t=0) == base_sessions_per_day`` exactly — no
discontinuity between "current observed fleet" and the simulated trajectory
(see "Design notes" below).

linear_growth_profile
    Genuinely linear year-on-year growth: ``base * (1 + rate * t)``. Best for
    short horizons or as a simple "conservative" baseline scenario.

s_curve_growth_profile
    Logistic (sigmoid) growth.  Captures the classic S-curve of technology
    adoption: slow start, rapid middle phase, plateau at saturation.
    ``midpoint_year``/``steepness`` default to values *relative to* the
    requested ``years`` horizon, so the curve spans the full horizon instead
    of saturating after a fixed number of years regardless of how long a
    simulation is requested.

bass_diffusion_profile
    Bass (1969) diffusion model, the standard in technology adoption
    research.  Explicitly models two populations: *innovators* (p) who
    adopt independently, and *imitators* (q) who adopt after social contact.
    Typical EV parameters: p ≈ 0.03, q ≈ 0.38.

Design notes
------------
Only three profiles are offered, one per canonical adoption scenario
(conservative / central / ambitious); extra hyperparameters would not
differentiate them. Each profile is built so that:

- ``growth(t=0) == base_sessions_per_day`` exactly. A scenario starts from the
  current observed fleet, not from zero: the logistic and Bass curves are
  rescaled so that no jump appears between the observed baseline and the
  simulated trajectory.
- the logistic curve spans the requested horizon. ``midpoint_year`` and
  ``steepness`` default to ``years / 2`` and ``6 / years``, so the curve
  reaches about 95 % of its asymptote at ``t = years`` whether 3 or 20 years
  are simulated (fixed constants would saturate it after the same number of
  years whatever the horizon, giving a plateau in long scenarios).

Simulation speed
----------------
The inner loop is vectorised by context: instead of calling gmm.sample()
for each individual day, days sharing the same (day_of_week, season) context
are batched together, dramatically reducing Python overhead on long horizons.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from gears._validation import as_naive_timestamp, check_float, check_int
from gears.data.schemas import _season

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Growth profiles
# ---------------------------------------------------------------------------

def linear_growth_profile(
    base_sessions_per_day: float,
    years: float,
    annual_growth_rate: float = 0.15,
    start_date: str | pd.Timestamp | None = None,
) -> pd.Series:
    """
    Daily session counts with constant year-on-year growth.

    Linear in elapsed time: ``base * (1 + annual_growth_rate * t)``, not the
    compound growth ``base * (1 + rate) ** t``.

    Parameters
    ----------
    base_sessions_per_day : float
        Sessions per day at the start of the simulation (t = 0). The returned
        series starts exactly at this value.
    years : float
        Simulation horizon (any positive value).
    annual_growth_rate : float
        Fractional annual growth, applied linearly (0.15 = +15% of the t=0
        value, added each year — so 1+15*0.15 = 3.25x after 15 years, not
        1.15**15 = 8.1x).
    start_date : str or Timestamp, optional
        Start of the date index. Defaults to today.

    Returns
    -------
    pd.Series
        Daily expected session counts indexed by date.
    """
    start = pd.Timestamp(start_date) if start_date else pd.Timestamp.today().normalize()
    n_days = max(1, round(365.25 * years))
    dates = pd.date_range(start, periods=n_days, freq="D")
    t = np.arange(n_days) / 365.25
    # A shrinking fleet cannot have a negative number of sessions: clamp at 0.
    counts = np.maximum(base_sessions_per_day * (1 + annual_growth_rate * t), 0.0)
    return pd.Series(counts, index=dates, name="n_sessions_expected")


def s_curve_growth_profile(
    base_sessions_per_day: float,
    years: float,
    saturation_factor: float = 3.0,
    midpoint_year: float | None = None,
    steepness: float | None = None,
    start_date: str | pd.Timestamp | None = None,
) -> pd.Series:
    """
    Logistic (S-curve) growth — technology adoption classic.

    Rescaled so the curve is anchored exactly at ``base_sessions_per_day`` at
    t=0 and approaches ``base_sessions_per_day * saturation_factor`` as an
    asymptote. A raw logistic curve starts at a few percent of its asymptote,
    which would be a discontinuity when a scenario is meant to start "from the
    current observed fleet, not zero": the raw curve's own t=0 value is
    subtracted and the remainder is rescaled to span exactly
    [base, base * saturation_factor].

    Parameters
    ----------
    base_sessions_per_day : float
        Sessions per day at t = 0. The returned series starts exactly here.
    years : float
        Simulation horizon.
    saturation_factor : float
        Max sessions as multiple of base (asymptote).
    midpoint_year : float, optional
        Year at which growth rate is maximal. Defaults to ``years / 2`` —
        i.e. scales with the requested horizon. (The old fixed default of
        2.5 meant the curve was ~100% saturated by year 8 regardless of
        whether `years` was 3 or 20.)
    steepness : float, optional
        Controls the slope of the S (larger = sharper transition). Defaults
        to ``6 / years`` so the curve reaches ~95% of the way to saturation
        at t=years, for any requested horizon.
    start_date : str or Timestamp, optional

    Returns
    -------
    pd.Series
        Daily expected session counts indexed by date.
    """
    if midpoint_year is None:
        midpoint_year = years / 2.0
    if steepness is None:
        steepness = 6.0 / years

    start = pd.Timestamp(start_date) if start_date else pd.Timestamp.today().normalize()
    n_days = max(1, round(365.25 * years))
    dates = pd.date_range(start, periods=n_days, freq="D")
    t = np.arange(n_days) / 365.25

    raw = 1.0 / (1.0 + np.exp(-steepness * (t - midpoint_year)))
    raw0 = 1.0 / (1.0 + np.exp(steepness * midpoint_year))  # raw curve's own value at t=0
    normalized = (raw - raw0) / (1.0 - raw0)  # rescaled so normalized(0) = 0, normalized(inf) = 1
    asymptote = base_sessions_per_day * saturation_factor
    growth = base_sessions_per_day + (asymptote - base_sessions_per_day) * normalized
    return pd.Series(growth, index=dates, name="n_sessions_expected")


def bass_diffusion_profile(
    base_sessions_per_day: float,
    years: float,
    market_potential_factor: float = 4.0,
    p: float = 0.03,
    q: float = 0.38,
    start_date: str | pd.Timestamp | None = None,
) -> pd.Series:
    """
    Bass (1969) diffusion model for EV adoption.

    The Bass model describes how *innovators* (coefficient p) adopt
    independently of others, while *imitators* (coefficient q) adopt
    proportionally to the existing installed base.  This is the standard
    model in technology-diffusion research and fits EV adoption data well.

    Typical EV values: p ≈ 0.01–0.05 (innovators), q ≈ 0.3–0.5 (imitators).

    Anchored so the curve equals ``base_sessions_per_day`` exactly at t=0 and
    approaches ``base_sessions_per_day * market_potential_factor`` as an
    asymptote. The textbook Bass formulation models the cumulative adopters
    of a brand-new product diffusing from zero, which is wrong when there is
    already a fleet: applied directly, the trajectory would start at zero
    sessions per day. Here the cumulative-adoption fraction F(t) tracks the
    growth *beyond* the current base instead of adoption from scratch.

    Parameters
    ----------
    base_sessions_per_day : float
        Sessions per day at t = 0. The returned series starts exactly here.
    years : float
    market_potential_factor : float
        Ultimate market size as multiple of base.
    p : float
        Coefficient of innovation (external influence).
    q : float
        Coefficient of imitation (internal influence / word-of-mouth).
    start_date : str or Timestamp, optional

    Returns
    -------
    pd.Series
        Daily expected session counts indexed by date.
    """
    start = pd.Timestamp(start_date) if start_date else pd.Timestamp.today().normalize()
    n_days = max(1, round(365.25 * years))
    dates = pd.date_range(start, periods=n_days, freq="D")
    t = np.arange(n_days) / 365.25

    # Cumulative adoption fraction F(t) from the closed-form Bass solution:
    # F(0) = 0, F(t) -> 1 as t -> infinity.
    exponent = np.exp(-(p + q) * t)
    F = (1 - exponent) / (1 + (q / p) * exponent)

    additional_ceiling = base_sessions_per_day * (market_potential_factor - 1.0)
    N = base_sessions_per_day + additional_ceiling * F
    return pd.Series(N, index=dates, name="n_sessions_expected")


# Growth profile registry: one profile per canonical adoption scenario
# (see the module docstring, "Design notes").
GROWTH_PROFILES = {
    "linear": linear_growth_profile,
    "s_curve": s_curve_growth_profile,
    "bass": bass_diffusion_profile,
}


# ---------------------------------------------------------------------------
# Charger mix presets
# ---------------------------------------------------------------------------

DEFAULT_CHARGER_MIX: dict[float, float] = {7.4: 0.5, 22.0: 0.5}

CHARGER_PRESETS: dict[str, dict[float, float]] = {
    "current_fr":       {3.7: 0.10, 7.4: 0.55, 22.0: 0.30, 50.0:  0.05},
    "fast_dominant":    {22.0: 0.40, 50.0: 0.40, 150.0: 0.20},
    "slow_residential": {3.7: 0.30, 7.4: 0.60, 11.0:  0.10},
    "workplace":        {7.4: 0.50, 11.0: 0.30, 22.0:  0.20},
    "public_mixed":     {22.0: 0.35, 50.0: 0.35, 150.0: 0.20, 350.0: 0.10},
}


# ---------------------------------------------------------------------------
# Main simulator
# ---------------------------------------------------------------------------

class MediumTermSimulator:
    """
    Medium-term EV demand simulator (months to multiple years, no fixed limit).

    Speed
    -----
    The inner simulation loop is **vectorised by context**: all days sharing
    the same (day_of_week, season) are batched into a single gmm.sample()
    call, reducing per-day Python overhead by ~20× on typical horizons.

    Parameters
    ----------
    gmm : EVSessionModel
        Fitted GMM for session-property sampling.
    base_sessions_per_day : float, optional
        Baseline daily session count. Inferred from gmm if not given.
    charger_mix : dict, optional
        Power distribution e.g. {7.4: 0.3, 22.0: 0.7} (auto-normalised).
    growth_model : str
        One of: 'linear', 's_curve', 'bass'. Set at construction time, not in
        `.simulate()`, which only takes the parameters of the chosen profile
        (anything else raises a ``ValueError``).
    n_scenarios : int
    seed : int

    Examples
    --------
    >>> sim = MediumTermSimulator(gmm, base_sessions_per_day=80, growth_model='bass')
    >>> result = sim.simulate(years=15, market_potential_factor=8)
    >>> sim2 = MediumTermSimulator(gmm, base_sessions_per_day=80, growth_model='s_curve')
    >>> result2 = sim2.simulate(years=15, saturation_factor=13)
    """

    def __init__(
        self,
        gmm,
        base_sessions_per_day: float | None = None,
        charger_mix: dict[float, float] | None = None,
        growth_model: str = "s_curve",
        n_scenarios: int = 10,
        seed: int = 42,
    ):
        self.gmm = gmm
        self.growth_model = growth_model
        self.n_scenarios = n_scenarios
        self.seed = seed

        cm = dict(charger_mix) if charger_mix else dict(DEFAULT_CHARGER_MIX)
        total = sum(cm.values())
        self.charger_mix = {k: v / total for k, v in cm.items()}

        self._base_given = base_sessions_per_day is not None
        if base_sessions_per_day is not None:
            self._base = float(base_sessions_per_day)
        elif gmm.n_sessions_per_day_:
            self._base = float(np.mean(list(gmm.n_sessions_per_day_.values())))
        else:
            self._base = 50.0

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate(
        self,
        years: float = 3,
        annual_growth_rate: float | None = None,
        start_date: str | pd.Timestamp | None = None,
        output: str = "daily_energy",
        weather_factor: dict[str, float] | None = None,
        n_scenarios: int | None = None,
        context: dict | None = None,
        **growth_kwargs,
    ) -> pd.DataFrame:
        """
        Run a medium-term simulation.

        Parameters
        ----------
        years : float
            Simulation horizon (any positive value, no upper limit).
        annual_growth_rate : float, optional
            Annual growth rate of the 'linear' growth model (default 0.15). The
            's_curve' and 'bass' models take their own shape parameters
            (``saturation_factor``/``midpoint_year``/``steepness`` or
            ``market_potential_factor``/``p``/``q``) through ``**growth_kwargs``
            below; giving them an ``annual_growth_rate`` raises a ``UserWarning``
            (it would change nothing).
        start_date : str or Timestamp, optional
            Start of the simulation. Defaults to today.
        output : str
            ``'daily_energy'`` — one row per (date, scenario) with kWh total.
            ``'hourly_energy'`` — one row per (date, hour, scenario).
            ``'sessions'``      — all individual sessions (memory-intensive).
        weather_factor : dict, optional
            Seasonal multipliers e.g. ``{'winter': 1.1, 'summer': 0.9}``.
        n_scenarios : int, optional
            Override the instance default.
        context : dict, optional
            The part of the context that dates cannot give, for a session model
            stratified by site: ``{"location_type": "work", "department": "75"}``.
            Every simulated day completes it with its own ``day_of_week`` and
            ``season``. Keys must be among the session model's ``stratify_by``.
            Unless ``base_sessions_per_day`` was given, the starting level is then
            the mean daily count of the fitted contexts that match ``context``,
            not of all of them.
        **growth_kwargs
            Shape parameters of the growth profile function (e.g.
            ``saturation_factor``, ``midpoint_year``, ``p``, ``q``). A name the
            growth model does not take raises a ``ValueError`` listing the valid ones.

        Returns
        -------
        pd.DataFrame
            Structure depends on *output*; see parameter description above. Rows
            are sorted by scenario, then date (then hour); with ``'daily_energy'``
            a day without any simulated session is a row with ``n_sessions = 0``.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If the session model is not fitted.
        ValueError
            If *output* or *growth_model* is not recognised, ``years`` is not a
            finite number > 0, ``annual_growth_rate`` is not a finite number
            > -1, ``n_scenarios`` is not an integer >= 1, ``start_date`` is
            not a valid date, ``context`` is not a dict or has a key the session
            model is not stratified by, or no fitted context matches it, or a
            ``growth_kwargs`` name is not a parameter of the growth model.

        Warns
        -----
        UserWarning
            ``annual_growth_rate`` given to a growth model other than 'linear'.
        """
        if output not in ("daily_energy", "hourly_energy", "sessions"):
            raise ValueError(
                f"output must be 'daily_energy', 'hourly_energy' or 'sessions', got {output!r}."
            )
        self.gmm._check_fitted()
        fixed = dict(context) if context else {}
        self.gmm._resolve_context(context, None)          # refuses a bad or unknown key now
        base = self._base_for(fixed)
        years = check_float(years, "years", gt=0, meaning="simulation horizon in years")
        if annual_growth_rate is not None and self.growth_model != "linear":
            warnings.warn(
                f"annual_growth_rate={annual_growth_rate} changes nothing with "
                f"growth_model={self.growth_model!r}, which takes its own shape parameters "
                f"({self._growth_parameters()}). Pass those instead.",
                UserWarning,
                stacklevel=2,
            )
        annual_growth_rate = check_float(
            0.15 if annual_growth_rate is None else annual_growth_rate,
            "annual_growth_rate", gt=-1,
            meaning="0.15 = +15% of the starting level per year; -1 would reach zero within a year",
        )
        n_sc = check_int(
            self.n_scenarios if n_scenarios is None else n_scenarios,
            "n_scenarios", minimum=1, meaning="number of stochastic scenarios",
        )
        start = (
            as_naive_timestamp(start_date, "start_date")
            if start_date is not None else pd.Timestamp.today().normalize()
        )
        rng = np.random.default_rng(self.seed)

        # Build growth profile
        profile_fn = GROWTH_PROFILES.get(self.growth_model)
        if profile_fn is None:
            raise ValueError(
                f"Unknown growth_model '{self.growth_model}'. "
                f"Available: {list(GROWTH_PROFILES)}"
            )

        # Shape parameters of the growth profile: refuse names it does not take
        valid_keys = self._growth_parameters(as_set=True)
        unknown = sorted(set(growth_kwargs) - valid_keys)
        if unknown:
            raise ValueError(
                f"Unknown growth parameter(s) {unknown} for growth_model="
                f"{self.growth_model!r}. It takes: {sorted(valid_keys)}."
            )
        profile_kwargs = dict(growth_kwargs)
        if "annual_growth_rate" in valid_keys:
            profile_kwargs["annual_growth_rate"] = annual_growth_rate

        growth = profile_fn(base, years, start_date=start, **profile_kwargs)

        seasonal_weights = {"winter": 1.0, "spring": 1.0, "summer": 1.0, "autumn": 1.0}
        if weather_factor:
            seasonal_weights.update(weather_factor)

        results = []
        for scenario_id in range(n_sc):
            sc_seed = int(rng.integers(0, 2**31))
            rows = self._simulate_scenario(
                growth=growth,
                seasonal_weights=seasonal_weights,
                output=output,
                scenario_id=scenario_id,
                seed=sc_seed,
                fixed_context=fixed,
            )
            results.extend(rows)

        if not results:
            return pd.DataFrame()

        # sessions and hourly_energy return lists of DataFrames; daily_energy returns dicts
        if output in ("sessions", "hourly_energy"):
            frame = pd.concat(results, ignore_index=True)
            order = ["scenario", "date"] + (["hour"] if output == "hourly_energy" else [])
            return frame.sort_values(order, kind="stable").reset_index(drop=True)
        return (
            pd.DataFrame(results)
            .sort_values(["scenario", "date"], kind="stable")
            .reset_index(drop=True)
        )

    def _growth_parameters(self, as_set: bool = False):
        """Names of the shape parameters of the growth model (``annual_growth_rate`` included)."""
        import inspect

        names = set(inspect.signature(GROWTH_PROFILES[self.growth_model]).parameters)
        names -= {"base_sessions_per_day", "years", "start_date"}
        return names if as_set else ", ".join(sorted(names))

    def _base_for(self, fixed: dict) -> float:
        """Starting sessions per day: the given one, or the mean over the contexts matching ``fixed``."""
        if self._base_given or not fixed:
            return self._base
        stratify = list(self.gmm.stratify_by)
        counts = [
            n for ctx, n in self.gmm.n_sessions_per_day_.items()
            if all(ctx[stratify.index(k)] == v for k, v in fixed.items())
        ]
        if not counts:
            raise ValueError(
                f"No fitted context matches {fixed}, so there is no starting level to grow from. "
                "Check the values (model.gmm_.bic_summary()['context'] lists the fitted contexts), "
                "or pass base_sessions_per_day."
            )
        return float(np.mean(counts))

    def _simulate_scenario(
        self,
        growth: pd.Series,
        seasonal_weights: dict,
        output: str,
        scenario_id: int,
        seed: int,
        fixed_context: dict | None = None,
    ) -> list:
        """
        Vectorised scenario: batch all same-context days together.

        For each unique (day_of_week, season) context we:
        1. Identify all dates with that context
        2. Compute the total expected sessions across those dates
        3. Call gmm.sample() ONCE for the whole batch
        4. Split the batch back to individual dates

        This is ~20× faster than looping over days individually.

        Parameters
        ----------
        growth : pd.Series
            Daily expected session counts from a growth profile function.
        seasonal_weights : dict
            Multiplicative seasonal adjustments keyed by season name.
        output : str
            One of ``'sessions'``, ``'daily_energy'``, ``'hourly_energy'``.
        scenario_id : int
            Scenario index written into every output row.
        seed : int
            Random seed for this scenario's draws.
        fixed_context : dict, optional
            Context keys that hold for every day (site type, département...);
            each batch adds its ``day_of_week`` / ``season``.

        Returns
        -------
        list
            List of dicts (for ``'daily_energy'``) or DataFrames (for
            ``'sessions'`` and ``'hourly_energy'``).
        """
        rng = np.random.default_rng(seed)
        rows = []

        # Build a frame with (date, day_of_week, season, expected_n)
        plan = pd.DataFrame({
            "date": growth.index,
            "expected_n": growth.values,
        })
        plan["day_of_week"] = plan["date"].dt.day_of_week
        plan["month"] = plan["date"].dt.month
        plan["season"] = plan["month"].apply(_season)
        plan["w"] = plan["season"].map(seasonal_weights).fillna(1.0)
        plan["adjusted_n"] = plan["expected_n"] * plan["w"]
        # Poisson draw per day to introduce realistic day-to-day count variability
        plan["n_sessions"] = rng.poisson(np.maximum(0, plan["adjusted_n"].values))

        if output == "daily_energy":
            # a day without any session is a day with zero energy, not a missing day
            for day in plan.loc[plan["n_sessions"] == 0, "date"]:
                rows.append({
                    "date": day.date(), "scenario": scenario_id, "n_sessions": 0,
                    "total_energy_kwh": 0.0, "mean_duration_h": np.nan,
                })

        # Group by context and batch-sample
        for (dow, season), group in plan.groupby(["day_of_week", "season"]):
            total_n = int(group["n_sessions"].sum())
            if total_n == 0:
                continue

            by_date = {"day_of_week": dow, "is_weekend": int(dow >= 5), "season": season}
            context = {
                **{k: v for k, v in by_date.items() if k in self.gmm.stratify_by},
                **(fixed_context or {}),
            }
            sessions_batch = self.gmm.sample(
                n_sessions=total_n,
                context=context,
                seed=int(rng.integers(0, 2**31)),
            )
            sessions_batch["power_kw"] = self._assign_power(total_n, rng)
            # Cap energy by power × duration (physical charger constraint)
            sessions_batch["energy"] = np.minimum(
                sessions_batch["energy"],
                sessions_batch["power_kw"] * sessions_batch["duration"],
            )

            # Distribute sessions back to individual dates
            cum = 0
            for _, day_row in group.iterrows():
                n_today = int(day_row["n_sessions"])
                if n_today == 0:
                    continue
                day_sessions = sessions_batch.iloc[cum: cum + n_today].copy()
                day_sessions["date"] = day_row["date"].date()
                day_sessions["scenario"] = scenario_id
                cum += n_today

                if output == "sessions":
                    rows.append(day_sessions)
                elif output == "daily_energy":
                    rows.append({
                        "date": day_row["date"].date(),
                        "scenario": scenario_id,
                        "n_sessions": n_today,
                        "total_energy_kwh": day_sessions["energy"].sum(),
                        "mean_duration_h": day_sessions["duration"].mean(),
                    })
                elif output == "hourly_energy":
                    hour_g = (
                        day_sessions.assign(hour=day_sessions["arrival_hour"].astype(int))
                        .groupby("hour")
                        .agg(energy_kwh=("energy", "sum"), n_sessions=("energy", "count"))
                        .reset_index()
                    )
                    hour_g["date"] = day_row["date"].date()
                    hour_g["scenario"] = scenario_id
                    rows.append(hour_g)

        return rows

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

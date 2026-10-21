"""
Medium-term EV demand simulator.

Generates aggregated daily energy demand over a configurable multi-year horizon
(no fixed limit) using GMM-based session sampling with EV adoption growth.

Growth profiles
---------------
Three growth models are available (see module-level functions). All three
are anchored so that ``growth(t=0) == base_sessions_per_day`` exactly — no
discontinuity between "current observed fleet" and the simulated trajectory
(see Session 6 fix note below).

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

Session 6 fix note
-------------------
Two earlier profiles (`s_curve_linear_tail_profile`, `double_s_curve_profile`)
were removed: with the base-anchoring and horizon-relative-timing fixes below
applied to `s_curve_growth_profile`, both added extra hyperparameters without
materially differentiated behaviour for this package's three canonical
adoption scenarios (conservative/central/ambitious). Two bugs were fixed in the three remaining profiles, both found by
tracing concrete numbers rather than assumed:
(1) `s_curve_growth_profile` and `bass_diffusion_profile` did not actually
start at `base_sessions_per_day` at t=0 (s_curve started at ~2-7% of the
asymptote depending on parameters; bass started at exactly 0) — a real
discontinuity when a scenario is meant to start "from the current observed
fleet, not zero". Both are now rescaled so growth(0) == base exactly.
(2) `s_curve_growth_profile`'s `midpoint_year`/`steepness` were fixed
absolute constants independent of the `years` horizon, so with the old
defaults the curve was 100% saturated by year 8 *regardless of whether a
3-year or a 20-year simulation was requested* — the actual mechanism behind
the "plateau" observed in notebook 3's long-term scenarios. Defaults are now
`years / 2` and `6 / years` respectively, so the curve reaches ~95% of its
asymptote at t=years for any requested horizon.

Simulation speed
----------------
The inner loop is vectorised by context: instead of calling gmm.sample()
for each individual day, days sharing the same (day_of_week, season) context
are batched together, dramatically reducing Python overhead on long horizons.
"""

from __future__ import annotations

import logging

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

    Genuinely linear in elapsed time: ``base * (1 + annual_growth_rate * t)``.
    (An earlier version of this function computed compound/exponential growth,
    ``base * (1 + rate) ** t``, despite its name — a real bug, not a design
    choice: fixed in Session 6, traced by comparing
    the two formulas' year-15 output directly.)

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
    asymptote. (A raw logistic curve with the old default parameters was only
    ~2.3% of its asymptote at t=0 regardless of `base_sessions_per_day` — a
    real discontinuity, not a design choice, when a scenario is meant to
    start "from the current observed fleet, not zero". Fixed in Session 6 by
    subtracting the raw curve's own t=0 value and rescaling the remainder to
    span exactly [base, base*saturation_factor].)

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
    asymptote. (The textbook Bass formulation models cumulative adopters of a
    brand-new product diffusing from zero — appropriate when there truly are
    no adopters yet, but wrong here: applied directly to
    `base_sessions_per_day`, it made the simulated trajectory start at a
    literal zero sessions/day, a hard discontinuity from whatever the real
    current baseline is. Confirmed by tracing the raw output: t=0 was exactly
    0 regardless of `base_sessions_per_day`. Fixed in Session 6 by treating
    the Bass cumulative-adoption fraction F(t) as tracking growth *beyond*
    the current base rather than adoption from scratch.)

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


# Growth profile registry
# Reduced from 5 to 3 in Session 6 (see module docstring "Session 6 fix note"
# for the justification: `s_curve_linear_tail_profile`
# and `double_s_curve_profile` added extra hyperparameters without materially
# differentiated behaviour once the base-anchoring and horizon-relative-timing
# bugs in `s_curve_growth_profile` were fixed, and neither mapped to any of
# this package's three canonical adoption scenarios).
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
        One of: 'linear', 's_curve', 'bass'. Set at construction time (not a
        `.simulate()` argument — passing `growth_model=` to `.simulate()` is
        silently ignored, since it isn't a growth-profile-function parameter;
        this was already true before Session 6 and the class-level example
        below was previously wrong about it).
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
        annual_growth_rate: float = 0.15,
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
        annual_growth_rate : float
            Annual growth rate, used only by the 'linear' growth model
            (silently ignored by 's_curve'/'bass', which take their own
            shape parameters — `saturation_factor`/`midpoint_year`/
            `steepness` or `market_potential_factor`/`p`/`q` — via
            `**growth_kwargs` below).
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
            Extra keyword arguments forwarded to the growth profile function
            (e.g. ``saturation_factor``, ``midpoint_year``, ``p``, ``q``).

        Returns
        -------
        pd.DataFrame
            Structure depends on *output*; see parameter description above.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If the session model is not fitted.
        ValueError
            If *output* or *growth_model* is not recognised, ``years`` is not a
            finite number > 0, ``annual_growth_rate`` is not a finite number
            > -1, ``n_scenarios`` is not an integer >= 1, ``start_date`` is
            not a valid date, ``context`` is not a dict or has a key the session
            model is not stratified by, or no fitted context matches it.
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
        annual_growth_rate = check_float(
            annual_growth_rate, "annual_growth_rate", gt=-1,
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

        # Build kwargs for profile function (only pass valid ones)
        import inspect
        sig = inspect.signature(profile_fn)
        valid_keys = set(sig.parameters) - {"base_sessions_per_day", "years", "start_date"}
        filtered_kwargs = {
            k: v for k, v in {
                "annual_growth_rate": annual_growth_rate,
                **growth_kwargs,
            }.items()
            if k in valid_keys
        }

        growth = profile_fn(base, years, start_date=start, **filtered_kwargs)

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
            return pd.concat(results, ignore_index=True)
        return pd.DataFrame(results)

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
        plan["day_of_week"] = plan["date"].dt.dayofweek
        plan["month"] = plan["date"].dt.month
        plan["season"] = plan["month"].apply(_season)
        plan["w"] = plan["season"].map(seasonal_weights).fillna(1.0)
        plan["adjusted_n"] = plan["expected_n"] * plan["w"]
        # Poisson draw per day to introduce realistic day-to-day count variability
        plan["n_sessions"] = rng.poisson(np.maximum(0, plan["adjusted_n"].values))

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

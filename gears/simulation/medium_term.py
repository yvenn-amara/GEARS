"""
Medium-term EV demand simulator.

Generates aggregated daily energy demand over a configurable multi-year horizon
(no fixed limit) using GMM-based session sampling with EV adoption growth.

Growth profiles
---------------
Five growth models are available (see module-level functions):

linear_growth_profile
    Simple year-on-year percentage growth.  Best for short horizons (1–3 yr)
    where penetration is still far from saturation.

s_curve_growth_profile
    Logistic (sigmoid) growth.  Captures the classic S-curve of technology
    adoption: slow start, rapid middle phase, plateau at saturation.

s_curve_linear_tail_profile
    S-curve that transitions to a small linear growth after saturation
    (slope = ``tail_rate`` × base_sessions/yr).  More realistic for EV
    charging than a pure plateau because new chargers and new users keep
    trickling in even after the initial wave saturates.

bass_diffusion_profile
    Bass (1969) diffusion model, the standard in technology adoption
    research.  Explicitly models two populations: *innovators* (p) who
    adopt independently, and *imitators* (q) who adopt after social contact.
    Typical EV parameters: p ≈ 0.03, q ≈ 0.38.

double_s_curve_profile
    Sum of two logistic curves offset in time.  Models two sequential
    adoption waves: e.g. early-adopters + premium cars first, then mass
    market + commercial vehicles.  Relevant for 2025–2040 where light and
    heavy EVs are on different trajectories.

Simulation speed
----------------
The inner loop is vectorised by context: instead of calling gmm.sample()
for each individual day, days sharing the same (day_of_week, season) context
are batched together, dramatically reducing Python overhead on long horizons.
"""

from __future__ import annotations

import logging
from typing import Optional, Union

import numpy as np
import pandas as pd

from gears.data.schemas import _season

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Growth profiles
# ---------------------------------------------------------------------------

def linear_growth_profile(
    base_sessions_per_day: float,
    years: float,
    annual_growth_rate: float = 0.15,
    start_date: Optional[Union[str, pd.Timestamp]] = None,
) -> pd.Series:
    """
    Daily session counts with constant year-on-year growth.

    Parameters
    ----------
    base_sessions_per_day : float
        Sessions per day at the start of the simulation (t = 0).
    years : float
        Simulation horizon (any positive value).
    annual_growth_rate : float
        Fractional annual growth (0.15 = +15%/yr).
    start_date : str or Timestamp, optional
        Start of the date index. Defaults to today.

    Returns
    -------
    pd.Series
        Daily expected session counts indexed by date.
    """
    start = pd.Timestamp(start_date) if start_date else pd.Timestamp.today().normalize()
    n_days = max(1, int(round(365.25 * years)))
    dates = pd.date_range(start, periods=n_days, freq="D")
    t = np.arange(n_days) / 365.25
    counts = base_sessions_per_day * (1 + annual_growth_rate) ** t
    return pd.Series(counts, index=dates, name="n_sessions_expected")


def s_curve_growth_profile(
    base_sessions_per_day: float,
    years: float,
    saturation_factor: float = 3.0,
    midpoint_year: float = 2.5,
    steepness: float = 1.5,
    start_date: Optional[Union[str, pd.Timestamp]] = None,
) -> pd.Series:
    """
    Logistic (S-curve) growth — technology adoption classic.

    Parameters
    ----------
    base_sessions_per_day : float
        Sessions per day at t = 0.
    years : float
        Simulation horizon.
    saturation_factor : float
        Max sessions as multiple of base (asymptote).
    midpoint_year : float
        Year at which growth rate is maximal.
    steepness : float
        Controls the slope of the S (larger = sharper transition).
    start_date : str or Timestamp, optional

    Returns
    -------
    pd.Series
        Daily expected session counts indexed by date.
    """
    start = pd.Timestamp(start_date) if start_date else pd.Timestamp.today().normalize()
    n_days = max(1, int(round(365.25 * years)))
    dates = pd.date_range(start, periods=n_days, freq="D")
    t = np.arange(n_days) / 365.25
    saturation = base_sessions_per_day * saturation_factor
    growth = saturation / (1 + np.exp(-steepness * (t - midpoint_year)))
    return pd.Series(growth, index=dates, name="n_sessions_expected")


def s_curve_linear_tail_profile(
    base_sessions_per_day: float,
    years: float,
    saturation_factor: float = 3.0,
    midpoint_year: float = 2.5,
    steepness: float = 1.5,
    tail_rate: float = 0.03,
    start_date: Optional[Union[str, pd.Timestamp]] = None,
) -> pd.Series:
    """
    S-curve with a linear tail after saturation.

    The logistic curve captures rapid early adoption; after the plateau
    a small linear slope ``tail_rate`` (fraction of base/yr) accounts for
    continued slow growth from new entrants and new geographies — more
    realistic than a hard asymptote.

    Parameters
    ----------
    base_sessions_per_day : float
    years : float
    saturation_factor : float
    midpoint_year : float
    steepness : float
    tail_rate : float
        Annual slope after saturation, as fraction of base_sessions_per_day.
        E.g. 0.03 = +3%/yr linear growth on top of the plateau.
    start_date : str or Timestamp, optional

    Returns
    -------
    pd.Series
        Daily expected session counts indexed by date.
    """
    s_curve = s_curve_growth_profile(
        base_sessions_per_day, years,
        saturation_factor=saturation_factor,
        midpoint_year=midpoint_year,
        steepness=steepness,
        start_date=start_date,
    )
    t = np.arange(len(s_curve)) / 365.25
    linear_tail = tail_rate * base_sessions_per_day * t
    combined = s_curve.values + linear_tail
    return pd.Series(combined, index=s_curve.index, name="n_sessions_expected")


def bass_diffusion_profile(
    base_sessions_per_day: float,
    years: float,
    market_potential_factor: float = 4.0,
    p: float = 0.03,
    q: float = 0.38,
    start_date: Optional[Union[str, pd.Timestamp]] = None,
) -> pd.Series:
    """
    Bass (1969) diffusion model for EV adoption.

    The Bass model describes how *innovators* (coefficient p) adopt
    independently of others, while *imitators* (coefficient q) adopt
    proportionally to the existing installed base.  This is the standard
    model in technology-diffusion research and fits EV adoption data well.

    Typical EV values: p ≈ 0.01–0.05 (innovators), q ≈ 0.3–0.5 (imitators).

    Parameters
    ----------
    base_sessions_per_day : float
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
    n_days = max(1, int(round(365.25 * years)))
    dates = pd.date_range(start, periods=n_days, freq="D")
    t = np.arange(n_days) / 365.25

    M = base_sessions_per_day * market_potential_factor
    # Cumulative adoption fraction F(t) from the closed-form Bass solution
    exponent = np.exp(-(p + q) * t)
    F = (1 - exponent) / (1 + (q / p) * exponent)
    N = M * F  # cumulative sessions
    return pd.Series(N, index=dates, name="n_sessions_expected")


def double_s_curve_profile(
    base_sessions_per_day: float,
    years: float,
    saturation_factor_1: float = 1.8,
    saturation_factor_2: float = 1.5,
    midpoint_year_1: float = 2.0,
    midpoint_year_2: float = 6.0,
    steepness_1: float = 2.0,
    steepness_2: float = 1.2,
    start_date: Optional[Union[str, pd.Timestamp]] = None,
) -> pd.Series:
    """
    Double S-curve — two sequential adoption waves.

    Models two distinct EV adoption cohorts.  For example:
    - Wave 1 (early, steeper): premium/early-adopter BEVs
    - Wave 2 (later, slower): mass-market / commercial vehicles

    Parameters
    ----------
    base_sessions_per_day : float
    years : float
    saturation_factor_1, saturation_factor_2 : float
        Saturation multipliers for each wave (relative to base).
    midpoint_year_1, midpoint_year_2 : float
        Year of maximum growth for each wave.
    steepness_1, steepness_2 : float
        Slope parameters for each wave.
    start_date : str or Timestamp, optional

    Returns
    -------
    pd.Series
        Daily expected session counts indexed by date (sum of both waves).
    """
    wave1 = s_curve_growth_profile(
        base_sessions_per_day, years,
        saturation_factor=saturation_factor_1,
        midpoint_year=midpoint_year_1,
        steepness=steepness_1,
        start_date=start_date,
    )
    # The second wave starts from a smaller base (30 % of base_sessions_per_day)
    # because it represents a secondary cohort that overlaps with the first.
    wave2 = s_curve_growth_profile(
        base_sessions_per_day * 0.3,
        years,
        saturation_factor=saturation_factor_2,
        midpoint_year=midpoint_year_2,
        steepness=steepness_2,
        start_date=start_date,
    )
    combined = wave1.values + wave2.values
    return pd.Series(combined, index=wave1.index, name="n_sessions_expected")


# Growth profile registry
GROWTH_PROFILES = {
    "linear": linear_growth_profile,
    "s_curve": s_curve_growth_profile,
    "s_curve_linear_tail": s_curve_linear_tail_profile,
    "bass": bass_diffusion_profile,
    "double_s_curve": double_s_curve_profile,
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
    gmm : EVSessionGMM
        Fitted GMM for session-property sampling.
    base_sessions_per_day : float, optional
        Baseline daily session count. Inferred from gmm if not given.
    charger_mix : dict, optional
        Power distribution e.g. {7.4: 0.3, 22.0: 0.7} (auto-normalised).
    growth_model : str
        One of: 'linear', 's_curve', 's_curve_linear_tail', 'bass', 'double_s_curve'.
    n_scenarios : int
    seed : int

    Examples
    --------
    >>> sim = MediumTermSimulator(gmm, base_sessions_per_day=80)
    >>> result = sim.simulate(years=5, growth_model='bass')
    >>> result = sim.simulate(years=15, growth_model='double_s_curve')
    """

    def __init__(
        self,
        gmm,
        base_sessions_per_day: Optional[float] = None,
        charger_mix: Optional[dict[float, float]] = None,
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
        start_date: Optional[Union[str, pd.Timestamp]] = None,
        output: str = "daily_energy",
        weather_factor: Optional[dict[str, float]] = None,
        n_scenarios: Optional[int] = None,
        **growth_kwargs,
    ) -> pd.DataFrame:
        """
        Run a medium-term simulation.

        Parameters
        ----------
        years : float
            Simulation horizon (any positive value, no upper limit).
        annual_growth_rate : float
            Annual growth rate for 'linear' and 's_curve*' models.
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
        **growth_kwargs
            Extra keyword arguments forwarded to the growth profile function
            (e.g. ``saturation_factor``, ``midpoint_year``, ``p``, ``q``).

        Returns
        -------
        pd.DataFrame
            Structure depends on *output*; see parameter description above.

        Raises
        ------
        ValueError
            If *output* or *growth_model* is not recognised.
        """
        if output not in ("daily_energy", "hourly_energy", "sessions"):
            raise ValueError("output must be 'daily_energy', 'hourly_energy', or 'sessions'.")

        n_sc = n_scenarios or self.n_scenarios
        start = pd.Timestamp(start_date) if start_date else pd.Timestamp.today().normalize()
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

        growth = profile_fn(self._base, years, start_date=start, **filtered_kwargs)

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
            )
            results.extend(rows)

        if not results:
            return pd.DataFrame()

        # sessions and hourly_energy return lists of DataFrames; daily_energy returns dicts
        if output in ("sessions", "hourly_energy"):
            return pd.concat(results, ignore_index=True)
        return pd.DataFrame(results)

    def _simulate_scenario(
        self,
        growth: pd.Series,
        seasonal_weights: dict,
        output: str,
        scenario_id: int,
        seed: int,
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

            context = {"day_of_week": dow, "season": season}
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

    # ------------------------------------------------------------------
    # Plotting helpers
    # ------------------------------------------------------------------

    def plot_energy_trajectory(
        self,
        result: pd.DataFrame,
        ax=None,
        resample: str = "ME",
        figsize: tuple = (14, 5),
        title: str = "Medium-term energy trajectory",
        color: str = "#2E86AB",
    ):
        """
        Monthly energy trajectory with 50 % and 80 % CI fan chart.

        Parameters
        ----------
        result : pd.DataFrame
            Output of :meth:`simulate` with ``output='daily_energy'``.
            Must contain columns: date, scenario, total_energy_kwh.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. Created if None.
        resample : str
            Pandas offset alias for temporal aggregation.
            ``'ME'`` = month-end (default); ``'QE'`` = quarter-end.
        figsize : tuple
            Figure size in inches, used only when creating a new figure.
        title : str
            Plot title.
        color : str
            Hex colour for the median line and CI shading.

        Returns
        -------
        matplotlib.axes.Axes
            The axes containing the fan chart.
        """
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        result = result.copy()
        result["date"] = pd.to_datetime(result["date"])

        monthly = (
            result.set_index("date")
            .groupby("scenario")["total_energy_kwh"]
            .resample(resample)
            .sum()
            .reset_index()
        )
        pivot = monthly.pivot(index="date", columns="scenario", values="total_energy_kwh")

        fig, ax_ = (None, ax) if ax is not None else plt.subplots(figsize=figsize)
        ax_ = ax_ or plt.gca()

        ax_.fill_between(pivot.index, pivot.quantile(0.10, axis=1),
                         pivot.quantile(0.90, axis=1),
                         alpha=0.15, color=color, label="80% CI")
        ax_.fill_between(pivot.index, pivot.quantile(0.25, axis=1),
                         pivot.quantile(0.75, axis=1),
                         alpha=0.30, color=color, label="50% CI")
        pivot.median(axis=1).plot(ax=ax_, label="Median", color=color, linewidth=2)

        ax_.set_ylabel("Monthly energy (kWh)")
        ax_.set_title(title)
        ax_.legend()
        ax_.grid(True, alpha=0.3)
        ax_.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax_.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax_.xaxis.get_majorticklabels(), rotation=30, ha="right")
        if fig:
            fig.tight_layout()
        return ax_

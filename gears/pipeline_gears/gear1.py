"""
Gear1Backend – GEAR 1st: GMM/VAE session modeling, SARIMA/probabilistic
session-count forecasting, short- and medium-term simulation, and V1G smart
charging.

Use it through :class:`gears.pipeline.GEARSModel` (``GEARSModel(gear=1, ...)``,
or simply ``GEARSModel(...)`` since 1 is the default gear).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from gears._logging import console_logging
from gears._validation import as_naive_timestamp, check_int, not_fitted_error
from gears.data.loader import load_sessions
from gears.models.forecaster import SessionForecaster
from gears.models.session_model import MIN_SESSIONS_PER_CONTEXT, EVSessionModel
from gears.output.aggregator import OutputAggregator
from gears.simulation.medium_term import MediumTermSimulator
from gears.simulation.short_term import ShortTermSimulator
from gears.smart_charging.optimizer import SmartChargingOptimizer

logger = logging.getLogger(__name__)

_FORECASTER_METHODS = ("sarima", "probabilistic")


class Gear1Backend:
    """
    GEAR 1st pipeline: GMM/VAE session modeling + SARIMA/probabilistic
    forecasting + simulation + smart charging.

    Not part of the public API directly — construct and use this via
    :class:`gears.pipeline.GEARSModel` (e.g. ``GEARSModel(gear=1, ...)``,
    or simply ``GEARSModel(...)`` since 1 is the default gear).

    Parameters
    ----------
    n_components : int or str
        GMM/VAE-context components passed to
        :class:`~gears.models.session_model.EVSessionModel`.
        Use ``'auto'`` for BIC-based selection (GMM only).
    max_components : int
        Upper bound of the BIC search when ``n_components='auto'``.
    stratify_by : list of str, optional
        Context columns for session-model stratification.
        Default: ``['day_of_week', 'season']``.
    forecaster_method : str
        ``'sarima'`` (recommended; always tried, and if it cannot be fitted — very
        short history — or pmdarima cannot be imported, a warning announces the
        substitute used, see :class:`~gears.models.forecaster.SessionForecaster`) or
        ``'probabilistic'`` (fast, no extra deps).
    charger_mix : dict, optional
        Charger power distribution, e.g. ``{7.4: 0.3, 22.0: 0.7}``.
        Use :data:`~gears.simulation.medium_term.CHARGER_PRESETS` for
        predefined mixes.
    n_scenarios : int
        Default number of stochastic scenarios.
    resolution_min : int
        Time resolution for load curves (minutes).
    max_samples_per_context : int or None
        Max training samples per context (for tractability on large
        datasets).
    forecaster_use_holidays : bool
        Passed through to :class:`~gears.models.forecaster.SessionForecaster`.
        Default ``True``, matching that class's own default.
    forecaster_country : str
        ISO 3166-1 alpha-2 country code for the SARIMA holiday exogenous
        regressor, passed through to
        :class:`~gears.models.forecaster.SessionForecaster`. Default ``'FR'``
        for backward compatibility — override this for non-French datasets
        (e.g. ``'US'``, ``'GB'``) so the holiday calendar actually matches
        the data.
    random_state : int
        Master random seed.
    model_type : {'gmm', 'vae'}
        Session-model family, forwarded to
        :class:`~gears.models.session_model.EVSessionModel`. Default
        ``'gmm'``, matching that class's own default.
    recency : bool or None
        If truthy, fit each context on a recency-weighted bootstrap resample
        instead of a uniform one (GMM path only — ignored with a warning for
        ``model_type='vae'``). Forwarded to
        :class:`~gears.models.session_model.EVSessionModel`. Default
        ``None`` (disabled).
    half_life_days : float or None
        Half-life of the exponential recency decay, only used when
        ``recency`` is truthy. Forwarded to
        :class:`~gears.models.session_model.EVSessionModel`. Default
        ``None`` (auto-derived per context group).
    """

    def __init__(
        self,
        n_components: int | str = "auto",
        max_components: int = 10,
        stratify_by: list[str] | None = None,
        forecaster_method: str = "sarima",
        charger_mix: dict[float, float] | None = None,
        n_scenarios: int = 10,
        resolution_min: int = 30,
        max_samples_per_context: int | None = None,
        forecaster_use_holidays: bool = True,
        forecaster_country: str = "FR",
        random_state: int = 42,
        model_type: Literal["gmm", "vae"] = "gmm",
        recency: bool | None = None,
        half_life_days: float | None = None,
    ):
        if forecaster_method not in _FORECASTER_METHODS:
            raise ValueError(
                f"forecaster_method must be one of {list(_FORECASTER_METHODS)}, "
                f"got {forecaster_method!r}."
            )
        n_scenarios = check_int(
            n_scenarios, "n_scenarios", minimum=1, meaning="default number of scenarios"
        )
        self.n_components = n_components
        self.max_components = max_components
        self.stratify_by = stratify_by or ["day_of_week", "season"]
        self.forecaster_method = forecaster_method
        self.charger_mix = charger_mix or {7.4: 0.5, 22.0: 0.5}
        self.n_scenarios = n_scenarios
        self.resolution_min = resolution_min
        self.max_samples_per_context = max_samples_per_context
        self.forecaster_use_holidays = forecaster_use_holidays
        self.forecaster_country = forecaster_country
        self.random_state = random_state
        self.model_type = model_type
        self.recency = recency
        self.half_life_days = half_life_days

        self.gmm_: EVSessionModel | None = None
        self.forecaster_: SessionForecaster | None = None
        self._short_sim: ShortTermSimulator | None = None
        self.aggregator_ = OutputAggregator(resolution_min=resolution_min)
        self.metadata_: dict = {}
        self.is_fitted_: bool = False

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        cache_dir: str | Path | None = None,
        path: str | Path | None = None,
        **kwargs,
    ) -> Gear1Backend:
        """Build a backend from a pre-trained session model."""
        from gears.pretrained import load_pretrained

        session_model = load_pretrained(model_id, path=path, cache_dir=cache_dir)

        instance = cls(**kwargs)
        instance.gmm_ = session_model
        instance.forecaster_ = SessionForecaster(method="probabilistic")
        instance.metadata_ = {"pretrained_model_id": model_id}
        instance._build_simulators()
        instance.is_fitted_ = True

        logger.info("Loaded pre-trained model '%s'.", model_id)
        return instance

    @classmethod
    def from_session_model(
        cls,
        session_model: EVSessionModel,
        **kwargs,
    ) -> Gear1Backend:
        """Build a Gear1Backend from a fitted session model."""
        instance = cls(**kwargs)
        instance.gmm_ = session_model
        instance.forecaster_ = SessionForecaster(method="probabilistic")
        instance.metadata_ = {"source": "provided_session_model"}
        instance._build_simulators()
        instance.is_fitted_ = True
        return instance

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        data: str | Path | pd.DataFrame,
        strict: bool = False,
        filter_failed: bool = True,
        verbose: bool = True,
        recent_months: int | None = None,
        **loader_kwargs,
    ) -> Gear1Backend:
        """Fit the full GEAR 1st pipeline on a sessions dataset."""
        with console_logging(verbose):
            df = load_sessions(
                data, strict=strict, filter_failed=filter_failed,
                verbose=verbose, **loader_kwargs
            )

            logger.info("Fitting %s on %d sessions …", self.model_type.upper(), len(df))
            self.gmm_ = EVSessionModel(
                n_components=self.n_components,
                max_components=self.max_components,
                stratify_by=self.stratify_by,
                max_samples_per_context=self.max_samples_per_context,
                recent_months=recent_months,
                random_state=self.random_state,
                model_type=self.model_type,
                recency=self.recency,
                half_life_days=self.half_life_days,
            ).fit(df)

            logger.info("Fitting session-count forecaster (%s) …", self.forecaster_method)
            self.forecaster_ = SessionForecaster(
                method=self.forecaster_method,
                use_holidays=self.forecaster_use_holidays,
                country=self.forecaster_country,
                random_state=self.random_state,
            ).fit(df)

            self._build_simulators()
            self.is_fitted_ = True

            logger.info("Fitting complete ✓")
            logger.info("%r", self.gmm_)
        return self

    def _build_simulators(self) -> None:
        """Instantiate the short-term simulator from fitted components."""
        self._short_sim = ShortTermSimulator(
            gmm=self.gmm_,
            forecaster=self.forecaster_,
            charger_mix=self.charger_mix,
        )

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate_short_term(
        self,
        start_date: str | pd.Timestamp,
        horizon: int = 7,
        n_scenarios: int | None = None,
        n_sessions: int | None = None,
        seed: int | None = None,
        context: dict | None = None,
        strict: bool = False,
    ) -> pd.DataFrame:
        """
        Simulate individual EV sessions for a short-term horizon.

        Parameters
        ----------
        start_date : str or Timestamp
            First day, read as local wall-clock time (a time zone is dropped,
            not converted). Simulated ``arrival_time`` values are naive local
            timestamps.
        horizon : int
            Number of days (an integer >= 1).
        n_scenarios : int, optional
            Number of scenarios (an integer >= 1); defaults to the value given
            at construction. Ignored when ``n_sessions`` is given.
        n_sessions : int, optional
            Simulate exactly this many sessions per day (one scenario, no
            ``scenario`` column) instead of drawing daily counts from the
            forecaster.
        seed : int, optional
            Random seed. The same seed gives the same sessions; ``None`` draws
            fresh randomness on every call.
        context : dict, optional
            Which context to simulate, for a model fitted per site:
            ``{"location_type": "work", "department": "75"}`` (the pre-trained
            French model has both). Each day completes it with its own
            ``day_of_week`` and ``season``; a key you give wins over the date's.
            Keys must be among the session model's ``stratify_by``
            (``model.gmm_.stratify_by``). Without ``context``, a model fitted
            per site cannot know which site you mean and serves the most
            populous one, with a warning. The daily counts come from the
            forecaster, which was fitted on all the sessions: to simulate one
            site, fit on that site's sessions, or give ``n_sessions``.
        strict : bool, default False
            Raise ``ValueError`` instead of warning when the session model has
            no mixture for a day's context and the closest one would be used.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If the model is not fitted.
        ValueError
            If ``horizon``, ``n_scenarios`` or ``n_sessions`` is not an integer
            >= 1, ``start_date`` is not a valid date, or ``context`` is not a
            dict or has a key the session model is not stratified by.
        """
        self._check_fitted()
        horizon = check_int(horizon, "horizon", minimum=1, meaning="number of days to simulate")
        n_sc = check_int(
            self.n_scenarios if n_scenarios is None else n_scenarios,
            "n_scenarios", minimum=1, meaning="number of stochastic scenarios",
        )
        start = as_naive_timestamp(start_date, "start_date")

        if n_sessions is not None:
            n_sessions = check_int(n_sessions, "n_sessions", minimum=1, meaning="sessions per day")
            # One independent seed per day: consecutive seeds must not share days.
            day_seeds = np.random.default_rng(seed).integers(0, 2**31, size=horizon)
            frames = [
                self._short_sim.simulate_single_day(
                    date=start + pd.Timedelta(days=day_offset),
                    n_sessions=n_sessions,
                    seed=int(day_seeds[day_offset]),
                    context=context,
                    strict=strict,
                )
                for day_offset in range(horizon)
            ]
            return pd.concat(frames, ignore_index=True)

        return self._short_sim.simulate(
            start_date=start,
            horizon=horizon,
            n_scenarios=n_sc,
            seed=seed,
            context=context,
            strict=strict,
        )

    def simulate_medium_term(
        self,
        years: float = 3,
        annual_growth_rate: float = 0.15,
        start_date: str | pd.Timestamp | None = None,
        output: str = "daily_energy",
        weather_factor: dict[str, float] | None = None,
        growth_model: str = "linear",
        n_scenarios: int | None = None,
        charger_mix: dict[float, float] | None = None,
        saturation_factor: float = 3.0,
        context: dict | None = None,
    ) -> pd.DataFrame:
        """Simulate aggregated EV demand over a multi-year horizon.

        ``context`` chooses the site of a model fitted per site, as in
        :meth:`simulate_short_term` (e.g. ``{"location_type": "work",
        "department": "75"}``); the starting level is then the mean daily
        count of the matching fitted contexts.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If the model is not fitted.
        ValueError
            If ``years`` is not > 0, ``annual_growth_rate`` is <= -1,
            ``n_scenarios`` is not an integer >= 1, ``start_date`` is not a valid
            date, or ``growth_model`` / ``output`` is not recognised.
        """
        self._check_fitted()
        sim = MediumTermSimulator(
            gmm=self.gmm_,
            charger_mix=charger_mix or self.charger_mix,
            growth_model=growth_model,
            n_scenarios=self.n_scenarios if n_scenarios is None else n_scenarios,
            seed=self.random_state,
        )
        return sim.simulate(
            years=years,
            annual_growth_rate=annual_growth_rate,
            start_date=start_date,
            output=output,
            weather_factor=weather_factor,
            context=context,
            saturation_factor=saturation_factor,
        )

    # ------------------------------------------------------------------
    # Smart charging
    # ------------------------------------------------------------------

    def smart_charge(
        self,
        sessions: pd.DataFrame,
        signal: pd.Series,
        signal_type: str = "price",
    ) -> pd.DataFrame:
        """Apply V1G smart charging optimisation to a sessions DataFrame."""
        self._check_fitted()
        opt = SmartChargingOptimizer(
            signal_type=signal_type,
            resolution_min=self.resolution_min,
        )
        result = opt.optimise(sessions, signal)
        summary = opt.savings_summary(result)
        logger.info("Smart charging summary: %s", summary)
        return result

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def daily_energy(self, sessions: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Shortcut for :meth:`aggregator_.daily_energy`."""
        return self.aggregator_.daily_energy(sessions, **kwargs)

    def hourly_profile(self, sessions: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Shortcut for :meth:`aggregator_.hourly_profile`."""
        return self.aggregator_.hourly_profile(sessions, **kwargs)

    def export(self, df: pd.DataFrame, path: str | Path, **kwargs) -> None:
        """Shortcut for :meth:`aggregator_.export`."""
        self.aggregator_.export(df, path, **kwargs)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        """Raise ``NotFittedError`` if the model has not been fitted yet."""
        if not self.is_fitted_:
            raise not_fitted_error(
                "GEARSModel",
                "Call .fit(data) first, or use GEARSModel.from_pretrained('french').",
            )

    def summary(self) -> str:
        """Return a human-readable summary of the fitted model."""
        lines = ["=" * 55, "GEARS Model Summary", "=" * 55]
        if not self.is_fitted_:
            lines.append("Status: NOT FITTED")
        else:
            lines.append("Status     : fitted ✓")
            lines.append(f"GMM        : {self.gmm_}")
            lines.extend(self._context_coverage_lines())
            lines.append(f"Forecaster : {self.forecaster_}")
            lines.append(f"Charger mix: {self.charger_mix}")
            lines.append(f"Scenarios  : {self.n_scenarios}")
            if self.metadata_:
                lines.append(f"Metadata   : {self.metadata_}")
        lines.append("=" * 55)
        return "\n".join(lines)

    def _context_coverage_lines(self) -> list[str]:
        """Summary lines on sparse contexts and on substitutions made so far."""
        gmm = self.gmm_
        lines = []
        skipped = getattr(gmm, "skipped_contexts_", None)
        if skipped:
            n_fitted = len(gmm.models_)
            lines.append(
                f"Contexts   : {n_fitted} fitted; {len(skipped)} with < "
                f"{MIN_SESSIONS_PER_CONTEXT} sessions have no model of their own and use "
                "the closest fitted context (model.gmm_.skipped_contexts_)"
            )
        fallbacks = getattr(gmm, "fallback_contexts_", None)
        if fallbacks:
            lines.append(
                f"Fallbacks  : {len(fallbacks)} requested context(s) not fitted, "
                "closest used (model.gmm_.fallback_contexts_)"
            )
        return lines

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted_ else "not fitted"
        return (
            f"Gear1Backend(n_components={self.n_components!r}, "
            f"stratify_by={self.stratify_by}, "
            f"forecaster={self.forecaster_method!r}, "
            f"model_type={self.model_type!r}, "
            f"status={status})"
        )

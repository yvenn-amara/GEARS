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

import pandas as pd

from gears._logging import console_logging
from gears.data.loader import load_sessions
from gears.models.forecaster import SessionForecaster
from gears.models.session_model import EVSessionModel
from gears.output.aggregator import OutputAggregator
from gears.simulation.medium_term import MediumTermSimulator
from gears.simulation.short_term import ShortTermSimulator
from gears.smart_charging.optimizer import SmartChargingOptimizer

logger = logging.getLogger(__name__)


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
        ``'sarima'`` (recommended, requires pmdarima) or
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
        **kwargs,
    ) -> Gear1Backend:
        """Build a backend from a pre-trained session model."""
        from gears.pretrained import load_pretrained

        session_model = load_pretrained(model_id, cache_dir=cache_dir)

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
    ) -> pd.DataFrame:
        """Simulate individual EV sessions for a short-term horizon."""
        self._check_fitted()
        n_sc = n_scenarios or self.n_scenarios

        if n_sessions is not None:
            frames = []
            for day_offset in range(horizon):
                day = pd.Timestamp(start_date) + pd.Timedelta(days=day_offset)
                day_df = self._short_sim.simulate_single_day(
                    date=day, n_sessions=n_sessions,
                    seed=(seed or 0) + day_offset,
                )
                frames.append(day_df)
            return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

        return self._short_sim.simulate(
            start_date=start_date,
            horizon=horizon,
            n_scenarios=n_sc,
            seed=seed,
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
    ) -> pd.DataFrame:
        """Simulate aggregated EV demand over a multi-year horizon."""
        self._check_fitted()
        sim = MediumTermSimulator(
            gmm=self.gmm_,
            charger_mix=charger_mix or self.charger_mix,
            growth_model=growth_model,
            n_scenarios=n_scenarios or self.n_scenarios,
            seed=self.random_state,
        )
        return sim.simulate(
            years=years,
            annual_growth_rate=annual_growth_rate,
            start_date=start_date,
            output=output,
            weather_factor=weather_factor,
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
        """Raise RuntimeError if the model has not been fitted yet."""
        if not self.is_fitted_:
            raise RuntimeError(
                "This GEARSModel instance is not fitted yet. "
                "Call .fit(data) or use GEARSModel.from_pretrained(model_id)."
            )

    def summary(self) -> str:
        """Return a human-readable summary of the fitted model."""
        lines = ["=" * 55, "GEARS Model Summary", "=" * 55]
        if not self.is_fitted_:
            lines.append("Status: NOT FITTED")
        else:
            lines.append("Status     : fitted ✓")
            lines.append(f"GMM        : {self.gmm_}")
            lines.append(f"Forecaster : {self.forecaster_}")
            lines.append(f"Charger mix: {self.charger_mix}")
            lines.append(f"Scenarios  : {self.n_scenarios}")
            if self.metadata_:
                lines.append(f"Metadata   : {self.metadata_}")
        lines.append("=" * 55)
        return "\n".join(lines)

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted_ else "not fitted"
        return (
            f"Gear1Backend(n_components={self.n_components!r}, "
            f"stratify_by={self.stratify_by}, "
            f"forecaster={self.forecaster_method!r}, "
            f"model_type={self.model_type!r}, "
            f"status={status})"
        )

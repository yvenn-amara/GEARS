"""
GEARSModel – unified facade for the full GEARS pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from gears.data.loader import load_sessions
from gears.models.forecaster import SessionForecaster
from gears.models.gmm import EVSessionGMM
from gears.models.registry import ModelRegistry, NativeGMMRegistry
from gears.output.aggregator import OutputAggregator
from gears.simulation.medium_term import MediumTermSimulator
from gears.simulation.short_term import ShortTermSimulator
from gears.smart_charging.optimizer import SmartChargingOptimizer

logger = logging.getLogger(__name__)


class GEARSModel:
    """
    Unified GEARS pipeline object.

    Combines GMM fitting, session-count forecasting, short- and medium-term
    simulation, V1G smart charging, and output aggregation behind a single
    facade.

    Parameters
    ----------
    n_components : int or str
        GMM components passed to :class:`~gears.models.gmm.EVSessionGMM`.
        Use ``'auto'`` for BIC-based selection.
    stratify_by : list of str, optional
        Context columns for GMM stratification.
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
        Max GMM training samples per context (for tractability on large
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

    Examples
    --------
    >>> model = GEARSModel()
    >>> model.fit("data/sessions_france.pkl")
    >>> sessions = model.simulate_short_term("2025-06-10", horizon=7)

    >>> model = GEARSModel.from_pretrained("work_fr_demo")
    >>> sessions = model.simulate_short_term("2025-06-10")

    >>> energy = model.simulate_medium_term(years=10, annual_growth_rate=0.15)
    """

    def __init__(
        self,
        n_components: int | str = "auto",
        stratify_by: list[str] | None = None,
        forecaster_method: str = "sarima",
        charger_mix: dict[float, float] | None = None,
        n_scenarios: int = 10,
        resolution_min: int = 30,
        max_samples_per_context: int | None = None,
        forecaster_use_holidays: bool = True,
        forecaster_country: str = "FR",
        random_state: int = 42,
    ):
        self.n_components = n_components
        self.stratify_by = stratify_by or ["day_of_week", "season"]
        self.forecaster_method = forecaster_method
        self.charger_mix = charger_mix or {7.4: 0.5, 22.0: 0.5}
        self.n_scenarios = n_scenarios
        self.resolution_min = resolution_min
        self.max_samples_per_context = max_samples_per_context
        self.forecaster_use_holidays = forecaster_use_holidays
        self.forecaster_country = forecaster_country
        self.random_state = random_state

        self.gmm_: EVSessionGMM | None = None
        self.forecaster_: SessionForecaster | None = None
        self._short_sim: ShortTermSimulator | None = None
        self._medium_sim: MediumTermSimulator | None = None
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
        hf_repo_id: str | None = None,
        cache_dir: str | Path | None = None,
        **kwargs,
    ) -> GEARSModel:
        """
        Load a pre-trained model bundle from the GEARS registry.

        Parameters
        ----------
        model_id : str
            Registry model identifier (e.g. ``'work_fr_demo'``).
        hf_repo_id : str, optional
            Override the default Hugging Face Hub repository.
        cache_dir : str or Path, optional
            Local cache directory for downloaded artefacts.
        **kwargs
            Forwarded to the :class:`GEARSModel` constructor.

        Returns
        -------
        GEARSModel
            Fitted instance loaded from the registry.
        """
        registry_kwargs = {}
        if hf_repo_id:
            registry_kwargs["hf_repo_id"] = hf_repo_id
        if cache_dir:
            registry_kwargs["cache_dir"] = cache_dir

        registry = ModelRegistry(**registry_kwargs)
        bundle = registry.load(model_id)

        instance = cls(**kwargs)
        instance.gmm_ = bundle["gmm"]
        instance.forecaster_ = bundle.get("forecaster") or SessionForecaster(
            method="probabilistic"
        )
        instance.metadata_ = bundle.get("metadata", {})
        instance._build_simulators()
        instance.is_fitted_ = True

        logger.info("Loaded pre-trained model '%s'.", model_id)
        return instance

    @classmethod
    def from_native_gmm(
        cls,
        gmm_id: str = "french",
        gmm_dir: Path | None = None,
        **kwargs,
    ) -> GEARSModel:
        """
        Build a GEARSModel from the pre-fitted unified French GMM.

        Parameters
        ----------
        gmm_id : str
            Registry bundle ID.  Currently only ``'french'`` is available —
            it contains all location types stratified by
            ``location_type × département × season × day_of_week``.
        gmm_dir : Path, optional
            Override the default GMM directory.
        **kwargs
            Forwarded to the :class:`GEARSModel` constructor.

        Returns
        -------
        GEARSModel
            Fitted instance using the native GMM bundle.
        """
        reg = NativeGMMRegistry(gmm_dir=gmm_dir)
        gmm = reg.load(gmm_id)

        instance = cls(**kwargs)
        instance.gmm_ = gmm
        instance.forecaster_ = SessionForecaster(method="probabilistic")
        instance.metadata_ = {"native_gmm_id": gmm_id, **gmm.metadata_}
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
    ) -> GEARSModel:
        """
        Fit the full GEARS pipeline on a sessions dataset.

        Parameters
        ----------
        data : str, Path, or pd.DataFrame
            Raw sessions data (any supported format, including French pkl).
        strict : bool
            Raise on data quality issues instead of dropping rows.
        filter_failed : bool
            French data: drop failed sessions (``succes_session != 't'``).
        verbose : bool
            Print loading and fitting progress.
        recent_months : int, optional
            Only use the most recent N months for GMM fitting.
        **loader_kwargs
            Passed to :func:`~gears.data.loader.load_sessions`
            (e.g. ``sep=';'``).

        Returns
        -------
        GEARSModel
            The fitted instance (``self``).
        """
        df = load_sessions(
            data, strict=strict, filter_failed=filter_failed,
            verbose=verbose, **loader_kwargs
        )

        if verbose:
            print(f"[GEARS] Fitting GMM on {len(df):,} sessions …")
        self.gmm_ = EVSessionGMM(
            n_components=self.n_components,
            stratify_by=self.stratify_by,
            max_samples_per_context=self.max_samples_per_context,
            recent_months=recent_months,
            random_state=self.random_state,
        ).fit(df)

        if verbose:
            print(f"[GEARS] Fitting session-count forecaster ({self.forecaster_method}) …")
        self.forecaster_ = SessionForecaster(
            method=self.forecaster_method,
            use_holidays=self.forecaster_use_holidays,
            country=self.forecaster_country,
            random_state=self.random_state,
        ).fit(df)

        self._build_simulators()
        self.is_fitted_ = True

        if verbose:
            print("[GEARS] Fitting complete ✓")
            print(repr(self.gmm_))
        return self

    def _build_simulators(self) -> None:
        """Instantiate short- and medium-term simulators from fitted components."""
        self._short_sim = ShortTermSimulator(
            gmm=self.gmm_,
            forecaster=self.forecaster_,
            charger_mix=self.charger_mix,
        )
        self._medium_sim = MediumTermSimulator(
            gmm=self.gmm_,
            charger_mix=self.charger_mix,
            n_scenarios=self.n_scenarios,
            seed=self.random_state,
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
        """
        Simulate individual EV sessions for a short-term horizon.

        Parameters
        ----------
        start_date : str or pd.Timestamp
            First day of the simulation.
        horizon : int
            Number of days.
        n_scenarios : int, optional
            Override the default number of scenarios.
        n_sessions : int, optional
            Fixed session count per day (bypasses the forecaster).
        seed : int, optional
            Random seed override.

        Returns
        -------
        pd.DataFrame
            One row per session with canonical GEARS columns.
        """
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
        """
        Simulate aggregated EV demand over a multi-year horizon.

        Parameters
        ----------
        years : float
            Horizon in years. No upper limit.
        annual_growth_rate : float
            Annual EV penetration growth (e.g. 0.15 = +15 %/yr).
        start_date : str or pd.Timestamp, optional
            Defaults to today.
        output : str
            ``'daily_energy'``, ``'hourly_energy'``, or ``'sessions'``.
        weather_factor : dict, optional
            Seasonal modifiers, e.g. ``{'winter': 1.1, 'summer': 0.9}``.
        growth_model : str
            ``'linear'`` or ``'s_curve'``.
        n_scenarios : int, optional
            Override the default number of scenarios.
        charger_mix : dict, optional
            Override the default charger mix for this simulation.
        saturation_factor : float
            S-curve only: maximum sessions as a multiple of the base.

        Returns
        -------
        pd.DataFrame
            Simulation output with columns depending on ``output``:
            ``date``, ``scenario``, and ``total_energy_kwh`` (or
            ``total_sessions`` for ``output='sessions'``).
        """
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
        """
        Apply V1G smart charging optimisation to a sessions DataFrame.

        Parameters
        ----------
        sessions : pd.DataFrame
            Output of :meth:`simulate_short_term`.
        signal : pd.Series
            Price (€/kWh) or RES fraction signal with a ``DatetimeIndex``.
        signal_type : str
            ``'price'`` or ``'res'``.

        Returns
        -------
        pd.DataFrame
            Sessions DataFrame with scheduling columns appended
            (``cost_smart``, ``cost_plug``, ``savings_pct``,
            ``scheduled_start``, ``scheduled_end``).
        """
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
        """
        Compute daily energy totals.

        Shortcut for :meth:`aggregator_.daily_energy`.

        Parameters
        ----------
        sessions : pd.DataFrame
        **kwargs
            Forwarded to
            :meth:`~gears.output.aggregator.OutputAggregator.daily_energy`.

        Returns
        -------
        pd.DataFrame
        """
        return self.aggregator_.daily_energy(sessions, **kwargs)

    def hourly_profile(self, sessions: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Compute hourly load profiles.

        Shortcut for :meth:`aggregator_.hourly_profile`.

        Parameters
        ----------
        sessions : pd.DataFrame
        **kwargs
            Forwarded to
            :meth:`~gears.output.aggregator.OutputAggregator.hourly_profile`.

        Returns
        -------
        pd.DataFrame
        """
        return self.aggregator_.hourly_profile(sessions, **kwargs)

    def export(self, df: pd.DataFrame, path: str | Path, **kwargs) -> None:
        """
        Export a DataFrame to file.

        Shortcut for :meth:`aggregator_.export`.

        Parameters
        ----------
        df : pd.DataFrame
        path : str or Path
        **kwargs
            Forwarded to
            :meth:`~gears.output.aggregator.OutputAggregator.export`.

        Returns
        -------
        None
        """
        self.aggregator_.export(df, path, **kwargs)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """
        Save the full GEARSModel to disk using joblib.

        Parameters
        ----------
        path : str or Path
            Destination file path (e.g. ``'models/my_model.joblib'``).

        Returns
        -------
        None
        """
        import joblib
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("GEARSModel saved to %s.", path)

    @classmethod
    def load(cls, path: str | Path) -> GEARSModel:
        """
        Load a GEARSModel from disk.

        Parameters
        ----------
        path : str or Path
            Path to a joblib-serialised GEARSModel.

        Returns
        -------
        GEARSModel

        Raises
        ------
        TypeError
            If the loaded object is not a GEARSModel instance.
        """
        import joblib
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected GEARSModel, got {type(obj)}")
        return obj

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
        """
        Return a human-readable summary of the fitted model.

        Returns
        -------
        str
        """
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
            f"GEARSModel(n_components={self.n_components!r}, "
            f"stratify_by={self.stratify_by}, "
            f"forecaster={self.forecaster_method!r}, "
            f"status={status})"
        )

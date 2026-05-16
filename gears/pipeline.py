"""
GEARSModel – unified facade for the full GEARS pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from gears.data.loader import load_sessions, make_demo_data
from gears.data.insee import DepartmentForecaster
from gears.models.gmm import EVSessionGMM
from gears.models.forecaster import SessionForecaster
from gears.models.registry import ModelRegistry, NativeGMMRegistry
from gears.simulation.short_term import ShortTermSimulator
from gears.simulation.medium_term import MediumTermSimulator, CHARGER_PRESETS
from gears.smart_charging.optimizer import SmartChargingOptimizer
from gears.output.aggregator import OutputAggregator

logger = logging.getLogger(__name__)


class GEARSModel:
    """
    Unified GEARS pipeline object.

    Parameters
    ----------
    n_components : int or 'auto'
        GMM components (passed to EVSessionGMM).
    stratify_by : list[str], optional
        Context columns for GMM stratification.
        Default: ['day_of_week', 'season'].
    forecaster_method : str
        'sarima' (recommended, requires pmdarima) or 'probabilistic' (fast, no deps).
    charger_mix : dict, optional
        e.g. {7.4: 0.3, 22.0: 0.7}. Use gears.simulation.medium_term.CHARGER_PRESETS
        for predefined mixes.
    n_scenarios : int
        Default number of stochastic scenarios.
    resolution_min : int
        Time resolution for load curves (minutes).
    max_samples_per_context : int or None
        Max GMM training samples per context (for tractability).
    random_state : int

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
        n_components: Union[int, str] = "auto",
        stratify_by: Optional[list[str]] = None,
        forecaster_method: str = "sarima",   # sarima > probabilistic for real data
        charger_mix: Optional[dict[float, float]] = None,
        n_scenarios: int = 10,
        resolution_min: int = 30,
        max_samples_per_context: Optional[int] = None,
        random_state: int = 42,
    ):
        self.n_components = n_components
        self.stratify_by = stratify_by or ["day_of_week", "season"]
        self.forecaster_method = forecaster_method
        self.charger_mix = charger_mix or {7.4: 0.5, 22.0: 0.5}
        self.n_scenarios = n_scenarios
        self.resolution_min = resolution_min
        self.max_samples_per_context = max_samples_per_context
        self.random_state = random_state

        self.gmm_: Optional[EVSessionGMM] = None
        self.forecaster_: Optional[SessionForecaster] = None
        self._short_sim: Optional[ShortTermSimulator] = None
        self._medium_sim: Optional[MediumTermSimulator] = None
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
        hf_repo_id: Optional[str] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> "GEARSModel":
        """
        Load a pre-trained model bundle from the GEARS registry.

        Parameters
        ----------
        model_id : str
            Registry model identifier (e.g. 'work_fr_demo').
        hf_repo_id : str, optional
            Override the default HF Hub repository.
        cache_dir : str or Path, optional
            Local cache directory.
        **kwargs
            Forwarded to GEARSModel constructor.

        Returns
        -------
        GEARSModel (fitted)
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
        gmm_dir: Optional[Path] = None,
        **kwargs,
    ) -> "GEARSModel":
        """
        Build a GEARSModel from the pre-fitted unified French GMM.

        Parameters
        ----------
        gmm_id : str
            Registry bundle ID.  Currently only ``'french'`` is available —
            it contains all location types stratified by
            location_type × département × season × day_of_week.
        gmm_dir : Path, optional
            Override the default GMM directory.
        **kwargs
            Forwarded to GEARSModel constructor.

        Returns
        -------
        GEARSModel (fitted, using native GMM)
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
        data: Union[str, Path, pd.DataFrame],
        strict: bool = False,
        filter_failed: bool = True,
        verbose: bool = True,
        recent_months: Optional[int] = None,
        **loader_kwargs,
    ) -> "GEARSModel":
        """
        Fit the full GEARS pipeline on a sessions dataset.

        Parameters
        ----------
        data : str, Path, or DataFrame
            Raw sessions data (any supported format, incl. French pkl).
        strict : bool
            Raise on data quality issues instead of dropping rows.
        filter_failed : bool
            French data: drop failed sessions (succes_session != 't').
        verbose : bool
            Print loading and fitting progress.
        recent_months : int, optional
            Only use the most recent N months for GMM fitting.
        **loader_kwargs
            Passed to load_sessions (e.g. sep=';').

        Returns
        -------
        self
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
            random_state=self.random_state,
        ).fit(df)

        self._build_simulators()
        self.is_fitted_ = True

        if verbose:
            print("[GEARS] Fitting complete ✓")
            print(repr(self.gmm_))
        return self

    def _build_simulators(self) -> None:
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
        start_date: Union[str, pd.Timestamp],
        horizon: int = 7,
        n_scenarios: Optional[int] = None,
        n_sessions: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Simulate individual EV sessions for a short-term horizon.

        Parameters
        ----------
        start_date : str or Timestamp
            First day of the simulation.
        horizon : int
            Number of days.
        n_scenarios : int, optional
            Override default.
        n_sessions : int, optional
            Fixed session count per day (bypasses forecaster).
        seed : int, optional

        Returns
        -------
        pd.DataFrame with one row per session.
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
        start_date: Optional[Union[str, pd.Timestamp]] = None,
        output: str = "daily_energy",
        weather_factor: Optional[dict[str, float]] = None,
        growth_model: str = "linear",
        n_scenarios: Optional[int] = None,
        charger_mix: Optional[dict[float, float]] = None,
        saturation_factor: float = 3.0,
    ) -> pd.DataFrame:
        """
        Simulate aggregated EV demand over a multi-year horizon.

        Parameters
        ----------
        years : float
            Horizon in years. No upper limit.
        annual_growth_rate : float
            Annual EV penetration growth (e.g. 0.15 = +15%/yr).
        start_date : str or Timestamp, optional
            Defaults to today.
        output : str
            'daily_energy', 'hourly_energy', or 'sessions'.
        weather_factor : dict, optional
            Seasonal modifiers e.g. {'winter': 1.1, 'summer': 0.9}.
        growth_model : str
            'linear' or 's_curve'.
        n_scenarios : int, optional
            Override default.
        charger_mix : dict, optional
            Override default charger mix for this simulation.
        saturation_factor : float
            s_curve only: max sessions as multiple of base.

        Returns
        -------
        pd.DataFrame
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
        Apply smart charging optimisation to a sessions DataFrame.

        Parameters
        ----------
        sessions : pd.DataFrame
            Output of simulate_short_term().
        signal : pd.Series
            Price (€/kWh) or RES fraction signal with DatetimeIndex.
        signal_type : str
            'price' or 'res'.

        Returns
        -------
        pd.DataFrame – sessions with scheduling columns appended.
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
        """Shortcut for aggregator_.daily_energy()."""
        return self.aggregator_.daily_energy(sessions, **kwargs)

    def hourly_profile(self, sessions: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Shortcut for aggregator_.hourly_profile()."""
        return self.aggregator_.hourly_profile(sessions, **kwargs)

    def export(self, df: pd.DataFrame, path: Union[str, Path], **kwargs) -> None:
        """Shortcut for aggregator_.export()."""
        self.aggregator_.export(df, path, **kwargs)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """Save the full GEARSModel to disk (joblib)."""
        import joblib
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("GEARSModel saved to %s.", path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "GEARSModel":
        """Load a GEARSModel from disk."""
        import joblib
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected GEARSModel, got {type(obj)}")
        return obj

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
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
            lines.append(f"Status     : fitted ✓")
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

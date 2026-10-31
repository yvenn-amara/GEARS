"""
GEARSModel – thin, gear-dispatching facade over per-gear backends.

GEAR 1st (GMM/VAE session modeling + SARIMA/probabilistic forecasting +
simulation + smart charging) is the only gear implemented today; GEAR 2nd to
5th are reserved for future model families. ``GEARSModel``'s public methods are thin dispatchers to the active
gear's backend (see ``gears/pipeline_gears/``), so a future gear is free to
define its own ``fit()``/``simulate_*()`` signatures.
"""

from __future__ import annotations

import logging
from pathlib import Path

from gears._validation import check_int
from gears.pipeline_gears.gear1 import Gear1Backend

logger = logging.getLogger(__name__)

# GEAR 1st is the only backend today. GEAR 2nd to 5th are reserved: the facade accepts the numbers
# (so that a new family of models can be added without changing GEARSModel) and raises
# NotImplementedError for those that are not in this dict yet. See CONTRIBUTING.md, "Adding a gear".
_GEAR_BACKENDS: dict[int, type] = {1: Gear1Backend}
_IMPLEMENTED_GEARS = sorted(_GEAR_BACKENDS)
_MAX_GEAR = 5


class GEARSModel:
    """
    Unified, gear-dispatching GEARS pipeline object.

    Combines GMM/VAE session modeling, session-count forecasting, short- and
    medium-term simulation, V1G smart charging, and output aggregation behind
    a single facade.

    Parameters
    ----------
    gear : int
        Which GEAR backend to use, from 1 to 5. ``1`` (the default) is GEAR 1st,
        the pipeline described in the README. ``2`` to ``5`` are reserved for
        future, structurally different model families and raise
        ``NotImplementedError`` until they exist; any other value raises
        ``ValueError``.
    **kwargs
        Forwarded to the selected gear's backend constructor. For GEAR 1st,
        see :class:`gears.pipeline_gears.gear1.Gear1Backend` for the full
        parameter list (``n_components``, ``max_components``, ``stratify_by``,
        ``forecaster_method``, ``charger_mix``, ``n_scenarios``,
        ``resolution_min``, ``max_samples_per_context``,
        ``forecaster_use_holidays``, ``forecaster_country``,
        ``random_state``, ``model_type``, ``recency``, ``half_life_days``).

    Examples
    --------
    >>> model = GEARSModel()
    >>> model.fit("data/sessions_france.pkl")
    >>> sessions = model.simulate_short_term("2025-06-10", horizon=7)

    >>> model = GEARSModel(gear=1, model_type="vae", recency=True)
    >>> model.fit("data/sessions_france.pkl")

    >>> model = GEARSModel.from_pretrained("french")
    >>> sessions = model.simulate_short_term("2025-06-10")

    >>> energy = model.simulate_medium_term(years=10, annual_growth_rate=0.15)
    """

    def __init__(self, gear: int = 1, **kwargs):
        gear = check_int(gear, "gear", minimum=1, meaning=f"1 to {_MAX_GEAR}; GEAR 1st is the default")
        if gear > _MAX_GEAR:
            raise ValueError(f"gear must be between 1 and {_MAX_GEAR}, got {gear}.")
        if gear not in _GEAR_BACKENDS:
            raise NotImplementedError(
                f"GEAR {gear} is reserved for a future release and is not implemented yet. "
                f"Implemented gears: {_IMPLEMENTED_GEARS}."
            )
        self.gear = gear
        self._backend = _GEAR_BACKENDS[gear](**kwargs)

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
    ) -> GEARSModel:
        """
        Load a pre-trained session model (see :func:`gears.load_pretrained`).

        Parameters
        ----------
        model_id : str
            Pre-trained model identifier (e.g. ``'french'``).
        cache_dir : str or Path, optional
            Local cache directory for downloaded artefacts.
        path : str or Path, optional
            Location of the model file, for machines without internet access.
        **kwargs
            Forwarded to the GEAR 1st backend constructor.

        Returns
        -------
        GEARSModel
            Fitted instance built from the pre-trained model (always GEAR 1st today
            — pre-trained bundles are native GMM/VAE session models).
        """
        instance = cls.__new__(cls)
        instance.gear = 1
        instance._backend = Gear1Backend.from_pretrained(
            model_id, cache_dir=cache_dir, path=path, **kwargs
        )
        return instance

    @classmethod
    def from_session_model(
        cls,
        session_model,
        **kwargs,
    ) -> GEARSModel:
        """
        Build a GEARSModel from a fitted session model.

        Parameters
        ----------
        session_model : EVSessionModel
            Fitted session model to use as the generator.
        **kwargs
            Forwarded to the GEAR 1st backend constructor.

        Returns
        -------
        GEARSModel
            Fitted instance using the provided session model (always GEAR 1st).
        """
        instance = cls.__new__(cls)
        instance.gear = 1
        instance._backend = Gear1Backend.from_session_model(
            session_model, **kwargs
        )
        return instance

    # ------------------------------------------------------------------
    # Dispatchers — thin forwarding to the active gear's backend.
    # ------------------------------------------------------------------

    def fit(self, data, *args, **kwargs) -> GEARSModel:
        """Fit the active gear's backend on a sessions dataset. See the
        backend class (e.g. :class:`~gears.pipeline_gears.gear1.Gear1Backend`)
        for the full parameter list."""
        self._backend.fit(data, *args, **kwargs)
        return self

    def simulate_short_term(self, *args, **kwargs):
        """Forwards to the active gear's backend. See
        :meth:`gears.pipeline_gears.gear1.Gear1Backend.simulate_short_term`."""
        return self._backend.simulate_short_term(*args, **kwargs)

    def simulate_medium_term(self, *args, **kwargs):
        """Forwards to the active gear's backend. See
        :meth:`gears.pipeline_gears.gear1.Gear1Backend.simulate_medium_term`."""
        return self._backend.simulate_medium_term(*args, **kwargs)

    def smart_charge(self, *args, **kwargs):
        """Forwards to the active gear's backend. See
        :meth:`gears.pipeline_gears.gear1.Gear1Backend.smart_charge`."""
        return self._backend.smart_charge(*args, **kwargs)

    def daily_energy(self, *args, **kwargs):
        """Forwards to the active gear's backend
        (:meth:`~gears.output.aggregator.OutputAggregator.daily_energy`)."""
        return self._backend.daily_energy(*args, **kwargs)

    def hourly_profile(self, *args, **kwargs):
        """Forwards to the active gear's backend
        (:meth:`~gears.output.aggregator.OutputAggregator.hourly_profile`)."""
        return self._backend.hourly_profile(*args, **kwargs)

    def export(self, *args, **kwargs) -> None:
        """Forwards to the active gear's backend
        (:meth:`~gears.output.aggregator.OutputAggregator.export`)."""
        self._backend.export(*args, **kwargs)

    def summary(self) -> str:
        """Return a human-readable summary of the fitted model."""
        return self._backend.summary()

    # ------------------------------------------------------------------
    # Persistence — the facade itself (gear + backend) is what's pickled.
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """
        Save the full GEARSModel to disk using joblib.

        Parameters
        ----------
        path : str or Path
            Destination file path (e.g. ``'models/my_model.joblib'``).
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
    # Read-through attributes (read-only): they live on the backend.
    # ------------------------------------------------------------------

    @property
    def gmm_(self):
        return self._backend.gmm_

    @property
    def forecaster_(self):
        return self._backend.forecaster_

    @property
    def aggregator_(self):
        return self._backend.aggregator_

    @property
    def metadata_(self):
        return self._backend.metadata_

    @property
    def is_fitted_(self) -> bool:
        return self._backend.is_fitted_

    @property
    def charger_mix(self):
        return self._backend.charger_mix

    @property
    def n_components(self):
        return self._backend.n_components

    @property
    def max_components(self):
        return self._backend.max_components

    @property
    def stratify_by(self):
        return self._backend.stratify_by

    @property
    def n_scenarios(self):
        return self._backend.n_scenarios

    @property
    def resolution_min(self):
        return self._backend.resolution_min

    @property
    def max_samples_per_context(self):
        return self._backend.max_samples_per_context

    @property
    def forecaster_method(self):
        return self._backend.forecaster_method

    @property
    def forecaster_use_holidays(self):
        return self._backend.forecaster_use_holidays

    @property
    def forecaster_country(self):
        return self._backend.forecaster_country

    @property
    def random_state(self):
        return self._backend.random_state

    @property
    def model_type(self):
        return self._backend.model_type

    @property
    def recency(self):
        return self._backend.recency

    @property
    def half_life_days(self):
        return self._backend.half_life_days

    def __repr__(self) -> str:
        status = "fitted" if self._backend.is_fitted_ else "not fitted"
        n_components, stratify_by, forecaster, _ = self._backend._settings()
        return (
            f"GEARSModel(gear={self.gear}, "
            f"n_components={n_components!r}, "
            f"stratify_by={stratify_by}, "
            f"forecaster={forecaster!r}, "
            f"status={status})"
        )

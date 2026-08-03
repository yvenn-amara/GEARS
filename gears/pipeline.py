"""
GEARSModel – thin, gear-dispatching facade over per-gear backends.

GEAR 1st (today's GMM/VAE session modeling + SARIMA/probabilistic forecasting
+ simulation + smart charging pipeline) is the only fully-implemented gear.
GEAR 2nd-5th are reserved for future, structurally different model families
and raise ``NotImplementedError`` until implemented.

``GEARSModel``'s public methods are thin, signature-agnostic dispatchers to
the active gear's backend object (see ``gears/pipeline_gears/``) — this is a
deliberate design choice, not an oversight: future gears are not forced into
GEAR 1st's ``fit()``/``simulate_*()`` signatures, since GEAR 2nd's internals
are confirmed to differ structurally. See ``PROPOSAL_GEAR_ARCHITECTURE.md``
at the repo root for the full design rationale this implements.
"""

from __future__ import annotations

import logging
from pathlib import Path

from gears.pipeline_gears.gear1 import Gear1Backend

logger = logging.getLogger(__name__)

# Gears 2-5 simply aren't in this dict yet, which is what produces the
# friendly NotImplementedError below — no separate "if gear == 2: raise ..."
# branch to maintain per future gear.
_GEAR_BACKENDS: dict[int, type] = {1: Gear1Backend}
_IMPLEMENTED_GEARS = sorted(_GEAR_BACKENDS)


class GEARSModel:
    """
    Unified, gear-dispatching GEARS pipeline object.

    Combines GMM/VAE session modeling, session-count forecasting, short- and
    medium-term simulation, V1G smart charging, and output aggregation behind
    a single facade. Only GEAR 1st is implemented today; ``gear=2..5`` are
    reserved for future, structurally different model families and raise
    ``NotImplementedError`` until implemented.

    Parameters
    ----------
    gear : int
        Which GEAR backend to use. Only ``1`` (the default) is implemented
        today. ``2``-``5`` raise ``NotImplementedError``.
    **kwargs
        Forwarded to the selected gear's backend constructor. For GEAR 1st,
        see :class:`gears.pipeline_gears.gear1.Gear1Backend` for the full
        parameter list (``n_components``, ``stratify_by``,
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

    >>> model = GEARSModel.from_pretrained("french_demo")
    >>> sessions = model.simulate_short_term("2025-06-10")

    >>> energy = model.simulate_medium_term(years=10, annual_growth_rate=0.15)
    """

    def __init__(self, gear: int = 1, **kwargs):
        if gear not in _GEAR_BACKENDS:
            raise NotImplementedError(
                f"GEAR {gear} is reserved for a future release and is not "
                f"implemented yet. GEAR 1st (the current GMM/VAE + "
                f"SARIMA/probabilistic pipeline) is fully supported via "
                f"GEARSModel(gear=1, ...). Implemented gears: "
                f"{_IMPLEMENTED_GEARS}."
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
        hf_repo_id: str | None = None,
        cache_dir: str | Path | None = None,
        **kwargs,
    ) -> GEARSModel:
        """
        Load a pre-trained model bundle from the GEARS registry.

        Parameters
        ----------
        model_id : str
            Registry model identifier (e.g. ``'french_demo'``).
        hf_repo_id : str, optional
            Override the default Hugging Face Hub repository.
        cache_dir : str or Path, optional
            Local cache directory for downloaded artefacts.
        **kwargs
            Forwarded to the GEAR 1st backend constructor.

        Returns
        -------
        GEARSModel
            Fitted instance loaded from the registry (always GEAR 1st today
            — pre-trained bundles are native GMM/VAE session models).
        """
        instance = cls.__new__(cls)
        instance.gear = 1
        instance._backend = Gear1Backend.from_pretrained(
            model_id, hf_repo_id=hf_repo_id, cache_dir=cache_dir, **kwargs
        )
        return instance

    @classmethod
    def from_native_gmm(
        cls,
        session_model_id: str = "french",
        session_model_dir: Path | None = None,
        **kwargs,
    ) -> GEARSModel:
        """
        Build a GEARSModel from the pre-fitted unified French GMM.

        Parameters
        ----------
        session_model_id : str
            Registry bundle ID.  Currently only ``'french'`` is available —
            it contains all location types stratified by
            ``location_type × département × season × day_of_week``.
        session_model_dir : Path, optional
            Override the default session-model directory.
        **kwargs
            Forwarded to the GEAR 1st backend constructor.

        Returns
        -------
        GEARSModel
            Fitted instance using the native GMM bundle (always GEAR 1st).
        """
        instance = cls.__new__(cls)
        instance.gear = 1
        instance._backend = Gear1Backend.from_native_gmm(
            session_model_id, session_model_dir=session_model_dir, **kwargs
        )
        return instance

    # ------------------------------------------------------------------
    # Dispatchers — thin, signature-agnostic forwarding to the active
    # gear's backend. Deliberately no abc.ABC / typing.Protocol base class
    # fixing an abstract interface across gears: a future Gear2Backend is
    # free to define completely different signatures here, since these
    # methods only forward *args/**kwargs (fit's `data` parameter aside —
    # see PROPOSAL_GEAR_ARCHITECTURE.md for why that one is standardized).
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
    # Persistence — the facade itself (gear + backend) is what's pickled,
    # so GEARSModel.load() correctly reconstructs a GEARSModel instance
    # rather than a bare backend.
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
    # Read-through attributes — backward compatibility with the
    # pre-gear-dispatch API, where these lived directly on GEARSModel.
    # Read-only: nothing in the codebase sets these from outside the
    # backend's own fit()/from_pretrained()/from_native_gmm().
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
        return (
            f"GEARSModel(gear={self.gear}, "
            f"n_components={self._backend.n_components!r}, "
            f"stratify_by={self._backend.stratify_by}, "
            f"forecaster={self._backend.forecaster_method!r}, "
            f"status={status})"
        )

"""
GEARS – Generating Electric vehicle recharging Sessions
=======================================================
A Python package for simulation, forecasting and smart charging of EV sessions.

Works with any EV dataset — the French national dataset is auto-detected,
but any dataset with arrival_time / duration / energy columns is supported.

Quick start
-----------
>>> from gears import GEARSModel, load_sessions
>>> model = GEARSModel()
>>> model.fit("data/sessions.pkl")
>>> sessions = model.simulate_short_term("2025-06-10", horizon=7)

Optional extras
---------------
- Deep-learning forecasters (PatchTST, NHiTS):  uv pip install "gears-ev[dl]"
"""

from ._fetch_models import ensure_models
ensure_models()
from gears.data.insee import DepartmentForecaster, aggregate_by_department, build_panel
from gears.data.loader import load_sessions, make_demo_data
from gears.evaluation.benchmark import (
    crps_ensemble,
    run_benchmark_for_datasets,
    run_rolling_origin_benchmark,
    run_sarima_sanity_check,
    summarize_sarima_sanity_check,
)
from gears.evaluation.windowing import sessions_in_last_n_occurrences
from gears.models.forecaster import (
    NHiTSForecaster,
    PersistenceForecaster,
    SessionForecaster,
    TransformerForecaster,
    sessions_to_daily_counts,
)
from gears.models.persistence_sampler import PersistenceSessionSampler
from gears.models.registry import (
    ModelRegistry,
    NativeSessionModelRegistry,
    get_session_model,
)
from gears.models.session_model import EVSessionModel
from gears.output.aggregator import LOCATION_POWER_PRESETS, OutputAggregator
from gears.pipeline import GEARSModel
from gears.simulation.medium_term import (
    CHARGER_PRESETS,
    GROWTH_PROFILES,
    MediumTermSimulator,
    bass_diffusion_profile,
    linear_growth_profile,
    s_curve_growth_profile,
)
from gears.simulation.short_term import ShortTermSimulator
from gears.smart_charging.optimizer import SmartChargingOptimizer

__version__ = "2.0.0"
__author__ = "Yvenn Amara-Ouali"

__all__ = [
    "CHARGER_PRESETS",
    "GROWTH_PROFILES",
    "LOCATION_POWER_PRESETS",
    "DepartmentForecaster",
    # Models
    "EVSessionModel",
    # Pipeline
    "GEARSModel",
    "MediumTermSimulator",
    "ModelRegistry",
    "NHiTSForecaster",
    "NativeSessionModelRegistry",
    # Output
    "OutputAggregator",
    "PersistenceForecaster",
    "PersistenceSessionSampler",
    "SessionForecaster",
    # Simulation
    "ShortTermSimulator",
    # Smart charging
    "SmartChargingOptimizer",
    "TransformerForecaster",
    "aggregate_by_department",
    "bass_diffusion_profile",
    "build_panel",
    "crps_ensemble",
    "get_session_model",
    "linear_growth_profile",
    # Data
    "load_sessions",
    "make_demo_data",
    "run_benchmark_for_datasets",
    "run_rolling_origin_benchmark",
    "run_sarima_sanity_check",
    "s_curve_growth_profile",
    "sessions_in_last_n_occurrences",
    "sessions_to_daily_counts",
    "summarize_sarima_sanity_check",
]

from gears import plotting, utils

__all__ += ["plotting", "utils"]

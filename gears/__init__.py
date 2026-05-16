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

from gears.models.gmm import EVSessionGMM
from gears.models.forecaster import (
    SessionForecaster,
    TransformerForecaster,
    NHiTSForecaster,
    PersistenceForecaster,
    sessions_to_daily_counts,
)
from gears.models.registry import ModelRegistry, NativeGMMRegistry, get_gmm
from gears.simulation.short_term import ShortTermSimulator
from gears.simulation.medium_term import (
    MediumTermSimulator,
    CHARGER_PRESETS,
    GROWTH_PROFILES,
    linear_growth_profile,
    s_curve_growth_profile,
    s_curve_linear_tail_profile,
    bass_diffusion_profile,
    double_s_curve_profile,
)
from gears.smart_charging.optimizer import SmartChargingOptimizer
from gears.output.aggregator import OutputAggregator, LOCATION_POWER_PRESETS
from gears.pipeline import GEARSModel
from gears.data.loader import load_sessions, make_demo_data
from gears.data.insee import DepartmentForecaster, aggregate_by_department, build_panel

__version__ = "1.0.0"
__author__ = "Yvenn Amara-Ouali"

__all__ = [
    # Pipeline
    "GEARSModel",
    # Models
    "EVSessionGMM",
    "SessionForecaster",
    "TransformerForecaster",
    "NHiTSForecaster",
    "PersistenceForecaster",
    "sessions_to_daily_counts",
    "ModelRegistry",
    "NativeGMMRegistry",
    "get_gmm",
    # Simulation
    "ShortTermSimulator",
    "MediumTermSimulator",
    "CHARGER_PRESETS",
    "GROWTH_PROFILES",
    "linear_growth_profile",
    "s_curve_growth_profile",
    "s_curve_linear_tail_profile",
    "bass_diffusion_profile",
    "double_s_curve_profile",
    # Smart charging
    "SmartChargingOptimizer",
    # Output
    "OutputAggregator",
    "LOCATION_POWER_PRESETS",
    # Data
    "load_sessions",
    "make_demo_data",
    "DepartmentForecaster",
    "aggregate_by_department",
    "build_panel",
]

from gears import utils, plotting
__all__ += ["utils", "plotting"]

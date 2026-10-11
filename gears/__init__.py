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

from gears.data.loader import load_sessions, make_demo_data
from gears.models import (
    PersistenceForecaster,
    SessionForecaster,
    sessions_to_daily_counts,
)
from gears.models.session_model import EVSessionModel
from gears.output.aggregator import LOCATION_POWER_PRESETS, OutputAggregator
from gears.pipeline import GEARSModel
from gears.pretrained import load_pretrained
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
    "EVSessionModel",
    "GEARSModel",
    "GROWTH_PROFILES",
    "LOCATION_POWER_PRESETS",
    "MediumTermSimulator",
    "OutputAggregator",
    "PersistenceForecaster",
    "SessionForecaster",
    "ShortTermSimulator",
    "SmartChargingOptimizer",
    "bass_diffusion_profile",
    "linear_growth_profile",
    "load_sessions",
    "make_demo_data",
    "s_curve_growth_profile",
    "sessions_to_daily_counts",
    "load_pretrained",
]

from gears import plotting, utils
__all__ += ["plotting", "utils"]

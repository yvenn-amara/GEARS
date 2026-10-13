"""
GEARS – Generating Electric vehicle recharging Sessions
=======================================================
Simulation, forecasting and smart charging of EV charging sessions.

``import gears`` is cheap: names are loaded on first use (PEP 562), so neither
``torch``, ``neuralforecast``, ``pmdarima``, ``statsmodels`` nor ``matplotlib``
is imported until a feature needs it, and nothing touches the network or the disk.

Three levels of use
-------------------
**Layer 1 – the session generator only** (a Gaussian mixture per context, or a VAE)::

    from gears.models import EVSessionModel
    gmm = EVSessionModel().fit(sessions)
    new_sessions = gmm.sample(1000)

**Layer 2 – independent building blocks**::

    from gears.models import SessionForecaster
    from gears.simulation import ShortTermSimulator, MediumTermSimulator
    from gears.smart_charging import SmartChargingOptimizer

**Layer 3 – the full GEAR 1st pipeline** (fit, forecast, simulate, smart-charge)::

    from gears import GEARSModel
    model = GEARSModel().fit(sessions)
    sessions = model.simulate_short_term("2025-06-10", horizon=7)

Without your own data, start from the pre-trained French model::

    from gears import GEARSModel
    model = GEARSModel.from_pretrained("french")

Optional extras
---------------
- ``pip install "gears-ev[vae]"``  – VAE session model (needs ``torch``)
- ``pip install "gears-ev[dl]"``   – deep-learning forecasters, NHiTS and PatchTST
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

__version__ = "2.0.0"
__author__ = "Yvenn Amara-Ouali"

# Public name -> module that defines it. Resolved on first access by __getattr__.
_EXPORTS: dict[str, str] = {
    "CHARGER_PRESETS": "gears.simulation.medium_term",
    "GROWTH_PROFILES": "gears.simulation.medium_term",
    "LOCATION_POWER_PRESETS": "gears.output.aggregator",
    "EVSessionModel": "gears.models.session_model",
    "GEARSModel": "gears.pipeline",
    "MediumTermSimulator": "gears.simulation.medium_term",
    "OutputAggregator": "gears.output.aggregator",
    "PersistenceForecaster": "gears.models.forecaster",
    "SessionForecaster": "gears.models.forecaster",
    "ShortTermSimulator": "gears.simulation.short_term",
    "SmartChargingOptimizer": "gears.smart_charging.optimizer",
    "bass_diffusion_profile": "gears.simulation.medium_term",
    "linear_growth_profile": "gears.simulation.medium_term",
    "load_pretrained": "gears.pretrained",
    "load_sessions": "gears.data.loader",
    "make_demo_data": "gears.data.loader",
    "s_curve_growth_profile": "gears.simulation.medium_term",
    "sessions_to_daily_counts": "gears.models.forecaster",
}
# Submodules that ``gears.<name>`` and ``from gears import <name>`` import on demand.
_SUBMODULES: tuple[str, ...] = ("plotting", "utils")

__all__ = [  # kept literal so linters and IDEs can read it; tests check it matches _EXPORTS
    "CHARGER_PRESETS",
    "GROWTH_PROFILES",
    "LOCATION_POWER_PRESETS",
    "EVSessionModel",
    "GEARSModel",
    "MediumTermSimulator",
    "OutputAggregator",
    "PersistenceForecaster",
    "SessionForecaster",
    "ShortTermSimulator",
    "SmartChargingOptimizer",
    "bass_diffusion_profile",
    "linear_growth_profile",
    "load_pretrained",
    "load_sessions",
    "make_demo_data",
    "plotting",
    "s_curve_growth_profile",
    "sessions_to_daily_counts",
    "utils",
]


def __getattr__(name: str):
    if name in _EXPORTS:
        value = getattr(importlib.import_module(_EXPORTS[name]), name)
    elif name in _SUBMODULES:
        value = importlib.import_module(f"gears.{name}")
    else:
        raise AttributeError(f"module 'gears' has no attribute {name!r}")
    globals()[name] = value  # cache: __getattr__ runs once per name
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})


if TYPE_CHECKING:  # what IDEs and type checkers see; never executed at runtime
    from gears import plotting, utils
    from gears.data.loader import load_sessions, make_demo_data
    from gears.models.forecaster import (
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

"""
GEARS model components.

Names are loaded on first use (PEP 562): ``from gears.models import EVSessionModel``
does not import the forecasters, and the deep-learning forecasters
(``NHiTSForecaster``, ``TransformerForecaster``, extra ``[dl]``) are only imported
when asked for. Importing them still does not load ``torch``; that happens when
you call ``fit()``.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

# Public name -> module that defines it. Resolved on first access by __getattr__.
_EXPORTS: dict[str, str] = {
    "EVSessionModel": "gears.models.session_model",
    "NHiTSForecaster": "gears.models.forecaster_dl",
    "PersistenceForecaster": "gears.models.forecaster",
    "SessionForecaster": "gears.models.forecaster",
    "TransformerForecaster": "gears.models.forecaster_dl",
    "sessions_to_daily_counts": "gears.models.forecaster",
}

__all__ = [  # kept literal so linters and IDEs can read it; tests check it matches _EXPORTS
    "EVSessionModel",
    "NHiTSForecaster",
    "PersistenceForecaster",
    "SessionForecaster",
    "TransformerForecaster",
    "sessions_to_daily_counts",
]


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module 'gears.models' has no attribute {name!r}")
    value = getattr(importlib.import_module(_EXPORTS[name]), name)
    globals()[name] = value  # cache: __getattr__ runs once per name
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})


if TYPE_CHECKING:  # what IDEs and type checkers see; never executed at runtime
    from gears.models.forecaster import (
        PersistenceForecaster,
        SessionForecaster,
        sessions_to_daily_counts,
    )
    from gears.models.forecaster_dl import NHiTSForecaster, TransformerForecaster
    from gears.models.session_model import EVSessionModel

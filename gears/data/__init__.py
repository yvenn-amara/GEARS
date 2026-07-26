"""GEARS data loading, validation and aggregation utilities."""

from gears.data.insee import (
    DepartmentForecaster,
    aggregate_by_department,
    build_panel,
    department_daily_energy,
)
from gears.data.loader import load_sessions, make_demo_data
from gears.data.schemas import _season, summary_stats, validate_dataframe

__all__ = [
    "DepartmentForecaster",
    "_season",
    "aggregate_by_department",
    "build_panel",
    "department_daily_energy",
    "load_sessions",
    "make_demo_data",
    "summary_stats",
    "validate_dataframe",
]

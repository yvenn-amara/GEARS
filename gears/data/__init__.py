"""GEARS data loading, validation and aggregation utilities."""

from gears.data.loader import load_sessions, make_demo_data
from gears.data.schemas import validate_dataframe, summary_stats, _season
from gears.data.insee import (
    aggregate_by_department,
    build_panel,
    department_daily_energy,
    DepartmentForecaster,
)

__all__ = [
    "load_sessions",
    "make_demo_data",
    "validate_dataframe",
    "summary_stats",
    "_season",
    "aggregate_by_department",
    "build_panel",
    "department_daily_energy",
    "DepartmentForecaster",
]

"""Tests for gears.data.schemas: column aliases, unit conversion, enrichment, filtering.

The error and warning *messages* (missing columns, units, time zones, dropped rows...)
are checked in ``test_sanity.py``.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gears.data.loader import load_sessions
from gears.data.schemas import summary_stats, validate_dataframe

REQUIRED = {"arrival_time", "duration", "energy"}


def make_generic(n=100):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "arrival_time": pd.date_range("2025-01-01", periods=n, freq="6h"),
        "duration": rng.uniform(0.5, 10, n),
        "energy": rng.uniform(1, 40, n),
    })


def make_french(n=50):
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "debut_session_timestamp": pd.date_range("2025-03-01", periods=n, freq="3h"),
        "duree_min": rng.uniform(30, 300, n),
        "energie_delivree_wh": rng.uniform(1000, 30000, n),
        "succes_session": ["t"] * n,
        "id_domaine_subvention": ["en_pri"] * (n // 2) + ["en_opu"] * (n // 2),
        "insee_code_departement": ["75"] * n,
    })


def make_evse(n=20, seed=0):
    """The raw schema of the EVSE datasets: Start/End/Arrival/Park.Duration/Energy."""
    rng = np.random.default_rng(seed)
    starts = pd.Timestamp("2023-01-02") + pd.to_timedelta(rng.uniform(0, 200, n), unit="D")
    park_duration_min = rng.uniform(10, 600, n)
    return pd.DataFrame({
        "Start": starts,
        "End": starts + pd.to_timedelta(park_duration_min, unit="m"),
        "Day": starts.day_name(),
        "Weekend": (starts.dayofweek >= 5).astype(int),
        "Arrival": starts.hour + starts.minute / 60.0,
        "Charge.Duration": park_duration_min * 0.9,
        "Idle.Duration": park_duration_min * 0.1,
        "Park.Duration": park_duration_min,
        "Energy": rng.uniform(1, 50, n),
    })


def test_a_generic_export_gets_its_calendar_features():
    # the 15th of each month of 2025 (a Wednesday for January), at 09:30 ... 12:30 or 08:30
    df = pd.DataFrame({
        "arrival_time": pd.to_datetime(
            [f"2025-{month:02d}-15 {8 + month % 5}:30" for month in range(1, 13)]
        ),
        "duration": 2.0,
        "energy": 10.0,
    })
    out = validate_dataframe(df)

    assert out["hour"].tolist() == [9.5, 10.5, 11.5, 12.5, 8.5, 9.5, 10.5, 11.5, 12.5, 8.5, 9.5, 10.5]
    assert out["day_of_week"].tolist() == [2, 5, 5, 1, 3, 6, 1, 4, 0, 2, 5, 0]       # 0 = Monday
    assert out["is_weekend"].tolist() == [0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0]
    assert out["season"].tolist() == (
        ["winter"] * 2 + ["spring"] * 3 + ["summer"] * 3 + ["autumn"] * 3 + ["winter"]
    )                                                              # meteorological seasons
    assert len(validate_dataframe(make_generic())) == 100       # a clean file loses nothing

    stats = summary_stats(validate_dataframe(make_generic()))
    assert {"mean", "missing_%"} <= set(stats.columns)


def test_a_french_export_is_detected_and_converted_to_kwh_and_hours():
    raw = make_french()
    raw.loc[0:4, "succes_session"] = "f"                        # 5 failed sessions
    out = validate_dataframe(raw)
    assert len(out) == 45                                        # failed ones are filtered

    assert out["energy"].iloc[0] == pytest.approx(raw["energie_delivree_wh"].iloc[5] / 1000)
    assert out["duration"].iloc[0] == pytest.approx(raw["duree_min"].iloc[5] / 60)
    # id_domaine_subvention: "en_pri" (private company) -> work, "en_opu" (open to the public)
    assert out["location_type"].value_counts().to_dict() == {"public": 25, "work": 20}
    assert len(validate_dataframe(raw, filter_failed=False)) == 50


def test_sessions_that_cannot_be_physical_are_filtered():
    df = make_generic(100)
    df.loc[0, "duration"] = -1       # not positive
    df.loc[1, "energy"] = -5         # negative
    df.loc[2, "duration"] = 200      # longer than a week
    df.loc[3, "energy"] = 1000       # more than any vehicle can take
    df.loc[4, "energy"] = 0          # a session that delivered nothing is a valid session
    assert len(validate_dataframe(df)) == 96


@pytest.mark.parametrize("names", [
    {"arrival_time": "start_time", "duration": "duration_hours", "energy": "energy_kwh"},
    {"arrival_time": "START", "duration": "Duration_H", "energy": "ENERGY_KWH"},
    {"arrival_time": "Start", "duration": "duration", "energy": "Energy"},
])
def test_column_aliases_are_matched_whatever_the_case(names):
    raw = make_generic(5).rename(columns=names)
    out = validate_dataframe(raw)
    assert REQUIRED <= set(out.columns)
    assert out["duration"].tolist() == make_generic(5)["duration"].tolist()


def test_duration_is_derived_from_the_end_time():
    df = pd.DataFrame({
        "arrival_time": pd.date_range("2025-01-01", periods=3, freq="8h"),
        "end_time": pd.date_range("2025-01-01 02:00", periods=3, freq="8h"),
        "energy": [10.0, 12.0, 8.0],
    })
    assert validate_dataframe(df)["duration"].tolist() == [2.0, 2.0, 2.0]


def test_the_evse_raw_schema_is_understood():
    """``Arrival`` is an hour of the day (4.13), never a timestamp; ``Park.Duration`` is
    in minutes; ``Day``, ``Weekend`` and index leftovers are noise that is dropped."""
    raw = make_evse()
    raw["Unnamed: 0"] = range(len(raw))
    raw["Start_Date"] = raw["Start"].dt.date.astype(str)

    out = validate_dataframe(raw)

    assert REQUIRED <= set(out.columns) and len(out) == len(raw)
    assert np.allclose(out["duration"], raw["Park.Duration"] / 60.0)         # minutes -> hours
    assert (out["arrival_time"].dt.date == raw["Start"].dt.date).all()       # from Start
    assert np.allclose(out["Arrival"], raw["Arrival"])                       # kept as it is
    for noisy in ("Day", "Weekend", "Unnamed: 0", "Start_Date"):
        assert noisy not in out.columns
    assert REQUIRED <= set(load_sessions(raw, verbose=False).columns)        # and via the loader

    # with only the hour of the day there is no arrival time: it is not guessed from `Arrival`
    hour_only = pd.DataFrame({"Arrival": [4.13, 6.75], "duration": [1.0, 2.0], "energy": [1.0, 2.0]})
    with pytest.raises(ValueError, match="arrival_time"):
        validate_dataframe(hour_only)


def test_a_combined_multi_site_file_warns_but_a_single_site_does_not():
    """acn.csv-style data (several sites in one file, flagged by a `data` column) would be
    counted several times next to the individual site files."""
    combined = make_evse(n=6)
    combined["data"] = ["caltech", "caltech", "jpl", "jpl", "office", "office"]
    with pytest.warns(UserWarning, match="combined multi-site"):
        validate_dataframe(combined)

    single = make_evse(n=6)
    single["data"] = "caltech"
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        validate_dataframe(single)       # must not warn


@pytest.mark.slow
def test_the_real_datasets_load_when_they_are_present():
    """The CSVs are not versioned (data/ only holds a README); run this on a checkout
    that has data/preprocessed_data/ with ``pytest -m slow``."""
    data_dir = Path(__file__).resolve().parent.parent / "data" / "preprocessed_data"
    files = sorted(data_dir.glob("*.csv")) if data_dir.exists() else []
    if not files:
        pytest.skip("data/preprocessed_data/ is not present in this checkout")
    for path in files:
        out = load_sessions(str(path), verbose=False)
        assert REQUIRED <= set(out.columns) and len(out) > 0, path.name

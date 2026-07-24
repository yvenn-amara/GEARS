"""Tests for gears.data.schemas."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gears.data.loader import load_sessions, make_demo_data
from gears.data.schemas import (
    _find_column,
    _normalize_key,
    _season,
    summary_stats,
    validate_dataframe,
)


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


def test_season():
    assert _season(1) == "winter"
    assert _season(4) == "spring"
    assert _season(7) == "summer"
    assert _season(10) == "autumn"
    assert _season(12) == "winter"


def test_generic_validate():
    df = make_generic()
    out = validate_dataframe(df)
    assert "hour" in out.columns
    assert "day_of_week" in out.columns
    assert "season" in out.columns
    assert "is_weekend" in out.columns
    assert out["duration"].min() > 0
    assert out["energy"].min() >= 0
    assert len(out) == len(df)


def test_french_autodetect():
    raw = make_french()
    out = validate_dataframe(raw)
    assert "energy" in out.columns
    assert "duration" in out.columns
    assert "location_type" in out.columns
    assert out["energy"].iloc[0] == pytest.approx(raw["energie_delivree_wh"].iloc[0] / 1000)
    assert out["duration"].iloc[0] == pytest.approx(raw["duree_min"].iloc[0] / 60)


def test_french_location_mapping():
    raw = make_french()
    out = validate_dataframe(raw)
    assert set(out["location_type"].unique()).issubset({"work", "home", "public", "heavy", "unknown"})
    assert out[out["location_type"] == "work"].shape[0] > 0


def test_failed_sessions_filtered():
    raw = make_french(n=10)
    raw.loc[0:4, "succes_session"] = "f"  # 5 failed
    out = validate_dataframe(raw, filter_failed=True)
    assert len(out) == 5


def test_quality_filters():
    df = make_generic(100)
    df.loc[0, "duration"] = -1    # invalid
    df.loc[1, "energy"] = -5      # invalid
    df.loc[2, "duration"] = 200   # too long
    out = validate_dataframe(df)
    assert len(out) == 97


def test_alias_resolution():
    df = pd.DataFrame({
        "start_time": pd.date_range("2025-01-01", periods=5, freq="8h"),
        "duration_hours": [2.0, 3.0, 1.5, 4.0, 2.5],
        "energy_kwh": [10.0, 15.0, 8.0, 20.0, 12.0],
    })
    out = validate_dataframe(df)
    assert "arrival_time" in out.columns
    assert "duration" in out.columns
    assert "energy" in out.columns


def test_derive_duration_from_end_time():
    df = pd.DataFrame({
        "arrival_time": pd.date_range("2025-01-01", periods=3, freq="8h"),
        "end_time": pd.date_range("2025-01-01 02:00", periods=3, freq="8h"),
        "energy": [10.0, 12.0, 8.0],
    })
    out = validate_dataframe(df)
    assert "duration" in out.columns
    assert all(out["duration"] > 0)


def test_summary_stats():
    df = make_generic()
    out = validate_dataframe(df)
    stats = summary_stats(out)
    assert "mean" in stats.columns
    assert "missing_%" in stats.columns


# --------------------------------------------------------------------------
# Session 1 — EVSE-style raw schema (Start/End/Arrival/Park.Duration/Energy),
# the case/punctuation-insensitive alias matching that supports it, and the
# guards it depends on (Arrival never -> arrival_time, noise columns dropped,
# combined-site acn.csv warning). See gears_persistence_vs_gmm_benchmark_prompt.md §1.7/§2.
# --------------------------------------------------------------------------

def make_evse(n=20, seed=0):
    """Mirrors the 11 preprocessed_data/*.csv datasets' raw schema."""
    rng = np.random.default_rng(seed)
    starts = pd.Timestamp("2023-01-02") + pd.to_timedelta(rng.uniform(0, 200, n), unit="D")
    park_duration_min = rng.uniform(10, 600, n)
    ends = starts + pd.to_timedelta(park_duration_min, unit="m")
    return pd.DataFrame({
        "Start": starts,
        "End": ends,
        "Day": starts.day_name(),
        "Weekend": (starts.dayofweek >= 5).astype(int),
        "Arrival": starts.hour + starts.minute / 60.0,
        "Charge.Duration": park_duration_min * 0.9,
        "Idle.Duration": park_duration_min * 0.1,
        "Park.Duration": park_duration_min,
        "Energy": rng.uniform(1, 50, n),
    })


def test_evse_style_schema_end_to_end():
    raw = make_evse()
    out = validate_dataframe(raw)
    assert {"arrival_time", "duration", "energy"}.issubset(out.columns)
    assert len(out) == len(raw)
    assert out["duration"].max() < 24  # hours, not raw minutes


def test_evse_style_via_load_sessions():
    raw = make_evse()
    out = load_sessions(raw, verbose=False)
    assert {"arrival_time", "duration", "energy"}.issubset(out.columns)


def test_park_duration_minutes_converted_to_hours():
    raw = make_evse()
    out = validate_dataframe(raw)
    assert np.allclose(out["duration"].to_numpy(), (raw["Park.Duration"] / 60.0).to_numpy())


def test_arrival_column_never_resolved_as_arrival_time():
    """`Arrival` is a float hour-of-day (e.g. 4.13), not a timestamp — must
    never be aliased to arrival_time, which must come from `Start` instead."""
    raw = make_evse()
    out = validate_dataframe(raw)
    assert (out["arrival_time"].dt.date == raw["Start"].dt.date).all()
    assert "Arrival" in out.columns
    assert np.allclose(out["Arrival"], raw["Arrival"])


def test_find_column_excludes_arrival_alias_explicitly():
    df = pd.DataFrame({"Arrival": [4.13, 6.75], "duration": [1.0, 2.0], "energy": [1.0, 2.0]})
    assert _find_column(df, "arrival_time") is None


def test_noise_columns_dropped():
    raw = make_evse()
    raw["Unnamed: 0"] = range(len(raw))
    raw["Start_Date"] = raw["Start"].dt.date.astype(str)
    out = validate_dataframe(raw)
    for noisy in ("Day", "Weekend", "Unnamed: 0", "Start_Date"):
        assert noisy not in out.columns


def test_normalize_key():
    assert _normalize_key("Park.Duration") == "park_duration"
    assert _normalize_key("Start Time") == "start_time"
    assert _normalize_key("Unnamed: 0") == "unnamed_0"


@pytest.mark.parametrize("cols", [
    {"START": "arrival_time", "Duration_H": "duration", "ENERGY_KWH": "energy"},
    {"Start": "arrival_time", "duration": "duration", "Energy": "energy"},
])
def test_case_insensitive_generic_aliasing(cols):
    keys = list(cols)
    raw = pd.DataFrame({
        keys[0]: ["2023-01-01 08:00:00", "2023-01-02 09:00:00"],
        keys[1]: [2.0, 3.0],
        keys[2]: [10.0, 12.0],
    })
    out = validate_dataframe(raw)
    assert {"arrival_time", "duration", "energy"}.issubset(out.columns)


def test_combined_site_dataset_warns():
    """acn.csv-style data (union of caltech+jpl+office, flagged by a `data`
    site column) must warn so it isn't silently double/triple-counted
    alongside the individual site files."""
    raw = make_evse(n=6)
    raw["data"] = ["caltech", "caltech", "jpl", "jpl", "office", "office"]
    with pytest.warns(UserWarning, match="combined multi-site"):
        validate_dataframe(raw)


def test_single_site_dataset_does_not_warn():
    raw = make_evse(n=6)
    raw["data"] = "caltech"
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        validate_dataframe(raw)  # must not raise


def test_make_demo_data_unaffected():
    df = make_demo_data(n=50, location_type="work", seed=0)
    assert len(df) == 50
    assert {"arrival_time", "duration", "energy", "location_type"}.issubset(df.columns)


ALL_11_DATASETS = [
    "acn", "boulder", "caltech", "domestics", "dundee", "jpl",
    "office", "palo_alto", "paris", "perth", "sap",
]
_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "preprocessed_data"


@pytest.mark.skipif(not _DATA_DIR.exists(), reason="data/preprocessed_data/ not present in this checkout")
@pytest.mark.parametrize("name", ALL_11_DATASETS)
def test_all_11_real_csvs_load_without_error(name):
    path = _DATA_DIR / f"{name}.csv"
    if not path.exists():
        pytest.skip(f"{path} not present in this checkout")
    out = load_sessions(str(path), verbose=False)
    assert {"arrival_time", "duration", "energy"}.issubset(out.columns)
    assert len(out) > 0

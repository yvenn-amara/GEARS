"""Tests for gears.data.schemas."""
import pytest
import numpy as np
import pandas as pd
from gears.data.schemas import validate_dataframe, _season, summary_stats


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

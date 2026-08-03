"""Tests for scripts/fit_session_model.py's CLI surface — specifically the
Phase 2 Session 4 additions (--departments, --exclude-last-n-days,
--output-name) and the --help contract. Runs the script as a real
subprocess against small synthetic CSVs (not real data), matching the
session's own ground rule to keep this test run fast.

The underlying stratify_by fallback fix itself (the actual bug) is covered
directly against EVSessionModel in test_session_model.py — this file only
covers the script's own argument wiring on top of it.
"""
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "fit_session_model.py"


def _write_synthetic_csv(path, n=300, seed=0, departments=("92", "69"), with_location_type=True):
    rng = np.random.default_rng(seed)
    data = {
        "arrival_time": pd.date_range("2024-01-01", periods=n, freq="3h"),
        "duration": rng.uniform(0.5, 8, n),
        "energy": rng.uniform(2, 40, n),
        "department": rng.choice(departments, n),
    }
    if with_location_type:
        data["location_type"] = rng.choice(["work", "home", "public"], n)
    pd.DataFrame(data).to_csv(path, index=False)
    return path


def _run(*args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True, text=True, cwd=REPO_ROOT, check=False,
    )


def test_help_parses_without_error():
    result = _run("--help")
    assert result.returncode == 0, result.stderr
    assert "--departments" in result.stdout
    assert "--exclude-last-n-days" in result.stdout
    assert "--output-name" in result.stdout


def test_departments_flag_filters_before_fitting(tmp_path):
    data_path = _write_synthetic_csv(
        tmp_path / "sessions.csv", n=6000, departments=("92", "69", "59"),
    )
    out_dir = tmp_path / "models"
    result = _run(
        "--input", str(data_path), "--output-dir", str(out_dir),
        "--departments", "92,69", "--n-components", "1", "--overwrite",
    )
    assert result.returncode == 0, result.stderr
    assert "Filtered to departments ['92', '69']" in result.stderr

    model = joblib.load(out_dir / "gmm_french.joblib")
    depts_used = {ctx[model.stratify_by.index("department")] for ctx in model.models_}
    assert depts_used <= {"92", "69"}


def test_departments_flag_errors_on_missing_column(tmp_path):
    data_path = _write_synthetic_csv(tmp_path / "sessions.csv")
    # Drop the department column entirely from the CSV.
    df = pd.read_csv(data_path).drop(columns=["department"])
    df.to_csv(data_path, index=False)

    result = _run(
        "--input", str(data_path), "--output-dir", str(tmp_path / "models"),
        "--departments", "92", "--n-components", "1",
    )
    assert result.returncode != 0
    assert "no 'department' column" in result.stderr


def test_exclude_last_n_days_filters_recent_sessions(tmp_path):
    data_path = _write_synthetic_csv(tmp_path / "sessions.csv", n=400)
    out_dir = tmp_path / "models"
    result = _run(
        "--input", str(data_path), "--output-dir", str(out_dir),
        "--exclude-last-n-days", "10", "--n-components", "1", "--overwrite",
    )
    assert result.returncode == 0, result.stderr
    assert "Excluded last 10 days" in result.stderr

    full = pd.read_csv(data_path, parse_dates=["arrival_time"])
    cutoff = full["arrival_time"].max() - pd.Timedelta(days=10)
    expected_kept = int((full["arrival_time"] < cutoff).sum())
    assert f"{expected_kept} / {len(full)} sessions retained" in result.stderr


def test_output_name_bypasses_registry_and_does_not_touch_default_bundle(tmp_path):
    data_path = _write_synthetic_csv(tmp_path / "sessions.csv")
    out_dir = tmp_path / "models"
    result = _run(
        "--input", str(data_path), "--output-dir", str(out_dir),
        "--n-components", "1", "--output-name", "gmm_french_holdout",
    )
    assert result.returncode == 0, result.stderr

    assert (out_dir / "gmm_french_holdout.joblib").exists()
    # The default registry-managed bundle must NOT have been created/touched.
    assert not (out_dir / "gmm_french.joblib").exists()


def test_output_name_overwrite_guard(tmp_path):
    data_path = _write_synthetic_csv(tmp_path / "sessions.csv")
    out_dir = tmp_path / "models"
    first = _run(
        "--input", str(data_path), "--output-dir", str(out_dir),
        "--n-components", "1", "--output-name", "gmm_custom",
    )
    assert first.returncode == 0, first.stderr

    second = _run(
        "--input", str(data_path), "--output-dir", str(out_dir),
        "--n-components", "1", "--output-name", "gmm_custom",
    )
    assert second.returncode != 0
    assert "already exists" in second.stderr

    third = _run(
        "--input", str(data_path), "--output-dir", str(out_dir),
        "--n-components", "1", "--output-name", "gmm_custom", "--overwrite",
    )
    assert third.returncode == 0, third.stderr

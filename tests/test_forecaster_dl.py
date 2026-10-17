"""Tests for the deep-learning forecasters (N-HiTS, PatchTST), which need neuralforecast.

Heavy: skipped when neuralforecast is not installed and excluded from the default run.
Run them with ``pip install "gears-ev[dl]"`` and ``pytest -m dl``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("neuralforecast")

from gears.data.schemas import validate_dataframe
from gears.models.forecaster_dl import NHiTSForecaster, TransformerForecaster

pytestmark = pytest.mark.dl


@pytest.fixture(autouse=True)
def _work_in_a_scratch_folder(tmp_path, monkeypatch):
    """PyTorch Lightning writes ``lightning_logs/`` in the working directory."""
    monkeypatch.chdir(tmp_path)


def make_sessions(n=300, seed=0):
    rng = np.random.default_rng(seed)
    return validate_dataframe(pd.DataFrame({
        "arrival_time": pd.date_range("2024-01-01", periods=n, freq="6h"),
        "duration": rng.uniform(0.5, 8, n),
        "energy": rng.uniform(2, 40, n),
    }))


def test_nhits_fit_and_predict():
    assert NHiTSForecaster.is_available() and TransformerForecaster.is_available()
    fc = NHiTSForecaster(horizon=7, max_steps=5).fit(make_sessions())
    assert fc.is_fitted_ and "fitted=True" in repr(fc)

    pred = fc.predict(horizon=7, n_scenarios=3, start_date="2025-01-01", seed=0)
    assert {"date", "scenario", "n_sessions"} <= set(pred.columns)
    assert len(pred) == 21 and (pred["n_sessions"] >= 0).all()
    assert pd.Timestamp(pred["date"].min()) == pd.Timestamp("2025-01-01")

    # the context window never exceeds half of a short training set
    short = NHiTSForecaster(horizon=7, input_size=100, max_steps=5).fit(make_sessions(n=60))
    assert short._effective_input_size <= 30

    # no temporal leakage: a forecast without a start date begins after the training data
    train = make_sessions(n=200, seed=99)
    train = train[train["arrival_time"] <= train["arrival_time"].quantile(0.75)]
    after = NHiTSForecaster(horizon=7, max_steps=5).fit(train).predict(horizon=7, n_scenarios=1)
    assert all(pd.Timestamp(d) > pd.Timestamp(train["date"].max()) for d in after["date"])


def test_the_transformer_fit_and_predict():
    fc = TransformerForecaster(horizon=7, max_steps=5).fit(make_sessions())
    assert fc.is_fitted_ and "fitted=True" in repr(fc)

    pred = fc.predict(horizon=7, n_scenarios=3, seed=0)
    assert {"date", "scenario", "n_sessions"} <= set(pred.columns)
    assert len(pred) == 21 and (pred["n_sessions"] >= 0).all()

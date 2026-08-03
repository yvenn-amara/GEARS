"""
Shared pytest fixtures for GEARS tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gears.data.loader import make_demo_data
from gears.models.forecaster import SessionForecaster
from gears.models.session_model import EVSessionModel
from gears.pipeline import GEARSModel

# ---------------------------------------------------------------------------
# Dataset fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def work_sessions():
    """500 synthetic workplace charging sessions."""
    return make_demo_data(n=500, location_type="work", seed=0)


@pytest.fixture(scope="session")
def home_sessions():
    """300 synthetic home charging sessions."""
    return make_demo_data(n=300, location_type="home", seed=1)


@pytest.fixture(scope="session")
def public_sessions():
    """400 synthetic public charging sessions."""
    return make_demo_data(n=400, location_type="public", seed=2)


@pytest.fixture(scope="session")
def mixed_sessions(work_sessions, home_sessions, public_sessions):
    """Mixed dataset with all location types."""
    return pd.concat([work_sessions, home_sessions, public_sessions], ignore_index=True)


# ---------------------------------------------------------------------------
# Model fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def fitted_gmm(work_sessions):
    """A GMM fitted on workplace sessions."""
    gmm = EVSessionModel(n_components=3, random_state=42)
    return gmm.fit(work_sessions)


@pytest.fixture(scope="session")
def fitted_forecaster(work_sessions):
    """A probabilistic forecaster fitted on workplace sessions."""
    fc = SessionForecaster(method="probabilistic")
    return fc.fit(work_sessions)


@pytest.fixture(scope="session")
def fitted_model(work_sessions):
    """A fully fitted GEARSModel (probabilistic, 3 components)."""
    model = GEARSModel(n_components=3, n_scenarios=3, random_state=42)
    return model.fit(work_sessions, verbose=False)


# ---------------------------------------------------------------------------
# Signal fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def price_signal():
    """48-hour electricity price signal at 30-min resolution."""
    idx = pd.date_range("2024-06-10", periods=96, freq="30min")
    rng = np.random.default_rng(99)
    prices = 0.15 + 0.08 * np.sin(np.linspace(0, 4 * np.pi, 96)) + rng.normal(0, 0.01, 96)
    return pd.Series(np.maximum(0.05, prices), index=idx, name="price_eur_kwh")


@pytest.fixture
def res_signal():
    """48-hour renewable energy fraction signal at 30-min resolution."""
    idx = pd.date_range("2024-06-10", periods=96, freq="30min")
    rng = np.random.default_rng(77)
    res = 0.4 + 0.3 * np.sin(np.linspace(0, 4 * np.pi, 96)) + rng.normal(0, 0.02, 96)
    return pd.Series(np.clip(res, 0, 1), index=idx, name="res_fraction")

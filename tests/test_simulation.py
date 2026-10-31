"""Tests for the short-term simulator and the charger mix both simulators share.

The output invariants (columns, physical values, window, reproducibility, scenarios) are
checked on the facade in ``test_sanity.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gears.models.forecaster import PersistenceForecaster, sessions_to_daily_counts
from gears.simulation.medium_term import MediumTermSimulator
from gears.simulation.short_term import ShortTermSimulator


def test_the_number_of_sessions_per_day_comes_from_the_forecaster(model, demo):
    forecaster = PersistenceForecaster(n_weeks=1).fit(demo)
    sim = ShortTermSimulator(model.gmm_, forecaster)
    counts = sessions_to_daily_counts(demo)
    start = counts.index[-1] + pd.Timedelta(days=1)

    sessions = sim.simulate(start, horizon=5, n_scenarios=1, seed=0)

    expected = [counts[d - pd.Timedelta(weeks=1)] for d in pd.date_range(start, periods=5)]
    per_day = sessions.groupby("date").size()
    assert [per_day.get(d.date(), 0) for d in pd.date_range(start, periods=5)] == expected


def test_without_a_forecaster_every_day_gets_the_default_number_of_sessions_and_says_so(model):
    with pytest.warns(UserWarning, match="No fitted forecaster.*20 sessions"):
        sessions = ShortTermSimulator(model.gmm_).simulate(
            "2025-06-10", horizon=3, n_scenarios=2, seed=0
        )
    assert len(sessions) == 3 * 2 * 20
    assert (sessions.groupby(["scenario", "date"]).size() == 20).all()


def test_a_single_day_has_the_requested_number_of_sessions_on_that_date(model):
    day = ShortTermSimulator(model.gmm_).simulate_single_day("2025-06-10", n_sessions=50, seed=0)
    assert len(day) == 50
    assert (day["date"] == pd.Timestamp("2025-06-10").date()).all()
    assert (day["arrival_time"].dt.date == day["date"]).all()


def test_the_charger_mix_is_normalised_and_decides_the_power_levels(model):
    gmm = model.gmm_
    for simulator in (ShortTermSimulator, MediumTermSimulator):
        assert simulator(gmm, charger_mix={7.4: 3, 22.0: 1}).charger_mix == {7.4: 0.75, 22.0: 0.25}

    only_11 = ShortTermSimulator(gmm, charger_mix={11.0: 1.0}).simulate_single_day(
        "2025-06-10", n_sessions=30, seed=0
    )
    assert (only_11["power_kw"] == 11.0).all()

    mixed = ShortTermSimulator(gmm, charger_mix={7.4: 0.5, 22.0: 0.5}).simulate_single_day(
        "2025-06-10", n_sessions=400, seed=0
    )
    assert set(mixed["power_kw"]) == {7.4, 22.0}
    assert (mixed["power_kw"] == 7.4).mean() == pytest.approx(0.5, abs=0.1)


def test_a_slow_charger_caps_the_energy_at_power_times_duration(model):
    """A charger cannot deliver more than its power over the session: the energy the model
    draws is cut to ``power_kw * duration`` (the other tests never reach that limit)."""
    date, n, seed = "2025-06-10", 300, 3
    raw = model.gmm_.sample(
        n_sessions=n, date=pd.Timestamp(date),
        seed=int(np.random.default_rng(seed).integers(0, 2**31)),
    )

    slow = ShortTermSimulator(model.gmm_, charger_mix={0.5: 1.0}).simulate_single_day(
        date, n_sessions=n, seed=seed
    )
    limit = 0.5 * raw["duration"].to_numpy()
    assert (raw["energy"].to_numpy() > limit).sum() > 20, "the test needs sessions above the limit"
    np.testing.assert_allclose(slow["energy"], np.minimum(raw["energy"], limit))
    assert (slow["energy"] <= limit + 1e-9).all()

    fast = ShortTermSimulator(model.gmm_, charger_mix={10_000.0: 1.0}).simulate_single_day(
        date, n_sessions=n, seed=seed
    )
    np.testing.assert_allclose(fast["energy"], raw["energy"])        # never binds: untouched

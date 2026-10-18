"""Tests for the V1G optimiser's summaries: savings_summary and compute_regret.

How sessions are scheduled (window, power, cost against an independent recomputation,
efficiency, warnings, invalid input) is checked in ``test_sanity.py`` (invariant 9).
All the figures below are computed by hand.
"""

from __future__ import annotations

import pandas as pd

from gears.data.loader import make_demo_data
from gears.smart_charging.optimizer import SmartChargingOptimizer
from gears.utils import make_price_signal

# 10 kW is 5 kWh per half-hour slot; the slots from 08:00 cost 1, 4, 2 and 3 EUR/kWh.
PRICES = pd.Series(
    [1.0, 4.0, 2.0, 3.0], index=pd.date_range("2025-06-10 08:00", periods=4, freq="30min")
)


def sessions(*energies_kwh):
    """Sessions plugged in at 08:00 for two hours on a 10 kW charger."""
    return pd.DataFrame({
        "arrival_time": [pd.Timestamp("2025-06-10 08:00")] * len(energies_kwh),
        "duration": [2.0] * len(energies_kwh),
        "energy": list(energies_kwh),
        "power_kw": [10.0] * len(energies_kwh),
    })


def test_savings_summary_adds_up_the_sessions():
    opt = SmartChargingOptimizer("price", resolution_min=30)
    # 15 kWh: plug 5*(1+4+2) = 35, smart 5*(1+2+3) = 30 (-14.29 %); 5 kWh twice: 5 in both (0 %)
    result = opt.optimise(sessions(15.0, 5.0, 5.0), PRICES)

    assert opt.savings_summary(result) == {
        "total_cost_smart_eur": 40.0,
        "total_cost_plug_eur": 45.0,
        "total_savings_eur": 5.0,
        "mean_savings_pct": 4.76,                    # the mean of 14.29, 0 and 0 (not their median)
        "n_sessions_optimised": 3,
    }
    # sessions the signal does not cover are not optimised: there is nothing to summarise
    not_covered = opt.optimise(sessions(5.0), PRICES.shift(freq="30D"))
    assert opt.savings_summary(not_covered) == {}


def test_a_session_without_power_uses_the_default_charger_power():
    demo = make_demo_data(n=20, seed=0, start_date="2025-06-01", end_date="2025-06-07")
    signal = make_price_signal(start="2025-06-01", periods=7 * 48, resolution_min=30)
    without = demo.drop(columns=["power_kw"], errors="ignore")

    default = SmartChargingOptimizer("price").optimise(without, signal)
    explicit = SmartChargingOptimizer("price").optimise(without.assign(power_kw=7.4), signal)
    slower = SmartChargingOptimizer("price", default_power_kw=3.7).optimise(without, signal)
    slower_explicit = SmartChargingOptimizer("price").optimise(without.assign(power_kw=3.7), signal)

    assert default["cost_plug"].notna().all()
    for column in ("cost_smart", "cost_plug"):
        pd.testing.assert_series_equal(default[column], explicit[column])        # 7.4 kW
        pd.testing.assert_series_equal(slower[column], slower_explicit[column])  # the argument
    assert not default["cost_plug"].equals(slower["cost_plug"])                  # and it matters


def test_regret_compares_what_the_forecast_costs_with_what_the_real_sessions_cost():
    opt = SmartChargingOptimizer("price", resolution_min=30)
    oracle, predicted, naive = sessions(15.0), sessions(10.0), sessions(5.0)
    # smart costs: 15 kWh -> 30, 10 kWh -> 5*1 + 5*2 = 15, 5 kWh -> 5; plug cost of 10 kWh: 25

    regret = opt.compute_regret(oracle, predicted, PRICES, persistence_sessions=naive)

    assert regret == {
        "cost_oracle_smart": 30.0,
        "cost_predicted_smart": 15.0,
        "cost_predicted_plug": 25.0,
        "regret_smart_vs_oracle": -15.0,
        "regret_plug_vs_oracle": -5.0,
        "value_of_smart_charging": 10.0,
        "cost_persistence_smart": 5.0,
        "value_of_forecast_vs_persistence": -10.0,
    }
    assert all(type(v) is float for v in regret.values())    # scalars, not one-column frames

    same = opt.compute_regret(oracle, oracle, PRICES)
    assert same["regret_smart_vs_oracle"] == 0.0 and same["value_of_smart_charging"] == 5.0
    assert "cost_persistence_smart" not in same

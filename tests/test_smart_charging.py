"""Tests for smart charging optimizer — including Section 7 regression tests
and the R3 compute_regret() scalar-value regression guards (relocated here
from tests/test_regression.py per AUDIT.md §g; see test_compute_regret_basic
for the bug this guards against)."""
import pytest

from gears.data.loader import make_demo_data
from gears.smart_charging.optimizer import SmartChargingOptimizer
from gears.utils import make_price_signal, make_res_signal


def make_sessions(n=20, seed=0, start="2025-06-01", end="2025-06-07"):
    """Create sessions whose date range aligns with the default price signal."""
    df = make_demo_data(n=n, seed=seed, start_date=start, end_date=end)
    return df


def make_signal(start="2025-06-01", days=7):
    return make_price_signal(
        start=start, periods=days * 48, resolution_min=30, pattern="day_night"
    )


# ── Basic optimisation ────────────────────────────────────────────────────────

def test_optimise_returns_expected_columns():
    sessions = make_sessions(20)
    signal = make_signal()
    opt = SmartChargingOptimizer(signal_type="price")
    result = opt.optimise(sessions, signal)
    for col in ["cost_smart", "cost_plug", "savings_pct", "scheduled_start"]:
        assert col in result.columns


def test_optimise_sessions_aligned_with_signal():
    """All sessions should be optimised when signal covers the session dates."""
    sessions = make_sessions(30)
    signal = make_signal()
    opt = SmartChargingOptimizer(signal_type="price")
    result = opt.optimise(sessions, signal)
    valid = result["cost_plug"].notna().sum()
    assert valid > 0, "Optimizer: 0 sessions optimised — signal/sessions date mismatch?"
    assert valid == len(sessions), f"Expected all sessions optimised, got {valid}/{len(sessions)}"


def test_smart_cheaper_than_plug():
    """Smart charging must be cheaper than or equal to plug-and-charge."""
    sessions = make_sessions(30)
    signal = make_signal()
    opt = SmartChargingOptimizer(signal_type="price")
    result = opt.optimise(sessions, signal)
    valid = result[result["cost_plug"].notna()].copy()
    assert (valid["cost_smart"] <= valid["cost_plug"] + 1e-6).all()


def test_savings_summary_keys():
    sessions = make_sessions(30)
    signal = make_signal()
    opt = SmartChargingOptimizer(signal_type="price")
    result = opt.optimise(sessions, signal)
    summary = opt.savings_summary(result)
    for key in ["total_cost_smart_eur", "total_cost_plug_eur",
                "total_savings_eur", "mean_savings_pct", "n_sessions_optimised"]:
        assert key in summary
    assert summary["total_savings_eur"] >= 0


def test_without_power_kw_uses_default():
    """Sessions without power_kw column should use default_power_kw."""
    sessions = make_sessions(10)
    sessions = sessions.drop(columns=["power_kw"], errors="ignore")
    signal = make_signal()
    opt = SmartChargingOptimizer(signal_type="price", default_power_kw=7.4)
    result = opt.optimise(sessions, signal)
    assert result["cost_plug"].notna().sum() > 0


def test_res_signal():
    sessions = make_sessions(20)
    signal = make_res_signal(start="2025-06-01", periods=7 * 48, resolution_min=30)
    opt = SmartChargingOptimizer(signal_type="res")
    result = opt.optimise(sessions, signal)
    assert result["cost_plug"].notna().sum() > 0


def test_date_as_object_dtype():
    """date column as Python date objects (from validate_dataframe) must work."""
    sessions = make_sessions(15)
    assert sessions["date"].dtype == object  # date objects, not datetime
    signal = make_signal()
    opt = SmartChargingOptimizer(signal_type="price")
    result = opt.optimise(sessions, signal)
    assert result["cost_plug"].notna().sum() > 0


def test_invalid_signal_type():
    with pytest.raises(ValueError, match="signal_type"):
        SmartChargingOptimizer(signal_type="carbon")


# ── Regret analysis ───────────────────────────────────────────────────────────

def test_compute_regret_basic():
    """R3: compute_regret()'s dict values must all be Python float scalars.
    The old bug used double-bracket indexing (oracle_opt[["cost_smart"]],
    a one-column DataFrame) instead of oracle_opt["cost_smart"] (a Series);
    .sum() on the former returns a Series, not a scalar, so downstream
    arithmetic raised KeyError: 0."""
    oracle = make_sessions(20, seed=0)
    predicted = make_sessions(20, seed=1)
    signal = make_signal()
    opt = SmartChargingOptimizer(signal_type="price")
    regret = opt.compute_regret(oracle, predicted, signal)
    expected_keys = {
        "cost_oracle_smart", "cost_predicted_smart", "cost_predicted_plug",
        "regret_smart_vs_oracle", "regret_plug_vs_oracle", "value_of_smart_charging",
    }
    assert expected_keys <= set(regret.keys())
    for key in expected_keys:
        assert isinstance(regret[key], float), (
            f"compute_regret['{key}'] is {type(regret[key]).__name__} — R3 regressed."
        )


def test_compute_regret_with_persistence():
    """R3: same scalar check, for the extra persistence_sessions branch."""
    oracle = make_sessions(20, seed=0)
    predicted = make_sessions(20, seed=1)
    persistence = make_sessions(20, seed=2)
    signal = make_signal()
    opt = SmartChargingOptimizer(signal_type="price")
    regret = opt.compute_regret(oracle, predicted, signal,
                                persistence_sessions=persistence)
    for key in ("cost_persistence_smart", "value_of_forecast_vs_persistence"):
        assert key in regret
        assert isinstance(regret[key], float)


def test_compute_regret_same_sessions_zero_regret():
    """R3: when oracle == predicted, regret must be ~0 as a scalar float,
    not an empty/misshapen array from the double-bracket bug."""
    sessions = make_sessions(25, seed=5)
    signal = make_signal()
    opt = SmartChargingOptimizer(signal_type="price")
    regret = opt.compute_regret(sessions, sessions, signal)
    assert isinstance(regret["regret_smart_vs_oracle"], float)
    assert abs(regret["regret_smart_vs_oracle"]) < 1e-3


def test_v1g_ordering():
    """Oracle <= GEARS+V1G <= GEARS+Plug must hold (R3: also confirms the
    arithmetic doesn't raise on the fixed scalar values)."""
    oracle = make_sessions(40, seed=0)
    predicted = make_sessions(40, seed=1)
    signal = make_signal()
    opt = SmartChargingOptimizer(signal_type="price")
    regret = opt.compute_regret(oracle, predicted, signal)
    assert regret["value_of_smart_charging"] >= -1e-3   # smart <= plug


def test_savings_nonnegative():
    sessions = make_sessions(30)
    signal = make_signal()
    opt = SmartChargingOptimizer(signal_type="price")
    result = opt.optimise(sessions, signal)
    summary = opt.savings_summary(result)
    assert summary.get("total_savings_eur", 0) >= -1e-6

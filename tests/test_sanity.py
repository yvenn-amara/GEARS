"""
S6 — sanity checks, error messages and output invariants.

Part 1 has at least one test per row of the S6 table of the refonte plan: what
happens with bad data, bad arguments, an unfitted model, an unknown context or a
SARIMA that cannot be fitted. Each test checks the exception type **and** that the
message tells the reader what to do, because that is what makes it actionable.

Part 2 tests the ten invariants every simulation output must satisfy. Everything
uses small demo datasets and fixed seeds, so nothing here is flaky.
"""

from __future__ import annotations

import datetime
import sys
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from gears import GEARSModel
from gears.data.loader import load_sessions, make_demo_data
from gears.data.schemas import validate_dataframe
from gears.models.forecaster import PersistenceForecaster, SessionForecaster
from gears.models.session_model import MIN_SESSIONS_PER_CONTEXT, EVSessionModel
from gears.simulation import MediumTermSimulator, ShortTermSimulator
from gears.simulation.medium_term import (
    bass_diffusion_profile,
    linear_growth_profile,
    s_curve_growth_profile,
)
from gears.smart_charging import SmartChargingOptimizer
from gears.utils import make_price_signal, make_res_signal


def _warnings_of(fn, *args, **kwargs):
    """Call ``fn`` and return ``(result, [UserWarning messages])``."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fn(*args, **kwargs)
    return result, [str(w.message) for w in caught if issubclass(w.category, UserWarning)]


def _raw(df: pd.DataFrame) -> pd.DataFrame:
    """The three required columns of a demo dataset, as a user's raw export would have them."""
    return df[["arrival_time", "duration", "energy"]].reset_index(drop=True)


@pytest.fixture(scope="module")
def demo() -> pd.DataFrame:
    return make_demo_data(n=1500, location_type="work", seed=0)


@pytest.fixture(scope="module")
def model(demo) -> GEARSModel:
    """A fitted model; 'probabilistic' counts keep the fit fast."""
    return GEARSModel(
        n_components=3, forecaster_method="probabilistic", n_scenarios=3, random_state=0
    ).fit(demo, verbose=False)


# =============================================================================
# Part 1 — the cases of the S6 table
# =============================================================================


# ── Required columns are missing ─────────────────────────────────────────────

def test_missing_columns_are_listed_with_the_recognised_aliases():
    df = pd.DataFrame({"foo": [1, 2, 3], "duration": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError) as err:
        validate_dataframe(df)
    msg = str(err.value)
    assert "arrival_time" in msg and "energy" in msg   # what is missing
    assert "start_time" in msg and "kwh" in msg        # which names are recognised
    assert "foo" in msg                                # what the file has instead


def test_aliases_listed_in_the_message_really_work(demo):
    raw = _raw(demo).rename(columns={"arrival_time": "start_time", "energy": "kwh"})
    out = validate_dataframe(raw)
    assert {"arrival_time", "energy", "duration"} <= set(out.columns)
    assert len(out) == len(demo)


# ── Empty DataFrame / fewer than N sessions ──────────────────────────────────

def test_empty_dataframe_raises_a_value_error_that_says_what_to_check(demo):
    with pytest.raises(ValueError, match="0 sessions|no rows"):
        validate_dataframe(_raw(demo).iloc[:0])


def test_too_few_sessions_name_the_minimum_instead_of_a_no_gmm_runtime_error(demo):
    few = load_sessions(_raw(demo).iloc[:5], verbose=False)
    with pytest.raises(ValueError, match=rf"{MIN_SESSIONS_PER_CONTEXT}") as err:
        EVSessionModel(n_components=2).fit(few)
    assert "5" in str(err.value)                      # how many it got
    assert "more" in str(err.value).lower()           # what to do


def test_too_few_sessions_through_the_facade_raise_the_same_error(demo):
    with pytest.raises(ValueError, match=rf"{MIN_SESSIONS_PER_CONTEXT}"):
        GEARSModel(n_components=2).fit(_raw(demo).iloc[:5], verbose=False)


def test_a_dataset_split_into_contexts_that_are_all_too_small_is_refused():
    df = make_demo_data(n=40, seed=0)   # 40 sessions over a year: no context reaches 10
    with pytest.raises(ValueError, match="Stratify by fewer columns"):
        EVSessionModel(n_components=2, stratify_by=["day_of_week", "season"]).fit(df)


# ── Contexts with few sessions ───────────────────────────────────────────────

@pytest.fixture(scope="module")
def sparse_fit():
    df = make_demo_data(n=400, seed=1)   # 28 contexts, a few of them below 10 sessions
    return _warnings_of(
        lambda: GEARSModel(n_components=2, forecaster_method="probabilistic").fit(
            df, verbose=False
        )
    )


def test_sparse_contexts_give_one_summary_warning_not_one_per_context(sparse_fit):
    _, msgs = sparse_fit
    sparse = [m for m in msgs if "no model of their own" in m]
    assert len(sparse) == 1
    assert "contexts have fewer than" in sparse[0]
    assert "stratify by fewer columns" in sparse[0]    # what to do


def test_sparse_contexts_are_visible_in_the_summary(sparse_fit):
    fitted, _ = sparse_fit
    skipped = fitted.gmm_.skipped_contexts_
    assert skipped and all(n < MIN_SESSIONS_PER_CONTEXT for n in skipped.values())
    text = fitted.summary()
    assert f"{len(skipped)} with < {MIN_SESSIONS_PER_CONTEXT} sessions" in text
    assert "closest fitted context" in text


def test_well_populated_contexts_give_no_sparse_warning_and_no_summary_line(model):
    assert not model.gmm_.skipped_contexts_
    assert "Contexts" not in model.summary()


# ── duration <= 0, energy < 0, NaN, duplicates ───────────────────────────────

def _dirty(demo) -> pd.DataFrame:
    raw = _raw(demo).iloc[:400].copy()
    raw.loc[0:1, "duration"] = [0.0, -2.0]        # 2 non-positive durations
    raw.loc[2, "energy"] = -5.0                   # 1 negative energy
    raw.loc[3, "energy"] = np.nan                 # 1 missing energy
    return raw


def test_bad_rows_are_filtered_and_reported_with_counts(demo):
    out, msgs = _warnings_of(validate_dataframe, _dirty(demo))
    assert len(out) == 396
    assert (out["duration"] > 0).all() and (out["energy"] >= 0).all()
    assert not out[["duration", "energy"]].isna().any().any()
    report = [m for m in msgs if "Dropped 4/400" in m]
    assert report, msgs
    for part in ("2 with duration <= 0", "1 with negative energy", "1 with a missing energy"):
        assert part in report[0]


def test_exact_duplicates_are_dropped_and_reported(demo):
    raw = _raw(demo).iloc[:400]
    dup = pd.concat([raw, raw.iloc[:20]], ignore_index=True)
    out, msgs = _warnings_of(validate_dataframe, dup)
    assert len(out) == 400
    assert any("20 duplicating an earlier row" in m for m in msgs)


def test_strict_mode_raises_instead_of_filtering(demo):
    with pytest.raises(ValueError, match="duration <= 0"):
        validate_dataframe(_dirty(demo), strict=True)


def test_everything_filtered_is_an_error_that_says_what_to_check(demo):
    raw = _raw(demo).iloc[:100].copy()
    raw["energy"] = -1.0
    with pytest.raises(ValueError) as err:
        validate_dataframe(raw)
    msg = str(err.value)
    assert "100/100" in msg and "Nothing is left" in msg
    assert "kWh" in msg                                # what to check


# ── Suspicious units ─────────────────────────────────────────────────────────

def test_energy_probably_in_wh_triggers_a_warning_that_states_the_heuristic(demo):
    raw = _raw(demo).iloc[:400].copy()
    raw["energy"] = raw["energy"] * 1000.0
    _, msgs = _warnings_of(validate_dataframe, raw)
    hint = [m for m in msgs if "probably in Wh" in m]
    assert hint, msgs
    assert "median energy" in hint[0] and "above 200" in hint[0]   # the heuristic
    assert "divide" in hint[0]                                      # what to do


def test_duration_probably_in_minutes_triggers_a_warning_that_states_the_heuristic(demo):
    raw = _raw(demo).iloc[:400].copy()
    raw["duration"] = raw["duration"] * 60.0
    _, msgs = _warnings_of(validate_dataframe, raw)
    hint = [m for m in msgs if "probably in minutes" in m]
    assert hint, msgs
    assert "median duration" in hint[0] and "above 48" in hint[0]


def test_plausible_units_give_no_unit_warning(demo):
    _, msgs = _warnings_of(validate_dataframe, _raw(demo))
    assert not [m for m in msgs if "probably in" in m]


# ── Time zones: convention documented and tested ─────────────────────────────

def test_a_time_zone_is_dropped_keeping_the_clock_reading_and_announced(demo):
    raw = _raw(demo).iloc[:300].copy()
    naive = pd.to_datetime(raw["arrival_time"])
    raw["arrival_time"] = naive.dt.tz_localize(datetime.timezone(datetime.timedelta(hours=2)))
    out, msgs = _warnings_of(validate_dataframe, raw)
    assert out["arrival_time"].dt.tz is None
    assert any("wall-clock" in m and "not converted" in m for m in msgs)
    # The clock reading is kept, not converted to UTC: 08:00+02:00 stays 08:00.
    assert np.array_equal(np.sort(out["arrival_time"].to_numpy()), np.sort(naive.to_numpy()))


def test_simulated_arrival_times_are_naive_local_times(model):
    sessions = model.simulate_short_term("2025-06-10", horizon=2, n_scenarios=2, seed=0)
    assert sessions["arrival_time"].dt.tz is None


def test_an_aware_start_date_is_read_as_the_same_clock_reading(model):
    naive = model.simulate_short_term("2025-06-10 00:00", horizon=2, n_scenarios=2, seed=7)
    aware = model.simulate_short_term(
        pd.Timestamp("2025-06-10 00:00", tz="Asia/Tokyo"), horizon=2, n_scenarios=2, seed=7
    )
    pd.testing.assert_frame_equal(naive, aware)


# ── horizon < 1, n_scenarios < 1, invalid date ───────────────────────────────

@pytest.mark.parametrize("bad", [0, -3, 2.5, True, None, "7"])
def test_invalid_horizon_is_a_value_error_not_a_key_error(model, bad):
    with pytest.raises(ValueError, match="horizon"):
        model.simulate_short_term("2025-06-10", horizon=bad)


@pytest.mark.parametrize("bad", [0, -1, 1.5])
def test_invalid_n_scenarios_is_a_value_error(model, bad):
    with pytest.raises(ValueError, match="n_scenarios"):
        model.simulate_short_term("2025-06-10", horizon=2, n_scenarios=bad)
    with pytest.raises(ValueError, match="n_scenarios"):
        model.simulate_medium_term(years=1, n_scenarios=bad)


@pytest.mark.parametrize("bad", ["not-a-date", "2025-13-45", "", None, float("nan")])
def test_invalid_start_date_is_a_value_error_with_an_example(model, bad):
    with pytest.raises(ValueError, match="start_date") as err:
        model.simulate_short_term(bad, horizon=2)
    assert "2025-06-10" in str(err.value)             # an example of a valid date


def test_invalid_n_sessions_is_a_value_error(model):
    with pytest.raises(ValueError, match="n_sessions"):
        model.simulate_short_term("2025-06-10", horizon=2, n_sessions=0)


def test_the_simulators_validate_when_used_directly(model):
    sim = ShortTermSimulator(model.gmm_)
    with pytest.raises(ValueError, match="horizon"):
        sim.simulate("2025-06-10", horizon=0)
    with pytest.raises(ValueError, match="date"):
        sim.simulate_single_day("nope")
    with pytest.raises(ValueError, match="n_sessions"):
        model.gmm_.sample(0)
    with pytest.raises(ValueError, match="date"):
        model.gmm_.sample(5, date="nope")


@pytest.mark.parametrize(
    ("kwargs", "name"),
    [
        ({"years": 0}, "years"),
        ({"years": -1}, "years"),
        ({"years": float("nan")}, "years"),
        ({"annual_growth_rate": -1.0}, "annual_growth_rate"),
        ({"annual_growth_rate": float("inf")}, "annual_growth_rate"),
        ({"start_date": "tomorrow-ish"}, "start_date"),
        ({"output": "weekly"}, "output"),
    ],
)
def test_medium_term_arguments_are_validated(model, kwargs, name):
    sim = MediumTermSimulator(model.gmm_, n_scenarios=1)
    with pytest.raises(ValueError, match=name):
        sim.simulate(**{"years": 1, "start_date": "2025-01-01", **kwargs})


# ── Simulation before fit ────────────────────────────────────────────────────

def test_every_entry_point_raises_not_fitted_error_with_the_way_forward(demo):
    unfitted = GEARSModel()
    calls = [
        lambda: unfitted.simulate_short_term("2025-06-10"),
        lambda: unfitted.simulate_medium_term(years=1),
        lambda: unfitted.smart_charge(demo.head(3), make_price_signal("2023-01-01", 48)),
        lambda: EVSessionModel().sample(5),
        lambda: SessionForecaster().predict(3),
        lambda: PersistenceForecaster().predict(3),
    ]
    for call in calls:
        with pytest.raises(NotFittedError, match=r"\.fit\(|from_pretrained|load_pretrained"):
            call()


def test_not_fitted_error_is_the_scikit_learn_one():
    # The point of NotFittedError: it is also a ValueError and an AttributeError, like
    # every scikit-learn estimator, so generic handlers keep working.
    with pytest.raises(ValueError, match="not fitted"):
        EVSessionModel().sample(5)


# ── Unknown context at simulation time ───────────────────────────────────────

@pytest.fixture
def two_key_model(demo):
    """A model stratified by (day_of_week, season): one fresh instance per test."""
    return EVSessionModel(n_components=2, random_state=0).fit(demo)


def test_unknown_context_warns_once_and_names_requested_and_used_context(two_key_model):
    m = two_key_model
    _, first = _warnings_of(m.sample, 5, context={"day_of_week": 99, "season": "monsoon"}, seed=0)
    assert len(first) == 1
    assert "'day_of_week': 99" in first[0] and "'season': 'monsoon'" in first[0]   # requested
    assert "Using the closest fitted context" in first[0]                          # used
    assert "strict=True" in first[0]                                               # what to do
    _, second = _warnings_of(m.sample, 5, context={"day_of_week": 98, "season": "x"}, seed=0)
    assert second == []                                                            # once only
    assert len(m.fallback_contexts_) == 2                                          # both recorded


def test_unknown_context_still_returns_sessions_when_not_strict(two_key_model):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = two_key_model.sample(7, context={"day_of_week": 99, "season": "monsoon"}, seed=0)
    assert len(out) == 7


def test_strict_true_raises_instead_of_falling_back(two_key_model):
    with pytest.raises(ValueError, match="No mixture model was fitted for context") as err:
        two_key_model.sample(5, context={"day_of_week": 99, "season": "monsoon"}, strict=True)
    assert "strict=False" in str(err.value)
    assert not two_key_model.fallback_contexts_         # nothing was substituted silently


def test_a_fitted_context_never_triggers_the_warning(two_key_model):
    ctx = {"day_of_week": 2, "season": "summer"}
    _, msgs = _warnings_of(two_key_model.sample, 5, context=ctx, seed=0, strict=True)
    assert msgs == [] and not two_key_model.fallback_contexts_


def test_a_date_alone_cannot_pick_a_location_type_and_says_so():
    df = make_demo_data(n=900, location_type="work", seed=3)
    df2 = make_demo_data(n=900, location_type="home", seed=4)
    both = pd.concat([df, df2], ignore_index=True)
    m = EVSessionModel(n_components=2, stratify_by=["location_type", "day_of_week"]).fit(both)
    _, msgs = _warnings_of(m.sample, 5, date="2025-06-10")
    assert len(msgs) == 1 and "location_type not given" in msgs[0]
    with pytest.raises(ValueError, match="location_type not given"):
        m.sample(5, date="2025-06-10", strict=True)


def test_strict_is_threaded_through_the_simulators(demo):
    both = pd.concat(
        [demo, make_demo_data(n=900, location_type="home", seed=4)], ignore_index=True
    )
    m = GEARSModel(
        n_components=2, forecaster_method="probabilistic",
        stratify_by=["location_type", "day_of_week"],
    ).fit(both, verbose=False)
    with pytest.raises(ValueError, match="location_type not given"):
        m.simulate_short_term("2025-06-10", horizon=3, n_scenarios=2, seed=0, strict=True)
    with pytest.raises(ValueError, match="location_type not given"):
        m.simulate_short_term("2025-06-10", horizon=1, n_sessions=3, seed=0, strict=True)


def test_context_must_be_a_dict(two_key_model):
    with pytest.raises(ValueError, match="context must be a dict"):
        two_key_model.sample(5, context=("monday", "summer"))


def test_fallbacks_are_listed_in_the_summary(demo):
    m = GEARSModel(n_components=2, forecaster_method="probabilistic").fit(demo, verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.gmm_.sample(3, context={"day_of_week": 99, "season": "x"})
    assert "Fallbacks" in m.summary()


# ── Unknown forecaster_method ────────────────────────────────────────────────

def test_unknown_forecaster_method_lists_the_valid_values_at_construction():
    with pytest.raises(ValueError) as err:
        GEARSModel(forecaster_method="prophet")      # fails at once, not after a long fit
    msg = str(err.value)
    assert "prophet" in msg and "sarima" in msg and "probabilistic" in msg
    with pytest.raises(ValueError, match="sarima"):
        SessionForecaster(method="prophet")


# ── SARIMA: short series, pmdarima absent ────────────────────────────────────

def _days(df: pd.DataFrame, n_days: int) -> pd.DataFrame:
    first = pd.to_datetime(df["date"]).min()
    keep = pd.to_datetime(df["date"]) < first + pd.Timedelta(days=n_days)
    return df[keep].reset_index(drop=True)


def test_sarima_on_a_short_history_announces_a_fall_back_to_probabilistic():
    short = make_demo_data(n=300, seed=5, start_date="2023-01-01", end_date="2023-01-10")
    fc = SessionForecaster(method="sarima")
    _, msgs = _warnings_of(fc.fit, short)
    assert len(msgs) == 1
    assert "at least 4 seasonal periods (28 days)" in msgs[0] and "10 day(s)" in msgs[0]
    assert "probabilistic" in msgs[0]
    assert fc.method == "sarima" and fc.method_ == "probabilistic"   # asked vs actually used
    assert "used='probabilistic'" in repr(fc)
    pred = fc.predict(horizon=5, n_scenarios=2, seed=0)
    assert len(pred) == 10 and (pred["n_sessions"] >= 0).all()


def test_sarima_on_a_single_day_still_gives_finite_counts():
    one_day = make_demo_data(n=50, seed=5, start_date="2023-01-02", end_date="2023-01-02")
    fc = SessionForecaster(method="sarima")
    _warnings_of(fc.fit, one_day)
    assert np.isfinite(fc.std_daily_)                    # one observation has no std: 0, not NaN
    assert np.isfinite(fc.predict(horizon=3, seed=0)["n_sessions"]).all()


def test_probabilistic_forecaster_needs_no_warning_on_a_short_history():
    short = make_demo_data(n=100, seed=5, start_date="2023-01-01", end_date="2023-01-10")
    _, msgs = _warnings_of(SessionForecaster(method="probabilistic").fit, short)
    assert msgs == []


def test_forecaster_refuses_an_empty_dataset(demo):
    with pytest.raises(ValueError, match="empty DataFrame"):
        SessionForecaster().fit(demo.iloc[:0])


def test_pmdarima_absent_announces_the_statsmodels_substitute(demo, monkeypatch):
    monkeypatch.setitem(sys.modules, "pmdarima", None)     # makes `import pmdarima` fail
    history = _days(demo, 90)
    fc = SessionForecaster(method="sarima", use_holidays=False)
    _, msgs = _warnings_of(fc.fit, history)
    absent = [m for m in msgs if "pmdarima could not be imported" in m]
    assert len(absent) == 1
    assert "SARIMAX(1,1,1)x(1,1,0,7)" in absent[0] and "pip install pmdarima" in absent[0]
    assert fc.is_fitted_ and fc.method_ == "sarima"
    pred = fc.predict(horizon=7, n_scenarios=2, seed=0)
    assert len(pred) == 14 and (pred["n_sessions"] >= 0).all()


def test_a_failing_sarima_prediction_is_announced_once_not_swallowed(demo):
    history = _days(demo, 60)
    fc = SessionForecaster(method="probabilistic").fit(history)
    fc.method_ = "sarima"                                 # pretend a SARIMA was fitted ...

    class Broken:
        def predict(self, **_):
            raise RuntimeError("singular matrix")

    fc._model = Broken()                                  # ... and that it cannot predict

    _, first = _warnings_of(fc.predict, horizon=5, n_scenarios=2, seed=0)
    assert len(first) == 1 and "singular matrix" in first[0] and "historical mean" in first[0]
    _, second = _warnings_of(fc.predict, horizon=5, n_scenarios=2, seed=0)
    assert second == []


# =============================================================================
# Part 2 — the ten output invariants
# =============================================================================

START = pd.Timestamp("2025-06-10")
HORIZON = 7


@pytest.fixture(scope="module")
def sessions(model) -> pd.DataFrame:
    return model.simulate_short_term(START, horizon=HORIZON, n_scenarios=3, seed=11)


# 1. Column schema and dtypes -------------------------------------------------

def test_inv1_columns_and_dtypes(sessions):
    assert set(sessions.columns) == {
        "date", "scenario", "arrival_time", "arrival_hour", "duration", "energy", "power_kw",
    }
    for col in ("arrival_hour", "duration", "energy", "power_kw"):
        assert pd.api.types.is_float_dtype(sessions[col]), col
    assert pd.api.types.is_integer_dtype(sessions["scenario"])
    assert sessions["arrival_time"].dtype.kind == "M"          # datetime64, any resolution
    assert sessions["arrival_time"].dt.tz is None              # naive local time
    assert all(isinstance(d, datetime.date) for d in sessions["date"])


# 2. energy >= 0, duration > 0, no NaN ----------------------------------------

def test_inv2_values_are_physical_and_complete(sessions):
    assert (sessions["energy"] >= 0).all()
    assert (sessions["duration"] > 0).all()
    assert (sessions["power_kw"] > 0).all()
    assert not sessions.isna().any().any()
    assert sessions["arrival_hour"].between(0, 24, inclusive="left").all()


# 3. energy <= power x duration (0.1 % tolerance) -----------------------------

def test_inv3_energy_never_exceeds_what_the_charger_can_deliver(sessions):
    cap = sessions["power_kw"] * sessions["duration"]
    assert (sessions["energy"] <= cap * 1.001).all()


# 4. arrival_time in [start, start + horizon + 1 day[ -------------------------

def test_inv4_arrivals_fall_inside_the_requested_window(sessions):
    assert (sessions["arrival_time"] >= START).all()
    assert (sessions["arrival_time"] < START + pd.Timedelta(days=HORIZON + 1)).all()
    # and each session sits on the day it is labelled with
    assert (sessions["arrival_time"].dt.date == sessions["date"]).all()


# 5. Reproducibility ----------------------------------------------------------

def test_inv5_same_seed_same_result_different_seed_different_result(model):
    kw = {"horizon": 4, "n_scenarios": 2}
    a = model.simulate_short_term(START, seed=5, **kw)
    b = model.simulate_short_term(START, seed=5, **kw)
    c = model.simulate_short_term(START, seed=6, **kw)
    pd.testing.assert_frame_equal(a, b)
    assert not (len(a) == len(c) and np.allclose(a["energy"], c["energy"]))


def test_inv5_also_holds_with_a_fixed_number_of_sessions(model):
    kw = {"horizon": 4, "n_sessions": 6}
    a = model.simulate_short_term(START, seed=5, **kw)
    pd.testing.assert_frame_equal(a, model.simulate_short_term(START, seed=5, **kw))
    assert not np.allclose(a["energy"], model.simulate_short_term(START, seed=6, **kw)["energy"])


def test_inv5_a_day_of_one_seed_is_not_the_day_of_another_seed(model):
    """Day 7 of seed 0 and day 0 of seed 7 have the same weekday and season, hence the same
    context: with per-day seeds computed as ``seed + day`` they were identical."""
    def day(df, i):
        return df[df["date"] == (START + pd.Timedelta(days=i)).date()]["energy"].to_numpy()

    s0 = model.simulate_short_term(START, horizon=8, n_sessions=8, seed=0)
    s7 = model.simulate_short_term(START, horizon=1, n_sessions=8, seed=7)
    assert (START + pd.Timedelta(days=7)).dayofweek == START.dayofweek
    assert not np.allclose(day(s0, 7), day(s7, 0))


def test_inv5_no_seed_means_fresh_randomness(model):
    kw = {"horizon": 2, "n_sessions": 6}
    a = model.simulate_short_term(START, **kw)
    b = model.simulate_short_term(START, **kw)
    assert not np.allclose(a["energy"], b["energy"])


# 6. Scenarios are numbered 0 ... n-1 -----------------------------------------

def test_inv6_scenarios_are_numbered_from_zero_without_gaps(sessions):
    assert sorted(sessions["scenario"].unique()) == [0, 1, 2]


# 7. Simulated means within +-15 % of the training data -----------------------

def test_inv7_simulated_means_match_the_training_data(model, demo):
    sim = model.simulate_short_term(START, horizon=7, n_sessions=300, seed=0)
    assert len(sim) == 7 * 300
    for col in ("energy", "duration"):
        ratio = sim[col].mean() / demo[col].mean()
        assert 0.85 <= ratio <= 1.15, f"{col}: simulated/training mean = {ratio:.3f}"


# 8. save -> load gives the same simulation -----------------------------------

def test_inv8_save_then_load_reproduces_the_simulation(model, tmp_path):
    path = tmp_path / "model.joblib"
    model.save(path)
    loaded = GEARSModel.load(path)
    before = model.simulate_short_term(START, horizon=3, n_scenarios=2, seed=21)
    after = loaded.simulate_short_term(START, horizon=3, n_scenarios=2, seed=21)
    pd.testing.assert_frame_equal(before, after)
    pd.testing.assert_frame_equal(
        model.gmm_.sample(20, date=START, seed=3), loaded.gmm_.sample(20, date=START, seed=3)
    )


# 9. V1G optimiser ------------------------------------------------------------

@pytest.fixture(scope="module")
def optimised(model):
    """Simulated sessions scheduled against a price signal, plus the settings used."""
    sess = model.simulate_short_term(START, horizon=3, n_scenarios=1, n_sessions=25, seed=2)
    res_min = 30
    signal = make_price_signal(
        start=START, periods=48 * 6, resolution_min=res_min, pattern="spot_like"
    )
    opt = SmartChargingOptimizer(signal_type="price", resolution_min=res_min)
    result, msgs = _warnings_of(opt.optimise, sess, signal)
    return opt, sess, result, msgs, res_min


def _slots(result: pd.DataFrame, res_min: int) -> pd.Series:
    return (result["scheduled_end"] - result["scheduled_start"]) / pd.Timedelta(minutes=res_min)


def test_inv9_every_session_is_scheduled_and_columns_are_complete(optimised):
    _, sess, result, _, _ = optimised
    assert len(result) == len(sess)
    for col in ("cost_smart", "cost_plug", "savings_pct", "scheduled_start", "scheduled_end"):
        assert col in result.columns
    assert result[["cost_smart", "cost_plug", "scheduled_start", "scheduled_end"]].notna().all().all()


def test_inv9_sessions_stay_inside_their_plug_in_window(optimised):
    _, _, result, _, res_min = optimised
    departure = result["arrival_time"] + pd.to_timedelta(result["duration"], unit="h")
    assert (result["scheduled_start"] >= result["arrival_time"]).all()
    # The last slot may overhang an unaligned departure by less than one signal step.
    assert (result["scheduled_end"] <= departure + pd.Timedelta(minutes=res_min)).all()


def test_inv9_planned_energy_matches_the_session_energy_within_one_slot(optimised):
    opt, _, result, _, res_min = optimised
    step_h = res_min / 60
    n = _slots(result, res_min)
    assert np.allclose(n, n.round())                                  # whole slots
    planned = n * result["power_kw"] * step_h * opt.efficiency        # kWh delivered to the car
    one_slot = result["power_kw"] * step_h * opt.efficiency
    fully_charged = planned >= result["energy"] - 1e-9
    # Sessions that fit in their window: planned >= energy, and not a whole slot too much.
    assert (planned[fully_charged] - result["energy"][fully_charged] < one_slot[fully_charged]).all()
    # The others are the truncated ones, and the optimiser said so (see the next test).
    assert fully_charged.mean() > 0.5


def test_inv9_each_charging_slot_uses_the_charger_power_and_no_more(optimised):
    """Recompute every cost from the signal: slots x power x step x price, nothing else."""
    _, _, result, _, res_min = optimised
    step_h = res_min / 60
    signal = make_price_signal(
        start=START, periods=48 * 6, resolution_min=res_min, pattern="spot_like"
    )
    n_slots = _slots(result, res_min).round().astype(int)
    for _, row, n in zip(range(len(result)), result.itertuples(), n_slots):
        departure = row.arrival_time + pd.Timedelta(hours=row.duration)
        window = signal[(signal.index >= row.arrival_time) & (signal.index < departure)]
        kwh_per_slot = row.power_kw * step_h
        assert row.cost_smart == pytest.approx(window.nsmallest(n).sum() * kwh_per_slot)
        assert row.cost_plug == pytest.approx(window.iloc[:n].sum() * kwh_per_slot)


def test_inv9_smart_charging_never_costs_more_than_immediate_charging(optimised):
    _, _, result, _, _ = optimised
    assert (result["cost_smart"] <= result["cost_plug"] + 1e-12).all()
    assert (result["savings_pct"].dropna() >= -1e-9).all()
    assert result["cost_smart"].sum() < result["cost_plug"].sum()   # and it saves something


def test_inv9_res_signal_never_schedules_less_green_than_immediate_charging(model):
    sess = model.simulate_short_term(START, horizon=2, n_scenarios=1, n_sessions=15, seed=4)
    signal = make_res_signal(start=START, periods=48 * 4, resolution_min=30)
    opt = SmartChargingOptimizer(signal_type="res")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = opt.optimise(sess, signal)
    # for a RES signal the 'cost' columns hold the green fraction x energy: more is better
    assert (result["cost_smart"] >= result["cost_plug"] - 1e-12).all()


def test_inv9_sessions_that_cannot_be_fully_charged_are_announced(model):
    sess = model.simulate_short_term(START, horizon=1, n_scenarios=1, n_sessions=4, seed=1)
    sess["power_kw"] = 3.7
    sess["duration"] = 1.0
    sess["energy"] = 3.7 * 1.0          # exactly power x duration, so efficiency 0.9 cannot fit
    signal = make_price_signal(start=START, periods=96, resolution_min=30)
    _, msgs = _warnings_of(SmartChargingOptimizer("price").optimise, sess, signal)
    assert any("cannot be fully charged" in m and "4 of 4" in m for m in msgs), msgs


def test_inv9_sessions_outside_the_signal_are_announced_not_silently_nan(model):
    sess = model.simulate_short_term(START, horizon=1, n_scenarios=1, n_sessions=5, seed=1)
    late_signal = make_price_signal(start=START + pd.Timedelta(days=30), periods=96)
    result, msgs = _warnings_of(SmartChargingOptimizer("price").optimise, sess, late_signal)
    assert any("5 of 5 sessions were not optimised" in m and "covers" in m for m in msgs), msgs
    assert result["cost_smart"].isna().all()


@pytest.mark.parametrize(
    ("kwargs", "name"),
    [
        ({"efficiency": 0}, "efficiency"),
        ({"efficiency": 1.2}, "efficiency"),
        ({"resolution_min": 0}, "resolution_min"),
        ({"default_power_kw": -7.4}, "default_power_kw"),
        ({"signal_type": "carbon"}, "signal_type"),
    ],
)
def test_optimiser_arguments_are_validated(kwargs, name):
    with pytest.raises(ValueError, match=name):
        SmartChargingOptimizer(**kwargs)


# 10. Medium term -------------------------------------------------------------

GROWTH_PROFILES = {
    "linear": lambda base, rate: linear_growth_profile(base, 5, annual_growth_rate=rate),
    "s_curve": lambda base, rate: s_curve_growth_profile(
        base, 5, saturation_factor=1.0 + 3 * rate
    ),
    "bass": lambda base, rate: bass_diffusion_profile(
        base, 5, market_potential_factor=1.0 + 3 * rate
    ),
}


@pytest.mark.parametrize("name", GROWTH_PROFILES)
def test_inv10_zero_growth_gives_the_base_trajectory(name):
    flat = GROWTH_PROFILES[name](80.0, 0.0)
    assert np.allclose(flat.to_numpy(), 80.0)


@pytest.mark.parametrize("name", GROWTH_PROFILES)
def test_inv10_positive_growth_is_monotone_from_the_base(name):
    grown = GROWTH_PROFILES[name](80.0, 0.5)
    values = grown.to_numpy()
    assert values[0] == pytest.approx(80.0)
    assert (np.diff(values) >= -1e-12).all()
    assert values[-1] > values[0]


@pytest.mark.parametrize("name", GROWTH_PROFILES)
def test_inv10_trajectories_are_never_negative(name):
    assert (GROWTH_PROFILES[name](80.0, 0.5).to_numpy() >= 0).all()


def test_inv10_a_shrinking_fleet_is_clamped_at_zero_not_negative():
    declining = linear_growth_profile(50.0, 10, annual_growth_rate=-0.5)   # would hit < 0 at year 2
    assert (declining.to_numpy() >= 0).all()
    assert declining.iloc[-1] == 0.0


def test_inv10_simulated_energy_is_non_negative_and_follows_the_growth(model):
    def yearly_mean(rate: float) -> tuple[float, float, float]:
        sim = MediumTermSimulator(model.gmm_, growth_model="linear", n_scenarios=2, seed=0)
        out = sim.simulate(years=3, annual_growth_rate=rate, start_date="2025-01-01")
        series = out.groupby("date")["total_energy_kwh"].mean()   # mean over scenarios
        year = (pd.to_datetime(series.index) - pd.Timestamp("2025-01-01")).days // 365
        return float(series.min()), float(series[year == 0].mean()), float(series[year == 2].mean())

    lowest, first0, last0 = yearly_mean(0.0)
    assert lowest >= 0
    assert last0 == pytest.approx(first0, rel=0.15)         # growth 0: no trend
    _, first1, last1 = yearly_mean(0.5)
    assert last1 > 1.5 * first1                             # growth > 0: it grows

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


def _short_history(n_days: int) -> pd.DataFrame:
    end = (pd.Timestamp("2023-01-01") + pd.Timedelta(days=n_days - 1)).strftime("%Y-%m-%d")
    return make_demo_data(n=max(60, 6 * n_days), seed=5, start_date="2023-01-01", end_date=end)


@pytest.mark.parametrize("n_days", [1, 3, 7, 10, 14, 20, 27])
def test_sarima_is_always_tried_and_a_short_history_never_breaks_the_fit(n_days):
    """Whatever the history, fit() works; if SARIMA could not be fitted it says so and
    ``method_`` tells that 'probabilistic' is used."""
    fc = SessionForecaster(method="sarima", use_holidays=False)
    _, msgs = _warnings_of(fc.fit, _short_history(n_days))
    switched = [m for m in msgs if "SARIMA could not be fitted" in m]
    assert fc.is_fitted_ and fc.method == "sarima"
    if fc.method_ == "probabilistic":
        assert len(switched) == 1
        assert f"{n_days} day(s)" in switched[0] and "forecaster_method='probabilistic'" in switched[0]
    else:
        assert switched == []                  # it worked: no warning, SARIMA is used
    pred = fc.predict(horizon=7, n_scenarios=2, seed=0)
    assert len(pred) == 14 and np.isfinite(pred["n_sessions"]).all()


def test_a_sarima_fit_that_raises_switches_to_probabilistic_and_says_why(monkeypatch):
    """pmdarima refuses short series ('no more samples after seasonal differencing')."""
    import pmdarima

    def refuse(*args, **kwargs):
        raise ValueError("There are no more samples after a first-order seasonal differencing. See http://x")

    monkeypatch.setattr(pmdarima, "auto_arima", refuse)
    fc = SessionForecaster(method="sarima", use_holidays=False)
    _, msgs = _warnings_of(fc.fit, _short_history(10))
    assert len(msgs) == 1
    assert "ValueError: There are no more samples after a first-order seasonal differencing" in msgs[0]
    assert "http://x" not in msgs[0]                       # the pmdarima URL is cut off
    assert "Switching to method='probabilistic'" in msgs[0]
    assert fc.method == "sarima" and fc.method_ == "probabilistic"
    assert "used='probabilistic'" in repr(fc)
    assert len(fc.predict(horizon=5, n_scenarios=2, seed=0)) == 10


def test_a_sarima_fit_that_diverges_is_treated_as_failed(monkeypatch):
    """A fit that 'succeeds' but forecasts 264 sessions/day from a history of ~10 has failed."""
    class Diverging:
        _is_statsmodels = True

        def forecast(self, horizon, exog=None):
            return np.array([11.0, 0.0, 24.0, 0.0, 65.0, 0.0, 264.0])[:horizon]

    def fake_fit(self, exog=None):
        self._model = Diverging()

    monkeypatch.setitem(sys.modules, "pmdarima", None)
    monkeypatch.setattr(SessionForecaster, "_fit_statsmodels_fallback", fake_fit)
    fc = SessionForecaster(method="sarima", use_holidays=False)
    _, msgs = _warnings_of(fc.fit, _short_history(14))
    failed = [m for m in msgs if "SARIMA could not be fitted" in m]
    assert len(failed) == 1 and "forecast diverges" in failed[0] and "264" in failed[0]
    assert fc.method_ == "probabilistic" and fc._model is None


def test_a_sarima_forecast_with_nan_is_treated_as_failed(monkeypatch):
    class Broken:
        _is_statsmodels = True

        def forecast(self, horizon, exog=None):
            return np.full(horizon, np.nan)

    monkeypatch.setitem(sys.modules, "pmdarima", None)
    monkeypatch.setattr(SessionForecaster, "_fit_statsmodels_fallback",
                        lambda self, exog=None: setattr(self, "_model", Broken()))
    fc = SessionForecaster(method="sarima", use_holidays=False)
    _, msgs = _warnings_of(fc.fit, _short_history(14))
    assert any("SARIMA could not be fitted" in m for m in msgs) and fc.method_ == "probabilistic"


def test_a_sarima_that_fits_on_a_long_enough_history_is_kept_without_any_warning():
    fc = SessionForecaster(method="sarima", use_holidays=False)
    _, msgs = _warnings_of(fc.fit, _short_history(120))
    assert fc.method_ == "sarima" and msgs == []


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

RES_MIN = 30


@pytest.fixture(scope="module")
def optimised(model):
    """Simulated sessions scheduled against a price signal."""
    sess = model.simulate_short_term(START, horizon=3, n_scenarios=1, n_sessions=25, seed=2)
    signal = make_price_signal(
        start=START, periods=48 * 6, resolution_min=RES_MIN, pattern="spot_like"
    )
    opt = SmartChargingOptimizer(signal_type="price", resolution_min=RES_MIN)
    result, msgs = _warnings_of(opt.optimise, sess, signal)
    return sess, signal, result, msgs


def _departure(result: pd.DataFrame) -> pd.Series:
    return result["arrival_time"] + pd.to_timedelta(result["duration"], unit="h")


def test_inv9_every_session_is_scheduled_without_any_warning(optimised):
    """GEARS caps energy at power x duration: every simulated session must be completable."""
    sess, _, result, msgs = optimised
    assert msgs == []
    assert len(result) == len(sess)
    for col in ("cost_smart", "cost_plug", "savings_pct", "scheduled_start", "scheduled_end",
                "energy_scheduled"):
        assert col in result.columns
    assert not result[["cost_smart", "cost_plug", "scheduled_start", "scheduled_end",
                       "energy_scheduled"]].isna().any().any()


def test_inv9_planned_energy_is_the_session_energy(optimised):
    _, _, result, _ = optimised
    assert np.allclose(result["energy_scheduled"], result["energy"], rtol=1e-9)


def test_inv9_sessions_stay_inside_their_plug_in_window(optimised):
    _, _, result, _ = optimised
    tol = pd.Timedelta(milliseconds=1)
    assert (result["scheduled_start"] >= result["arrival_time"]).all()
    assert (result["scheduled_end"] <= _departure(result) + tol).all()


def test_inv9_the_maximum_power_is_respected(optimised):
    """Delivering E kWh at no more than P kW takes at least E / P hours."""
    _, _, result, _ = optimised
    span_h = (result["scheduled_end"] - result["scheduled_start"]) / pd.Timedelta(hours=1)
    assert (span_h >= result["energy"] / result["power_kw"] - 1e-9).all()


def test_inv9_costs_match_an_independent_slot_by_slot_recomputation(optimised):
    """Fill the slots by hand (plain Python): best price first / chronologically."""
    _, signal, result, _ = optimised
    step = pd.Timedelta(minutes=RES_MIN)
    for row in result.itertuples():
        arrival, departure = row.arrival_time, row.arrival_time + pd.Timedelta(hours=row.duration)
        slots = []   # (slot start, price, kWh the charger can give inside the window)
        for t, price in signal.items():
            overlap = min(t + step, departure) - max(t, arrival)
            if overlap > pd.Timedelta(0):
                slots.append((t, price, row.power_kw * overlap / pd.Timedelta(hours=1)))

        def cost(order, energy=row.energy):
            left, total = energy, 0.0
            for _, price, capacity in order:
                kwh = min(capacity, left)
                total += price * kwh
                left -= kwh
            return total

        assert row.cost_plug == pytest.approx(cost(slots))
        assert row.cost_smart == pytest.approx(cost(sorted(slots, key=lambda s: s[1])))


def test_inv9_smart_charging_never_costs_more_than_immediate_charging(optimised):
    _, _, result, _ = optimised
    assert (result["cost_smart"] <= result["cost_plug"] + 1e-12).all()
    assert (result["savings_pct"].dropna() >= -1e-9).all()
    assert result["cost_smart"].sum() < result["cost_plug"].sum()   # and it saves something


def test_inv9_res_signal_never_schedules_less_green_than_immediate_charging(model):
    sess = model.simulate_short_term(START, horizon=2, n_scenarios=1, n_sessions=15, seed=4)
    signal = make_res_signal(start=START, periods=48 * 4, resolution_min=RES_MIN)
    result = SmartChargingOptimizer(signal_type="res").optimise(sess, signal)
    # for a RES signal the 'cost' columns hold the green fraction x energy: more is better
    assert (result["cost_smart"] >= result["cost_plug"] - 1e-12).all()


def _one_session(arrival: str, duration: float, energy: float, power: float = 10.0):
    return pd.DataFrame({
        "arrival_time": [pd.Timestamp(arrival)], "duration": [duration],
        "energy": [energy], "power_kw": [power],
    })


def _signal(prices, start="2025-06-10 08:00"):
    idx = pd.date_range(start, periods=len(prices), freq="30min")
    return pd.Series(prices, index=idx)


def test_inv9_hand_computed_example_with_a_full_and_a_partial_slot():
    # 10 kW => 5 kWh per 30-min slot; prices 08:00..09:30 = 1, 4, 2, 3 EUR/kWh.
    opt = SmartChargingOptimizer("price", resolution_min=30)
    prices = [1.0, 4.0, 2.0, 3.0]
    full = opt.optimise(_one_session("2025-06-10 08:00", 2.0, 15.0), _signal(prices)).iloc[0]
    assert full["cost_plug"] == pytest.approx(5 * (1 + 4 + 2))      # slots 08:00, 08:30, 09:00
    assert full["cost_smart"] == pytest.approx(5 * (1 + 2 + 3))     # slots 08:00, 09:00, 09:30
    assert full["savings_pct"] == pytest.approx(100 * 5 / 35)
    assert full["scheduled_start"] == pd.Timestamp("2025-06-10 08:00")
    assert full["scheduled_end"] == pd.Timestamp("2025-06-10 10:00")
    assert full["energy_scheduled"] == pytest.approx(15.0)

    partial = opt.optimise(_one_session("2025-06-10 08:00", 2.0, 12.0), _signal(prices)).iloc[0]
    assert partial["cost_plug"] == pytest.approx(5 * 1 + 5 * 4 + 2 * 2)     # last slot half full
    assert partial["cost_smart"] == pytest.approx(5 * 1 + 5 * 2 + 2 * 3)
    assert partial["scheduled_end"] == pd.Timestamp("2025-06-10 09:42")     # 2 kWh at 10 kW = 12 min


def test_inv9_a_slot_only_partly_inside_the_window_offers_only_the_time_inside_it():
    # 08:10 -> 09:10 at 10 kW: 3.33 kWh in the 08:00 slot, 5 in 08:30, 1.67 in 09:00.
    result = SmartChargingOptimizer("price", 30).optimise(
        _one_session("2025-06-10 08:10", 1.0, 10.0), _signal([3.0, 1.0, 2.0])
    ).iloc[0]
    assert result["energy_scheduled"] == pytest.approx(10.0)
    assert result["cost_plug"] == pytest.approx(10 / 3 * 3 + 5 * 1 + 10 / 6 * 2)
    # the session fills its whole window, so there is nothing to shift
    assert result["cost_smart"] == pytest.approx(result["cost_plug"])
    assert result["scheduled_start"] == pd.Timestamp("2025-06-10 08:10")
    assert result["scheduled_end"] == pd.Timestamp("2025-06-10 09:10")


def test_inv9_sessions_at_the_simulators_energy_cap_are_fully_scheduled():
    """energy == power x duration whatever the duration and the alignment on the grid."""
    rng = np.random.default_rng(0)
    n = 200
    duration = rng.uniform(0.3, 12.0, n)
    power = rng.choice([3.7, 7.4, 22.0], n)
    sess = pd.DataFrame({
        "arrival_time": START + pd.to_timedelta(rng.uniform(0, 12, n), unit="h"),
        "duration": duration, "energy": power * duration, "power_kw": power,
    })
    signal = make_price_signal(start=START, periods=48 * 3, resolution_min=RES_MIN)
    result, msgs = _warnings_of(SmartChargingOptimizer("price", RES_MIN).optimise, sess, signal)
    assert msgs == []
    assert np.allclose(result["energy_scheduled"], result["energy"], rtol=1e-9)
    assert np.allclose(result["cost_smart"], result["cost_plug"])   # no room to shift


def test_inv9_a_session_that_cannot_be_completed_is_announced_and_partly_scheduled():
    sess = _one_session("2025-06-10 08:00", 1.0, 25.0, power=10.0)       # 25 kWh in 1 h at 10 kW
    result, msgs = _warnings_of(
        SmartChargingOptimizer("price", 30).optimise, sess, _signal([1.0, 2.0, 3.0])
    )
    assert len(msgs) == 1 and "1 of 1 sessions cannot be completed" in msgs[0]
    assert "energy_scheduled" in msgs[0]
    assert result.iloc[0]["energy_scheduled"] == pytest.approx(10.0)       # what fits


def test_inv9_efficiency_below_one_adds_losses_on_top_of_the_energy():
    sess = _one_session("2025-06-10 08:00", 2.0, 18.0, power=10.0)       # 20 kWh capacity
    prices = _signal([1.0, 1.0, 1.0, 1.0])
    result, msgs = _warnings_of(SmartChargingOptimizer("price", 30, efficiency=0.9).optimise,
                                sess, prices)
    assert msgs == []
    assert result.iloc[0]["cost_plug"] == pytest.approx(18.0 / 0.9)          # the charger draws more
    assert result.iloc[0]["energy_scheduled"] == pytest.approx(18.0)
    _, msgs = _warnings_of(SmartChargingOptimizer("price", 30, efficiency=0.9).optimise,
                           _one_session("2025-06-10 08:00", 2.0, 20.0, power=10.0), prices)
    assert any("energy / efficiency (0.9)" in m for m in msgs)               # 22.2 kWh > 20


def test_inv9_sessions_outside_the_signal_are_announced_not_silently_nan(model):
    sess = model.simulate_short_term(START, horizon=1, n_scenarios=1, n_sessions=5, seed=1)
    late_signal = make_price_signal(start=START + pd.Timedelta(days=30), periods=96)
    result, msgs = _warnings_of(SmartChargingOptimizer("price").optimise, sess, late_signal)
    assert any("5 of 5 sessions were not optimised" in m and "covers" in m for m in msgs), msgs
    assert result["cost_smart"].isna().all()


def test_inv9_unusable_sessions_are_counted_and_the_others_still_scheduled():
    good = _one_session("2025-06-10 08:00", 2.0, 10.0)
    bad = pd.concat([good, good, good], ignore_index=True)
    bad.loc[1, "energy"] = np.nan
    bad.loc[2, "duration"] = 0.0
    result, msgs = _warnings_of(SmartChargingOptimizer("price", 30).optimise, bad,
                                _signal([1.0, 2.0, 3.0, 4.0]))
    assert any("2 of 3 sessions were not optimised" in m and "no usable arrival" in m for m in msgs)
    assert result["cost_smart"].notna().tolist() == [True, False, False]


def test_inv9_a_signal_with_another_step_than_resolution_min_is_announced():
    sess = _one_session("2025-06-10 08:00", 2.0, 10.0)
    _, msgs = _warnings_of(SmartChargingOptimizer("price", 15).optimise, sess,
                           _signal([1.0, 2.0, 3.0, 4.0]))
    assert any("30-minute step" in m and "resolution_min=15" in m for m in msgs)


def test_inv9_bad_inputs_are_refused_with_a_clear_message():
    sess = _one_session("2025-06-10 08:00", 2.0, 10.0)
    with pytest.raises(TypeError, match="indexed by timestamps"):
        SmartChargingOptimizer("price", 30).optimise(sess, pd.Series([1.0, 2.0]))
    sess["power_kw"] = 0.0
    with pytest.raises(ValueError, match="power_kw must be > 0"):
        SmartChargingOptimizer("price", 30).optimise(sess, _signal([1.0, 2.0]))


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


def test_the_optimiser_assumes_no_extra_losses_by_default():
    assert SmartChargingOptimizer().efficiency == 1.0


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

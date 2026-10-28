"""
Sanity checks, error messages and output invariants.

Part 1 has one test per situation a user can get wrong: bad data, bad arguments, an unfitted
model, an unknown context, or a SARIMA that cannot be fitted. Each test checks the exception
type **and** that the message tells the reader what to do, because that is what makes it
actionable. A test covers several variants of a situation in a loop; the assertion messages
name the variant that failed.

Part 2 has one test per output invariant (the V1G optimiser has four).

Everything uses small demo datasets and fixed seeds, so nothing here is flaky.
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


def _raw(df: pd.DataFrame, n: int | None = None) -> pd.DataFrame:
    """The three required columns of a demo dataset, as a user's raw export would have them."""
    return df[["arrival_time", "duration", "energy"]].iloc[:n].reset_index(drop=True)


def _one_session(arrival: str, duration: float, energy: float, power: float = 10.0):
    return pd.DataFrame({
        "arrival_time": [pd.Timestamp(arrival)], "duration": [duration],
        "energy": [energy], "power_kw": [power],
    })


def _signal(prices, start="2025-06-10 08:00"):
    return pd.Series(prices, index=pd.date_range(start, periods=len(prices), freq="30min"))


# =============================================================================
# Part 1 — what a user can get wrong
# =============================================================================


def test_missing_columns_are_listed_with_the_recognised_aliases(demo):
    with pytest.raises(ValueError) as err:
        validate_dataframe(pd.DataFrame({"foo": [1, 2, 3], "duration": [1.0, 2.0, 3.0]}))
    msg = str(err.value)
    assert "arrival_time" in msg and "energy" in msg   # what is missing
    assert "start_time" in msg and "kwh" in msg        # which names are recognised
    assert "foo" in msg                                # what the file has instead

    # the aliases listed in the message really work
    out = validate_dataframe(_raw(demo).rename(columns={"arrival_time": "start_time", "energy": "kwh"}))
    assert {"arrival_time", "energy", "duration"} <= set(out.columns) and len(out) == len(demo)


def test_nothing_to_learn_from_is_refused_saying_what_to_check(demo):
    with pytest.raises(ValueError, match="0 sessions|no rows"):
        validate_dataframe(_raw(demo, 0))

    everything_filtered = _raw(demo, 100).assign(energy=-1.0)
    with pytest.raises(ValueError) as err:
        validate_dataframe(everything_filtered)
    assert "100/100" in str(err.value) and "Nothing is left" in str(err.value)
    assert "kWh" in str(err.value)                                 # what to check

    # fewer sessions than the minimum: the minimum and the count are named (it used to be
    # "RuntimeError: No GMM was fitted"), through the model and through the facade
    few = load_sessions(_raw(demo, 5), verbose=False)
    with pytest.raises(ValueError, match=rf"{MIN_SESSIONS_PER_CONTEXT}") as err:
        EVSessionModel(n_components=2).fit(few)
    assert "5" in str(err.value) and "more" in str(err.value).lower()
    with pytest.raises(ValueError, match=rf"{MIN_SESSIONS_PER_CONTEXT}"):
        GEARSModel(n_components=2).fit(_raw(demo, 5), verbose=False)

    # enough sessions overall, but split into contexts that are all too small
    year = make_demo_data(n=40, seed=0)       # 40 sessions over a year: no context reaches 10
    with pytest.raises(ValueError, match="Stratify by fewer columns"):
        EVSessionModel(n_components=2, stratify_by=["day_of_week", "season"]).fit(year)


def test_sparse_contexts_give_one_summary_warning_and_show_in_the_summary(model):
    sparse_data = make_demo_data(n=400, seed=1)   # 28 contexts, a few of them below 10 sessions
    fitted, msgs = _warnings_of(
        lambda: GEARSModel(n_components=2, forecaster_method="probabilistic").fit(
            sparse_data, verbose=False
        )
    )
    sparse = [m for m in msgs if "no model of their own" in m]
    assert len(sparse) == 1                                       # one, not one per context
    assert "contexts have fewer than" in sparse[0]
    assert "stratify by fewer columns" in sparse[0]               # what to do

    skipped = fitted.gmm_.skipped_contexts_
    assert skipped and all(n < MIN_SESSIONS_PER_CONTEXT for n in skipped.values())
    text = fitted.summary()
    assert f"{len(skipped)} with < {MIN_SESSIONS_PER_CONTEXT} sessions" in text
    assert "closest fitted context" in text

    # a well-populated model says nothing about it
    assert not model.gmm_.skipped_contexts_ and "Contexts" not in model.summary()


def test_bad_rows_are_filtered_and_counted_and_strict_mode_raises_instead(demo):
    dirty = _raw(demo, 400)
    dirty.loc[0:1, "duration"] = [0.0, -2.0]        # 2 non-positive durations
    dirty.loc[2, "energy"] = -5.0                   # 1 negative energy
    dirty.loc[3, "energy"] = np.nan                 # 1 missing energy

    out, msgs = _warnings_of(validate_dataframe, dirty)
    assert len(out) == 396
    assert (out["duration"] > 0).all() and (out["energy"] >= 0).all()
    assert not out[["duration", "energy"]].isna().any().any()
    report = [m for m in msgs if "Dropped 4/400" in m]
    assert report, msgs
    for part in ("2 with duration <= 0", "1 with negative energy", "1 with a missing energy"):
        assert part in report[0]

    with pytest.raises(ValueError, match="duration <= 0"):
        validate_dataframe(dirty, strict=True)

    raw = _raw(demo, 400)
    out, msgs = _warnings_of(validate_dataframe, pd.concat([raw, raw.iloc[:20]], ignore_index=True))
    assert len(out) == 400
    assert any("20 duplicating an earlier row" in m for m in msgs)


def test_suspicious_units_warn_with_the_heuristic_and_plausible_ones_do_not(demo):
    raw = _raw(demo, 400)
    for column, factor, unit, heuristic in (
        ("energy", 1000.0, "Wh", ("median energy", "above 200", "divide")),
        ("duration", 60.0, "minutes", ("median duration", "above 48")),
    ):
        _, msgs = _warnings_of(validate_dataframe, raw.assign(**{column: raw[column] * factor}))
        hint = [m for m in msgs if f"probably in {unit}" in m]
        assert hint, (column, msgs)
        for part in heuristic:                                    # the heuristic, and what to do
            assert part in hint[0], (column, part)

    _, msgs = _warnings_of(validate_dataframe, _raw(demo))
    assert not [m for m in msgs if "probably in" in m]


def test_a_time_zone_is_dropped_keeping_the_clock_reading_and_announced(demo):
    raw = _raw(demo, 300)
    naive = pd.to_datetime(raw["arrival_time"])
    raw["arrival_time"] = naive.dt.tz_localize(datetime.timezone(datetime.timedelta(hours=2)))
    out, msgs = _warnings_of(validate_dataframe, raw)
    assert out["arrival_time"].dt.tz is None
    assert any("wall-clock" in m and "not converted" in m for m in msgs)
    # The clock reading is kept, not converted to UTC: 08:00+02:00 stays 08:00.
    assert np.array_equal(np.sort(out["arrival_time"].to_numpy()), np.sort(naive.to_numpy()))


def test_simulated_arrival_times_are_naive_and_an_aware_start_date_keeps_its_clock_reading(model):
    naive = model.simulate_short_term("2025-06-10 00:00", horizon=2, n_scenarios=2, seed=7)
    assert naive["arrival_time"].dt.tz is None
    aware = model.simulate_short_term(
        pd.Timestamp("2025-06-10 00:00", tz="Asia/Tokyo"), horizon=2, n_scenarios=2, seed=7
    )
    pd.testing.assert_frame_equal(naive, aware)


def test_invalid_short_term_arguments_are_value_errors_that_name_the_argument(model):
    cases = (
        [({"horizon": h}, "horizon") for h in (0, -3, 2.5, True, None, "7")]   # was KeyError: 'date'
        + [({"n_scenarios": n}, "n_scenarios") for n in (0, -1, 1.5)]
        + [({"n_sessions": 0}, "n_sessions")]
    )
    for kwargs, name in cases:
        with pytest.raises(ValueError, match=name):
            model.simulate_short_term("2025-06-10", **{"horizon": 2, **kwargs})

    for bad in ("not-a-date", "2025-13-45", "", None, float("nan")):
        with pytest.raises(ValueError, match="start_date") as err:
            model.simulate_short_term(bad, horizon=2)
        assert "2025-06-10" in str(err.value), bad                  # an example of a valid date

    # the simulators and the sampler validate too when they are used directly
    sim = ShortTermSimulator(model.gmm_)
    with pytest.raises(ValueError, match="horizon"):
        sim.simulate("2025-06-10", horizon=0)
    with pytest.raises(ValueError, match="date"):
        sim.simulate_single_day("nope")
    with pytest.raises(ValueError, match="n_sessions"):
        model.gmm_.sample(0)
    with pytest.raises(ValueError, match="date"):
        model.gmm_.sample(5, date="nope")


def test_invalid_medium_term_arguments_are_value_errors_that_name_the_argument(model):
    for bad in (0, -1, 1.5):
        with pytest.raises(ValueError, match="n_scenarios"):
            model.simulate_medium_term(years=1, n_scenarios=bad)

    sim = MediumTermSimulator(model.gmm_, n_scenarios=1)
    for kwargs, name in (
        ({"years": 0}, "years"),
        ({"years": -1}, "years"),
        ({"years": float("nan")}, "years"),
        ({"annual_growth_rate": -1.0}, "annual_growth_rate"),
        ({"annual_growth_rate": float("inf")}, "annual_growth_rate"),
        ({"start_date": "tomorrow-ish"}, "start_date"),
        ({"output": "weekly"}, "output"),
    ):
        with pytest.raises(ValueError, match=name):
            sim.simulate(**{"years": 1, "start_date": "2025-01-01", **kwargs})


def test_every_entry_point_raises_not_fitted_error_with_the_way_forward(demo):
    unfitted = GEARSModel()
    calls = {
        "simulate_short_term": lambda: unfitted.simulate_short_term("2025-06-10"),
        "simulate_medium_term": lambda: unfitted.simulate_medium_term(years=1),
        "smart_charge": lambda: unfitted.smart_charge(demo.head(3), make_price_signal("2023-01-01", 48)),
        "EVSessionModel.sample": lambda: EVSessionModel().sample(5),
        "SessionForecaster.predict": lambda: SessionForecaster().predict(3),
        "PersistenceForecaster.predict": lambda: PersistenceForecaster().predict(3),
    }
    for call in calls.values():
        with pytest.raises(NotFittedError, match=r"\.fit\(|from_pretrained|load_pretrained"):
            call()
        # it is the scikit-learn error, hence also a ValueError: generic handlers keep working
        with pytest.raises(ValueError, match="not fitted"):
            call()


@pytest.fixture
def fresh_model(demo) -> GEARSModel:
    """A model stratified by (day_of_week, season); one instance per test, since sampling an
    unknown context records it on the model."""
    return GEARSModel(n_components=2, forecaster_method="probabilistic", random_state=0).fit(
        demo, verbose=False
    )


def test_an_unknown_context_warns_once_naming_the_requested_and_the_used_context(fresh_model):
    gmm = fresh_model.gmm_
    _, msgs = _warnings_of(gmm.sample, 5, context={"day_of_week": 2, "season": "summer"},
                           seed=0, strict=True)
    assert msgs == [] and not gmm.fallback_contexts_               # a fitted context is silent

    out, first = _warnings_of(gmm.sample, 5, context={"day_of_week": 99, "season": "monsoon"}, seed=0)
    assert len(out) == 5                                           # still sessions, not an error
    assert len(first) == 1
    assert "'day_of_week': 99" in first[0] and "'season': 'monsoon'" in first[0]   # requested
    assert "Using the closest fitted context" in first[0]                          # used
    assert "strict=True" in first[0]                                               # what to do

    _, second = _warnings_of(gmm.sample, 5, context={"day_of_week": 98, "season": "x"}, seed=0)
    assert second == []                                            # once only ...
    assert len(gmm.fallback_contexts_) == 2                        # ... but both recorded
    assert "Fallbacks" in fresh_model.summary()


def test_strict_refuses_to_fall_back_and_says_how_to_allow_it(fresh_model, demo):
    gmm = fresh_model.gmm_
    with pytest.raises(ValueError, match="No mixture model was fitted for context") as err:
        gmm.sample(5, context={"day_of_week": 99, "season": "monsoon"}, strict=True)
    assert "strict=False" in str(err.value)
    assert not gmm.fallback_contexts_                              # nothing substituted silently

    with pytest.raises(ValueError, match="context must be a dict"):
        gmm.sample(5, context=("monday", "summer"))

    # a date alone cannot pick a location type: a warning, or an error under strict
    both = pd.concat([demo, make_demo_data(n=900, location_type="home", seed=4)], ignore_index=True)
    by_site = GEARSModel(
        n_components=2, forecaster_method="probabilistic", stratify_by=["location_type", "day_of_week"]
    ).fit(both, verbose=False)
    _, msgs = _warnings_of(by_site.gmm_.sample, 5, date="2025-06-10")
    assert len(msgs) == 1 and "location_type not given" in msgs[0]
    with pytest.raises(ValueError, match="location_type not given"):
        by_site.gmm_.sample(5, date="2025-06-10", strict=True)
    # strict goes through the simulators, one scenario per day or a fixed number of sessions
    with pytest.raises(ValueError, match="location_type not given"):
        by_site.simulate_short_term("2025-06-10", horizon=3, n_scenarios=2, seed=0, strict=True)
    with pytest.raises(ValueError, match="location_type not given"):
        by_site.simulate_short_term("2025-06-10", horizon=1, n_sessions=3, seed=0, strict=True)


def test_an_unknown_forecaster_method_lists_the_valid_values_at_construction():
    with pytest.raises(ValueError) as err:
        GEARSModel(forecaster_method="prophet")      # fails at once, not after a long fit
    msg = str(err.value)
    assert "prophet" in msg and "sarima" in msg and "probabilistic" in msg
    with pytest.raises(ValueError, match="sarima"):
        SessionForecaster(method="prophet")


# ── SARIMA: short series, pmdarima absent, forecasts that cannot be trusted ──

def _short_history(n_days: int) -> pd.DataFrame:
    end = (pd.Timestamp("2023-01-01") + pd.Timedelta(days=n_days - 1)).strftime("%Y-%m-%d")
    return make_demo_data(n=max(60, 6 * n_days), seed=5, start_date="2023-01-01", end_date=end)


def test_sarima_is_always_tried_and_a_short_history_never_breaks_the_fit():
    """Whatever the history, fit() works; if SARIMA could not be fitted it says so, and
    ``method_`` tells that 'probabilistic' is what is used."""
    for n_days in (1, 3, 7, 10, 14, 20, 27, 120):
        fc = SessionForecaster(method="sarima", use_holidays=False)
        _, msgs = _warnings_of(fc.fit, _short_history(n_days))
        switched = [m for m in msgs if "SARIMA could not be fitted" in m]
        assert fc.is_fitted_ and fc.method == "sarima", n_days
        if fc.method_ == "probabilistic":
            assert len(switched) == 1, n_days
            assert f"{n_days} day(s)" in switched[0], n_days
            assert "forecaster_method='probabilistic'" in switched[0], n_days
        else:
            assert switched == [], n_days                          # it worked: no switch announced
        pred = fc.predict(horizon=7, n_scenarios=2, seed=0)
        assert len(pred) == 14 and np.isfinite(pred["n_sessions"]).all(), n_days
        if n_days == 120:                                          # long enough: kept, and silent
            assert fc.method_ == "sarima" and msgs == [], msgs


def test_a_sarima_that_fails_or_cannot_be_trusted_switches_to_probabilistic_and_says_why(
    monkeypatch,
):
    import pmdarima

    # 1. pmdarima refuses a short series ("no more samples after seasonal differencing")
    def refuse(*args, **kwargs):
        raise ValueError(
            "There are no more samples after a first-order seasonal differencing. See http://x"
        )

    with monkeypatch.context() as patch:
        patch.setattr(pmdarima, "auto_arima", refuse)
        fc = SessionForecaster(method="sarima", use_holidays=False)
        _, msgs = _warnings_of(fc.fit, _short_history(10))
    assert len(msgs) == 1
    assert "ValueError: There are no more samples after a first-order seasonal differencing" in msgs[0]
    assert "http://x" not in msgs[0]                       # the pmdarima URL is cut off
    assert "Switching to method='probabilistic'" in msgs[0]
    assert fc.method == "sarima" and fc.method_ == "probabilistic"
    assert "used='probabilistic'" in repr(fc)
    assert len(fc.predict(horizon=5, n_scenarios=2, seed=0)) == 10

    # 2. a fit that "succeeds" but forecasts nonsense has failed too
    class Forecasts:
        _is_statsmodels = True

        def __init__(self, values):
            self.values = values

        def forecast(self, horizon, exog=None):
            return np.resize(self.values, horizon)

    for values, expected in (
        ([11.0, 0.0, 24.0, 0.0, 65.0, 0.0, 264.0], ("forecast diverges", "264")),   # ~10 a day
        ([np.nan], ()),
    ):
        with monkeypatch.context() as patch:
            patch.setitem(sys.modules, "pmdarima", None)
            patch.setattr(
                SessionForecaster, "_fit_statsmodels_fallback",
                lambda self, exog=None, v=values: setattr(self, "_model", Forecasts(v)),
            )
            fc = SessionForecaster(method="sarima", use_holidays=False)
            _, msgs = _warnings_of(fc.fit, _short_history(14))
        failed = [m for m in msgs if "SARIMA could not be fitted" in m]
        assert len(failed) == 1 and fc.method_ == "probabilistic" and fc._model is None, values
        for part in expected:
            assert part in failed[0], (values, part)


def test_a_missing_pmdarima_and_a_failing_sarima_prediction_are_announced_once(demo, monkeypatch):
    dates = pd.to_datetime(demo["date"])
    history = demo[dates < dates.min() + pd.Timedelta(days=90)].reset_index(drop=True)

    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "pmdarima", None)         # makes `import pmdarima` fail
        fc = SessionForecaster(method="sarima", use_holidays=False)
        _, msgs = _warnings_of(fc.fit, history)
    absent = [m for m in msgs if "pmdarima could not be imported" in m]
    assert len(absent) == 1
    assert "SARIMAX(1,1,1)x(1,1,0,7)" in absent[0] and "pip install pmdarima" in absent[0]
    assert fc.is_fitted_ and fc.method_ == "sarima"
    pred = fc.predict(horizon=7, n_scenarios=2, seed=0)
    assert len(pred) == 14 and (pred["n_sessions"] >= 0).all()

    class Cannot:
        def predict(self, **_):
            raise RuntimeError("singular matrix")

    fc._model = Cannot()                                      # fitted, but it cannot predict
    _, first = _warnings_of(fc.predict, horizon=5, n_scenarios=2, seed=0)
    assert len(first) == 1 and "singular matrix" in first[0] and "historical mean" in first[0]
    _, second = _warnings_of(fc.predict, horizon=5, n_scenarios=2, seed=0)
    assert second == []


def test_forecasters_cope_with_a_single_day_a_short_history_and_an_empty_dataset(demo):
    one_day = make_demo_data(n=50, seed=5, start_date="2023-01-02", end_date="2023-01-02")
    fc = SessionForecaster(method="sarima")
    _warnings_of(fc.fit, one_day)
    assert np.isfinite(fc.std_daily_)                  # one observation has no std: 0, not NaN
    assert np.isfinite(fc.predict(horizon=3, seed=0)["n_sessions"]).all()

    short = make_demo_data(n=100, seed=5, start_date="2023-01-01", end_date="2023-01-10")
    _, msgs = _warnings_of(SessionForecaster(method="probabilistic").fit, short)
    assert msgs == []                                  # nothing to announce for 'probabilistic'

    with pytest.raises(ValueError, match="empty DataFrame"):
        SessionForecaster().fit(demo.iloc[:0])


# =============================================================================
# Part 2 — the output invariants
# =============================================================================

START = pd.Timestamp("2025-06-10")
HORIZON = 7
RES_MIN = 30


@pytest.fixture(scope="module")
def sessions(model) -> pd.DataFrame:
    return model.simulate_short_term(START, horizon=HORIZON, n_scenarios=3, seed=11)


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


def test_inv2_values_are_physical_and_complete(sessions):
    assert (sessions["energy"] >= 0).all()
    assert (sessions["duration"] > 0).all()
    assert (sessions["power_kw"] > 0).all()
    assert not sessions.isna().any().any()
    assert sessions["arrival_hour"].between(0, 24, inclusive="left").all()


def test_inv3_energy_never_exceeds_what_the_charger_can_deliver(sessions):
    cap = sessions["power_kw"] * sessions["duration"]
    assert (sessions["energy"] <= cap * 1.001).all()


def test_inv4_arrivals_fall_inside_the_requested_window(sessions):
    assert (sessions["arrival_time"] >= START).all()
    assert (sessions["arrival_time"] < START + pd.Timedelta(days=HORIZON + 1)).all()
    assert (sessions["arrival_time"].dt.date == sessions["date"]).all()   # on its own day


def test_inv5_same_seed_same_result_different_seed_different_result(model):
    kw = {"horizon": 4, "n_scenarios": 2}
    a = model.simulate_short_term(START, seed=5, **kw)
    pd.testing.assert_frame_equal(a, model.simulate_short_term(START, seed=5, **kw))
    c = model.simulate_short_term(START, seed=6, **kw)
    assert not (len(a) == len(c) and np.allclose(a["energy"], c["energy"]))

    kw = {"horizon": 4, "n_sessions": 6}          # also with a fixed number of sessions
    a = model.simulate_short_term(START, seed=5, **kw)
    pd.testing.assert_frame_equal(a, model.simulate_short_term(START, seed=5, **kw))
    assert not np.allclose(a["energy"], model.simulate_short_term(START, seed=6, **kw)["energy"])

    # day 7 of seed 0 and day 0 of seed 7 share weekday and season, hence the context: with
    # per-day seeds computed as ``seed + day`` they were identical
    def day(df, i):
        return df[df["date"] == (START + pd.Timedelta(days=i)).date()]["energy"].to_numpy()

    s0 = model.simulate_short_term(START, horizon=8, n_sessions=8, seed=0)
    s7 = model.simulate_short_term(START, horizon=1, n_sessions=8, seed=7)
    assert (START + pd.Timedelta(days=7)).day_of_week == START.day_of_week
    assert not np.allclose(day(s0, 7), day(s7, 0))

    # no seed: fresh randomness
    kw = {"horizon": 2, "n_sessions": 6}
    assert not np.allclose(
        model.simulate_short_term(START, **kw)["energy"],
        model.simulate_short_term(START, **kw)["energy"],
    )


def test_inv6_scenarios_are_numbered_from_zero_without_gaps(sessions):
    assert sorted(sessions["scenario"].unique()) == [0, 1, 2]


def test_inv7_simulated_means_match_the_training_data(model, demo):
    sim = model.simulate_short_term(START, horizon=7, n_sessions=300, seed=0)
    assert len(sim) == 7 * 300
    for col in ("energy", "duration"):
        ratio = sim[col].mean() / demo[col].mean()
        assert 0.85 <= ratio <= 1.15, f"{col}: simulated/training mean = {ratio:.3f}"


def test_inv8_save_then_load_reproduces_the_simulation(model, tmp_path):
    path = tmp_path / "model.joblib"
    model.save(path)
    loaded = GEARSModel.load(path)
    pd.testing.assert_frame_equal(
        model.simulate_short_term(START, horizon=3, n_scenarios=2, seed=21),
        loaded.simulate_short_term(START, horizon=3, n_scenarios=2, seed=21),
    )
    pd.testing.assert_frame_equal(
        model.gmm_.sample(20, date=START, seed=3), loaded.gmm_.sample(20, date=START, seed=3)
    )


# ── 9. V1G optimiser ─────────────────────────────────────────────────────────

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


def test_inv9_every_session_is_scheduled_inside_its_window_within_the_power_limit(optimised):
    """GEARS caps energy at power x duration: every simulated session must be completable."""
    sess, _, result, msgs = optimised
    assert msgs == []
    assert len(result) == len(sess)
    columns = ["cost_smart", "cost_plug", "scheduled_start", "scheduled_end", "energy_scheduled"]
    assert {*columns, "savings_pct"} <= set(result.columns)
    assert not result[columns].isna().any().any()

    assert np.allclose(result["energy_scheduled"], result["energy"], rtol=1e-9)   # all of it
    departure = result["arrival_time"] + pd.to_timedelta(result["duration"], unit="h")
    assert (result["scheduled_start"] >= result["arrival_time"]).all()
    assert (result["scheduled_end"] <= departure + pd.Timedelta(milliseconds=1)).all()
    # delivering E kWh at no more than P kW takes at least E / P hours
    span_h = (result["scheduled_end"] - result["scheduled_start"]) / pd.Timedelta(hours=1)
    assert (span_h >= result["energy"] / result["power_kw"] - 1e-9).all()


def test_inv9_costs_are_recomputed_independently_and_smart_never_costs_more(optimised, model):
    _, signal, result, _ = optimised

    # fill the slots by hand (plain Python): chronologically, and best price first
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

    assert (result["cost_smart"] <= result["cost_plug"] + 1e-12).all()
    assert (result["savings_pct"].dropna() >= -1e-9).all()
    assert result["cost_smart"].sum() < result["cost_plug"].sum()   # and it saves something

    # for a renewable-share signal the 'cost' columns hold green fraction x energy: more is better
    sess = model.simulate_short_term(START, horizon=2, n_scenarios=1, n_sessions=15, seed=4)
    res = SmartChargingOptimizer(signal_type="res").optimise(
        sess, make_res_signal(start=START, periods=48 * 4, resolution_min=RES_MIN)
    )
    assert (res["cost_smart"] >= res["cost_plug"] - 1e-12).all()
    # ... and what it gains is reported as a gain, not as a negative saving in euros
    assert (res["savings_pct"].dropna() >= -1e-9).all() and res["savings_pct"].max() > 0
    summary = SmartChargingOptimizer(signal_type="res").savings_summary(res)
    assert set(summary) == {
        "green_kwh_smart", "green_kwh_plug", "green_kwh_gain", "mean_gain_pct",
        "n_sessions_optimised",
    }
    assert summary["green_kwh_gain"] > 0 and summary["mean_gain_pct"] > 0
    assert summary["green_kwh_gain"] == pytest.approx(
        summary["green_kwh_smart"] - summary["green_kwh_plug"], abs=0.02
    )
    assert all(type(v) in (float, int) for v in summary.values())      # not numpy scalars


def test_inv9_hand_computed_schedules_including_partial_slots():
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

    # 08:10 -> 09:10: only the time inside the window is on offer in the edge slots
    # (3.33 kWh in the 08:00 slot, 5 in 08:30, 1.67 in 09:00)
    edge = opt.optimise(_one_session("2025-06-10 08:10", 1.0, 10.0), _signal([3.0, 1.0, 2.0])).iloc[0]
    assert edge["energy_scheduled"] == pytest.approx(10.0)
    assert edge["cost_plug"] == pytest.approx(10 / 3 * 3 + 5 * 1 + 10 / 6 * 2)
    assert edge["cost_smart"] == pytest.approx(edge["cost_plug"])           # the window is full
    assert edge["scheduled_start"] == pd.Timestamp("2025-06-10 08:10")
    assert edge["scheduled_end"] == pd.Timestamp("2025-06-10 09:10")

    # energy == power x duration (the simulators' cap), whatever duration and grid alignment
    rng = np.random.default_rng(0)
    n = 200
    duration, power = rng.uniform(0.3, 12.0, n), rng.choice([3.7, 7.4, 22.0], n)
    capped = pd.DataFrame({
        "arrival_time": START + pd.to_timedelta(rng.uniform(0, 12, n), unit="h"),
        "duration": duration, "energy": power * duration, "power_kw": power,
    })
    signal = make_price_signal(start=START, periods=48 * 3, resolution_min=RES_MIN)
    result, msgs = _warnings_of(SmartChargingOptimizer("price", RES_MIN).optimise, capped, signal)
    assert msgs == []
    assert np.allclose(result["energy_scheduled"], result["energy"], rtol=1e-9)
    assert np.allclose(result["cost_smart"], result["cost_plug"])           # no room to shift


def test_inv9_what_cannot_be_scheduled_is_announced_and_bad_inputs_are_refused(model):
    prices = _signal([1.0, 2.0, 3.0, 4.0])

    # 25 kWh in 1 h at 10 kW cannot be completed: announced, and what fits is scheduled
    result, msgs = _warnings_of(
        SmartChargingOptimizer("price", 30).optimise,
        _one_session("2025-06-10 08:00", 1.0, 25.0), _signal([1.0, 2.0, 3.0]),
    )
    assert len(msgs) == 1 and "1 of 1 sessions cannot be completed" in msgs[0]
    assert "energy_scheduled" in msgs[0]
    assert result.iloc[0]["energy_scheduled"] == pytest.approx(10.0)

    # efficiency below 1 adds the losses on top of the energy
    lossy = SmartChargingOptimizer("price", 30, efficiency=0.9)
    result, msgs = _warnings_of(lossy.optimise, _one_session("2025-06-10 08:00", 2.0, 18.0), _signal([1.0] * 4))
    assert msgs == []
    assert result.iloc[0]["cost_plug"] == pytest.approx(18.0 / 0.9)          # the charger draws more
    assert result.iloc[0]["energy_scheduled"] == pytest.approx(18.0)
    _, msgs = _warnings_of(lossy.optimise, _one_session("2025-06-10 08:00", 2.0, 20.0), _signal([1.0] * 4))
    assert any("energy / efficiency (0.9)" in m for m in msgs)               # 22.2 kWh > 20
    assert SmartChargingOptimizer().efficiency == 1.0                        # no losses by default

    # sessions outside the signal are announced, not silently NaN
    sess = model.simulate_short_term(START, horizon=1, n_scenarios=1, n_sessions=5, seed=1)
    late = make_price_signal(start=START + pd.Timedelta(days=30), periods=96)
    result, msgs = _warnings_of(SmartChargingOptimizer("price").optimise, sess, late)
    assert any("5 of 5 sessions were not optimised" in m and "covers" in m for m in msgs), msgs
    assert result["cost_smart"].isna().all()

    # unusable sessions are counted, and the others are still scheduled
    good = _one_session("2025-06-10 08:00", 2.0, 10.0)
    bad = pd.concat([good, good, good], ignore_index=True)
    bad.loc[1, "energy"] = np.nan
    bad.loc[2, "duration"] = 0.0
    result, msgs = _warnings_of(SmartChargingOptimizer("price", 30).optimise, bad, prices)
    assert any("2 of 3 sessions were not optimised" in m and "no usable arrival" in m for m in msgs)
    assert result["cost_smart"].notna().tolist() == [True, False, False]

    # a signal whose step differs from resolution_min
    _, msgs = _warnings_of(SmartChargingOptimizer("price", 15).optimise, good, prices)
    assert any("30-minute step" in m and "resolution_min=15" in m for m in msgs)

    # refusals
    with pytest.raises(TypeError, match="indexed by timestamps"):
        SmartChargingOptimizer("price", 30).optimise(good, pd.Series([1.0, 2.0]))
    with pytest.raises(ValueError, match="power_kw must be > 0"):
        SmartChargingOptimizer("price", 30).optimise(good.assign(power_kw=0.0), _signal([1.0, 2.0]))
    for kwargs, name in (
        ({"efficiency": 0}, "efficiency"),
        ({"efficiency": 1.2}, "efficiency"),
        ({"resolution_min": 0}, "resolution_min"),
        ({"default_power_kw": -7.4}, "default_power_kw"),
        ({"signal_type": "carbon"}, "signal_type"),
    ):
        with pytest.raises(ValueError, match=name):
            SmartChargingOptimizer(**kwargs)


# ── 10. Medium term ──────────────────────────────────────────────────────────

GROWTH = {
    "linear": lambda base, rate: linear_growth_profile(base, 5, annual_growth_rate=rate),
    "s_curve": lambda base, rate: s_curve_growth_profile(base, 5, saturation_factor=1.0 + 3 * rate),
    "bass": lambda base, rate: bass_diffusion_profile(base, 5, market_potential_factor=1.0 + 3 * rate),
}


def test_inv10_growth_profiles_are_flat_at_zero_monotone_above_it_and_never_negative():
    for name, profile in GROWTH.items():
        assert np.allclose(profile(80.0, 0.0).to_numpy(), 80.0), name      # growth 0: the base
        values = profile(80.0, 0.5).to_numpy()
        assert values[0] == pytest.approx(80.0), name                       # growth > 0: from the base
        assert (np.diff(values) >= -1e-12).all() and values[-1] > values[0], name
        assert (values >= 0).all() and profile(80.0, 0.5).index.is_monotonic_increasing, name

    # a shrinking fleet is clamped at zero, not negative (it would hit < 0 at year 2)
    declining = linear_growth_profile(50.0, 10, annual_growth_rate=-0.5)
    assert (declining.to_numpy() >= 0).all() and declining.iloc[-1] == 0.0


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

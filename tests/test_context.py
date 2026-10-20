"""Choosing the context: which site, which département, in the facade, the simulators and the CLI.

A model fitted per site (the French one is) holds a mixture for each (site type, département,
weekday, season). The simulators take a ``context`` for the part a date cannot give; each day
completes it with its own weekday and season. The fixture is a two-site model whose sites look
nothing alike: workplaces in "75" (arrivals in the morning, 4 sessions a day) and homes in "92"
(arrivals in the evening, 2 sessions a day).
"""

from __future__ import annotations

import warnings

import pandas as pd
import pytest
from click.testing import CliRunner

from gears.cli import main
from gears.data.loader import make_demo_data
from gears.pipeline import GEARSModel
from gears.simulation.medium_term import MediumTermSimulator
from gears.simulation.short_term import ShortTermSimulator

KEYS = ["location_type", "department", "day_of_week", "season"]
WORK = {"location_type": "work", "department": "75"}
HOME = {"location_type": "home", "department": "92"}


@pytest.fixture(scope="module")
def two_sites() -> GEARSModel:
    work = make_demo_data(n=1500, location_type="work", seed=0).assign(department="75")
    home = make_demo_data(n=700, location_type="home", seed=1).assign(department="92")
    return GEARSModel(
        n_components=2, forecaster_method="probabilistic", random_state=0, stratify_by=KEYS
    ).fit(pd.concat([work, home], ignore_index=True), verbose=False)


def test_the_context_picks_the_site_and_each_day_completes_it(two_sites):
    week = {}
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)   # strict=True and full coverage: nothing to warn
        for name, site in (("work", WORK), ("home", HOME)):
            week[name] = two_sites.simulate_short_term(
                "2024-06-03", horizon=7, n_sessions=300, seed=0, context=site, strict=True
            )
    assert week["work"]["date"].nunique() == week["home"]["date"].nunique() == 7
    assert len(week["work"]) == len(week["home"]) == 7 * 300
    assert week["work"]["arrival_hour"].median() < 10      # mornings at work
    assert week["home"]["arrival_hour"].median() > 16      # evenings at home
    assert not two_sites.gmm_.fallback_contexts_

    # "completes it with the day's weekday and season" means exactly this: the partial context
    # for a Monday in June is the full context of a summer Monday
    gmm = two_sites.gmm_
    sim = ShortTermSimulator(gmm)
    partial = sim.simulate_single_day("2024-06-03", n_sessions=50, seed=1, context=HOME)
    full = sim.simulate_single_day(
        "2024-06-03", n_sessions=50, seed=1, context={**HOME, "day_of_week": 0, "season": "summer"}
    )
    pd.testing.assert_frame_equal(partial, full)
    saturday = sim.simulate_single_day("2024-06-08", n_sessions=50, seed=1, context=HOME)
    assert not saturday["energy"].equals(partial["energy"])      # another day, another mixture


def test_what_a_context_gives_wins_over_what_the_date_gives(two_sites):
    gmm = two_sites.gmm_
    monday = pd.Timestamp("2024-06-03")                          # a Monday, in summer
    resolve = gmm._resolve_context
    assert resolve(HOME, monday) == ("home", "92", 0, "summer")           # the date completes
    assert resolve({**HOME, "day_of_week": 5}, monday) == ("home", "92", 5, "summer")   # a key wins
    assert resolve({**HOME, "season": "winter"}, monday) == ("home", "92", 0, "winter")
    assert resolve(HOME, None) == ("home", "92", None, None)              # no date: "not given"
    assert resolve(None, monday) == (None, None, 0, "summer")             # no context: ditto

    # "is_weekend" is read from the date as well, for a model stratified by it
    weekend = GEARSModel(
        n_components=2, forecaster_method="probabilistic", stratify_by=["is_weekend", "season"]
    )
    weekend.fit(make_demo_data(n=600, seed=3), verbose=False)
    assert weekend.gmm_._resolve_context({}, pd.Timestamp("2024-06-08")) == (1, "summer")
    assert weekend.gmm_._resolve_context({"season": "winter"}, monday) == (0, "winter")


def test_a_key_the_model_is_not_stratified_by_is_refused_not_ignored(two_sites, model):
    typo = {"location_type": "home", "departement": "92"}
    with pytest.raises(ValueError, match=r"\['departement'\] that this model is not stratified by") as err:
        two_sites.simulate_short_term("2024-06-03", n_sessions=5, context=typo)
    assert "location_type" in str(err.value)                   # it lists the keys that exist
    with pytest.raises(ValueError, match="not stratified by"):
        two_sites.simulate_short_term("2024-06-03", n_scenarios=1, context=typo)
    with pytest.raises(ValueError, match="not stratified by"):
        two_sites.simulate_medium_term(years=1, context=typo)
    with pytest.raises(ValueError, match="not stratified by"):
        two_sites.gmm_.sample(5, context=typo)

    # the demo model is stratified by weekday and season only: a site means nothing to it
    with pytest.raises(ValueError, match=r"\['location_type'\] that this model is not stratified by"):
        model.simulate_short_term("2024-06-03", n_sessions=5, context={"location_type": "work"})


def test_a_context_that_was_not_fitted_warns_or_raises_as_before(two_sites_file):
    fresh = GEARSModel.load(two_sites_file)          # the warning is shown once per model
    nowhere = {"location_type": "work", "department": "00"}
    with pytest.raises(ValueError, match="No mixture model was fitted for context"):
        fresh.simulate_short_term("2024-06-03", n_sessions=5, context=nowhere, strict=True)
    assert not fresh.gmm_.fallback_contexts_
    with pytest.warns(UserWarning, match="Using the closest fitted context"):
        out = fresh.simulate_short_term("2024-06-03", horizon=1, n_sessions=5, context=nowhere)
    assert len(out) == 5 and fresh.gmm_.fallback_contexts_


def test_the_forecaster_path_takes_the_context_too(two_sites):
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        home = two_sites.simulate_short_term(
            "2024-06-03", horizon=3, n_scenarios=2, seed=0, context=HOME, strict=True
        )
    assert set(home["scenario"]) == {0, 1} and home["arrival_hour"].median() > 16


def test_the_medium_term_site_sets_the_shape_and_the_starting_level(two_sites):
    def daily(site):
        return two_sites.simulate_medium_term(
            years=0.5, annual_growth_rate=0.0, growth_model="linear", start_date="2024-01-01",
            n_scenarios=1, context=site,
        )

    work, home = daily(WORK), daily(HOME)
    gmm = two_sites.gmm_
    for site, result in ((WORK, work), (HOME, home)):
        matching = [n for k, n in gmm.n_sessions_per_day_.items() if k[0] == site["location_type"]]
        expected = sum(matching) / len(matching)                # the mean daily count of that site
        assert result["n_sessions"].sum() / (0.5 * 365) == pytest.approx(expected, rel=0.2)
    assert work["n_sessions"].sum() > 1.4 * home["n_sessions"].sum()     # 4.2 against 2.2 a day

    # the sessions themselves come from the chosen site: mornings at work, evenings at home
    hours = {
        name: two_sites.simulate_medium_term(
            years=0.1, annual_growth_rate=0.0, growth_model="linear", start_date="2024-01-01",
            n_scenarios=1, output="sessions", context=site,
        )["arrival_hour"].median()
        for name, site in (("work", WORK), ("home", HOME))
    }
    assert hours["work"] < 10 and hours["home"] > 16

    # a starting level given by the user is never replaced by the site's
    given = MediumTermSimulator(gmm, base_sessions_per_day=10.0, growth_model="linear", n_scenarios=1)
    assert given._base_for(HOME) == 10.0

    with pytest.raises(ValueError, match="No fitted context matches"):
        two_sites.simulate_medium_term(
            years=1, context={"location_type": "work", "department": "00"}
        )


def test_a_model_without_forecaster_says_so_unless_the_count_is_given(two_sites):
    by_hand = GEARSModel.from_session_model(two_sites.gmm_)      # what from_pretrained builds
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)             # n_sessions: nothing to warn about
        by_hand.simulate_short_term("2024-06-03", n_sessions=10, context=HOME, seed=0, strict=True)

    with pytest.warns(UserWarning, match="No fitted forecaster.*20 sessions.*n_sessions"):
        out = by_hand.simulate_short_term("2024-06-03", horizon=2, n_scenarios=1, context=HOME)
    assert len(out) == 2 * 20


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def two_sites_file(two_sites, tmp_path_factory):
    path = tmp_path_factory.mktemp("two_sites") / "two_sites.joblib"
    two_sites.save(path)
    return path


def simulate(model_file, *args):
    return CliRunner().invoke(
        main, ["simulate", "--model", str(model_file), "--start", "2024-06-03", "--horizon", "2",
               "--n-sessions", "40", "--seed", "0", *args]
    )


def test_the_command_line_chooses_the_site_with_repeated_context_options(two_sites_file, tmp_path):
    arrivals = {}
    for name, flags in (
        ("work", ["-c", "location_type=work", "--context", "department=75"]),
        ("home", ["-c", "location_type=home", "-c", "department=92", "--strict"]),
    ):
        out = tmp_path / f"{name}.csv"
        result = simulate(two_sites_file, *flags, "--output", str(out))
        assert result.exit_code == 0, result.output
        sessions = pd.read_csv(out)
        assert len(sessions) == 2 * 40
        arrivals[name] = sessions["arrival_hour"].median()
    assert arrivals["work"] < 10 and arrivals["home"] > 16

    # a whole number where the model holds whole numbers: day_of_week=0 is an int, not "0"
    result = simulate(two_sites_file, "-c", "location_type=home", "-c", "department=92",
                      "-c", "day_of_week=5", "--strict", "--output", str(tmp_path / "sat.csv"))
    assert result.exit_code == 0, result.output


def test_the_command_line_refuses_a_bad_context_in_one_line(two_sites_file, tmp_path):
    out = ["--output", str(tmp_path / "never.csv")]
    for flags, message in (
        (["-c", "location_type"], "expected KEY=VALUE"),
        (["-c", "departement=92"], "'departement' is not a context key of this model"),
        (["-c", "department=92", "-c", "department=75"], "'department' is given twice"),
        (["-c", "day_of_week=monday"], "day_of_week is a whole number"),
    ):
        result = simulate(two_sites_file, *flags, *out)
        assert result.exit_code == 2 and message in result.output, (flags, result.output)

    result = simulate(two_sites_file, "-c", "department=00", "--strict", *out)
    assert result.exit_code == 1 and "No mixture model was fitted for context" in result.output
    assert "Traceback" not in result.output and not (tmp_path / "never.csv").exists()

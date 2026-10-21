"""Bring your own data: file -> load_sessions -> fit -> save -> load -> simulate.

    python examples/04_bring_your_own_data.py                 # writes a small CSV, then uses it
    python examples/04_bring_your_own_data.py sessions.csv    # your own file (csv, parquet, xlsx...)
    python examples/04_bring_your_own_data.py --pretrained    # + the official French model

Your file needs one row per charging session: the arrival time, the duration in hours and the
energy in kWh (``data/README.md`` lists the optional columns). Common column names are
recognised. ``--pretrained`` downloads the French model once (about 21 MB, checked against a
SHA-256 and cached in ``~/.cache/gears``): it needs an internet connection, the rest does not.
Files are written to ``$GEARS_EXAMPLES_OUT`` (default: ``./example_outputs``).
"""

import argparse
import os
from pathlib import Path

import pandas as pd

from gears import GEARSModel, load_sessions, make_demo_data

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("csv", nargs="?", help="your sessions file (default: a demo file is written)")
parser.add_argument("--pretrained", action="store_true", help="also use the official French model")
args = parser.parse_args()

OUT = Path(os.environ.get("GEARS_EXAMPLES_OUT", "example_outputs"))
OUT.mkdir(parents=True, exist_ok=True)

# 1. A file of sessions. Without an argument, write one whose columns do not carry the names
#    GEARS uses internally: start_time, duration_h, energy_kwh and site_type are recognised.
if args.csv:
    path = Path(args.csv)
else:
    path = OUT / "my_sessions.csv"
    demo = make_demo_data(n=2000, location_type="work", seed=1)
    demo = demo.rename(columns={"arrival_time": "start_time", "duration": "duration_h",
                                "energy": "energy_kwh", "location_type": "site_type"})
    demo["start_time"] = demo["start_time"].dt.floor("s")
    demo = demo.round({"duration_h": 2, "energy_kwh": 2})
    demo[["start_time", "duration_h", "energy_kwh", "site_type"]].to_csv(path, index=False)
print(f"sessions file: {path}")

# 2. Load and check it: columns renamed, calendar columns added, impossible rows dropped
#    (with one warning that counts them by reason; strict=True would raise instead).
sessions = load_sessions(path, verbose=False)
print(f"{len(sessions)} sessions from {sessions['arrival_time'].min():%Y-%m-%d} "
      f"to {sessions['arrival_time'].max():%Y-%m-%d}; columns: {list(sessions.columns)}")

# 3. Fit, then save the model to a file you can keep or share. Open a model file only if you trust
#    its source: it is a pickle.
model = GEARSModel(n_components=3, forecaster_method="probabilistic", n_scenarios=5, random_state=0)
model.fit(sessions, verbose=False)
model.save(OUT / "my_model.joblib")
reloaded = GEARSModel.load(OUT / "my_model.joblib")

# 4. Simulate the week after your last session, with the model read back from the file.
next_day = sessions["arrival_time"].max().normalize() + pd.Timedelta(days=1)
simulated = reloaded.simulate_short_term(next_day, horizon=7, seed=0)
reloaded.export(simulated, OUT / "my_simulated_sessions.csv")
print(f"{len(simulated)} simulated sessions from {next_day:%Y-%m-%d}, "
      f"saved to {OUT / 'my_simulated_sessions.csv'}")

# 5. Optional: the official model of French charging sessions, no data of your own needed. It is
#    fitted per site type and per département: `context` says which one (the weekday and the season
#    come from the dates), and `n_sessions` gives the sessions per day, since it holds no forecaster
#    of daily counts. (A warning about the scikit-learn version it was trained with appears if yours
#    differs.)
if args.pretrained:
    french = GEARSModel.from_pretrained("french")  # downloads on first use, then cached
    paris_workplaces = {"location_type": "work", "department": "75"}
    week = french.simulate_short_term(
        "2025-06-02", horizon=7, n_sessions=40, context=paris_workplaces, seed=0, strict=True
    )
    print(f"French model, workplaces of Paris: {len(week)} sessions over {week['date'].nunique()} "
          f"days from 2025-06-02, median energy {week['energy'].median():.1f} kWh")

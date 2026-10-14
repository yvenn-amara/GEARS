# `data/` — put your session files here

Drop your charging-session files (CSV, Parquet, Excel, JSON or pickle) in this
folder. Git ignores everything in `data/` except this README (see the root
`.gitignore`), so your data is never committed by accident.

## Expected format

One row per charging session, with the columns below
(`gears.data.schemas` validates them):

| Column | Required? | Meaning |
|---|---|---|
| `arrival_time` | **required** | date and time of plug-in |
| `duration` | **required** | session duration in hours (> 0) |
| `energy` | **required** | energy delivered in kWh (≥ 0) |
| `power` | optional | charger power in kW |
| `location_type` | optional | one of `work`, `home`, `public`, `heavy` |
| `user_id` | optional | anonymised user identifier |
| `department` | optional | INSEE department code (French data only) |

Your column names need not match exactly: `load_sessions()` recognises common
aliases (`start_time`, `kwh`, `energy_kwh`, …). The authoritative list is
`COLUMN_ALIASES` and `REQUIRED_COLS` in
[`gears/data/schemas.py`](../gears/data/schemas.py). The French national IRVE
export is detected automatically and converted (Wh → kWh, minutes → hours).

## Conventions

- **Time.** `arrival_time` is **local wall-clock time** at the charging site, and simulated
  `arrival_time` values are naive timestamps in that same local time. A time zone or UTC
  offset is dropped and the clock reading kept (`08:00+02:00` becomes `08:00`), **not
  converted**, and a warning says so. If your timestamps are in UTC, convert them first
  (`df["arrival_time"].dt.tz_convert("Europe/Paris")`).
- **Units.** Hours and kWh. A median energy above 200 or a median duration above 48 triggers a
  warning ("probably Wh", "probably minutes").
- **Filters.** Rows with `duration <= 0`, a negative or missing `energy`, `duration >= 168` h,
  `energy >= 400` kWh, or an exact duplicate are dropped and counted by reason in one warning
  (`strict=True` raises instead).
- **Size.** At least 10 sessions overall; a stratification context (e.g. Monday x summer) with
  fewer than 10 sessions gets no model of its own and is served by the closest fitted one
  (one warning at `fit`, listed in `model.summary()`).

## Use it

```python
from gears import GEARSModel, load_sessions

sessions = load_sessions("data/my_sessions.csv")   # pass sep=";" etc. to the reader if needed
model = GEARSModel().fit(sessions)
```

No data of your own? `gears.make_demo_data()` generates synthetic sessions, and
`GEARSModel.from_pretrained("french")` loads the pre-trained French model.

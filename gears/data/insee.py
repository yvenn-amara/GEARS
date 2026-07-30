"""
INSEE department-level aggregation and forecasting for GEARS.

Provides utilities to:
- Aggregate raw EV sessions by INSEE department code
- Build daily/monthly energy time series per department
- Forecast medium-term energy demand per department using SARIMA
- Aggregate forecasts across departments or regions

This module is the primary entry point for the medium-term notebook.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def aggregate_by_department(
    df: pd.DataFrame,
    freq: str = "D",
    metric: str = "energy_kwh",
) -> pd.DataFrame:
    """
    Aggregate sessions by department and time frequency.

    Parameters
    ----------
    df : pd.DataFrame
        Validated sessions DataFrame (output of ``load_sessions``).
        Must have columns: ``arrival_time``, ``energy``, ``department``.
    freq : str
        Pandas offset alias: ``'D'`` (daily), ``'W'`` (weekly),
        ``'ME'`` (month-end).
    metric : str
        Which metric to aggregate:
        ``'energy_kwh'`` | ``'n_sessions'`` | ``'mean_energy_kwh'`` |
        ``'mean_duration_h'``.

    Returns
    -------
    pd.DataFrame
        Long-format table with columns: ``date``, ``department``,
        ``<metric>``.

    Raises
    ------
    ValueError
        If ``df`` has no ``department`` column, or ``metric`` is unknown.
    """
    if "department" not in df.columns:
        raise ValueError("DataFrame must have a 'department' column.")

    df = df.copy()
    df["arrival_time"] = pd.to_datetime(df["arrival_time"])

    grouped = (
        df.groupby(["department", pd.Grouper(key="arrival_time", freq=freq)])
        .agg(
            energy_kwh=("energy", "sum"),
            n_sessions=("energy", "count"),
            mean_energy_kwh=("energy", "mean"),
            mean_duration_h=("duration", "mean"),
        )
        .reset_index()
        .rename(columns={"arrival_time": "date"})
    )

    if metric not in grouped.columns:
        raise ValueError(
            f"metric must be one of {list(grouped.columns[2:])}, got '{metric}'"
        )

    return grouped[["date", "department", metric]]


def build_panel(
    df: pd.DataFrame,
    freq: str = "D",
    metric: str = "energy_kwh",
    fill_value: float = 0.0,
) -> pd.DataFrame:
    """
    Build a wide-format panel: rows = dates, columns = departments.

    Missing (date, department) pairs are filled with ``fill_value``.

    Parameters
    ----------
    df : pd.DataFrame
        Validated sessions DataFrame.
    freq : str
        Aggregation frequency.
    metric : str
        Metric to aggregate.
    fill_value : float
        Value for missing periods.

    Returns
    -------
    pd.DataFrame
        Wide panel with ``DatetimeIndex`` and one column per department.
    """
    long = aggregate_by_department(df, freq=freq, metric=metric)
    panel = long.pivot(index="date", columns="department", values=metric)
    panel.index = pd.to_datetime(panel.index)

    # Fill in all dates for every department, including those with no sessions.
    full_range = pd.date_range(panel.index.min(), panel.index.max(), freq=freq)
    panel = panel.reindex(full_range, fill_value=fill_value)

    return panel


def department_daily_energy(
    df: pd.DataFrame,
    departments: list | None = None,
) -> pd.DataFrame:
    """
    Compute daily energy (kWh) per department.

    Parameters
    ----------
    df : pd.DataFrame
        Validated sessions DataFrame.
    departments : list, optional
        Subset of departments to include. If None, all are used.

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns: ``date``, ``department``,
        ``energy_kwh``.
    """
    long = aggregate_by_department(df, freq="D", metric="energy_kwh")
    if departments is not None:
        long = long[long["department"].isin(departments)]
    return long


# ---------------------------------------------------------------------------
# Department-level SARIMA forecaster
# ---------------------------------------------------------------------------

class DepartmentForecaster:
    """
    Forecast daily energy demand per INSEE department using SARIMA.

    Fits one time series model per department on historical daily energy
    data and produces multi-step forecasts with uncertainty intervals.

    Parameters
    ----------
    seasonal_period : int
        Seasonal period for SARIMA (7 = weekly, default).
    max_p, max_q, max_P, max_Q : int
        SARIMA order search bounds (used by auto_arima).
    min_obs : int
        Minimum number of observations required to fit a department model.
        Departments with fewer observations use the global mean.
    use_log : bool
        If True, log-transform the target before fitting (recommended for
        count/energy data with positive skew).
    random_state : int
        Random seed for pmdarima's stepwise search.

    Examples
    --------
    >>> fc = DepartmentForecaster()
    >>> fc.fit(df)                         # df from load_sessions
    >>> forecast = fc.predict(horizon=30)  # 30-day forecast per department
    """

    def __init__(
        self,
        seasonal_period: int = 7,
        max_p: int = 3,
        max_q: int = 3,
        max_P: int = 1,
        max_Q: int = 1,
        min_obs: int = 60,
        use_log: bool = True,
        random_state: int = 42,
    ):
        self.seasonal_period = seasonal_period
        self.max_p = max_p
        self.max_q = max_q
        self.max_P = max_P
        self.max_Q = max_Q
        self.min_obs = min_obs
        self.use_log = use_log
        self.random_state = random_state

        self._models: dict[str, object] = {}
        self._dept_stats: dict[str, dict] = {}
        self._panel: pd.DataFrame | None = None
        self.is_fitted_: bool = False
        self.departments_: list[str] = []

    def fit(
        self,
        df: pd.DataFrame,
        departments: list | None = None,
        verbose: bool = True,
    ) -> DepartmentForecaster:
        """
        Fit SARIMA models on historical daily energy per department.

        Parameters
        ----------
        df : pd.DataFrame
            Validated sessions DataFrame (from ``load_sessions``).
        departments : list, optional
            Subset of departments to fit. Defaults to all.
        verbose : bool
            Print fitting progress.

        Returns
        -------
        DepartmentForecaster
            The fitted instance (``self``).
        """
        panel = build_panel(df, freq="D", metric="energy_kwh")
        self._panel = panel

        depts = departments or list(panel.columns)
        self.departments_ = depts

        for dept in depts:
            series = panel[dept].dropna()
            n = len(series)
            self._dept_stats[dept] = {
                "mean": float(series.mean()),
                "std": float(series.std()),
                "n_obs": n,
                "last_date": series.index[-1] if n > 0 else None,
            }

            if n < self.min_obs:
                if verbose:
                    logger.debug(
                        "Dept %s: only %d obs, using mean fallback.", dept, n
                    )
                continue

            try:
                model = self._fit_sarima(series)
                self._models[dept] = model
                if verbose:
                    logger.debug("Dept %s: SARIMA fitted (n=%d).", dept, n)
            except Exception as e:  # noqa: BLE001 - fall back to naive mean forecast on any SARIMA failure
                logger.warning("Dept %s: SARIMA failed (%s), using mean.", dept, e)

        if verbose:
            fitted = len(self._models)
            print(
                f"[GEARS] DepartmentForecaster fitted: "
                f"{fitted}/{len(depts)} departments with SARIMA, "
                f"{len(depts)-fitted} using mean fallback."
            )
        self.is_fitted_ = True
        return self

    def _fit_sarima(self, series: pd.Series) -> object:
        """
        Fit SARIMA to a daily energy series via pmdarima or statsmodels.

        Tries ``pmdarima.auto_arima`` first; falls back to a fixed
        SARIMAX(1,1,1)(1,1,0,s) via statsmodels if pmdarima is unavailable.

        Parameters
        ----------
        series : pd.Series
            Daily energy values (positive reals).

        Returns
        -------
        object
            A fitted pmdarima or statsmodels SARIMAX model.
        """
        y = np.log1p(series.values) if self.use_log else series.values

        try:
            import pmdarima as pm
            model = pm.auto_arima(
                y,
                seasonal=True,
                m=self.seasonal_period,
                max_p=self.max_p,
                max_q=self.max_q,
                max_P=self.max_P,
                max_Q=self.max_Q,
                information_criterion="bic",
                stepwise=True,
                suppress_warnings=True,
                error_action="ignore",
                random_state=self.random_state,
            )
            return model
        except ImportError:
            pass

        from statsmodels.tsa.statespace.sarimax import SARIMAX
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mod = SARIMAX(
                y,
                order=(1, 1, 1),
                seasonal_order=(1, 1, 0, self.seasonal_period),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            result = mod.fit(disp=False)
            result._is_statsmodels = True
            result._use_log = self.use_log
            return result

    def predict(
        self,
        horizon: int = 30,
        departments: list | None = None,
        start_date: str | None = None,
        n_scenarios: int = 1,
        seed: int = 42,
    ) -> pd.DataFrame:
        """
        Forecast daily energy demand per department.

        Parameters
        ----------
        horizon : int
            Number of days to forecast.
        departments : list, optional
            Subset of departments. Defaults to all fitted departments.
        start_date : str, optional
            First forecasted date. Defaults to the day after the last
            training date.
        n_scenarios : int
            Number of stochastic scenarios (for uncertainty quantification).
        seed : int
            Random seed for scenario noise.

        Returns
        -------
        pd.DataFrame
            Long-format table with columns: ``date`` (DatetimeIndex),
            ``department``, ``scenario`` (int), ``energy_kwh_forecast``.
        """
        if not self.is_fitted_:
            raise RuntimeError("Call .fit() before predict().")

        rng = np.random.default_rng(seed)
        depts = departments or self.departments_
        rows = []

        for dept in depts:
            stats = self._dept_stats.get(dept, {"mean": 0.0, "std": 0.0})

            if start_date is not None:
                first_date = pd.Timestamp(start_date)
            elif stats.get("last_date") is not None:
                first_date = stats["last_date"] + pd.Timedelta(days=1)
            else:
                first_date = pd.Timestamp.today().normalize()

            dates = pd.date_range(first_date, periods=horizon, freq="D")

            for s in range(n_scenarios):
                fc = self._forecast_dept(dept, horizon, rng)
                for d, v in zip(dates, fc):
                    rows.append({
                        "date": d,
                        "department": dept,
                        "scenario": s,
                        "energy_kwh_forecast": max(0.0, float(v)),
                    })

        return pd.DataFrame(rows)

    def _forecast_dept(
        self, dept: str, horizon: int, rng: np.random.Generator
    ) -> np.ndarray:
        """
        Generate a stochastic forecast for one department.

        Uses the fitted SARIMA if available; otherwise falls back to a
        mean-plus-noise baseline drawn from historical statistics.

        Parameters
        ----------
        dept : str
            Department code.
        horizon : int
            Number of forecast days.
        rng : np.random.Generator

        Returns
        -------
        np.ndarray, shape (horizon,)
            Non-negative daily energy forecasts (kWh).
        """
        stats = self._dept_stats.get(dept, {"mean": 0.0, "std": 1.0})
        model = self._models.get(dept)

        if model is None:
            noise = rng.normal(0, stats["std"] * 0.1, horizon)
            return np.maximum(0, stats["mean"] + noise)

        try:
            if hasattr(model, "_is_statsmodels"):
                raw_fc = model.forecast(horizon)
            else:
                raw_fc = model.predict(n_periods=horizon)

            fc = np.expm1(raw_fc) if self.use_log else raw_fc
            # Use the full training std as noise scale, not a small fraction of it.
            # AUDIT.md §e (Mechanism 1) traced this: with the old `std * 0.05` scale,
            # the resulting 80% CI band stayed at ~2% of the median at every horizon
            # step — visually a flat, pinched line rather than a genuine forecast
            # cone, which is what produced the "artificial plateau" look in notebook
            # 3's medium-term fan charts. `gears.models.forecaster.SessionForecaster.
            # _forecast_one_scenario` already made this exact fix (full std achieves
            # ~87% empirical CI coverage vs. ~33% for a small fraction) but it was
            # never ported to this class. Fixed in Session 6; see REFACTOR_STATE.md.
            noise_scale = max(stats["std"], 1.0)
            fc = fc + rng.normal(0, noise_scale, horizon)
            return np.maximum(0, fc)
        except Exception:  # noqa: BLE001 - fall back to stats-based sample on any forecast failure
            noise = rng.normal(0, stats["std"] * 0.1, horizon)
            return np.maximum(0, stats["mean"] + noise)

    def plot_forecast(
        self,
        department: str,
        horizon: int = 30,
        n_scenarios: int = 200,
        seed: int = 0,
        ax=None,
        figsize: tuple = (12, 4),
    ):
        """
        Plot historical energy and forecast fan chart for one department.

        Parameters
        ----------
        department : str
            INSEE department code.
        horizon : int
            Number of days to forecast.
        n_scenarios : int
            Number of scenarios for the uncertainty band.
        seed : int
            Random seed.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. A new figure is created if None.
        figsize : tuple
            Figure size (width, height) in inches.

        Returns
        -------
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt

        if not self.is_fitted_:
            raise RuntimeError("Call .fit() first.")

        fc_df = self.predict(
            horizon=horizon,
            departments=[department],
            n_scenarios=n_scenarios,
            seed=seed,
        )
        pivot = fc_df.pivot(index="date", columns="scenario", values="energy_kwh_forecast")

        fig, ax_ = (None, ax) if ax is not None else plt.subplots(figsize=figsize)
        ax_ = ax_ or plt.gca()

        if self._panel is not None and department in self._panel.columns:
            hist = self._panel[department].dropna().tail(90)  # show last 90 days
            hist.plot(ax=ax_, label="Historical (kWh/day)", color="#2E86AB", linewidth=1.5)

        ax_.fill_between(
            pivot.index,
            pivot.quantile(0.1, axis=1),
            pivot.quantile(0.9, axis=1),
            alpha=0.2, color="#E84855", label="80% CI",
        )
        ax_.fill_between(
            pivot.index,
            pivot.quantile(0.25, axis=1),
            pivot.quantile(0.75, axis=1),
            alpha=0.3, color="#E84855", label="50% CI",
        )
        pivot.median(axis=1).plot(ax=ax_, label="Median forecast", color="#E84855", linewidth=2)

        ax_.set_ylabel("Daily energy (kWh)")
        ax_.set_title(f"Energy forecast – Department {department}")
        ax_.legend()
        ax_.grid(True, alpha=0.3)
        if fig:
            fig.tight_layout()
        return ax_

    def __repr__(self) -> str:
        if not self.is_fitted_:
            return "DepartmentForecaster(not fitted)"
        return (
            f"DepartmentForecaster("
            f"n_departments={len(self.departments_)}, "
            f"n_fitted_sarima={len(self._models)}, "
            f"seasonal_period={self.seasonal_period})"
        )

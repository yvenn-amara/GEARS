"""
Session-count forecasters for GEARS.

Three forecasters with a unified interface:

1. **SessionForecaster** (SARIMA / probabilistic)
   - SARIMA noise scale uses full training std (not 10%) for realistic 90% CI coverage.
   Fits on historical daily session counts using SARIMA (via pmdarima or
   statsmodels) or a simple normal-distribution assumption.

2. **TransformerForecaster** (PatchTST, optional)
   State-of-the-art transformer for time-series (Nie et al., ICLR 2023).
   Requires the ``[transformer]`` optional extra::

       pip install "gears-ev[transformer]"

   The package works fully without it — if PyTorch / neuralforecast are not
   installed a clear ImportError is raised with installation instructions,
   and all other GEARS functionality remains available.

3. **PersistenceForecaster**
   Naive baseline: forecast = value from the same weekday N weeks ago.
   Useful as a baseline comparison to quantify the value of SARIMA / transformer.

4. **NHiTSForecaster** (NHiTS, optional)
   NHiTS (Challu et al., AAAI 2023). Requires PyTorch and neuralforecast::

       uv pip install "gears-ev[dl]"

   Implementation notes
   --------------------
   - input_size default: ``2 * horizon`` (reduced from ``4 * horizon``),
     additionally hard-capped at ``n_train // 2`` inside ``fit()``.  The
     old default of 360 days for horizon=90 consumed ≥ 80 % of a typical
     training set as context window, leaving too few gradient windows and
     causing instability or memorisation of recent values.
   - scaler_type="standard": NeuralForecast normalises y internally when
     this is set.  Without it, raw session counts (0–200) produce erratic
     gradients whose scale is ~100× larger than normalised targets.
   - The public ``fit()`` method exposes ``input_size`` as a keyword
     argument so callers can override without subclassing.

All forecasters share the same ``fit(df)`` / ``predict(horizon, ...)`` interface
so they can be swapped transparently in notebooks and pipeline code.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sessions_to_daily_counts(df: pd.DataFrame) -> pd.Series:
    """
    Aggregate a sessions DataFrame to a daily count Series indexed by date.

    Missing dates between the first and last observation are filled with 0
    so that the returned Series covers a contiguous date range.

    Parameters
    ----------
    df : pd.DataFrame
        Validated sessions DataFrame containing a ``date`` column.

    Returns
    -------
    pd.Series
        Daily session counts with a ``DatetimeIndex``; name is ``'count'``.
    """
    daily = df.groupby("date").size().rename("count")
    daily.index = pd.to_datetime(daily.index)
    full_range = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    return daily.reindex(full_range, fill_value=0)


def _make_forecast_df(dates: pd.DatetimeIndex, counts_matrix: np.ndarray) -> pd.DataFrame:
    """Build the standard forecast DataFrame from a (n_scenarios, horizon) matrix.

    Parameters
    ----------
    dates : pd.DatetimeIndex
        Forecast dates, one per column of *counts_matrix*.
    counts_matrix : np.ndarray
        Shape ``(n_scenarios, horizon)``; raw (possibly fractional) counts.

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns: date, scenario, n_sessions.
        Each count is clipped to 0 and rounded to the nearest integer.
    """
    rows = []
    for s, counts in enumerate(counts_matrix):
        for d, c in zip(dates, counts):
            rows.append({"date": d, "scenario": s, "n_sessions": max(0, round(float(c)))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. SessionForecaster — SARIMA / probabilistic
# ---------------------------------------------------------------------------

class SessionForecaster:
    """
    Forecast daily session counts using SARIMA or simple probabilistic assumptions.

    Parameters
    ----------
    method : str
        ``'sarima'`` — auto-selects the best ARIMA+seasonal order via BIC
        (requires pmdarima; falls back to statsmodels SARIMAX(1,1,1)x(1,1,0,7)
        if pmdarima is not installed).
        ``'probabilistic'`` — samples from a normal distribution fitted on the
        historical mean and standard deviation. Fast, no dependencies.
    seasonal_period : int
        Seasonal period for SARIMA (7 = weekly, default).
    use_holidays : bool
        If True, add a French bank-holiday dummy as an exogenous regressor
        in SARIMA.  Requires the ``holidays`` package (installed by default).
    country : str
        ISO 3166-1 alpha-2 country code for the holiday calendar (default 'FR').
    max_p, max_q, max_P, max_Q : int
        SARIMA order search bounds (used by auto_arima).
    random_state : int

    Examples
    --------
    >>> fc = SessionForecaster(method='sarima', use_holidays=True)
    >>> fc.fit(df)
    >>> forecast = fc.predict(horizon=14, n_scenarios=50)
    """

    def __init__(
        self,
        method: str = "sarima",
        seasonal_period: int = 7,
        use_holidays: bool = True,
        country: str = "FR",
        max_p: int = 3,
        max_q: int = 3,
        max_P: int = 2,
        max_Q: int = 2,
        random_state: int = 42,
    ):
        if method not in ("sarima", "probabilistic"):
            raise ValueError(f"method must be 'sarima' or 'probabilistic', got '{method}'")
        self.method = method
        self.seasonal_period = seasonal_period
        self.use_holidays = use_holidays
        self.country = country
        self.max_p = max_p
        self.max_q = max_q
        self.max_P = max_P
        self.max_Q = max_Q
        self.random_state = random_state

        self._model = None
        self._daily_counts: pd.Series | None = None
        self._exog_train: np.ndarray | None = None
        self.is_fitted_: bool = False

    def fit(self, df: pd.DataFrame) -> SessionForecaster:
        """
        Fit on historical daily session counts.

        Parameters
        ----------
        df : pd.DataFrame
            Validated sessions DataFrame (output of load_sessions).

        Returns
        -------
        SessionForecaster
            The fitted model (``self``), allowing method chaining.
        """
        self._daily_counts = sessions_to_daily_counts(df)
        self.mean_daily_ = float(self._daily_counts.mean())
        self.std_daily_ = float(self._daily_counts.std())

        if self.method == "probabilistic":
            logger.info("Probabilistic mode: stored mean/std of daily counts.")
            self.is_fitted_ = True
            return self

        # Build holiday exogenous regressor for training dates
        self._exog_train = self._build_holiday_exog(self._daily_counts.index)

        n = len(self._daily_counts)
        logger.info("Fitting SARIMA on %d days of data.", n)
        try:
            import pmdarima as pm
            self._model = pm.auto_arima(
                self._daily_counts.values,
                exogenous=self._exog_train,
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
            logger.info("SARIMA order: %s seasonal: %s", self._model.order,
                        self._model.seasonal_order)
        except ImportError:
            logger.warning(
                "pmdarima not installed; falling back to statsmodels SARIMAX(1,1,1)x(1,1,0,7)."
            )
            self._fit_statsmodels_fallback(exog=self._exog_train)

        self.is_fitted_ = True
        return self

    def _build_holiday_exog(
        self,
        date_index: pd.DatetimeIndex,
    ) -> np.ndarray | None:
        """Build a [n, 1] holiday dummy array for the given date range.

        Returns None when use_holidays=False or when the holidays package
        is unavailable (falls back gracefully without raising).

        Parameters
        ----------
        date_index : pd.DatetimeIndex
            Dates for which the dummy should be constructed.

        Returns
        -------
        np.ndarray of shape (n, 1) or None
            1.0 on public holidays, 0.0 otherwise.
        """
        if not self.use_holidays:
            return None
        try:
            import holidays as hol
            cal = hol.country_holidays(self.country)
            dummy = np.array(
                [1.0 if d.date() in cal else 0.0 for d in date_index],
                dtype=float,
            ).reshape(-1, 1)
            n_holidays = int(dummy.sum())
            logger.info(
                "Holiday exog: %d public holidays in %d-day window (%s).",
                n_holidays, len(date_index), self.country,
            )
            return dummy
        except Exception as e:  # noqa: BLE001 - holiday exog is optional; any failure falls back to None
            logger.warning("Could not build holiday exog: %s. Proceeding without.", e)
            return None

    def _fit_statsmodels_fallback(self, exog=None) -> None:
        """Fit a fixed SARIMAX(1,1,1)x(1,1,0,7) when pmdarima is unavailable.

        Used as a fallback so the forecaster remains functional without the
        pmdarima dependency.  The fixed order is a sensible default for
        weekly-seasonal daily count data.

        Parameters
        ----------
        exog : np.ndarray or None
            Holiday dummy array of shape ``(n_train, 1)``, or None.
        """
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mod = SARIMAX(
                self._daily_counts.values,
                exog=exog,
                order=(1, 1, 1),
                seasonal_order=(1, 1, 0, self.seasonal_period),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            self._model = mod.fit(disp=False)
            self._model._is_statsmodels = True

    def predict(
        self,
        horizon: int = 7,
        n_scenarios: int = 1,
        start_date: str | pd.Timestamp | None = None,
        seed: int | None = None,
    ) -> pd.DataFrame:
        """
        Predict daily session counts for the next ``horizon`` days.

        Parameters
        ----------
        horizon : int
            Number of days to forecast.
        n_scenarios : int
            Number of stochastic scenarios (> 1 enables Monte-Carlo uncertainty).
        start_date : str or Timestamp, optional
            First forecasted date. Defaults to the day after the last training date.
        seed : int, optional
            Random seed for scenario noise.

        Returns
        -------
        pd.DataFrame
            Columns: date, scenario, n_sessions.
        """
        if not self.is_fitted_:
            raise RuntimeError("Call .fit() before predict().")

        rng = np.random.default_rng(seed)
        start_date = self._resolve_start(start_date)
        dates = pd.date_range(start_date, periods=horizon, freq="D")

        # Build holiday exog for the forecast period (same structure as training)
        future_exog = self._build_holiday_exog(dates)

        matrix = np.stack(
            [self._forecast_one_scenario(horizon, rng, future_exog) for _ in range(n_scenarios)]
        )
        return _make_forecast_df(dates, matrix)

    def _resolve_start(self, start_date) -> pd.Timestamp:
        """Resolve the forecast start date.

        Parameters
        ----------
        start_date : str, Timestamp, or None

        Returns
        -------
        pd.Timestamp
            The first date of the forecast window.
        """
        if start_date is not None:
            return pd.Timestamp(start_date)
        if self._daily_counts is not None:
            return self._daily_counts.index[-1] + pd.Timedelta(days=1)
        return pd.Timestamp.today().normalize()

    def _forecast_one_scenario(
        self, horizon: int, rng: np.random.Generator,
        future_exog: np.ndarray | None = None,
    ) -> np.ndarray:
        """Generate one stochastic forecast trajectory.

        Draws the deterministic SARIMA point forecast and adds Gaussian noise
        scaled to the full training standard deviation.  Using the full std
        (rather than a fractional one) is necessary to achieve empirically
        realistic 80–90 % confidence interval coverage.

        Parameters
        ----------
        horizon : int
            Number of forecast days.
        rng : np.random.Generator
            Seeded random generator for noise draws.
        future_exog : np.ndarray or None
            Holiday dummy for the forecast window, shape ``(horizon, 1)``.

        Returns
        -------
        np.ndarray
            Non-negative forecast counts of shape ``(horizon,)``.
        """
        if self.method == "probabilistic":
            return np.maximum(0, rng.normal(self.mean_daily_, self.std_daily_, horizon))

        if self._model is None:
            return np.full(horizon, self.mean_daily_)

        try:
            if hasattr(self._model, "_is_statsmodels"):
                fc = self._model.forecast(horizon, exog=future_exog)
            else:
                fc = self._model.predict(n_periods=horizon, X=future_exog)
        except Exception:  # noqa: BLE001 - predict fallback to mean on any SARIMA failure
            return np.full(horizon, self.mean_daily_)

        # Use the full training std as noise scale so that the 90 % CI achieves
        # ~87 % empirical coverage.  A small fraction (e.g. 0.1 × std) was used
        # historically and produced only ~33 % coverage — a serious underestimate
        # of real forecast uncertainty.
        noise_scale = max(self.std_daily_, 1.0)
        fc = fc + rng.normal(0, noise_scale, horizon)
        return np.maximum(0, fc)

    def plot_forecast(self, horizon: int = 14, n_scenarios: int = 50,
                      seed: int = 0, ax=None, figsize: tuple = (12, 4)):
        """Fan chart of history and forecast (see :func:`gears.plotting.plot_forecast_fan`)."""
        from gears.plotting import plot_forecast_fan

        return plot_forecast_fan(self, horizon, n_scenarios, seed, ax, figsize)

    def __repr__(self) -> str:
        return f"SessionForecaster(method={self.method!r}, fitted={self.is_fitted_})"


# ---------------------------------------------------------------------------
# 2. TransformerForecaster — PatchTST (optional)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 3. PersistenceForecaster — naive same-weekday baseline
# ---------------------------------------------------------------------------

class PersistenceForecaster:
    """
    Naive persistence baseline: forecast = value from ``n_weeks`` ago on the
    same day-of-week.

    This represents the simplest possible "forecaster" — what an operator
    would do with no model at all, just looking at last week's numbers.
    It is the standard baseline for evaluating the **added value** of SARIMA
    or transformer models.

    Parameters
    ----------
    n_weeks : int
        Look-back window in weeks (default 1 = same weekday last week).

    Examples
    --------
    >>> fc = PersistenceForecaster(n_weeks=1)
    >>> fc.fit(df)
    >>> forecast = fc.predict(horizon=14)
    """

    def __init__(self, n_weeks: int = 1):
        self.n_weeks = n_weeks
        self._daily_counts: pd.Series | None = None
        self.is_fitted_: bool = False

    def fit(self, df: pd.DataFrame) -> PersistenceForecaster:
        """
        Fit by storing historical daily session counts.

        Parameters
        ----------
        df : pd.DataFrame
            Validated sessions DataFrame (output of load_sessions).

        Returns
        -------
        PersistenceForecaster
            The fitted model (``self``), allowing method chaining.
        """
        self._daily_counts = sessions_to_daily_counts(df)
        self.mean_daily_ = float(self._daily_counts.mean())
        self.std_daily_ = float(self._daily_counts.std())
        self.is_fitted_ = True
        return self

    def predict(
        self,
        horizon: int = 7,
        n_scenarios: int = 1,
        start_date: str | pd.Timestamp | None = None,
        seed: int | None = None,
    ) -> pd.DataFrame:
        """
        Forecast by repeating the value from ``n_weeks`` weeks ago.

        Parameters
        ----------
        horizon : int
            Number of days to forecast.
        n_scenarios : int
            Number of scenarios. For n_scenarios > 1, a small jitter is added
            for multi-scenario compatibility; the forecaster is otherwise
            deterministic.
        start_date : str or Timestamp, optional
            First forecasted date.
        seed : int, optional
            Random seed for jitter noise.

        Returns
        -------
        pd.DataFrame
            Columns: date, scenario, n_sessions.
        """
        if not self.is_fitted_:
            raise RuntimeError("Call .fit() before predict().")

        if start_date is not None:
            start = pd.Timestamp(start_date)
        elif self._daily_counts is not None:
            start = self._daily_counts.index[-1] + pd.Timedelta(days=1)
        else:
            start = pd.Timestamp.today().normalize()

        dates = pd.date_range(start, periods=horizon, freq="D")
        look_back = pd.Timedelta(weeks=self.n_weeks)

        fc_values = []
        for d in dates:
            ref = d - look_back
            if self._daily_counts is not None and ref in self._daily_counts.index:
                fc_values.append(float(self._daily_counts[ref]))
            elif self._daily_counts is not None:
                # Carry forward: find the closest available same-weekday value
                # going further back in time so we never return 0 or mean
                candidate = ref
                found = False
                for _ in range(52):  # search up to 52 weeks back
                    candidate -= pd.Timedelta(weeks=1)
                    if candidate in self._daily_counts.index:
                        fc_values.append(float(self._daily_counts[candidate]))
                        found = True
                        break
                if not found:
                    fc_values.append(self.mean_daily_)
            else:
                fc_values.append(self.mean_daily_)

        rng = np.random.default_rng(seed)
        # Persistence is deterministic; tile the single trajectory across scenarios
        matrix = np.tile(np.array(fc_values), (n_scenarios, 1))
        # Add tiny noise so multi-scenario consumers have distinct columns
        if n_scenarios > 1:
            noise = rng.normal(0, max(self.std_daily_ * 0.03, 0.1), matrix.shape)
            matrix = np.maximum(0, matrix + noise)
        return _make_forecast_df(dates, matrix)

    def __repr__(self) -> str:
        return (
            f"PersistenceForecaster(n_weeks={self.n_weeks}, "
            f"fitted={self.is_fitted_})"
        )


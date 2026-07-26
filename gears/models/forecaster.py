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

    def plot_forecast(
        self,
        horizon: int = 14,
        n_scenarios: int = 50,
        seed: int = 0,
        ax=None,
        figsize: tuple = (12, 4),
    ):
        """Plot historical counts and a fan-chart forecast.

        Parameters
        ----------
        horizon : int
            Number of forecast days to display.
        n_scenarios : int
            Number of stochastic scenarios used for the fan chart.
        seed : int
            Random seed for scenario generation.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. Created if None.
        figsize : tuple
            Figure size in inches, used only when creating a new figure.

        Returns
        -------
        matplotlib.axes.Axes
            The axes containing the plot.
        """
        import matplotlib.pyplot as plt

        forecast = self.predict(horizon=horizon, n_scenarios=n_scenarios, seed=seed)
        pivot = forecast.pivot(index="date", columns="scenario", values="n_sessions")

        fig, ax_ = (None, ax) if ax is not None else plt.subplots(figsize=figsize)
        ax_ = ax_ or plt.gca()

        if self._daily_counts is not None:
            self._daily_counts.plot(ax=ax_, label="Historical", color="#2E86AB", linewidth=1.2)
        ax_.fill_between(
            pivot.index,
            pivot.quantile(0.1, axis=1),
            pivot.quantile(0.9, axis=1),
            alpha=0.2, color="#E84855", label="80% CI",
        )
        pivot.median(axis=1).plot(
            ax=ax_, label="Median forecast", color="#E84855", linewidth=2
        )
        ax_.set_ylabel("Daily sessions")
        ax_.set_title("Session count forecast")
        ax_.legend()
        ax_.grid(True, alpha=0.3)
        if fig:
            fig.tight_layout()
        return ax_

    def __repr__(self) -> str:
        return f"SessionForecaster(method={self.method!r}, fitted={self.is_fitted_})"


# ---------------------------------------------------------------------------
# 2. TransformerForecaster — PatchTST (optional)
# ---------------------------------------------------------------------------

class TransformerForecaster:
    """
    Transformer-based session-count forecaster using PatchTST.

    PatchTST (Nie et al., ICLR 2023) divides the input time series into
    fixed-length patches (like image patches in ViT) and applies a standard
    Transformer encoder.  It achieves state-of-the-art performance on
    long-range univariate forecasting benchmarks while remaining lightweight
    and fast to train on CPU.

    **This forecaster is optional.**  It requires PyTorch and neuralforecast::

        pip install "gears-ev[transformer]"
        # or: pip install neuralforecast torch

    If these are not installed, a clear ``ImportError`` is raised.
    All other GEARS functionality works without them.

    Same interface as ``SessionForecaster``.

    Parameters
    ----------
    horizon : int
        Fixed forecast horizon (days).  PatchTST is horizon-specific;
        if ``predict(horizon=...)`` is called with a different horizon,
        the model is refitted automatically.
    input_size : int
        Number of past days used as input context. Defaults to 4 × horizon.
    max_steps : int
        Training steps.  200 is enough for typical EV datasets; increase
        for very long or noisy series.
    patch_len : int
        Patch length (days). Default 7 aligns with weekly seasonality.
    random_state : int

    Examples
    --------
    >>> fc = TransformerForecaster(horizon=14)
    >>> fc.fit(df)
    >>> forecast = fc.predict(horizon=14, n_scenarios=50)
    """

    _IMPORT_MSG = (
        "TransformerForecaster requires neuralforecast and PyTorch.\n"
        "Install with:  pip install 'gears-ev[transformer]'\n"
        "or:            pip install neuralforecast torch"
    )

    def __init__(
        self,
        horizon: int = 14,
        input_size: int | None = None,
        max_steps: int = 200,
        patch_len: int = 7,
        random_state: int = 42,
    ):
        self.horizon = horizon
        self.input_size = input_size or max(4 * horizon, 28)
        self.max_steps = max_steps
        self.patch_len = patch_len
        self.random_state = random_state

        self._nf = None
        self._daily_counts: pd.Series | None = None
        self.is_fitted_: bool = False

    @staticmethod
    def is_available() -> bool:
        """Return True if neuralforecast and torch are installed.

        Returns
        -------
        bool
        """
        try:
            import neuralforecast  # noqa: F401
            import torch  # noqa: F401
            return True
        except ImportError:
            return False

    def fit(self, df: pd.DataFrame) -> TransformerForecaster:
        """
        Fit PatchTST on historical daily session counts.

        Parameters
        ----------
        df : pd.DataFrame
            Validated sessions DataFrame (output of load_sessions).

        Returns
        -------
        TransformerForecaster
            The fitted model (``self``), allowing method chaining.

        Raises
        ------
        ImportError
            If neuralforecast or torch are not installed.
        """
        try:
            from neuralforecast import NeuralForecast
            from neuralforecast.models import PatchTST
        except ImportError:
            raise ImportError(self._IMPORT_MSG)

        self._daily_counts = sessions_to_daily_counts(df)
        self.mean_daily_ = float(self._daily_counts.mean())
        self.std_daily_ = float(self._daily_counts.std())

        n = len(self._daily_counts)
        logger.info("Fitting PatchTST on %d days of data.", n)

        train_df = pd.DataFrame({
            "unique_id": "series_0",
            "ds": self._daily_counts.index,
            "y": self._daily_counts.values.astype(float),
        })

        model = PatchTST(
            h=self.horizon,
            input_size=min(self.input_size, n - 1),
            patch_len=self.patch_len,
            max_steps=self.max_steps,
            random_seed=self.random_state,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import logging as _logging
            _logging.getLogger("pytorch_lightning").setLevel(_logging.ERROR)
            self._nf = NeuralForecast(models=[model], freq="D")
            self._nf.fit(train_df)

        self._fitted_horizon = self.horizon
        self.is_fitted_ = True
        logger.info("PatchTST fitted.")
        return self

    def predict(
        self,
        horizon: int = 14,
        n_scenarios: int = 1,
        start_date: str | pd.Timestamp | None = None,
        seed: int | None = None,
    ) -> pd.DataFrame:
        """
        Predict daily session counts.

        If ``horizon`` differs from the training horizon the model is refitted.

        Parameters
        ----------
        horizon : int
        n_scenarios : int
            Number of stochastic scenarios (noise is added to the deterministic
            PatchTST output to produce scenario uncertainty).
        start_date : str or Timestamp, optional
        seed : int, optional

        Returns
        -------
        pd.DataFrame
            Columns: date, scenario, n_sessions.
        """
        if not self.is_fitted_:
            raise RuntimeError("Call .fit() before predict().")

        if horizon != self._fitted_horizon:
            logger.warning(
                "Requested horizon %d != fitted horizon %d; refitting.",
                horizon, self._fitted_horizon,
            )
            self.horizon = horizon
            self._fitted_horizon = horizon
            # Refit with the new horizon so the model head matches the output length
            from neuralforecast import NeuralForecast
            from neuralforecast.models import PatchTST
            n = len(self._daily_counts)
            train_df = pd.DataFrame({
                "unique_id": "series_0",
                "ds": self._daily_counts.index,
                "y": self._daily_counts.values.astype(float),
            })
            model = PatchTST(
                h=horizon,
                input_size=min(self.input_size, n - 1),
                patch_len=self.patch_len,
                max_steps=self.max_steps,
                random_seed=self.random_state,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                import logging as _logging
                _logging.getLogger("pytorch_lightning").setLevel(_logging.ERROR)
                self._nf = NeuralForecast(models=[model], freq="D")
                self._nf.fit(train_df)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw_pred = self._nf.predict()

        fc_values = raw_pred["PatchTST"].values
        fc_dates = pd.to_datetime(raw_pred["ds"].values)

        # Align to requested start_date if provided; keep the forecast pattern,
        # just re-anchor the date axis.
        if start_date is not None:
            req_start = pd.Timestamp(start_date)
            if req_start != fc_dates[0]:
                fc_dates = pd.date_range(req_start, periods=len(fc_dates), freq="D")

        fc_dates = fc_dates[:horizon]
        fc_values = fc_values[:horizon]

        rng = np.random.default_rng(seed)
        noise_scale = max(self.std_daily_ * 0.08, 0.5)

        matrix = np.stack([
            np.maximum(0, fc_values + rng.normal(0, noise_scale, len(fc_values)))
            for _ in range(n_scenarios)
        ])
        return _make_forecast_df(pd.DatetimeIndex(fc_dates), matrix)

    def plot_forecast(self, horizon: int = 14, n_scenarios: int = 50,
                      seed: int = 0, ax=None, figsize: tuple = (12, 4)):
        """Plot historical counts and a fan-chart forecast.

        Parameters
        ----------
        horizon : int
            Number of forecast days to display.
        n_scenarios : int
            Number of stochastic scenarios used for the fan chart.
        seed : int
        ax : matplotlib.axes.Axes, optional
        figsize : tuple

        Returns
        -------
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt
        forecast = self.predict(horizon=horizon, n_scenarios=n_scenarios, seed=seed)
        pivot = forecast.pivot(index="date", columns="scenario", values="n_sessions")

        fig, ax_ = (None, ax) if ax is not None else plt.subplots(figsize=figsize)
        ax_ = ax_ or plt.gca()

        if self._daily_counts is not None:
            self._daily_counts.plot(ax=ax_, label="Historical", color="#2E86AB", linewidth=1.2)
        ax_.fill_between(pivot.index, pivot.quantile(0.1, axis=1),
                         pivot.quantile(0.9, axis=1), alpha=0.2, color="#9B5DE5", label="80% CI")
        pivot.median(axis=1).plot(ax=ax_, label="PatchTST median",
                                  color="#9B5DE5", linewidth=2)
        ax_.set_ylabel("Daily sessions")
        ax_.set_title("PatchTST session-count forecast")
        ax_.legend()
        ax_.grid(True, alpha=0.3)
        if fig:
            fig.tight_layout()
        return ax_

    def __repr__(self) -> str:
        return (
            f"TransformerForecaster(horizon={self.horizon}, "
            f"max_steps={self.max_steps}, fitted={self.is_fitted_})"
        )


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


# ---------------------------------------------------------------------------
# 4. NHiTSForecaster — Neural Hierarchical Interpolation for Time Series
# ---------------------------------------------------------------------------

class NHiTSForecaster:
    """
    NHiTS-based session-count forecaster.

    NHiTS (Challu et al., AAAI 2023) uses hierarchical interpolation and
    multi-rate input sampling to efficiently capture long- and short-range
    temporal patterns.  It is often faster and more accurate than
    transformer-based models on univariate datasets.

    **This forecaster is optional.**  It requires PyTorch and neuralforecast::

        uv pip install "gears-ev[dl]"

    Same interface as ``SessionForecaster`` and ``TransformerForecaster``.

    ⚠ SCOPE CONSTRAINT
    -------------------
    This forecaster predicts **session counts** (``n_sessions``), NOT energy.
    It must be fitted on the **same department scope** as the actual counts
    it will be evaluated against.  Fitting on multi-department data and
    evaluating against a single department produces ~N_dept× systematic bias.

    Concretely:
    - Notebook 2 (dept 92 evaluation): ``fc_nhits.fit(train_sarima)``
    - Notebook 3 (national evaluation): ``fc_nhits.fit(train_focus_depts)``
      Then convert predictions to energy via ``mean_energy_per_session``.

    Parameters
    ----------
    horizon : int
        Fixed forecast horizon (days).
    input_size : int, optional
        Number of past days used as input context.

        **Default: ``2 * horizon``** (reduced from ``4 * horizon``).
        The old default of ``4 * horizon`` caused ``input_size=360`` for
        horizon=90.  On a 450-day training set this meant 80 % of the data
        was used as a single context window, leaving only 90 points to
        compute the loss — effectively <1 full training epoch of gradient
        signal, producing random-walk behaviour.

        The effective value is additionally hard-capped at ``n_train // 2``
        inside ``fit()`` to ensure at least ``n_train // 2`` usable
        training windows.
    max_steps : int
        Training gradient steps.

        **Default: 200** (reduced from 300).  On small datasets
        (≤600 days), 300 steps combined with a large input_size caused
        overfitting.  Use 500+ only when ``n_train ≥ 2000``.
    n_stacks : int
        Number of stacked NHiTS blocks.
    scaler_type : str
        Internal NeuralForecast scaler applied to ``y`` before training.
        Default ``"standard"`` (zero-mean, unit-variance).  Set to
        ``"robust"`` for heavy-tailed count data.  Set to ``"identity"``
        to disable.  Normalisation is critical: without it, raw session
        counts (range 0–200) produce erratic gradients on default
        learning rates.
    random_state : int
    """

    _IMPORT_MSG = (
        "NHiTSForecaster requires neuralforecast and PyTorch.\n"
        "Install with:  uv pip install 'gears-ev[dl]'\n"
        "or:            pip install neuralforecast torch"
    )

    def __init__(
        self,
        horizon: int = 14,
        input_size: int | None = None,
        max_steps: int = 200,           # Reduced from 300: fewer steps prevent overfitting on small datasets
        n_stacks: int = 3,
        scaler_type: str = "standard",  # Explicitly set: normalisation is critical for raw count data (range 0–200)
        random_state: int = 42,
    ):
        self.horizon = horizon
        # Default context window is 2 × horizon (down from 4 ×): the larger
        # value caused input_size=360 for horizon=90, consuming 80 % of a
        # typical training set as context and leaving too few windows for
        # gradient descent.  An additional hard cap at n_train // 2 is
        # applied inside fit().
        self._input_size_arg = input_size         # None means "auto"
        self.input_size = input_size or max(2 * horizon, 14)
        self.max_steps = max_steps
        self.n_stacks = n_stacks
        self.scaler_type = scaler_type
        self.random_state = random_state

        self._nf = None
        self._daily_counts: pd.Series | None = None
        self.is_fitted_: bool = False
        self.mean_daily_: float = 0.0
        self.std_daily_: float = 1.0

    @staticmethod
    def is_available() -> bool:
        """Return True if neuralforecast and torch are installed.

        Returns
        -------
        bool
        """
        try:
            import neuralforecast  # noqa: F401
            import torch  # noqa: F401
            return True
        except ImportError:
            return False

    def fit(self, df: pd.DataFrame) -> NHiTSForecaster:
        """
        Fit NHiTS on historical daily session counts.

        ⚠ No-leakage contract
        ----------------------
        ``df`` must contain ONLY training data (i.e. sessions strictly before
        the evaluation start date).  No shuffling is applied; the temporal
        ordering is preserved as-is.

        Parameters
        ----------
        df : pd.DataFrame
            Validated sessions DataFrame (output of load_sessions).
            Must cover the same département scope as the actual counts
            against which predictions will be evaluated.

        Returns
        -------
        NHiTSForecaster
            The fitted model (``self``), allowing method chaining.

        Raises
        ------
        ImportError
            If neuralforecast or torch are not installed.
        """
        try:
            from neuralforecast import NeuralForecast
        except ImportError:
            raise ImportError(self._IMPORT_MSG)

        self._daily_counts = sessions_to_daily_counts(df)
        self.mean_daily_ = float(self._daily_counts.mean())
        self.std_daily_ = float(self._daily_counts.std())

        n = len(self._daily_counts)

        # Hard-cap input_size at n // 2: ensures the model has at least n // 2
        # usable training windows of width input_size.  Without this cap, a
        # 450-day dataset with input_size=360 yields only ~90 non-overlapping
        # windows — less than one full pass of gradient signal, causing the
        # model to memorise the last few values (random-walk collapse).
        effective_input_size = min(self.input_size, max(n // 2, self.horizon + 1))
        logger.info(
            "Fitting NHiTS on %d days of data.  input_size=%d (requested=%d, cap=n//2=%d).",
            n, effective_input_size, self.input_size, n // 2,
        )

        train_df = pd.DataFrame({
            "unique_id": "series_0",
            "ds": self._daily_counts.index,
            "y": self._daily_counts.values.astype(float),
        })

        model = self._build_nhits_model(
            h=self.horizon,
            input_size=effective_input_size,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import logging as _logging
            _logging.getLogger("pytorch_lightning").setLevel(_logging.ERROR)
            self._nf = NeuralForecast(models=[model], freq="D")
            self._nf.fit(train_df)

        self._fitted_horizon = self.horizon
        self._effective_input_size = effective_input_size
        self.is_fitted_ = True
        logger.info("NHiTS fitted.")
        return self

    def _build_nhits_model(self, h: int, input_size: int):
        """Build an NHiTS model, robust to neuralforecast API changes across versions.

        ``n_stacks`` was removed in neuralforecast ≥ 2.0 (replaced by
        ``stack_types``).  We inspect the NHITS constructor at runtime so
        the code works on both the old and the new API without a hard version pin.

        ``scaler_type`` is forwarded to NeuralForecast when the installed
        version supports it (≥ 1.7).  Normalising the target series internally
        before gradient updates is critical: raw session counts (range 0–200)
        produce gradients ~100× larger than normalised targets, causing erratic
        training dynamics.

        Parameters
        ----------
        h : int
            Forecast horizon (days).
        input_size : int
            Context window length (already capped by the caller).

        Returns
        -------
        NHITS
            Configured model instance, not yet fitted.
        """
        import inspect

        from neuralforecast.models import NHITS

        sig = inspect.signature(NHITS.__init__)
        kwargs: dict = {
            "h": h,
            "input_size": input_size,
            "max_steps": self.max_steps,
            "random_seed": self.random_state,
        }
        if "n_stacks" in sig.parameters:
            kwargs["n_stacks"] = self.n_stacks          # old API (< 2.0)
        elif "stack_types" in sig.parameters:
            kwargs["stack_types"] = ["identity"] * self.n_stacks  # new API (≥ 2.0)

        # Pass scaler_type only when the installed version supports it (≥ 1.7)
        if "scaler_type" in sig.parameters:
            kwargs["scaler_type"] = self.scaler_type

        return NHITS(**kwargs)

    def predict(
        self,
        horizon: int = 14,
        n_scenarios: int = 1,
        start_date: str | pd.Timestamp | None = None,
        seed: int | None = None,
    ) -> pd.DataFrame:
        """
        Predict daily session counts.

        If ``horizon`` differs from the training horizon the model is refitted.

        Parameters
        ----------
        horizon : int
        n_scenarios : int
            Number of stochastic scenarios.
        start_date : str or Timestamp, optional
        seed : int, optional

        Returns
        -------
        pd.DataFrame
            Columns: date, scenario, n_sessions.

        Note
        ----
        The output column is ``n_sessions`` (integer session counts).
        To compare with energy targets (kWh), multiply by the mean energy
        per session from the training data::

            mean_kwh = train_df["energy"].mean()
            pivot_energy = pivot_nhits * mean_kwh
        """
        if not self.is_fitted_:
            raise RuntimeError("Call .fit() before predict().")

        if horizon != self._fitted_horizon:
            logger.warning(
                "Requested horizon %d != fitted horizon %d; refitting.",
                horizon, self._fitted_horizon,
            )
            self.horizon = horizon
            self._fitted_horizon = horizon
            n = len(self._daily_counts)
            effective_input_size = min(self.input_size, max(n // 2, horizon + 1))
            train_df = pd.DataFrame({
                "unique_id": "series_0",
                "ds": self._daily_counts.index,
                "y": self._daily_counts.values.astype(float),
            })
            model = self._build_nhits_model(
                h=horizon,
                input_size=effective_input_size,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                import logging as _logging
                _logging.getLogger("pytorch_lightning").setLevel(_logging.ERROR)
                from neuralforecast import NeuralForecast
                self._nf = NeuralForecast(models=[model], freq="D")
                self._nf.fit(train_df)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw_pred = self._nf.predict()

        fc_values = raw_pred["NHITS"].values
        fc_dates = pd.to_datetime(raw_pred["ds"].values)

        if start_date is not None:
            req_start = pd.Timestamp(start_date)
            if req_start != fc_dates[0]:
                fc_dates = pd.date_range(req_start, periods=len(fc_dates), freq="D")

        fc_dates = fc_dates[:horizon]
        fc_values = fc_values[:horizon]

        rng = np.random.default_rng(seed)
        noise_scale = max(self.std_daily_ * 0.08, 0.5)

        matrix = np.stack([
            np.maximum(0, fc_values + rng.normal(0, noise_scale, len(fc_values)))
            for _ in range(n_scenarios)
        ])
        return _make_forecast_df(pd.DatetimeIndex(fc_dates), matrix)

    def plot_forecast(self, horizon: int = 14, n_scenarios: int = 50,
                      seed: int = 0, ax=None, figsize: tuple = (12, 4)):
        """Plot historical counts and a fan-chart forecast.

        Parameters
        ----------
        horizon : int
            Number of forecast days to display.
        n_scenarios : int
        seed : int
        ax : matplotlib.axes.Axes, optional
        figsize : tuple

        Returns
        -------
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt
        forecast = self.predict(horizon=horizon, n_scenarios=n_scenarios, seed=seed)
        pivot = forecast.pivot(index="date", columns="scenario", values="n_sessions")

        fig, ax_ = (None, ax) if ax is not None else plt.subplots(figsize=figsize)
        ax_ = ax_ or plt.gca()

        if self._daily_counts is not None:
            self._daily_counts.plot(ax=ax_, label="Historical", color="#2E86AB", linewidth=1.2)
        ax_.fill_between(pivot.index, pivot.quantile(0.1, axis=1),
                         pivot.quantile(0.9, axis=1), alpha=0.2, color="#F18F01", label="80% CI")
        pivot.median(axis=1).plot(ax=ax_, label="NHiTS median",
                                  color="#F18F01", linewidth=2)
        ax_.set_ylabel("Daily sessions")
        ax_.set_title("NHiTS session-count forecast")
        ax_.legend()
        ax_.grid(True, alpha=0.3)
        if fig:
            fig.tight_layout()
        return ax_

    def __repr__(self) -> str:
        return (
            f"NHiTSForecaster(horizon={self.horizon}, "
            f"input_size={self.input_size}, "
            f"max_steps={self.max_steps}, fitted={self.is_fitted_})"
        )

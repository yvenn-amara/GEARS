from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from gears.models.forecaster import _make_forecast_df, sessions_to_daily_counts

logger = logging.getLogger(__name__)


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
        Number of past days used as input context. Defaults to 4 Ã— horizon.
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

    âš  SCOPE CONSTRAINT
    -------------------
    This forecaster predicts **session counts** (``n_sessions``), NOT energy.
    It must be fitted on the **same department scope** as the actual counts
    it will be evaluated against.  Fitting on multi-department data and
    evaluating against a single department produces ~N_deptÃ— systematic bias.

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
        compute the loss â€” effectively <1 full training epoch of gradient
        signal, producing random-walk behaviour.

        The effective value is additionally hard-capped at ``n_train // 2``
        inside ``fit()`` to ensure at least ``n_train // 2`` usable
        training windows.
    max_steps : int
        Training gradient steps.

        **Default: 200** (reduced from 300).  On small datasets
        (â‰¤600 days), 300 steps combined with a large input_size caused
        overfitting.  Use 500+ only when ``n_train â‰¥ 2000``.
    n_stacks : int
        Number of stacked NHiTS blocks.
    scaler_type : str
        Internal NeuralForecast scaler applied to ``y`` before training.
        Default ``"standard"`` (zero-mean, unit-variance).  Set to
        ``"robust"`` for heavy-tailed count data.  Set to ``"identity"``
        to disable.  Normalisation is critical: without it, raw session
        counts (range 0â€“200) produce erratic gradients on default
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
        scaler_type: str = "standard",  # Explicitly set: normalisation is critical for raw count data (range 0â€“200)
        random_state: int = 42,
    ):
        self.horizon = horizon
        # Default context window is 2 Ã— horizon (down from 4 Ã—): the larger
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

        âš  No-leakage contract
        ----------------------
        ``df`` must contain ONLY training data (i.e. sessions strictly before
        the evaluation start date).  No shuffling is applied; the temporal
        ordering is preserved as-is.

        Parameters
        ----------
        df : pd.DataFrame
            Validated sessions DataFrame (output of load_sessions).
            Must cover the same dÃ©partement scope as the actual counts
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
        # windows â€” less than one full pass of gradient signal, causing the
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

        ``n_stacks`` was removed in neuralforecast â‰¥ 2.0 (replaced by
        ``stack_types``).  We inspect the NHITS constructor at runtime so
        the code works on both the old and the new API without a hard version pin.

        ``scaler_type`` is forwarded to NeuralForecast when the installed
        version supports it (â‰¥ 1.7).  Normalising the target series internally
        before gradient updates is critical: raw session counts (range 0â€“200)
        produce gradients ~100Ã— larger than normalised targets, causing erratic
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
            kwargs["stack_types"] = ["identity"] * self.n_stacks  # new API (â‰¥ 2.0)

        # Pass scaler_type only when the installed version supports it (â‰¥ 1.7)
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

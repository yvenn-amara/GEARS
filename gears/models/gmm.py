"""
Gaussian Mixture Model for EV session (arrival_hour, duration, energy) triplet.

The model fits a joint GMM over the 3-dimensional feature space
[hour_of_day, log(1+duration), log(1+energy)] with optional conditioning
on categorical context variables.

Key design choices
------------------
- log1p-transform duration and energy for better Gaussian fit
- separate GMMs per context group (configurable via stratify_by)
- BIC-based automatic component selection
- full covariance matrices to capture correlations

Default stratification
----------------------
The canonical GEARS stratification for the French national dataset is:
    ['location_type', 'department', 'day_of_week', 'season']
producing up to 4 × 101 × 7 × 4 = 11 312 strata.

A single ``EVSessionGMM`` object covers all location types; the
``location_type`` dimension is just another context key.  Retrieve strata via:

    gears.get_gmm("work", "75", "winter", 0)

Pre-fitted GMMs
---------------
The package ships with a single pre-fitted GMM bundle ``gmm_french.joblib``
in gears/data/gmm/.  Use ``NativeGMMRegistry`` to load it without raw data.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from gears.data.schemas import _season

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _add_context(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure context columns (day_of_week, season, is_weekend) are present.

    Parameters
    ----------
    df : pd.DataFrame
        Sessions DataFrame; must contain an ``arrival_time`` column.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with ``day_of_week``, ``season``, and ``is_weekend``
        columns added when absent.
    """
    df = df.copy()
    if "day_of_week" not in df.columns:
        df["day_of_week"] = df["arrival_time"].dt.dayofweek
    if "season" not in df.columns:
        df["season"] = df["arrival_time"].dt.month.apply(_season)
    if "is_weekend" not in df.columns:
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    return df


def _features(df: pd.DataFrame) -> np.ndarray:
    """
    Extract the feature matrix [hour, log1p(duration), log1p(energy)].

    Why log-transform duration and energy but NOT arrival hour?
    -----------------------------------------------------------
    Gaussian Mixture Models assume that each component follows a multivariate
    Normal distribution.  This assumption is best met when the marginal
    distributions of each feature are roughly bell-shaped.

    - **duration** and **energy** are strictly positive and right-skewed
      (many short/low-energy sessions, a long tail of long/high-energy ones).
      The log1p transform (log(1 + x)) compresses the tail, making the
      distribution much more symmetric and Gaussian-like.  This substantially
      improves the GMM fit quality (lower BIC, more stable covariance estimates).

    - **arrival hour** already lives on a bounded [0, 24) interval and its
      marginal distribution is naturally multimodal but not skewed in the
      same way.  Transforming it would distort the temporal interpretation
      (e.g. the difference between 8h and 9h should remain 1 hour) and would
      not improve the Gaussian fit.  We therefore leave it in its natural unit.

    Back-transformation at sampling time: duration = expm1(raw[:, 1]),
    energy = expm1(raw[:, 2]).

    Parameters
    ----------
    df : pd.DataFrame
        Sessions DataFrame with ``hour``, ``duration``, and ``energy`` columns.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_samples, 3)`` containing
        ``[hour, log1p(duration), log1p(energy)]``.
    """
    hour = df["hour"].values
    log_dur = np.log1p(df["duration"].values)
    log_ene = np.log1p(df["energy"].values)
    return np.column_stack([hour, log_dur, log_ene])


# ---------------------------------------------------------------------------
# Main EVSessionGMM class
# ---------------------------------------------------------------------------

class EVSessionGMM:
    """
    Joint Gaussian Mixture Model over (arrival_hour, duration, energy).

    Parameters
    ----------
    n_components : int or 'auto'
        Number of Gaussian components per context. If 'auto', BIC selects
        the best n in [min_components, max_components].
    min_components : int
        Minimum components when n_components='auto'.
    max_components : int
        Maximum components when n_components='auto'.
    covariance_type : str
        GMM covariance type: 'full' | 'tied' | 'diag' | 'spherical'.
    stratify_by : list[str]
        Context columns to split data on before fitting separate GMMs.
        Default: ['day_of_week', 'season'].
        For department-level: ['department', 'day_of_week', 'season'].
    max_samples_per_context : int or None
        Maximum training samples per context group.  Stratified subsampling
        is applied when a group exceeds this limit.
        Recommended: 5_000 for large datasets.
    recent_months : int or None
        If set, only the most recent N months of data are kept before
        subsampling.  Useful when the distribution has shifted over time.
    random_state : int
        Random seed for reproducibility.

    Attributes
    ----------
    models_ : dict
        Fitted GaussianMixture objects keyed by context tuple.
    n_sessions_per_day_ : dict
        Average sessions per day per context group.
    is_fitted_ : bool
    context_counts_ : dict
        Training sample counts per context.
    is_sample_ : bool
        True if this GMM was fitted on a data subset (provisional).
    """

    def __init__(
        self,
        n_components: int | str = "auto",
        min_components: int = 2,
        max_components: int = 10,
        covariance_type: str = "full",
        stratify_by: list[str] | None = None,
        max_samples_per_context: int | None = None,
        recent_months: int | None = None,
        random_state: int = 42,
        # VAE-specific parameters (used when model_type="vae")
        model_type: Literal["gmm", "vae"] = "gmm",
        vae_latent_dim: int = 16,
        vae_hidden_dim: int = 256,
        vae_n_layers: int = 2,
        vae_epochs: int = 50,
        vae_batch_size: int = 512,
        vae_lr: float = 3e-3,
        vae_beta: float = 1.0,
        vae_score_n_samples: int = 20,
    ):
        self.n_components = n_components
        self.min_components = min_components
        self.max_components = max_components
        self.covariance_type = covariance_type
        self.stratify_by = stratify_by if stratify_by is not None else ["day_of_week", "season"]
        self.max_samples_per_context = max_samples_per_context
        self.recent_months = recent_months
        self.random_state = random_state
        self.model_type = model_type
        self.vae_latent_dim = vae_latent_dim
        self.vae_hidden_dim = vae_hidden_dim
        self.vae_n_layers = vae_n_layers
        self.vae_epochs = vae_epochs
        self.vae_batch_size = vae_batch_size
        self.vae_lr = vae_lr
        self.vae_beta = vae_beta
        self.vae_score_n_samples = vae_score_n_samples

        self.models_: dict[tuple, GaussianMixture] = {}
        self.n_sessions_per_day_: dict[tuple, float] = {}
        self.context_counts_: dict[tuple, int] = {}
        self.is_fitted_: bool = False
        self.is_sample_: bool = False   # True = fitted on data subset
        self.metadata_: dict = {}

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        df: pd.DataFrame,
        is_sample: bool = False,
        metadata: dict | None = None,
    ) -> EVSessionGMM:
        """
        Fit GMM(s) on a validated EV sessions DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Output of load_sessions() or validate_dataframe().
        is_sample : bool
            Flag that this GMM was fitted on a data subset (not the full dataset).
        metadata : dict, optional
            Arbitrary metadata to embed in the model.

        Returns
        -------
        EVSessionGMM
            The fitted model (``self``), allowing method chaining.
        """
        df = _add_context(df)
        self.is_sample_ = is_sample
        self.metadata_ = metadata or {}

        # Optional: keep only the most recent N months
        if self.recent_months is not None:
            df["_arrival_dt"] = pd.to_datetime(df["arrival_time"])
            cutoff = df["_arrival_dt"].max() - pd.DateOffset(months=self.recent_months)
            n_before = len(df)
            df = df[df["_arrival_dt"] >= cutoff].reset_index(drop=True)
            df.drop(columns=["_arrival_dt"], inplace=True)
            logger.info(
                "recent_months=%d: kept %d/%d sessions.",
                self.recent_months, len(df), n_before,
            )

        # Check that all stratify_by columns exist
        missing_cols = [c for c in self.stratify_by if c not in df.columns]
        if missing_cols:
            logger.warning(
                "Stratification columns %s not in data. "
                "Falling back to ['day_of_week', 'season'].",
                missing_cols,
            )
            self.stratify_by = ["day_of_week", "season"]

        groups = df.groupby(self.stratify_by, observed=True)

        retained_groups: list[tuple[tuple, pd.DataFrame]] = []

        for ctx_key, group_df in groups:
            ctx_tuple = ctx_key if isinstance(ctx_key, tuple) else (ctx_key,)
            n = len(group_df)
            if n < 10:
                warnings.warn(
                    f"Context {ctx_tuple} has only {n} samples – skipping.",
                    UserWarning,
                    stacklevel=2,
                )
                continue

            # Optional stratified subsampling
            if self.max_samples_per_context is not None and n > self.max_samples_per_context:
                rng_sub = np.random.default_rng(self.random_state)
                idx = rng_sub.choice(n, size=self.max_samples_per_context, replace=False)
                group_df = group_df.iloc[idx].reset_index(drop=True)
                n = self.max_samples_per_context

            if self.model_type == "vae":
                # Defer fitting; just collect groups
                retained_groups.append((ctx_tuple, group_df))
                # Still track counts (needed by aggregator / medium_term)
                n_days = group_df["date"].nunique() if "date" in group_df.columns else 1
                self.n_sessions_per_day_[ctx_tuple] = n / max(n_days, 1)
                self.context_counts_[ctx_tuple] = n
                continue

            X = _features(group_df)
            gmm = self._fit_single(X)
            if gmm is None:
                continue

            self.models_[ctx_tuple] = gmm

            n_days = group_df["date"].nunique() if "date" in group_df.columns else 1
            self.n_sessions_per_day_[ctx_tuple] = n / max(n_days, 1)
            self.context_counts_[ctx_tuple] = n

            logger.info(
                "Fitted GMM(k=%d) for context %s on %d samples.",
                gmm.n_components, ctx_tuple, n,
            )

        # VAE path: train a single shared model across all collected groups
        if self.model_type == "vae":
            if not retained_groups:
                raise RuntimeError("No contexts with enough samples for VAE. Check data quality.")
            self._fit_vae(df, retained_groups)

        if not self.models_:
            raise RuntimeError("No GMM was fitted. Check data quality.")

        self.is_fitted_ = True
        return self

    # ------------------------------------------------------------------
    # VAE fitting helpers
    # ------------------------------------------------------------------

    def _fit_vae(
        self,
        df: pd.DataFrame,
        retained_groups: list[tuple[tuple, pd.DataFrame]],
    ) -> None:
        """
        Train a single shared ConditionalVAE across all retained context groups,
        then populate self.models_ with one VAEContextSlice per context.

        Parameters
        ----------
        df : pd.DataFrame
            Full (possibly subsampled) sessions data – not used directly here;
            retained_groups already contains the per-context slices.
        retained_groups : list of (ctx_tuple, group_df)
            Pre-filtered groups (n >= 10, subsampled if needed).
        """
        from sklearn.preprocessing import StandardScaler

        from gears.models.vae import (
            ConditionalVAE,
            ContextEncoder,
            VAEContextSlice,
            train_cvae,
        )

        logger.info(
            "Fitting shared CVAE across %d contexts (latent_dim=%d, hidden=%d, epochs=%d).",
            len(retained_groups), self.vae_latent_dim, self.vae_hidden_dim, self.vae_epochs,
        )

        # ── Build vocabulary for context dimensions ───────────────────
        all_ctx_keys = [ctx for ctx, _ in retained_groups]
        ctx_enc = ContextEncoder(self.stratify_by)
        ctx_enc.fit(all_ctx_keys)

        # Embedding dims: sqrt rule, minimum 4
        emb_dims = [max(4, int(np.ceil(np.sqrt(d)))) for d in ctx_enc.context_dims]

        # ── Assemble combined training dataset ────────────────────────
        X_parts: list[np.ndarray] = []
        ctx_parts: list[np.ndarray] = []
        for ctx_tuple, group_df in retained_groups:
            X_g = _features(group_df)
            ctx_idx = ctx_enc.encode_batch([ctx_tuple] * len(group_df))
            X_parts.append(X_g)
            ctx_parts.append(ctx_idx)

        X_all = np.vstack(X_parts)
        ctx_all = np.vstack(ctx_parts)

        # ── Standardise features ──────────────────────────────────────
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_all)

        # ── Build and train CVAE ──────────────────────────────────────
        cvae = ConditionalVAE(
            context_dims=ctx_enc.context_dims,
            emb_dims=emb_dims,
            feature_dim=3,
            latent_dim=self.vae_latent_dim,
            hidden_dim=self.vae_hidden_dim,
            n_layers=self.vae_n_layers,
        )
        train_cvae(
            cvae,
            X=X_scaled,
            ctx_indices=ctx_all,
            epochs=self.vae_epochs,
            batch_size=self.vae_batch_size,
            lr=self.vae_lr,
            beta=self.vae_beta,
            seed=self.random_state,
            verbose=True,
        )

        # ── Create one VAEContextSlice per context ────────────────────
        for ctx_tuple, group_df in retained_groups:
            ctx_index = ctx_enc.encode(ctx_tuple)  # (1, n_dims)
            slice_ = VAEContextSlice(
                cvae=cvae,
                ctx_index=ctx_index,
                scaler_mean=scaler.mean_,
                scaler_std=scaler.scale_,
                score_n_samples=self.vae_score_n_samples,
            )
            self.models_[ctx_tuple] = slice_

            n_days = group_df["date"].nunique() if "date" in group_df.columns else 1
            n = len(group_df)
            self.n_sessions_per_day_[ctx_tuple] = n / max(n_days, 1)
            self.context_counts_[ctx_tuple] = n

        logger.info("CVAE fitted; %d context slices created.", len(self.models_))

    def _fit_single(self, X: np.ndarray) -> GaussianMixture | None:
        """Fit one GMM, optionally selecting n_components via BIC.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape ``(n_samples, 3)``.

        Returns
        -------
        GaussianMixture or None
            Fitted sklearn GMM, or ``None`` if fitting failed for all
            candidate component counts.
        """
        if self.n_components == "auto":
            best_gmm, best_bic = None, np.inf
            k_max = min(self.max_components, len(X) // 5)
            for k in range(self.min_components, max(k_max + 1, self.min_components + 1)):
                try:
                    gmm = GaussianMixture(
                        n_components=k,
                        covariance_type=self.covariance_type,
                        random_state=self.random_state,
                        n_init=3,
                        max_iter=300,
                        reg_covar=1e-5,
                    ).fit(X)
                    bic = gmm.bic(X)
                    if bic < best_bic:
                        best_bic, best_gmm = bic, gmm
                except Exception as e:  # noqa: BLE001 - skip this k on any GaussianMixture fit failure
                    logger.debug("GMM k=%d failed: %s", k, e)
                    continue
            return best_gmm
        else:
            try:
                return GaussianMixture(
                    n_components=int(self.n_components),
                    covariance_type=self.covariance_type,
                    random_state=self.random_state,
                    n_init=3,
                    max_iter=300,
                    reg_covar=1e-5,
                ).fit(X)
            except Exception as e:  # noqa: BLE001 - fall back to None on any GaussianMixture fit failure
                logger.warning("GMM fitting failed: %s", e)
                return None

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(
        self,
        n_sessions: int,
        context: dict | None = None,
        date: str | pd.Timestamp | None = None,
        seed: int | None = None,
    ) -> pd.DataFrame:
        """
        Generate synthetic EV sessions.

        Parameters
        ----------
        n_sessions : int
            Number of sessions to generate.
        context : dict, optional
            Explicit context, e.g. {'day_of_week': 1, 'season': 'winter'}.
            If None, inferred from ``date``.
        date : str or Timestamp, optional
            Reference date used to infer context if context is None.
        seed : int, optional
            Random seed.

        Returns
        -------
        pd.DataFrame
            Columns: arrival_hour, duration, energy, [arrival_time].
        """
        self._check_fitted()
        ctx_tuple = self._resolve_context(context, date)
        gmm = self._get_model(ctx_tuple)

        rng = np.random.default_rng(seed)
        gmm.random_state = int(rng.integers(0, 2**31))

        raw, _ = gmm.sample(n_sessions)
        return self._raw_to_sessions(raw, date=date, seed=seed)

    def _raw_to_sessions(
        self,
        raw: np.ndarray,
        date: str | pd.Timestamp | None = None,
        seed: int | None = None,
    ) -> pd.DataFrame:
        """Convert raw GMM samples [hour, log_dur, log_ene] to a sessions DataFrame.

        Applies ``expm1`` back-transformation and clips values to physically
        plausible bounds (arrival hour in [0, 24), duration in [0.08, 48] h,
        energy in [0.01, 350] kWh).

        Parameters
        ----------
        raw : np.ndarray
            Array of shape ``(n_samples, 3)`` in GMM feature space.
        date : str or Timestamp, optional
            If given, an ``arrival_time`` column is added anchored to this date.
        seed : int, optional
            Unused; kept for API symmetry with :meth:`sample`.

        Returns
        -------
        pd.DataFrame
            Columns: arrival_hour, duration, energy, [arrival_time].
        """
        hour = np.clip(raw[:, 0], 0, 23.99)
        duration = np.clip(np.expm1(raw[:, 1]), 0.08, 48.0)
        energy = np.clip(np.expm1(raw[:, 2]), 0.01, 350.0)

        sessions = pd.DataFrame(
            {"arrival_hour": hour, "duration": duration, "energy": energy}
        )

        if date is not None:
            ref = pd.Timestamp(date)
            td = pd.to_timedelta(sessions["arrival_hour"], unit="h")
            sessions["arrival_time"] = ref.normalize() + td

        return sessions

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(self, df: pd.DataFrame) -> float:
        """Return mean log-likelihood per sample on a validated DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Validated sessions DataFrame with the same columns used during
            ``fit()``.

        Returns
        -------
        float
            Mean per-sample log-likelihood across all contexts present in
            *df*.  Returns ``-inf`` when no fitted context matches.
        """
        self._check_fitted()
        df = _add_context(df)
        scores = []
        for ctx_key, group_df in df.groupby(self.stratify_by, observed=True):
            ctx_tuple = ctx_key if isinstance(ctx_key, tuple) else (ctx_key,)
            if ctx_tuple not in self.models_:
                continue
            X = _features(group_df)
            scores.extend(self.models_[ctx_tuple].score_samples(X).tolist())
        return float(np.mean(scores)) if scores else float("-inf")

    def bic_summary(self) -> pd.DataFrame:
        """Return BIC and n_components per context group.

        Returns
        -------
        pd.DataFrame
            One row per fitted context with columns:
            ``context``, ``n_components``, ``n_samples``.
        """
        rows = []
        for ctx, gmm in self.models_.items():
            rows.append({
                "context": ctx,
                "n_components": gmm.n_components,
                "n_samples": self.context_counts_.get(ctx, -1),
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Expose underlying sklearn objects
    # ------------------------------------------------------------------

    def get_sklearn_gmm(
        self,
        context: dict | None = None,
        date: str | pd.Timestamp | None = None,
    ) -> GaussianMixture:
        """
        Return the underlying sklearn GaussianMixture for a given context.

        This provides full access to all sklearn attributes:
        means_, covariances_, weights_, predict(), score_samples(), etc.

        Parameters
        ----------
        context : dict, optional
            Context dict, e.g. {'day_of_week': 0, 'season': 'winter'}.
        date : str or Timestamp, optional
            Date to infer context from.

        Returns
        -------
        sklearn.mixture.GaussianMixture
            The fitted GaussianMixture for the resolved context.
        """
        self._check_fitted()
        ctx_tuple = self._resolve_context(context, date)
        return self._get_model(ctx_tuple)

    def list_contexts(self) -> list[tuple]:
        """Return all fitted context tuples.

        Returns
        -------
        list[tuple]
            Each tuple corresponds to one fitted stratum, with values ordered
            according to ``stratify_by``.
        """
        return list(self.models_.keys())

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Serialize model to a joblib file.

        Parameters
        ----------
        path : str or Path
            Destination file path. Parent directories are created if missing.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("GMM saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> EVSessionGMM:
        """Load a serialised GMM from disk.

        Parameters
        ----------
        path : str or Path
            Path to a ``.joblib`` file previously saved with :meth:`save`.

        Returns
        -------
        EVSessionGMM
            The loaded and fitted model.

        Raises
        ------
        TypeError
            If the file contains an object that is not an ``EVSessionGMM``.
        """
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj)}")
        return obj

    # ------------------------------------------------------------------
    # Diagnostics / plots
    # ------------------------------------------------------------------

    def plot_components(
        self,
        context: dict | None = None,
        date: str | None = None,
        ax=None,
        figsize: tuple = (8, 4),
    ):
        """
        Plot GMM component weights for a given context.

        Parameters
        ----------
        context : dict, optional
            Context dict passed to :meth:`_resolve_context`.
        date : str, optional
            Reference date used to infer context when *context* is None.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. Created if None.
        figsize : tuple
            Figure size in inches, used only when creating a new figure.

        Returns
        -------
        matplotlib.axes.Axes
            The axes containing the bar chart.
        """
        import matplotlib.pyplot as plt

        self._check_fitted()
        ctx_tuple = self._resolve_context(context, date)
        gmm = self._get_model(ctx_tuple)
        weights = gmm.weights_
        k = len(weights)

        fig, ax_ = (None, ax) if ax is not None else plt.subplots(figsize=figsize)
        ax_ = ax_ or plt.gca()

        colors = plt.cm.Blues(np.linspace(0.4, 0.9, k))
        bars = ax_.bar(range(k), weights, color=colors, edgecolor="white")
        ax_.set_xlabel("Component")
        ax_.set_ylabel("Weight")
        ax_.set_title(f"GMM component weights – context {ctx_tuple}")
        ax_.set_xticks(range(k))

        for bar, w in zip(bars, weights):
            ax_.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{w:.2f}",
                ha="center", va="bottom", fontsize=9,
            )
        if fig:
            fig.tight_layout()
        return ax_

    def plot_marginals(
        self,
        df: pd.DataFrame | None = None,
        context: dict | None = None,
        date: str | None = None,
        n_samples: int = 2000,
        bins: int = 40,
        figsize: tuple = (14, 4),
    ):
        """
        Side-by-side histograms of real vs. sampled sessions.

        Parameters
        ----------
        df : pd.DataFrame, optional
            Real data to compare against.
        context : dict, optional
            Context dict for sampling.
        date : str, optional
            Reference date used to infer context.
        n_samples : int
            Number of synthetic sessions to generate for comparison.
        bins : int
            Number of histogram bins.
        figsize : tuple
            Figure size in inches.

        Returns
        -------
        matplotlib.figure.Figure
            The figure containing the three marginal histograms.
        """
        import matplotlib.pyplot as plt

        self._check_fitted()
        synth = self.sample(n_samples, context=context, date=date)

        fig, axes = plt.subplots(1, 3, figsize=figsize)
        pairs = [
            ("arrival_hour", "Arrival hour"),
            ("duration", "Duration (h)"),
            ("energy", "Energy (kWh)"),
        ]

        for ax, (col, label) in zip(axes, pairs):
            ax.hist(synth[col], bins=bins, alpha=0.6, label="Simulated", density=True,
                    color="#2E86AB")
            if df is not None:
                real_col = "hour" if col == "arrival_hour" and "hour" in df.columns else col
                if real_col in df.columns:
                    ax.hist(df[real_col], bins=bins, alpha=0.6, label="Real",
                            density=True, color="#E84855")
            ax.set_xlabel(label)
            ax.set_ylabel("Density")
            ax.legend()
            ax.grid(True, alpha=0.3)

        ctx_tuple = self._resolve_context(context, date)
        fig.suptitle(f"GMM marginal distributions – context {ctx_tuple}")
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        """Raise RuntimeError if the model has not been fitted yet."""
        if not self.is_fitted_:
            raise RuntimeError("Call .fit() before using this method.")

    def _resolve_context(
        self,
        context: dict | None,
        date: str | pd.Timestamp | None,
    ) -> tuple:
        """Return a context tuple matching the stratify_by keys.

        Resolution order:
        1. Explicit *context* dict.
        2. Date-derived context (day_of_week, season, is_weekend).
        3. Most populous context as fallback.

        Parameters
        ----------
        context : dict or None
        date : str, Timestamp, or None

        Returns
        -------
        tuple
            Values ordered according to ``stratify_by``.
        """
        if context is not None:
            return tuple(context.get(k) for k in self.stratify_by)

        if date is not None:
            ts = pd.Timestamp(date)
            ctx: dict = {}
            if "day_of_week" in self.stratify_by:
                ctx["day_of_week"] = ts.dayofweek
            if "is_weekend" in self.stratify_by:
                ctx["is_weekend"] = int(ts.dayofweek >= 5)
            if "season" in self.stratify_by:
                ctx["season"] = _season(ts.month)
            # location_type cannot be inferred from a date alone; it is left
            # as None so the fallback chain in _get_model can handle it.
            return tuple(ctx.get(k) for k in self.stratify_by)

        # Fallback: most populous context
        if self.context_counts_:
            return max(self.context_counts_, key=lambda k: self.context_counts_[k])
        return next(iter(self.models_))

    def _get_model(self, ctx_tuple: tuple) -> GaussianMixture:
        """Return GMM for context, with smart fallback chain.

        The fallback tries progressively shorter prefix matches before
        settling on the globally most-populous model.  This handles rare
        contexts (e.g. uncommon departments) gracefully without raising.

        Parameters
        ----------
        ctx_tuple : tuple
            Context tuple produced by :meth:`_resolve_context`.

        Returns
        -------
        GaussianMixture
            The best available fitted model for the requested context.
        """
        if ctx_tuple in self.models_:
            return self.models_[ctx_tuple]

        # Try partial matches (ignore department, then ignore season)
        for width in range(len(ctx_tuple) - 1, 0, -1):
            prefix = ctx_tuple[:width]
            for key in self.models_:
                if key[:width] == prefix:
                    logger.debug(
                        "Context %s not found; falling back to %s.", ctx_tuple, key
                    )
                    return self.models_[key]

        # Last resort: most populous model
        best = max(self.context_counts_, key=lambda k: self.context_counts_[k])
        logger.debug("Context %s not found; using global fallback %s.", ctx_tuple, best)
        return self.models_[best]

    def __repr__(self) -> str:
        if not self.is_fitted_:
            return (
                f"EVSessionGMM(n_components={self.n_components!r}, "
                f"stratify_by={self.stratify_by}, fitted=False)"
            )
        extras = []
        if self.is_sample_:
            extras.append("is_sample=True")
        if self.max_samples_per_context is not None:
            extras.append(f"max_samples={self.max_samples_per_context}")
        extra_str = (", " + ", ".join(extras)) if extras else ""
        return (
            f"EVSessionGMM("
            f"n_contexts={len(self.models_)}, "
            f"stratify_by={self.stratify_by}, "
            f"n_components={self.n_components!r}"
            f"{extra_str}, fitted=True)"
        )

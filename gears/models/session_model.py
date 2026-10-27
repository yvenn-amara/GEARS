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

A single ``EVSessionModel`` object covers all location types; the
``location_type`` dimension is just another context key.  Retrieve strata via:

    model = load_pretrained("french")
    sk = model.get_sklearn_component(
        context={"location_type": "work", "department": "75",
                 "season": "winter", "day_of_week": 0}
    )

Pre-fitted GMMs
---------------
The package ships with a single pre-fitted GMM bundle ``gmm_french.joblib``.
Use ``gears.load_pretrained("french")`` to load it without raw data.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from gears._validation import as_naive_timestamp, check_int, not_fitted_error
from gears.data.schemas import _season

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Recency-weighting defaults (see EVSessionModel's ``recency`` parameter)
# ---------------------------------------------------------------------------
# Cap on the size of the weighted-bootstrap resample used to approximate a
# recency-weighted GMM fit (sklearn's GaussianMixture has no sample_weight
# argument). Kept as a named constant rather than a literal buried in
# _recency_resample so the performance/fidelity trade-off is easy to find
# and override.
DEFAULT_RECENCY_RESAMPLE_CAP = 5000

# Default divisor used to derive half_life_days from a context group's
# observed history span (in days) when half_life_days is not given
# explicitly: half_life_days = span_days / DEFAULT_RECENCY_HALFLIFE_DIVISOR.
DEFAULT_RECENCY_HALFLIFE_DIVISOR = 3.5

# A context (one combination of the stratify_by columns) needs at least this many
# sessions to get its own model; smaller ones are served by the closest fitted
# context (see EVSessionModel.skipped_contexts_). The same number is the minimum
# size of the whole dataset.
MIN_SESSIONS_PER_CONTEXT = 10
MIN_SESSIONS_TOTAL = MIN_SESSIONS_PER_CONTEXT


def _library_versions() -> dict[str, str]:
    """Versions of GEARS and scikit-learn, stored in ``EVSessionModel.metadata_``."""
    import sklearn

    from gears import __version__

    return {"gears_version": __version__, "sklearn_version": sklearn.__version__}


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _add_context(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the derived columns (day_of_week, season, is_weekend, hour, date) are present.

    Parameters
    ----------
    df : pd.DataFrame
        Sessions DataFrame; must contain an ``arrival_time`` column.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with ``day_of_week``, ``season``, ``is_weekend``, ``hour`` (arrival
        time as a decimal hour) and ``date`` columns added when absent, so that a plain
        DataFrame with ``arrival_time``, ``duration`` and ``energy`` can be fitted.
    """
    df = df.copy()
    if "hour" not in df.columns:
        df["hour"] = df["arrival_time"].dt.hour + df["arrival_time"].dt.minute / 60.0
    if "date" not in df.columns:
        df["date"] = df["arrival_time"].dt.date
    if "day_of_week" not in df.columns:
        df["day_of_week"] = df["arrival_time"].dt.day_of_week
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
# Main EVSessionModel class
# ---------------------------------------------------------------------------

class EVSessionModel:
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
    recency : bool or None
        If truthy, fit each context group on a recency-weighted bootstrap
        resample instead of a uniform one. sklearn's ``GaussianMixture`` has
        no ``sample_weight`` argument, so weighting is implemented by
        resampling with replacement from a probability distribution over
        sessions, weight(session) = 0.5 ** (days_since_session /
        half_life_days), then fitting a standard unweighted GMM on the
        resample -- a classic weighted-bootstrap approximation of a
        weighted fit. Default ``None`` (disabled); behavior is then
        byte-for-byte identical to before this parameter existed. Only
        applies to the GMM path (``model_type="gmm"``); ignored (with a
        warning) when ``model_type="vae"``.
        Note: the resample cap (see ``recency_resample_cap``) only bounds
        the *training* sample fed to ``GaussianMixture.fit`` -- it is not
        allowed to distort ``n_sessions_per_day_``, which is computed from
        the true pre-resample pool so downstream simulation still sees the
        real historical session rate rather than a rate implied by however
        small the capped training sample happens to be.
    half_life_days : float or None
        Half-life of the exponential recency decay, only used when
        ``recency`` is truthy. If ``None`` (default), it is derived per
        context group from that group's own observed history span:
        ``half_life_days = span_days / recency_halflife_divisor``.
        Pass an explicit value to override the automatic scaling.
    recency_reference_date : str, pd.Timestamp, or None
        The "as of" date that ``days_since_session`` is measured against.
        If ``None`` (default), the most recent ``arrival_time`` in the
        fitting data is used, i.e. weighting is relative to "today" =
        the last observed session.
    recency_resample_cap : int
        Upper bound on the size of the recency-weighted resample per
        context group (see ``DEFAULT_RECENCY_RESAMPLE_CAP``). For small
        groups the resample size is simply ``n`` (no inflation); this cap
        only matters for large pools (e.g. ~40k sessions at long history
        windows), where resampling to the full pool size would be far
        slower for no fidelity benefit. If ``max_samples_per_context`` is
        also set, the smaller of the two is used.
    recency_halflife_divisor : float
        Divisor used to derive the default ``half_life_days`` from a
        context group's history span in days, when ``half_life_days`` is
        not given explicitly (see ``DEFAULT_RECENCY_HALFLIFE_DIVISOR``).

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
    half_life_days_used_ : dict
        Half-life actually used per context tuple, populated only when
        ``recency`` is enabled (useful for inspecting the automatic scaling).
    recency_reference_date_used_ : pd.Timestamp or None
        The reference date actually used for recency weighting, populated
        only when ``recency`` is enabled.
    skipped_contexts_ : dict or None
        ``{context: n_sessions}`` for the contexts with fewer than
        ``MIN_SESSIONS_PER_CONTEXT`` (10) sessions, which got no model of their
        own and are served by the closest fitted context. ``None`` for bundles
        saved before this was recorded.
    fallback_contexts_ : dict
        ``{requested context: fitted context used}`` for every request that was
        not an exact match, filled as you sample. The first one raises a
        ``UserWarning``; ``sample(..., strict=True)`` raises instead.
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
        # Recency-weighting parameters (opt-in; GMM path only)
        recency: bool | None = None,
        half_life_days: float | None = None,
        recency_reference_date: str | pd.Timestamp | None = None,
        recency_resample_cap: int = DEFAULT_RECENCY_RESAMPLE_CAP,
        recency_halflife_divisor: float = DEFAULT_RECENCY_HALFLIFE_DIVISOR,
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
        self.recency = recency
        self.half_life_days = half_life_days
        self.recency_reference_date = recency_reference_date
        self.recency_resample_cap = recency_resample_cap
        self.recency_halflife_divisor = recency_halflife_divisor

        self.models_: dict[tuple, GaussianMixture] = {}
        self.n_sessions_per_day_: dict[tuple, float] = {}
        self.context_counts_: dict[tuple, int] = {}
        self.is_fitted_: bool = False
        self.is_sample_: bool = False   # True = fitted on data subset
        self.metadata_: dict = {}
        self.half_life_days_used_: dict[tuple, float] = {}
        self.recency_reference_date_used_: pd.Timestamp | None = None
        # context tuple -> number of sessions, for contexts too small to be fitted
        self.skipped_contexts_: dict[tuple, int] | None = {}
        # requested context -> fitted context used instead (filled while sampling)
        self.fallback_contexts_: dict[tuple, tuple] = {}
        self._fallback_warned = False

    def __setstate__(self, state: dict) -> None:
        """Restore instance state from unpickling.

        Older on-disk bundles (e.g. the committed ``gmm_french.joblib``,
        pickled before Session 2 added recency-weighting support) don't have
        the recency-related attributes below in their pickled ``__dict__``.
        Without this, plain attribute access on those attributes elsewhere
        in the class (``__repr__`` in particular: ``if self.recency:``)
        raises ``AttributeError`` on load — surfaced by notebook 1's
        ``load_pretrained()`` call. Backfilling with the same defaults ``__init__``
        uses keeps old bundles loadable under the current code without
        needing to refit and re-commit them.

        ``model_type`` has the same problem: the two oldest release bundles
        (``gmm_french.joblib``, ``gmm_french_sample.joblib``) were pickled
        before that attribute existed at all (confirmed by inspecting their
        unpickled ``__dict__`` directly — the three newer bundles,
        ``gmm_french_holdout.joblib``, ``gmm_french_recency.joblib``, and
        ``vae_french_sample.joblib``, all already have it). Surfaced by
        Session 6's ``plot_marginals()`` fix, which reads ``self.model_type``
        to pick GMM- vs VAE-consistent plot colors and raised
        ``AttributeError`` on the old GMM bundles. Every bundle predating this
        attribute is a GMM (the VAE path didn't exist yet), so ``"gmm"`` is
        the correct backfill default — it's also ``__init__``'s own default.
        """
        defaults = {
            "recency": None,
            "half_life_days": None,
            "recency_reference_date": None,
            "recency_resample_cap": DEFAULT_RECENCY_RESAMPLE_CAP,
            "recency_halflife_divisor": DEFAULT_RECENCY_HALFLIFE_DIVISOR,
            "half_life_days_used_": {},
            "recency_reference_date_used_": None,
            "model_type": "gmm",
            "skipped_contexts_": None,   # not recorded by older bundles
            "fallback_contexts_": {},
            "_fallback_warned": False,
        }
        for key, default in defaults.items():
            state.setdefault(key, default)
        self.__dict__.update(state)

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def _check_n_components(self) -> None:
        """``n_components`` is 'auto' or an integer >= 1: refuse anything else before fitting."""
        n = self.n_components
        if isinstance(n, str) and n == "auto":
            return
        if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
            raise ValueError(
                f"n_components must be 'auto' (choose the number of mixture components "
                f"by BIC) or an integer >= 1, got {n!r}."
            )

    def fit(
        self,
        df: pd.DataFrame,
        is_sample: bool = False,
        metadata: dict | None = None,
    ) -> EVSessionModel:
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
        EVSessionModel
            The fitted model (``self``), allowing method chaining.
        """
        missing = [c for c in ("arrival_time", "duration", "energy") if c not in df.columns]
        if missing:
            raise ValueError(
                f"EVSessionModel.fit needs the columns {missing}. Pass the output of "
                "gears.load_sessions(...) (or validate_dataframe), which also converts "
                "units and parses dates."
            )
        self._check_n_components()
        n_total = len(df)
        if n_total < MIN_SESSIONS_TOTAL:
            raise ValueError(
                f"Cannot fit a session model on {n_total} session(s): at least "
                f"{MIN_SESSIONS_TOTAL} are needed, and each stratification context needs "
                f"{MIN_SESSIONS_PER_CONTEXT} of its own. Provide more data, or check that "
                "your filters did not remove most of it."
            )
        df = _add_context(df)
        self.is_sample_ = is_sample
        # Start from a clean state: refitting must not keep contexts of an earlier fit.
        self.models_ = {}
        self.n_sessions_per_day_ = {}
        self.context_counts_ = {}
        self.skipped_contexts_ = {}
        self.fallback_contexts_ = {}
        self._fallback_warned = False
        # Versions of the libraries the model is pickled with; load_pretrained()
        # warns when scikit-learn differs at load time.
        self.metadata_ = {**(metadata or {}), **_library_versions()}

        # Optional: keep only the most recent N months
        if self.recent_months is not None:
            df["_arrival_dt"] = pd.to_datetime(df["arrival_time"])
            cutoff = df["_arrival_dt"].max() - pd.DateOffset(months=self.recent_months)
            n_before = len(df)
            df = df[df["_arrival_dt"] >= cutoff].reset_index(drop=True)
            df = df.drop(columns=["_arrival_dt"])
            logger.info(
                "recent_months=%d: kept %d/%d sessions.",
                self.recent_months, len(df), n_before,
            )

        # Check that all stratify_by columns exist. Only the columns that are
        # actually absent are dropped — a dataset missing just `location_type`
        # (e.g. a single-site export) still keeps `department` if it has it,
        # rather than silently collapsing all the way down to
        # ['day_of_week', 'season']. This is also the one log statement that
        # reports the *final* stratify_by actually used for fitting, so any
        # caller logging its own pre-fit summary should treat this message as
        # authoritative rather than re-deriving (and potentially
        # contradicting) it independently.
        missing_cols = [c for c in self.stratify_by if c not in df.columns]
        if missing_cols:
            kept_cols = [c for c in self.stratify_by if c not in missing_cols]
            logger.warning(
                "Stratification columns %s not in data. "
                "Stratifying by %s instead.",
                missing_cols, kept_cols,
            )
            self.stratify_by = kept_cols
            if not kept_cols:
                raise ValueError(
                    f"None of the stratify_by columns {missing_cols} exists in the data "
                    f"(columns: {list(df.columns)}). Name at least one existing column, "
                    "e.g. stratify_by=['day_of_week', 'season']."
                )

        # Recency-weighted resampling (opt-in) is only implemented for the
        # GMM path; VAE fitting collects raw groups and trains a single
        # shared model differently, so recency weighting is out of scope
        # here and is ignored (loudly) rather than silently.
        reference_date: pd.Timestamp | None = None
        if self.recency:
            if self.model_type == "vae":
                logger.warning(
                    "recency=%r is set but model_type='vae'; recency weighting "
                    "is only implemented for the GMM path and will be ignored.",
                    self.recency,
                )
            else:
                reference_date = (
                    pd.Timestamp(self.recency_reference_date)
                    if self.recency_reference_date is not None
                    else pd.to_datetime(df["arrival_time"]).max()
                )
                self.recency_reference_date_used_ = reference_date
                self.half_life_days_used_ = {}

        groups = df.groupby(self.stratify_by, observed=True)

        # Contexts too small for a model of their own: one summary, not one warning each.
        sizes = {
            (k if isinstance(k, tuple) else (k,)): int(v) for k, v in groups.size().items()
        }
        self.skipped_contexts_ = {
            k: v for k, v in sizes.items() if v < MIN_SESSIONS_PER_CONTEXT
        }
        if len(self.skipped_contexts_) == len(sizes):
            raise ValueError(
                f"No stratification context has {MIN_SESSIONS_PER_CONTEXT} sessions: the "
                f"{n_total} sessions are spread over {len(sizes)} contexts "
                f"(stratify_by={self.stratify_by}, largest context: {max(sizes.values())}). "
                "Stratify by fewer columns, e.g. stratify_by=['day_of_week'], or provide "
                "more data."
            )
        if self.skipped_contexts_:
            n_skipped = sum(self.skipped_contexts_.values())
            warnings.warn(
                f"{len(self.skipped_contexts_)} of {len(sizes)} contexts have fewer than "
                f"{MIN_SESSIONS_PER_CONTEXT} sessions ({n_skipped:,} sessions, "
                f"{100 * n_skipped / n_total:.1f}% of the data) and get no model of their "
                "own: requests for them use the closest fitted context. To reduce this, "
                "stratify by fewer columns or provide more data; the list is in "
                "model.skipped_contexts_ (and model.summary() when using GEARSModel).",
                UserWarning,
                stacklevel=2,
            )

        retained_groups: list[tuple[tuple, pd.DataFrame]] = []

        for ctx_key, group_df in groups:
            ctx_tuple = ctx_key if isinstance(ctx_key, tuple) else (ctx_key,)
            n = len(group_df)
            if ctx_tuple in self.skipped_contexts_:
                continue

            # Recency-weighted bootstrap resample (opt-in). Replaces the
            # uniform max_samples_per_context subsampling below for this
            # group -- the resample has its own size cap, and drawing with
            # replacement from a recency-weighted distribution already
            # subsumes plain subsampling.
            #
            # IMPORTANT: capture the *true* underlying volume (n_true,
            # n_days_true) before any resampling or subsampling. Those caps
            # exist purely to keep the GMM *fit* tractable; they must not
            # distort the sessions-per-day *rate* used downstream by
            # simulation (aggregator.py / medium_term). A 40k-session pool
            # capped to 5000 for fitting would otherwise undercount the daily
            # volume ~8x, for exactly the high-volume contexts the caps target.
            n_true = n
            n_days_true = group_df["date"].nunique() if "date" in group_df.columns else 1
            if reference_date is not None:
                group_df, n = self._recency_resample(group_df, ctx_tuple, reference_date)
            # Optional stratified subsampling (unweighted)
            elif self.max_samples_per_context is not None and n > self.max_samples_per_context:
                rng_sub = np.random.default_rng(self.random_state)
                idx = rng_sub.choice(n, size=self.max_samples_per_context, replace=False)
                group_df = group_df.iloc[idx].reset_index(drop=True)
                n = self.max_samples_per_context

            if self.model_type == "vae":
                # Defer fitting; just collect groups
                retained_groups.append((ctx_tuple, group_df))
                # Still track counts (needed by aggregator / medium_term)
                self.n_sessions_per_day_[ctx_tuple] = n_true / max(n_days_true, 1)
                self.context_counts_[ctx_tuple] = n
                continue

            X = _features(group_df)
            gmm = self._fit_single(X)
            if gmm is None:
                continue

            self.models_[ctx_tuple] = gmm

            # the true volume, not the (possibly capped) training sample used for X above
            self.n_sessions_per_day_[ctx_tuple] = n_true / max(n_days_true, 1)
            self.context_counts_[ctx_tuple] = n

            logger.debug(
                "Fitted GMM(k=%d) for context %s on %d samples.",
                gmm.n_components, ctx_tuple, n,
            )

        if self.model_type == "gmm":
            logger.info("Fitted %d GMM contexts.", len(self.models_))

        # VAE path: train a single shared model across all collected groups
        if self.model_type == "vae":
            self._fit_vae(df, retained_groups)

        if not self.models_:
            raise RuntimeError(
                f"No mixture model could be fitted ({len(sizes) - len(self.skipped_contexts_)} "
                "contexts were large enough, but every fit failed). The data are probably "
                "degenerate (e.g. constant duration or energy): check them with "
                "gears.data.schemas.summary_stats(df)."
            )

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

    def _recency_resample(
        self,
        group_df: pd.DataFrame,
        ctx_tuple: tuple,
        reference_date: pd.Timestamp,
    ) -> tuple[pd.DataFrame, int]:
        """
        Weighted bootstrap resample favoring recent sessions.

        sklearn's ``GaussianMixture`` has no ``sample_weight`` parameter, so
        half-life recency weighting is implemented via resampling rather than
        a custom EM: each session gets
        ``weight = 0.5 ** (days_since_session / half_life_days)``, weights
        are normalized into a probability distribution over the group's
        sessions, and a new set is drawn *with replacement* from that
        distribution. Recent sessions are drawn often (possibly more than
        once); old sessions rarely or never appear. Fitting a standard,
        unweighted GMM on this resample statistically approximates a true
        weighted fit on the original data (classic weighted bootstrap).

        Parameters
        ----------
        group_df : pd.DataFrame
            One context group's pooled sessions, pre-resample.
        ctx_tuple : tuple
            Context key; used only to record the half-life actually used in
            ``half_life_days_used_`` for diagnostics.
        reference_date : pd.Timestamp
            The "as of" date ``days_since_session`` is measured against.

        Returns
        -------
        (pd.DataFrame, int)
            The resampled group (index reset) and its new size.
        """
        n = len(group_df)
        arrival = pd.to_datetime(group_df["arrival_time"])
        days_since = (reference_date - arrival).dt.total_seconds().to_numpy() / 86400.0

        if self.half_life_days is not None:
            half_life = float(self.half_life_days)
        else:
            span_days = float(days_since.max() - days_since.min()) if n > 1 else 0.0
            half_life = max(1.0, span_days / self.recency_halflife_divisor)
        self.half_life_days_used_[ctx_tuple] = half_life

        # Clip to guard against numerical overflow/underflow at the extremes
        # (e.g. an explicit reference_date earlier than some sessions, or a
        # very short half_life on a very old session); does not affect the
        # normal case where all days_since >= 0.
        weight = np.clip(0.5 ** (days_since / half_life), 1e-300, 1e12)
        probs = weight / weight.sum()

        cap = self.recency_resample_cap
        if self.max_samples_per_context is not None:
            cap = min(cap, self.max_samples_per_context)
        resample_size = min(n, cap)

        rng_rec = np.random.default_rng(self.random_state)
        idx = rng_rec.choice(n, size=resample_size, replace=True, p=probs)
        resampled = group_df.iloc[idx].reset_index(drop=True)
        return resampled, resample_size

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
        strict: bool = False,
    ) -> pd.DataFrame:
        """
        Generate synthetic EV sessions.

        Parameters
        ----------
        n_sessions : int
            Number of sessions to generate (an integer >= 1).
        context : dict, optional
            The context to sample, e.g. ``{'day_of_week': 1, 'season': 'winter'}``
            or, for a model stratified by site, ``{'location_type': 'work',
            'department': '75'}``. It may be partial: the keys it leaves out
            (``day_of_week``, ``is_weekend``, ``season``) are read from ``date``,
            and the others are "not given" (see ``strict``). Its keys must be
            among ``stratify_by``.
        date : str or Timestamp, optional
            Date that completes ``context`` (a value in ``context`` wins over
            the one the date gives). Read as local wall-clock time (a time zone
            is dropped, not converted). With neither, the most populous context
            is used.
        seed : int, optional
            Random seed.
        strict : bool, default False
            What to do when no mixture model was fitted for the requested
            context. ``False``: use the closest fitted context and warn once
            (see ``fallback_contexts_``). ``True``: raise ``ValueError``.

        Returns
        -------
        pd.DataFrame
            Columns: arrival_hour, duration, energy, [arrival_time].

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If the model is not fitted.
        ValueError
            If ``n_sessions`` is not an integer >= 1, ``context`` is not a
            dict or has a key the model is not stratified by, ``date`` is not
            a valid date, or ``strict`` is True and the context was not fitted.
        """
        self._check_fitted()
        n_sessions = check_int(n_sessions, "n_sessions", minimum=1)
        if date is not None:
            date = as_naive_timestamp(date, "date")
        ctx_tuple = self._resolve_context(context, date)
        gmm = self._get_model(ctx_tuple, strict=strict)

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

    def get_sklearn_component(
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
    def load(cls, path: str | Path) -> EVSessionModel:
        """Load a serialised GMM from disk.

        Parameters
        ----------
        path : str or Path
            Path to a ``.joblib`` file previously saved with :meth:`save`.

        Returns
        -------
        EVSessionModel
            The loaded and fitted model.

        Raises
        ------
        TypeError
            If the file contains an object that is not an ``EVSessionModel``.
        """
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj)}")
        return obj

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

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

        # Consistent model-identity color pairing used throughout the notebooks:
        # GMM = "#2E86AB" (blue), VAE = "#F4A261" (orange). Picking the color from
        # self.model_type (rather than hardcoding blue) keeps GMM-vs-VAE comparison
        # charts visually consistent regardless of which instance calls this method.
        is_vae = self.model_type == "vae"
        sim_color = "#F4A261" if is_vae else "#2E86AB"
        model_label = "VAE" if is_vae else "GMM"

        fig, axes = plt.subplots(1, 3, figsize=figsize)
        pairs = [
            ("arrival_hour", "Arrival hour"),
            ("duration", "Duration (h)"),
            ("energy", "Energy (kWh)"),
        ]

        for ax, (col, label) in zip(axes, pairs):
            ax.hist(synth[col], bins=bins, alpha=0.6, label="Simulated", density=True,
                    color=sim_color)
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
        fig.suptitle(f"{model_label} marginal distributions – context {ctx_tuple}")
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        """Raise ``NotFittedError`` if the model has not been fitted yet."""
        if not self.is_fitted_:
            raise not_fitted_error(
                "EVSessionModel",
                "Call .fit(sessions) first, or load a fitted model with "
                "gears.load_pretrained('french').",
            )

    def _date_context(self, date: pd.Timestamp) -> dict:
        """The part of the context that a date determines, for the keys this model uses."""
        derived: dict = {}
        if "day_of_week" in self.stratify_by:
            derived["day_of_week"] = date.day_of_week
        if "is_weekend" in self.stratify_by:
            derived["is_weekend"] = int(date.day_of_week >= 5)
        if "season" in self.stratify_by:
            derived["season"] = _season(date.month)
        return derived

    def _resolve_context(
        self,
        context: dict | None,
        date: str | pd.Timestamp | None,
    ) -> tuple:
        """Return a context tuple matching the stratify_by keys.

        Resolution order, key by key:
        1. The value given in *context*.
        2. The value the *date* determines (``day_of_week``, ``is_weekend``,
           ``season``). A partial context such as
           ``{"location_type": "work", "department": "75"}`` is therefore
           completed by the date, day after day.
        3. Nothing: the key stays ``None`` (``location_type`` and ``department``
           cannot be read from a date) and :meth:`_get_model` serves the closest
           fitted context. With neither context nor date, the most populous one.

        Parameters
        ----------
        context : dict or None
        date : str, Timestamp, or None

        Returns
        -------
        tuple
            Values ordered according to ``stratify_by``.

        Raises
        ------
        ValueError
            If *context* is not a dict, or names a key this model is not
            stratified by (it would have no effect, so it is refused).
        """
        if context is not None:
            if not isinstance(context, Mapping):
                raise ValueError(
                    "context must be a dict such as "
                    f"{{'day_of_week': 1, 'season': 'winter'}}, got {context!r} "
                    f"of type {type(context).__name__}. This model is stratified by "
                    f"{self.stratify_by}."
                )
            unknown = [k for k in context if k not in self.stratify_by]
            if unknown:
                raise ValueError(
                    f"context has key(s) {unknown} that this model is not stratified by "
                    f"(its context keys are {self.stratify_by}), so they would have no effect. "
                    "Check the spelling, or fit the model with these keys in stratify_by."
                )

        derived = self._date_context(as_naive_timestamp(date, "date")) if date is not None else {}
        if context is not None or date is not None:
            given = context if context is not None else {}
            return tuple(given[k] if k in given else derived.get(k) for k in self.stratify_by)

        # Fallback: most populous context
        if self.context_counts_:
            return max(self.context_counts_, key=lambda k: self.context_counts_[k])
        return next(iter(self.models_))

    def _closest_context(self, ctx_tuple: tuple) -> tuple:
        """Key of the fitted context used when ``ctx_tuple`` was not fitted.

        Tries progressively shorter prefixes of ``ctx_tuple`` (so the last
        ``stratify_by`` keys are dropped first), then the most populous context.
        """
        for width in range(len(ctx_tuple) - 1, 0, -1):
            prefix = ctx_tuple[:width]
            for key in self.models_:
                if key[:width] == prefix:
                    return key
        return max(self.context_counts_, key=lambda k: self.context_counts_[k])

    def _get_model(self, ctx_tuple: tuple, strict: bool = False) -> GaussianMixture:
        """Return the GMM for a context, falling back to the closest fitted one.

        Parameters
        ----------
        ctx_tuple : tuple
            Context tuple produced by :meth:`_resolve_context`.
        strict : bool
            Raise ``ValueError`` instead of falling back.

        Returns
        -------
        GaussianMixture
            The fitted model for the requested context or, if there is none and
            ``strict`` is False, the closest one. Each substitution is recorded
            in ``fallback_contexts_`` and announced by a warning (once per model).
        """
        if ctx_tuple in self.models_:
            return self.models_[ctx_tuple]

        asked = dict(zip(self.stratify_by, ctx_tuple))
        given = {k: v for k, v in asked.items() if v is not None}
        not_given = [k for k, v in asked.items() if v is None]
        what = (
            f"No mixture model was fitted for context {given or '{}'}"
            + (f" ({', '.join(not_given)} not given)" if not_given else "")
            + f"; this model is stratified by {self.stratify_by}."
        )
        if strict:
            raise ValueError(
                f"{what} Pass a context that was fitted (see "
                "model.bic_summary()['context']), or strict=False to fall back to "
                "the closest fitted context."
            )

        used = self._closest_context(ctx_tuple)
        self.fallback_contexts_[ctx_tuple] = used
        if not self._fallback_warned:
            self._fallback_warned = True
            warnings.warn(
                f"{what} Using the closest fitted context "
                f"{dict(zip(self.stratify_by, used))} instead, so the simulated "
                "sessions do not reflect the requested context. Pass a full `context` "
                "that was fitted, or strict=True to raise instead. Every substitution is listed "
                "in model.fallback_contexts_ (this warning is shown once per model).",
                UserWarning,
                stacklevel=3,
            )
        return self.models_[used]

    def __repr__(self) -> str:
        if not self.is_fitted_:
            return (
                f"EVSessionModel(n_components={self.n_components!r}, "
                f"stratify_by={self.stratify_by}, fitted=False)"
            )
        extras = []
        if self.is_sample_:
            extras.append("is_sample=True")
        if self.max_samples_per_context is not None:
            extras.append(f"max_samples={self.max_samples_per_context}")
        if self.recency:
            extras.append("recency=True")
        extra_str = (", " + ", ".join(extras)) if extras else ""
        return (
            f"EVSessionModel("
            f"n_contexts={len(self.models_)}, "
            f"stratify_by={self.stratify_by}, "
            f"n_components={self.n_components!r}"
            f"{extra_str}, fitted=True)"
        )

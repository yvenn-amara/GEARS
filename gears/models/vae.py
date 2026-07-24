"""
Conditional Variational Autoencoder (CVAE) for EV session generation.

Provides a drop-in replacement for sklearn GaussianMixture objects stored in
EVSessionGMM.models_, with identical duck-type API used by:
  - gears/output/aggregator.py   (sk_gmm.random_state, sk_gmm.sample)
  - gears/plotting.py            (sk_gmm.n_components, sk_gmm.means_, sk_gmm.weights_)

Architecture
------------
A single shared CVAE is trained across **all** context groups simultaneously.
Context information is injected via learnable embeddings (one per stratify_by
dimension) concatenated and passed to both encoder and decoder.  This avoids
fitting a separate model per context (some contexts have <50 samples) and
lets the model share statistical structure across similar contexts.

Feature space
-------------
Identical to the GMM: [hour, log1p(duration), log1p(energy)], standardised
before entering the network and inverse-standardised at generation time.

Log-likelihood
--------------
score_samples() returns per-sample IWAE bounds (Burda et al. 2016) using K
importance samples from the approximate posterior, providing a tighter estimate
than the vanilla ELBO.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy torch import — gives a clear message when torch is absent
# ---------------------------------------------------------------------------

def _require_torch():
    try:
        import torch
        return torch
    except ImportError as e:
        raise ImportError(
            "torch is required for VAE support.  Install it with:\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
            "or add it to your project with `uv add torch`."
        ) from e


# ---------------------------------------------------------------------------
# Encoder / Decoder MLP blocks
# ---------------------------------------------------------------------------

def _build_mlp(in_dim: int, hidden_dim: int, out_dim: int, n_layers: int = 2):
    """Build a simple MLP with ReLU activations and BatchNorm."""
    torch = _require_torch()
    nn = torch.nn
    layers: list[nn.Module] = []
    prev = in_dim
    for _ in range(n_layers):
        layers += [nn.Linear(prev, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU()]
        prev = hidden_dim
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# ConditionalVAE – shared model trained across all contexts
# ---------------------------------------------------------------------------

class ConditionalVAE:
    """
    Conditional VAE (PyTorch, CPU) sharing parameters across all contexts.

    Parameters
    ----------
    context_dims : list[int]
        Number of unique values for each stratify_by dimension.
        E.g. for [location_type, department, day_of_week, season]:
        [4, 5, 7, 4].
    emb_dims : list[int]
        Embedding dimension for each context variable.
    feature_dim : int
        Input feature dimension (3 for GEARS: hour, log_dur, log_ene).
    latent_dim : int
        VAE latent space dimension.
    hidden_dim : int
        Hidden layer width for encoder/decoder MLPs.
    n_layers : int
        Number of hidden layers in encoder and decoder.
    """

    def __init__(
        self,
        context_dims: list[int],
        emb_dims: list[int],
        feature_dim: int = 3,
        latent_dim: int = 16,
        hidden_dim: int = 256,
        n_layers: int = 2,
    ):
        self.context_dims = context_dims
        self.emb_dims = emb_dims
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self._build_network()

    def _build_network(self) -> None:
        torch = _require_torch()
        nn = torch.nn

        total_emb = sum(self.emb_dims)

        # Context embedding layers
        self.embeddings = nn.ModuleList([
            nn.Embedding(n_vals, emb_d)
            for n_vals, emb_d in zip(self.context_dims, self.emb_dims)
        ])

        # Encoder: [x | ctx_emb] → (mu_z, logvar_z)
        enc_in = self.feature_dim + total_emb
        self.encoder_body = _build_mlp(enc_in, self.hidden_dim, self.hidden_dim, self.n_layers)
        self.fc_mu = nn.Linear(self.hidden_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.hidden_dim, self.latent_dim)

        # Decoder: [z | ctx_emb] → x_recon (mu of p(x|z,c))
        dec_in = self.latent_dim + total_emb
        self.decoder = _build_mlp(dec_in, self.hidden_dim, self.feature_dim, self.n_layers)
        # Log-variance of p(x|z,c) — learnable global parameter (per feature)
        self.log_recon_var = nn.Parameter(torch.zeros(self.feature_dim))

        self._torch = torch
        self._nn = nn

    def parameters(self):
        """Yield all trainable parameters."""
        torch = self._torch
        nn = self._nn
        # Collect all nn.Module parameters
        modules = [*self.embeddings, self.encoder_body, self.fc_mu,
                   self.fc_logvar, self.decoder]
        seen = set()
        for m in modules:
            for p in m.parameters():
                if id(p) not in seen:
                    seen.add(id(p))
                    yield p
        yield self.log_recon_var

    def train(self) -> None:
        for m in [*self.embeddings, self.encoder_body, self.fc_mu,
                  self.fc_logvar, self.decoder]:
            m.train()

    def eval(self) -> None:
        for m in [*self.embeddings, self.encoder_body, self.fc_mu,
                  self.fc_logvar, self.decoder]:
            m.eval()

    def state_dict(self) -> dict:
        torch = self._torch
        sd: dict[str, Any] = {}
        for i, emb in enumerate(self.embeddings):
            sd[f"emb_{i}"] = emb.state_dict()
        sd["encoder_body"] = self.encoder_body.state_dict()
        sd["fc_mu"] = self.fc_mu.state_dict()
        sd["fc_logvar"] = self.fc_logvar.state_dict()
        sd["decoder"] = self.decoder.state_dict()
        sd["log_recon_var"] = self.log_recon_var.data.clone()
        return sd

    def load_state_dict(self, sd: dict) -> None:
        for i, emb in enumerate(self.embeddings):
            emb.load_state_dict(sd[f"emb_{i}"])
        self.encoder_body.load_state_dict(sd["encoder_body"])
        self.fc_mu.load_state_dict(sd["fc_mu"])
        self.fc_logvar.load_state_dict(sd["fc_logvar"])
        self.decoder.load_state_dict(sd["decoder"])
        with self._torch.no_grad():
            self.log_recon_var.copy_(sd["log_recon_var"])

    def to(self, device) -> "ConditionalVAE":
        for emb in self.embeddings:
            emb.to(device)
        self.encoder_body.to(device)
        self.fc_mu.to(device)
        self.fc_logvar.to(device)
        self.decoder.to(device)
        self.log_recon_var.data = self.log_recon_var.data.to(device)
        return self

    def _embed_context(self, ctx_idx: "torch.Tensor") -> "torch.Tensor":
        """Embed categorical context indices and concatenate."""
        torch = self._torch
        parts = [
            self.embeddings[i](ctx_idx[:, i])
            for i in range(len(self.embeddings))
        ]
        return torch.cat(parts, dim=-1)

    def encode(self, x: "torch.Tensor", ctx_emb: "torch.Tensor"):
        """Encode x conditioned on ctx_emb → (mu, logvar)."""
        h = self.encoder_body(torch_cat([x, ctx_emb], self._torch))
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z: "torch.Tensor", ctx_emb: "torch.Tensor") -> "torch.Tensor":
        """Decode z conditioned on ctx_emb → x_recon."""
        return self.decoder(torch_cat([z, ctx_emb], self._torch))

    def reparameterise(self, mu: "torch.Tensor", logvar: "torch.Tensor") -> "torch.Tensor":
        """Sample z = mu + eps * exp(0.5 * logvar)."""
        torch = self._torch
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def elbo_loss(
        self,
        x: "torch.Tensor",
        ctx_idx: "torch.Tensor",
        beta: float = 1.0,
    ) -> "torch.Tensor":
        """Beta-VAE ELBO loss (negative, to minimise)."""
        torch = self._torch
        ctx_emb = self._embed_context(ctx_idx)
        mu, logvar = self.encode(x, ctx_emb)
        z = self.reparameterise(mu, logvar)
        x_recon = self.decode(z, ctx_emb)

        # Gaussian reconstruction log-likelihood: sum over features
        recon_var = torch.exp(self.log_recon_var).clamp(1e-4, 10.0)
        recon_loss = 0.5 * torch.sum(
            ((x - x_recon) ** 2) / recon_var + torch.log(recon_var) + np.log(2 * np.pi),
            dim=-1,
        ).mean()

        # KL divergence: KL(q(z|x,c) || p(z)) where p(z) = N(0,I)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()

        return recon_loss + beta * kl

    def iwae_log_prob(
        self,
        x: "torch.Tensor",
        ctx_idx: "torch.Tensor",
        K: int = 20,
    ) -> "torch.Tensor":
        """IWAE log-likelihood bound per sample (Burda et al. 2016).

        Returns
        -------
        torch.Tensor, shape (N,)
            Per-sample IWAE log-probability bound.
        """
        torch = self._torch
        N = x.shape[0]
        ctx_emb = self._embed_context(ctx_idx)  # (N, emb_dim)

        # Expand to (N*K, ...)
        x_rep = x.unsqueeze(1).expand(N, K, -1).reshape(N * K, -1)
        ctx_emb_rep = ctx_emb.unsqueeze(1).expand(N, K, -1).reshape(N * K, -1)

        mu, logvar = self.encode(x_rep, ctx_emb_rep)
        z = self.reparameterise(mu, logvar)
        x_recon = self.decode(z, ctx_emb_rep)

        recon_var = torch.exp(self.log_recon_var).clamp(1e-4, 10.0)

        # log p(x|z,c)
        log_px_z = -0.5 * torch.sum(
            ((x_rep - x_recon) ** 2) / recon_var + torch.log(recon_var) + np.log(2 * np.pi),
            dim=-1,
        )

        # log p(z) = N(0,I)
        log_pz = -0.5 * torch.sum(z ** 2 + np.log(2 * np.pi), dim=-1)

        # log q(z|x,c) = N(mu, exp(logvar))
        log_qz_x = -0.5 * torch.sum(
            ((z - mu) ** 2) / logvar.exp() + logvar + np.log(2 * np.pi),
            dim=-1,
        )

        log_w = (log_px_z + log_pz - log_qz_x).reshape(N, K)  # (N, K)
        # IWAE bound: E[logsumexp(log_w) - log K]
        iwae = torch.logsumexp(log_w, dim=1) - np.log(K)
        return iwae

    def sample_prior(
        self,
        n: int,
        ctx_idx: "torch.Tensor",
        seed: Optional[int] = None,
    ) -> "torch.Tensor":
        """Sample n points from the prior p(z) → decode → x_recon."""
        torch = self._torch
        if seed is not None:
            torch.manual_seed(seed)
        with torch.no_grad():
            ctx_emb = self._embed_context(ctx_idx)  # (n, emb_dim)
            z = torch.randn(n, self.latent_dim)
            x_recon = self.decode(z, ctx_emb)
        return x_recon  # (n, feature_dim)


def torch_cat(tensors, torch):
    return torch.cat(tensors, dim=-1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_cvae(
    cvae: ConditionalVAE,
    X: np.ndarray,
    ctx_indices: np.ndarray,
    epochs: int = 50,
    batch_size: int = 512,
    lr: float = 3e-3,
    beta: float = 1.0,
    seed: int = 42,
    verbose: bool = True,
) -> list[float]:
    """
    Train a ConditionalVAE with Adam optimiser.

    Parameters
    ----------
    cvae : ConditionalVAE
    X : np.ndarray, shape (N, 3)
        Standardised feature matrix.
    ctx_indices : np.ndarray, shape (N, n_ctx_dims)
        Integer context indices per sample.
    epochs : int
    batch_size : int
    lr : float
    beta : float
        Beta-VAE coefficient for KL term.
    seed : int
    verbose : bool

    Returns
    -------
    list[float]
        Per-epoch training losses.
    """
    torch = _require_torch()
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_t = torch.tensor(X, dtype=torch.float32)
    ctx_t = torch.tensor(ctx_indices, dtype=torch.long)

    N = len(X_t)
    optimizer = torch.optim.Adam(list(cvae.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 10)

    cvae.train()
    losses = []
    for epoch in range(epochs):
        perm = torch.randperm(N)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, N, batch_size):
            idx = perm[i: i + batch_size]
            xb = X_t[idx]
            cb = ctx_t[idx]
            optimizer.zero_grad()
            loss = cvae.elbo_loss(xb, cb, beta=beta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(cvae.parameters()), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        mean_loss = epoch_loss / max(n_batches, 1)
        losses.append(mean_loss)
        if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
            logger.info("  VAE epoch %3d/%d  loss=%.4f", epoch + 1, epochs, mean_loss)

    cvae.eval()
    return losses


# ---------------------------------------------------------------------------
# VAEContextSlice – duck-types sklearn GaussianMixture API
# ---------------------------------------------------------------------------

class VAEContextSlice:
    """
    Wraps a trained ConditionalVAE for a fixed context, duck-typing the
    sklearn GaussianMixture interface used by GEARS internals.

    Used attributes/methods from the sklearn API:
    - .random_state          (settable int, used as seed before .sample())
    - .sample(n)             → (np.ndarray shape (n,3), None)
    - .score_samples(X)      → np.ndarray (per-sample IWAE log-likelihood)
    - .n_components          (= 1)
    - .means_                (shape (1,3), from generated samples)
    - .weights_              (= [1.0])

    Parameters
    ----------
    cvae : ConditionalVAE
        Trained shared CVAE model (in eval mode, on CPU).
    ctx_index : np.ndarray, shape (1, n_ctx_dims)
        Integer context indices for this stratum.
    scaler_mean : np.ndarray, shape (3,)
        Feature mean used for standardisation.
    scaler_std : np.ndarray, shape (3,)
        Feature std used for standardisation.
    score_n_samples : int
        K for IWAE importance sampling.
    """

    def __init__(
        self,
        cvae: ConditionalVAE,
        ctx_index: np.ndarray,
        scaler_mean: np.ndarray,
        scaler_std: np.ndarray,
        score_n_samples: int = 20,
    ):
        self.cvae = cvae
        self.ctx_index = ctx_index  # shape (1, n_ctx_dims)
        self.scaler_mean = scaler_mean
        self.scaler_std = scaler_std
        self.score_n_samples = score_n_samples

        # sklearn GaussianMixture compatibility
        self.n_components: int = 1
        self.weights_: np.ndarray = np.array([1.0])
        # means_ is set lazily to avoid sampling at construction time
        self._means_cache: Optional[np.ndarray] = None

        # Settable seed (used by aggregator.py / medium_term.py)
        self.random_state: Optional[int] = None

    # ------------------------------------------------------------------
    # sklearn duck-type API
    # ------------------------------------------------------------------

    def sample(self, n: int):
        """
        Sample n points from the CVAE for this context.

        Returns
        -------
        tuple
            (np.ndarray shape (n, 3), None)  — matches sklearn GMM API.
        """
        torch = _require_torch()
        ctx_t = torch.tensor(
            np.repeat(self.ctx_index, n, axis=0), dtype=torch.long
        )  # (n, n_ctx_dims)

        seed = self.random_state
        with torch.no_grad():
            x_std = self.cvae.sample_prior(n, ctx_t, seed=seed).cpu().numpy()

        # Inverse-standardise
        x = x_std * self.scaler_std + self.scaler_mean
        return x, None

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Compute per-sample IWAE log-likelihood for X.

        Parameters
        ----------
        X : np.ndarray, shape (N, 3)
            Feature matrix in original (non-standardised) space.

        Returns
        -------
        np.ndarray, shape (N,)
        """
        torch = _require_torch()
        N = len(X)
        X_std = (X - self.scaler_mean) / self.scaler_std
        X_t = torch.tensor(X_std, dtype=torch.float32)
        ctx_t = torch.tensor(
            np.repeat(self.ctx_index, N, axis=0), dtype=torch.long
        )

        # Process in chunks to avoid OOM with IWAE K expansion
        chunk = 256
        log_probs = []
        with torch.no_grad():
            for i in range(0, N, chunk):
                lp = self.cvae.iwae_log_prob(
                    X_t[i: i + chunk],
                    ctx_t[i: i + chunk],
                    K=self.score_n_samples,
                )
                log_probs.append(lp.cpu().numpy())

        return np.concatenate(log_probs)

    @property
    def means_(self) -> np.ndarray:
        """Return approximate component mean (shape (1, 3)) from 500 samples."""
        if self._means_cache is None:
            samples, _ = self.sample(500)
            self._means_cache = samples.mean(axis=0, keepdims=True)  # (1, 3)
        return self._means_cache

    @means_.setter
    def means_(self, value: np.ndarray) -> None:
        self._means_cache = value

    # ------------------------------------------------------------------
    # Pickle support (for joblib.dump / joblib.load)
    # ------------------------------------------------------------------

    def __getstate__(self):
        # Store the CVAE state dict (plain tensors, joblib-serialisable)
        state = self.__dict__.copy()
        state["_cvae_state_dict"] = self.cvae.state_dict()
        state["_cvae_init_kwargs"] = {
            "context_dims":  self.cvae.context_dims,
            "emb_dims":      self.cvae.emb_dims,
            "feature_dim":   self.cvae.feature_dim,
            "latent_dim":    self.cvae.latent_dim,
            "hidden_dim":    self.cvae.hidden_dim,
            "n_layers":      self.cvae.n_layers,
        }
        # Don't store the live model object (it contains non-picklable bits sometimes)
        del state["cvae"]
        return state

    def __setstate__(self, state):
        cvae_sd = state.pop("_cvae_state_dict")
        cvae_kw = state.pop("_cvae_init_kwargs")
        self.__dict__.update(state)
        cvae = ConditionalVAE(**cvae_kw)
        cvae.load_state_dict(cvae_sd)
        cvae.eval()
        self.cvae = cvae


# ---------------------------------------------------------------------------
# Context encoder helper – maps stratify_by values to integer indices
# ---------------------------------------------------------------------------

class ContextEncoder:
    """
    Maps categorical context values to integer indices for VAE embedding.

    Parameters
    ----------
    stratify_by : list[str]
        Names of context dimensions.
    """

    def __init__(self, stratify_by: list[str]):
        self.stratify_by = stratify_by
        self.vocab_: dict[str, dict[Any, int]] = {}  # dim_name → {value: idx}

    def fit(self, context_keys: Sequence[tuple]) -> "ContextEncoder":
        """Build vocabulary from context tuples."""
        for dim_i, dim_name in enumerate(self.stratify_by):
            values = sorted(set(k[dim_i] for k in context_keys))
            self.vocab_[dim_name] = {v: i for i, v in enumerate(values)}
        return self

    @property
    def context_dims(self) -> list[int]:
        """Number of unique values per dimension."""
        return [len(self.vocab_[d]) for d in self.stratify_by]

    def encode(self, ctx_tuple: tuple) -> np.ndarray:
        """Convert a context tuple to integer indices, shape (1, n_dims)."""
        idx = [
            self.vocab_[self.stratify_by[i]].get(v, 0)
            for i, v in enumerate(ctx_tuple)
        ]
        return np.array([idx], dtype=np.int64)

    def encode_batch(self, ctx_tuples: list[tuple]) -> np.ndarray:
        """Convert a list of context tuples to array of shape (N, n_dims)."""
        return np.array([
            [self.vocab_[self.stratify_by[i]].get(v, 0) for i, v in enumerate(ctx)]
            for ctx in ctx_tuples
        ], dtype=np.int64)

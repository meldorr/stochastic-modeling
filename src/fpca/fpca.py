"""Stage 2: per-variable discrete functional PCA.

Every trajectory lives on the same uniform 200-point grid, so the L2 inner
product of the Karhunen-Loeve expansion reduces to a plain dot product and
"discrete fPCA" is exactly PCA on the (N, T) matrix of each feature's profiles
(Dinh et al. 2025, Sec. 3.1). We fit one basis per feature and concatenate the
scores into a single latent weight vector, mirroring the joint drag+CAS modelling
of Hodgkin et al. 2025.

    encode:  x(t)  ->  w = [ <x_f - mu_f, phi_f> ]  concatenated over features
    decode:  w     ->  x_hat_f(t) = mu_f(t) + sum_k w_{f,k} phi_{f,k}(t)

The basis is frozen after ``fit`` and never updates during DDPM training.
"""

from __future__ import annotations

import numpy as np


def _fit_one(profiles: np.ndarray, threshold: float, cap: int) -> dict:
    """Discrete fPCA for one feature. ``profiles``: (N, T) standardized curves."""
    mu = profiles.mean(0)                      # mean function (T,)
    centered = profiles - mu
    # economy SVD: Vt rows are orthonormal eigenfunctions, ordered by eigenvalue.
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    var = s**2
    evr = var / var.sum()
    cum = np.cumsum(evr)
    k = int(np.searchsorted(cum, threshold) + 1)
    k = max(1, min(k, cap, vt.shape[0]))
    return {
        "mean": mu.astype(np.float64),
        "components": vt[:k].astype(np.float64),        # (k, T)
        "evr": evr[:k].astype(np.float64),              # retained per-comp variance frac
        "evr_full": evr.astype(np.float64),             # full spectrum (for diagnostics)
        "k": k,
    }


class FPCA:
    """Bank of per-feature discrete fPCA bases with a concatenated latent."""

    def __init__(self, feature_names: list[str], bases: list[dict]):
        self.feature_names = list(feature_names)
        self.bases = bases
        self.ks = [b["k"] for b in bases]
        self.slices = self._make_slices(self.ks)
        self.m = sum(self.ks)                            # total latent dim

    @staticmethod
    def _make_slices(ks: list[int]) -> list[slice]:
        out, start = [], 0
        for k in ks:
            out.append(slice(start, start + k))
            start += k
        return out

    @classmethod
    def fit(
        cls,
        X_std: np.ndarray,
        feature_names: list[str],
        explained_variance: float = 0.95,
        max_components: int = 40,
    ) -> "FPCA":
        """Fit one basis per feature on standardized profiles ``X_std`` (N, T, F)."""
        assert X_std.shape[-1] == len(feature_names)
        bases = [
            _fit_one(X_std[:, :, f], explained_variance, max_components)
            for f in range(len(feature_names))
        ]
        return cls(feature_names, bases)

    def encode(self, X_std: np.ndarray) -> np.ndarray:
        """(N, T, F) standardized profiles -> (N, m) latent weights."""
        parts = []
        # errstate: numpy>=2 can emit spurious FP warnings from the matmul SIMD
        # path even when outputs are finite (verified); silence that noise.
        with np.errstate(all="ignore"):
            for f, b in enumerate(self.bases):
                centered = X_std[:, :, f] - b["mean"]
                parts.append(centered @ b["components"].T)   # (N, k_f)
        return np.concatenate(parts, axis=1).astype(np.float32)

    def decode(self, W: np.ndarray) -> np.ndarray:
        """(N, m) latent weights -> (N, T, F) standardized profiles."""
        n = W.shape[0]
        t = self.bases[0]["mean"].shape[0]
        out = np.empty((n, t, len(self.bases)), np.float32)
        with np.errstate(all="ignore"):
            for f, (b, sl) in enumerate(zip(self.bases, self.slices)):
                out[:, :, f] = b["mean"] + W[:, sl] @ b["components"]
        return out

    def reconstruction_error(self, X_std: np.ndarray) -> np.ndarray:
        """Per-feature RMSE of encode->decode round trip (standardized units)."""
        rec = self.decode(self.encode(X_std))
        return np.sqrt(((rec - X_std) ** 2).mean(axis=(0, 1)))       # (F,)

    def total_explained_variance(self) -> dict[str, float]:
        return {n: float(b["evr"].sum()) for n, b in zip(self.feature_names, self.bases)}

    # --- serialization -----------------------------------------------------
    def state(self) -> dict:
        return {
            "feature_names": self.feature_names,
            "bases": [
                {
                    "mean": b["mean"],
                    "components": b["components"],
                    "evr": b["evr"],
                    "evr_full": b["evr_full"],
                    "k": b["k"],
                }
                for b in self.bases
            ],
        }

    @classmethod
    def from_state(cls, s: dict) -> "FPCA":
        return cls(s["feature_names"], s["bases"])


class LatentScaler:
    """Zero-mean, unit-variance per-dim scaler for the fPCA weight vectors.

    Applied before the DDPM (which then works on a roughly N(0, I) latent) and
    inverted after sampling.
    """

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = np.asarray(mean, np.float32)
        self.std = np.asarray(std, np.float32)
        self.std[self.std == 0] = 1.0

    @classmethod
    def fit(cls, W: np.ndarray) -> "LatentScaler":
        return cls(W.mean(0), W.std(0))

    def transform(self, W: np.ndarray) -> np.ndarray:
        return ((W - self.mean) / self.std).astype(np.float32)

    def inverse_transform(self, W: np.ndarray) -> np.ndarray:
        return (W * self.std + self.mean).astype(np.float32)

    def state(self) -> dict:
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_state(cls, s: dict) -> "LatentScaler":
        return cls(s["mean"], s["std"])

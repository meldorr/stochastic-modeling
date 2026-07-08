"""Stage 2: per-variable functional PCA, with two basis choices.

``basis: discrete`` — PCA straight on the (N, T) sampled profiles. Valid because
every flight shares one uniform grid, so the Karhunen-Loeve L2 inner product
collapses to a dot product (Dinh et al. 2025, Sec. 3.1, "discrete FPCA").

``basis: bspline`` — Jarry et al. 2022 (Sec. 3.1-3.2): first represent each curve
in a **cubic B-spline basis with a Sobolev roughness penalty** (∫ f''^2, the W^2
seminorm), giving smooth spline coefficients; then run functional PCA in that
coefficient space, metrized by the basis Gram matrix G = ∫ φ_i φ_j. Working with
whitened coefficients z = (c - c̄) G^{1/2} makes the score Euclidean norm equal the
L2 function-space distance (Jarry Sec. 4.2.2), and band-limits the reconstruction
to the spline space (built-in smoothing that discrete PCA only gets via truncation).

Both bases expose the *same* affine encode/decode on the grid, so everything
downstream (DDPM, reconstruction) is identical:

    encode:  x(t)  ->  w = [ scores per feature ]  (concatenated)
    decode:  w     ->  x_hat_f(t)                    (on the 200-pt grid)

The basis is frozen after ``fit`` and never updates during DDPM training.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline


# --- discrete FPCA ---------------------------------------------------------
def _fit_discrete(profiles: np.ndarray, threshold: float, cap: int) -> dict:
    """Discrete fPCA for one feature. ``profiles``: (N, T) standardized curves."""
    mu = profiles.mean(0)
    centered = profiles - mu
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    evr = (s**2) / (s**2).sum()
    k = int(np.searchsorted(np.cumsum(evr), threshold) + 1)
    k = max(1, min(k, cap, vt.shape[0]))
    return {
        "basis": "discrete",
        "mean": mu.astype(np.float64),           # (T,)
        "components": vt[:k].astype(np.float64),  # (k, T) orthonormal eigenfunctions
        "evr": evr[:k].astype(np.float64),
        "evr_full": evr.astype(np.float64),
        "k": k,
    }


# --- Jarry-style B-spline FPCA --------------------------------------------
def _bspline_design(n_points: int, n_basis: int, degree: int) -> np.ndarray:
    """(n_points, n_basis) cubic B-spline basis evaluated on a uniform grid."""
    k = degree
    n_basis = max(n_basis, k + 1)
    n_interior = n_basis - k - 1
    interior = np.linspace(0.0, 1.0, n_interior + 2)[1:-1] if n_interior > 0 else np.array([])
    knots = np.concatenate([np.zeros(k + 1), interior, np.ones(k + 1)])
    x = np.linspace(0.0, 1.0, n_points)
    return np.asarray(BSpline.design_matrix(x, knots, k, extrapolate=True).todense())


def _sym_sqrt(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric square root and inverse square root of an SPD matrix."""
    w, q = np.linalg.eigh(mat)
    w = np.clip(w, 1e-10, None)
    return (q * np.sqrt(w)) @ q.T, (q * (1.0 / np.sqrt(w))) @ q.T


def _fit_bspline(profiles, threshold, cap, n_basis, degree, smoothing) -> dict:
    """Penalized-spline functional PCA for one feature. ``profiles``: (N, T)."""
    t = profiles.shape[1]
    B = _bspline_design(t, n_basis, degree)          # (T, K)
    B2 = np.diff(B, n=2, axis=0)                      # 2nd-difference -> roughness
    R = B2.T @ B2                                     # ∫ f''^2 penalty matrix (K, K)
    G = B.T @ B                                       # Gram / L2 metric (K, K)

    # penalized least-squares smoothing-spline fit: c = (G + λR)^{-1} B^T x
    fit_op = np.linalg.solve(G + smoothing * R, B.T)  # (K, T)
    C = profiles @ fit_op.T                            # (N, K) spline coefficients

    Gh, Gh_inv = _sym_sqrt(G)                          # G^{1/2}, G^{-1/2}
    cbar = C.mean(0)
    Z = (C - cbar) @ Gh                                # whitened coeffs -> L2-metric PCA
    _, s, vt = np.linalg.svd(Z, full_matrices=False)
    evr = (s**2) / (s**2).sum()
    k = int(np.searchsorted(np.cumsum(evr), threshold) + 1)
    k = max(1, min(k, cap, vt.shape[0]))
    U = vt[:k].T                                       # (K, k)

    # collapse the whole chain into one affine map on the grid (shared with decode)
    return {
        "basis": "bspline",
        "enc_matrix": (fit_op.T @ (Gh @ U)).astype(np.float64),   # (T, k): s = x @ E + b
        "enc_bias": (-(cbar @ Gh @ U)).astype(np.float64),        # (k,)
        "dec_matrix": ((U.T @ Gh_inv) @ B.T).astype(np.float64),  # (k, T): x = s @ D + m
        "mean_grid": (cbar @ B.T).astype(np.float64),             # (T,)
        "evr": evr[:k].astype(np.float64),
        "evr_full": evr.astype(np.float64),
        "k": k,
        "n_basis": int(B.shape[1]),
    }


class FPCA:
    """Bank of per-feature functional PCA bases with a concatenated latent."""

    def __init__(self, feature_names: list[str], bases: list[dict]):
        self.feature_names = list(feature_names)
        self.bases = bases
        self.ks = [b["k"] for b in bases]
        self.slices = self._make_slices(self.ks)
        self.m = sum(self.ks)

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
        explained_variance: float | dict = 0.95,
        max_components: int = 40,
        basis: str = "discrete",
        bspline_cfg: dict | None = None,
    ) -> "FPCA":
        """Fit one basis per feature on standardized profiles ``X_std`` (N, T, F).

        ``explained_variance`` may be a scalar or a per-feature dict (with an
        optional ``default``). ``basis`` selects discrete vs. Jarry B-spline FPCA.
        """
        assert X_std.shape[-1] == len(feature_names)
        if isinstance(explained_variance, dict):
            dflt = explained_variance.get("default", 0.95)
            thr = [float(explained_variance.get(n, dflt)) for n in feature_names]
        else:
            thr = [float(explained_variance)] * len(feature_names)

        if basis == "bspline":
            bc = bspline_cfg or {}
            n_basis = int(bc.get("n_basis", 40))
            degree = int(bc.get("degree", 3))
            smoothing = float(bc.get("smoothing", 1e-3))
            bases = [
                _fit_bspline(X_std[:, :, f], thr[f], max_components, n_basis, degree, smoothing)
                for f in range(len(feature_names))
            ]
        elif basis == "discrete":
            bases = [
                _fit_discrete(X_std[:, :, f], thr[f], max_components)
                for f in range(len(feature_names))
            ]
        else:
            raise ValueError(f"unknown fpca basis: {basis!r} (use 'discrete' or 'bspline')")
        return cls(feature_names, bases)

    def encode(self, X_std: np.ndarray) -> np.ndarray:
        """(N, T, F) standardized profiles -> (N, m) latent weights."""
        parts = []
        # errstate: numpy>=2 can emit spurious FP flags from the matmul SIMD path.
        with np.errstate(all="ignore"):
            for f, b in enumerate(self.bases):
                x = X_std[:, :, f]
                if b["basis"] == "bspline":
                    parts.append(x @ b["enc_matrix"] + b["enc_bias"])
                else:
                    parts.append((x - b["mean"]) @ b["components"].T)
        return np.concatenate(parts, axis=1).astype(np.float32)

    def decode(self, W: np.ndarray) -> np.ndarray:
        """(N, m) latent weights -> (N, T, F) standardized profiles."""
        b0 = self.bases[0]
        t = (b0["mean"] if b0["basis"] == "discrete" else b0["mean_grid"]).shape[0]
        out = np.empty((W.shape[0], t, len(self.bases)), np.float32)
        with np.errstate(all="ignore"):
            for f, (b, sl) in enumerate(zip(self.bases, self.slices)):
                s = W[:, sl]
                if b["basis"] == "bspline":
                    out[:, :, f] = b["mean_grid"] + s @ b["dec_matrix"]
                else:
                    out[:, :, f] = b["mean"] + s @ b["components"]
        return out

    def reconstruction_error(self, X_std: np.ndarray) -> np.ndarray:
        """Per-feature RMSE of encode->decode round trip (standardized units)."""
        rec = self.decode(self.encode(X_std))
        return np.sqrt(((rec - X_std) ** 2).mean(axis=(0, 1)))

    def total_explained_variance(self) -> dict[str, float]:
        return {n: float(b["evr"].sum()) for n, b in zip(self.feature_names, self.bases)}

    # --- serialization -----------------------------------------------------
    def state(self) -> dict:
        return {"feature_names": self.feature_names, "bases": [dict(b) for b in self.bases]}

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

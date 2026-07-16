"""Persistable trajectory clusterer.

Wraps the (StandardScaler + KMeans/GMM) fitted on the chosen clustering features
so a flight can be assigned to a cluster reproducibly at train and eval time, and
so the clusterer travels inside the checkpoint. Only ``xy_flat`` is supported here
(the chosen basis); it needs no fPCA, so assignment is self-contained.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


class ClusterAssigner:
    def __init__(self, spec: dict, scaler: StandardScaler, model):
        self.spec = spec              # {basis, features, stride, method, k}
        self.scaler = scaler          # fitted StandardScaler
        self.model = model            # fitted KMeans / GaussianMixture

    def _raw(self, X_std: np.ndarray, feature_names) -> np.ndarray:
        idx = [list(feature_names).index(c) for c in self.spec["features"]]
        if self.spec["basis"] != "xy_flat":
            raise ValueError(f"ClusterAssigner supports xy_flat only, got {self.spec['basis']!r}")
        stride = int(self.spec["stride"])
        sub = X_std[:, ::stride, :][:, :, idx]        # (N, T', |features|)
        return sub.reshape(len(sub), -1)

    @classmethod
    def fit(cls, data: dict, cfg_cl: dict, seed: int) -> tuple["ClusterAssigner", np.ndarray]:
        spec = {
            "basis": cfg_cl.get("basis", "xy_flat"),
            "features": list(cfg_cl.get("features", ["x", "y"])),
            "stride": int(cfg_cl.get("stride", 4)),
            "method": cfg_cl.get("method", "kmeans"),
            "k": int(cfg_cl["k"]),
        }
        self = cls(spec, StandardScaler(), None)
        feats = self.scaler.fit_transform(self._raw(data["X_std"], data["feature_names"]))
        if spec["method"] == "gmm":
            self.model = GaussianMixture(n_components=spec["k"], covariance_type="full",
                                         random_state=seed).fit(feats)
        else:
            self.model = KMeans(n_clusters=spec["k"], n_init=10, random_state=seed).fit(feats)
        return self, self.model.predict(feats).astype(np.int64)

    def assign(self, X_std: np.ndarray, feature_names) -> np.ndarray:
        feats = self.scaler.transform(self._raw(X_std, feature_names))
        return self.model.predict(feats).astype(np.int64)

    def state(self) -> dict:
        return {"spec": self.spec, "scaler": self.scaler, "model": self.model}

    @classmethod
    def from_state(cls, s: dict) -> "ClusterAssigner":
        return cls(s["spec"], s["scaler"], s["model"])

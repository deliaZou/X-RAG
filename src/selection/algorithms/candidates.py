"""
Candidate Algorithm Registry
============================
Wraps every detection candidate behind one uniform interface so the
comparison runner does not need to know about sklearn / pyod / eif API
differences.

Each Candidate knows:
  - how to fit
  - how to produce an anomaly score (higher = more anomalous)
  - how to produce a binary prediction (1 = anomaly)
  - which SHAP variant applies (tree = exact, kernel = approximate)

EIF handling:
  If the real `eif` package is importable, EIF_MODE = "real".
  Otherwise it falls back to sklearn IsolationForest(max_features=0.8),
  an extended-cut approximation that stays TreeSHAP-compatible.
  Disclose whichever mode was used in the methodology.
"""

import time
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# ── optional deps ─────────────────────────────────────────────────────────────
try:
    from pyod.models.knn import KNN
    from pyod.models.hbos import HBOS
    from pyod.models.cblof import CBLOF
    from pyod.models.copod import COPOD
    from pyod.models.pca import PCA as PYOD_PCA
    PYOD_AVAILABLE = True
except ImportError:
    PYOD_AVAILABLE = False

try:
    import eif as _eif
    EIF_MODE = "real"
except ImportError:
    EIF_MODE = "approx"


TYPE_MAP = {
    "iForest": "Trees", "EIF": "Trees",
    "KNN": "Distance", "LOF": "Distance", "HBOS": "Distance", "CBLOF": "Distance",
    "COPOD": "Distribution", "PCC": "Reconstruction",
}


class Candidate:
    """Uniform wrapper around one detection algorithm."""

    def __init__(self, name, shap_type, params):
        self.name = name
        self.shap_type = shap_type          # "tree" | "kernel"
        self.compat = "Exact" if shap_type == "tree" else "Approximate"
        self.type = TYPE_MAP.get(name, "Other")
        self.params = params
        self.model = None
        self._eif_forest = None             # only used when EIF_MODE == "real"

    # -- to be provided by subclasses --------------------------------------
    def fit(self, X_train):
        raise NotImplementedError

    def score(self, X):
        """Anomaly score, higher = more anomalous."""
        raise NotImplementedError

    def predict(self, X):
        """Binary prediction, 1 = anomaly."""
        raise NotImplementedError

    def score_fn(self):
        """Callable X -> scores, used as the KernelSHAP prediction function."""
        return self.score


class _IForest(Candidate):
    def fit(self, X_train):
        self.model = IsolationForest(
            n_estimators=self.params["n_estimators"],
            contamination=self.params["contamination"],
            random_state=self.params.get("random_state", 42), n_jobs=-1,
        )
        self.model.fit(X_train)

    def score(self, X):
        return -self.model.decision_function(X)

    def predict(self, X):
        return (self.model.predict(X) == -1).astype(int)


class _EIF(Candidate):
    def fit(self, X_train):
        if EIF_MODE == "real":
            self._X_train = np.ascontiguousarray(X_train.astype("float64"))
            ntrees = self.params["n_estimators"]
            sample = min(256, len(X_train))
            # ExtensionLevel = n_features - 1 -> fully extended
            self._eif_forest = _eif.iForest(
                self._X_train, ntrees=ntrees, sample_size=sample,
                ExtensionLevel=X_train.shape[1] - 1,
            )
        else:
            # approximation: sklearn IF with partial feature subsampling
            self.model = IsolationForest(
                n_estimators=self.params["n_estimators"],
                max_features=self.params.get("max_features", 0.8),
                contamination=self.params["contamination"],
                random_state=self.params.get("random_state", 42), n_jobs=-1,
            )
            self.model.fit(X_train)

    def score(self, X):
        if EIF_MODE == "real":
            Xc = np.ascontiguousarray(X.astype("float64"))
            return self._eif_forest.compute_paths(X_in=Xc)
        return -self.model.decision_function(X)

    def predict(self, X):
        s = self.score(X)
        thr = np.percentile(s, 100 * (1 - self.params["contamination"]))
        return (s >= thr).astype(int)


class _LOF(Candidate):
    def fit(self, X_train):
        # novelty=True so we can score unseen test points and drive KernelSHAP
        self.model = LocalOutlierFactor(
            n_neighbors=self.params["n_neighbors"],
            contamination=self.params["contamination"],
            novelty=True, n_jobs=-1,
        )
        self.model.fit(X_train)

    def score(self, X):
        return -self.model.decision_function(X)

    def predict(self, X):
        s = self.score(X)
        thr = np.percentile(s, 100 * (1 - self.params["contamination"]))
        return (s >= thr).astype(int)


class _PyODCandidate(Candidate):
    """Generic pyod wrapper (KNN, HBOS, COPOD, PCC)."""

    def __init__(self, name, shap_type, params, factory):
        super().__init__(name, shap_type, params)
        self._factory = factory

    def fit(self, X_train):
        self.model = self._factory()
        self.model.fit(X_train)

    def score(self, X):
        return self.model.decision_function(X)

    def predict(self, X):
        return (self.model.predict(X) == 1).astype(int)


class _CBLOF(Candidate):
    """CBLOF needs auto-retry: some cluster settings raise on certain data."""

    # fallback combinations tried only if the configured preferred value fails
    _FALLBACK = [(n, a, b)
                 for n in (5, 10, 3, 12, 6)
                 for a, b in ((0.75, 3), (0.9, 5), (0.6, 2))]

    def fit(self, X_train):
        preferred = (self.params["n_clusters"],
                     self.params["alpha"], self.params["beta"])
        grid = [preferred] + [g for g in self._FALLBACK if g != preferred]
        rs = self.params.get("random_state", 42)
        last_err = None
        for n_clust, alpha, beta in grid:
            try:
                m = CBLOF(n_clusters=n_clust,
                          contamination=self.params["contamination"],
                          alpha=alpha, beta=beta, random_state=rs)
                m.fit(X_train)
                self.model = m
                self.params = {**self.params, "n_clusters": n_clust,
                               "alpha": alpha, "beta": beta}
                return
            except Exception as e:  # noqa: BLE001
                last_err = e
        raise RuntimeError(f"CBLOF failed for all grid settings: {last_err}")

    def score(self, X):
        return self.model.decision_function(X)

    def predict(self, X):
        return (self.model.predict(X) == 1).astype(int)


def _load_config(cfg=None):
    """
    Resolve the parameter config.
      cfg=None            -> load configs/selection.yaml
      cfg="path/to.yaml"  -> load that file
      cfg=dict            -> use as-is
    Falls back to built-in defaults for any missing key.
    """
    import os
    import yaml

    _DEFAULTS = {
        "shared": {"contamination": 0.05},
        "iForest": {"n_estimators": 100, "random_state": 42},
        "EIF": {"n_estimators": 100, "random_state": 42, "max_features": 0.8},
        "LOF": {"n_neighbors": 20},
        "KNN": {"n_neighbors": 5},
        "HBOS": {"n_bins": 10},
        "CBLOF": {"n_clusters": 8, "alpha": 0.75, "beta": 3},
        "COPOD": {},
        "PCC": {"n_components": 0.95},
        "experiment": {"explain_samples": 100, "kernel_background": 50},
    }

    if cfg is None:
        here = os.path.dirname(os.path.abspath(__file__))
        cfg = os.path.abspath(
            os.path.join(here, "..", "..", "..", "configs", "selection.yaml"))
    if isinstance(cfg, str):
        with open(cfg,  encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    merged = {}
    for section, defaults in _DEFAULTS.items():
        merged[section] = {**defaults, **(cfg.get(section) or {})}
    return merged


def get_candidates(cfg=None):
    """
    Build the full candidate registry from a config file/dict.
    All 28 machines pass the SAME cfg, enforcing unified parameters.
    pyod-dependent candidates are skipped (with a warning) if pyod is missing.
    """
    c = _load_config(cfg)
    contamination = c["shared"]["contamination"]
    reg = {}

    reg["iForest"] = _IForest("iForest", "tree",
                              {**c["iForest"], "contamination": contamination})

    reg["EIF"] = _EIF("EIF", "tree",
                      {**c["EIF"], "contamination": contamination,
                       "mode": EIF_MODE})

    reg["LOF"] = _LOF("LOF", "kernel",
                      {**c["LOF"], "contamination": contamination})

    if PYOD_AVAILABLE:
        knn = c["KNN"]
        reg["KNN"] = _PyODCandidate(
            "KNN", "kernel", {**knn, "contamination": contamination},
            lambda: KNN(n_neighbors=knn["n_neighbors"],
                        contamination=contamination))
        hbos = c["HBOS"]
        reg["HBOS"] = _PyODCandidate(
            "HBOS", "kernel", {**hbos, "contamination": contamination},
            lambda: HBOS(n_bins=hbos["n_bins"], contamination=contamination))
        reg["CBLOF"] = _CBLOF(
            "CBLOF", "kernel", {**c["CBLOF"], "contamination": contamination})
        reg["COPOD"] = _PyODCandidate(
            "COPOD", "kernel", {"contamination": contamination},
            lambda: COPOD(contamination=contamination))
        pcc = c["PCC"]
        reg["PCC"] = _PyODCandidate(
            "PCC", "kernel", {**pcc, "contamination": contamination},
            lambda: PYOD_PCA(n_components=pcc["n_components"],
                             contamination=contamination))
    else:
        print("[WARN] pyod not installed. KNN/HBOS/CBLOF/COPOD/PCC skipped.")

    return reg
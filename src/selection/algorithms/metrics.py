"""
Metrics & Timing
================
Detection metrics (AUC-ROC, AUC-PR, F1, confusion matrix) and SHAP
explanation-time measurement. Kept as pure-ish functions so Layer 1 can
reuse the detection metrics later.
"""

import time
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, confusion_matrix
)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def predict_by_rate(y_score, rate):
    """
    Unified evaluation-side thresholding: flag the top `rate` fraction of
    instances (by anomaly score) as anomalies.

    rate is the test-set true anomaly rate (from metadata). Using it here,
    on the evaluation side only, keeps thresholding consistent across all
    algorithms on a given machine and adapts to each machine's real anomaly
    level. It does NOT touch training, so no label leakage into the model.
    """
    if rate is None or not (0 < rate < 1):
        raise ValueError(f"rate must be in (0,1), got {rate}")
    thr = np.quantile(y_score, 1 - rate)
    return (y_score >= thr).astype(int)

def detection_metrics(y_true, y_score, y_pred):
    """Return a dict of detection metrics; individually guarded."""
    out = {}
    try:
        out["AUC-ROC"] = round(roc_auc_score(y_true, y_score), 4)
    except Exception:
        out["AUC-ROC"] = np.nan
    try:
        out["AUC-PR"] = round(average_precision_score(y_true, y_score), 4)
    except Exception:
        out["AUC-PR"] = np.nan
    try:
        out["F1"] = round(f1_score(y_true, y_pred, zero_division=0), 4)
    except Exception:
        out["F1"] = np.nan
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        n_norm = tn + fp
        out["Confusion Matrix"] = f"TP={tp} FP={fp} TN={tn} FN={fn}"
        out["FPR"] = round(fp / n_norm, 4) if n_norm else np.nan
    except Exception:
        out["Confusion Matrix"] = "N/A"
        out["FPR"] = np.nan
    return out


def measure_explain_time(candidate, X_train, X_test,
                         n_sample=100, n_background=50):
    """
    Time SHAP attribution on a sample of test instances.
    Dispatches to TreeSHAP (exact) or KernelSHAP (approximate) based on
    candidate.shap_type. Returns (total_seconds, per_sample_seconds).
    """
    if not SHAP_AVAILABLE:
        return np.nan, np.nan

    X_exp = X_test[:min(n_sample, len(X_test))]

    if candidate.shap_type == "tree":
        # TreeSHAP only supports the sklearn-tree candidates (iForest / EIF-approx)
        model = getattr(candidate, "model", None)
        if model is None:
            return np.nan, np.nan   # e.g. real-EIF has no sklearn tree model
        t0 = time.time()
        explainer = shap.TreeExplainer(model)
        explainer.shap_values(X_exp)
        total = time.time() - t0
    else:
        bg = shap.sample(X_train, min(n_background, len(X_train)),
                         random_state=42)
        t0 = time.time()
        explainer = shap.KernelExplainer(candidate.score_fn(), bg)
        explainer.shap_values(X_exp, silent=True)
        total = time.time() - t0

    return round(total, 4), round(total / len(X_exp), 6)
"""Structural integrity, calibration and risk-coverage metrics (§4.8, E14)."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

FEATURES = [
    "peak_ratio",        # J1 - J2 (second peak > 50 m away): map distinctiveness at this pose
    "J1",                # optimum value
    "log_sigma_pos",     # Laplace position std (log m)
    "log_nb",            # log(1 + #buildings)
    "spread",            # centroid spread / footprint (0..1)
    "persistence",       # mean boundary persistence
    "A",                 # Ekeland shape agreement
]


def feature_matrix(rows: Sequence[Dict]) -> np.ndarray:
    return np.array([[float(r.get(k, 0.0)) for k in FEATURES] for r in rows], np.float64)


class IntegrityModel:
    """Logistic model P(success | structural evidence), fit on validation sites only."""

    def __init__(self, C: float = 1.0):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        self.sc = StandardScaler()
        self.lr = LogisticRegression(C=C, max_iter=2000)

    def fit(self, X, y):
        X = np.nan_to_num(X, nan=0.0, posinf=10.0, neginf=-10.0)
        self.lr.fit(self.sc.fit_transform(X), y)
        return self

    def predict(self, X):
        X = np.nan_to_num(X, nan=0.0, posinf=10.0, neginf=-10.0)
        return self.lr.predict_proba(self.sc.transform(X))[:, 1]


def auroc(conf, y):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    return float(roc_auc_score(y, conf)) if 0 < y.sum() < len(y) else float("nan")


def ece(conf, y, bins=10):
    conf, y = np.asarray(conf), np.asarray(y, float)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        m = (conf >= a) & (conf < b if b < 1 else conf <= b)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - y[m].mean())
    return float(e)


def brier(conf, y):
    return float(np.mean((np.asarray(conf) - np.asarray(y, float)) ** 2))


def risk_coverage(conf, err, thresholds_m=(25.0, 100.0)):
    """Rows sorted by descending confidence; risk at every coverage level."""
    order = np.argsort(-np.asarray(conf))
    e = np.asarray(err)[order]
    n = len(e)
    cov = np.arange(1, n + 1) / n
    out = {"coverage": cov,
           "median": np.array([np.median(e[: i + 1]) for i in range(n)]),
           "p95": np.array([np.percentile(e[: i + 1], 95) for i in range(n)])}
    for t in thresholds_m:
        out[f"fail@{int(t)}"] = np.cumsum(e > t) / np.arange(1, n + 1)
    return out


def aurc(conf, err, fail_m=25.0):
    rc = risk_coverage(conf, err, (fail_m,))
    return float(np.trapz(rc[f"fail@{int(fail_m)}"], rc["coverage"]))

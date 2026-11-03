"""Non-neural reference models (supplementary, NOT the core DL methodology).

EntityPrior : diagnostic.  Uses ONLY molecule identity - per-drug and per-excipient incompatibility
              rates seen in training, combined naive-Bayes style.  Unseen molecules fall back to the
              global prior.  If it scores well on a split, that split rewards memorising molecules,
              not learning chemistry.
RFBaseline  : random forest on the concatenated fingerprints (same input as the paper's ANN).
"""
from __future__ import annotations
import numpy as np


def _logit(p): p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))


class EntityPrior:
    def __init__(self, smoothing: float = 2.0): self.m = smoothing

    def fit(self, drug_cid, exc_cid, y):
        self.prior = float(np.mean(y))
        self.dstat, self.estat = {}, {}
        for stat, ids in ((self.dstat, drug_cid), (self.estat, exc_cid)):
            for i, yy in zip(ids, y):
                n, k = stat.get(i, (0, 0)); stat[i] = (n + 1, k + yy)
        return self

    def _rate(self, stat, i):
        n, k = stat.get(i, (0, 0))
        return (k + self.m * self.prior) / (n + self.m)

    def predict_score(self, drug_cid, exc_cid):
        pd_ = np.array([self._rate(self.dstat, i) for i in drug_cid])
        pe_ = np.array([self._rate(self.estat, i) for i in exc_cid])
        z = _logit(pd_) + _logit(pe_) - _logit(self.prior)
        return 1 / (1 + np.exp(-z))


class RFBaseline:
    def __init__(self, n_estimators: int = 300, seed: int = 0):
        from sklearn.ensemble import RandomForestClassifier
        self.clf = RandomForestClassifier(n_estimators=n_estimators, class_weight="balanced_subsample",
                                          n_jobs=-1, random_state=seed)

    def fit(self, Xa, Xb, y): self.clf.fit(np.hstack([Xa, Xb]), y); return self
    def predict_score(self, Xa, Xb): return self.clf.predict_proba(np.hstack([Xa, Xb]))[:, 1]

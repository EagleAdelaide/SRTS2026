"""WP8: the single predeclared shrinkage variant (AI-09).

SRTS-Shrink: log T_g(alpha) = log T_pool + alpha * (log T_g - log T_pool),
alpha in {0, .25, .5, .75, 1}. alpha is selected by an inner 5-fold
top-label-BCE CV *within the n calibration labels*: in each inner fold the
full SRTS pipeline (scaler, router, thresholds, temperatures) is refit on
the fold-train part only, shrunk temperatures are applied to the held-out
part, and the alpha with lowest mean held-out BCE wins; the final model is
refit on all n at the selected alpha. alpha=1 recovers SRTS-BCE exactly;
alpha=0 recovers TvA-TS exactly. One pilot + one confirmation; no further
variants regardless of outcome (protocol WEB_PROTOCOL.md §9).
"""
from __future__ import annotations
import numpy as np
from sklearn.model_selection import StratifiedKFold

from srts import SRTS, softmax

ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def _shrunk_temps(model: SRTS, alpha: float) -> np.ndarray:
    return np.exp(np.log(model.pooled_T)
                  + alpha * (np.log(model.temps) - np.log(model.pooled_T)))


def _topbce(probs, targets):
    c = np.clip(probs.max(1), 1e-12, 1 - 1e-12)
    b = (probs.argmax(1) == np.asarray(targets)).astype(float)
    return float(-(b * np.log(c) + (1 - b) * np.log(1 - c)).mean())


class SRTSShrink:
    def __init__(self, K=3, seed=0, inner_folds=5):
        self.K, self.seed, self.inner_folds = K, seed, inner_folds
        self.report = {}

    def fit(self, vl, vt):
        vl = np.asarray(vl, dtype=np.float64)
        vt = np.asarray(vt, dtype=np.int64)
        y_err = (vl.argmax(1) != vt).astype(int)
        minority = min(y_err.sum(), len(y_err) - y_err.sum())
        F = min(self.inner_folds, max(2, minority)) if minority >= 2 else 0
        scores = {a: [] for a in ALPHAS}
        if F >= 2:
            skf = StratifiedKFold(F, shuffle=True, random_state=self.seed)
            for tr, te in skf.split(vl, y_err):
                sub = SRTS(K=self.K, seed=self.seed, scaler="fold",
                           fold_fallback="train").fit(vl[tr], vt[tr])
                g_te = sub.route(vl[te])
                for a in ALPHAS:
                    T = _shrunk_temps(sub, a)[g_te][:, None]
                    scores[a].append(_topbce(softmax(vl[te] / T), vt[te]))
            means = {a: float(np.mean(v)) for a, v in scores.items()}
            self.alpha = min(means, key=means.get)
        else:
            means = {}
            self.alpha = 0.0
        self.base = SRTS(K=self.K, seed=self.seed, scaler="fold",
                         fold_fallback="train").fit(vl, vt)
        self.temps = _shrunk_temps(self.base, self.alpha)
        self.report = {"alpha": self.alpha, "inner_folds_used": F,
                       "cv_bce": means,
                       "temps": self.temps.tolist(),
                       "pooled_T": self.base.pooled_T}
        return self

    def predict_proba(self, logits):
        logits = np.asarray(logits, dtype=np.float64)
        T = self.temps[self.base.route(logits)][:, None]
        return softmax(logits / T)

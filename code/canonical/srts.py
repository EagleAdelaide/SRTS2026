"""Canonical SRTS-BCE (WWW 2027 revision).

Byte-equivalent six-signal construction to the reviewed artifact
(`calibrate_signal_router._logit_signals`); fitting semantics match
`srts_bce.SRTSBCE` with the following REVISION-CONTROLLED changes, each
switchable so the earlier-version behaviour remains reproducible:

  scaler="fold"   (new default) z-scoring moments fit on the fold-TRAIN part
                  only inside cross-fitting; deployment scaler refit on the
                  full calibration split. scaler="full" reproduces the
                  submitted (transductive-covariate) behaviour exactly.
  fold_fallback="train" (new default) a fold whose training part has <2
                  correctness classes imputes the FOLD-TRAIN error rate
                  (not the full-pool rate, which read held-out fold labels).
                  "pool" reproduces the submitted behaviour.
  Feasible fold count: F is reduced to min(F, minority correctness count)
                  when needed, recorded in the fit report (never silent).

Accounting comes from the fitted object (`accounting()`): optimized
coefficients, stored thresholds, stored moments, serialized bytes.

Partitions: equal-frequency quantiles for any K (deployed SRTS uses K=3),
plus `dyadic=True` genuinely nested partitions (K in {1,2,4,8}: thresholds
of the finer level contain the coarser level's thresholds by construction).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import io
import pickle
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

N_SIGNALS = 6
OOF_FOLDS = 5
MIN_GROUP = 50
T_CLAMP = (0.05, 20.0)
L2_C = 1.0
SIGNAL_NAMES = ["confidence", "logit_margin", "prob_margin",
                "pred_entropy", "logit_norm", "top_logit"]


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def six_signals(logits: np.ndarray) -> np.ndarray:
    """Identical math to artifact calibrate_signal_router._logit_signals."""
    log = np.asarray(logits, dtype=np.float64)
    log_shift = log - log.max(axis=1, keepdims=True)
    expv = np.exp(log_shift)
    probs = expv / expv.sum(axis=1, keepdims=True)
    sorted_logits = np.sort(log, axis=1)
    sorted_probs = np.sort(probs, axis=1)
    top_logit = sorted_logits[:, -1]
    second_logit = sorted_logits[:, -2]
    top_prob = sorted_probs[:, -1]
    second_prob = sorted_probs[:, -2]
    pred_entropy = -np.sum(probs * np.log(np.clip(probs, 1e-30, None)), axis=1)
    return np.stack([top_prob, top_logit - second_logit, top_prob - second_prob,
                     pred_entropy, np.linalg.norm(log, axis=1), top_logit],
                    axis=1).astype(np.float64)


def fit_bce_temperature(logits: np.ndarray, targets: np.ndarray,
                        clamp=T_CLAMP) -> tuple[float, dict]:
    """Bounded 1-D top-label-BCE temperature fit; returns (T, status)."""
    if len(targets) == 0:
        return 1.0, {"status": "empty", "boundary": False}
    b = (logits.argmax(axis=1) == targets).astype(np.float64)

    def obj(T):
        q = np.clip(softmax(logits / max(float(T), 1e-4)).max(axis=1),
                    1e-12, 1 - 1e-12)
        return float(-(b * np.log(q) + (1 - b) * np.log(1 - q)).mean())

    res = minimize_scalar(obj, bounds=clamp, method="bounded",
                          options={"xatol": 1e-8, "maxiter": 500})
    T = float(res.x)
    boundary = (abs(T - clamp[0]) < 1e-4) or (abs(T - clamp[1]) < 1e-3)
    return T, {"status": "ok" if res.success else "noconv",
               "fun": float(res.fun), "boundary": boundary,
               "nit": int(getattr(res, "nfev", -1))}


@dataclass
class SRTS:
    K: int = 3
    seed: int = 0
    folds: int = OOF_FOLDS
    min_group: int = MIN_GROUP
    l2_C: float = L2_C
    scaler: str = "fold"          # "fold" (revised) | "full" (earlier version)
    fold_fallback: str = "train"  # "train" (revised) | "pool" (earlier version)
    dyadic: bool = False          # nested dyadic quantile partition (WP3)
    # fitted state
    pooled_T: float = 1.0
    temps: np.ndarray = field(default=None)
    thresholds: np.ndarray = field(default=None)
    mean: np.ndarray = field(default=None)
    std: np.ndarray = field(default=None)
    coef: np.ndarray = field(default=None)
    intercept: float = 0.0
    report: dict = field(default_factory=dict)

    # ---------------- fitting ------------------------------------------
    def _fit_router_oof(self, S, y_risk):
        """Returns q_oof plus deployment (mean,std,coef,intercept)."""
        n = len(y_risk)
        q_oof = np.zeros(n, dtype=np.float64)
        n_err = int(y_risk.sum())
        minority = min(n_err, n - n_err)
        fold_log = []
        if minority == 0:
            # degenerate: all right or all wrong
            p = np.clip(y_risk.mean(), 1e-6, 1 - 1e-6)
            self.report["router"] = {"degenerate": True, "folds_used": 0}
            return (np.full(n, y_risk.mean()), S.mean(0),
                    np.where(S.std(0) < 1e-12, 1.0, S.std(0)),
                    np.zeros(S.shape[1]), float(np.log(p / (1 - p))))
        F = min(self.folds, max(2, minority)) if minority >= 2 else 0
        if F >= 2:
            skf = StratifiedKFold(n_splits=F, shuffle=True,
                                  random_state=self.seed)
            for tr, te in skf.split(S, y_risk):
                if self.scaler == "fold":
                    mu = S[tr].mean(0)
                    sd = np.where(S[tr].std(0) < 1e-12, 1.0, S[tr].std(0))
                else:
                    mu = S.mean(0)
                    sd = np.where(S.std(0) < 1e-12, 1.0, S.std(0))
                if len(set(y_risk[tr].tolist())) < 2:
                    fill = (y_risk[tr].mean() if self.fold_fallback == "train"
                            else y_risk.mean())
                    q_oof[te] = fill
                    fold_log.append({"fold": len(fold_log), "fallback": True})
                    continue
                clf = LogisticRegression(penalty="l2", C=self.l2_C,
                                         solver="lbfgs", max_iter=1000,
                                         random_state=self.seed)
                clf.fit((S[tr] - mu) / sd, y_risk[tr])
                q_oof[te] = clf.predict_proba((S[te] - mu) / sd)[:, 1]
                fold_log.append({"fold": len(fold_log), "fallback": False,
                                 "converged": int(clf.n_iter_[0]) < 1000})
        else:
            fill = y_risk.mean()
            q_oof[:] = fill
            fold_log.append({"fold": 0, "fallback": True,
                             "reason": f"minority={minority}"})
        # deployment refit on full split (scaler refit on full split too)
        mu_f = S.mean(0)
        sd_f = np.where(S.std(0) < 1e-12, 1.0, S.std(0))
        clf_final = LogisticRegression(penalty="l2", C=self.l2_C,
                                       solver="lbfgs", max_iter=1000,
                                       random_state=self.seed)
        clf_final.fit((S - mu_f) / sd_f, y_risk)
        self.report["router"] = {"degenerate": False,
                                 "requested_folds": self.folds,
                                 "folds_used": F, "minority": minority,
                                 "folds": fold_log,
                                 "final_converged": int(clf_final.n_iter_[0]) < 1000}
        return (q_oof, mu_f, sd_f, clf_final.coef_[0].astype(np.float64),
                float(clf_final.intercept_[0]))

    def _partition_thresholds(self, q_oof):
        if self.dyadic:
            # genuinely nested: K must be power of 2; thresholds at j/K
            assert self.K & (self.K - 1) == 0, "dyadic needs K power of 2"
            qs = [j / self.K for j in range(1, self.K)]
        else:
            qs = [j / self.K for j in range(1, self.K)]
        return np.quantile(q_oof, qs) if qs else np.array([])

    def fit(self, val_logits, val_targets):
        vl = np.asarray(val_logits, dtype=np.float64)
        vt = np.asarray(val_targets, dtype=np.int64)
        self.report = {"n_fit": int(len(vt)), "scaler": self.scaler,
                       "fold_fallback": self.fold_fallback, "K": self.K}
        self.pooled_T, st = fit_bce_temperature(vl, vt)
        self.report["pooled_T"] = {"T": self.pooled_T, **st}
        if self.K == 1:
            self.temps = np.array([self.pooled_T])
            self.thresholds = np.array([])
            self.mean = np.zeros(N_SIGNALS)
            self.std = np.ones(N_SIGNALS)
            self.coef = np.zeros(N_SIGNALS)
            self.intercept = 0.0
            self.report["groups"] = [{"g": 0, "n": int(len(vt)),
                                      "fallback": False, "T": self.pooled_T}]
            return self
        S = six_signals(vl)
        y_risk = (vl.argmax(axis=1) != vt).astype(np.int64)
        q_oof, self.mean, self.std, self.coef, self.intercept = \
            self._fit_router_oof(S, y_risk)
        self.thresholds = self._partition_thresholds(q_oof)
        g = np.clip(np.digitize(q_oof, self.thresholds, right=False),
                    0, self.K - 1)
        temps, glog = [], []
        for grp in range(self.K):
            m = g == grp
            if m.sum() >= self.min_group:
                T, st = fit_bce_temperature(vl[m], vt[m])
                fb = False
            else:
                T, st, fb = self.pooled_T, {"status": "fallback"}, True
            temps.append(T)
            glog.append({"g": grp, "n": int(m.sum()),
                         "err": int(y_risk[m].sum()), "fallback": fb,
                         "T": float(T), **st})
        self.temps = np.asarray(temps, dtype=np.float64)
        self.report["groups"] = glog
        self.report["active_temperature_count"] = int(
            sum(1 for r in glog if not r["fallback"]))
        self.report["fallback_rate"] = float(
            sum(1 for r in glog if r["fallback"]) / self.K)
        return self

    # ---------------- deployment ---------------------------------------
    def score(self, logits):
        S = (six_signals(np.asarray(logits, dtype=np.float64)) - self.mean) / self.std
        return 1.0 / (1.0 + np.exp(-(S @ self.coef + self.intercept)))

    def route(self, logits):
        if self.K == 1:
            return np.zeros(len(logits), dtype=int)
        return np.clip(np.digitize(self.score(logits), self.thresholds,
                                   right=False), 0, self.K - 1).astype(int)

    def predict_proba(self, logits):
        logits = np.asarray(logits, dtype=np.float64)
        T = self.temps[self.route(logits)][:, None]
        return softmax(logits / T)

    # ---------------- accounting ----------------------------------------
    def accounting(self) -> dict:
        opt = (0 if self.K == 1 else N_SIGNALS + 1) + self.K
        stored = {"thresholds": 0 if self.K == 1 else int(len(self.thresholds)),
                  "moments": 0 if self.K == 1 else 2 * N_SIGNALS}
        buf = io.BytesIO()
        pickle.dump({"temps": self.temps, "thresholds": self.thresholds,
                     "mean": self.mean, "std": self.std, "coef": self.coef,
                     "intercept": self.intercept, "K": self.K}, buf)
        return {"optimized_coefficients": opt,
                "stored_state": stored,
                "serialized_bytes": int(buf.getbuffer().nbytes)}


# ---------------------------------------------------------------------------
def fit_tva(vl, vt):
    """TvA-TS = SRTS K=1 (exact reduction)."""
    return SRTS(K=1).fit(vl, vt)


@dataclass
class MarginK:
    """Raw logit-margin quantile groups, one BCE temperature each (Margin-K3)."""
    K: int = 3
    min_group: int = MIN_GROUP
    pooled_T: float = 1.0
    temps: np.ndarray = None
    thresholds: np.ndarray = None

    @staticmethod
    def _margin(logits):
        s = np.sort(np.asarray(logits, dtype=np.float64), axis=1)
        return s[:, -1] - s[:, -2]

    def fit(self, vl, vt):
        vl = np.asarray(vl, dtype=np.float64)
        vt = np.asarray(vt, dtype=np.int64)
        self.pooled_T, _ = fit_bce_temperature(vl, vt)
        m = self._margin(vl)
        self.thresholds = np.quantile(m, [j / self.K for j in range(1, self.K)])
        g = np.clip(np.digitize(m, self.thresholds), 0, self.K - 1)
        temps = []
        for grp in range(self.K):
            mm = g == grp
            temps.append(fit_bce_temperature(vl[mm], vt[mm])[0]
                         if mm.sum() >= self.min_group else self.pooled_T)
        self.temps = np.asarray(temps)
        return self

    def predict_proba(self, logits):
        logits = np.asarray(logits, dtype=np.float64)
        g = np.clip(np.digitize(self._margin(logits), self.thresholds),
                    0, self.K - 1)
        return softmax(logits / self.temps[g][:, None])

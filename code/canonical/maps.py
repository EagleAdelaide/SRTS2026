"""Scalar, HTS, and continuous risk-conditioned temperature maps (canonical).

Ports of the reviewed-artifact implementations with unified fit logging
(convergence, boundary hits, wall time). All fitting uses only the provided
calibration (vl, vt); no test labels anywhere.

Contents:
  fit_ts_nll            classical scalar NLL temperature (TS-NLL)
  HTS                   entropy map T=softplus(w*log Hbar + b); objective
                        "nll" (native) or "bce"; Nelder-Mead from scalar init
  RiskMap               continuous maps on the SRTS OOF risk score:
                        kind in {"linear","pw3","spline"} (matched to the earlier version) and
                        "pwK" with n_anchors for the WP3 nested-knot family
"""
from __future__ import annotations
import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar

from srts import softmax, T_CLAMP


def softplus(x):
    return np.logaddexp(0.0, np.asarray(x, dtype=np.float64))


def softplus_inv(y):
    y = float(y)
    return y + np.log1p(-np.exp(-y))


def toplabel_bce_perT(logits, correct, T):
    T = np.clip(np.asarray(T, dtype=np.float64), 1e-4, None)
    p = np.clip(softmax(logits / T[:, None]).max(axis=1), 1e-12, 1 - 1e-12)
    return float(-np.mean(correct * np.log(p) + (1 - correct) * np.log(1 - p)))


def nll_perT(logits, targets, T):
    T = np.clip(np.asarray(T, dtype=np.float64), 1e-4, None)
    P = softmax(logits / T[:, None])
    return float(-np.mean(np.log(np.clip(P[np.arange(len(targets)), targets],
                                         1e-12, None))))


def fit_ts_nll(vl, vt, clamp=T_CLAMP):
    """Scalar temperature minimizing multiclass val NLL (TS-NLL)."""
    vl = np.asarray(vl, dtype=np.float64)
    vt = np.asarray(vt, dtype=np.int64)

    def obj(T):
        return nll_perT(vl, vt, np.full(len(vt), float(T)))

    res = minimize_scalar(obj, bounds=clamp, method="bounded",
                          options={"xatol": 1e-8, "maxiter": 500})
    T = float(res.x)
    report = {"T": T, "fun": float(res.fun),
              "boundary": abs(T - clamp[0]) < 1e-4 or abs(T - clamp[1]) < 1e-3}

    class _M:
        temps = np.array([T])

        @staticmethod
        def predict_proba(logits):
            return softmax(np.asarray(logits, dtype=np.float64) / T)
    _M.report = report
    return _M


def norm_log_entropy(z):
    P = softmax(np.asarray(z, dtype=np.float64))
    H = -np.sum(P * np.log(np.clip(P, 1e-30, None)), axis=1)
    Hbar = np.clip(H / np.log(P.shape[1]), 1e-12, None)
    return np.log(Hbar)


class HTS:
    """Balanya et al. entropy-conditioned 2-parameter temperature map."""

    def __init__(self, objective="nll"):
        assert objective in ("nll", "bce")
        self.objective = objective
        self.params = None
        self.report = {}

    def fit(self, vl, vt):
        vl = np.asarray(vl, dtype=np.float64)
        vt = np.asarray(vt, dtype=np.int64)
        u = norm_log_entropy(vl)
        correct = (vl.argmax(1) == vt).astype(np.float64)
        # init at matching-objective scalar
        if self.objective == "nll":
            T0 = fit_ts_nll(vl, vt).temps[0]
            def loss(params):
                return nll_perT(vl, vt, softplus(params[0] * u + params[1]))
        else:
            def sobj(T):
                return toplabel_bce_perT(vl, correct, np.full(len(vt), T))
            T0 = float(minimize_scalar(sobj, bounds=T_CLAMP, method="bounded",
                                       options={"xatol": 1e-8}).x)
            def loss(params):
                return toplabel_bce_perT(vl, correct,
                                         softplus(params[0] * u + params[1]))
        x0 = np.array([0.0, softplus_inv(T0)])
        t0 = time.time()
        res = minimize(loss, x0, method="Nelder-Mead",
                       options={"maxiter": 4000, "xatol": 1e-8, "fatol": 1e-10})
        self.params = np.asarray(res.x, dtype=np.float64)
        self.report = {"objective": self.objective, "converged": bool(res.success),
                       "fun": float(res.fun), "init_T0": float(T0),
                       "nit": int(res.nit), "seconds": time.time() - t0}
        return self

    def temperature(self, logits):
        return softplus(self.params[0] * norm_log_entropy(logits)
                        + self.params[1])

    def predict_proba(self, logits):
        logits = np.asarray(logits, dtype=np.float64)
        return softmax(logits / self.temperature(logits)[:, None])


def natural_cubic_basis(q, knots):
    q = np.asarray(q, dtype=np.float64)
    k1, k2, k3 = knots

    def d(k):
        return (np.maximum(q - k, 0.0) ** 3
                - np.maximum(q - k3, 0.0) ** 3) / (k3 - k)

    return np.column_stack([np.ones_like(q), q, d(k1) - d(k2)])


class RiskMap:
    """Continuous temperature map on a shared risk score (BCE objective).

    kind: "linear" (2 coef), "pw3" (3 anchors @ q{1/6,1/2,5/6}),
          "spline" (natural cubic, 3 coef, knots @ q{.25,.5,.75}),
          "pwK" with n_anchors in {2,4,6,...} for the WP3 nested-knot family
          (anchors at equally spaced quantiles; n_anchors=1 == scalar).
    Fitting: L-BFGS-B with Nelder-Mead fallback, init at pooled BCE scalar.
    """

    def __init__(self, kind="pw3", n_anchors=None):
        self.kind = kind
        self.n_anchors = n_anchors
        self.theta = None
        self.anchors = None
        self.knots = None
        self.report = {}

    def _predictor(self, q_fit):
        if self.kind == "linear":
            def f(th, q):
                return softplus(th[0] + th[1] * np.asarray(q))
            return f, 2
        if self.kind == "pw3" or (self.kind == "pwK" and self.n_anchors):
            m = 3 if self.kind == "pw3" else int(self.n_anchors)
            qs = ([1 / 6, 1 / 2, 5 / 6] if self.kind == "pw3"
                  else [(2 * j + 1) / (2 * m) for j in range(m)])
            self.anchors = np.quantile(q_fit, qs)

            def f(th, q):
                return np.interp(np.asarray(q), self.anchors,
                                 softplus(np.asarray(th)))
            return f, m
        if self.kind == "spline":
            self.knots = np.quantile(q_fit, [0.25, 0.5, 0.75])

            def f(th, q):
                return softplus(natural_cubic_basis(q, self.knots)
                                @ np.asarray(th))
            return f, 3
        if self.kind == "poly":
            # genuinely NESTED family: T = softplus(sum_j th_j q^j),
            # coef count m in {1,2,3,4}; m=1 == pooled BCE scalar.
            m = int(self.n_anchors)

            def f(th, q):
                q = np.asarray(q, dtype=np.float64)
                acc = np.zeros_like(q)
                for j, c in enumerate(np.asarray(th)):
                    acc = acc + c * q ** j
                return softplus(acc)
            return f, m
        raise ValueError(self.kind)

    def fit(self, vl, vt, q_fit):
        vl = np.asarray(vl, dtype=np.float64)
        vt = np.asarray(vt, dtype=np.int64)
        correct = (vl.argmax(1) == vt).astype(np.float64)

        def sobj(T):
            return toplabel_bce_perT(vl, correct, np.full(len(vt), T))
        T0 = float(minimize_scalar(sobj, bounds=T_CLAMP, method="bounded",
                                   options={"xatol": 1e-8}).x)
        f_T, m = self._predictor(np.asarray(q_fit))
        a0 = softplus_inv(T0)
        if self.kind == "linear":
            x0 = np.array([a0, 0.0])
        elif self.kind == "spline":
            x0 = np.array([a0, 0.0, 0.0])
        elif self.kind == "poly":
            x0 = np.zeros(m); x0[0] = a0
        else:
            x0 = np.full(m, a0)

        def obj(th):
            return toplabel_bce_perT(vl, correct, f_T(th, q_fit))

        t0 = time.time()
        res = minimize(obj, x0, method="L-BFGS-B")
        fallback = False
        if (not res.success) or (not np.isfinite(res.fun)):
            res_nm = minimize(obj, x0, method="Nelder-Mead",
                              options={"maxiter": 5000, "xatol": 1e-8,
                                       "fatol": 1e-10})
            if np.isfinite(res_nm.fun) and (not np.isfinite(res.fun)
                                            or res_nm.fun <= res.fun):
                res, fallback = res_nm, True
        self.theta = np.asarray(res.x, dtype=np.float64)
        self._f = f_T
        self.report = {"kind": self.kind, "n_coef": int(m),
                       "converged": bool(res.success),
                       "fallback": fallback, "fun": float(res.fun),
                       "init_T0": T0, "seconds": time.time() - t0}
        return self

    def predict_proba(self, logits, q):
        logits = np.asarray(logits, dtype=np.float64)
        T = np.clip(self._f(self.theta, q), 1e-4, None)
        return softmax(logits / T[:, None])

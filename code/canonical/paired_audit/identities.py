"""Exact identities for accepted-set mean calibration bias (plan A8).

Notation: for a map m with top-label confidence c_m (array over N items),
shared correctness b in {0,1}^N (argmax preserved), and an accepted set
A_m (boolean mask), g_m = mean_{A_m}(c_m - b).

Identity 1 (mechanism accounting; exact):
    g_C - g_B = S + P,
    S = mean_{A_C}(c_C - c_B)                      (score change on C's set)
    P = mean_{A_C}(c_B - b) - mean_{A_B}(c_B - b)  (selection change, baseline residual)
Proof: g_C = mean_{A_C}(c_C - b) = mean_{A_C}(c_C - c_B) + mean_{A_C}(c_B - b);
subtract g_B = mean_{A_B}(c_B - b).  Both A_C and A_B must be non-empty.
The identity is for the signed difference; |g_C| - |g_B| is not additive in S, P.

Identity 2 (audit weights; exact):
    g_C - g_B = d_conf - sum_i w_i b_i,
    w_i = 1[i in A_C]/|A_C| - 1[i in A_B]/|A_B|,
    d_conf = mean_{A_C}(c_C) - mean_{A_B}(c_B)   (label-free).
Proof: g_C - g_B = [mean_{A_C} c_C - mean_{A_B} c_B] - [mean_{A_C} b - mean_{A_B} b]
and mean_{A_C} b - mean_{A_B} b = sum_i (1[A_C]/|A_C| - 1[A_B]/|A_B|) b_i.
Special case |A_C| = |A_B| = k: w_i = 0 on the overlap, +1/k on A_C \\ A_B,
-1/k on A_B \\ A_C, so the label term depends only on the symmetric difference.
"""
import numpy as np


def accepted_fixed(c, t):
    """Fixed-threshold accept set {c >= t}."""
    return np.asarray(c) >= t


def accepted_matched(c, k):
    """Top-k by own confidence (stable sort; ties broken by index), k >= 0."""
    c = np.asarray(c)
    mask = np.zeros(len(c), dtype=bool)
    if k > 0:
        mask[np.argsort(-c, kind="stable")[:k]] = True
    return mask


def gap(c, b, A):
    A = np.asarray(A, dtype=bool)
    if not A.any():
        return float("nan")
    return float(np.mean(np.asarray(c, dtype=np.float64)[A] - np.asarray(b, dtype=np.float64)[A]))


def decompose(cC, cB, b, AC, AB):
    """Identity 1 components. Returns dict with gC, gB, S, P, residual (= gC-gB-S-P)."""
    cC = np.asarray(cC, dtype=np.float64); cB = np.asarray(cB, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    AC = np.asarray(AC, dtype=bool); AB = np.asarray(AB, dtype=bool)
    gC, gB = gap(cC, b, AC), gap(cB, b, AB)
    S = float(np.mean(cC[AC] - cB[AC])) if AC.any() else float("nan")
    P = float(np.mean(cB[AC] - b[AC]) - np.mean(cB[AB] - b[AB])) if (AC.any() and AB.any()) else float("nan")
    return {"gC": gC, "gB": gB, "S": S, "P": P, "residual": gC - gB - S - P,
            "absdiff": abs(gC) - abs(gB), "cov_C": float(AC.mean()), "cov_B": float(AB.mean()),
            "overlap": float((AC & AB).sum() / max(1, (AC | AB).sum()))}


def audit_weights(AC, AB):
    """Identity 2 weights w_i and the label-free part needs c; returns w."""
    AC = np.asarray(AC, dtype=bool); AB = np.asarray(AB, dtype=bool)
    nC, nB = AC.sum(), AB.sum()
    if nC == 0 or nB == 0:
        raise ValueError("empty accepted set")
    return AC / nC - AB / nB


def d_conf(cC, cB, AC, AB):
    cC = np.asarray(cC, dtype=np.float64); cB = np.asarray(cB, dtype=np.float64)
    return float(cC[np.asarray(AC, bool)].mean() - cB[np.asarray(AB, bool)].mean())


def identity2_check(cC, cB, b, AC, AB):
    w = audit_weights(AC, AB)
    lhs = gap(cC, b, AC) - gap(cB, b, AB)
    rhs = d_conf(cC, cB, AC, AB) - float(np.dot(w, np.asarray(b, dtype=np.float64)))
    return lhs, rhs

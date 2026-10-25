"""Accepted sets inside a target pool (plan A10 i).

Three branches:
  fixed                          A_t = {i in P : c_i >= t}
  stream_matched_then_restricted the A8/A9 construction (kept for the record): top-k
                                 sets on the full split with k = #{c_B >= t}, then
                                 intersected with P; counts inside P need NOT match
  exact_matched_within_pool      k_P = #{i in P : c_B,i >= t}; A_B^P, A_C^P = the top-k_P
                                 items of P by c_B and by c_C; ties broken by
                                 (-c, item index) with a stable sort; |A_B^P| = |A_C^P|
                                 is asserted, so the overlap weight 1/|A_C|-1/|A_B| is 0.
"""
import numpy as np

BRANCHES = ("fixed", "stream_matched_then_restricted", "exact_matched_within_pool")


def topk_in(c, pool, k):
    """Top-k items of `pool` (bool mask) by c, ties by index (stable)."""
    idx = np.where(pool)[0]
    out = np.zeros(len(c), dtype=bool)
    if k > 0:
        order = np.argsort(-np.asarray(c)[idx], kind="stable")
        out[idx[order[:k]]] = True
    return out


def accepted_sets(cB, cC, pool, t, branch):
    """Return (A_C, A_B) as bool masks restricted to `pool`."""
    cB = np.asarray(cB); cC = np.asarray(cC); pool = np.asarray(pool, bool)
    if branch == "fixed":
        return (cC >= t) & pool, (cB >= t) & pool
    if branch == "stream_matched_then_restricted":
        k = int((cB >= t).sum())
        full = np.ones(len(cB), bool)
        return topk_in(cC, full, k) & pool, topk_in(cB, full, k) & pool
    if branch == "exact_matched_within_pool":
        kP = int(((cB >= t) & pool).sum())
        AC, AB = topk_in(cC, pool, kP), topk_in(cB, pool, kP)
        assert AC.sum() == AB.sum() == kP, "exact matching failed"
        return AC, AB
    raise ValueError(branch)

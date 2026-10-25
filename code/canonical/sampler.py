"""Exact-size calibration budget sampler (fixes NEW-01 / audit issue B1).

The earlier artifact's `strat_subsample` rounded per-class quotas down and only
truncated overshoot, so nominal n=250 produced 244-247 actual samples. This
module provides a deterministic exact-size allocator:

  * largest-remainder (Hamilton) apportionment of n over classes proportional
    to pool class counts, with deterministic tie-breaking (by fractional part,
    then class id);
  * per-class caps at the available count, with deficit redistributed to the
    classes with the most remaining items (deterministic);
  * assertion: len(idx)==n, no duplicates, all indices in the allowed pool.

Also provides `natural_subsample` (unstratified uniform draw) for the
stratification-sensitivity analysis, and a provenance record.
"""
from __future__ import annotations
import hashlib
import numpy as np


def _pool_hash(labels: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(labels).tobytes()).hexdigest()[:16]


def exact_stratified_subsample(labels: np.ndarray, n: int, rng: np.random.Generator,
                               pool_idx: np.ndarray | None = None):
    """Return (idx, record). idx has EXACTLY n unique entries drawn from
    pool_idx (default: all), class-proportional by largest remainder."""
    labels = np.asarray(labels)
    if pool_idx is None:
        pool_idx = np.arange(len(labels))
    pool_idx = np.asarray(pool_idx)
    y = labels[pool_idx]
    N = len(pool_idx)
    if n > N:
        raise ValueError(f"requested n={n} > pool size {N}")
    classes, counts = np.unique(y, return_counts=True)
    quota = n * counts / N
    base = np.floor(quota).astype(int)
    rem = n - base.sum()
    frac = quota - base
    # deterministic: largest fractional part first, ties by class id
    order = np.lexsort((classes, -frac))
    take = base.copy()
    for j in order[:rem]:
        take[j] += 1
    # cap at available and redistribute deficit deterministically
    over = take - counts
    deficit = int(np.clip(over, 0, None).sum())
    take = np.minimum(take, counts)
    if deficit:
        room = counts - take
        order2 = np.lexsort((classes, -room))
        for j in order2:
            if deficit == 0:
                break
            add = int(min(room[j], deficit))
            take[j] += add
            deficit -= add
    assert take.sum() == n, (take.sum(), n)
    idx = []
    for c, k in zip(classes, take):
        ci = pool_idx[np.where(y == c)[0]]
        if k > 0:
            idx.extend(rng.choice(ci, size=int(k), replace=False).tolist())
    idx = np.array(sorted(idx))
    assert len(idx) == n and len(np.unique(idx)) == n
    assert np.isin(idx, pool_idx).all()
    sel_counts = {int(c): int(k) for c, k in zip(classes, take) if k > 0}
    record = dict(requested_n=int(n), actual_n=int(len(idx)),
                  sampler="exact_stratified_largest_remainder_v1",
                  pool_size=int(N), pool_hash=_pool_hash(labels[pool_idx]),
                  class_counts=sel_counts,
                  bitgen=type(rng.bit_generator).__name__)
    return idx, record


def natural_subsample(labels: np.ndarray, n: int, rng: np.random.Generator,
                      pool_idx: np.ndarray | None = None):
    """Unstratified uniform draw of exactly n (sensitivity protocol)."""
    labels = np.asarray(labels)
    if pool_idx is None:
        pool_idx = np.arange(len(labels))
    pool_idx = np.asarray(pool_idx)
    if n > len(pool_idx):
        raise ValueError("n > pool")
    idx = np.sort(rng.choice(pool_idx, size=n, replace=False))
    classes, counts = np.unique(labels[idx], return_counts=True)
    record = dict(requested_n=int(n), actual_n=int(len(idx)),
                  sampler="natural_uniform_v1", pool_size=int(len(pool_idx)),
                  pool_hash=_pool_hash(labels[pool_idx]),
                  class_counts={int(c): int(k) for c, k in zip(classes, counts)},
                  bitgen=type(rng.bit_generator).__name__)
    return idx, record


def legacy_quota_subsample(labels: np.ndarray, n: int, rng: np.random.Generator):
    """EXACT reimplementation of the earlier artifact sampler (round-down +
    truncate-only). Used ONLY to document the legacy actual-n deficit."""
    labels = np.asarray(labels)
    idx = []
    N = len(labels)
    for c in np.unique(labels):
        ci = np.where(labels == c)[0]
        k = min(len(ci), max(1, int(round(n * len(ci) / N))))
        idx.extend(rng.choice(ci, k, replace=False).tolist())
    idx = np.array(idx)
    if len(idx) > n:
        idx = rng.choice(idx, n, replace=False)
    return idx

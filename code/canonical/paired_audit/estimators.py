"""Finite-population estimators of the label term L = sum_i w_i b_i (plan A8 iii)
and of the baseline level g_B, with Horvitz-Thompson variance estimates.

All designs are stratified simple random sampling without replacement
(SRSWOR) with known strata; the HT estimator of a population total
T = sum_{i in U} y_i under SRSWOR of n_h from stratum h (size N_h) is
    T_hat = sum_h (N_h / n_h) sum_{sampled i in h} y_i,
    Var_hat = sum_h N_h^2 (1 - n_h/N_h) s_h^2 / n_h,   s_h^2 = sample variance of y in h
(exact unbiasedness of T_hat for SRSWOR; Var_hat unbiased when n_h >= 2;
strata with n_h = 1 contribute zero variance, i.e. the variance is
under-estimated there, so allocations enforce n_h >= 2 where N_h >= 2).
Designs (unique labels m):
  E1 stream   : one stratum = the whole evaluation stream (labels wasted where w_i = 0)
  E2 separate : SRS of m/2 from A_B and m/2 from A_C (estimates mean_{A_B} b and
                mean_{A_C} b separately; an item drawn in both costs one label but
                the two estimates are formed from the two samples)
  E3 union    : one stratum = A_B u A_C
  E4 switcher : strata O = A_B n A_C, C\\B, B\\C with Neyman-type allocation
                n_h proportional to N_h |w_h| (w constant within a stratum),
                floor 2 per non-empty stratum (if N_h >= 2), remainder to the
                largest N_h|w_h| stratum; total unique labels = m
  E0 level    : SRS of m from A_B for g_B = mean_{A_B} c_B - mean_{A_B} b
Plan A9 comparators (need a label-free surrogate s_i ~ P(b_i = 1); we use c_B):
  E5 active IS : m draws WITH replacement from A_B u A_C, q_i prop. to |w_i| sqrt(s_i)
                 (Kossen et al. 2021 style), Hansen-Hurwitz estimator
                 (1/m) sum_j y_j/q_j, variance s^2(y/q)/m; unique labels <= m
  E6 active PPI: sum_i w_i s_i + Hansen-Hurwitz estimate of sum_i w_i (b_i - s_i),
                 q = 0.8 q* + 0.2 q_w, q* prop. to |w_i| sqrt(s_i(1 - s_i)),
                 q_w prop. to |w_i| (fixed defensive mixture)
  E7 strat-diff: E4 strata and allocation, HT estimate of sum_i w_i (b_i - s_i)
                 plus the known sum_i w_i s_i
Plan A10 error-centred controls (sum_i w_i = 0, so L = sum w b = -sum w e, e = 1 - b;
surrogate error s_e = 1 - c_B):
  E8  error-centred IS : with replacement over supp(w), q = 0.8 q* + 0.2 q_w,
                         q*_i prop. to |w_i| sqrt(s_e,i), q_w prop. to |w_i|; Hansen-Hurwitz
  E9  error-centred union: SRSWOR on A_B u A_C, Horvitz-Thompson total of w e
  E10 active inference (Zrnic & Candes 2024 form): Poisson sampling with
                         pi_i = min(1, kappa a_i), a = 0.8 |w| sqrt(s_e(1-s_e)) + 0.2 |w|
                         (normalised), kappa set so sum pi = m; AIPW estimator
                         sum w s_e + sum_sampled w (e - s_e)/pi; label cost random, mean m
  E11 LURE active testing (Kossen et al. 2021 form): sequential sampling without
                         replacement over supp(w) from a static proposal 0.8 q*_e + 0.2 uniform,
                         LURE weights (Farquhar et al. 2021); exactly m distinct labels;
                         variance = empirical variance of the weighted terms / m (approximate)
"""
import numpy as np


def _srswor_total(y_pop, n, rng):
    """HT total and variance for SRSWOR of n from y_pop (array)."""
    N = len(y_pop)
    n = int(min(max(n, 0), N))
    if N == 0 or n == 0:
        return 0.0, 0.0, 0, np.zeros(0, dtype=int)
    idx = rng.choice(N, size=n, replace=False)
    y = y_pop[idx]
    tot = N * y.mean()
    var = N * N * (1 - n / N) * (y.var(ddof=1) if n >= 2 else 0.0) / n
    return float(tot), float(var), n, idx


def _allocate(sizes, weights, m, floor=2):
    """Neyman-type allocation proportional to N_h |w_h| with a floor."""
    sizes = np.asarray(sizes, dtype=int); weights = np.abs(np.asarray(weights, dtype=float))
    n = np.zeros(len(sizes), dtype=int)
    live = sizes > 0
    for h in np.where(live)[0]:
        n[h] = min(floor, sizes[h])
    rem = m - n.sum()
    if rem > 0:
        score = sizes * weights * live
        if score.sum() > 0:
            add = np.floor(rem * score / score.sum()).astype(int)
            add = np.minimum(add, sizes - n)
            n += add
            rem = m - n.sum()
            order = np.argsort(-score)
            for h in order:
                if rem <= 0:
                    break
                room = sizes[h] - n[h]
                take = min(room, rem); n[h] += take; rem -= take
    return n


def _hansen_hurwitz(q, y, m, rng):
    """m i.i.d. draws from q (sums to 1); unbiased estimate of sum(y) where
    q > 0 covers every y != 0; returns est, var, unique draws."""
    cdf = np.cumsum(q); cdf /= cdf[-1]
    j = np.minimum(np.searchsorted(cdf, rng.random(m), side="right"), len(q) - 1)
    z = y[j] / q[j]
    var = z.var(ddof=1) / m if m >= 2 else 0.0
    return float(z.mean()), float(var), int(len(np.unique(j)))


class LabelTerm:
    """Estimate L = sum_i w_i b_i with unique-label budget m."""

    def __init__(self, w, AC, AB, surrogate=None):
        self.w = np.asarray(w, dtype=np.float64)
        self.AC = np.asarray(AC, bool); self.AB = np.asarray(AB, bool)
        self.N = len(self.w)
        self.s = None if surrogate is None else np.clip(np.asarray(surrogate, dtype=np.float64), 0.0, 1.0)
        if self.s is not None:
            self.U = np.where(self.AC | self.AB)[0]
            aw, sU = np.abs(self.w[self.U]), self.s[self.U]
            if aw.sum() == 0:                       # identical sets: no label information needed
                aw = np.ones_like(aw)
            q5 = aw * np.sqrt(sU)
            self.q5 = q5 / q5.sum()
            qw = aw / aw.sum()
            qs = aw * np.sqrt(sU * (1 - sU))
            self.q6 = 0.8 * (qs / qs.sum() if qs.sum() > 0 else qw) + 0.2 * qw
            self.offset = float(np.dot(self.w, self.s))

    def estimate(self, design, b, m, rng):
        b = np.asarray(b, dtype=np.float64)
        y = self.w * b
        if design == "E1":
            tot, var, n, idx = _srswor_total(y, m, rng)
            return tot, var, n
        if design == "E3":
            U = np.where(self.AC | self.AB)[0]
            tot, var, n, idx = _srswor_total(y[U], m, rng)
            return tot, var, n
        if design == "E2":
            iB, iC = np.where(self.AB)[0], np.where(self.AC)[0]
            mB, mC = m // 2, m - m // 2
            tB, vB, nB, idB = _srswor_total(b[iB], mB, rng)   # total of b over A_B
            tC, vC, nC, idC = _srswor_total(b[iC], mC, rng)
            est = tC / len(iC) - tB / len(iB)
            var = vC / len(iC) ** 2 + vB / len(iB) ** 2
            uniq = len(set(iB[idB].tolist()) | set(iC[idC].tolist()))
            return float(est), float(var), uniq
        if design == "E4":
            strata = [np.where(self.AC & self.AB)[0], np.where(self.AC & ~self.AB)[0], np.where(self.AB & ~self.AC)[0]]
            wh = [1 / self.AC.sum() - 1 / self.AB.sum(), 1 / self.AC.sum(), -1 / self.AB.sum()]
            sizes = [len(s) for s in strata]
            alloc = _allocate(sizes, wh, m)
            tot = var = 0.0; n = 0
            for s_idx, nh in zip(strata, alloc):
                t_, v_, n_, _ = _srswor_total(y[s_idx], nh, rng)
                tot += t_; var += v_; n += n_
            return float(tot), float(var), n
        if design == "E4ac":
            # E4's sample and estimator; stratum variance from an Agresti-Coull error
            # proportion p~ = (x + 2)/(n + 4), never zero (plan A10, coverage repair)
            strata = [np.where(self.AC & self.AB)[0], np.where(self.AC & ~self.AB)[0], np.where(self.AB & ~self.AC)[0]]
            wh = [1 / self.AC.sum() - 1 / self.AB.sum(), 1 / self.AC.sum(), -1 / self.AB.sum()]
            alloc = _allocate([len(x) for x in strata], wh, m)
            tot = var = 0.0; n = 0
            for s_idx, nh, w_ in zip(strata, alloc, wh):
                Nh = len(s_idx); nh = int(min(nh, Nh))
                if Nh == 0 or nh == 0:
                    continue
                sel = s_idx[rng.choice(Nh, size=nh, replace=False)]
                tot += w_ * Nh * b[sel].mean()
                x = float((1 - b[sel]).sum()); pt = (x + 2) / (nh + 4)
                var += (w_ * Nh) ** 2 * (1 - nh / Nh) * pt * (1 - pt) / nh
                n += nh
            return float(tot), float(var), n
        if design in ("E5", "E6", "E7") and self.s is None:
            raise ValueError("designs E5-E7 need a surrogate")
        if design == "E5":
            return _hansen_hurwitz(self.q5, y[self.U], m, rng)
        if design == "E6":
            e, v, n = _hansen_hurwitz(self.q6, (self.w * (b - self.s))[self.U], m, rng)
            return self.offset + e, v, n
        if design == "E7":
            r = self.w * (b - self.s)
            strata = [np.where(self.AC & self.AB)[0], np.where(self.AC & ~self.AB)[0], np.where(self.AB & ~self.AC)[0]]
            wh = [1 / self.AC.sum() - 1 / self.AB.sum(), 1 / self.AC.sum(), -1 / self.AB.sum()]
            alloc = _allocate([len(x) for x in strata], wh, m)
            tot = var = 0.0; n = 0
            for s_idx, nh in zip(strata, alloc):
                t_, v_, n_, _ = _srswor_total(r[s_idx], nh, rng)
                tot += t_; var += v_; n += n_
            return self.offset + float(tot), float(var), n
        if design in ("E8", "E9", "E10", "E11"):
            if self.s is None:
                raise ValueError("designs E8-E11 need a surrogate")
            e = 1.0 - b
            return self._error_centred(design, e, m, rng)
        raise ValueError(design)

    def _error_centred(self, design, e, m, rng):
        """Return (estimate of L = -sum w e, variance, distinct labels)."""
        se = np.clip(1.0 - self.s, 1e-6, 1 - 1e-6)
        if design == "E9":
            U = self.U
            tot, var, n, _ = _srswor_total((self.w * e)[U], m, rng)
            return -tot, var, n
        S = np.where(self.w != 0)[0]
        if len(S) == 0:
            return 0.0, 0.0, 0
        aw = np.abs(self.w[S]); seS = se[S]; y = (self.w * e)[S]
        if design == "E8":
            qs = aw * np.sqrt(seS)
            q = 0.8 * qs / qs.sum() + 0.2 * aw / aw.sum()
            est, var, n = _hansen_hurwitz(q, y, m, rng)
            return -est, var, n
        if design == "E10":
            a = 0.8 * (aw * np.sqrt(seS * (1 - seS))) / (aw * np.sqrt(seS * (1 - seS))).sum() + 0.2 * aw / aw.sum()
            pi = _poisson_inclusion(a, min(m, len(S)))
            take = rng.random(len(S)) < pi
            r = self.w[S] * (e[S] - seS)
            est = float((self.w[S] * seS).sum() + (r[take] / pi[take]).sum())
            var = float(((1 - pi[take]) * r[take] ** 2 / pi[take] ** 2).sum())
            return -est, var, int(take.sum())
        if design == "E11":
            N = len(S); M = int(min(m, N))
            qs = aw * np.sqrt(seS)
            a = 0.8 * qs / qs.sum() + 0.2 / N
            # successive sampling without replacement proportional to a (Gumbel-top-k)
            g = np.log(a) - np.log(-np.log(rng.random(N)))
            order = np.argsort(-g)[:M]
            prev = np.concatenate([[0.0], np.cumsum(a[order])[:-1]])
            Q = a[order] / (a.sum() - prev)
            mm = np.arange(1, M + 1)
            V = 1 + (N - M) / (N - mm) * (1 / ((N - mm + 1) * Q) - 1) if M < N else np.ones(M)
            ell = N * y[order]
            terms = V * ell
            est = float(terms.mean())
            var = float(terms.var(ddof=1) / M) if M >= 2 else 0.0
            if M == N:
                var = 0.0
            return -est, var, M
        raise ValueError(design)


def _poisson_inclusion(a, m):
    """Inclusion probabilities pi_i = min(1, kappa a_i) with sum pi = m."""
    a = np.asarray(a, float); pi = np.zeros_like(a)
    free = a > 0; target = float(m)
    for _ in range(100):
        if target <= 0 or not free.any():
            break
        kappa = target / a[free].sum()
        cand = kappa * a[free]
        over = cand >= 1
        if not over.any():
            pi[free] = cand
            break
        idx = np.where(free)[0][over]
        pi[idx] = 1.0; free[idx] = False; target -= len(idx)
    return np.clip(pi, 0, 1)


def level_gB(cB, b, AB, m, rng):
    """E0: g_B estimate from an SRS of m labels in A_B; returns est, var, n."""
    iB = np.where(np.asarray(AB, bool))[0]
    cB = np.asarray(cB, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    tot, var, n, idx = _srswor_total(b[iB], m, rng)
    NB = len(iB)
    return float(cB[iB].mean() - tot / NB), float(var / NB ** 2), n


def guardrail_decision(gB_hat, gB_var, delta_hat, delta_var, z=1.96):
    """Switch from B to C only if the 95% upper bound of |g_B + Delta| lies
    below the 95% lower bound of |g_B| (independent samples; delta method
    with the two variances added).  Returns (switch: bool, details)."""
    gC_hat = gB_hat + delta_hat
    se_B = np.sqrt(gB_var); se_C = np.sqrt(gB_var + delta_var)
    lo_B = max(0.0, abs(gB_hat) - z * se_B)
    hi_C = abs(gC_hat) + z * se_C
    return bool(hi_C < lo_B), {"gC_hat": float(gC_hat), "lo_absB": float(lo_B), "hi_absC": float(hi_C)}

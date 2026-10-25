"""Decision procedures for "is |g_C| < |g_B|?" (plan A10 iii).

Exact identity: g_C^2 - g_B^2 = 2 D H with D = g_C - g_B and H = (g_C + g_B)/2,
so |g_C| < |g_B|  iff  D H < 0.  With e = 1 - b (errors), sum_i w_i = 0 and
sum_i v_i = 1:
    D = d_conf + sum_i w_i e_i,          w = 1[A_C]/|A_C| - 1[A_B]/|A_B|
    H = h_conf - 1 + sum_i v_i e_i,      v = (1[A_C]/|A_C| + 1[A_B]/|A_B|)/2,
    h_conf = (mean_{A_C} c_C + mean_{A_B} c_B)/2.
Strata of U = A_B u A_C: O = A_B n A_C, Cx = A_C \\ A_B, Bx = A_B \\ A_C; w and v are
constant within each stratum.

Every procedure spends at most m distinct labels and returns, for one simulated
audit, (switch_fpc, switch_nofpc, distinct_labels):
  switch_fpc   : decision with finite-population (fixed-pool) variances
  switch_nofpc : the same draws decided with future-population variances (no FPC)
Procedures (fixed-horizon Wald / delta method; no finite-sample guarantee):
  G0  historical A8 guardrail: level from m//2 SRS labels of A_B (Wald variance, zero
      when no error is observed), D from E4 with the remaining labels
  G1  G0 with an Agresti-Coull level variance, p~ = (x + 2)/(n + 4)
  G2  plug-in H test: switch iff sign(D_hat) H_hat < 0 with the one-sided 97.5% bound of
      H_hat = g_B_hat + D_hat/2 excluding 0, where D_hat is ESTIMATED (switcher strata) and its
      uncertainty is IGNORED.  Approximate and uncalibrated; NOT a known-D oracle.
  G3  joint (D,H): one stratified SRSWOR sample, allocation prop. to
      N_h s_h sqrt(w_h^2 + v_h^2); intersection-union test, each margin one-sided 97.5%
  G3d same sample as G3; delta-method test of the product D H using the design
      covariance of (D_hat, H_hat)
  G4  power-targeted allocation prop. to N_h s_h |v_h| (all effort on the decisive
      contrast H; a motivated adaptation of Sawade et al. 2012), IUT as G3
Stratum variances use Agresti-Coull proportions (never zero) except in G0.
"""
import numpy as np

from paired_audit.estimators import LabelTerm, _allocate, _srswor_total

Z = 1.96


def strata(AC, AB):
    O, Cx, Bx = AC & AB, AC & ~AB, AB & ~AC
    nC, nB = AC.sum(), AB.sum()
    w = (1 / nC - 1 / nB, 1 / nC, -1 / nB)
    v = ((1 / nC + 1 / nB) / 2, 1 / (2 * nC), 1 / (2 * nB))
    return [np.where(x)[0] for x in (O, Cx, Bx)], np.array(w), np.array(v)


def truth(cC, cB, b, AC, AB):
    gC = float((cC - b)[AC].mean()); gB = float((cB - b)[AB].mean())
    return dict(gC=gC, gB=gB, D=gC - gB, H=(gC + gB) / 2, better=abs(gC) < abs(gB),
                margin=abs(gB) - abs(gC))


class Decider:
    def __init__(self, cC, cB, b, AC, AB):
        self.cC, self.cB, self.e = cC, cB, 1.0 - np.asarray(b, float)
        self.b = np.asarray(b, float)
        self.AC, self.AB = AC, AB
        self.idx, self.w, self.v = strata(AC, AB)
        self.N = np.array([len(x) for x in self.idx])
        self.dconf = float(cC[AC].mean() - cB[AB].mean())
        self.hconf = float((cC[AC].mean() + cB[AB].mean()) / 2)
        se = np.clip(1 - cB, 1e-6, 1 - 1e-6)
        self.sh = np.array([np.sqrt(np.mean(se[x]) * (1 - np.mean(se[x]))) if len(x) else 0.0 for x in self.idx])
        self.iB = np.where(AB)[0]; self.cBmean = float(cB[AB].mean())
        wfull = np.zeros(len(b));
        for x, wh in zip(self.idx, self.w):
            wfull[x] = wh
        self.lt = LabelTerm(wfull, AC, AB, surrogate=cB)
        self.U = int(AC.sum() + AB.sum() - (AC & AB).sum())

    # --- level estimate of g_B from n SRS labels of A_B
    def _level(self, n, rng, ac):
        NB = len(self.iB); n = int(min(n, NB))
        idx = self.iB[rng.choice(NB, size=n, replace=False)] if n > 0 else np.zeros(0, int)
        x = float(self.e[idx].sum())
        gB = self.cBmean - 1 + x / max(n, 1)
        if ac:
            p = (x + 2) / (n + 4); s2 = p * (1 - p)
        else:
            s2 = self.e[idx].var(ddof=1) if n >= 2 else 0.0
        return gB, s2 / max(n, 1), (1 - n / NB), set(idx.tolist())

    def _D_E4(self, n, rng):
        # E4 with raw correctness (as in A8); returns D_hat, var (FPC), var (no FPC), labels
        sizes = self.N; alloc = _allocate(sizes, self.w, n)
        tot = vf = vn = 0.0; lab = set()
        for x, wh, nh in zip(self.idx, self.w, alloc):
            nh = int(min(nh, len(x)))
            if nh == 0:
                continue
            sel = x[rng.choice(len(x), size=nh, replace=False)]
            yb = self.b[sel]
            Nh = len(x)
            tot += wh * Nh * yb.mean()
            s2 = yb.var(ddof=1) if nh >= 2 else 0.0
            vf += (wh * Nh) ** 2 * (1 - nh / Nh) * s2 / nh
            vn += (wh * Nh) ** 2 * s2 / nh
            lab |= set(sel.tolist())
        D = self.dconf - tot
        return D, vf, vn, lab

    def _strat_DH(self, n, rng, alloc_w):
        alloc = _allocate(self.N, alloc_w, n)
        Dt, Ht = self.dconf, self.hconf - 1
        vD = [0.0, 0.0]; vH = [0.0, 0.0]; cDH = [0.0, 0.0]; lab = 0
        for x, wh, vh, nh in zip(self.idx, self.w, self.v, alloc):
            Nh = len(x); nh = int(min(nh, Nh))
            if Nh == 0:
                continue
            if nh == 0:
                continue
            sel = x[rng.choice(Nh, size=nh, replace=False)]
            xe = float(self.e[sel].sum()); ph = xe / nh
            Dt += wh * Nh * ph; Ht += vh * Nh * ph
            pt = (xe + 2) / (nh + 4); s2 = pt * (1 - pt)
            for j, f in enumerate(((1 - nh / Nh), 1.0)):
                vp = Nh ** 2 * f * s2 / nh
                vD[j] += wh * wh * vp; vH[j] += vh * vh * vp; cDH[j] += wh * vh * vp
            lab += nh
        # strata with zero allocation are unestimated: treat as unknown -> no decision
        unest = any(len(x) > 0 and nh == 0 for x, nh in zip(self.idx, alloc))
        return Dt, Ht, vD, vH, cDH, lab, unest

    @staticmethod
    def _iut(D, H, sD, sH):
        return bool((D - Z * sD > 0 and H + Z * sH < 0) or (D + Z * sD < 0 and H - Z * sH > 0))

    def decide(self, proc, m, rng):
        if proc in ("G0", "G1", "G2"):
            n1 = m // 2
            gB, s2n, fpc, lab1 = self._level(n1, rng, ac=(proc != "G0"))
            D, vDf, vDn, lab2 = self._D_E4(m - n1, rng)
            labels = len(lab1 | lab2)
            out = []
            for vB, vD in ((s2n * fpc, vDf), (s2n, vDn)):
                if proc in ("G0", "G1"):
                    gC = gB + D
                    lo = max(0.0, abs(gB) - Z * np.sqrt(vB)); hi = abs(gC) + Z * np.sqrt(vB + vD)
                    out.append(bool(hi < lo))
                else:
                    H = gB + D / 2
                    out.append(bool((D > 0 and H + Z * np.sqrt(vB) < 0) or (D < 0 and H - Z * np.sqrt(vB) > 0)))
            return out[0], out[1], labels
        if proc in ("G3", "G3d", "G4"):
            # per-item allocation weights; _allocate multiplies by N_h itself
            aw = self.sh * (np.sqrt(self.w ** 2 + self.v ** 2) if proc != "G4" else np.abs(self.v))
            aw = np.where(self.N > 0, np.maximum(aw, 1e-12), 0)
            D, H, vD, vH, cDH, lab, unest = self._strat_DH(m, rng, aw)
            out = []
            for j in (0, 1):
                if unest:
                    out.append(False); continue
                if proc == "G3d":
                    P = D * H; vP = H * H * vD[j] + D * D * vH[j] + 2 * D * H * cDH[j]
                    out.append(bool(P + Z * np.sqrt(max(vP, 0.0)) < 0))
                else:
                    out.append(self._iut(D, H, np.sqrt(vD[j]), np.sqrt(vH[j])))
            return out[0], out[1], lab
        raise ValueError(proc)

#!/usr/bin/env python3
"""Regression tests for plan A10: exact within-pool matching, error-centred
estimators E8-E11, and the decision procedures G0-G4.
Run: python -m unittest paired_audit.test_closure   (from scripts/canonical)
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paired_audit.decisions import Decider, truth  # noqa: E402
from paired_audit.estimators import LabelTerm  # noqa: E402
from paired_audit.identities import audit_weights  # noqa: E402
from paired_audit.pools import accepted_sets  # noqa: E402


def world(rng, N=6000, shift=0.04, acc=0.92):
    b = (rng.random(N) < acc).astype(float)
    cB = np.clip(0.55 + 0.45 * rng.beta(4, 1.5, N) * (0.75 + 0.25 * b), 0.5, 0.9995)
    cC = np.clip(cB + shift * rng.random(N) - 0.01 + 0.02 * rng.standard_normal(N), 0.5, 0.9995)
    pool = rng.random(N) < 0.5
    return cC, cB, b, pool


class TestPools(unittest.TestCase):
    def test_old_construction_breaks_exact_counts_new_one_holds(self):
        rng = np.random.default_rng(10)
        broke = 0
        for trial in range(10):
            cC, cB, b, pool = world(rng)
            for t in (0.8, 0.9):
                AC, AB = accepted_sets(cB, cC, pool, t, "stream_matched_then_restricted")
                broke += AC.sum() != AB.sum()
                AC2, AB2 = accepted_sets(cB, cC, pool, t, "exact_matched_within_pool")
                self.assertEqual(AC2.sum(), AB2.sum())
                self.assertEqual(AB2.sum(), ((cB >= t) & pool).sum())
                w = audit_weights(AC2, AB2)
                self.assertTrue(np.all(w[AC2 & AB2] == 0.0))          # exact cancellation on the overlap
                self.assertTrue(np.all(~AC2 | pool) and np.all(~AB2 | pool))
        self.assertGreater(broke, 10)   # the A8 construction rarely gives equal counts inside a pool

    def test_ties_are_deterministic(self):
        rng = np.random.default_rng(11)
        cC, cB, b, pool = world(rng)
        cC = np.round(cC, 2); cB = np.round(cB, 2)                    # heavy ties
        r1 = accepted_sets(cB, cC, pool, 0.9, "exact_matched_within_pool")
        r2 = accepted_sets(cB.copy(), cC.copy(), pool.copy(), 0.9, "exact_matched_within_pool")
        self.assertTrue(np.array_equal(r1[0], r2[0]) and np.array_equal(r1[1], r2[1]))
        self.assertEqual(r1[0].sum(), r1[1].sum())


class TestErrorCentred(unittest.TestCase):
    def test_E8_E11_unbiased(self):
        rng = np.random.default_rng(12)
        cC, cB, b, pool = world(rng)
        for branch in ("fixed", "exact_matched_within_pool"):
            AC, AB = accepted_sets(cB, cC, pool, 0.9, branch)
            w = audit_weights(AC, AB); L = float(np.dot(w, b))
            lt = LabelTerm(w, AC, AB, surrogate=cB)
            R = 3000
            for d in ("E8", "E9", "E10", "E11"):
                out = np.array([lt.estimate(d, b, 150, rng) for _ in range(R)])
                est, var, lab = out[:, 0], out[:, 1], out[:, 2]
                sd = est.std()
                self.assertLess(abs(est.mean() - L), 4 * sd / np.sqrt(R) + 1e-12, (branch, d))
                if sd > 1e-6 and d != "E11":
                    self.assertLess(abs(np.sqrt(var.mean()) - sd) / sd, 0.25, (branch, d))
                if d == "E10":
                    self.assertLess(abs(lab.mean() - min(150, (w != 0).sum())), 3.0)
                else:
                    self.assertTrue(np.all(lab <= 150))


class TestDecisions(unittest.TestCase):
    def test_identities_D_H(self):
        rng = np.random.default_rng(13)
        cC, cB, b, pool = world(rng)
        for branch in ("fixed", "exact_matched_within_pool"):
            AC, AB = accepted_sets(cB, cC, pool, 0.9, branch)
            dc = Decider(cC, cB, b, AC, AB); tr = truth(cC, cB, b, AC, AB)
            e = 1 - b
            D = dc.dconf + sum(wh * e[x].sum() for x, wh in zip(dc.idx, dc.w))
            H = dc.hconf - 1 + sum(vh * e[x].sum() for x, vh in zip(dc.idx, dc.v))
            self.assertAlmostEqual(D, tr["D"], places=12); self.assertAlmostEqual(H, tr["H"], places=12)
            self.assertEqual(tr["better"], tr["D"] * tr["H"] < 0)

    def test_zero_observed_errors_floor(self):
        rng = np.random.default_rng(14)
        N = 4000
        b = np.ones(N); b[:3] = 0                       # 3 errors in 4000: level samples usually see none
        cB = np.full(N, 0.97); cC = np.full(N, 0.975); pool = np.ones(N, bool)
        AC, AB = accepted_sets(cB, cC, pool, 0.9, "fixed")
        dc = Decider(cC, cB, b, AC, AB)
        _, s2_wald, _, _ = dc._level(50, np.random.default_rng(0), ac=False)
        _, s2_ac, _, _ = dc._level(50, np.random.default_rng(0), ac=True)
        self.assertEqual(s2_wald, 0.0); self.assertGreater(s2_ac, 0.0)

    def test_identical_sets_and_census(self):
        rng = np.random.default_rng(15)
        cC, cB, b, pool = world(rng, N=1500)
        AB = (cB >= 0.9) & pool
        dc = Decider(cB + 0.004, cB, b, AB, AB)        # identical sets, constant shift
        self.assertEqual(len(dc.idx[1]) + len(dc.idx[2]), 0)
        self.assertAlmostEqual(dc.dconf, 0.004, places=12)
        tr = truth(cB + 0.004, cB, b, AB, AB)
        U = int(AB.sum())
        for proc in ("G3", "G4", "G3d"):
            sw = [dc.decide(proc, U + 10, rng)[0] for _ in range(5)]  # census
            if abs(tr["H"]) > 1e-9:
                self.assertTrue(all(s == tr["better"] for s in sw), proc)

    def test_small_strata_and_near_tie(self):
        rng = np.random.default_rng(16)
        cC, cB, b, pool = world(rng, N=3000, shift=0.0)
        AC, AB = accepted_sets(cB, cC, pool, 0.9, "fixed")
        dc = Decider(cC, cB, b, AC, AB)
        for proc in ("G0", "G1", "G2", "G3", "G3d", "G4"):
            res = [dc.decide(proc, 60, rng) for _ in range(50)]
            self.assertTrue(all(r[2] <= 60 for r in res), proc)
        # near tie: shift so that H is ~0 -> switching should be rare for the IUT
        AB2 = (cB >= 0.9) & pool
        g_b = float((cB - b)[AB2].mean())
        cC2 = cB - 2 * g_b + 1e-5                        # D ~ -2 g_B -> H ~ 0
        dc2 = Decider(cC2, cB, b, AB2, AB2)
        sw = np.mean([dc2.decide("G3", 200, rng)[0] for _ in range(400)])
        self.assertLess(sw, 0.08)


if __name__ == "__main__":
    unittest.main()

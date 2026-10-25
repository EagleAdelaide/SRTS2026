#!/usr/bin/env python3
"""Unit tests for the exact identities and the estimators (plan A8).
Run: python -m unittest paired_audit.test_identities  (from scripts/canonical)
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paired_audit.identities import (accepted_fixed, accepted_matched, audit_weights,  # noqa: E402
                                     d_conf, decompose, gap, identity2_check)
from paired_audit.estimators import LabelTerm, guardrail_decision, level_gB  # noqa: E402


def synthetic(rng, N=5000, shift=0.05):
    b = (rng.random(N) < 0.8).astype(float)
    cB = np.clip(rng.beta(5, 2, N) * (0.6 + 0.4 * b), 0.01, 0.999)
    cC = np.clip(cB + shift * rng.random(N) - 0.01, 0.01, 0.999)
    return cC, cB, b


class TestIdentities(unittest.TestCase):
    def test_identity1_fixed_and_matched(self):
        rng = np.random.default_rng(0)
        for trial in range(20):
            cC, cB, b = synthetic(rng, shift=rng.uniform(-0.1, 0.1))
            for t in (0.5, 0.8, 0.9, 0.95):
                AC, AB = accepted_fixed(cC, t), accepted_fixed(cB, t)
                if AC.any() and AB.any():
                    d = decompose(cC, cB, b, AC, AB)
                    self.assertLess(abs(d["residual"]), 1e-12)
            k = int(accepted_fixed(cB, 0.9).sum())
            d = decompose(cC, cB, b, accepted_matched(cC, k), accepted_matched(cB, k))
            self.assertLess(abs(d["residual"]), 1e-12)

    def test_identity2_fixed_and_matched_and_edge_cases(self):
        rng = np.random.default_rng(1)
        cC, cB, b = synthetic(rng)
        for t in (0.5, 0.9):
            lhs, rhs = identity2_check(cC, cB, b, accepted_fixed(cC, t), accepted_fixed(cB, t))
            self.assertAlmostEqual(lhs, rhs, places=12)
        k = 300
        AC, AB = accepted_matched(cC, k), accepted_matched(cB, k)
        w = audit_weights(AC, AB)
        self.assertTrue(np.all(np.abs(w[AC & AB]) < 1e-15))           # overlap weight is zero at matched coverage
        self.assertAlmostEqual(w[AC & ~AB].sum(), (AC & ~AB).sum() / k, places=12)
        lhs, rhs = identity2_check(cC, cB, b, AC, AB)
        self.assertAlmostEqual(lhs, rhs, places=12)
        # identical sets -> label-free
        lhs, rhs = identity2_check(cC, cB, b, AB, AB)
        self.assertAlmostEqual(lhs, d_conf(cC, cB, AB, AB), places=12)
        # all correct / all wrong
        for const in (0.0, 1.0):
            bb = np.full_like(b, const)
            lhs, rhs = identity2_check(cC, cB, bb, AC, AB)
            self.assertAlmostEqual(lhs, rhs, places=12)
        with self.assertRaises(ValueError):
            audit_weights(np.zeros(10, bool), np.ones(10, bool))

    def test_estimators_unbiased_and_variance(self):
        rng = np.random.default_rng(2)
        cC, cB, b = synthetic(rng, N=4000, shift=0.06)
        AC, AB = accepted_fixed(cC, 0.9), accepted_fixed(cB, 0.9)
        w = audit_weights(AC, AB)
        L = float(np.dot(w, b))
        lt = LabelTerm(w, AC, AB)
        R = 3000
        for design in ("E1", "E2", "E3", "E4"):
            ests, vars_ = [], []
            for _ in range(R):
                e, v, n = lt.estimate(design, b, 200, rng)
                ests.append(e); vars_.append(v)
            ests = np.array(ests); vars_ = np.array(vars_)
            mc_sd = ests.std()
            # unbiasedness: |mean - L| within 4 MC standard errors
            self.assertLess(abs(ests.mean() - L), 4 * mc_sd / np.sqrt(R) + 1e-12, design)
            # variance estimate tracks the Monte Carlo variance (within 25%) where it is not degenerate
            if mc_sd > 1e-6:
                self.assertLess(abs(np.sqrt(vars_.mean()) - mc_sd) / mc_sd, 0.25, design)
        # E4 should not be worse than E3 here (same unique labels)
        sd = {}
        for design in ("E3", "E4"):
            sd[design] = np.std([lt.estimate(design, b, 200, rng)[0] for _ in range(R)])
        self.assertLessEqual(sd["E4"], sd["E3"] * 1.05)

    def test_active_designs_unbiased_and_variance(self):
        """Plan A9: E5 (IS with replacement), E6 (active difference estimator),
        E7 (stratified difference estimator) on fixed thresholds and at
        matched coverage (overlap weight 0 -> zero proposal mass there)."""
        rng = np.random.default_rng(4)
        cC, cB, b = synthetic(rng, N=4000, shift=0.06)
        k = int(accepted_fixed(cB, 0.9).sum())
        for AC, AB in ((accepted_fixed(cC, 0.9), accepted_fixed(cB, 0.9)),
                       (accepted_matched(cC, k), accepted_matched(cB, k))):
            w = audit_weights(AC, AB)
            L = float(np.dot(w, b))
            lt = LabelTerm(w, AC, AB, surrogate=cB)
            self.assertAlmostEqual(lt.q5.sum(), 1.0, places=12); self.assertAlmostEqual(lt.q6.sum(), 1.0, places=12)
            R = 3000
            for design in ("E5", "E6", "E7"):
                out = np.array([lt.estimate(design, b, 200, rng) for _ in range(R)])
                ests, vars_, uniq = out[:, 0], out[:, 1], out[:, 2]
                mc_sd = ests.std()
                self.assertLess(abs(ests.mean() - L), 4 * mc_sd / np.sqrt(R) + 1e-12, design)
                if mc_sd > 1e-6:
                    self.assertLess(abs(np.sqrt(vars_.mean()) - mc_sd) / mc_sd, 0.25, design)
                self.assertTrue(np.all(uniq <= 200), design)
        with self.assertRaises(ValueError):
            LabelTerm(w, AC, AB).estimate("E5", b, 50, rng)

    def test_level_and_guardrail(self):
        rng = np.random.default_rng(3)
        cC, cB, b = synthetic(rng, N=4000, shift=0.0)
        AB = accepted_fixed(cB, 0.9)
        gB = gap(cB, b, AB)
        ests = [level_gB(cB, b, AB, 150, rng)[0] for _ in range(2000)]
        self.assertLess(abs(np.mean(ests) - gB), 4 * np.std(ests) / np.sqrt(2000) + 1e-12)
        sw, _ = guardrail_decision(0.02, 1e-6, -0.019, 1e-6)
        self.assertTrue(sw)
        sw, _ = guardrail_decision(0.02, 1e-4, -0.019, 1e-4)
        self.assertFalse(sw)


if __name__ == "__main__":
    unittest.main()

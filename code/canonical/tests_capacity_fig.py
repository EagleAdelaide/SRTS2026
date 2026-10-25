#!/usr/bin/env python3
"""Regression tests for gen_capacity_fig.py (fix F1, 2026-10-16).

1. rows mixing test splits are rejected (check_rows and seed_mean_std);
2. (cell, split, seed, budget, draw, method) keys are unique, an injected
   duplicate is rejected;
3. every plotted series is paired over the same (seed, draw) set, a dropped
   row is rejected;
4. the K=1 no-floor curve and Poly1 equal the core TvA-TS rows of the same
   cell / split / seed / budget / draw (scalar consistency).
Run: python tests_capacity_fig.py
"""
import copy
import json
import unittest
from collections import defaultdict

import gen_capacity_fig as G

CASES = [("c100_vit", "test"), ("amazon_distilbert", "id_test")]


def core_rows(cell, split):
    rows = []
    for f in sorted(G.SUITE.glob(f"*/{cell}_core_natural.jsonl")):
        rows += [json.loads(l) for l in open(f)]
    return [r for r in rows if r["cell"] == cell and r["test_split"] == split]


class T(unittest.TestCase):
    def test_mixed_split_rejected(self):
        rows, _ = G.load_family("amazon_distilbert", "familyA", "id_test")
        other, _ = G.load_family("amazon_distilbert", "familyA", "ood_val")
        with self.assertRaises(G.SplitError):
            G.check_rows(rows + other)
        with self.assertRaises(G.SplitError):
            G.seed_mean_std(rows + other, "SRTS-K1-floor0", 250)
        with self.assertRaises(G.SplitError):
            G.load_family("amazon_distilbert", "familyA", "no_such_split")

    def test_unique_keys(self):
        for cell, split in CASES:
            for fam in ("familyA", "familyB"):
                rows, _ = G.load_family(cell, fam, split)
                G.check_rows(rows)
                with self.assertRaises(ValueError):
                    G.check_rows(rows + [copy.deepcopy(rows[0])])

    def test_pairing(self):
        for cell, split in CASES:
            for fam, ser in (("familyA", G.SER_A), ("familyB", G.SER_B)):
                rows = G.check_rows(G.load_family(cell, fam, split)[0])
                methods = [m for _, m, _, _ in ser]
                G.check_pairing(rows, methods)
                drop = [r for r in rows if not (
                    r["method"] == methods[1] and r["budget"] == 250
                    and r["draw"] == 0 and r["seed"] == rows[0]["seed"])]
                with self.assertRaises(ValueError):
                    G.check_pairing(drop, methods)

    def test_scalar_curves_equal_core_tva(self):
        for cell, split in CASES:
            core = {(r["seed"], r["budget"], r["draw"]): r["ece15"]
                    for r in core_rows(cell, split) if r["method"] == "TvA-TS"}
            for fam, m in (("familyA", "SRTS-K1-floor0"),
                           ("familyB", "Poly1(=scalarBCE)")):
                rows = G.load_family(cell, fam, split)[0]
                n = 0
                for r in rows:
                    if r["method"] == m:
                        k = (r["seed"], r["budget"], r["draw"])
                        self.assertIn(k, core)
                        self.assertAlmostEqual(r["ece15"], core[k], delta=1e-9)
                        n += 1
                self.assertGreater(n, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

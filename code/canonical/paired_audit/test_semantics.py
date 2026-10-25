#!/usr/bin/env python3
"""Semantic regression tests (plan A11 iii).  They fail if a claim drifts back to a
withdrawn interpretation.  Run from scripts/canonical:  python -m unittest paired_audit.test_semantics
"""
import glob
import json
import re
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("__SRTS_ROOT__/calattn-repo")   # rewritten by the artifact's configure.py, like every other script
W = REPO / "results/www2027"
TEX = re.sub(r"\s+", " ", (REPO / "www2027/paper/main.tex").read_text())
DEC_SRC = (Path(__file__).parent / "decisions.py").read_text()
FIG_SRC = (Path(__file__).parents[1] / "gen_fig_final_audit.py").read_text()


def sentences(text):
    return re.split(r"(?<=[.;])\s", text)


class TestSemantics(unittest.TestCase):
    def test_signed_excess_is_not_assumed_nonnegative(self):
        d = pd.concat([pd.read_csv(f) for f in glob.glob(str(W / "posthoc_closure/dec_*.csv"))], ignore_index=True)
        z = d.query('budget == 2500 and split == "id_test" and rule == "fixed" and proc == "G3" and m == 5000')
        best = np.minimum(abs(z.gC_e), abs(z.gB_e))
        tot = z.sw_nofpc * abs(z.gC_e) + (1 - z.sw_nofpc) * abs(z.gB_e) - best
        orc = np.where(z.better_a, abs(z.gC_e), abs(z.gB_e)) - best
        neg = int(((tot - orc) < -1e-12).sum())
        self.assertGreater(neg, 0)                                   # the excess really is signed
        s = json.loads((W / "posthoc_final/summary_final.json").read_text())["rows"]["2500/id_test/fixed/G3/cap=5000"]
        self.assertEqual(s["negative_excess_cells"], neg)            # the summary does not clip it
        self.assertNotIn("finite-label loss", TEX)

    def test_g2_is_not_called_an_oracle(self):
        self.assertIn("plug-in", TEX)
        for snt in sentences(TEX):
            if "G2" in snt and "oracle" in snt:
                self.assertRegex(snt, r"not an oracle|not a known-D oracle")
        self.assertNotIn("D$ treated as known (G2)", TEX)
        self.assertIn("NOT a known-D oracle", DEC_SRC)

    def test_exact_matched_counts_are_equal(self):
        e = pd.concat([pd.read_csv(f) for f in glob.glob(str(W / "posthoc_closure/est_*.csv"))], ignore_index=True)
        x = e[e.rule == "exact_matched_within_pool"]
        self.assertGreater(len(x), 0)
        self.assertTrue((x.nA_C == x.nA_B).all())
        self.assertTrue((x.overlap_weight == 0).all())

    def test_caps_are_not_labelled_as_exact_costs(self):
        self.assertIn("label-budget cap", TEX)
        self.assertIn('"Label-budget cap $m$"', FIG_SRC)   # figure-polish revision: capitalized axis label, still a cap
        s = json.loads((W / "posthoc_final/summary_final.json").read_text())["rows"]
        self.assertLess(s["2500/id_test/fixed/G3/cap=5000"]["actual_distinct_labels_mean"], 5000)
        self.assertNotRegex(TEX, r"power 0\.31, 0\.58 and 0\.84 at 1\{,\}000")   # old exact-m phrasing

    def test_oracle_reference_is_not_a_floor_or_bound(self):
        for bad in ("no audit budget removes", "not an information limit", "transfer, not the label count",
                    "thousands are needed", "irreducible"):
            self.assertNotIn(bad, TEX)
        for snt in sentences(TEX):
            if "oracle" in snt and re.search(r"floor|lower bound|minimum attainable", snt):
                self.assertRegex(snt, r"not a floor|not a lower bound|is not a")


if __name__ == "__main__":
    unittest.main()

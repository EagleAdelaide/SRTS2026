#!/usr/bin/env python3
"""V-06 bin-count / equal-mass sensitivity at full-pool fits (development
splits). Recovered verbatim from the command that wrote
results/www2027/bin_sensitivity.json on 2026-10-09 (session transcript);
output path is an argument. Layer A (needs logits)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import registry as R
from run_suite import load_cell, fit_methods

OUT = sys.argv[1] if len(sys.argv) > 1 else \
    "__SRTS_ROOT__/calattn-repo/results/www2027/bin_sensitivity.json"
out = {}
for cell, seeds in (("c100_vit", range(5)), ("amazon_distilbert", range(5)), ("yahoo_distilbert", range(3))):
    recs = []
    for s in seeds:
        vl, vt, tests, _ = load_cell(cell, s, False)
        split = "test" if "test" in tests else "id_test"
        tl, tt = tests[split]
        fitted = fit_methods("core", vl, vt, s)
        for name in ("TvA-TS", "HTS-BCE", "SMART+BCE", "SRTS-BCE"):
            P = fitted[name][0](tl)
            rec = {"seed": s, "method": name}
            for bins in (10, 15, 20, 30, 50):
                rec[f"ece{bins}"] = R.ece(P, tt, bins)
                rec[f"ada{bins}"] = R.adaece(P, tt, bins)
            recs.append(rec)
    out[cell] = recs
json.dump(out, open(OUT, "w"), indent=0)
print("DONE", OUT)

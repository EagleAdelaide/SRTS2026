#!/usr/bin/env python3
"""Gate for plan A4(iii): the refit's fixed-threshold metrics must reproduce
the saved rows (exact for TvA-TS, HTS-BCE, SRTS-BCE, SMART+BCE; <= 1e-3 ECE
points for PWLinear-3). Prints the maxima; exit 1 on violation."""
import glob
import json
import sys
from pathlib import Path

W = Path("__SRTS_ROOT__/calattn-repo/results/www2027")
TOL = {"PWLinear-3": 1e-3}
ref = {}
for t, pat in (("amazon_distilbert", "suite/amazon_s*/amazon_distilbert_core_natural.jsonl"),
               ("yahoo_distilbert", "suite/yahoo_s*/yahoo_distilbert_core_natural.jsonl")):
    for f in glob.glob(str(W / pat)):
        for r in map(json.loads, open(f)):
            ref[(t, r["test_split"], r["seed"], r["budget"], r["draw"], r["method"])] = r
    for r in map(json.loads, open(W / f"CONFIRMATION/{t}.jsonl")):
        ref[(t, "ood_test_CONFIRMATION", r["seed"], r["budget"], r["draw"], r["method"])] = r
worst, bad, n = {}, [], 0
for f in sorted(glob.glob(str(W / "posthoc_matched/*.jsonl"))):
    for r in map(json.loads, open(f)):
        k = (r["task"], r["split"], r["seed"], r["budget"], r["draw"], r["method"])
        o = ref.get(k)
        if o is None:
            bad.append(("missing", k)); continue
        n += 1
        dd = max(abs(o["ece15"] - r["ece15"]), 100 * abs(o["count_rel_err"] - r["count_rel_err"]),
                 *[100 * abs(o[f"errgap_a{t}"] - r[f"errgap_a{t}"]) for t in (80, 90, 95)
                   if o[f"errgap_a{t}"] == o[f"errgap_a{t}"]])
        worst[r["method"]] = max(worst.get(r["method"], 0), dd)
        if dd > TOL.get(r["method"], 1e-9):
            bad.append((k, dd))
print(f"matched {n} rows; violations {len(bad)}; max diff (points) {worst}")
for b in bad[:8]:
    print("  ", b)
sys.exit(1 if bad or n == 0 else 0)

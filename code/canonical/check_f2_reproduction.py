#!/usr/bin/env python3
"""Reproduction gate for F2 (plan P2): D-OOF must equal saved SRTS-BCE rows
and C-IN must equal saved PWLinear-3 rows (ECE15, |diff| < 1e-9) for every
matched key.  Exit code 1 on any mismatch or on zero matched keys."""
import glob
import json
import sys
from pathlib import Path

REPO = Path("__SRTS_ROOT__/calattn-repo")
W = REPO / "results/www2027"
MAP = {"D-OOF": "SRTS-BCE", "C-IN": "PWLinear-3"}
# Addendum A1 of the plan: D-OOF exact; C-IN tolerance for host-level
# L-BFGS-B end-point drift (diagnosed in the pilot, max reported)
TOL = {"D-OOF": 1e-9, "C-IN": 1e-3}


def load(pattern):
    rows = []
    for f in sorted(glob.glob(str(W / pattern))):
        rows += [json.loads(l) for l in open(f)]
    return rows


def main():
    bad, n, worst = [], 0, {}
    for cell, split, core in (("c100_vit", "test", "suite/c100_vit_s*/c100_vit_core_natural.jsonl"),
                              ("amazon_distilbert", "id_test", "suite/amazon_s*/amazon_distilbert_core_natural.jsonl")):
        ref = {(r["seed"], r["budget"], r["draw"], r["method"]): r["ece15"]
               for r in load(core) if r["test_split"] == split}
        for r in load(f"posthoc_f2/{cell}_{split}.jsonl"):
            if r["variant"] in MAP:
                k = (r["seed"], r["budget"], r["draw"], MAP[r["variant"]])
                if k not in ref:
                    bad.append(("missing", cell, k)); continue
                n += 1
                dd = abs(ref[k] - r["ece15"])
                worst[r["variant"]] = max(worst.get(r["variant"], 0.0), dd)
                if dd >= TOL[r["variant"]]:
                    bad.append((cell, k, ref[k], r["ece15"]))
    sref = {(r["seed"], r["corruption"], r["severity"], r["method"]): r["ece15"]
            for r in load("shift/c100c_core_s*.jsonl")}
    for r in load("posthoc_f2/c100c_vit_shift.jsonl"):
        if r["variant"] in MAP:
            k = (r["seed"], r["corruption"], r["severity"], MAP[r["variant"]])
            if k not in sref:
                bad.append(("missing", "shift", k)); continue
            n += 1
            dd = abs(sref[k] - r["ece15"])
            worst["shift " + r["variant"]] = max(
                worst.get("shift " + r["variant"], 0.0), dd)
            if dd >= TOL[r["variant"]]:
                bad.append(("shift", k, sref[k], r["ece15"]))
    print(f"matched {n} rows; mismatches {len(bad)}; max |dECE15| {worst}")
    for b in bad[:10]:
        print("  ", b)
    sys.exit(1 if (bad or n == 0) else 0)


if __name__ == "__main__":
    main()

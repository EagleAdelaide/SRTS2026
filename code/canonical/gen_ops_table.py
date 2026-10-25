#!/usr/bin/env python3
"""Operational-endpoints table (LaTeX) from canonical rows.

Columns per method at chosen budgets: |count_rel_err| (batch error-count
relative error, whole split as one batch, denominator = realized errors),
coverage@.90, accerr@.90, errgap@.90, risk@90%cov. Values = mean over
seeds of per-seed draw-means +- across-seed std. Usage:
  gen_ops_table.py <cell> <split> <rows_glob> [budgets]
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path("__SRTS_ROOT__/calattn-repo")
ORDER = ["NoTS", "TvA-TS", "HTS-BCE", "SMART+BCE", "PWLinear-3",
         "SRTS-BCE", "SRTS-Shrink"]


def agg(rows, method, budget, key, absval=False):
    per = defaultdict(list)
    for r in rows:
        if r["method"] == method and r["budget"] == budget and key in r:
            v = r[key]
            per[r["seed"]].append(abs(v) if absval else v)
    if not per:
        return None
    sm = np.array([np.mean(v) for v in per.values()])
    return float(sm.mean()), float(sm.std())


def main():
    cell, split, glob_ = sys.argv[1], sys.argv[2], sys.argv[3]
    budgets = [int(b) for b in (sys.argv[4].split(",") if len(sys.argv) > 4
                                else ["250", "2500"])]
    rows = []
    for f in sorted(REPO.glob(glob_)):
        rows += [json.loads(l) for l in open(f)]
    rows = [r for r in rows if r.get("test_split") == split]
    lines = []
    for b in budgets:
        lines.append(rf"\multicolumn{{6}}{{l}}{{\emph{{$n={b}$}}}}\\")
        for m in ORDER:
            cols = []
            for key, absv, fmt in [("count_rel_err", True, "{:.3f}"),
                                   ("cov_a90", False, "{:.3f}"),
                                   ("accerr_a90", False, "{:.3f}"),
                                   ("errgap_a90", False, "{:+.3f}"),
                                   ("risk_cov90", False, "{:.3f}")]:
                a = agg(rows, m, b, key, absv)
                cols.append("---" if a is None else
                            (fmt.format(a[0]) + rf"$\pm$" + f"{a[1]:.3f}"))
            if all(c == "---" for c in cols):
                continue
            lines.append(f"{m} & " + " & ".join(cols) + r" \\")
    out = REPO / "www2027/paper/generated" / f"ops_{cell}_{split}.tex"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

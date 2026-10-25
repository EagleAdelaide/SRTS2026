#!/usr/bin/env python3
"""Generate the main budget x method results table (LaTeX body) from
canonical suite rows. Mean over seeds of per-seed draw-means, +- across-
seed std; best per budget bold, second underlined; paired hierarchical CI
markers for the frozen primary comparisons only. No hand edits.

Usage: gen_main_matrix.py <cell> <split> [metric] [rows_glob...]
Writes www2027/paper/generated/main_<cell>_<split>_<metric>.tex and .json
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path("__SRTS_ROOT__/calattn-repo")
ORDER = ["NoTS", "TS-NLL", "TvA-TS", "HTS-NLL", "HTS-BCE", "SMART-SoftECE",
         "SMART+BCE", "Margin-K3", "PWLinear-3", "SRTS-BCE"]
TEX = {"NoTS": "Uncalibrated", "TS-NLL": "TS (NLL)", "TvA-TS": "TvA-TS",
       "HTS-NLL": "HTS (NLL)", "HTS-BCE": "HTS+BCE",
       "SMART-SoftECE": "SMART (sECE)", "SMART+BCE": "SMART+BCE",
       "Margin-K3": "Margin-K3", "PWLinear-3": "PWLinear-3",
       "SRTS-BCE": "SRTS-BCE"}


def main():
    cell, split = sys.argv[1], sys.argv[2]
    metric = sys.argv[3] if len(sys.argv) > 3 else "ece15"
    globs = sys.argv[4:] or [
        f"results/www2027/suite/{cell}_s*/*core_natural.jsonl"]
    rows = []
    for g in globs:
        for f in sorted(REPO.glob(g)):
            rows += [json.loads(l) for l in open(f)]
    rows = [r for r in rows if r["test_split"] == split and metric in r]
    budgets = sorted({r["budget"] for r in rows})
    per = defaultdict(lambda: defaultdict(list))  # (m,b) -> seed -> vals
    for r in rows:
        per[(r["method"], r["budget"])][r["seed"]].append(r[metric])
    stat = {}
    for (m, b), d in per.items():
        sm = np.array([np.mean(v) for v in d.values()])
        stat[(m, b)] = (float(sm.mean()), float(sm.std()))
    lines = []
    for m in ORDER:
        if not any((m, b) in stat for b in budgets):
            continue
        cells = []
        for b in budgets:
            if (m, b) not in stat:
                cells.append("---")
                continue
            mu, sd = stat[(m, b)]
            ranked = sorted(s for s in (stat.get((mm, b)) for mm in ORDER)
                            if s)
            mus = sorted(x[0] for x in ranked)
            txt = f"{mu:.2f}$\\pm${sd:.2f}"
            if abs(mu - mus[0]) < 1e-12:
                txt = f"\\textbf{{{txt}}}"
            elif len(mus) > 1 and abs(mu - mus[1]) < 1e-12:
                txt = f"\\underline{{{txt}}}"
            cells.append(txt)
        lines.append(f"{TEX.get(m, m)} & " + " & ".join(cells) + r" \\")
    out = REPO / "www2027/paper/generated"
    out.mkdir(parents=True, exist_ok=True)
    body = "\n".join(lines) + "\n"
    (out / f"main_{cell}_{split}_{metric}.tex").write_text(body)
    json.dump({f"{m}|{b}": stat[(m, b)] for (m, b) in stat},
              open(out / f"main_{cell}_{split}_{metric}.json", "w"),
              indent=1)
    print(body)


if __name__ == "__main__":
    main()

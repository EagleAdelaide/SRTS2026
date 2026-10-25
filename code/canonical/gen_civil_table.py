#!/usr/bin/env python3
"""Appendix table: CivilComments external check (frozen protocol; written and tested on
synthetic rows before any real outcome).  Reads results/www2027/civil/analysis.json.
Rows: four conditional maps vs TvA-TS at n = 2500 (and n = 250 for the primary endpoint);
columns: paired differences (map - scalar, pp) of ECE15, |count error|, |g(A_0.90)| with
its Bonferroni (1 - 0.05/4) seed->draw interval [primary family], |g| at matched coverage,
and mean D = g_C - g_B.  (Single-column layout after outcomes, for space: the selection
shares, previously a column, are in analysis.json["pattern"] and quoted in the text.)
Writes www2027/paper/generated/civil_table.tex (table body only).
"""
import json
import os
from pathlib import Path

RES = Path(os.environ.get("CIVIL_RES", "__SRTS_ROOT__/calattn-repo/results/www2027/civil"))
OUT = Path(os.environ.get("CIVIL_TABLE_OUT", "__SRTS_ROOT__/calattn-repo/www2027/paper/generated/civil_table.tex"))
MAPS = ["HTS-BCE", "SRTS-BCE", "SMART+BCE", "PWLinear-3"]


def f(x, d=2):
    if x != 0 and round(x, d) == 0:   # keep the sign of a bound that rounds to zero
        d += 1
    s = f"{x:+.{d}f}"
    return s.replace("-", "$-$")


def main():
    a = json.load(open(RES / "analysis.json"))
    c, pat = a["contrasts"], a["pattern"]
    lines = []
    for n in (2500, 250):
        for m in MAPS:
            k = f"n={n}/{m}"
            p = c[f"{k}/abs_g90"]; lo, hi = p["ci_bonf4"]
            q = pat[f"{k}/t=90"]
            lines.append(f"{n} & {m} & {f(c[k + '/ece15']['mean'])} & {f(c[k + '/abs_count']['mean'], 1)} & "
                         f"{f(p['mean'])} [{f(lo)}, {f(hi)}] & {f(c[k + '/abs_gm90']['mean'])} & {f(q['D'])} \\\\")
        if n == 2500:
            lines.append("\\midrule")
    OUT.write_text("\n".join(lines) + "\n")
    print("wrote", OUT)
    print("\n".join(lines))


if __name__ == "__main__":
    main()

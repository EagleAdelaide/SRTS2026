#!/usr/bin/env python3
"""WP6 canonical analysis of suite JSONL rows.

Inference design (frozen in WEB_PROTOCOL.md §6):
  * primary: seed->draw hierarchical bootstrap of paired per-draw deltas
    (resample seeds with replacement, then draws within seed; B=20000;
    one shared RNG stream per cell so all comparisons use the same draws);
  * sensitivity: seed-cluster interval (percentile over per-seed means,
    S<=5 -> reported as spread, not a calibrated CI);
  * interaction: Delta(n_small) - Delta(n_big) with the same hierarchy
    (paired within seed x draw index);
  * wording: CI crossing 0 => "not statistically resolved" (the JSON stores
    excludes_zero; prose must follow the rule);
  * win rates and bootstrap sign fractions are DESCRIPTIVE.

Usage:
  python analyze_suite.py ROWS.jsonl [ROWS2.jsonl ...] --metric ece15 \
      --comparisons "SRTS-BCE:TvA-TS,SRTS-BCE:SMART+BCE" \
      --budgets 250,2500 --test-split test --out out.json
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

B_BOOT = 20000


def load_rows(paths):
    rows = []
    for p in paths:
        for line in open(p):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def hier_boot(per_seed_deltas, rng, B=B_BOOT):
    svals = list(per_seed_deltas.values())
    S = len(svals)
    boots = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, S, S)
        m = 0.0
        for j in idx:
            a = svals[j]
            m += a[rng.integers(0, len(a), len(a))].mean()
        boots[i] = m / S
    return boots


def analyze(rows, metric, comps, budgets, split, sampling):
    rows = [r for r in rows if r["test_split"] == split
            and r.get("sampling", "natural") == sampling
            and metric in r]
    out = {}
    cells = sorted({r["cell"] for r in rows})
    for cell in cells:
        cr = [r for r in rows if r["cell"] == cell]
        seeds = sorted({r["seed"] for r in cr})
        idxm = {(r["seed"], r["budget"], r["draw"], r["method"]): r[metric]
                for r in cr}
        cell_out = {}
        rng = np.random.default_rng(20261009)  # shared stream per cell
        for b in budgets:
            rec = {}
            for ma, mb in comps:
                per = {}
                ok = True
                for s in seeds:
                    draws = sorted({r["draw"] for r in cr
                                    if r["seed"] == s and r["budget"] == b
                                    and r["method"] == ma})
                    try:
                        a = np.array(
                            [idxm[(s, b, d, ma)] - idxm[(s, b, d, mb)]
                             for d in draws])
                        # collapse duplicated full-pool draws (legacy R>1)
                        if len(a) > 1 and np.ptp(a) == 0.0:
                            a = a[:1]
                        per[s] = a
                    except KeyError:
                        ok = False
                if not ok or not per:
                    continue
                boots = hier_boot(per, rng)
                lo, hi = np.percentile(boots, [2.5, 97.5])
                seed_means = np.array([a.mean() for a in per.values()])
                alln = np.concatenate(list(per.values()))
                rec[f"{ma} - {mb}"] = dict(
                    point=float(seed_means.mean()),
                    hier_ci=[float(lo), float(hi)],
                    excludes_zero=bool(lo > 0 or hi < 0),
                    resolved="yes" if (lo > 0 or hi < 0)
                             else "not statistically resolved",
                    per_seed={str(s): float(a.mean())
                              for s, a in per.items()},
                    seed_cluster_range=[float(seed_means.min()),
                                        float(seed_means.max())],
                    win_rate_desc=float((alln < 0).mean()),
                    sign_frac_boot_desc=float((boots < 0).mean()),
                    n_seeds=len(per),
                    n_draws=int(sum(len(a) for a in per.values())))
            cell_out[int(b)] = rec
        # interaction Delta(min budget) - Delta(max budget)
        if len(budgets) >= 2:
            b1, b2 = min(budgets), max(budgets)
            inter = {}
            for ma, mb in comps:
                per = {}
                try:
                    for s in seeds:
                        dr1 = sorted({r["draw"] for r in cr
                                      if r["seed"] == s and r["budget"] == b1
                                      and r["method"] == ma})
                        dr2 = sorted({r["draw"] for r in cr
                                      if r["seed"] == s and r["budget"] == b2
                                      and r["method"] == ma})
                        d1 = np.array([idxm[(s, b1, d, ma)]
                                       - idxm[(s, b1, d, mb)] for d in dr1])
                        d2 = np.array([idxm[(s, b2, d, ma)]
                                       - idxm[(s, b2, d, mb)] for d in dr2])
                        if len(d2) > 1 and np.ptp(d2) == 0.0:
                            d2 = d2[:1]
                        if set(dr1) == set(dr2) and len(d1) == len(d2):
                            per[s] = d1 - d2       # paired by draw index
                        else:
                            per[s] = d1 - d2.mean()  # broadcast (R=1 case)
                except KeyError:
                    continue
                boots = hier_boot(per, rng)
                lo, hi = np.percentile(boots, [2.5, 97.5])
                inter[f"{ma} - {mb}"] = dict(
                    point=float(np.mean([a.mean() for a in per.values()])),
                    hier_ci=[float(lo), float(hi)],
                    excludes_zero=bool(lo > 0 or hi < 0))
            cell_out[f"interaction_{b1}_minus_{b2}"] = inter
        out[cell] = cell_out
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rows", nargs="+")
    ap.add_argument("--metric", default="ece15")
    ap.add_argument("--comparisons",
                    default="SRTS-BCE:TvA-TS,SRTS-BCE:SMART+BCE")
    ap.add_argument("--budgets", default="250,625,1250,2500")
    ap.add_argument("--test-split", default="test")
    ap.add_argument("--sampling", default="natural")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    comps = [tuple(c.split(":")) for c in args.comparisons.split(",")]
    budgets = [int(b) for b in args.budgets.split(",")]
    rows = load_rows(args.rows)
    res = analyze(rows, args.metric, comps, budgets, args.test_split,
                  args.sampling)
    txt = json.dumps(res, indent=1)
    if args.out:
        Path(args.out).write_text(txt)
    for cell, co in res.items():
        print(f"== {cell} [{args.metric}, split={args.test_split}, "
              f"{args.sampling}]")
        for b, rec in co.items():
            for name, v in rec.items():
                e = "EXCL0" if v["excludes_zero"] else "incl0"
                print(f"  {b} {name}: {v['point']:+.4f} "
                      f"[{v['hier_ci'][0]:+.4f},{v['hier_ci'][1]:+.4f}] {e}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""CivilComments validation, CPU stage 4: summaries of the audit rows (written before any
real audit outcome was read; pure aggregation in the Amazon reporting format, no new test).
Estimation (per budget, rule, design, m): mean RMSE of D_hat over cells (pp), mean distinct
labels, median and worst-cell 95% coverage; median number of items with nonzero audit
weight (|A_C xor A_B| for the exact branch).
Decisions (per budget, rule, procedure, label-budget cap): cells better / worse /
equivalent at |margin| >= 0.1 pp within the audit pool; mean power over beneficial cells;
mean and worst-cell false switches over harmful cells; mean distinct labels; census cells
(labels >= support); cap at which each beneficial cell first reaches 80% power.
Writes results/www2027/civil/audit_summary.json.
"""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

RES = Path(__import__("os").environ.get("CIVIL_RES", "__SRTS_ROOT__/calattn-repo/results/www2027/civil"))
EPS = 0.001


def main():
    est = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(RES / "audit_est_s*.csv")))], ignore_index=True)
    dec = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(RES / "audit_dec_s*.csv")))], ignore_index=True)
    out = dict(note="within the audit pool only (item-level hash split); truth = audit pool full labels; "
                    "finite-population variances; caps are maximum label budgets", estimation={}, decisions={}, cells={})
    for (n, rule, d, m), y in est.groupby(["budget", "rule", "design", "m"]):
        out["estimation"][f"n={n}/{rule}/{d}/m={m}"] = dict(
            cells=len(y), rmse_pp=100 * float(y.rmse.mean()), unique_labels=float(y.unique_labels.mean()),
            coverage_median=float(y.coverage95.median()), coverage_min=float(y.coverage95.min()))
    for (n, rule), y in est[est.m == est.m.min()].groupby(["budget", "rule"]):
        c = y.drop_duplicates(["seed", "t", "method"])
        out["cells"][f"n={n}/{rule}"] = dict(
            cells=len(c), support_w_median=float(c.support_w.median()), nA_B_median=float(c.nA_B.median()),
            better=int((c.margin_a >= EPS).sum()), worse=int((c.margin_a <= -EPS).sum()),
            equivalent=int((c.margin_a.abs() < EPS).sum()),
            D_pp_median=100 * float(c.D_a.median()), H_pp_median=100 * float(c.H_a.median()),
            gB_pp_mean=100 * float(c.gB_a.mean()), gC_pp_mean=100 * float(c.gC_a.mean()))
    for (n, rule, p, m), y in dec.groupby(["budget", "rule", "proc", "m"]):
        good, bad = y.margin_a >= EPS, y.margin_a <= -EPS
        out["decisions"][f"n={n}/{rule}/{p}/cap={m}"] = dict(
            cells=len(y), beneficial=int(good.sum()), harmful=int(bad.sum()),
            power=float(y.loc[good, "sw_fpc"].mean()) if good.any() else None,
            false_switch_mean=float(y.loc[bad, "sw_fpc"].mean()) if bad.any() else None,
            false_switch_worst=float(y.loc[bad, "sw_fpc"].max()) if bad.any() else None,
            distinct_labels_mean=float(y.labels.mean()), census_cells=int((y.labels >= y.support_U).sum()))
    first = {}
    for (n, rule, p), y in dec[dec.margin_a >= EPS].groupby(["budget", "rule", "proc"]):
        caps = []
        for _, c in y.groupby(["seed", "t", "method"]):
            ok = c[c.sw_fpc >= 0.8].m
            caps.append(int(ok.min()) if len(ok) else None)
        first[f"n={n}/{rule}/{p}"] = dict(beneficial=len(caps), **{f"by_{k}": int(sum(c is not None and c <= k for c in caps))
                                                                   for k in (250, 1000, 2500, 5000)},
                                          never=int(sum(c is None for c in caps)))
    out["first_cap_80pct_power"] = first
    (RES / "audit_summary.json").write_text(json.dumps(out, indent=1) + "\n")
    for k in ("n=2500/fixed", "n=2500/exact_matched_within_pool"):
        print(k, out["cells"].get(k))
    for k in ("n=2500/fixed/G3/cap=1000", "n=2500/fixed/G3/cap=5000", "n=2500/fixed/G1/cap=5000"):
        print(k, out["decisions"].get(k))
    for d in ("E3", "E9", "E4", "E10", "E11"):
        print(d, {m: round(out["estimation"][f"n=2500/fixed/{d}/m={m}"]["rmse_pp"], 3) for m in (100, 1000)
                  if f"n=2500/fixed/{d}/m={m}" in out["estimation"]})


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""P3 post-hoc operational-endpoint analysis on the frozen confirmation rows
(`python analyze_endpoints.py`) and, per plan addendum A4(i), on the
development splits (`python analyze_endpoints.py development`).

Plan: www2027/docs/POSTHOC_PLAN_2026-10-16.md (P3). Post-hoc; does not alter
the six frozen primary comparisons.

Per row (fields of registry.ops_metrics; b = correct, c = confidence,
A_t = {c >= t}, g(B) = mean_B(c - b)):
  g_all      = -count_rel_err * act_err_count / n      (= mean(c - b))
  R_all      = act_err_count / n
  gap_t      = errgap_t = accerr_t - prederr_t          (= g(A_t))
  acc_cnt_t  = (prederr_t - accerr_t) / accerr_t        accepted-set count
               relative error (NaN if A_t empty or accerr_t == 0)
  grej_t     = (g_all - cov_t * gap_t) / (1 - cov_t)    rejected-set gap,
               derived from g_all = cov g(A) + (1-cov) g(A^c)
               (NaN if cov_t in {0, 1})
Saved-field consistency checks: nmis_t == accerr_t * cov_t * n;
count_rel_err == (pred_err_count - act_err_count) / act_err_count.

Inference: paired per-draw deltas (method - TvA-TS), seed->draw
hierarchical bootstrap, B = 20000, RNG 20261016, one index array per
(task, budget) shared by all endpoints; 95% and Bonferroni 1-0.05/64 for
the 64 core contrasts (|count|, |gap_.80|, |gap_.90|, |gap_.95|).
Outputs: results/www2027/posthoc_endpoints/{endpoints.json, endpoints.csv}
"""
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path("__SRTS_ROOT__/calattn-repo")
W = REPO / "results/www2027"
OUT = W / "posthoc_endpoints"
TASKS = {"amazon": W / "CONFIRMATION/amazon_distilbert.jsonl",
         "yahoo": W / "CONFIRMATION/yahoo_distilbert.jsonl"}
REF = "TvA-TS"
METHODS = ["HTS-BCE", "SRTS-BCE", "SMART+BCE", "PWLinear-3"]
SHOW = ["NoTS", REF] + METHODS
BUDGETS = [250, 2500]
TS = [80, 90, 95]
CORE = ["abs_count"] + [f"abs_gap_{t}" for t in TS]
DESC = (["ece15", "nll", "brier", "topbce", "aurc", "count_rel_err"]
        + [f"{q}_{t}" for t in TS for q in
           ("cov", "accerr", "prederr", "gap", "acc_cnt", "abs_acc_cnt",
            "grej", "abs_grej")]
        + [f"risk_cov{t}" for t in TS])
B = 20000
N_CORE = len(CORE) * len(METHODS) * len(BUDGETS) * len(TASKS)   # 64
Z_BONF = 100 * 0.05 / N_CORE / 2


def derive(r):
    d = {k: r[k] for k in ("ece15", "nll", "brier", "topbce", "aurc",
                           "count_rel_err")}
    n = r["n"]
    d["abs_count"] = abs(r["count_rel_err"])
    g_all = -r["count_rel_err"] * r["act_err_count"] / n
    d["g_all"] = g_all
    chk = []
    # consistency of saved fields
    cre = (r["pred_err_count"] - r["act_err_count"]) / r["act_err_count"]
    chk.append(abs(cre - r["count_rel_err"]))
    for t in TS:
        cov, ae, pe = r[f"cov_a{t}"], r[f"accerr_a{t}"], r[f"prederr_a{t}"]
        gap = r[f"errgap_a{t}"]
        d[f"cov_{t}"], d[f"accerr_{t}"], d[f"prederr_{t}"] = cov, ae, pe
        d[f"gap_{t}"], d[f"abs_gap_{t}"] = gap, abs(gap)
        if cov > 0:
            chk.append(abs(r[f"nmis_a{t}"] - ae * cov * n) / n)
            chk.append(abs(gap - (ae - pe)))
        d[f"acc_cnt_{t}"] = ((pe - ae) / ae if (cov > 0 and ae > 0)
                             else float("nan"))
        d[f"abs_acc_cnt_{t}"] = abs(d[f"acc_cnt_{t}"])
        d[f"grej_{t}"] = ((g_all - cov * gap) / (1 - cov)
                          if 0 < cov < 1 else float("nan"))
        d[f"abs_grej_{t}"] = abs(d[f"grej_{t}"])
        d[f"risk_cov{t}"] = r[f"risk_cov{t}"]
    return d, max(chk)


def boot_index(S, D, rng):
    return rng.integers(0, S, (B, S)), rng.integers(0, D, (B, S, D))


def hier(A, idx):
    """A: S x D paired deltas; returns B bootstrap means."""
    i_s, i_d = idx
    v = A[i_s[:, :, None], i_d]
    return np.nanmean(np.nanmean(v, axis=2), axis=1)


DEV = {"amazon_id_test": ("suite/amazon_s*/amazon_distilbert_core_natural.jsonl", "id_test"),
       "amazon_ood_val": ("suite/amazon_s*/amazon_distilbert_core_natural.jsonl", "ood_val"),
       "yahoo_id_test": ("suite/yahoo_s*/yahoo_distilbert_core_natural.jsonl", "id_test")}


def load_rows(task, mode):
    if mode == "confirmation":
        return [json.loads(l) for l in open(TASKS[task])]
    pat, split = DEV[task]
    rows = []
    for f in sorted(W.glob(pat)):
        rows += [json.loads(l) for l in open(f)]
    rows = [r for r in rows if r["test_split"] == split
            and r.get("sampling", "natural") == "natural"]
    assert rows, task
    return rows


def main():
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "confirmation"
    assert mode in ("confirmation", "development")
    tasks = list(TASKS) if mode == "confirmation" else list(DEV)
    n_core = N_CORE if mode == "confirmation" else len(CORE) * len(METHODS) * len(BUDGETS)
    z_bonf = 100 * 0.05 / n_core / 2
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"plan": "POSTHOC_PLAN_2026-10-16.md " + ("P3" if mode == "confirmation" else "A4(i)"),
           "B": B, "rng_seed": 20261016, "n_core": n_core,
           "bonferroni_level": 1 - 0.05 / n_core, "tasks": {}}
    if mode == "development":
        res["bonferroni_family"] = "per split (32 contrasts)"
    csv_rows = []
    rng = np.random.default_rng(20261016)
    for task in tasks:
        rows = load_rows(task, mode)
        tab, maxchk = {}, 0.0
        for r in rows:
            if r["method"] in SHOW and r["budget"] in BUDGETS:
                d, c = derive(r)
                maxchk = max(maxchk, c)
                tab[(r["method"], r["budget"], r["seed"], r["draw"])] = d
        seeds = sorted({k[2] for k in tab})
        draws = sorted({k[3] for k in tab})
        tr = {"saved_field_consistency_max_abs": maxchk, "seeds": seeds,
              "draws": len(draws), "levels": {}, "contrasts": {}}
        for b in BUDGETS:
            # descriptive levels: mean over seeds of per-seed draw means
            lv = {}
            for m in SHOW:
                lv[m] = {}
                for q in ["abs_count", "g_all"] + DESC + CORE:
                    M = np.array([[tab[(m, b, s, dr)][q] for dr in draws]
                                  for s in seeds])
                    lv[m][q] = float(np.nanmean(np.nanmean(M, axis=1)))
                    nn = int(np.isnan(M).sum())
                    if nn:
                        lv[m][q + "_n_undefined"] = nn
            tr["levels"][b] = lv
            idx = boot_index(len(seeds), len(draws), rng)
            ct = {}
            for m in METHODS:
                ct[m] = {}
                for q in CORE + DESC:
                    A = np.array([[tab[(m, b, s, dr)][q] - tab[(REF, b, s, dr)][q]
                                   for dr in draws] for s in seeds])
                    bt = hier(A, idx)
                    pt = float(np.nanmean(np.nanmean(A, axis=1)))
                    lo, hi = np.nanpercentile(bt, [2.5, 97.5])
                    rec = {"point": pt, "ci95": [float(lo), float(hi)],
                           "excl0_95": bool(lo > 0 or hi < 0),
                           "per_seed": [float(x) for x in np.nanmean(A, axis=1)],
                           "core": q in CORE}
                    if q in CORE:
                        lb, hb = np.nanpercentile(bt, [z_bonf, 100 - z_bonf])
                        rec["ci_bonf"] = [float(lb), float(hb)]
                        rec["excl0_bonf"] = bool(lb > 0 or hb < 0)
                    ct[m][q] = rec
                    csv_rows.append(dict(task=task, budget=b, method=m,
                                         endpoint=q, core=q in CORE,
                                         point=pt, lo95=float(lo),
                                         hi95=float(hi),
                                         lo_bonf=rec.get("ci_bonf", [None])[0],
                                         hi_bonf=rec.get("ci_bonf", [None, None])[1]))
            tr["contrasts"][b] = ct
        res["tasks"][task] = tr
    stem = "endpoints" if mode == "confirmation" else "endpoints_dev"
    (OUT / f"{stem}.json").write_text(json.dumps(res, indent=1))
    with open(OUT / f"{stem}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(csv_rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(csv_rows)
    print("wrote", OUT)


if __name__ == "__main__":
    main()

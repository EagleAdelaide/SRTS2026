#!/usr/bin/env python3
"""Plan A8 (ii): exact mechanism accounting on the per-item cache.
For map C in {HTS-BCE, SRTS-BCE, SMART+BCE, PWLinear-3} vs B = TvA-TS, per
seed, draw, split, budget, rule (fixed t in {.80,.90,.95}; matched top-k
with k = |A_B| at t): S, P, S+P, gC, gB, |gC|-|gB|, coverages, overlap.
Summaries: mean over draws within seed, then over seeds; descriptive 95%
seed->draw intervals (B = 20000, RNG 20261016); per-seed means.
Gate: S + P - (gC - gB) < 1e-9 everywhere; gC/gB equal the A4(iii) rows
(errgap_a{t}) to 1e-6.
Output: results/www2027/posthoc_mechanism/{mechanism_rows.csv, analysis_mechanism.json}
"""
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paired_audit.identities import accepted_fixed, accepted_matched, decompose  # noqa: E402

W = Path("__SRTS_ROOT__/calattn-repo/results/www2027")
OUT = W / "posthoc_mechanism"
MAPS = ["TvA-TS", "HTS-BCE", "SRTS-BCE", "SMART+BCE", "PWLinear-3"]
SPLITS = ["id_test", "ood_val", "ood_test_CONFIRMATION"]
TS = [0.80, 0.90, 0.95]
B, RNG = 20000, 20261016


def hier(A, i_s, i_d):
    return np.nanmean(np.nanmean(A[i_s[:, :, None], i_d], axis=2), axis=1)


def main():
    ref = {}
    for f in glob.glob(str(W / "posthoc_matched/amazon_distilbert_s*.jsonl")):
        for r in map(json.loads, open(f)):
            ref[(r["split"], r["seed"], r["budget"], r["draw"], r["method"])] = r
    rows = []
    gate = {"max_identity_residual": 0.0, "max_row_diff_points": 0.0, "cells": 0}
    files = sorted(glob.glob(str(OUT / "cache_s*_n*.npz")))
    for f in files:
        z = np.load(f)
        s = int(Path(f).stem.split("_s")[1].split("_n")[0]); b = int(Path(f).stem.split("_n")[1])
        for sp in SPLITS:
            conf, corr = z[f"conf_{sp}"], z[f"corr_{sp}"].astype(np.float64)
            for d in range(conf.shape[0]):
                cB = conf[d, 0].astype(np.float64)
                for t in TS:
                    AB = accepted_fixed(cB, t); k = int(AB.sum())
                    for mi, m in enumerate(MAPS[1:], start=1):
                        cC = conf[d, mi].astype(np.float64)
                        for rule, AC_, AB_ in (("fixed", accepted_fixed(cC, t), AB),
                                               ("matched", accepted_matched(cC, k), accepted_matched(cB, k))):
                            dd = decompose(cC, cB, corr, AC_, AB_)
                            gate["cells"] += 1
                            if dd["residual"] == dd["residual"]:
                                gate["max_identity_residual"] = max(gate["max_identity_residual"], abs(dd["residual"]))
                            if rule == "fixed":
                                o = ref[(sp, s, b, d, m)]; oB = ref[(sp, s, b, d, "TvA-TS")]
                                key = f"errgap_a{int(round(100*t))}"
                                for val, oo in ((dd["gC"], o[key]), (dd["gB"], oB[key])):
                                    if val == val and oo == oo:
                                        gate["max_row_diff_points"] = max(gate["max_row_diff_points"], 100 * abs(val - oo))
                            rows.append(dict(seed=s, budget=b, draw=d, split=sp, t=t, rule=rule, method=m, **dd))
    with open(OUT / "mechanism_rows.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator="\n"); w.writeheader(); w.writerows(rows)
    # summaries
    draws = sorted({r["draw"] for r in rows})
    summ = {}
    idx = {(r["seed"], r["draw"], r["split"], r["t"], r["rule"], r["method"], r["budget"]): r for r in rows}
    for b in sorted({r["budget"] for r in rows}):
        seeds = sorted({r["seed"] for r in rows if r["budget"] == b})
        S_, D_ = len(seeds), len(draws)
        rng = np.random.default_rng(RNG); i_s, i_d = rng.integers(0, S_, (B, S_)), rng.integers(0, D_, (B, S_, D_))
        for sp in SPLITS:
            for t in TS:
                for rule in ("fixed", "matched"):
                    for m in MAPS[1:]:
                        e = {}
                        for q in ("S", "P", "gC", "gB", "absdiff", "overlap", "cov_C", "cov_B"):
                            A = np.array([[idx[(s, d, sp, t, rule, m, b)][q] for d in draws] for s in seeds], dtype=float)
                            bt = hier(A, i_s, i_d)
                            e[q] = {"mean": float(np.nanmean(np.nanmean(A, 1))), "ci95": [float(x) for x in np.nanpercentile(bt, [2.5, 97.5])],
                                    "per_seed": [float(x) for x in np.nanmean(A, 1)]}
                        e["seeds"] = S_
                        e["share_P"] = float(abs(e["P"]["mean"]) / (abs(e["S"]["mean"]) + abs(e["P"]["mean"])))
                        e["pred_absdiff_from_gB"] = float(abs(e["gB"]["mean"] + e["S"]["mean"] + e["P"]["mean"]) - abs(e["gB"]["mean"]))
                        summ[f"n={b}/{sp}/t={t}/{rule}/{m}"] = e
    out = {"plan": "A8(ii)", "B": B, "rng": RNG, "gate": gate, "summary": summ}
    (OUT / "analysis_mechanism.json").write_text(json.dumps(out, indent=1) + "\n")
    print("gate", gate)
    for key, e in summ.items():
        if "/t=0.9/" in key and "n=2500" in key:
            print(f"{key:55s} S={100*e['S']['mean']:+.2f} P={100*e['P']['mean']:+.2f} gB={100*e['gB']['mean']:+.2f} gC={100*e['gC']['mean']:+.2f} |d|={100*e['absdiff']['mean']:+.2f} shareP={e['share_P']:.2f} ovl={e['overlap']['mean']:.2f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""CivilComments validation, CPU stage 2: endpoint analysis (frozen before outcomes).
Primary family (4): |g(A_0.90)|_map - |g(A_0.90)|_TvA-TS at n = 2500, fixed threshold,
seed->draw hierarchical bootstrap (B = 20000, RNG 20261016), Bonferroni 1 - 0.05/4.
Descriptive: paired differences of ECE15, |count error|, |g| at t in {.80,.90,.95}
(fixed and matched coverage) for n in {250, 2500}; the endpoint-disagreement pattern
(lower ECE and count error with larger |g|); S/P accounting in both orders with
selection shares; the narrowing condition g_B < -D/2 against the observed change;
g within predicted-toxic and predicted-non-toxic accepted items.
Writes results/www2027/civil/analysis.json.
"""
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

RES = Path(__import__("os").environ.get("CIVIL_RES", "__SRTS_ROOT__/calattn-repo/results/www2027/civil"))
MAPS = ["HTS-BCE", "SRTS-BCE", "SMART+BCE", "PWLinear-3"]
TS = (80, 90, 95)
B, RNG = 20000, 20261016
ZB = 2.498  # two-sided 1 - 0.05/4 normal quantile is only used for labels; intervals are bootstrap percentiles


def hier(A, i_s, i_d):
    return np.nanmean(np.nanmean(A[i_s[:, :, None], i_d], axis=2), axis=1)


def main():
    rows = [json.loads(l) for f in sorted(glob.glob(str(RES / "rows_s*.jsonl"))) for l in open(f)]
    tab = {(r["seed"], r["budget"], r["draw"], r["method"]): r for r in rows}
    seeds = sorted({r["seed"] for r in rows}); draws = sorted({r["draw"] for r in rows})
    rng = np.random.default_rng(RNG)
    i_s, i_d = rng.integers(0, len(seeds), (B, len(seeds))), rng.integers(0, len(draws), (B, len(seeds), len(draws)))
    out = dict(seeds=seeds, draws=len(draws), B=B, rng=RNG, contrasts={}, levels={}, accounting={}, action={})

    def contrast(fn):
        A = np.array([[fn(s, d) for d in draws] for s in seeds], dtype=float)
        bt = hier(A, i_s, i_d)
        return dict(mean=float(np.nanmean(A)), ci95=[float(x) for x in np.nanpercentile(bt, [2.5, 97.5])],
                    ci_bonf4=[float(x) for x in np.nanpercentile(bt, [100 * 0.05 / 8, 100 * (1 - 0.05 / 8)])],
                    per_seed=[float(x) for x in np.nanmean(A, 1)])

    for n in (250, 2500):
        for m in MAPS:
            k = f"n={n}/{m}"
            g = lambda s, d, f: tab[(s, n, d, m)][f]
            gb = lambda s, d, f: tab[(s, n, d, "TvA-TS")][f]
            out["contrasts"][k + "/ece15"] = contrast(lambda s, d: g(s, d, "ece15") - gb(s, d, "ece15"))
            out["contrasts"][k + "/abs_count"] = contrast(lambda s, d: 100 * (abs(g(s, d, "count_rel_err")) - abs(gb(s, d, "count_rel_err"))))
            for t in TS:
                out["contrasts"][k + f"/abs_g{t}"] = contrast(lambda s, d: 100 * (abs(g(s, d, f"g_{t}")) - abs(gb(s, d, f"g_{t}"))))
                out["contrasts"][k + f"/abs_gm{t}"] = contrast(lambda s, d: 100 * (abs(g(s, d, f"gm_{t}")) - abs(gb(s, d, f"gm_{t}"))))
                for q in ("S", "P", "S2", "P2"):
                    out["accounting"][k + f"/{q}{t}"] = contrast(lambda s, d: 100 * g(s, d, f"{q}_{t}"))
                for nm in ("toxic", "nontoxic"):
                    out["action"][k + f"/g{t}_{nm}"] = contrast(lambda s, d: 100 * g(s, d, f"g_{t}_{nm}"))
                    out["action"][k + f"/gB{t}_{nm}"] = contrast(lambda s, d: 100 * gb(s, d, f"g_{t}_{nm}"))
        for m in ["TvA-TS"] + MAPS:
            for f in ["ece15", "count_rel_err"] + [f"g_{t}" for t in TS] + [f"cov_{t}" for t in TS]:
                v = np.array([[tab[(s, n, d, m)][f] for d in draws] for s in seeds], dtype=float)
                out["levels"][f"n={n}/{m}/{f}"] = float(np.nanmean(v))
    # primary family
    prim = {m: out["contrasts"][f"n=2500/{m}/abs_g90"] for m in MAPS}
    out["primary"] = {m: dict(mean=v["mean"], ci_bonf4=v["ci_bonf4"],
                              resolved=bool(v["ci_bonf4"][0] > 0 or v["ci_bonf4"][1] < 0)) for m, v in prim.items()}
    # disagreement pattern and narrowing condition (means over seeds and draws)
    pat = {}
    for n in (250, 2500):
        for m in MAPS:
            for t in TS:
                c = out["contrasts"]
                D = out["accounting"][f"n={n}/{m}/S{t}"]["mean"] + out["accounting"][f"n={n}/{m}/P{t}"]["mean"]
                gB = 100 * out["levels"][f"n={n}/TvA-TS/g_{t}"]
                pat[f"n={n}/{m}/t={t}"] = dict(
                    dECE=c[f"n={n}/{m}/ece15"]["mean"], dCount=c[f"n={n}/{m}/abs_count"]["mean"], dAbsG=c[f"n={n}/{m}/abs_g{t}"]["mean"],
                    disagreement=bool(c[f"n={n}/{m}/ece15"]["mean"] < 0 and c[f"n={n}/{m}/abs_count"]["mean"] < 0 and c[f"n={n}/{m}/abs_g{t}"]["mean"] > 0),
                    D=D, gB=gB, condition_narrows=bool((D > 0 and gB < -D / 2) or (D < 0 and gB > -D / 2)),
                    observed_narrows=bool(c[f"n={n}/{m}/abs_g{t}"]["mean"] < 0),
                    share_sel_fwd=abs(out["accounting"][f"n={n}/{m}/P{t}"]["mean"]) / (abs(out["accounting"][f"n={n}/{m}/S{t}"]["mean"]) + abs(out["accounting"][f"n={n}/{m}/P{t}"]["mean"]) + 1e-12),
                    share_sel_rev=abs(out["accounting"][f"n={n}/{m}/P2{t}"]["mean"]) / (abs(out["accounting"][f"n={n}/{m}/S2{t}"]["mean"]) + abs(out["accounting"][f"n={n}/{m}/P2{t}"]["mean"]) + 1e-12))
    out["pattern"] = pat
    (RES / "analysis.json").write_text(json.dumps(out, indent=1) + "\n")
    for m, v in out["primary"].items():
        print("PRIMARY", m, round(v["mean"], 3), [round(x, 3) for x in v["ci_bonf4"]], "resolved" if v["resolved"] else "not resolved")


if __name__ == "__main__":
    main()

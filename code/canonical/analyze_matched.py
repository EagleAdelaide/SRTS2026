#!/usr/bin/env python3
"""Plan addendum A4(iii): accepted-set gap at matched coverage.
For each split, budget and threshold t: paired per-draw deltas of
|g_matched| (map minus TvA-TS), where every map is evaluated on its ceil(qN)
most confident items and q is TvA-TS's coverage at t in the same draw.
Seed->draw hierarchical bootstrap (B = 20000, RNG 20261016), 95% and
Bonferroni over the 120 contrasts; seed-level t interval as sensitivity.
Also reports the fixed-threshold |g(A_t)| from the same refit for reference.
Output: results/www2027/posthoc_matched/analysis_matched.json
"""
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

W = Path("__SRTS_ROOT__/calattn-repo/results/www2027/posthoc_matched")
M = ["HTS-BCE", "SRTS-BCE", "SMART+BCE", "PWLinear-3"]
B, NF = 20000, 120
ZB = 100 * 0.05 / NF / 2
SPLITS = [("amazon_distilbert", "id_test"), ("amazon_distilbert", "ood_val"),
          ("amazon_distilbert", "ood_test_CONFIRMATION"),
          ("yahoo_distilbert", "id_test"), ("yahoo_distilbert", "ood_test_CONFIRMATION")]


def main():
    rows = [json.loads(l) for f in sorted(glob.glob(str(W / "*.jsonl"))) for l in open(f)]
    tab = {(r["task"], r["split"], r["seed"], r["budget"], r["draw"], r["method"]): r for r in rows}
    rng = np.random.default_rng(20261016)
    out = {"plan": "POSTHOC_PLAN_2026-10-16.md A4(iii)", "B": B, "family": NF, "contrasts": {}, "levels": {}}
    n_c = 0
    for task, sp in SPLITS:
        seeds = sorted({k[2] for k in tab if k[0] == task and k[1] == sp})
        for b in (250, 2500):
            draws = sorted({k[4] for k in tab if k[0] == task and k[1] == sp and k[3] == b})
            S, D = len(seeds), len(draws)
            i_s = rng.integers(0, S, (B, S)); i_d = rng.integers(0, D, (B, S, D))
            for t in (80, 90, 95):
                for m in ["TvA-TS"] + M:
                    vals = np.array([[tab[(task, sp, s, b, d, m)][f"g_matched_a{t}"] for d in draws] for s in seeds])
                    fixed = np.array([[tab[(task, sp, s, b, d, m)][f"errgap_a{t}"] for d in draws] for s in seeds])
                    out["levels"][f"{task}/{sp}/n={b}/t={t}/{m}"] = {
                        "g_matched": float(vals.mean()), "abs_g_matched": float(np.abs(vals).mean()),
                        "abs_g_fixed": float(np.abs(fixed).mean())}
                for m in M:
                    A = np.array([[abs(tab[(task, sp, s, b, d, m)][f"g_matched_a{t}"])
                                   - abs(tab[(task, sp, s, b, d, "TvA-TS")][f"g_matched_a{t}"])
                                   for d in draws] for s in seeds])
                    bt = A[i_s[:, :, None], i_d].mean(2).mean(1)
                    lo, hi = np.percentile(bt, [2.5, 97.5]); lb, hb = np.percentile(bt, [ZB, 100 - ZB])
                    ps = A.mean(1); h = stats.t.ppf(0.975, S - 1) * ps.std(ddof=1) / np.sqrt(S)
                    out["contrasts"][f"{task}/{sp}/n={b}/t={t}/{m}"] = {
                        "point": float(ps.mean()), "ci95": [float(lo), float(hi)],
                        "ci_bonf": [float(lb), float(hb)], "excl0_95": bool(lo > 0 or hi < 0),
                        "excl0_bonf": bool(lb > 0 or hb < 0), "per_seed": [float(x) for x in ps],
                        "seed_t95": [float(ps.mean() - h), float(ps.mean() + h)],
                        "seed_t95_excl0": bool(ps.mean() - h > 0 or ps.mean() + h < 0)}
                    n_c += 1
    assert n_c == NF, n_c
    (W / "analysis_matched.json").write_text(json.dumps(out, indent=1))
    print("wrote", W / "analysis_matched.json")


if __name__ == "__main__":
    main()

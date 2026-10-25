#!/usr/bin/env python3
"""EXPLORATORY SENSITIVITY (post-unblinding, labelled): reviewer-cluster
bootstrap of paired ECE deltas on the Amazon confirmation set for fixed
calibrators (draw 0 per seed), B = 500. Recovered from the command that
wrote EXPLORATORY_cluster_bootstrap.json on 2026-10-10."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import registry as R
from run_suite import load_cell, fit_methods, WEB
from sampler import natural_subsample
from exploratory_reviewer_level import reviewers

B = 500
OUT = sys.argv[1] if len(sys.argv) > 1 else str(WEB / "EXPLORATORY_cluster_bootstrap.json")
out = {}
for b in (250, 2500):
    deltas = {"SRTS-TvA": [], "SRTS-SMART": [], "HTS-TvA": []}
    for s in range(5):
        vl_p, vt_p, tests, _ = load_cell("amazon_distilbert", s, True)
        tl, tt = tests["ood_test_CONFIRMATION"]
        rev = reviewers(s)
        urev = np.unique(rev)
        rid_idx = {r: np.where(rev == r)[0] for r in urev}
        rng0 = np.random.default_rng(77_000_000 + 1_000_000 * s + 1000 * b + 0)
        idx, _ = natural_subsample(vt_p, b, rng0)
        fitted = fit_methods("core", vl_p[idx], vt_p[idx], s)
        P = {m: fitted[m][0](tl) for m in ("TvA-TS", "SMART+BCE", "SRTS-BCE", "HTS-BCE")}
        rng = np.random.default_rng(909_000 + s)
        reps = {k: [] for k in deltas}
        for _ in range(B):
            pick = rng.integers(0, len(urev), len(urev))
            sel = np.concatenate([rid_idx[urev[j]] for j in pick])
            e = {m: R.ece(P[m][sel], tt[sel]) for m in P}
            reps["SRTS-TvA"].append(e["SRTS-BCE"] - e["TvA-TS"])
            reps["SRTS-SMART"].append(e["SRTS-BCE"] - e["SMART+BCE"])
            reps["HTS-TvA"].append(e["HTS-BCE"] - e["TvA-TS"])
        for k in deltas:
            deltas[k].append(reps[k])
    res = {}
    for k, v in deltas.items():
        arr = np.array(v)
        seed_means = arr.mean(1); pooled = arr.mean(0)
        res[k] = dict(point=float(seed_means.mean()),
                      cluster_ci=[float(np.percentile(pooled, 2.5)), float(np.percentile(pooled, 97.5))],
                      per_seed_point=[float(x) for x in seed_means])
    out[b] = res
json.dump(out, open(OUT, "w"), indent=1)
print("DONE", OUT)

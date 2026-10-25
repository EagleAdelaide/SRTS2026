#!/usr/bin/env python3
"""CivilComments validation, CPU stage 3: within-pool audit (frozen before outcomes).
Uses the draw-0 confidence caches.  Audit pool = test_pool 0 among the evaluated test
rows (an item-level hash split: the source has no user or article ids, so cross-cohort
transfer is NOT tested).  Branches: fixed and exact_matched_within_pool.
Estimation of D (truth = audit pool): E3, E4, E4ac, E6, E8, E9, E10, E11 at
m in {25, 50, 100, 250, 500, 1000}.  Decisions (truth = audit pool, finite-population
variances): G1 (corrected guardrail), G2 (plug-in H test; estimates D and ignores its
uncertainty), G3 (joint (D,H) intersection-union test) at label-budget caps
m in {25, 50, 100, 250, 500, 1000, 2500, 5000}.  R = 400 audits per cell;
RNG 20261028 + crc32(cell key) mod 1e8.
Writes results/www2027/civil/audit_{est,dec}_s{seed}.csv
"""
import csv
import os
import sys
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "canonical"))
from paired_audit.decisions import Decider, truth  # noqa: E402
from paired_audit.estimators import LabelTerm  # noqa: E402
from paired_audit.identities import audit_weights, d_conf  # noqa: E402
from paired_audit.pools import accepted_sets  # noqa: E402

RES = Path(__import__("os").environ.get("CIVIL_RES", "__SRTS_ROOT__/calattn-repo/results/www2027/civil"))
MAPS = ["TvA-TS", "HTS-BCE", "SRTS-BCE", "SMART+BCE", "PWLinear-3"]
TS = [0.80, 0.90, 0.95]
MEST = [25, 50, 100, 250, 500, 1000]
MDEC = [25, 50, 100, 250, 500, 1000, 2500, 5000]
EST = ["E3", "E4", "E4ac", "E6", "E8", "E9", "E10", "E11"]
DEC = ["G1", "G2", "G3"]
R = int(os.environ.get("CIVIL_R", "400"))
RNG0 = 20261028


def rng_for(*key):
    return np.random.default_rng(RNG0 + zlib.crc32("/".join(map(str, key)).encode()) % 10 ** 8)


def task(seed):
    est_rows, dec_rows = [], []
    for n in (250, 2500):
        z = np.load(RES / f"cache_s{seed}_n{n}.npz")
        conf = z["conf"].astype(np.float64); b = z["corr"].astype(np.float64); aud = z["pool"] == 0
        cB = conf[0]
        for t in TS:
            for mi, mname in enumerate(MAPS[1:], start=1):
                cC = conf[mi]
                for branch in ("fixed", "exact_matched_within_pool"):
                    AC, AB = accepted_sets(cB, cC, aud, t, branch)
                    if min(AC.sum(), AB.sum()) < 10:
                        continue
                    w = audit_weights(AC, AB); dc = d_conf(cC, cB, AC, AB); tr = truth(cC, cB, b, AC, AB)
                    base = dict(seed=seed, budget=n, t=t, rule=branch, method=mname, nA_C=int(AC.sum()), nA_B=int(AB.sum()),
                                support_w=int((w != 0).sum()), **{f"{k}_a": v for k, v in tr.items()})
                    lt = LabelTerm(w, AC, AB, surrogate=cB)
                    for m in MEST:
                        for d in EST:
                            rng = rng_for("est", seed, n, t, mname, branch, m, d)
                            o = np.array([lt.estimate(d, b, m, rng) for _ in range(R)])
                            est = dc - o[:, 0]; se = np.sqrt(np.maximum(o[:, 1], 0)); err = est - tr["D"]
                            est_rows.append(dict(base, m=m, design=d, rmse=float(np.sqrt(np.mean(err ** 2))), bias=float(err.mean()),
                                                 coverage95=float(np.mean(np.abs(err) <= 1.96 * se + 1e-12)),
                                                 unique_labels=float(o[:, 2].mean())))
                    dcd = Decider(cC, cB, b, AC, AB)
                    for m in MDEC:
                        for p in DEC:
                            rng = rng_for("dec", seed, n, t, mname, branch, m, p)
                            o = np.array([dcd.decide(p, m, rng) for _ in range(R)], dtype=float)
                            dec_rows.append(dict(base, m=m, proc=p, sw_fpc=float(o[:, 0].mean()), labels=float(o[:, 2].mean()),
                                                 support_U=dcd.U))
    for name, rows in (("est", est_rows), ("dec", dec_rows)):
        with open(RES / f"audit_{name}_s{seed}.csv", "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator="\n"); wr.writeheader(); wr.writerows(rows)
    return seed


if __name__ == "__main__":
    seeds = [int(x) for x in sys.argv[1:]] or [0, 1, 2, 3, 4]
    with Pool(len(seeds)) as p:
        for s in p.imap_unordered(task, seeds):
            print("done", s, flush=True)

#!/usr/bin/env python3
"""Plan A10: estimand repair, error-centred controls, decisions within pool and
under transfer.  One task per (cache file, split); draw 0 of each seed (as A8).
Estimation (target = the audit pool's own D):
  exact_matched_within_pool : E1-E11
  fixed, stream_matched_then_restricted : E8-E11 (E1-E7 are in the A8/A9 rows)
Decisions (branches fixed and exact_matched_within_pool): G0, G1, G2, G3, G3d, G4 at
m in MGRID_DEC; per audit both the fixed-pool (FPC) and the future-population (no FPC)
decision; truths for the audit pool (experiment A) and the evaluation pool (B).
RNG: 20261027 + crc32(cell key) mod 10^8 (deterministic, order-independent).
Coverage counts |error| <= 1.96 SE + 1e-12 (census cells have SE 0 and float error ~1e-17);
support_w = #{w != 0}; census_share = share of audits whose labels covered supp(w).
Output: results/www2027/posthoc_closure/{est,dec}_s{seed}_n{budget}_{split}.csv
"""
import csv
import glob
import os
import sys
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paired_audit.decisions import Decider, truth  # noqa: E402
from paired_audit.estimators import LabelTerm  # noqa: E402
from paired_audit.identities import audit_weights, d_conf  # noqa: E402
from paired_audit.pools import accepted_sets  # noqa: E402
from paired_audit.simulate_audit import CACHE, MAPS, pools  # noqa: E402

OUT = Path("__SRTS_ROOT__/calattn-repo/results/www2027/posthoc_closure")
TS = [0.80, 0.90, 0.95]
MGRID_EST = [25, 50, 100, 250, 500, 1000]
MGRID_DEC = [25, 50, 100, 250, 500, 1000, 2500, 5000]
R = int(os.environ.get("CLOSURE_R", "400"))
RNG0 = 20261027
EST_ALL = ["E1", "E2", "E3", "E4", "E4ac", "E5", "E6", "E7", "E8", "E9", "E10", "E11"]
EST_NEW = ["E4ac", "E8", "E9", "E10", "E11"]
PROCS = ["G0", "G1", "G2", "G3", "G3d", "G4"]


def rng_for(*key):
    return np.random.default_rng(RNG0 + zlib.crc32("/".join(map(str, key)).encode()) % 10 ** 8)


def task(args):
    f, sp = args
    z = np.load(f)
    s = int(Path(f).stem.split("_s")[1].split("_n")[0]); bud = int(Path(f).stem.split("_n")[1])
    conf = z[f"conf_{sp}"][0].astype(np.float64); corr = z[f"corr_{sp}"].astype(np.float64)
    aud, ev = pools(z[f"rev_{sp}"]); cB = conf[0]
    est_rows, dec_rows = [], []
    for t in TS:
        for mi, mname in enumerate(MAPS[1:], start=1):
            cC = conf[mi]
            for branch in ("fixed", "stream_matched_then_restricted", "exact_matched_within_pool"):
                ACa, ABa = accepted_sets(cB, cC, aud, t, branch)
                ACe, ABe = accepted_sets(cB, cC, ev, t, branch)
                if min(ACa.sum(), ABa.sum(), ACe.sum(), ABe.sum()) < 10:
                    continue
                w = audit_weights(ACa, ABa); dconf = d_conf(cC, cB, ACa, ABa)
                tr_a, tr_e = truth(cC, cB, corr, ACa, ABa), truth(cC, cB, corr, ACe, ABe)
                ovw = float(abs(w[ACa & ABa]).max()) if (ACa & ABa).any() else 0.0
                if branch == "exact_matched_within_pool":
                    assert ovw == 0.0 and ACa.sum() == ABa.sum() and ACe.sum() == ABe.sum()
                base = dict(seed=s, budget=bud, split=sp, t=t, rule=branch, method=mname)
                lt = LabelTerm(w, ACa, ABa, surrogate=cB)
                designs = EST_ALL if branch == "exact_matched_within_pool" else EST_NEW
                for m in MGRID_EST:
                    for d in designs:
                        rng = rng_for("est", s, bud, sp, t, mname, branch, m, d)
                        out = np.array([lt.estimate(d, corr, m, rng) for _ in range(R)])
                        est = dconf - out[:, 0]; se = np.sqrt(np.maximum(out[:, 1], 0))
                        err = est - tr_a["D"]
                        est_rows.append(dict(base, m=m, design=d, delta_true=tr_a["D"], bias=float(err.mean()),
                                             rmse=float(np.sqrt(np.mean(err ** 2))), mc_sd=float(est.std()),
                                             mean_se=float(se.mean()), coverage95=float(np.mean(np.abs(err) <= 1.96 * se + 1e-12)),
                                             support_w=int((w != 0).sum()), census_share=float(np.mean(out[:, 2] >= (w != 0).sum())),
                                             unique_labels=float(out[:, 2].mean()), nA_C=int(ACa.sum()), nA_B=int(ABa.sum()),
                                             symdiff=int((ACa ^ ABa).sum()), overlap_weight=ovw))
                if branch == "stream_matched_then_restricted" or os.environ.get("CLOSURE_SKIP_DEC"):
                    continue
                dc = Decider(cC, cB, corr, ACa, ABa)
                for m in MGRID_DEC:
                    for proc in PROCS:
                        rng = rng_for("dec", s, bud, sp, t, mname, branch, m, proc)
                        res = np.array([dc.decide(proc, m, rng) for _ in range(R)], dtype=float)
                        dec_rows.append(dict(base, m=m, proc=proc, sw_fpc=float(res[:, 0].mean()), sw_nofpc=float(res[:, 1].mean()),
                                             labels=float(res[:, 2].mean()), support_U=dc.U, nA_B=int(ABa.sum()),
                                             **{f"{k}_a": v for k, v in tr_a.items()}, **{f"{k}_e": v for k, v in tr_e.items()}))
    tag = f"s{s}_n{bud}_{sp}"
    for name, rows in (("est", est_rows), ("dec", dec_rows)):
        if not rows:
            continue
        with open(OUT / f"{name}_{tag}.csv", "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator="\n"); wr.writeheader(); wr.writerows(rows)
    return tag, len(est_rows), len(dec_rows)


def main():
    OUT.mkdir(exist_ok=True)
    files = sorted(glob.glob(str(CACHE / "cache_s*_n*.npz")))
    tasks = [(f, sp) for f in files for sp in ("id_test", "ood_val")]
    only = os.environ.get("CLOSURE_ONLY")
    if only:
        tasks = [x for x in tasks if only in Path(x[0]).stem + "_" + x[1]]
    with Pool(int(os.environ.get("CLOSURE_PROCS", "20"))) as p:
        for tag, ne, nd in p.imap_unordered(task, tasks):
            print("done", tag, ne, nd, flush=True)


if __name__ == "__main__":
    main()

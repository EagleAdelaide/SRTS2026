#!/usr/bin/env python3
"""CivilComments validation, CPU stage 1 (amended protocol; frozen before any test
outcome is inspected).  For seeds 0-4, n in {250, 2500}, draws 0-19: exact-budget
uniform draw from the calibration pool (RNG 77e6 + 1e6 s + 1e3 n + d, as for Amazon),
fit the five core maps (run_suite.fit_methods("core")), evaluate on the test rows
kept by the duplicate rule (test_eval_mask):
  ece15, count_rel_err (whole test), and per t in {0.80, 0.90, 0.95}: coverage,
  g = mean(c - b) on A_t (= R.ops_metrics err_gap), matched-coverage g (each map's
  top-k, k = scalar's |A_t|), the S/P accounting in both orders against TvA-TS, and
  g and coverage within A_t restricted to predicted toxic and predicted non-toxic
  labels (action-specific descriptive check).
HTS temperatures are clipped to [1e-4, 1e4] at prediction (amendment F1).
Writes results/www2027/civil/rows_s{seed}.jsonl and cache_s{seed}_n{n}.npz (draw-0
confidences of the five maps on the evaluated test rows, correctness, predicted
class, audit/evaluation pool).
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "canonical"))
import registry as R  # noqa: E402
from run_suite import fit_methods  # noqa: E402
from sampler import natural_subsample  # noqa: E402
import maps as _maps  # noqa: E402

# Amendment F1 (2026-10-22, before any test outcome; CIVILCOMMENTS_FROZEN_PROTOCOL.md section 6):
# numerically stable HTS evaluation.  The HTS temperature is clipped to [1e-4, 1e4] at
# prediction; the lower bound is the fit objective's own clip (maps.toplabel_bce_perT), the
# upper bound binds only for diverged fits.  Fits and non-degenerate draws are unchanged.
_hts_temperature = _maps.HTS.temperature
_maps.HTS.temperature = lambda self, logits: np.clip(_hts_temperature(self, logits), 1e-4, 1e4)

DATA = Path("__SRTS_ROOT__/data/civil_comments_hf/prepared_v1")
RES = Path(__import__("os").environ.get("CIVIL_RES", "__SRTS_ROOT__/calattn-repo/results/www2027/civil"))
MAPS = ["TvA-TS", "HTS-BCE", "SRTS-BCE", "SMART+BCE", "PWLinear-3"]
TS = (0.80, 0.90, 0.95)
BUDGETS = (250, 2500)
DRAWS = range(20)


def gap(c, b, A):
    return float((c[A] - b[A]).mean()) if A.any() else float("nan")


def topk(c, k):
    m = np.zeros(len(c), bool)
    if k > 0:
        m[np.argsort(-c, kind="stable")[:k]] = True
    return m


def main(seed):
    meta = np.load(DATA / "meta.npz")
    cal = np.load(RES / f"distilbert_s{seed}/logits_val_calib.npz")
    te = np.load(RES / f"distilbert_s{seed}/logits_test.npz")
    off = int(meta["test"][0])
    keep = meta["test_eval_mask"][te["row_idx"] - off]
    pool = meta["test_pool"][te["row_idx"] - off][keep]
    vl, vt = cal["logits"].astype(np.float64), cal["targets"].astype(np.int64)
    tl, tt = te["logits"][keep].astype(np.float64), te["targets"][keep].astype(np.int64)
    rows = []
    for n in BUDGETS:
        cache = dict(conf=np.zeros((len(MAPS), len(tt)), np.float32))
        for d in DRAWS:
            rng = np.random.default_rng(77_000_000 + 1_000_000 * seed + 1000 * n + d)
            idx, _ = natural_subsample(vt, n, rng)
            fm = fit_methods("core", vl[idx], vt[idx], seed)
            P = {m: fm[m][0](tl) for m in MAPS}
            C = {}
            for m in MAPS:
                c, pred, b = R._conf_pred_corr(P[m], tt)
                C[m] = (c, pred, b)
            b = C["TvA-TS"][2]; pred = C["TvA-TS"][1]
            assert all(np.array_equal(C[m][1], pred) for m in MAPS), "argmax changed"
            cB = C["TvA-TS"][0]
            for m in MAPS:
                c = C[m][0]
                ops = R.ops_metrics(P[m], tt)
                r = dict(seed=seed, budget=n, draw=d, method=m, ece15=R.ece(P[m], tt),
                         count_rel_err=float(ops["count_rel_err"]), n_eval=int(len(tt)))
                for t in TS:
                    A, AB = c >= t, cB >= t
                    k = int(AB.sum()); AM, ABM = topk(c, k), topk(cB, k)
                    r[f"cov_{int(100*t)}"] = float(A.mean())
                    r[f"g_{int(100*t)}"] = gap(c, b, A)
                    r[f"gm_{int(100*t)}"] = gap(c, b, AM)
                    r[f"gB_{int(100*t)}"] = gap(cB, b, AB)
                    r[f"gBm_{int(100*t)}"] = gap(cB, b, ABM)
                    # accounting (fixed threshold): forward S on A_C, P with scalar residual; reverse S' on A_B, P' with map residual
                    r[f"S_{int(100*t)}"] = float((c - cB)[A].mean()) if A.any() else float("nan")
                    r[f"P_{int(100*t)}"] = (gap(cB, b, A) - gap(cB, b, AB))
                    r[f"S2_{int(100*t)}"] = float((c - cB)[AB].mean()) if AB.any() else float("nan")
                    r[f"P2_{int(100*t)}"] = (gap(c, b, A) - gap(c, b, AB))
                    for cls, nm in ((1, "toxic"), (0, "nontoxic")):
                        Ak = A & (pred == cls)
                        r[f"g_{int(100*t)}_{nm}"] = gap(c, b, Ak)
                        r[f"cov_{int(100*t)}_{nm}"] = float(Ak.mean())
                rows.append(r)
            if d == 0:
                for mi, m in enumerate(MAPS):
                    cache["conf"][mi] = C[m][0]
                cache["corr"] = b.astype(np.int8); cache["pred"] = pred.astype(np.int8); cache["pool"] = pool.astype(np.int8)
        np.savez_compressed(RES / f"cache_s{seed}_n{n}.npz", maps=np.array(MAPS), **cache)
        print("done", seed, n, flush=True)
    with open(RES / f"rows_s{seed}.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main(int(sys.argv[1]))

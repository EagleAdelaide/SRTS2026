#!/usr/bin/env python3
"""Plan addendum A4(iii): accepted-set gap at matched coverage, plus a
refit of the confirmation / development rows from logits (ungated; writes to
results/www2027/posthoc_matched/, never to CONFIRMATION/).

For each task, seed, budget in {250, 2500} and draw: refit TvA-TS, HTS-BCE,
SRTS-BCE, SMART+BCE and PWLinear-3 (original construction, in-sample scores)
through run_suite.fit_methods("core"); on every evaluation split
compute ECE15, the frozen ops metrics, and for t in {.80,.90,.95} the gap
g = mean(c - b) over each map's ceil(q N) most confident items, where q is
TvA-TS's coverage at t in the same draw.
Usage: python run_matched_coverage.py --task amazon_distilbert --seed 0
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")  # same as run_suite; SMART (torch) depends on it
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import registry as R
from maps import HTS, RiskMap
from run_suite import load_cell, fit_methods
from sampler import natural_subsample
from smart import train_smart, smart_predict_proba
from srts import SRTS, fit_tva

OUT = Path("__SRTS_ROOT__/calattn-repo/results/www2027/posthoc_matched")
TS = (80, 90, 95)


def fit5(vl, vt, seed):
    """Use run_suite.fit_methods("core") itself: fitting order matters for
    SMART (an unseeded global state), so a separate fitter would not
    reproduce the saved rows."""
    fm = fit_methods("core", vl, vt, seed)
    return {m: fm[m][0] for m in ("TvA-TS", "HTS-BCE", "SRTS-BCE",
                                  "SMART+BCE", "PWLinear-3")}


def matched(conf, corr, k):
    o = np.argsort(-conf, kind="stable")[:k]
    return float((conf[o] - corr[o]).mean()) if k > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--budgets", default="250,2500")
    ap.add_argument("--draws", default="0-19")
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.draws.split("-"))
    OUT.mkdir(parents=True, exist_ok=True)
    fout = OUT / f"{a.task}_s{a.seed}.jsonl"
    done = set()
    if fout.exists():
        done = {(r["budget"], r["draw"]) for r in map(json.loads, open(fout))}
    vlp, vtp, tests, _ = load_cell(a.task, a.seed, True)
    splits = {k: v for k, v in tests.items()
              if k in ("id_test", "ood_val", "ood_test_CONFIRMATION")}
    with fout.open("a") as fh:
        for b in [int(x) for x in a.budgets.split(",")]:
            for d in range(lo, hi + 1):
                if (b, d) in done:
                    continue
                t0 = time.time()
                rng = np.random.default_rng(77_000_000 + 1_000_000 * a.seed + 1000 * b + d)
                idx, _ = natural_subsample(vtp, b, rng)
                fits = fit5(vlp[idx], vtp[idx], a.seed)
                rows = []
                for sp, (tl, tt) in splits.items():
                    P = {m: f(tl) for m, f in fits.items()}
                    N = len(tt)
                    qref = {t: float((P["TvA-TS"].max(1) >= t / 100).mean()) for t in TS}
                    for m, p in P.items():
                        conf = p.max(1); corr = (p.argmax(1) == tt).astype(float)
                        ops = R.ops_metrics(p, tt)
                        row = dict(task=a.task, split=sp, seed=a.seed, budget=b, draw=d,
                                   method=m, ece15=R.ece(p, tt),
                                   count_rel_err=ops["count_rel_err"], n=N)
                        for t in TS:
                            k = int(math.ceil(qref[t] * N))
                            row[f"cov_a{t}"] = ops[f"cov_a{t}"]
                            row[f"errgap_a{t}"] = ops[f"errgap_a{t}"]
                            row[f"qref_a{t}"] = qref[t]
                            row[f"g_matched_a{t}"] = matched(conf, corr, k)
                        rows.append(row)
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
                fh.flush()
                print(f"{a.task} s{a.seed} n{b} d{d} {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()

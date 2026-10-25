#!/usr/bin/env python3
"""Plan addendum A6, stage 1: per-reviewer accepted-set sums for the
post-hoc reviewer-cluster sensitivity (n = 2500, t = 0.90, fixed threshold).

For every seed (0-4) and draw (0-19) the calibrators are fit once with
run_suite.fit_methods("core") and the stored RNG rule (4 CPU threads, as in
A4(iii)); for TvA-TS and the four conditional maps and for the ID test, OOD
validation and confirmation splits, the cache stores per reviewer
S = sum over accepted items of (c - b) and N = number of accepted items
(accepted: c >= 0.90). Gate (A6 iv): recomputed gap and coverage equal the
A4(iii) rows (<= 1e-12; PWLinear-3 <= 1e-3 points) and S/N reproduces the
gap. Cost gate (A6 v): stops after seed 0 if 5 x seed-0 CPU + 0.25 h >
2.0 CPU-hours, and whenever cumulative CPU exceeds 2.0 h.
Output: results/www2027/posthoc_reviewer_cluster/cache_s{seed}.npz, gate.json
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")  # as run_suite / run_matched_coverage
import glob
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import registry as R  # noqa: E402
from exploratory_reviewer_level import META  # noqa: E402
from run_suite import WEB, fit_methods, load_cell  # noqa: E402
from sampler import natural_subsample  # noqa: E402

OUT = WEB / "posthoc_reviewer_cluster"
TASK, BUDGET, T = "amazon_distilbert", 2500, 0.90
MAPS = ["TvA-TS", "HTS-BCE", "SRTS-BCE", "SMART+BCE", "PWLinear-3"]
SPLITS = {"id_test": "logits_id_test.npz", "ood_val": "logits_val.npz",
          "ood_test_CONFIRMATION": "logits_test.npz"}
SEEDS, DRAWS = range(5), range(20)
BUDGET_CPU_H, BOOT_ALLOW_H = 2.0, 0.25
TOL = {"PWLinear-3": 1e-3}


def cpu_hours():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return (r.ru_utime + r.ru_stime) / 3600.0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ref = {}
    for f in glob.glob(str(WEB / f"posthoc_matched/{TASK}_s*.jsonl")):
        for r in map(json.loads, open(f)):
            if r["budget"] == BUDGET:
                ref[(r["split"], r["seed"], r["draw"], r["method"])] = r
    rev_all = None  # official metadata, used when the npz has no joined reviewer field
    gate = {"max_abs_diff_points": {m: 0.0 for m in MAPS}, "violations": [],
            "sum_identity_max": 0.0, "empty_accepted": 0, "cpu_hours_after_seed": {}}
    t0 = time.time()
    for s in SEEDS:
        vlp, vtp, tests, _ = load_cell(TASK, s, True)
        inv, R_n = {}, {}
        for sp, fn in SPLITS.items():
            z = np.load(WEB / f"wilds_amazon/distilbert_s{s}/{fn}")
            if "reviewer" in z.files:  # layout written by join_web_labels.py
                rv = z["reviewer"]
            else:
                rev_all = np.load(META)["reviewer"] if rev_all is None else rev_all
                rv = rev_all[z["row_idx"]]
            u, inv[sp] = np.unique(rv, return_inverse=True)
            R_n[sp] = len(u)
        S = {sp: np.zeros((len(DRAWS), len(MAPS), R_n[sp])) for sp in SPLITS}
        N = {sp: np.zeros((len(DRAWS), len(MAPS), R_n[sp]), dtype=np.int64) for sp in SPLITS}
        for d in DRAWS:
            rng = np.random.default_rng(77_000_000 + 1_000_000 * s + 1000 * BUDGET + d)
            idx, _ = natural_subsample(vtp, BUDGET, rng)
            fm = fit_methods("core", vlp[idx], vtp[idx], s)
            for sp in SPLITS:
                tl, tt = tests[sp]
                for mi, m in enumerate(MAPS):
                    p = fm[m][0](tl)
                    conf, _, corr = R._conf_pred_corr(p, tt)
                    acc = conf >= T
                    S[sp][d, mi] = np.bincount(inv[sp][acc], weights=(conf - corr)[acc],
                                               minlength=R_n[sp])
                    N[sp][d, mi] = np.bincount(inv[sp][acc], minlength=R_n[sp])
                    ops = R.ops_metrics(p, tt)
                    o = ref[(sp, s, d, m)]
                    if N[sp][d, mi].sum() == 0:
                        gate["empty_accepted"] += 1
                        continue
                    g_sum = S[sp][d, mi].sum() / N[sp][d, mi].sum()
                    gate["sum_identity_max"] = max(gate["sum_identity_max"],
                                                   abs(g_sum - ops["errgap_a90"]))
                    dd = 100 * max(abs(ops["errgap_a90"] - o["errgap_a90"]),
                                   abs(ops["cov_a90"] - o["cov_a90"]))
                    gate["max_abs_diff_points"][m] = max(gate["max_abs_diff_points"][m], dd)
                    if dd > TOL.get(m, 1e-10):
                        gate["violations"].append([sp, s, d, m, dd])
            print(f"s{s} d{d} {time.time() - t0:.0f}s cpu {cpu_hours():.3f}h", flush=True)
        np.savez_compressed(OUT / f"cache_s{s}.npz",
                            **{f"S_{sp}": S[sp] for sp in SPLITS},
                            **{f"N_{sp}": N[sp] for sp in SPLITS},
                            maps=np.array(MAPS))
        ch = cpu_hours()
        gate["cpu_hours_after_seed"][s] = ch
        hard = [v for v in gate["violations"] if v[3] != "PWLinear-3"]
        stop = None
        if hard:
            stop = "reproduction gate failed for a non-PWLinear-3 map"
        elif s == 0 and 5 * ch + BOOT_ALLOW_H > BUDGET_CPU_H:
            stop = f"cost gate: projected {5 * ch + BOOT_ALLOW_H:.2f} CPU-h > {BUDGET_CPU_H}"
        elif ch > BUDGET_CPU_H:
            stop = f"cost gate: cumulative {ch:.2f} CPU-h > {BUDGET_CPU_H}"
        gate["stopped"] = stop
        (OUT / "gate.json").write_text(json.dumps(gate, indent=1) + "\n")
        if stop:
            print("STOP:", stop)
            sys.exit(1)
    gate["completed"] = True
    gate["reviewers"] = R_n
    (OUT / "gate.json").write_text(json.dumps(gate, indent=1) + "\n")
    print("done", json.dumps({k: v for k, v in gate.items() if k != "violations"}))


if __name__ == "__main__":
    main()

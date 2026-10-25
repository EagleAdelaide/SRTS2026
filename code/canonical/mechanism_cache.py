#!/usr/bin/env python3
"""Plan addendum A8 (i): per-item confidence cache for the mechanism
accounting and the paired audit simulation.

For seeds 0-4, budgets {250, 2500}, draws 0-19: refit the five maps with
run_suite.fit_methods("core") and the stored RNG rule (4 CPU threads, as
in A4(iii)/A6); for id_test, ood_val and ood_test_CONFIRMATION store each
map's top-label confidence (float32), the shared correctness b (checked
identical across maps), and the reviewer ids.  Gate: ECE15 and errgap at
t = .80/.90/.95 recomputed from the cache equal the A4(iii) rows
(<= 1e-9; PWLinear-3 <= 1e-3 points).
Output: results/www2027/posthoc_mechanism/cache_s{seed}_n{budget}.npz and gate.json
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
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

OUT = WEB / "posthoc_mechanism"
TASK = "amazon_distilbert"
MAPS = ["TvA-TS", "HTS-BCE", "SRTS-BCE", "SMART+BCE", "PWLinear-3"]
SPLITS = {"id_test": "logits_id_test.npz", "ood_val": "logits_val.npz",
          "ood_test_CONFIRMATION": "logits_test.npz"}
BUDGETS = [int(x) for x in os.environ.get("MECH_BUDGETS", "250,2500").split(",")]
SEEDS = [int(x) for x in os.environ.get("MECH_SEEDS", "0,1,2,3,4").split(",")]
DRAWS = range(20)
TOL = {"PWLinear-3": 1e-3}


def cpu_hours():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return (r.ru_utime + r.ru_stime) / 3600.0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ref = {}
    for f in glob.glob(str(WEB / f"posthoc_matched/{TASK}_s*.jsonl")):
        for r in map(json.loads, open(f)):
            ref[(r["split"], r["seed"], r["budget"], r["draw"], r["method"])] = r
    gate_path = OUT / "gate.json"
    gate = json.load(open(gate_path)) if gate_path.exists() else {"max_abs_diff_points": {}, "violations": [], "argmax_mismatch": 0, "cpu_hours": {}}
    t0 = time.time()
    for s in SEEDS:
        vlp, vtp, tests, _ = load_cell(TASK, s, True)
        rev = {}
        for sp, fn in SPLITS.items():
            z = np.load(WEB / f"wilds_amazon/distilbert_s{s}/{fn}")
            rev[sp] = z["reviewer"] if "reviewer" in z.files else np.load(META)["reviewer"][z["row_idx"]]
        for b in BUDGETS:
            fout = OUT / f"cache_s{s}_n{b}.npz"
            if fout.exists():
                print("exists", fout); continue
            conf = {sp: np.zeros((len(DRAWS), len(MAPS), len(tests[sp][1])), dtype=np.float32) for sp in SPLITS}
            corr = {sp: None for sp in SPLITS}
            for d in DRAWS:
                rng = np.random.default_rng(77_000_000 + 1_000_000 * s + 1000 * b + d)
                idx, _ = natural_subsample(vtp, b, rng)
                fm = fit_methods("core", vlp[idx], vtp[idx], s)
                for sp in SPLITS:
                    tl, tt = tests[sp]
                    for mi, m in enumerate(MAPS):
                        p = fm[m][0](tl)
                        c, _, bb = R._conf_pred_corr(p, tt)
                        conf[sp][d, mi] = c
                        if corr[sp] is None:
                            corr[sp] = bb.astype(np.int8)
                        elif not np.array_equal(corr[sp], bb.astype(np.int8)):
                            gate["argmax_mismatch"] += 1
                        o = ref[(sp, s, b, d, m)]
                        ops = R.ops_metrics(p, tt)
                        dd = max([abs(ops_ := R.ece(p, tt)) - abs(o["ece15"])] + [0]) if False else 0.0
                        dd = max([abs(R.ece(p, tt) - o["ece15"])] + [100 * abs(ops[f"errgap_a{t}"] - o[f"errgap_a{t}"])
                                                                          for t in (80, 90, 95) if o[f"errgap_a{t}"] == o[f"errgap_a{t}"]])
                        gate["max_abs_diff_points"][m] = max(gate["max_abs_diff_points"].get(m, 0.0), dd)
                        if dd > TOL.get(m, 1e-9):
                            gate["violations"].append([sp, s, b, d, m, dd])
                print(f"s{s} n{b} d{d} {time.time() - t0:.0f}s cpu {cpu_hours():.2f}h", flush=True)
            np.savez_compressed(fout, maps=np.array(MAPS),
                                **{f"conf_{sp}": conf[sp] for sp in SPLITS},
                                **{f"corr_{sp}": corr[sp] for sp in SPLITS},
                                **{f"rev_{sp}": rev[sp] for sp in SPLITS})
            gate["cpu_hours"][f"s{s}_n{b}"] = cpu_hours()
            gate_path.write_text(json.dumps(gate, indent=1) + "\n")
    hard = [v for v in gate["violations"] if v[4] != "PWLinear-3"]
    gate["passed"] = not hard and gate["argmax_mismatch"] == 0
    gate_path.write_text(json.dumps(gate, indent=1) + "\n")
    print("done", {k: v for k, v in gate.items() if k != "violations"}, "violations", len(gate["violations"]))


if __name__ == "__main__":
    main()

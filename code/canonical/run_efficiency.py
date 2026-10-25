#!/usr/bin/env python3
"""F-10 / NEW-05: measured efficiency protocol. Fit and inference are
reported SEPARATELY; nothing is inferred from parameter counts.

Measured on the declared hardware (CPU limited to 4 threads; this host's
model recorded). For each calibrator, on the Amazon seed-0 cell:
  fit_seconds        median over R fits on n=2500 (warm)
  inf_batch1_us      per-sample latency, batch=1, p50/p95 over 2000 calls
                     (after 100 warmup)
  inf_batch256_us    per-sample latency at batch 256
  serialized_bytes   pickle size of deployed state
  peak_rss_mb        resident set delta during fit
End-to-end context: DistilBERT fp32 CPU forward latency measured on the
same host (batch 1 and 32, tokenized length 512) so calibrator overhead
can be stated as a fraction of model inference. Backbone GPU latency is
reported separately from its training logs' throughput; we do not claim
GPU figures from CPU runs.
Output: results/www2027/efficiency.json
"""
import json
import os
import pickle
import resource
import sys
import time
from pathlib import Path

os.environ.update({k: "4" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                    "OPENBLAS_NUM_THREADS")})
import numpy as np
import torch
torch.set_num_threads(4)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_suite import load_cell
from srts import SRTS, fit_tva
from maps import HTS, fit_ts_nll, RiskMap
from smart import train_smart, smart_predict_proba

OUT = Path("__SRTS_ROOT__/calattn-repo/results/www2027")


def timeit_inf(fn, z1, zB, n=2000, warm=100):
    for _ in range(warm):
        fn(z1)
    t = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn(z1)
        t.append(time.perf_counter() - t0)
    t = np.array(t) * 1e6
    tb = []
    for _ in range(200):
        t0 = time.perf_counter()
        fn(zB)
        tb.append((time.perf_counter() - t0) / len(zB) * 1e6)
    return dict(batch1_us_p50=float(np.percentile(t, 50)),
                batch1_us_p95=float(np.percentile(t, 95)),
                batch256_us_p50=float(np.percentile(tb, 50)))


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def main():
    cell = sys.argv[1] if len(sys.argv) > 1 else "amazon_distilbert"
    vl, vt, tests, _ = load_cell(cell, 0, False)
    split = "id_test" if "id_test" in tests else "test"
    tl, tt = tests[split]
    z1, zB = tl[:1], tl[:256]
    res = {"host": os.uname().nodename, "threads": 4, "cell": cell,
           "n_fit": int(len(vt))}

    def bench(name, fit_fn, pred_fn_of, state_of, R=5):
        fits = []
        r0 = rss_mb()
        for _ in range(R):
            t0 = time.perf_counter()
            m = fit_fn()
            fits.append(time.perf_counter() - t0)
        pred = pred_fn_of(m)
        buf = pickle.dumps(state_of(m))
        res[name] = dict(fit_seconds_p50=float(np.median(fits)),
                         serialized_bytes=len(buf),
                         peak_rss_delta_mb=max(0.0, rss_mb() - r0),
                         **timeit_inf(pred, z1, zB))
        print(name, res[name], flush=True)

    bench("TS-NLL", lambda: fit_ts_nll(vl, vt),
          lambda m: m.predict_proba, lambda m: float(m.temps[0]))
    bench("TvA-TS", lambda: fit_tva(vl, vt),
          lambda m: m.predict_proba, lambda m: float(m.temps[0]))
    bench("HTS-BCE", lambda: HTS(objective="bce").fit(vl, vt),
          lambda m: m.predict_proba, lambda m: m.params)
    bench("SRTS-BCE", lambda: SRTS(K=3, scaler="fold").fit(vl, vt),
          lambda m: m.predict_proba,
          lambda m: dict(t=m.temps, th=m.thresholds, mu=m.mean, sd=m.std,
                         c=m.coef, b=m.intercept))
    bench("SMART+BCE",
          lambda: train_smart(vl, vt, objective="bce", seed=0)[0],
          lambda m: (lambda z: smart_predict_proba(m, z)),
          lambda m: m.state_dict(), R=3)

    # backbone CPU forward reference (fp32, len512)
    try:
        from transformers import AutoModelForSequenceClassification
        mdl = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=5)
        mdl.eval()
        ids1 = torch.randint(0, 30000, (1, 512))
        ids32 = torch.randint(0, 30000, (32, 512))
        with torch.no_grad():
            for _ in range(5):
                mdl(input_ids=ids1)
            t = []
            for _ in range(50):
                t0 = time.perf_counter()
                mdl(input_ids=ids1)
                t.append(time.perf_counter() - t0)
            tb = []
            for _ in range(10):
                t0 = time.perf_counter()
                mdl(input_ids=ids32)
                tb.append((time.perf_counter() - t0) / 32)
        res["backbone_cpu_fp32"] = dict(
            batch1_ms_p50=float(np.percentile(t, 50) * 1e3),
            batch32_ms_per_sample_p50=float(np.percentile(tb, 50) * 1e3))
    except Exception as e:
        res["backbone_cpu_fp32"] = {"error": str(e)}
    json.dump(res, open(OUT / f"efficiency_{cell}.json", "w"), indent=1)
    print("DONE")


if __name__ == "__main__":
    main()

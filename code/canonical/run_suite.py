#!/usr/bin/env python3
"""Unified calibration-suite driver (canonical engine for all WWW2027 cells).

Per (seed, budget, draw): draw an EXACT-n calibration subset from the cell's
pool, fit every requested method on that subset only, evaluate on the cell's
test splits, and append fully-provenanced JSONL rows.

Cells:
  c100_vit                 reviewed-artifact CIFAR-100/ViT-B/16 cells (5 seeds)
  amazon_distilbert        WILDS Amazon trainer outputs (pool=id_val;
                           tests: id_test, ood_val; ood_test only with
                           --include-confirmation)
  amazon_roberta, yahoo_distilbert   same layout
Method sets (--methods):
  core    NoTS, TS-NLL, TvA-TS, HTS-NLL, HTS-BCE, SMART-SoftECE, SMART+BCE,
          Margin-K3, PWLinear-3, SRTS-BCE
  familyA nested dyadic SRTS K in {1,2,4,8} + deployed K=3 reference
          (+ floor0 diagnostic variants with min_group=0)
  familyB nested polynomial risk maps, coef count {1(=scalar BCE),2,3,4}
Sampling (--sampling): natural (primary) | stratified (sensitivity)
All methods inside a draw share the identical calibration subset; SRTS's
OOF risk score is shared with the continuous risk maps (same discipline as
the earlier matched-capacity protocol).

Usage:
  python run_suite.py --cell c100_vit --seeds 0,1,2,3,4 \
      --budgets 250,625,1250,2500 --draws 20 --sampling natural \
      --methods core --out results/www2027/suite
"""
from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import numpy as np
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import registry as R
from sampler import exact_stratified_subsample, natural_subsample
from srts import SRTS, fit_tva, softmax
from smart import train_smart, smart_predict_proba
from maps import HTS, RiskMap, fit_ts_nll
from srts import MarginK

REPO = Path("__SRTS_ROOT__/calattn-repo")
ART_CELLS = REPO / "www2027/reviewed_artifact/srts_bce_artifact/data/cells"
WEB = REPO / "results/www2027"


# ---------------------------------------------------------------- cells ----
def load_cell(cell: str, seed: int, include_confirmation: bool):
    """Returns (pool_logits, pool_labels, tests: dict[str,(tl,tt)], meta)."""
    if cell == "c100_vit":
        d = ART_CELLS / f"ANCHOR_C100_ViT224_CE_v1_NoTS_s{seed}"
        v = np.load(d / "logits_val.npz")
        t = np.load(d / "logits_test.npz")
        tests = {"test": (t["logits"].astype(np.float64),
                          t["targets"].astype(np.int64))}
        return (v["logits"].astype(np.float64), v["targets"].astype(np.int64),
                tests, {"source": str(d)})
    if cell in ("amazon_distilbert", "amazon_roberta", "yahoo_distilbert"):
        base = {"amazon_distilbert": WEB / "wilds_amazon" / f"distilbert_s{seed}",
                "amazon_roberta": WEB / "wilds_amazon" / f"roberta_s{seed}",
                "yahoo_distilbert": WEB / "yahoo" / f"distilbert_s{seed}"}[cell]
        pool = np.load(base / "logits_id_val.npz")
        tests = {}
        for name, fn in [("id_test", "logits_id_test.npz"),
                         ("ood_val", "logits_val.npz")]:
            if (base / fn).exists():
                z = np.load(base / fn)
                tests[name] = (z["logits"].astype(np.float64),
                               z["targets"].astype(np.int64))
        if include_confirmation and (base / "logits_test.npz").exists():
            z = np.load(base / "logits_test.npz")
            tests["ood_test_CONFIRMATION"] = (
                z["logits"].astype(np.float64), z["targets"].astype(np.int64))
        return (pool["logits"].astype(np.float64),
                pool["targets"].astype(np.int64), tests, {"source": str(base)})
    raise ValueError(cell)


# -------------------------------------------------------------- methods ----
def fit_methods(methodset, vl, vt, seed):
    """Returns dict name -> (predict_fn(test_logits)->probs, fit_report)."""
    out = {}

    def add(name, fn, rep):
        out[name] = (fn, rep)

    if methodset in ("core", "familyA", "familyB"):
        srts = SRTS(K=3, seed=seed, scaler="fold", fold_fallback="train"
                    ).fit(vl, vt)
    if methodset == "core":
        add("NoTS", lambda z: softmax(np.asarray(z, dtype=np.float64)),
            {"note": "identity"})
        ts = fit_ts_nll(vl, vt)
        add("TS-NLL", ts.predict_proba, ts.report)
        tva = fit_tva(vl, vt)
        add("TvA-TS", tva.predict_proba, tva.report["pooled_T"])
        for obj in ("nll", "bce"):
            h = HTS(objective=obj).fit(vl, vt)
            add(f"HTS-{obj.upper()}", h.predict_proba, h.report)
        for obj, nm in (("softece", "SMART-SoftECE"), ("bce", "SMART+BCE")):
            m, rep = train_smart(vl, vt, objective=obj, seed=seed)
            add(nm, (lambda mm: (lambda z: smart_predict_proba(mm, z)))(m), rep)
        mk = MarginK(K=3).fit(vl, vt)
        add("Margin-K3", mk.predict_proba, {"T": mk.temps.tolist()})
        # PWLinear-3 shares SRTS's OOF risk + deployed router score
        q_oof_vl = srts.score(vl)  # deployment-router score on cal split
        pw = RiskMap(kind="pw3").fit(vl, vt, q_oof_vl)
        add("PWLinear-3",
            (lambda p, s: (lambda z: p.predict_proba(z, s.score(z))))(pw, srts),
            pw.report)
        add("SRTS-BCE", srts.predict_proba,
            {k: v for k, v in srts.report.items()})
    elif methodset == "familyA":
        for K in (1, 2, 4, 8):
            mdl = SRTS(K=K, seed=seed, scaler="fold", fold_fallback="train",
                       dyadic=True).fit(vl, vt)
            add(f"SRTS-K{K}", mdl.predict_proba, mdl.report)
            mdl0 = SRTS(K=K, seed=seed, scaler="fold", fold_fallback="train",
                        dyadic=True, min_group=0).fit(vl, vt)
            add(f"SRTS-K{K}-floor0", mdl0.predict_proba, mdl0.report)
        add("SRTS-K3-ref", srts.predict_proba, srts.report)
    elif methodset == "familyB":
        q_vl = srts.score(vl)
        for m, nm in ((1, "Poly1(=scalarBCE)"), (2, "Poly2"), (3, "Poly3"),
                      (4, "Poly4")):
            pm = RiskMap(kind="poly", n_anchors=m).fit(vl, vt, q_vl)
            add(nm, (lambda p, s: (lambda z: p.predict_proba(z, s.score(z))))(
                pm, srts), pm.report)
        add("SRTS-K3-ref", srts.predict_proba, srts.report)
    else:
        raise ValueError(methodset)
    return out


# ------------------------------------------------------------------ run ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--budgets", default="250,625,1250,2500")
    ap.add_argument("--draws", type=int, default=20)
    ap.add_argument("--sampling", default="natural",
                    choices=["natural", "stratified"])
    ap.add_argument("--methods", default="core")
    ap.add_argument("--include-confirmation", action="store_true")
    ap.add_argument("--skip-smece-over", type=int, default=200000)
    ap.add_argument("--out", default=str(WEB / "suite"))
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    budgets = [int(b) for b in args.budgets.split(",")]
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.cell}_{args.methods}_{args.sampling}"
    fout = outdir / f"{tag}.jsonl"
    try:
        import subprocess
        commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short",
                                 "HEAD"], capture_output=True, text=True
                                ).stdout.strip()
    except Exception:
        commit = "unknown"

    done = set()
    if fout.exists():  # resume support
        for line in fout.open():
            try:
                r = json.loads(line)
                done.add((r["seed"], r["budget"], r["draw"], r["method"],
                          r["test_split"]))
            except Exception:
                pass

    with fout.open("a") as fh:
        for s in seeds:
            vl_pool, vt_pool, tests, meta = load_cell(
                args.cell, s, args.include_confirmation)
            t_start = time.time()
            for b in budgets:
                # R=1 at full pool: repeating the same full set is not a
                # new draw (WEB_PROTOCOL §4)
                n_draws_b = 1 if b >= len(vt_pool) else args.draws
                for r_ in range(n_draws_b):
                    rng = np.random.default_rng(
                        77_000_000 + 1_000_000 * s + 1000 * b + r_)
                    if args.sampling == "natural":
                        idx, rec = natural_subsample(vt_pool, b, rng)
                    else:
                        idx, rec = exact_stratified_subsample(vt_pool, b, rng)
                    vl, vt = vl_pool[idx], vt_pool[idx]
                    fitted = fit_methods(args.methods, vl, vt, s)
                    for name, (pred, rep) in fitted.items():
                        for split, (tl, tt) in tests.items():
                            if (s, b, r_, name, split) in done:
                                continue
                            P = pred(tl)
                            with_sm = len(tt) <= args.skip_smece_over
                            m = R.all_metrics(P, tt, with_smece=with_sm)
                            m.update(R.ops_metrics(P, tt))
                            m.update(R.risk_at_coverage(P, tt))
                            row = dict(cell=args.cell, seed=s, budget=b,
                                       draw=r_, sampling=args.sampling,
                                       method=name, test_split=split,
                                       argmax_changes=int(
                                           (P.argmax(1) != tl.argmax(1)).sum()),
                                       sampler=rec, commit=commit,
                                       methodset=args.methods,
                                       fit_report=rep, **m)
                            fh.write(json.dumps(row, default=float) + "\n")
                    fh.flush()
                print(f"[{args.cell}] seed {s} budget {b}: {args.draws} draws "
                      f"done ({time.time() - t_start:.0f}s)", flush=True)
    print("DONE", fout)


if __name__ == "__main__":
    main()

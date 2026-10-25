#!/usr/bin/env python3
"""F2 post-hoc controlled analysis: 2x2 score-construction control.

Plan: www2027/docs/POSTHOC_PLAN_2026-10-16.md (P2), written before running.

Variants fit inside one draw on the same calibration subset:
  D-OOF  deployed SRTS-BCE (K=3): groups + temperatures from cross-fitted
         (out-of-fold) router scores               == saved "SRTS-BCE" rows
  D-IN   same discrete map, groups + temperatures from the full-refit
         router's scores on its own training rows
  C-IN   PWLinear-3 fitted on the full-refit router's in-sample scores
                                                    == saved "PWLinear-3" rows
  C-OOF  PWLinear-3 fitted on the same OOF scores as D-OOF
Shared per draw: subset, StratifiedKFold(seed) folds, fold-train
standardization for OOF scores, one full-refit router (full-split
standardization) used for IN scores and to route every test sample of every
variant. No test label is used by any fit or choice.

Modes:
  suite  --cell c100_vit|amazon_distilbert --split test|id_test
         budgets 250,625,1250,2500; 20 natural draws (1 at full pool)
  shift  C100-C, 95 corruption x severity cells, fit once on the full clean
         split (n=2500), protocol of run_shift_suite.py
Rows: results/www2027/posthoc_f2/<tag>.jsonl
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import registry as R
from maps import RiskMap
from run_suite import load_cell
from sampler import natural_subsample
from srts import SRTS, fit_bce_temperature, softmax

REPO = Path("__SRTS_ROOT__/calattn-repo")
OUT = REPO / "results/www2027/posthoc_f2"
SOTA = Path("__SRTS_ROOT__/anchor-sota/results")
VARIANTS = ("D-OOF", "D-IN", "C-OOF", "C-IN")


class SRTSX(SRTS):
    """SRTS that keeps its cross-fitted scores; fitting is unchanged."""

    def _fit_router_oof(self, S, y_risk):
        out = super()._fit_router_oof(S, y_risk)
        self.q_oof_ = np.asarray(out[0], dtype=np.float64).copy()
        return out


def fit_variants(vl, vt, seed):
    d = SRTSX(K=3, seed=seed, scaler="fold", fold_fallback="train").fit(vl, vt)
    q_oof = d.q_oof_
    q_in = d.score(vl)
    # D-IN: same discrete map, in-sample router scores
    thr = np.quantile(q_in, [1 / 3, 2 / 3])
    g = np.clip(np.digitize(q_in, thr, right=False), 0, 2)
    temps, glog = [], []
    for k in range(3):
        m = g == k
        if m.sum() >= d.min_group:
            T, _ = fit_bce_temperature(vl[m], vt[m])
            fb = False
        else:
            T, fb = d.pooled_T, True
        temps.append(T)
        glog.append({"g": k, "n": int(m.sum()), "T": float(T), "fallback": fb})
    temps = np.asarray(temps)

    def din(z):
        z = np.asarray(z, dtype=np.float64)
        r = np.clip(np.digitize(d.score(z), thr, right=False), 0, 2)
        return softmax(z / temps[r][:, None])

    c_in = RiskMap(kind="pw3").fit(vl, vt, q_in)
    c_oof = RiskMap(kind="pw3").fit(vl, vt, q_oof)
    fits = {
        "D-OOF": (d.predict_proba,
                  {"thresholds": d.thresholds.tolist(),
                   "temps": d.temps.tolist(),
                   "fallback_rate": d.report.get("fallback_rate")}),
        "D-IN": (din, {"thresholds": thr.tolist(), "temps": temps.tolist(),
                       "groups": glog}),
        "C-IN": ((lambda p: (lambda z: p.predict_proba(z, d.score(z))))(c_in),
                 {"anchors": c_in.anchors.tolist(),
                  "T_anchor": np.logaddexp(0, c_in.theta).tolist(),
                  **c_in.report}),
        "C-OOF": ((lambda p: (lambda z: p.predict_proba(z, d.score(z))))(c_oof),
                  {"anchors": c_oof.anchors.tolist(),
                   "T_anchor": np.logaddexp(0, c_oof.theta).tolist(),
                   **c_oof.report}),
    }
    diag = {"spearman_oof_in": float(
        np.corrcoef(np.argsort(np.argsort(q_oof)),
                    np.argsort(np.argsort(q_in)))[0, 1]),
            "group_agree_oof_in": float(np.mean(
                np.clip(np.digitize(q_oof, d.thresholds), 0, 2) == g))}
    return fits, diag


def metrics(P, t):
    return {"ece15": R.ece(P, t), "adaece15": R.adaece(P, t),
            "nll": R.nll(P, t), "topbce": R.topbce(P, t),
            "brier": R.brier(P, t), "top_brier": R.top_brier(P, t)}


def commit():
    import subprocess
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


def run_suite(cell, split, seeds, budgets, draws, draw_list=None):
    OUT.mkdir(parents=True, exist_ok=True)
    fout = OUT / f"{cell}_{split}.jsonl"
    done = set()
    if fout.exists():
        for l in fout.open():
            r = json.loads(l)
            done.add((r["seed"], r["budget"], r["draw"]))
    c = commit()
    with fout.open("a") as fh:
        for s in seeds:
            vl_pool, vt_pool, tests, _ = load_cell(cell, s, False)
            tl, tt = tests[split]
            for b in budgets:
                nd = 1 if b >= len(vt_pool) else draws
                for r_ in (draw_list if draw_list is not None else range(nd)):
                    if r_ >= nd or (s, b, r_) in done:
                        continue
                    t0 = time.time()
                    rng = np.random.default_rng(
                        77_000_000 + 1_000_000 * s + 1000 * b + r_)
                    idx, rec = natural_subsample(vt_pool, b, rng)
                    fits, diag = fit_variants(vl_pool[idx], vt_pool[idx], s)
                    for v in VARIANTS:
                        pred, rep = fits[v]
                        P = pred(tl)
                        row = dict(cell=cell, test_split=split, seed=s,
                                   budget=b, draw=r_, sampling="natural",
                                   variant=v, sampler=rec, commit=c,
                                   argmax_changes=int(
                                       (P.argmax(1) != tl.argmax(1)).sum()),
                                   fit=rep, diag=diag, **metrics(P, tt))
                        fh.write(json.dumps(row, default=float) + "\n")
                    fh.flush()
                    print(f"[{cell}/{split}] s{s} n{b} d{r_} "
                          f"{time.time() - t0:.1f}s", flush=True)
    return fout


def run_shift(seeds):
    OUT.mkdir(parents=True, exist_ok=True)
    fout = OUT / "c100c_vit_shift.jsonl"
    done = set()
    if fout.exists():
        for l in fout.open():
            r = json.loads(l)
            done.add((r["seed"], r["corruption"], r["severity"]))
    c = commit()
    pat_s = "ANCHOR_C100C_ViT224_CE_v1_NoTS_s{}_(.+)_s(\\d)$"
    with fout.open("a") as fh:
        for s in seeds:
            vl, vt, _, _ = load_cell("c100_vit", s, False)
            fits, diag = fit_variants(vl, vt, s)
            pat = re.compile(pat_s.format(s))
            cells = sorted((m.group(1), int(m.group(2)), d)
                           for d in SOTA.iterdir()
                           for m in [pat.match(d.name)]
                           if m and (d / "logits_test.npz").exists())
            t0 = time.time()
            for corr, sev, d in cells:
                if (s, corr, sev) in done:
                    continue
                z = np.load(d / "logits_test.npz")
                cl = z["logits"].astype(np.float64)
                ct = z["targets"].astype(np.int64)
                for v in VARIANTS:
                    pred, rep = fits[v]
                    P = pred(cl)
                    fh.write(json.dumps(dict(
                        cell="c100c_vit", seed=s, corruption=corr,
                        severity=sev, variant=v, budget=int(len(vt)),
                        commit=c, fit=rep, diag=diag,
                        argmax_changes=int((P.argmax(1) != cl.argmax(1)).sum()),
                        **metrics(P, ct)), default=float) + "\n")
                fh.flush()
            print(f"[shift] seed {s}: {len(cells)} cells "
                  f"{time.time() - t0:.0f}s", flush=True)
    return fout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["suite", "shift"])
    ap.add_argument("--cell")
    ap.add_argument("--split")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--budgets", default="250,625,1250,2500")
    ap.add_argument("--draws", type=int, default=20)
    ap.add_argument("--draw-list", default=None,
                    help="pilot only: comma list of draw indices")
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]
    if a.mode == "suite":
        dl = ([int(x) for x in a.draw_list.split(",")] if a.draw_list
              else None)
        f = run_suite(a.cell, a.split, seeds,
                      [int(x) for x in a.budgets.split(",")], a.draws, dl)
    else:
        f = run_shift(seeds)
    print("DONE", f)


if __name__ == "__main__":
    main()

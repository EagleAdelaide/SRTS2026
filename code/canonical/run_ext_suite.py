#!/usr/bin/env python3
"""Extended / nearest-neighbour baseline suite (WP4 tier) on anchor budgets.

Methods: PTS (native MSE) + PTS-BCE, ETS (Mix-n-Match), Dirichlet-ODIR,
MetaCal (native coverage semantics; comparable view flagged), SAG (official
code; REAL frozen features), AdaTS (faithful CondVAE port; REAL features),
LogitMLP-T (honest rename of the former "SATS-Logit" generic head:
logits -> 128 -> softplus temperature).

Draws share the main-suite RNG formula (77e6 + 1e6*seed + 1000*b + r).
Budgets default {250, 2500} (anchor tier). Features for c100_vit come from
calattn-repo-sota NoTS cells (verified byte-identical logits).
Rows appended to the same JSONL schema as run_suite.
"""
import argparse
import json
import os
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import numpy as np
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import registry as R
from sampler import natural_subsample, exact_stratified_subsample
from srts import softmax
from run_suite import load_cell, REPO, WEB
from baselines_ext import PTS, ETS, DirichletODIR, MetaCal, SAG, AdaTS

SOTA = Path("__SRTS_ROOT__/anchor-sota/results")


def load_features(cell, seed):
    if cell == "c100_vit":
        d = SOTA / f"ANCHOR_C100_ViT224_CE_v1_NoTS_s{seed}"
        fv = np.load(d / "features_val.npz")["features"].astype(np.float64)
        ft = np.load(d / "features_test.npz")["features"].astype(np.float64)
        return fv, {"test": ft}
    base = {"amazon_distilbert": WEB / "wilds_amazon" / f"distilbert_s{seed}",
            "amazon_roberta": WEB / "wilds_amazon" / f"roberta_s{seed}",
            "yahoo_distilbert": WEB / "yahoo" / f"distilbert_s{seed}"}[cell]
    fv = np.load(base / "logits_id_val.npz")["features"].astype(np.float64)
    out = {}
    for name, fn in [("id_test", "logits_id_test.npz"),
                     ("ood_val", "logits_val.npz")]:
        p = base / fn
        if p.exists():
            z = np.load(p)
            if "features" in z:
                out[name] = z["features"].astype(np.float64)
    return fv, out


class LogitMLPT:
    """Honest rename of the former 'SATS-Logit' row: generic MLP
    logits->h128->softplus T, trained on val NLL (as in the earlier repository's
    calibrate_sats.py). NOT the official AdaTS. ~ (C+2)*128+1 params."""

    def __init__(self, seed=0, hidden=128, iters=2000, lr=1e-3):
        self.seed, self.hidden, self.iters, self.lr = seed, hidden, iters, lr
        self.report = {}

    def fit(self, vl, vt):
        import torch
        import torch.nn as nn
        torch.manual_seed(self.seed)
        z = torch.tensor(np.asarray(vl, dtype=np.float32))
        y = torch.tensor(np.asarray(vt, dtype=np.int64))
        self.net = nn.Sequential(nn.Linear(z.shape[1], self.hidden),
                                 nn.ReLU(), nn.Linear(self.hidden, 1),
                                 nn.Softplus())
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        t0 = time.time()
        for _ in range(self.iters):
            T = self.net(z).squeeze(-1) + 1e-6
            loss = nn.functional.cross_entropy(z / T.unsqueeze(-1), y)
            opt.zero_grad(); loss.backward(); opt.step()
        self.report = {"n_params": sum(p.numel()
                                       for p in self.net.parameters()),
                       "iters": self.iters, "seconds": time.time() - t0}
        return self

    def predict_proba(self, logits):
        import torch
        with torch.no_grad():
            z = torch.tensor(np.asarray(logits, dtype=np.float32))
            T = self.net(z).squeeze(-1) + 1e-6
            return torch.softmax(z / T.unsqueeze(-1), 1).double().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--budgets", default="250,2500")
    ap.add_argument("--draws", type=int, default=20)
    ap.add_argument("--sampling", default="natural")
    ap.add_argument("--only", default="", help="comma list to restrict")
    ap.add_argument("--out", default=str(WEB / "suite_ext"))
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    budgets = [int(b) for b in args.budgets.split(",")]
    only = set(args.only.split(",")) if args.only else None
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    fout = outdir / f"{args.cell}_ext_{args.sampling}.jsonl"
    done = set()
    if fout.exists():
        for line in fout.open():
            try:
                r = json.loads(line)
                done.add((r["seed"], r["budget"], r["draw"], r["method"],
                          r["test_split"]))
            except Exception:
                pass
    with fout.open("a") as fh:
        for s in seeds:
            vl_pool, vt_pool, tests, _ = load_cell(args.cell, s, False)
            feat_pool, feat_tests = load_features(args.cell, s)
            for b in budgets:
                nd = 1 if b >= len(vt_pool) else args.draws
                for r_ in range(nd):
                    rng = np.random.default_rng(
                        77_000_000 + 1_000_000 * s + 1000 * b + r_)
                    if args.sampling == "natural":
                        idx, rec = natural_subsample(vt_pool, b, rng)
                    else:
                        idx, rec = exact_stratified_subsample(vt_pool, b, rng)
                    vl, vt, fv = vl_pool[idx], vt_pool[idx], feat_pool[idx]
                    methods = {}
                    def want(n):
                        return only is None or n in only
                    if want("PTS"):
                        methods["PTS"] = (PTS(objective="mse", seed=s
                                              ).fit(vl, vt), None)
                    if want("PTS-BCE"):
                        methods["PTS-BCE"] = (PTS(objective="bce", seed=s
                                                  ).fit(vl, vt), None)
                    if want("ETS"):
                        methods["ETS"] = (ETS().fit(vl, vt), None)
                    if want("DirODIR"):
                        methods["DirODIR"] = (DirichletODIR(seed=s
                                                            ).fit(vl, vt), None)
                    if want("MetaCal"):
                        methods["MetaCal"] = (MetaCal(seed=s).fit(vl, vt), None)
                    if want("LogitMLP-T"):
                        methods["LogitMLP-T"] = (LogitMLPT(seed=s
                                                           ).fit(vl, vt), None)
                    if want("SAG"):
                        methods["SAG"] = (SAG(seed=s).fit(vl, vt, fv), "feat")
                    if want("AdaTS"):
                        methods["AdaTS"] = (AdaTS(seed=s).fit(vl, vt, fv),
                                            "feat")
                    for name, (m, needs) in methods.items():
                        for split, (tl, tt) in tests.items():
                            if (s, b, r_, name, split) in done:
                                continue
                            if needs == "feat":
                                if split not in feat_tests:
                                    continue
                                P = m.predict_proba(tl, feat_tests[split])
                            else:
                                P = m.predict_proba(tl)
                            mm = R.all_metrics(P, tt,
                                               with_smece=len(tt) <= 200000)
                            mm.update(R.ops_metrics(P, tt))
                            row = dict(cell=args.cell, seed=s, budget=b,
                                       draw=r_, sampling=args.sampling,
                                       method=name, test_split=split,
                                       methodset="ext",
                                       argmax_changes=int((P.argmax(1)
                                                           != tl.argmax(1)).sum()),
                                       sampler=rec,
                                       fit_report=getattr(m, "report", {}),
                                       **mm)
                            if name == "MetaCal":
                                am = m.accept_mask(tl)
                                row["metacal_coverage"] = float(am.mean())
                                row["metacal_acc_err"] = float(
                                    (P.argmax(1) != tt)[am].mean()) \
                                    if am.any() else None
                            fh.write(json.dumps(row, default=float) + "\n")
                    fh.flush()
                print(f"[{args.cell}] s{s} b{b} ext done", flush=True)
    print("DONE", fout)


if __name__ == "__main__":
    main()

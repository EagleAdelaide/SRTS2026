#!/usr/bin/env python3
"""WP5: OOF-fit vs full-refit deployment diagnostics + fold-count sensitivity.

Per (cell, seed, budget, draw):
  * fit SRTS with F in {3,5,10} on the same exact-n draw;
  * on the calibration subset: Spearman(q_OOF, q_fullrefit), group-assignment
    confusion (OOF tertile id vs deployed-router tertile id), occupancy,
    per-group error counts and temperatures (OOF-group vs deploy-group);
  * on a DEVELOPMENT probe (first half of the cell's dev test split —
    labelled development, never confirmation): fold-router ensemble score vs
    full-router score drift (Spearman, assignment agreement) and ECE of two
    named diagnostics: "no-crossfit" (router fit once on the full subset,
    in-sample scores define thresholds) and "fold-ensemble" (deploy score =
    mean of fold routers);
  * all methods share draws with the main suite RNG formula.
Output: JSON rows -> results/www2027/oof_diag/<cell>.json
"""
import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import registry as R
from sampler import natural_subsample
from srts import SRTS, six_signals, softmax, fit_bce_temperature
from run_suite import load_cell

OUT = Path("__SRTS_ROOT__/calattn-repo/results/www2027/oof_diag")
OUT.mkdir(parents=True, exist_ok=True)


def fit_with_fold_models(vl, vt, seed, F):
    """SRTS fit that also returns the fold routers for ensemble scoring."""
    S = six_signals(vl)
    y = (vl.argmax(1) != vt).astype(np.int64)
    folds = []
    q_oof = np.zeros(len(y))
    minority = min(y.sum(), len(y) - y.sum())
    F_eff = min(F, max(2, minority)) if minority >= 2 else 0
    if F_eff >= 2:
        skf = StratifiedKFold(F_eff, shuffle=True, random_state=seed)
        for tr, te in skf.split(S, y):
            mu = S[tr].mean(0)
            sd = np.where(S[tr].std(0) < 1e-12, 1.0, S[tr].std(0))
            if len(set(y[tr].tolist())) < 2:
                q_oof[te] = y[tr].mean()
                continue
            clf = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
            clf.fit((S[tr] - mu) / sd, y[tr])
            q_oof[te] = clf.predict_proba((S[te] - mu) / sd)[:, 1]
            folds.append((mu, sd, clf))
    mu_f = S.mean(0)
    sd_f = np.where(S.std(0) < 1e-12, 1.0, S.std(0))
    clf_full = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
    if minority >= 1:
        clf_full.fit((S - mu_f) / sd_f, y)
    thr = np.quantile(q_oof, [1 / 3, 2 / 3])
    g_oof = np.clip(np.digitize(q_oof, thr), 0, 2)
    pooled_T, _ = fit_bce_temperature(vl, vt)
    temps, sizes, errs = [], [], []
    for g in range(3):
        m = g_oof == g
        sizes.append(int(m.sum()))
        errs.append(int(y[m].sum()))
        temps.append(fit_bce_temperature(vl[m], vt[m])[0]
                     if m.sum() >= 50 else pooled_T)
    return dict(S=S, y=y, q_oof=q_oof, folds=folds, clf_full=clf_full,
                mu=mu_f, sd=sd_f, thr=thr, g_oof=g_oof,
                temps=np.array(temps), pooled_T=pooled_T,
                sizes=sizes, errs=errs, F_eff=F_eff)


def q_full(fit, logits):
    S = (six_signals(logits) - fit["mu"]) / fit["sd"]
    return fit["clf_full"].predict_proba(S)[:, 1]


def q_fold_ens(fit, logits):
    S6 = six_signals(logits)
    qs = [clf.predict_proba((S6 - mu) / sd)[:, 1]
          for mu, sd, clf in fit["folds"]]
    return np.mean(qs, axis=0) if qs else np.zeros(len(logits))


def groups_from(q, thr):
    return np.clip(np.digitize(q, thr), 0, 2)


def ece_with(fit, logits, targets, q):
    g = groups_from(q, fit["thr"])
    T = fit["temps"][g][:, None]
    return R.ece(softmax(logits / T), targets)


def main():
    cell = sys.argv[1] if len(sys.argv) > 1 else "c100_vit"
    seeds = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2
                              else ["0", "1", "2", "3", "4"])]
    budgets = [250, 2500]
    draws = 20
    rows = []
    for s in seeds:
        vl_pool, vt_pool, tests, _ = load_cell(cell, s, False)
        split = "test" if "test" in tests else "ood_val"
        tl, tt = tests[split]
        half = len(tt) // 2
        pl, pt = tl[:half], tt[:half]  # development probe half
        for b in budgets:
            nd = 1 if b >= len(vt_pool) else draws
            for r_ in range(nd):
                rng = np.random.default_rng(
                    77_000_000 + 1_000_000 * s + 1000 * b + r_)
                idx, _ = natural_subsample(vt_pool, b, rng)
                vl, vt = vl_pool[idx], vt_pool[idx]
                for F in (3, 5, 10):
                    fit = fit_with_fold_models(vl, vt, s, F)
                    qf_cal = q_full(fit, vl)
                    g_dep = groups_from(qf_cal, fit["thr"])
                    conf = np.zeros((3, 3), dtype=int)
                    for a_, b_ in zip(fit["g_oof"], g_dep):
                        conf[a_, b_] += 1
                    # probe drift + diagnostics
                    qp_full = q_full(fit, pl)
                    qp_ens = q_fold_ens(fit, pl)
                    g_full_p = groups_from(qp_full, fit["thr"])
                    g_ens_p = groups_from(qp_ens, fit["thr"])
                    rows.append(dict(
                        cell=cell, seed=s, budget=b, draw=r_, F=F,
                        F_eff=fit["F_eff"],
                        spearman_cal=float(spearmanr(
                            fit["q_oof"], qf_cal).correlation),
                        assign_agree_cal=float(
                            (fit["g_oof"] == g_dep).mean()),
                        confusion=conf.tolist(),
                        occupancy_dep=np.bincount(
                            g_dep, minlength=3).tolist(),
                        group_sizes_oof=fit["sizes"],
                        group_errs_oof=fit["errs"],
                        temps=fit["temps"].tolist(),
                        pooled_T=fit["pooled_T"],
                        probe_spearman_full_vs_ens=float(spearmanr(
                            qp_full, qp_ens).correlation),
                        probe_assign_agree=float(
                            (g_full_p == g_ens_p).mean()),
                        probe_ece_deploy_full=ece_with(fit, pl, pt, qp_full),
                        probe_ece_fold_ens=ece_with(fit, pl, pt, qp_ens),
                    ))
            print(f"seed {s} budget {b} done", flush=True)
    out = OUT / f"{cell}.json"
    json.dump(rows, open(out, "w"), indent=0)
    print("DONE", out, len(rows), "rows")


if __name__ == "__main__":
    main()

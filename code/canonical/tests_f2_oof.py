#!/usr/bin/env python3
"""Verify the OOF property used by the F2 control (plan P2).

For sampled draws, rebuild the cross-fitting independently of srts.py:
StratifiedKFold(seed) on the correctness labels, fold-train standardization,
one L2 logistic router per fold, prediction on the held-out fold only. Check
(1) every calibration index is held out exactly once and never appears in
its own training fold, (2) the rebuilt scores equal the q_oof used by the F2
variants (D-OOF, C-OOF) to 1e-12, (3) the in-sample scores used by D-IN and
C-IN equal the full-refit router's scores. Writes the fold IDs of the checked
draws to results/www2027/posthoc_f2/oof_fold_ids_check.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_f2_score_control import SRTSX
from run_suite import load_cell
from sampler import natural_subsample
from srts import six_signals

OUT = Path("__SRTS_ROOT__/calattn-repo/results/www2027/posthoc_f2")
CASES = [("c100_vit", 0, 250, 0), ("c100_vit", 3, 250, 7), ("c100_vit", 1, 2500, 0),
         ("amazon_distilbert", 0, 250, 0), ("amazon_distilbert", 2, 250, 11),
         ("amazon_distilbert", 4, 2500, 5)]


def main():
    rec, ok = {}, True
    for cell, s, b, d in CASES:
        vlp, vtp, _, _ = load_cell(cell, s, False)
        rng = np.random.default_rng(77_000_000 + 1_000_000 * s + 1000 * b + d)
        idx, _ = natural_subsample(vtp, b, rng)
        vl, vt = vlp[idx], vtp[idx]
        m = SRTSX(K=3, seed=s, scaler="fold", fold_fallback="train").fit(vl, vt)
        S = six_signals(vl)
        y = (vl.argmax(1) != vt).astype(np.int64)
        minority = int(min(y.sum(), len(y) - y.sum()))
        F = min(5, max(2, minority))
        fold = np.full(len(y), -1)
        q = np.zeros(len(y))
        for k, (tr, te) in enumerate(StratifiedKFold(F, shuffle=True, random_state=s).split(S, y)):
            assert not set(tr) & set(te)
            fold[te] = k
            mu = S[tr].mean(0); sd = np.where(S[tr].std(0) < 1e-12, 1.0, S[tr].std(0))
            clf = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                                     max_iter=1000, random_state=s)
            clf.fit((S[tr] - mu) / sd, y[tr])
            q[te] = clf.predict_proba((S[te] - mu) / sd)[:, 1]
        held_once = bool((fold >= 0).all())
        d_oof = float(np.abs(q - m.q_oof_).max())
        mu_f = S.mean(0); sd_f = np.where(S.std(0) < 1e-12, 1.0, S.std(0))
        clf_f = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                                   max_iter=1000, random_state=s).fit((S - mu_f) / sd_f, y)
        d_in = float(np.abs(clf_f.predict_proba((S - mu_f) / sd_f)[:, 1] - m.score(vl)).max())
        good = held_once and d_oof < 1e-12 and d_in < 1e-12
        ok &= good
        key = f"{cell}/seed{s}/n{b}/draw{d}"
        rec[key] = {"folds": int(F), "held_out_once": held_once,
                    "max_abs_diff_oof": d_oof, "max_abs_diff_in": d_in,
                    "calib_sample_ids": [int(i) for i in idx],
                    "fold_id": [int(f) for f in fold]}
        print(f"{key}: folds={F} held_once={held_once} |dOOF|={d_oof:.1e} |dIN|={d_in:.1e} {'OK' if good else 'FAIL'}")
    (OUT / "oof_fold_ids_check.json").write_text(json.dumps(rec))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

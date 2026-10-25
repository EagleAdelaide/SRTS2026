#!/usr/bin/env python3
"""AI-07 / WP7 §4: held-out gradient-decomposition diagnostic.

Per (cell, seed), full calibration pool split into halves A/B
(deterministic RNG): on A fit the pooled top-label-BCE temperature T*_A
and the SRTS router (K=3). On B (held out from all fitting):
  d_i = analytic dBCE_i/dT at T*_A  (gradient of the sample loss)
  groups g_i from A's router.
Report  B_hat = sum_g pi_g (mean_g d)^2,  sigma2_hat = sum_g pi_g var_g d,
ratio vs the finite-sample floor (K-1)/n for n in {250, 2500}, plus
per-group correctness residual after scalar calibration
(mean confidence - accuracy per risk group at T*_A) — the direct
group-residual / conditional-reliability evidence the no-benefit wording
requires. Descriptive mechanism evidence; never a win/loss label.
"""
import json
import os
import sys
from pathlib import Path

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "4")
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_suite import load_cell
from srts import SRTS, softmax, fit_bce_temperature

OUT = Path("__SRTS_ROOT__/calattn-repo/results/www2027")


def grad_T(logits, targets, T):
    """Analytic d/dT of per-sample top-label BCE at temperature T."""
    z = np.asarray(logits, dtype=np.float64)
    P = softmax(z / T)
    c = np.clip(P.max(1), 1e-12, 1 - 1e-12)
    b = (z.argmax(1) == np.asarray(targets)).astype(np.float64)
    # dc/dT: c = softmax(z/T)_k at fixed k = argmax; d/dT (z/T) = -z/T^2
    # dc/dT = c * ( -(z_k - sum_j P_j z_j) / T^2 )
    zk = z[np.arange(len(z)), z.argmax(1)]
    zbar = (P * z).sum(1)
    dc = c * (-(zk - zbar) / T ** 2)
    return (c - b) / (c * (1 - c)) * dc


def main():
    cells = sys.argv[1:] or ["c100_vit", "yahoo_distilbert"]
    res = {}
    for cell in cells:
        seeds = range(5) if "c100" in cell or "amazon" in cell else range(3)
        rows = []
        for s in seeds:
            try:
                vl, vt, _, _ = load_cell(cell, s, False)
            except Exception as e:
                print(cell, s, "skip:", e)
                continue
            rng = np.random.default_rng(555_000 + s)
            perm = rng.permutation(len(vt))
            A, Bh = perm[: len(vt) // 2], perm[len(vt) // 2:]
            T_A, _ = fit_bce_temperature(vl[A], vt[A])
            srts = SRTS(K=3, seed=s, scaler="fold",
                        fold_fallback="train").fit(vl[A], vt[A])
            g = srts.route(vl[Bh])
            d = grad_T(vl[Bh], vt[Bh], T_A)
            pis, mus, vars_, resid = [], [], [], []
            P_B = softmax(vl[Bh] / T_A)
            conf = P_B.max(1)
            corr = (vl[Bh].argmax(1) == vt[Bh]).astype(float)
            for k in range(3):
                m = g == k
                pis.append(m.mean())
                mus.append(d[m].mean() if m.any() else 0.0)
                vars_.append(d[m].var() if m.any() else 0.0)
                resid.append(float(conf[m].mean() - corr[m].mean())
                             if m.any() else 0.0)
            pis = np.array(pis); mus = np.array(mus)
            B_hat = float((pis * mus ** 2).sum())
            s2 = float((pis * np.array(vars_)).sum())
            rows.append(dict(seed=s, T_A=T_A, B_hat=B_hat, sigma2_hat=s2,
                             ratio=B_hat / s2 if s2 > 0 else None,
                             floor_n250=2 / 250, floor_n2500=2 / 2500,
                             group_mass=pis.tolist(),
                             group_grad_mean=mus.tolist(),
                             group_conf_minus_acc=resid))
            print(f"{cell} s{s}: B/sigma2={B_hat/s2:.4f} "
                  f"(floors .008/.0008) resid={['%+.3f'%r for r in resid]}")
        res[cell] = rows
    json.dump(res, open(OUT / "bsigma_diag.json", "w"), indent=1)
    print("DONE")


if __name__ == "__main__":
    main()

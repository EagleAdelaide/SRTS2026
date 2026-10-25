"""WP0 unit + substantive checks for the canonical package.

Synthetic arrays are used ONLY for software correctness, never as paper
evidence. Run: python tests_unit.py
"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import registry as R
from sampler import exact_stratified_subsample, natural_subsample, legacy_quota_subsample
from srts import SRTS, fit_tva, fit_bce_temperature, six_signals, softmax
from smart import train_smart, smart_predict_proba

rng = np.random.default_rng(0)
FAIL = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        FAIL.append(name)


# ---------- synthetic data ----------
N, C = 3000, 10
z = rng.normal(0, 2, (N, C))
y = np.array([rng.choice(C, p=softmax(zz[None])[0]) for zz in z])

# 1. ECE basics: perfectly calibrated binary-ish check
p_cal = np.full((20000, 2), 0.5)
y_cal = rng.integers(0, 2, 20000)
check("ece≈0 for conf=.5 coin", R.ece(p_cal, y_cal) < 1.5,
      f"={R.ece(p_cal, y_cal):.3f}pp")

# 2. CW-ECE OvR differs from legacy predclass metric on skewed probs
probs = softmax(z)
a, b = R.cw_ece_ovr(probs, y), R.predclass_top_ece(probs, y)
check("cw_ece_ovr != predclass_top_ece (distinguishable)", abs(a - b) > 1e-3,
      f"ovr={a:.4f} legacy={b:.4f}")

# 3. official smECE present and sane
sm = R.smece(probs, y)
check("relplot smECE finite", np.isfinite(sm), f"={sm:.4f}pp")

# 4. exact sampler: exact n, no dup, in-pool; legacy deficit reproduced
counts = rng.integers(12, 44, 100)  # uneven classes like the real val split
labels = np.repeat(np.arange(100), counts)[:2500]
labels = np.concatenate([labels, rng.integers(0, 100, max(0, 2500 - len(labels)))])[:2500]
for n in (250, 625, 1250):
    idx, rec = exact_stratified_subsample(labels, n, np.random.default_rng(1))
    check(f"exact sampler n={n}", len(idx) == n == rec["actual_n"])
leg = [len(legacy_quota_subsample(labels, 250, np.random.default_rng(s)))
       for s in range(5)]
check("legacy sampler deficit reproduced (<250)", all(l < 250 for l in leg),
      f"actual={leg}")
idx_nat, _ = natural_subsample(labels, 250, np.random.default_rng(2))
check("natural sampler exact n", len(idx_nat) == 250)

# 5. K=1 == TvA exactly; argmax invariance
vl, vt = z[:2000], y[:2000]
tl, tt = z[2000:], y[2000:]
tva = fit_tva(vl, vt)
srts1 = SRTS(K=1).fit(vl, vt)
check("K=1 == TvA (temps equal)", abs(tva.temps[0] - srts1.temps[0]) < 1e-10)
srts3 = SRTS(K=3, scaler="full", fold_fallback="pool").fit(vl, vt)
P3 = srts3.predict_proba(tl)
check("argmax invariance", (P3.argmax(1) == tl.argmax(1)).all())

# 6. accounting: 10 optimized coefficients at K=3
acc = srts3.accounting()
check("10 optimized coefficients", acc["optimized_coefficients"] == 10, str(acc))

# 7. fold scaler variant runs and differs only via preprocessing
srts3f = SRTS(K=3, scaler="fold", fold_fallback="train", seed=0).fit(vl, vt)
check("fold-scaler fit report present",
      srts3f.report["router"]["folds_used"] >= 2, str(srts3f.report["router"]))

# 8. rare-error fallback: fit with 1 error only
vl_r = rng.normal(0, 1, (300, C)) + 8 * np.eye(C)[rng.integers(0, C, 300)]
vt_r = vl_r.argmax(1).copy()
vt_r[0] = (vt_r[0] + 1) % C  # exactly one error
m = SRTS(K=3, scaler="fold", fold_fallback="train").fit(vl_r, vt_r)
check("rare-error fit survives, no crash",
      np.isfinite(m.predict_proba(vl_r)).all(),
      f"folds_used={m.report['router'].get('folds_used')}")
vt_r2 = vl_r.argmax(1)  # zero errors
m0 = SRTS(K=3).fit(vl_r, vt_r2)
check("zero-error degenerate OK", np.isfinite(m0.predict_proba(vl_r)).all())

# 9. BCE gradient identity (AI-11): dl/dc = (c-b)/(c(1-c)); numeric check
c0 = 0.7
for bb in (0.0, 1.0):
    h = 1e-6
    f = lambda c: -(bb * np.log(c) + (1 - bb) * np.log(1 - c))
    num = (f(c0 + h) - f(c0 - h)) / (2 * h)
    ana = (c0 - bb) / (c0 * (1 - c0))
    check(f"BCE grad identity b={bb}", abs(num - ana) < 1e-5,
          f"num={num:.6f} ana={ana:.6f}")

# 10. SMART fixed pairing: loaded state reproduces best recorded loss
model, rep = train_smart(vl, vt, objective="bce", n_iter=400, patience=100, seed=0)
P = smart_predict_proba(model, vl)
bce_now = R.topbce(P, vt)
check("SMART best_state reproduces best_loss",
      abs(bce_now - rep["best_loss"]) < 5e-4,
      f"reloaded={bce_now:.6f} best={rep['best_loss']:.6f}")
check("SMART 49 params", rep["n_params"] == 49)
model_l, rep_l = train_smart(vl, vt, objective="bce", n_iter=400, patience=100,
                             seed=0, legacy_pairing=True)
P_l = smart_predict_proba(model_l, vl)
print(f"      legacy pairing reloaded BCE={R.topbce(P_l, vt):.6f} "
      f"vs best={rep_l['best_loss']:.6f} (pairing offset documented)")

# 11. ops metrics sanity
om = R.ops_metrics(probs, y)
check("ops coverage monotone", om["cov_a80"] >= om["cov_a90"] >= om["cov_a95"])

print()
if FAIL:
    print("FAILURES:", FAIL)
    sys.exit(1)
print("ALL CHECKS PASSED")

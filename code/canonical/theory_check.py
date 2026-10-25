"""WP7 numerical check of the corrected fixed-partition statement.

Synthetic quadratic model (software check only, never paper evidence):
per-sample loss in group g: l(t) = .5 L2 (t - tau_g)^2 + eps_i t,
eps ~ N(0, sig^2)  => gradient at t*: d_i = L2(t* - tau_g) + eps_i.
Closed forms: B = mass-weighted Var_g of group-mean gradients at t*
            = L2^2 Var_g(tau_g);  sigma_bar^2 = sig^2.
Prediction: E[Delta] = -(2 L2)^-1 (B - (K-1)/n sig^2)   (grouped - pooled
population excess loss; negative = grouping helps).
"""
import numpy as np

rng = np.random.default_rng(7)
L2, sig = 2.0, 1.5
REPS = 20000

print(f"{'K':>3} {'n':>5} {'spread':>7} {'B':>8} {'pred':>10} {'sim':>10}")
for K in (2, 4, 8):
    for n in (250, 1000, 4000):
        for spread in (0.0, 0.3):  # spread=0 => B=0 (random partition case)
            tau = spread * (np.arange(K) - (K - 1) / 2) / max(1, (K - 1) / 2)
            tstar = tau.mean()
            B = L2 ** 2 * np.mean((tau - tstar) ** 2)
            pred = -(B - (K - 1) / n * sig ** 2) / (2 * L2)
            deltas = np.empty(REPS)
            for r in range(REPS):
                g = rng.integers(0, K, n)
                eps = rng.normal(0, sig, n)
                d = L2 * (tstar - tau[g]) + eps
                t_pool = tstar - d.mean() / L2
                loss_p = loss_g = 0.0
                for k in range(K):
                    m = g == k
                    t_k = tstar - d[m].mean() / L2 if m.any() else t_pool
                    loss_g += m.mean() * 0.5 * L2 * (t_k - tau[k]) ** 2
                    loss_p += m.mean() * 0.5 * L2 * (t_pool - tau[k]) ** 2
                deltas[r] = loss_g - loss_p
            print(f"{K:>3} {n:>5} {spread:>7.1f} {B:>8.4f} {pred:>10.6f} "
                  f"{deltas.mean():>10.6f}")
print("\nExpect sim ~= pred; at spread=0 (B=0) sim must be POSITIVE "
      "(variance cost), refuting the old 'inert/zero' wording.")

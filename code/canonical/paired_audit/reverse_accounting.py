#!/usr/bin/env python3
"""Plan A9 note 3 (a): reverse-order accounting and the direction condition.
Forward order (A8 ii):  S = mean_{A_C}(c_C - c_B),  P = mean_{A_C}(c_B - b) - mean_{A_B}(c_B - b).
Reverse order:          S' = mean_{A_B}(c_C - c_B), P' = mean_{A_C}(c_C - b) - mean_{A_B}(c_C - b).
Both satisfy S + P = S' + P' = g_C - g_B (checked per cell).  Direction:
for Delta = g_C - g_B, |g_C| < |g_B| iff sign and size satisfy
|g_B + Delta| < |g_B|; for Delta > 0 this is g_B < -Delta/2.  Summaries as
in A8 (ii): mean over draws then seeds, descriptive 95% seed->draw
intervals (B = 20000, RNG 20261016).
Output: results/www2027/posthoc_mechanism/reverse_accounting.json
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paired_audit.identities import accepted_fixed, accepted_matched  # noqa: E402
from paired_audit.run_mechanism import B, MAPS, OUT, RNG, SPLITS, TS, hier  # noqa: E402


def mean(x, A):
    return float(x[A].mean()) if A.any() else float("nan")


def main():
    rec = {}
    worst = 0.0
    for f in sorted(glob.glob(str(OUT / "cache_s*_n*.npz"))):
        z = np.load(f)
        s = int(Path(f).stem.split("_s")[1].split("_n")[0]); b = int(Path(f).stem.split("_n")[1])
        for sp in SPLITS:
            conf, corr = z[f"conf_{sp}"], z[f"corr_{sp}"].astype(np.float64)
            for d in range(conf.shape[0]):
                cB = conf[d, 0].astype(np.float64)
                for t in TS:
                    AB = accepted_fixed(cB, t); k = int(AB.sum())
                    for mi, m in enumerate(MAPS[1:], start=1):
                        cC = conf[d, mi].astype(np.float64)
                        for rule, AC_, AB_ in (("fixed", accepted_fixed(cC, t), AB),
                                               ("matched", accepted_matched(cC, k), accepted_matched(cB, k))):
                            gC, gB = mean(cC - corr, AC_), mean(cB - corr, AB_)
                            S = mean(cC - cB, AC_); P = mean(cB - corr, AC_) - gB
                            S2 = mean(cC - cB, AB_); P2 = gC - mean(cC - corr, AB_)
                            if gC == gC and gB == gB:
                                worst = max(worst, abs(S + P - (gC - gB)), abs(S2 + P2 - (gC - gB)))
                            rec[(b, sp, t, rule, m, s, d)] = dict(S=S, P=P, S2=S2, P2=P2, gB=gB, gC=gC)
    draws = sorted({k[6] for k in rec})
    summ = {}
    for b in sorted({k[0] for k in rec}):
        seeds = sorted({k[5] for k in rec if k[0] == b})
        rng = np.random.default_rng(RNG); i_s, i_d = rng.integers(0, len(seeds), (B, len(seeds))), rng.integers(0, len(draws), (B, len(seeds), len(draws)))
        for sp in SPLITS:
            for t in TS:
                for rule in ("fixed", "matched"):
                    for m in MAPS[1:]:
                        e = {}
                        for q in ("S", "P", "S2", "P2", "gB", "gC"):
                            A = np.array([[rec[(b, sp, t, rule, m, s, d)][q] for d in draws] for s in seeds], dtype=float)
                            e[q] = {"mean": float(np.nanmean(np.nanmean(A, 1))),
                                    "ci95": [float(x) for x in np.nanpercentile(hier(A, i_s, i_d), [2.5, 97.5])]}
                        D = e["gC"]["mean"] - e["gB"]["mean"]
                        e["delta"] = D
                        e["share_P_forward"] = abs(e["P"]["mean"]) / (abs(e["S"]["mean"]) + abs(e["P"]["mean"]))
                        e["share_P_reverse"] = abs(e["P2"]["mean"]) / (abs(e["S2"]["mean"]) + abs(e["P2"]["mean"]))
                        e["narrows_mean"] = bool(abs(e["gC"]["mean"]) < abs(e["gB"]["mean"]))
                        e["condition_gB_below_minus_half_delta"] = bool(e["gB"]["mean"] < -D / 2) if D > 0 else None
                        summ[f"n={b}/{sp}/t={t}/{rule}/{m}"] = e
    out = {"plan": "A9 note 3 (a)", "max_identity_residual_both_orders": worst, "summary": summ}
    (OUT / "reverse_accounting.json").write_text(json.dumps(out, indent=1) + "\n")
    print("max residual (both orders)", worst)
    for b in (2500, 250):
        for rule in ("fixed", "matched"):
            cells = [(k, v) for k, v in summ.items() if k.startswith(f"n={b}/") and f"/{rule}/" in k]
            for t in TS:
                ct = [v for k, v in cells if f"/t={t}/" in k]
                print(f"n={b} {rule:7s} t={t}: P' {100*min(v['P2']['mean'] for v in ct):+.2f}..{100*max(v['P2']['mean'] for v in ct):+.2f}"
                      f" share' max {max(v['share_P_reverse'] for v in ct):.2f} | P {100*min(v['P']['mean'] for v in ct):+.2f}..{100*max(v['P']['mean'] for v in ct):+.2f}"
                      f" share max {max(v['share_P_forward'] for v in ct):.2f} | S' {100*min(v['S2']['mean'] for v in ct):+.2f}..{100*max(v['S2']['mean'] for v in ct):+.2f}")
    # direction: does sign(gB) alone predict narrowing? (fixed, n=2500)
    for b in (2500,):
        for rule in ("fixed", "matched"):
            cells = [(k, v) for k, v in summ.items() if k.startswith(f"n={b}/") and f"/{rule}/" in k]
            neg = [(k, v) for k, v in cells if v["gB"]["mean"] < 0]
            print(f"n={b} {rule}: cells {len(cells)}, gB<0 in {len(neg)}, of which narrow {sum(v['narrows_mean'] for k, v in neg)};"
                  f" condition gB<-delta/2 matches narrowing in {sum((v['condition_gB_below_minus_half_delta'] == v['narrows_mean']) for k, v in cells if v['delta'] > 0)}/{sum(v['delta'] > 0 for k, v in cells)}")


if __name__ == "__main__":
    main()

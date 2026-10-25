#!/usr/bin/env python3
"""Plan addendum A3: within-family capacity contrasts with intervals.
K8-K1 (family A, no floor) and Poly4-Poly1 (family B) on c100_vit/test and
amazon_distilbert/id_test at n=250, n=2500 and the interaction.
Output: results/www2027/posthoc_f2/analysis_family_capacity.json"""
import json
from pathlib import Path

import numpy as np

import gen_capacity_fig as G

B, NF = 20000, 12
ZB = 100 * 0.05 / NF / 2
OUT = Path("__SRTS_ROOT__/calattn-repo/results/www2027/posthoc_f2")


def deltas(rows, hi, lo, b):
    tab = {(r["seed"], r["draw"], r["method"]): r["ece15"] for r in rows
           if r["budget"] == b}
    seeds = sorted({k[0] for k in tab})
    draws = sorted({k[1] for k in tab})
    return np.array([[tab[(s, d, hi)] - tab[(s, d, lo)] for d in draws]
                     for s in seeds])


def boot(M, rng):
    S, D = M.shape
    i_s = rng.integers(0, S, (B, S))
    if D == 1:
        return M[i_s, 0].mean(1)
    i_d = rng.integers(0, D, (B, S, D))
    return M[i_s[:, :, None], i_d].mean(2).mean(1)


def rec(M, bt):
    lo, hi = np.percentile(bt, [2.5, 97.5])
    lb, hb = np.percentile(bt, [ZB, 100 - ZB])
    ps = M.mean(1)
    return {"point": float(ps.mean()), "ci95": [float(lo), float(hi)],
            "ci_bonf12": [float(lb), float(hb)],
            "excl0_bonf12": bool(lb > 0 or hb < 0),
            "excl0_95": bool(lo > 0 or hi < 0),
            "per_seed": [float(x) for x in ps]}


def main():
    rng = np.random.default_rng(20261016)
    out = {"plan": "POSTHOC_PLAN_2026-10-16.md A3", "B": B, "family_size": NF}
    for cell, split in (("c100_vit", "test"), ("amazon_distilbert", "id_test")):
        for fam, hi, lo in (("familyA", "SRTS-K8-floor0", "SRTS-K1-floor0"),
                            ("familyB", "Poly4", "Poly1(=scalarBCE)")):
            rows = G.check_rows(G.load_family(cell, fam, split)[0])
            M250, M2500 = deltas(rows, hi, lo, 250), deltas(rows, hi, lo, 2500)
            S = M250.shape[0]
            i_s = rng.integers(0, S, (B, S))
            i_d250 = rng.integers(0, M250.shape[1], (B, S, M250.shape[1]))
            b250 = M250[i_s[:, :, None], i_d250].mean(2).mean(1)
            if M2500.shape[1] == 1:
                b2500 = M2500[i_s, 0].mean(1)
            else:
                i_d2 = rng.integers(0, M2500.shape[1], (B, S, M2500.shape[1]))
                b2500 = M2500[i_s[:, :, None], i_d2].mean(2).mean(1)
            key = f"{cell}/{split}/{fam}: {hi} - {lo}"
            out[key] = {"n=250": rec(M250, b250), "n=2500": rec(M2500, b2500),
                        "interaction 250-2500": rec(
                            M250.mean(1, keepdims=True) - M2500.mean(1, keepdims=True),
                            b250 - b2500)}
    (OUT / "analysis_family_capacity.json").write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        if isinstance(v, dict) and "n=250" in v:
            print(k)
            for kk, c in v.items():
                print(f"   {kk:22s} {c['point']:+.3f} [{c['ci95'][0]:+.3f},{c['ci95'][1]:+.3f}] "
                      f"bonf12[{c['ci_bonf12'][0]:+.3f},{c['ci_bonf12'][1]:+.3f}] "
                      f"{'B' if c['excl0_bonf12'] else ('*' if c['excl0_95'] else '')}")


if __name__ == "__main__":
    main()

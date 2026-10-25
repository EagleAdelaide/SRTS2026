#!/usr/bin/env python3
"""Analysis of the F2 2x2 score-construction control (plan P2 + A1).

Contrasts (ECE15, lower is better), per setting:
  K1 C-OOF - D-OOF   K2 C-IN - D-IN   K3 C-IN - D-OOF (as originally reported)
  K4 C-OOF - C-IN    K5 D-OOF - D-IN  K6 K1 - K2
Settings: c100_vit/test n in {250, 2500}; amazon_distilbert/id_test n in
{250, 2500}; c100c_vit (95-cell mean, fit at n = 2500)  -> 30 contrasts.
Inference: paired deltas; seed->draw hierarchical bootstrap (B = 20000, RNG
20261016, one index array per setting shared by K1..K6); for settings with
one draw per seed only seeds are resampled.  95% and Bonferroni 1-0.05/30.
Descriptive: means of every variant for ECE15, NLL, top-label BCE, Brier at
all four budgets; per-seed signs; OOF/IN score diagnostics.
Output: results/www2027/posthoc_f2/analysis_f2.json
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

W = Path("__SRTS_ROOT__/calattn-repo/results/www2027/posthoc_f2")
B = 20000
N_FAM = 30
ZB = 100 * 0.05 / N_FAM / 2
V = ("D-OOF", "D-IN", "C-OOF", "C-IN")
K = {"K1 C-OOF - D-OOF": (("C-OOF", 1), ("D-OOF", -1)),
     "K2 C-IN - D-IN": (("C-IN", 1), ("D-IN", -1)),
     "K3 C-IN - D-OOF": (("C-IN", 1), ("D-OOF", -1)),
     "K4 C-OOF - C-IN": (("C-OOF", 1), ("C-IN", -1)),
     "K5 D-OOF - D-IN": (("D-OOF", 1), ("D-IN", -1)),
     "K6 (C-OOF - D-OOF) - (C-IN - D-IN)": (("C-OOF", 1), ("D-OOF", -1),
                                            ("C-IN", -1), ("D-IN", 1))}
SEC = ("ece15", "nll", "topbce", "brier")


def contrast_matrix(tab, seeds, draws, metric="ece15"):
    """seeds x draws arrays per contrast (NaN-free by construction)."""
    out = {}
    for name, terms in K.items():
        out[name] = np.array([[sum(w * tab[(s, d, v)][metric]
                                   for v, w in terms) for d in draws]
                              for s in seeds])
    return out


def boot(M, rng):
    S, D = M.shape
    i_s = rng.integers(0, S, (B, S))
    if D == 1:
        return M[i_s, 0].mean(1), (i_s, None)
    i_d = rng.integers(0, D, (B, S, D))
    return M[i_s[:, :, None], i_d].mean(2).mean(1), (i_s, i_d)


def summarize(mats, rng):
    S, D = next(iter(mats.values())).shape
    i_s = rng.integers(0, S, (B, S))
    i_d = rng.integers(0, D, (B, S, D)) if D > 1 else None
    res = {}
    for name, M in mats.items():
        bt = (M[i_s, 0].mean(1) if i_d is None
              else M[i_s[:, :, None], i_d].mean(2).mean(1))
        lo, hi = np.percentile(bt, [2.5, 97.5])
        lb, hb = np.percentile(bt, [ZB, 100 - ZB])
        ps = M.mean(1)
        res[name] = {"point": float(ps.mean()), "ci95": [float(lo), float(hi)],
                     "ci_bonf30": [float(lb), float(hb)],
                     "excl0_95": bool(lo > 0 or hi < 0),
                     "excl0_bonf30": bool(lb > 0 or hb < 0),
                     "per_seed": [float(x) for x in ps],
                     "seeds_negative": int((ps < 0).sum()),
                     "n_seeds": int(S), "draws_per_seed": int(D)}
    return res


def main():
    rng = np.random.default_rng(20261016)
    out = {"plan": "POSTHOC_PLAN_2026-10-16.md P2 + Addendum A1",
           "B": B, "family_size": N_FAM, "bonferroni_level": 1 - 0.05 / N_FAM,
           "settings": {}, "descriptive": {}}
    for cell, split in (("c100_vit", "test"), ("amazon_distilbert", "id_test")):
        rows = [json.loads(l) for l in open(W / f"{cell}_{split}.jsonl")]
        tab = {(r["seed"], r["budget"], r["draw"], r["variant"]): r
               for r in rows}
        assert len(tab) == len(rows), "duplicate rows"
        seeds = sorted({r["seed"] for r in rows})
        budgets = sorted({r["budget"] for r in rows})
        desc = {}
        for b in budgets:
            draws = sorted({r["draw"] for r in rows if r["budget"] == b})
            for s in seeds:
                for d in draws:
                    for v in V:
                        assert (s, b, d, v) in tab, ("missing", s, b, d, v)
            desc[b] = {v: {m: float(np.mean([np.mean([tab[(s, b, d, v)][m]
                                                      for d in draws])
                                             for s in seeds])) for m in SEC}
                       for v in V}
            desc[b]["diag"] = {
                "spearman_oof_in_mean": float(np.mean(
                    [tab[(s, b, d, "D-OOF")]["diag"]["spearman_oof_in"]
                     for s in seeds for d in draws])),
                "group_agree_oof_in_mean": float(np.mean(
                    [tab[(s, b, d, "D-OOF")]["diag"]["group_agree_oof_in"]
                     for s in seeds for d in draws]))}
            if b in (250, 2500):
                t2 = {(s, d, v): tab[(s, b, d, v)] for s in seeds
                      for d in draws for v in V}
                mats = contrast_matrix(t2, seeds, draws)
                out["settings"][f"{cell}/{split}/n={b}"] = summarize(mats, rng)
        out["descriptive"][f"{cell}/{split}"] = desc
    rows = [json.loads(l) for l in open(W / "c100c_vit_shift.jsonl")]
    seeds = sorted({r["seed"] for r in rows})
    cells = sorted({(r["corruption"], r["severity"]) for r in rows})
    assert len(cells) == 95 and len(rows) == len(seeds) * 95 * 4, len(rows)
    tab = {(r["seed"], r["corruption"], r["severity"], r["variant"]): r
           for r in rows}
    agg = {(s, 0, v): {m: float(np.mean([tab[(s, c, sv, v)][m]
                                         for c, sv in cells])) for m in SEC}
           for s in seeds for v in V}
    out["settings"]["c100c_vit/95-cell mean/n=2500"] = summarize(
        contrast_matrix(agg, seeds, [0]), rng)
    out["descriptive"]["c100c_vit"] = {v: {m: float(np.mean(
        [agg[(s, 0, v)][m] for s in seeds])) for m in SEC} for v in V}
    # per-cell win counts for K1 and K3 (descriptive)
    for name, (a, b_) in (("K1", ("C-OOF", "D-OOF")), ("K3", ("C-IN", "D-OOF"))):
        wins = sum(1 for s in seeds for c, sv in cells
                   if tab[(s, c, sv, a)]["ece15"] < tab[(s, c, sv, b_)]["ece15"])
        out["descriptive"]["c100c_vit"][f"{name}_cells_continuous_lower"] = \
            f"{wins}/{len(seeds) * 95}"
    n = sum(len(v) for v in out["settings"].values())
    assert n == N_FAM, n
    (W / "analysis_f2.json").write_text(json.dumps(out, indent=1))
    print("wrote", W / "analysis_f2.json")


if __name__ == "__main__":
    main()

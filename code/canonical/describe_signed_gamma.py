#!/usr/bin/env python3
"""Plan A5 note 3 (descriptive, post hoc): signed counterpart of Gamma.
Δs_X = g_map(A_map) − g_scalar(A_scalar) on split X (no absolute values),
Γ_s = Δs_unseen − Δs_seen, within each (map, budget, threshold, seed, draw),
from the A4(iii) rows; plus the scalar's own signed g(A) per split and the
share of draws with g > 0. 95% intervals from the primary A5 index arrays
(description only, no multiplicity claim). Output:
results/www2027/posthoc_gamma/signed_descriptive.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_gamma import (B, BUDGETS, M, OODS, OUT, REF, RULES, SEEDS_RNG, TS,  # noqa: E402
                           boot_index, hier, load_matched)


def main():
    tab, _ = load_matched()
    seeds = sorted({k[1] for k in tab})
    draws = sorted({k[3] for k in tab})
    idx = boot_index(len(seeds), len(draws), np.random.default_rng(SEEDS_RNG[0]))

    def mat(f):
        return np.array([[f(s, d) for d in draws] for s in seeds])

    def summ(A):
        bt = hier(A, idx)
        return {"mean": float(np.nanmean(np.nanmean(A, axis=1))),
                "ci95": [float(x) for x in np.nanpercentile(bt, [2.5, 97.5])]}

    out = {"note": "plan A5 note 3; descriptive, post hoc; B=%d, RNG %d" % (B, SEEDS_RNG[0]),
           "contrasts": [], "scalar": []}
    for rule, fp in RULES.items():
        for b in BUDGETS:
            for t in TS:
                f = fp.format(t=t)
                for sp in ("id_test",) + OODS:
                    g = mat(lambda s, d: tab[(sp, s, b, d, REF)][f])
                    out["scalar"].append(dict(rule=rule, split=sp, budget=b, t=t / 100, **summ(g),
                                              share_pos=float(np.nanmean(g > 0))))
                for ood in OODS:
                    for m in M:
                        ds = {sp: mat(lambda s, d: tab[(sp, s, b, d, m)][f] - tab[(sp, s, b, d, REF)][f])
                              for sp in ("id_test", ood)}
                        out["contrasts"].append(dict(
                            rule=rule, ood=ood, budget=b, t=t / 100, method=m,
                            dsigned_seen=summ(ds["id_test"]), dsigned_unseen=summ(ds[ood]),
                            gamma_signed=summ(ds[ood] - ds["id_test"])))
    gs = [c["gamma_signed"] for c in out["contrasts"]]
    out["summary"] = {
        "gamma_signed_range": [min(x["mean"] for x in gs), max(x["mean"] for x in gs)],
        "gamma_signed_ci95_excl0_pos": sum(x["ci95"][0] > 0 for x in gs),
        "gamma_signed_ci95_excl0_neg": sum(x["ci95"][1] < 0 for x in gs),
        "dsigned_seen_range": [min(c["dsigned_seen"]["mean"] for c in out["contrasts"]),
                               max(c["dsigned_seen"]["mean"] for c in out["contrasts"])],
        "dsigned_unseen_range": [min(c["dsigned_unseen"]["mean"] for c in out["contrasts"]),
                                 max(c["dsigned_unseen"]["mean"] for c in out["contrasts"])]}
    (OUT / "signed_descriptive.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out["summary"], indent=1))
    for c in out["contrasts"]:
        if c["budget"] == 2500 and c["t"] == 0.9:
            print(c["rule"], c["ood"][:8], c["method"],
                  "seen %+.2f unseen %+.2f Gs %+.2f [%+.2f,%+.2f]" % (
                      100 * c["dsigned_seen"]["mean"], 100 * c["dsigned_unseen"]["mean"],
                      100 * c["gamma_signed"]["mean"], *[100 * x for x in c["gamma_signed"]["ci95"]]))
    for c in out["scalar"]:
        if c["budget"] == 2500:
            print("scalar", c["rule"], c["split"][:8], c["t"], "g %+.2f [%+.2f,%+.2f] share>0 %.2f" % (
                100 * c["mean"], *[100 * x for x in c["ci95"]], c["share_pos"]))


if __name__ == "__main__":
    main()

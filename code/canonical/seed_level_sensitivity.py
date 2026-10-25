#!/usr/bin/env python3
"""Plan addendum A4(ii): seed-level sensitivity for every reported contrast.
From the per-seed means stored in the analysis outputs, compute a t interval
(df = seeds - 1) and the number of seeds sharing the sign of the mean. This
treats backbone seeds as the unit of replication (inference to new
backbones), whereas the frozen seed->draw bootstrap describes variation
across these backbones and draws. Output: posthoc_f2/seed_level_sensitivity.json
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

W = Path("__SRTS_ROOT__/calattn-repo/results/www2027")


def seedlevel(ps):
    ps = np.asarray(ps, dtype=float)
    S = len(ps); m = ps.mean()
    se = ps.std(ddof=1) / np.sqrt(S)
    h = stats.t.ppf(0.975, S - 1) * se
    same = int((np.sign(ps) == np.sign(m)).sum())
    return {"n_seeds": S, "mean": float(m), "t95": [float(m - h), float(m + h)],
            "t95_excl0": bool(m - h > 0 or m + h < 0), "same_sign": same}


def main():
    out = {}
    for name, f, cell in (("frozen amazon", "analysis_CONFIRM_amazon_distilbert_ece15.json", "amazon_distilbert"),
                          ("frozen yahoo", "analysis_CONFIRM_yahoo_distilbert_ece15.json", "yahoo_distilbert"),
                          ("frozen anchor", "analysis_c100_core_natural_ece15.json", "c100_vit")):
        d = json.load(open(W / f))[cell]
        for b in ("250", "2500"):
            for k, v in d[b].items():
                out[f"{name}/n={b}/{k}"] = {**seedlevel(list(v["per_seed"].values())),
                                           "boot95_excl0": v["excludes_zero"]}
    for fn, tag in (("posthoc_endpoints/endpoints.json", "conf"), ("posthoc_endpoints/endpoints_dev.json", "dev")):
        r = json.load(open(W / fn))["tasks"]
        for task, tr in r.items():
            for b, cs in tr["contrasts"].items():
                for m, qs in cs.items():
                    for q, c in qs.items():
                        if c["core"] or q == "ece15":
                            out[f"endpoints {tag}/{task}/n={b}/{m}/{q}"] = {
                                **seedlevel(c["per_seed"]), "boot95_excl0": c["excl0_95"],
                                "bonf_excl0": c.get("excl0_bonf")}
    for k, v in json.load(open(W / "posthoc_f2/analysis_f2.json"))["settings"].items():
        for kk, c in v.items():
            out[f"f2/{k}/{kk}"] = {**seedlevel(c["per_seed"]), "boot95_excl0": c["excl0_95"],
                                   "bonf_excl0": c["excl0_bonf30"]}
    for k, v in json.load(open(W / "posthoc_f2/analysis_family_capacity.json")).items():
        if isinstance(v, dict) and "n=250" in v:
            for kk, c in v.items():
                out[f"family/{k}/{kk}"] = {**seedlevel(c["per_seed"]), "boot95_excl0": c["excl0_95"],
                                           "bonf_excl0": c["excl0_bonf12"]}
    (W / "posthoc_f2/seed_level_sensitivity.json").write_text(json.dumps(out, indent=1))
    print(len(out), "contrasts written")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""EXPLORATORY (post-unblinding, labelled): reviewer-level (support 75)
error-count estimation on the Amazon OOD test; n=2500, draw 0 per seed.
Recovered from the command that wrote EXPLORATORY_reviewer_level.json on
2026-10-10. Reviewer ids come from the joined logits ('reviewer', from the
official WILDS metadata) or, internally, from the tokenizer cache."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_suite import load_cell, fit_methods, WEB
from sampler import natural_subsample

META = Path("__SRTS_ROOT__/data/wilds/tokcache/distilbert_meta.npz")
OUT = sys.argv[1] if len(sys.argv) > 1 else str(WEB / "EXPLORATORY_reviewer_level.json")


def reviewers(s):
    z = np.load(WEB / f"wilds_amazon/distilbert_s{s}/logits_test.npz")
    if "reviewer" in z.files:
        return z["reviewer"]
    return np.load(META)["reviewer"][z["row_idx"]]


def main():
    res = {}
    for s in range(5):
        vl_p, vt_p, tests, _ = load_cell("amazon_distilbert", s, True)
        tl, tt = tests["ood_test_CONFIRMATION"]
        rev = reviewers(s)
        rng = np.random.default_rng(77_000_000 + 1_000_000 * s + 1000 * 2500 + 0)
        idx, _ = natural_subsample(vt_p, 2500, rng)
        fitted = fit_methods("core", vl_p[idx], vt_p[idx], s)
        for name in ("TvA-TS", "SMART+BCE", "SRTS-BCE"):
            P = fitted[name][0](tl)
            conf = P.max(1); corr = (P.argmax(1) == tt).astype(float)
            errs = np.array([abs((1 - conf[rev == r]).sum() - (1 - corr[rev == r]).sum())
                             for r in np.unique(rev)])
            res.setdefault(name, []).append(dict(seed=s, p50=float(np.percentile(errs, 50)),
                                                 p90=float(np.percentile(errs, 90)),
                                                 mean=float(errs.mean()), n_rev=len(errs)))
        print("seed", s, "done", flush=True)
    json.dump(res, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()

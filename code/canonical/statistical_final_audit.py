#!/usr/bin/env python3
"""STATISTICAL FINAL AUDIT (author P0, 2026-10-10).

Facts audited:
(a) The earlier 'Bonferroni' run regenerated bootstrap replicates with a
    fresh stream, so its endpoints were NOT quantiles of the same arrays
    as the frozen 95% CIs -> apparent nesting violations were
    Monte-Carlo artifacts. Original replicate arrays were not persisted.
(b) This audit computes, for each of the 12 confirmation primary
    comparisons (6 per task), ONE replicate array (same estimand, same
    hierarchical seed->draw algorithm as analyze_suite.hier_boot, same
    per-seed delta construction incl. full-pool dedupe and interaction
    broadcast), with documented per-comparison seeds and B=200,000, and
    takes the 95% / 99.1667% (6-way Bonferroni per task) / 99.5833%
    (12-way global Bonferroni) percentile CIs FROM THAT SAME ARRAY.
(c) Programmatic assertions: CI nesting (higher confidence contains
    lower) for every comparison; plus reproduction of the frozen
    endpoints by rerunning the ORIGINAL analyze_suite stream design
    (seed 20261009, B=20000, original comparison order).
Both corrected levels are POST-UNBLINDING SENSITIVITIES, never
preregistered tests. Frozen nominal-95% CIs remain the primary
inference and are not modified.
Outputs: results/www2027/statistical_final_audit.json and the
replicate-array summaries; markdown written separately.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze_suite import load_rows, hier_boot, analyze  # noqa: E402

REPO = Path("__SRTS_ROOT__/calattn-repo")
OUT = REPO / "results/www2027"
B_AUDIT = 200_000
# CI level 1-a: percentiles (100*a/2, 100-100*a/2).
# bonf6:  a = .05/6  -> (0.41667, 99.58333)  [99.1667% CI]
# bonf12: a = .05/12 -> (0.20833, 99.79167)  [99.5833% CI]
LEVELS = {"95": (2.5, 97.5),
          "99.1667_bonf6": (100 * 0.05 / 6 / 2, 100 - 100 * 0.05 / 6 / 2),
          "99.5833_bonf12": (100 * 0.05 / 12 / 2, 100 - 100 * 0.05 / 12 / 2)}


def per_seed_deltas(rows, idxm, seeds, ma, mb, b):
    per = {}
    for s in seeds:
        dr = sorted({r["draw"] for r in rows
                     if r["seed"] == s and r["budget"] == b
                     and r["method"] == ma})
        a = np.array([idxm[(s, b, d, ma)] - idxm[(s, b, d, mb)] for d in dr])
        if len(a) > 1 and np.ptp(a) == 0.0:
            a = a[:1]
        per[s] = a
    return per


def interaction_deltas(rows, idxm, seeds, ma, mb, b1, b2):
    p1 = per_seed_deltas(rows, idxm, seeds, ma, mb, b1)
    p2 = per_seed_deltas(rows, idxm, seeds, ma, mb, b2)
    return {s: (p1[s] - p2[s] if len(p1[s]) == len(p2[s])
                else p1[s] - p2[s].mean()) for s in seeds}


def main():
    audit = {"script": str(Path(__file__).resolve()),
             "B_audit": B_AUDIT,
             "levels_percentiles": LEVELS,
             "note": "both corrected levels are post-unblinding "
                     "sensitivities; frozen nominal-95% remains primary"}
    failures = []
    for cell in ("amazon_distilbert", "yahoo_distilbert"):
        rows = [r for r in load_rows(
            [str(OUT / f"CONFIRMATION/{cell}.jsonl")])
            if r["test_split"] == "CONFIRMATION" and "ece15" in r]
        seeds = sorted({r["seed"] for r in rows})
        idxm = {(r["seed"], r["budget"], r["draw"], r["method"]): r["ece15"]
                for r in rows}
        comps = {
            "SRTS-TvA@250":   per_seed_deltas(rows, idxm, seeds, "SRTS-BCE", "TvA-TS", 250),
            "SRTS-SMART@250": per_seed_deltas(rows, idxm, seeds, "SRTS-BCE", "SMART+BCE", 250),
            "SRTS-TvA@2500":  per_seed_deltas(rows, idxm, seeds, "SRTS-BCE", "TvA-TS", 2500),
            "SRTS-SMART@2500": per_seed_deltas(rows, idxm, seeds, "SRTS-BCE", "SMART+BCE", 2500),
            "IX-TvA(250-2500)": interaction_deltas(rows, idxm, seeds, "SRTS-BCE", "TvA-TS", 250, 2500),
            "IX-SMART(250-2500)": interaction_deltas(rows, idxm, seeds, "SRTS-BCE", "SMART+BCE", 250, 2500),
        }
        cellres = {}
        for i, (name, per) in enumerate(comps.items()):
            seed_val = 424_242 + 1000 * (0 if cell.startswith("amazon") else 1) + i
            rng = np.random.default_rng(seed_val)
            boots = hier_boot(per, rng, B=B_AUDIT)   # ONE array per comp
            point = float(np.mean([a.mean() for a in per.values()]))
            entry = {"point": point, "B": B_AUDIT, "rng_seed": seed_val,
                     "n_seeds": len(per),
                     "n_draw_deltas": int(sum(len(a) for a in per.values())),
                     "boot_mean": float(boots.mean()),
                     "boot_sd": float(boots.std())}
            for lv, (qlo, qhi) in LEVELS.items():
                lo, hi = np.percentile(boots, [qlo, qhi])
                entry[f"ci_{lv}"] = [float(lo), float(hi)]
                entry[f"excl0_{lv}"] = bool(lo > 0 or hi < 0)
            # programmatic nesting assertions on the SAME array
            l95, l6, l12 = (entry["ci_95"], entry["ci_99.1667_bonf6"],
                            entry["ci_99.5833_bonf12"])
            ok = (l6[0] <= l95[0] and l6[1] >= l95[1]
                  and l12[0] <= l6[0] and l12[1] >= l6[1])
            entry["nesting_ok"] = bool(ok)
            if not ok:
                failures.append((cell, name, l95, l6, l12))
            cellres[name] = entry
        audit[cell] = cellres
    # (c) reproduce the FROZEN endpoints with the original stream design
    repro = {}
    for cell in ("amazon_distilbert", "yahoo_distilbert"):
        rows = load_rows([str(OUT / f"CONFIRMATION/{cell}.jsonl")])
        res = analyze(rows, "ece15",
                      [("SRTS-BCE", "TvA-TS"), ("SRTS-BCE", "SMART+BCE")],
                      [250, 625, 1250, 2500], "CONFIRMATION", "natural")
        frozen = json.load(open(OUT / f"analysis_CONFIRM_{cell}_ece15.json"))
        agree = []
        for b in ("250", "2500"):
            for comp in ("SRTS-BCE - TvA-TS", "SRTS-BCE - SMART+BCE"):
                a = res[cell][int(b)][comp]["hier_ci"]
                f = frozen[cell][b][comp]["hier_ci"]
                agree.append(dict(budget=b, comp=comp, rerun=a, frozen=f,
                                  max_abs_diff=float(max(abs(a[0] - f[0]),
                                                         abs(a[1] - f[1])))))
        repro[cell] = agree
    audit["frozen_reproduction_same_stream"] = repro
    audit["nesting_failures"] = failures
    json.dump(audit, open(OUT / "statistical_final_audit.json", "w"),
              indent=1)
    # console summary
    for cell in ("amazon_distilbert", "yahoo_distilbert"):
        print(f"== {cell}")
        for name, e in audit[cell].items():
            print(f"  {name:20s} pt {e['point']:+.4f} "
                  f"95[{e['ci_95'][0]:+.4f},{e['ci_95'][1]:+.4f}] "
                  f"b6[{e['ci_99.1667_bonf6'][0]:+.4f},{e['ci_99.1667_bonf6'][1]:+.4f}]"
                  f"{'*' if e['excl0_99.1667_bonf6'] else '°'} "
                  f"b12[{e['ci_99.5833_bonf12'][0]:+.4f},{e['ci_99.5833_bonf12'][1]:+.4f}]"
                  f"{'*' if e['excl0_99.5833_bonf12'] else '°'} "
                  f"nest={'OK' if e['nesting_ok'] else 'FAIL'}")
    print("nesting failures:", len(failures))
    mx = max(x["max_abs_diff"] for c in repro.values() for x in c)
    print(f"frozen-endpoint reproduction (same stream, B=20000): "
          f"max |diff| = {mx:.2e}")
    assert not failures, "CI nesting violated on a shared replicate array"
    print("ALL ASSERTIONS PASSED")


if __name__ == "__main__":
    main()

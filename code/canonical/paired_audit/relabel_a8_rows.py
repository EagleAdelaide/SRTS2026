#!/usr/bin/env python3
"""One-off repair of the A8(iii) raw rows: simulate_audit.py (A8 version)
shadowed the map name with the audit budget m, so the "method" column of
audit_rows.csv / policy_rows.csv held m.  Estimates were unaffected (the
map's confidences are indexed by position).  This script regenerates the
deterministic row order without Monte Carlo, checks that every row's
map-dependent truth (delta_true; gB_eval/gC_eval) matches to 1e-12, and
rewrites the method column.  Aggregates (AUDIT_LABEL_EFFICIENCY.csv,
summary.json) do not use the column and are unchanged."""
import csv
import glob
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paired_audit.identities import accepted_fixed, accepted_matched, gap  # noqa: E402
from paired_audit.simulate_audit import CACHE, MAPS, MGRID, OUT, SPLITS, TS, pools  # noqa: E402


def keys():
    for f in sorted(glob.glob(str(CACHE / "cache_s*_n*.npz"))):
        z = np.load(f)
        for sp in SPLITS:
            conf = z[f"conf_{sp}"][0].astype(np.float64); corr = z[f"corr_{sp}"].astype(np.float64)
            aud, ev = pools(z[f"rev_{sp}"]); cB = conf[0]
            for t in TS:
                for mi, mname in enumerate(MAPS[1:], start=1):
                    cC = conf[mi]
                    for rule in ("fixed", "matched"):
                        if rule == "fixed":
                            AC, AB = accepted_fixed(cC, t), accepted_fixed(cB, t)
                        else:
                            k = int(accepted_fixed(cB, t).sum()); AC, AB = accepted_matched(cC, k), accepted_matched(cB, k)
                        tr = {}
                        for name, mask in (("audit", aud), ("eval", ev)):
                            A1, A0 = AC & mask, AB & mask
                            if A1.sum() < 10 or A0.sum() < 10:
                                tr = None; break
                            tr[name] = (gap(cB, corr, A0), gap(cC, corr, A1))
                        if tr is None:
                            continue
                        yield mname, tr["audit"][1] - tr["audit"][0], tr["eval"]


def main():
    seq = list(keys())
    for name, per in (("audit_rows.csv", 4 * len(MGRID)), ("policy_rows.csv", len(MGRID))):
        rows = list(csv.DictReader(open(OUT / name)))
        assert len(rows) == per * len(seq), (name, len(rows), per * len(seq))
        worst = 0.0
        for i, r in enumerate(rows):
            mname, dtrue, (gBe, gCe) = seq[i // per]
            if name == "audit_rows.csv":
                worst = max(worst, abs(float(r["delta_true"]) - dtrue))
            else:
                worst = max(worst, abs(float(r["gB_eval"]) - gBe), abs(float(r["gC_eval"]) - gCe))
            assert r["method"] == r["m"] or r["method"] == mname
            r["method"] = mname
        assert worst < 1e-12, (name, worst)
        with open(OUT / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator="\n"); w.writeheader(); w.writerows(rows)
        print(name, "relabelled", len(rows), "rows; max truth diff", worst)


if __name__ == "__main__":
    main()

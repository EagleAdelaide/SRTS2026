#!/usr/bin/env python3
"""Plan A9: merge the A8(iii) audit rows (E1-E4) with the A9 rows (E5-E7)
into AUDIT_LABEL_EFFICIENCY.csv and the "efficiency" block of summary.json
(policy block unchanged).  Before overwriting, checks that the E1-E4
aggregates reproduce the existing A8 file exactly."""
import csv
import json
from pathlib import Path

import numpy as np

OUT = Path("__SRTS_ROOT__/calattn-repo/results/www2027/posthoc_audit")
NUM = ("delta_true", "bias", "rmse", "mc_sd", "mean_se", "coverage95", "unique_labels", "overlap")


def load(name):
    rows = list(csv.DictReader(open(OUT / name)))
    for r in rows:
        for k in NUM:
            r[k] = float(r[k])
        r["budget"], r["m"] = int(r["budget"]), int(r["m"])
    return rows


def aggregate(rows):
    agg = {}
    for r in rows:
        agg.setdefault((r["budget"], r["split"], r["rule"], r["m"], r["design"]), []).append(r)
    eff = []
    for k, rs in sorted(agg.items()):
        rm = np.mean([x["rmse"] for x in rs]); e3 = agg[(k[0], k[1], k[2], k[3], "E3")]
        eff.append(dict(budget=k[0], split=k[1], rule=k[2], m=k[3], design=k[4], n_cells=len(rs),
                        rmse_mean_pts=100 * rm, rmse_rel_to_E3=rm / np.mean([x["rmse"] for x in e3]),
                        abs_bias_mean_pts=100 * np.mean([abs(x["bias"]) for x in rs]),
                        coverage95_mean=np.mean([x["coverage95"] for x in rs]),
                        unique_labels_mean=np.mean([x["unique_labels"] for x in rs])))
    return eff


def main():
    a8 = load("audit_rows.csv")
    summ = json.load(open(OUT / "summary.json"))
    old = {(e["budget"], e["split"], e["rule"], e["m"], e["design"]): e for e in summ["efficiency"] if e["design"] in ("E1", "E2", "E3", "E4")}
    re8 = {(e["budget"], e["split"], e["rule"], e["m"], e["design"]): e for e in aggregate(a8)}
    worst = max(abs(re8[k][q] - old[k][q]) for k in old for q in ("rmse_mean_pts", "coverage95_mean", "unique_labels_mean"))
    assert set(re8) == set(old) and worst < 1e-9, f"A8 aggregates not reproduced ({worst})"
    a9 = load("audit_rows_active.csv")
    eff = aggregate(a8 + a9)
    with open(OUT / "AUDIT_LABEL_EFFICIENCY.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(eff[0].keys()), lineterminator="\n"); w.writeheader(); w.writerows(eff)
    summ["efficiency"] = eff
    summ["plan"] = "A8(iii) + A9 (E5-E7 estimation only; policy unchanged)"
    summ["rng0_A9"] = 20261025
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1) + "\n")
    print("merged; A8 reproduction max diff", worst)
    for e in eff:
        if e["budget"] == 2500 and e["rule"] == "fixed" and e["m"] in (100, 1000):
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in e.items()})


if __name__ == "__main__":
    main()

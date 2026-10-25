#!/usr/bin/env python3
"""Print the numbers quoted in Sections Q4/Q5 from the A8 result files
(used to write the text and by verify_paper_numbers.py)."""
import json
from pathlib import Path

import numpy as np

W = Path("__SRTS_ROOT__/calattn-repo/results/www2027")
MAPS = ["HTS-BCE", "SRTS-BCE", "SMART+BCE", "PWLinear-3"]
SPL = ["id_test", "ood_val", "ood_test_CONFIRMATION"]


def numbers():
    a = json.load(open(W / "posthoc_mechanism/analysis_mechanism.json"))
    S = a["summary"]; out = {"gate": a["gate"]}
    # representative condition
    for rule in ("fixed", "matched"):
        for sp in SPL:
            es = [S[f"n=2500/{sp}/t=0.9/{rule}/{m}"] for m in MAPS]
            out[f"S_{rule}_{sp}"] = [100 * e["S"]["mean"] for e in es]
            out[f"P_{rule}_{sp}"] = [100 * e["P"]["mean"] for e in es]
            out[f"shareP_{rule}_{sp}"] = [e["share_P"] for e in es]
            out[f"gB_{rule}_{sp}"] = 100 * es[0]["gB"]["mean"]
            out[f"gBci_{rule}_{sp}"] = [100 * x for x in es[0]["gB"]["ci95"]]
    # all cells at n=2500 (both rules, 3 t, 3 splits, 4 maps)
    cells = [v for k, v in S.items() if k.startswith("n=2500/")]
    out["n2500_S_range"] = [100 * min(e["S"]["mean"] for e in cells), 100 * max(e["S"]["mean"] for e in cells)]
    out["n2500_absP_max"] = 100 * max(abs(e["P"]["mean"]) for e in cells)
    out["n2500_shareP_max_fixed"] = max(e["share_P"] for k, e in S.items() if k.startswith("n=2500/") and "/fixed/" in k)
    out["n2500_shareP_max_matched"] = max(e["share_P"] for k, e in S.items() if k.startswith("n=2500/") and "/matched/" in k)
    out["n2500_S_pos_all"] = all(e["S"]["ci95"][0] > 0 for e in cells)
    out["n2500_P_ci_excl0"] = sum(e["P"]["ci95"][0] > 0 or e["P"]["ci95"][1] < 0 for e in cells)
    out["n2500_cells"] = len(cells)
    # sign rule check: predicted |gC|-|gB| from gB+S+P vs measured absdiff (means)
    out["n2500_absdiff_pred_err_max"] = 100 * max(abs(e["pred_absdiff_from_gB"] - e["absdiff"]["mean"]) for e in cells)
    # n=250
    cells250 = [v for k, v in S.items() if k.startswith("n=250/")]
    out["n250_S_range"] = [100 * min(e["S"]["mean"] for e in cells250), 100 * max(e["S"]["mean"] for e in cells250)]
    out["n250_absP_max"] = 100 * max(abs(e["P"]["mean"]) for e in cells250)
    out["n250_shareP_max"] = max(e["share_P"] for e in cells250)
    # audit
    s = json.load(open(W / "posthoc_audit/summary.json"))
    eff = {(e["budget"], e["split"], e["rule"], e["m"], e["design"]): e for e in s["efficiency"]}
    for sp in ("ood_val", "id_test"):
        for m in (100, 1000):
            out[f"rmse_{sp}_m{m}"] = {d: eff[(2500, sp, "fixed", m, d)]["rmse_mean_pts"] for d in ("E1", "E2", "E3", "E4")}
            out[f"rel_{sp}_m{m}"] = eff[(2500, sp, "fixed", m, "E4")]["rmse_rel_to_E3"]
    cov = [e["coverage95_mean"] for e in s["efficiency"] if e["budget"] == 2500 and e["design"] == "E4" and e["m"] >= 100]
    out["E4_coverage_range_m>=100"] = [min(cov), max(cov)]
    cov25 = [e["coverage95_mean"] for e in s["efficiency"] if e["budget"] == 2500 and e["design"] == "E4" and e["m"] == 25]
    out["E4_coverage_m25"] = [min(cov25), max(cov25)]
    pol = s["policy"]
    for sp in ("ood_val", "id_test"):
        for m in (25, 100, 250, 1000):
            k = f"n=2500/{sp}/fixed/m={m}"
            out[f"policy_{sp}_m{m}"] = {q: pol[k][q] for q in ("switch", "false_switch", "missed", "regret_pts", "share_C_better")}
    return out


if __name__ == "__main__":
    for k, v in numbers().items():
        if isinstance(v, (list, dict)):
            if isinstance(v, list):
                v = [round(x, 3) if isinstance(x, float) else x for x in v]
            else:
                v = {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()}
        elif isinstance(v, float):
            v = round(v, 4)
        print(f"{k}: {v}")

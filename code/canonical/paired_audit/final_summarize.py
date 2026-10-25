#!/usr/bin/env python3
"""Plan A11 (i): corrected summaries of the A10 decision rows (no new simulation).
Reads results/www2027/posthoc_closure/dec_*.csv (unchanged historical rows).
For each (budget, split, rule, procedure, label-budget cap):
  actual mean distinct labels, census cells (labels == support), FPC within-pool power
  and false switches (truth = audit pool), and for the SAME no-FPC decisions: regret on
  the audit pool and on the evaluation pool; the source-pool oracle's regret on the
  evaluation pool (a reference choice, not a floor); and the signed
  excess_target_regret_vs_source_oracle (may be negative; cells with negative excess
  are counted).  Margin bins report cell counts, power and mean sampled fraction.
Writes results/www2027/posthoc_final/{decision_same_policy.csv, margin_bins.csv, summary_final.json}.
"""
import csv
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

W = Path("__SRTS_ROOT__/calattn-repo/results/www2027")
OUT = W / "posthoc_final"
EPS = 0.001
BINS = [0, 0.001, 0.0025, 0.005, 0.01, 1.0]


def reg(sw, gB, gC):
    best = np.minimum(np.abs(gB), np.abs(gC))
    return sw * np.abs(gC) + (1 - sw) * np.abs(gB) - best


def main():
    OUT.mkdir(exist_ok=True)
    d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(W / "posthoc_closure/dec_*.csv")))], ignore_index=True)
    d["reg_fpc_audit"] = reg(d.sw_fpc, d.gB_a, d.gC_a)
    d["reg_nofpc_audit"] = reg(d.sw_nofpc, d.gB_a, d.gC_a)
    d["reg_nofpc_eval"] = reg(d.sw_nofpc, d.gB_e, d.gC_e)
    d["reg_source_oracle_eval"] = reg(d.better_a.astype(float), d.gB_e, d.gC_e)
    d["excess_target_regret_vs_source_oracle"] = d.reg_nofpc_eval - d.reg_source_oracle_eval
    d["census"] = d.labels >= d.support_U
    d["sample_frac"] = d.labels / d.support_U
    rows = []
    for k, y in d.groupby(["budget", "split", "rule", "proc", "m"]):
        good, bad = y.margin_a >= EPS, y.margin_a <= -EPS
        rows.append(dict(budget=k[0], split=k[1], rule=k[2], proc=k[3], cap=k[4], cells=len(y),
                         actual_distinct_labels_mean=float(y.labels.mean()), census_cells=int(y.census.sum()),
                         census_beneficial_cells=int((y.census & good).sum()),
                         fpc_power=float(y.loc[good, "sw_fpc"].mean()) if good.any() else None,
                         fpc_false_switch_mean=float(y.loc[bad, "sw_fpc"].mean()) if bad.any() else None,
                         fpc_false_switch_worst_cell=float(y.loc[bad, "sw_fpc"].max()) if bad.any() else None,
                         nofpc_power=float(y.loc[good, "sw_nofpc"].mean()) if good.any() else None,
                         regret_fpc_audit_pp=100 * float(y.reg_fpc_audit.mean()),
                         regret_nofpc_audit_pp=100 * float(y.reg_nofpc_audit.mean()),
                         regret_nofpc_eval_pp=100 * float(y.reg_nofpc_eval.mean()),
                         regret_source_oracle_eval_pp=100 * float(y.reg_source_oracle_eval.mean()),
                         excess_target_regret_vs_source_oracle_pp=100 * float(y.excess_target_regret_vs_source_oracle.mean()),
                         negative_excess_cells=int((y.excess_target_regret_vs_source_oracle < -1e-12).sum()),
                         winner_flips=int((y.better_a != y.better_e).sum())))
    pd.DataFrame(rows).to_csv(OUT / "decision_same_policy.csv", index=False)
    b = d[d.margin_a >= EPS].copy()
    b["H_bin"] = pd.cut(np.abs(b.H_a), BINS, right=False)
    mb = b.groupby(["budget", "split", "rule", "proc", "m", "H_bin"], observed=True).agg(
        cells=("H_a", "size"), fpc_power=("sw_fpc", "mean"), sample_frac=("sample_frac", "mean"),
        census_cells=("census", "sum")).reset_index()
    mb["H_bin"] = mb.H_bin.astype(str)
    mb.to_csv(OUT / "margin_bins.csv", index=False)
    s = {r_["budget"].__str__() + "/" + r_["split"] + "/" + r_["rule"] + "/" + r_["proc"] + f"/cap={r_['cap']}": r_ for r_ in rows}
    (OUT / "summary_final.json").write_text(json.dumps(dict(plan="A11 (i)", note="caps are maximum label budgets; actual distinct labels reported; excess may be negative; source-oracle regret is a reference, not a floor", rows=s), indent=1, default=float) + "\n")
    for key in ("2500/id_test/fixed/G3/cap=1000", "2500/id_test/fixed/G3/cap=5000", "2500/id_test/fixed/G1/cap=5000"):
        print(key, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s[key].items() if k not in ("budget", "split", "rule", "proc")})


if __name__ == "__main__":
    main()

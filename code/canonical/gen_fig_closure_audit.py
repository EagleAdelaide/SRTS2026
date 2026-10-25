#!/usr/bin/env python3
"""Figure: what a small target-label audit can estimate, decide and transfer (plan A10).
n = 2500, fixed thresholds, calibrators fixed at draw 0 of each seed, 400 simulated
audits per cell, audit/evaluation pools = disjoint reviewer halves.
(a) RMSE of the estimated D = g_map - g_scalar against mean distinct labels on unseen
    reviewers (OOD validation; mean over 5 seeds x 4 maps x 3 thresholds = 60 cells)
    for raw-correctness and error-centred designs.
(b) per-cell 95% interval coverage (both populations, 120 cells): median (marker),
    10th percentile (bar end) and minimum (open marker), at m = 100 and m = 1000.
(c) within-pool decision power on seen reviewers (ID test; truth = the audit pool,
    fixed-pool variances) by margin |H| = |g_map + g_scalar|/2, for the corrected
    historical rule (G1) and the joint (D, H) test (G3) at m = 1000, 2500, 5000.
(d) transfer to the evaluation pool (seen reviewers; future-population variances):
    regret in |g| for G3 and G1, the audit-pool oracle's transfer regret, and G3's
    within-pool regret.
Reads results/www2027/posthoc_closure/{summary_closure.json, decision_by_margin.csv}.
Writes www2027/paper/figures/fig_closure_audit.pdf.  Type 42 fonts.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 7})
import matplotlib.pyplot as plt
import numpy as np

REPO = Path("__SRTS_ROOT__/calattn-repo")
IN = REPO / "results/www2027/posthoc_closure"
OUT = REPO / "www2027/paper/figures/fig_closure_audit.pdf"
DES = [("E3", "union", "#0072B2", "s", "--"), ("E9", "union, error-centred", "#0072B2", "s", "-"),
       ("E5", "importance s.", "#56B4E9", "v", "--"), ("E8", "importance s., error-c.", "#56B4E9", "v", "-"),
       ("E4", "switcher strata", "#D55E00", "D", "-"), ("E10", "active inference [ZC]", "#CC79A7", "P", "-"),
       ("E11", "LURE [K]", "#000000", "x", "-")]
COVD = [("E3", "union"), ("E9", "union e-c"), ("E4", "strata"), ("E4ac", "strata AC"), ("E8", "IS e-c"), ("E10", "AIPW")]


def main():
    S = json.load(open(IN / "summary_closure.json"))
    est = [e for e in S["estimation"] if e["budget"] == 2500 and e["rule"] == "fixed"]
    dec = [d for d in S["decisions"] if d["budget"] == 2500 and d["rule"] == "fixed" and d["split"] == "id_test"]
    bm = [r for r in csv.DictReader(open(IN / "decision_by_margin.csv"))
          if r["budget"] == "2500" and r["rule"] == "fixed" and r["split"] == "id_test" and r["C_better"] == "True"]
    fig, ax = plt.subplots(1, 4, figsize=(7.0, 1.95), gridspec_kw={"wspace": 0.55, "width_ratios": [1.05, 1.0, 1.0, 1.0]})
    a, b, c, d = ax
    for k, lab, col, mk, ls in DES:
        pts = sorted([(e["unique_labels"], e["rmse_pts"]) for e in est if e["design"] == k and e["split"] == "ood_val"])
        if pts:
            a.plot([p[0] for p in pts], [p[1] for p in pts], color=col, marker=mk, ms=2.4, lw=0.8, ls=ls, label=lab)
    a.set_xscale("log"); a.set_yscale("log")
    a.set_yticks([0.2, 0.5, 1, 2, 5]); a.set_yticklabels(["0.2", "0.5", "1", "2", "5"]); a.minorticks_off()
    a.set_xlabel("distinct labels (mean)", fontsize=6.2); a.set_ylabel("RMSE of $\\hat D$ (pp)", fontsize=6.2)
    a.set_title("(a) estimate D (unseen)", fontsize=6.6)
    # (b) per-cell coverage, pooled over the 120 cells of both populations (raw per-cell rows)
    import glob as _glob
    cells = defaultdict(list)
    for f in _glob.glob(str(IN / "est_*.csv")):
        for r in csv.DictReader(open(f)):
            if r["budget"] == "2500" and r["rule"] == "fixed":
                cells[(r["design"], int(r["m"]))].append(float(r["coverage95"]))
    for r in csv.DictReader(open(REPO / "results/www2027/posthoc_audit/audit_rows.csv")):
        if r["budget"] == "2500" and r["rule"] == "fixed" and r["design"] in ("E3", "E4"):
            cells[(r["design"], int(r["m"]))].append(float(r["coverage95"]))
    xs = np.arange(len(COVD))
    covstats = {}
    for j, (m, off, col) in enumerate(((100, -0.17, "#999999"), (1000, 0.17, "#000000"))):
        for i, (k, _) in enumerate(COVD):
            v = np.array(cells.get((k, m), []))
            if not len(v):
                continue
            med, p10, mn = float(np.median(v)), float(np.percentile(v, 10)), float(v.min())
            covstats[f"{k}/m={m}"] = dict(cells=len(v), median=med, p10=p10, min=mn)
            b.plot([i + off, i + off], [p10, med], color=col, lw=1.0)
            b.plot(i + off, med, marker="o", color=col, ms=2.6)
            b.plot(i + off, mn, marker="o", mfc="white", mec=col, ms=2.6, mew=0.7)
    (REPO / "results/www2027/posthoc_figdata").mkdir(exist_ok=True)
    (REPO / "results/www2027/posthoc_figdata/fig_closure_coverage.json").write_text(json.dumps(covstats, indent=1) + "\n")
    b.axhline(0.95, color="#D55E00", lw=0.6, ls=":")
    b.set_xticks(xs); b.set_xticklabels([l for _, l in COVD], rotation=45, ha="right", fontsize=5.4)
    b.set_ylim(-0.02, 1.02); b.set_ylabel("95% interval coverage", fontsize=6.2)
    b.set_title("(b) coverage per cell", fontsize=6.6)
    b.plot([], [], color="#999999", marker="o", ms=2.6, label="m = 100"); b.plot([], [], color="#000000", marker="o", ms=2.6, label="m = 1000")
    b.plot([], [], ls="none", marker="o", mfc="white", mec="#555555", ms=2.6, label="worst cell")
    b.legend(fontsize=5.2, frameon=False, loc="lower left", handlelength=1.0)
    # (c) power vs margin
    bins = sorted({(float(r["H_bin_lo_pts"]), float(r["H_bin_hi_pts"])) for r in bm})
    lab_bins = [f"{lo:.2f}-{hi:.2f}" if hi < 50 else f">{lo:.0f}" for lo, hi in bins]
    for proc, m, col, ls, lab in (("G3", 1000, "#E69F00", "-", "m = 1000"), ("G3", 2500, "#009E73", "-", "m = 2500"),
                                  ("G3", 5000, "#0072B2", "-", "m = 5000"), ("G1", 5000, "#0072B2", "--", "guardrail, 5000")):
        ys = []
        for lo, hi in bins:
            r = [x for x in bm if x["proc"] == proc and int(x["m"]) == m and float(x["H_bin_lo_pts"]) == lo]
            ys.append(float(r[0]["switch_rate"]) if r else np.nan)
        c.plot(range(len(bins)), ys, color=col, ls=ls, marker="o" if proc == "G3" else None, ms=2.4, lw=0.9, label=lab)
    c.set_xticks(range(len(bins))); c.set_xticklabels(lab_bins, rotation=45, ha="right", fontsize=5.2)
    c.set_xlabel("margin |H| (pp)", fontsize=6.2, labelpad=0); c.set_ylabel("power (seen, within pool)", fontsize=6.2)
    c.set_ylim(-0.02, 1.32); c.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); c.set_title("(c) decide which |g| is smaller", fontsize=6.6)
    c.legend(fontsize=5.2, frameon=False, loc="upper center", ncol=2, handlelength=1.6, labelspacing=0.2, columnspacing=0.6)
    # (d) transfer regret
    for proc, col, ls, key, lab in (("G3", "#0072B2", "-", "B_regret_pts", "joint, eval."),
                                    ("G1", "#999999", "--", "B_regret_pts", "guardrail, eval."),
                                    ("G3", "#0072B2", ":", "A_regret_pts", "joint, within")):
        R = sorted([x for x in dec if x["proc"] == proc], key=lambda x: x["m"])
        d.plot([x["m"] for x in R], [x[key] for x in R], color=col, ls=ls, marker="o" if ls == "-" else None, ms=2.4, lw=0.9, label=lab)
    orc = [x for x in dec if x["proc"] == "G3"][0]["B_oracle_transfer_regret_pts"]
    d.axhline(orc, color="#D55E00", lw=0.8, ls="-.", label="oracle, eval.")
    d.set_xscale("log"); d.set_ylim(0, 0.74); d.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    d.set_xlabel("distinct labels $m$", fontsize=6.2); d.set_ylabel("regret in |g| (pp)", fontsize=6.2)
    d.set_title("(d) transfer (seen)", fontsize=6.6)
    d.legend(fontsize=5.2, frameon=False, loc="upper center", ncol=2, handlelength=1.6, labelspacing=0.2, columnspacing=0.6)
    for x in ax:
        x.tick_params(labelsize=5.6); x.spines[["top", "right"]].set_visible(False); x.grid(alpha=0.25, lw=0.4)
    h, l = a.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=7, fontsize=5.2, frameon=False, bbox_to_anchor=(0.5, -0.04), handlelength=1.8, columnspacing=0.8)
    fig.subplots_adjust(bottom=0.33)
    fig.savefig(OUT, bbox_inches="tight", metadata={"CreationDate": None})
    print("wrote", OUT)


if __name__ == "__main__":
    main()

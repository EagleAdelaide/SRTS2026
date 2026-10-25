#!/usr/bin/env python3
"""Figure 5 (plan A11): estimate, decide, generalize (Amazon, n = 2500, fixed thresholds).
(a) RMSE of D_hat vs mean distinct labels, unseen reviewers (60 cells), five designs.
(b) Within-pool power of the joint (D,H) test G3 (finite-population variances; truth =
    the audit pool) vs the label-budget cap, by margin |H| stratum (cell counts in the
    legend); rings mark caps at which some cells of the stratum are a full census; the
    corrected guardrail G1 over all beneficial cells for comparison.
(c) Reviewer-partition sensitivity (seen reviewers): mean target regret over the 60 cells
    of the source half's full-label choice, of always keeping the scalar and of always
    switching, for 50 partitions x 2 directions; the original partition is marked.
Also writes the appendix coverage figure fig_coverage.pdf (per-cell coverage).
Reads posthoc_closure/{summary_closure.json, dec_*.csv}, posthoc_figdata/fig_closure_coverage.json,
posthoc_partition/partition_summary.json.  Type 42 fonts.
"""
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 7})
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FS, FT = 8, 7.5   # final-size regular text / ticks (final-readability revision)

REPO = Path("__SRTS_ROOT__/calattn-repo")
W = REPO / "results/www2027"
OUT = REPO / "www2027/paper/figures/fig_final_audit.pdf"
OUTC = REPO / "www2027/paper/figures/fig_coverage.pdf"
# figure-polish revision (2026-10-25): raw-correctness sampling in grey dashes, error-centred union sampling
# in solid blue, the three near-identical designs keep distinct colours and markers and are NOT separated;
# margin strata on an ordered light-to-dark scale with a marker each.  Coordinates are unchanged.
DES = [("E3", "Union, raw correctness", "#808080", "o", "--"), ("E9", "Union, error-centred", "#0072B2", "s", "-"),
       ("E4", "Switcher-stratified", "#D55E00", "D", "-"), ("E10", "Active inference", "#CC79A7", "P", "-"),
       ("E11", "LURE", "#000000", "x", "-")]
STRATA = [((0.0, 0.0025), "< 0.25", "#9ac8e0", "o"), ((0.0025, 0.005), "0.25\u20130.5", "#5ba3cf", "s"),
          ((0.005, 0.01), "0.5\u20131", "#2676b8", "^"), ((0.01, 1.0), "> 1", "#08306b", "D")]


def main():
    S = json.load(open(W / "posthoc_closure/summary_closure.json"))
    est = [e for e in S["estimation"] if e["budget"] == 2500 and e["rule"] == "fixed" and e["split"] == "ood_val"]
    d = pd.concat([pd.read_csv(f) for f in glob.glob(str(W / "posthoc_closure/dec_*.csv"))], ignore_index=True)
    d = d.query('budget == 2500 and split == "id_test" and rule == "fixed"')
    P = json.load(open(W / "posthoc_partition/partition_summary.json"))["per_partition"]
    # final-readability revision: rendered at the exact text width (506.30 pt), no bbox tightening, so
    # sizes below are final printed sizes (8 pt titles/labels/legends, 7.5 pt ticks).
    W_IN, H_IN = 506.295 / 72.27, 2.28
    fig = plt.figure(figsize=(W_IN, H_IN))
    L, B, PH, GAP = 0.50, 0.44, 1.58, 0.60
    PW = (W_IN - L - 0.06 - 2 * GAP) / 3
    a, b, c = [fig.add_axes([(L + i * (PW + GAP)) / W_IN, B / H_IN, PW / W_IN, PH / H_IN]) for i in range(3)]
    for k, lab, col, mk, ls in DES:
        pts = sorted([(e["unique_labels"], e["rmse_pts"]) for e in est if e["design"] == k])
        a.plot([p[0] for p in pts], [p[1] for p in pts], color=col, marker=mk, ms=3.6, lw=1.1, ls=ls, label=lab)
    a.set_xscale("log"); a.set_yscale("log"); a.minorticks_off()
    a.set_yticks([0.2, 0.5, 1, 2, 5]); a.set_yticklabels(["0.2", "0.5", "1", "2", "5"])
    a.set_xlabel("Mean distinct labels", fontsize=FS); a.set_ylabel("RMSE of $\\hat D$ (pp)", fontsize=FS)
    a.set_title("(a) Bias-Difference Estimation", fontsize=FS)
    a.set_ylim(0.006, 9)   # empty lower band holds the legend below every curve (display range only)
    a.legend(fontsize=FS, frameon=False, loc="lower left", ncol=1, handlelength=2.0, labelspacing=0.12,
             handletextpad=0.4, borderaxespad=0.2)
    g3 = d[(d.proc == "G3") & (d.margin_a >= 0.001)].copy(); g3["H"] = g3.H_a.abs(); g3["census"] = g3.labels >= g3.support_U
    caps = sorted(g3.m.unique())
    for (lo, hi), lab, col, mk in STRATA:
        q = g3[(g3.H >= lo) & (g3.H < hi)]
        ncell = q.groupby(["seed", "t", "method"]).ngroups
        pw = [q[q.m == m].sw_fpc.mean() for m in caps]
        b.plot(caps, pw, color=col, marker=mk, ms=3.4, lw=1.15, label=f"{lab} ({ncell})")
        for m, y in zip(caps, pw):
            if q[q.m == m].census.any():
                b.plot(m, y, marker="o", ms=8, mfc="none", mec=col, mew=0.8)
    g1 = d[(d.proc == "G1") & (d.margin_a >= 0.001)]
    b.plot(caps, [g1[g1.m == m].sw_fpc.mean() for m in caps], color="#555555", ls="--", lw=1.1, label="G1 rule (34)")
    b.set_xscale("log"); b.set_ylim(-0.02, 1.02); b.set_xlim(9, 7000)
    b.set_xlabel("Label-budget cap $m$", fontsize=FS); b.set_ylabel("Mean power (audit pool)", fontsize=FS)
    b.set_title("(b) Calibrator-Selection Power", fontsize=FS)
    b.plot([], [], ls="none", marker="o", ms=8, mfc="none", mec="#555555", mew=0.8, label="Full census")
    b.legend(fontsize=FS, frameon=False, loc="upper left", handlelength=1.6, labelspacing=0.15, handletextpad=0.4,
             borderaxespad=0.2, title="G3 by margin $|H|$\n(pp; cells)", title_fontsize=FS, alignment="left")
    keys = [(k, dr) for k in range(1, 51) for dr in ("0->1", "1->0")]
    series = [("regret_src_opt_pp", "Source-pool\nchoice", "#0072B2"), ("regret_always_scalar_pp", "Always\nscalar", "#999999"),
              ("regret_always_map_pp", "Always\nswitch", "#D55E00")]
    rng = np.random.default_rng(0)
    for i, (key, lab, col) in enumerate(series):
        v = np.array([P[f"{k}/{dr}/2500/id_test"][key] for k, dr in keys])
        c.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=6, color=col, alpha=0.5, lw=0)   # categorical jitter only
        c.plot([i - 0.27, i + 0.27], [np.median(v)] * 2, color="black", lw=1.3)
        for dr in ("0->1", "1->0"):
            c.plot(i, P[f"0/{dr}/2500/id_test"][key], marker="D", ms=4.6, mfc="black", mec="white", mew=0.6)
    c.set_xticks(range(3)); c.set_xticklabels([s[1] for s in series], fontsize=FT)
    c.set_ylabel("Target-half regret (pp)", fontsize=FS); c.set_ylim(0, 0.92); c.set_xlim(-0.5, 2.5)
    c.set_title("(c) Reviewer-Partition Sensitivity", fontsize=FS)
    c.plot([], [], ls="none", marker="D", ms=4.6, mfc="black", mec="white", label="Original partition"); c.plot([], [], color="black", lw=1.3, label="Median")
    c.legend(fontsize=FS, frameon=False, loc="upper right", handletextpad=0.4, handlelength=1.4, borderaxespad=0.1, labelspacing=0.15)
    for x in (a, b, c):
        x.tick_params(labelsize=FT, length=2.5, pad=1.5); x.spines[["top", "right"]].set_visible(False); x.grid(alpha=0.25, lw=0.4)
    fig.savefig(OUT, metadata={"CreationDate": None})
    # artifact-only: per-cell coverage
    cv = json.load(open(W / "posthoc_figdata/fig_closure_coverage.json"))
    COVD = [("E3", "union, raw"), ("E9", "union, e-c"), ("E4", "strata"), ("E4ac", "strata, AC"), ("E8", "IS, e-c"), ("E10", "active inf.")]
    fig2, ax = plt.subplots(figsize=(3.33, 1.7))
    for m, off, col in ((100, -0.15, "#999999"), (1000, 0.15, "#000000")):
        for i, (k, _) in enumerate(COVD):
            e = cv[f"{k}/m={m}"]
            ax.plot([i + off] * 2, [e["p10"], e["median"]], color=col, lw=1.0)
            ax.plot(i + off, e["median"], marker="o", color=col, ms=3)
            ax.plot(i + off, e["min"], marker="v", color=col, ms=3.4, ls="none")   # worst cell: solid triangle
    ax.axhline(0.95, color="#D55E00", lw=0.6, ls=":")
    ax.set_xticks(range(len(COVD))); ax.set_xticklabels([l for _, l in COVD], fontsize=6, rotation=30, ha="right")
    ax.set_ylabel("Empirical coverage of\nnominal 95% intervals", fontsize=6.6, linespacing=1.0); ax.set_ylim(-0.02, 1.02); ax.tick_params(labelsize=6)
    ax.plot([], [], color="#999999", marker="o", ms=3, label="m = 100"); ax.plot([], [], color="#000000", marker="o", ms=3, label="m = 1000")
    ax.plot([], [], ls="none", marker="v", color="#555555", ms=3.4, label="worst cell")
    ax.legend(fontsize=5.6, frameon=False, loc="lower right"); ax.spines[["top", "right"]].set_visible(False)
    fig2.savefig(OUTC, bbox_inches="tight", metadata={"CreationDate": None})
    print("wrote", OUT, OUTC)


if __name__ == "__main__":
    main()

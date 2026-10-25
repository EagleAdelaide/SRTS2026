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
DES = [("E3", "union, raw", "#0072B2", "s", "--"), ("E9", "union, errors", "#0072B2", "s", "-"),
       ("E4", "strata", "#D55E00", "D", "-"), ("E10", "active inf.", "#CC79A7", "P", "-"),
       ("E11", "LURE", "#000000", "x", "-")]
STRATA = [((0.0, 0.0025), "< 0.25", "#E69F00"), ((0.0025, 0.005), "0.25\u20130.5", "#009E73"),
          ((0.005, 0.01), "0.5\u20131", "#56B4E9"), ((0.01, 1.0), "> 1", "#0072B2")]


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
    a.set_xlabel("distinct labels (mean)", fontsize=FS); a.set_ylabel("RMSE of $\\hat D$ (pp)", fontsize=FS)
    a.set_title("(a) Estimate", fontsize=FS)
    a.set_ylim(0.016, 9)
    a.legend(fontsize=FS, frameon=False, loc="lower left", ncol=2, handlelength=1.8, labelspacing=0.15, columnspacing=0.7,
             handletextpad=0.4, borderaxespad=0.2)
    g3 = d[(d.proc == "G3") & (d.margin_a >= 0.001)].copy(); g3["H"] = g3.H_a.abs(); g3["census"] = g3.labels >= g3.support_U
    caps = sorted(g3.m.unique())
    for (lo, hi), lab, col in STRATA:
        q = g3[(g3.H >= lo) & (g3.H < hi)]
        ncell = q.groupby(["seed", "t", "method"]).ngroups
        pw = [q[q.m == m].sw_fpc.mean() for m in caps]
        b.plot(caps, pw, color=col, marker="o", ms=3.4, lw=1.1, label=f"{lab} ({ncell})")
        for m, y in zip(caps, pw):
            if q[q.m == m].census.any():
                b.plot(m, y, marker="o", ms=8, mfc="none", mec=col, mew=0.9)
    g1 = d[(d.proc == "G1") & (d.margin_a >= 0.001)]
    b.plot(caps, [g1[g1.m == m].sw_fpc.mean() for m in caps], color="#555555", ls="--", lw=1.1, label="guardrail (34)")
    b.set_xscale("log"); b.set_ylim(-0.02, 1.02); b.set_xlim(9, 7000)
    b.set_xlabel("label-budget cap $m$", fontsize=FS); b.set_ylabel("power in the audit pool", fontsize=FS)
    b.set_title("(b) Decide", fontsize=FS)
    b.plot([], [], ls="none", marker="o", ms=8, mfc="none", mec="#555555", mew=0.9, label="census (ring)")
    b.legend(fontsize=FS, frameon=False, loc="upper left", handlelength=1.6, labelspacing=0.15, handletextpad=0.4,
             borderaxespad=0.2, title="$|H|$ margin, pp (cells)", title_fontsize=FS)
    keys = [(k, dr) for k in range(1, 51) for dr in ("0->1", "1->0")]
    series = [("regret_src_opt_pp", "source\nchoice", "#0072B2"), ("regret_always_scalar_pp", "always\nscalar", "#999999"),
              ("regret_always_map_pp", "always\nswitch", "#D55E00")]
    rng = np.random.default_rng(0)
    for i, (key, lab, col) in enumerate(series):
        v = np.array([P[f"{k}/{dr}/2500/id_test"][key] for k, dr in keys])
        c.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=5, color=col, alpha=0.55, lw=0)
        c.plot([i - 0.27, i + 0.27], [np.median(v)] * 2, color="black", lw=1.3)
        for dr in ("0->1", "1->0"):
            c.plot(i, P[f"0/{dr}/2500/id_test"][key], marker="D", ms=4.6, mfc="black", mec="white", mew=0.6)
    c.set_xticks(range(3)); c.set_xticklabels([s[1] for s in series], fontsize=FT)
    c.set_ylabel("target-half regret (pp)", fontsize=FS); c.set_ylim(0, 0.92); c.set_xlim(-0.5, 2.5)
    c.set_title("(c) Partition sensitivity", fontsize=FS)
    c.plot([], [], ls="none", marker="D", ms=4.6, mfc="black", mec="white", label="original split"); c.plot([], [], color="black", lw=1.3, label="median")
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
    ax.set_ylabel("95% interval coverage", fontsize=6.6); ax.set_ylim(-0.02, 1.02); ax.tick_params(labelsize=6)
    ax.plot([], [], color="#999999", marker="o", ms=3, label="m = 100"); ax.plot([], [], color="#000000", marker="o", ms=3, label="m = 1000")
    ax.plot([], [], ls="none", marker="v", color="#555555", ms=3.4, label="worst cell")
    ax.legend(fontsize=5.6, frameon=False, loc="lower right"); ax.spines[["top", "right"]].set_visible(False)
    fig2.savefig(OUTC, bbox_inches="tight", metadata={"CreationDate": None})
    print("wrote", OUT, OUTC)


if __name__ == "__main__":
    main()

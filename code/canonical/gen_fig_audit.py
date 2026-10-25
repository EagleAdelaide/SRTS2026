#!/usr/bin/env python3
"""Figure: audit labels versus estimation / decision quality (plans A8 iii, A9).
(a) RMSE of the paired gap-difference estimate on unseen reviewers (OOD
validation, n = 2500, fixed thresholds; mean over seeds, maps, thresholds)
against the mean number of unique labels, for the seven designs (E1-E4 of
A8; E5-E7 of A9: active-testing IS, active difference estimator,
stratified difference estimator); (b) guardrail policy (A8, unchanged):
switch and false-switch rates vs m on seen and unseen reviewers, with the
share of cells where the map is better on the evaluation pool.  Type 42
fonts.  Reads results/www2027/posthoc_audit/summary.json.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 7})
import matplotlib.pyplot as plt

REPO = Path("__SRTS_ROOT__/calattn-repo")
SRC = REPO / "results/www2027/posthoc_audit/summary.json"
OUT = REPO / "www2027/paper/figures/fig_audit.pdf"
DES = [("E1", "uniform stream", "#999999", "o"), ("E2", "separate sets", "#009E73", "^"),
       ("E3", "union", "#0072B2", "s"), ("E5", "active IS", "#56B4E9", "v"),
       ("E6", "active diff.", "#CC79A7", "P"), ("E4", "switcher-strat.", "#D55E00", "D"),
       ("E7", "switcher-strat. + diff.", "#000000", "x")]
SPL = [("ood_val", "unseen", "-"), ("id_test", "seen", "--")]


def main():
    s = json.load(open(SRC))
    eff = [e for e in s["efficiency"] if e["budget"] == 2500 and e["rule"] == "fixed"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(3.33, 1.85), gridspec_kw={"wspace": 0.55})
    for d, lab, col, mk in DES:
        pts = sorted([(e["unique_labels_mean"], e["rmse_mean_pts"]) for e in eff if e["design"] == d and e["split"] == "ood_val"])
        if pts:
            a.plot([p[0] for p in pts], [p[1] for p in pts], color=col, marker=mk, ms=2.8, lw=0.9, label=lab)
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("unique audit labels (mean)", fontsize=6.3)
    a.set_ylabel("RMSE of $\\hat\\Delta$ (pp)", fontsize=6.3); a.set_title("(a) paired difference, unseen", fontsize=6.6)
    a.tick_params(labelsize=6); a.spines[["top", "right"]].set_visible(False); a.grid(alpha=0.25, lw=0.5)
    pol = s["policy"]
    for sp, slab, ls in SPL:
        ms_ = sorted({int(k.split("m=")[1]) for k in pol if k.startswith(f"n=2500/{sp}/fixed/")})
        sw = [pol[f"n=2500/{sp}/fixed/m={m}"]["switch"] for m in ms_]
        fs = [pol[f"n=2500/{sp}/fixed/m={m}"]["false_switch"] for m in ms_]
        better = pol[f"n=2500/{sp}/fixed/m={ms_[0]}"]["share_C_better"]
        b.plot(ms_, sw, color="#0072B2", marker="o", ms=2.8, lw=0.9, ls=ls, label=f"switch ({slab})")
        b.plot(ms_, fs, color="#D55E00", marker="x", ms=3.2, lw=0.9, ls=ls, label=f"false switch ({slab})")
        b.axhline(better, color="#555555", lw=0.5, ls=ls)
    b.set_xscale("log"); b.set_ylim(-0.02, 1.02); b.set_xlabel("unique audit labels $m$", fontsize=6.3)
    b.set_ylabel("rate", fontsize=6.3); b.set_title("(b) guardrail policy", fontsize=6.8)
    b.tick_params(labelsize=6); b.spines[["top", "right"]].set_visible(False); b.grid(alpha=0.25, lw=0.5)
    b.legend(fontsize=5.3, frameon=False, loc="upper right", handlelength=2.1, labelspacing=0.25, borderaxespad=0.1)
    h, l = a.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=5.3, frameon=False, bbox_to_anchor=(0.5, -0.02),
               handlelength=1.4, columnspacing=0.9)
    fig.subplots_adjust(bottom=0.38)
    fig.savefig(OUT, bbox_inches="tight", metadata={"CreationDate": None})
    print("wrote", OUT)


if __name__ == "__main__":
    main()

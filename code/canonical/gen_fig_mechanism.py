#!/usr/bin/env python3
"""Figure: score-change (S) versus selection (P) accounting of the signed
accepted-set bias change, map − scalar (plan A8 ii).  Panels: seen (ID
test), unseen OOD validation, unseen confirmation; rows: 4 maps × 3
thresholds at n = 2500, fixed threshold; bars for S and P (means over
seeds × draws) with descriptive 95% seed->draw intervals on S and on P
(thin lines), a tick for S + P (the signed difference) and markers for
g_B (scalar level).  Type 42 fonts.
Reads results/www2027/posthoc_mechanism/analysis_mechanism.json.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 7})
import matplotlib.pyplot as plt
import numpy as np

REPO = Path("__SRTS_ROOT__/calattn-repo")
SRC = REPO / "results/www2027/posthoc_mechanism/analysis_mechanism.json"
OUT = REPO / "www2027/paper/figures/fig_mechanism.pdf"
MAPS = [("HTS-BCE", "HTS+BCE"), ("SRTS-BCE", "SRTS-BCE"), ("SMART+BCE", "SMART+BCE"), ("PWLinear-3", "PWLinear-3")]
TS = [0.8, 0.9, 0.95]
PANELS = [("id_test", "seen reviewers (ID test)"), ("ood_val", "unseen (OOD validation)"), ("ood_test_CONFIRMATION", "unseen (confirmation)")]
CS, CP = "#0072B2", "#D55E00"


def main():
    a = json.load(open(SRC))["summary"]
    rows = [(m, nm, t) for t in TS for m, nm in MAPS]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 1.8), sharey=True, sharex=True)
    ys = np.arange(len(rows))[::-1]
    for ax, (sp, title) in zip(axes, PANELS):
        for y, (m, nm, t) in zip(ys, rows):
            e = a[f"n=2500/{sp}/t={t}/fixed/{m}"]
            S, P = 100 * e["S"]["mean"], 100 * e["P"]["mean"]
            ax.barh(y + 0.18, S, height=0.34, color=CS, alpha=0.9)
            ax.barh(y - 0.18, P, height=0.34, color=CP, alpha=0.9)
            for q, v, dy in (("S", S, 0.18), ("P", P, -0.18)):
                lo, hi = (100 * x for x in e[q]["ci95"])
                ax.plot([lo, hi], [y + dy, y + dy], color="#222222", lw=0.6, solid_capstyle="butt")
            tot = S + P
            ax.plot([tot], [y], marker="|", color="black", ms=7, mew=1.2, ls="none")
            gB = 100 * e["gB"]["mean"]
            ax.plot([gB], [y], marker="o", mfc="white", mec="#555555", ms=3.2, ls="none")
        for k in (3.5, 7.5):
            ax.axhline(k, color="#cccccc", lw=0.5)
        ax.axvline(0, color="#777777", lw=0.7, zorder=0)
        ax.set_title(title, fontsize=7); ax.tick_params(labelsize=6.3)
        ax.grid(axis="x", alpha=0.25, lw=0.5); ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("percentage points", fontsize=6.5)
    axes[0].set_yticks(ys); axes[0].set_yticklabels([f"{nm}, t={t:.2f}" for _, nm, t in rows], fontsize=6)
    handles = [plt.Rectangle((0, 0), 1, 1, color=CS, label="S: score change on the map's accepted set"),
               plt.Rectangle((0, 0), 1, 1, color=CP, label="P: selection change under the scalar residual"),
               plt.Line2D([], [], color="#222222", lw=0.6, label="95% seed→draw interval (S, P)"),
               plt.Line2D([], [], marker="|", color="black", ls="none", ms=7, mew=1.2, label="S + P = g_map − g_scalar"),
               plt.Line2D([], [], marker="o", mfc="white", mec="#555555", ls="none", ms=3.2, label="scalar's own g(A_t)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=6.0, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(OUT, bbox_inches="tight", metadata={"CreationDate": None})
    print("wrote", OUT)


if __name__ == "__main__":
    main()

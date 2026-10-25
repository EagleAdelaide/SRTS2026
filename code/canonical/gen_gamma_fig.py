#!/usr/bin/env python3
"""Figure: direct seen-vs-unseen split contrast (plan addendum A5).
Four panels (selection rule x unseen set); rows = budget x threshold; one
marker per map (Okabe-Ito, same order and shapes as the endpoint figure);
thick bar = 95% hierarchical interval, thin bar = Bonferroni(96) interval
(primary stream); filled marker = Bonferroni interval excludes 0 in both
Monte Carlo streams, open otherwise.  x = Gamma x 100 =
(Delta_unseen - Delta_seen) x 100, Delta = |g_map| - |g_scalar|; positive =
the map's accepted-set gap widens more on unseen than on seen reviewers.
Reads posthoc_gamma/analysis_gamma.json.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})  # no Type 3 fonts
import matplotlib.pyplot as plt

REPO = Path("__SRTS_ROOT__/calattn-repo")
SRC = REPO / "results/www2027/posthoc_gamma/analysis_gamma.json"
OUT = REPO / "www2027/paper/figures/gamma_split_contrast.pdf"
METHODS = [("HTS-BCE", "#0072B2", "o"), ("SRTS-BCE", "#E69F00", "s"),
           ("SMART+BCE", "#009E73", "^"), ("PWLinear-3", "#D55E00", "D")]
ROWS = [(250, 0.80), (250, 0.90), (250, 0.95), (2500, 0.80), (2500, 0.90), (2500, 0.95)]
PANELS = [("fixed", "ood_val", "fixed t, OOD validation"),
          ("fixed", "ood_test_CONFIRMATION", "fixed t, confirmation"),
          ("matched", "ood_val", "matched cov., OOD validation"),
          ("matched", "ood_test_CONFIRMATION", "matched cov., confirmation")]


def main():
    r = json.load(open(SRC))
    tab = {(e["rule"], e["ood"], e["budget"], e["t"], e["method"]): e for e in r["contrasts"]}
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.4), sharey=True, sharex=True)
    for ax, (rule, ood, title) in zip(axes, PANELS):
        for i, (b, t) in enumerate(ROWS):
            for j, (m, col, mk) in enumerate(METHODS):
                e = tab[(rule, ood, b, t, m)]
                y = i + (j - 1.5) * 0.17
                ax.plot([100 * e["cibonf"][0], 100 * e["cibonf"][1]], [y, y], color=col,
                        lw=0.5, solid_capstyle="butt")
                ax.plot([100 * e["ci95"][0], 100 * e["ci95"][1]], [y, y], color=col,
                        lw=1.5, solid_capstyle="round")
                stable = e["cibonf_excl0"] and not e["cibonf_mc_flip"]
                ax.plot([100 * e["gamma"]], [y], marker=mk, ms=4.0, color=col,
                        mfc=col if stable else "white", mew=1.0, ls="none")
        ax.axvline(0, color="#777777", lw=0.8, zorder=0)
        ax.axhline(2.5, color="#bbbbbb", lw=0.5, zorder=0)
        ax.set_title(title, fontsize=7)
        ax.tick_params(labelsize=6.5)
        ax.grid(axis="x", alpha=0.25, lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel(r"$\Gamma$ = unseen $-$ seen ($\times$100)", fontsize=6.5)
    axes[0].set_yticks(range(len(ROWS)))
    axes[0].set_yticklabels([f"n={b}, t={t:.2f}" for b, t in ROWS], fontsize=6.5)
    axes[0].invert_yaxis()
    handles = [plt.Line2D([], [], color=c, marker=mk, ls="-", lw=1.5, ms=4.0,
                          label=m.replace("HTS-BCE", "HTS+BCE")) for m, c, mk in METHODS]
    handles.append(plt.Line2D([], [], color="#555555", marker="o", mfc="white",
                              ls="none", ms=4.0,
                              label="open: Bonferroni(96) incl. 0 in a stream"))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=6.3,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight",
                metadata={"CreationDate": None})  # no local date in the shipped PDF
    print("wrote", OUT)


if __name__ == "__main__":
    main()

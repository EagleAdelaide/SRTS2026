#!/usr/bin/env python3
"""Figure: operational endpoints vs the scalar map on both confirmation sets
(post-hoc, plan P3).  Four panels (task x budget); rows = whole-split
|count rel. error| and |gap| of the auto-accept set at t = .80/.90/.95;
one marker per method (Okabe-Ito, fixed order + distinct marker shapes);
bar = 95% hierarchical interval; filled marker = Bonferroni(64) interval
excludes 0, open marker otherwise.  x = (method - TvA-TS) x 100, negative =
smaller absolute error than the scalar.  Reads posthoc_endpoints/endpoints.json.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})  # no Type 3 fonts
import matplotlib.pyplot as plt

REPO = Path("__SRTS_ROOT__/calattn-repo")
SRC = REPO / "results/www2027/posthoc_endpoints/endpoints.json"
OUT = REPO / "www2027/paper/figures/endpoints_vs_scalar.pdf"
METHODS = [("HTS-BCE", "#0072B2", "o"), ("SRTS-BCE", "#E69F00", "s"),
           ("SMART+BCE", "#009E73", "^"), ("PWLinear-3", "#D55E00", "D")]
ROWS = [("abs_count", "|batch count err.| (%)"), ("abs_gap_80", "|g(A)|, t=.80 (pp)"),
        ("abs_gap_90", "|g(A)|, t=.90 (pp)"), ("abs_gap_95", "|g(A)|, t=.95 (pp)"),
        ("matched_80", "|g|, matched .80 (pp)"), ("matched_90", "|g|, matched .90 (pp)"),
        ("matched_95", "|g|, matched .95 (pp)")]
MATCHED = REPO / "results/www2027/posthoc_matched/analysis_matched.json"
MKEY = {"amazon": "amazon_distilbert/ood_test_CONFIRMATION", "yahoo": "yahoo_distilbert/ood_test_CONFIRMATION"}
PANELS = [("amazon", "250", "Amazon, n=250"),
          ("amazon", "2500", "Amazon, n=2500"),
          ("yahoo", "250", "Yahoo, n=250"),
          ("yahoo", "2500", "Yahoo, n=2500")]


def main():
    r = json.load(open(SRC))
    mt = json.load(open(MATCHED))["contrasts"]
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.3), sharey=True)
    for ax, (task, b, title) in zip(axes, PANELS):
        ct = r["tasks"][task]["contrasts"][b]
        for i, (q, _) in enumerate(ROWS):
            for j, (m, col, mk) in enumerate(METHODS):
                if q.startswith("matched_"):
                    c = mt[f"{MKEY[task]}/n={b}/t={q[-2:]}/{m}"]
                else:
                    c = ct[m][q]
                y = i + (j - 1.5) * 0.17
                lo, hi = (100 * c["ci95"][0], 100 * c["ci95"][1])
                ax.plot([lo, hi], [y, y], color=col, lw=1.2,
                        solid_capstyle="round")
                ax.plot([100 * c["point"]], [y], marker=mk, ms=4.2,
                        color=col, mfc=col if c["excl0_bonf"] else "white",
                        mew=1.0, ls="none")
        ax.axvline(0, color="#777777", lw=0.8, zorder=0)
        ax.axhline(3.5, color="#bbbbbb", lw=0.5, zorder=0)
        ax.set_title(title, fontsize=7)
        ax.tick_params(labelsize=6.5)
        ax.grid(axis="x", alpha=0.25, lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel(r"method $-$ scalar ($\times$100)", fontsize=6.5)
    axes[0].set_yticks(range(len(ROWS)))
    axes[0].set_yticklabels([lab for _, lab in ROWS], fontsize=6.5)
    axes[0].invert_yaxis()
    handles = [plt.Line2D([], [], color=c, marker=mk, ls="-", lw=1.2, ms=4.2,
                          label=m.replace("HTS-BCE", "HTS+BCE")) for m, c, mk in METHODS]
    handles.append(plt.Line2D([], [], color="#555555", marker="o", mfc="white",
                              ls="none", ms=4.2,
                              label="open: Bonferroni interval incl. 0 (64 fixed-t / 120 matched)"))
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=6.0,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight",
                metadata={"CreationDate": None})  # no local date in the shipped PDF
    print("wrote", OUT)


if __name__ == "__main__":
    main()

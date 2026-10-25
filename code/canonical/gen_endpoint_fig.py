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
ROWS = [("abs_count", "|Batch count error| (%)"), ("abs_gap_80", "|g|, fixed t = 0.80 (pp)"),
        ("abs_gap_90", "|g|, fixed t = 0.90 (pp)"), ("abs_gap_95", "|g|, fixed t = 0.95 (pp)"),
        ("matched_80", "|g|, matched 0.80 (pp)"), ("matched_90", "|g|, matched 0.90 (pp)"),
        ("matched_95", "|g|, matched 0.95 (pp)")]
MATCHED = REPO / "results/www2027/posthoc_matched/analysis_matched.json"
MKEY = {"amazon": "amazon_distilbert/ood_test_CONFIRMATION", "yahoo": "yahoo_distilbert/ood_test_CONFIRMATION"}
PANELS = [("amazon", "250", "Amazon, n=250"),
          ("amazon", "2500", "Amazon, n=2500"),
          ("yahoo", "250", "Yahoo, n=250"),
          ("yahoo", "2500", "Yahoo, n=2500")]


def main():
    r = json.load(open(SRC))
    mt = json.load(open(MATCHED))["contrasts"]
    # figure-polish revision (2026-10-25): rendered at the exact text width (506.30 pt) with the aspect ratio
    # of the earlier file (496.49 x 165.75 pt), no bbox tightening, so sizes are final (8 pt text, 7.5 pt ticks);
    # one shared x label; thinner lines; within-row method offsets 0.20 of a row (disclosed in the caption);
    # marker fill (filled = Bonferroni interval excludes 0) unchanged.
    FS, FT = 8, 7.5
    W_IN = 506.295 / 72.27
    H_IN = W_IN * 165.75 / 496.49
    L, R, GAP, BOT, TOP = 1.50, 0.05, 0.15, 0.66, 0.21
    PW, PH = (W_IN - L - R - 3 * GAP) / 4, H_IN - BOT - TOP
    fig = plt.figure(figsize=(W_IN, H_IN))
    axes = []
    for k in range(4):
        axes.append(fig.add_axes([(L + k * (PW + GAP)) / W_IN, BOT / H_IN, PW / W_IN, PH / H_IN],
                                 sharey=axes[0] if axes else None))
    for ax, (task, b, title) in zip(axes, PANELS):
        ct = r["tasks"][task]["contrasts"][b]
        for i, (q, _) in enumerate(ROWS):
            for j, (m, col, mk) in enumerate(METHODS):
                if q.startswith("matched_"):
                    c = mt[f"{MKEY[task]}/n={b}/t={q[-2:]}/{m}"]
                else:
                    c = ct[m][q]
                y = i + (j - 1.5) * 0.20
                lo, hi = (100 * c["ci95"][0], 100 * c["ci95"][1])
                ax.plot([lo, hi], [y, y], color=col, lw=0.9, solid_capstyle="butt")
                ax.plot([100 * c["point"]], [y], marker=mk, ms=3.3,
                        color=col, mfc=col if c["excl0_bonf"] else "white",
                        mew=0.8, ls="none")
        ax.axvline(0, color="#666666", lw=0.7, zorder=0)
        ax.axhline(3.5, color="#bbbbbb", lw=0.5, zorder=0)
        ax.set_title(title.replace("n=", "$n$ = "), fontsize=FS, pad=3)
        ax.tick_params(labelsize=FT, length=2.5, pad=1.5)
        ax.grid(axis="x", alpha=0.25, lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(range(len(ROWS)))
    axes[0].set_yticklabels([lab for _, lab in ROWS], fontsize=FS)
    axes[0].tick_params(axis="y", length=0, pad=3)
    for ax in axes[1:]:
        ax.tick_params(axis="y", labelleft=False, length=0)
    axes[0].set_ylim(len(ROWS) - 0.5, -0.5)   # same rows as before, top to bottom
    fig.text((L + (W_IN - L - R) / 2) / W_IN, 0.24 / H_IN, "Method $-$ scalar TvA-TS ($\\times$100)",
             ha="center", va="bottom", fontsize=FS)
    handles = [plt.Line2D([], [], color=c, marker=mk, ls="-", lw=0.9, ms=3.3,
                          label=m.replace("HTS-BCE", "HTS+BCE")) for m, c, mk in METHODS]
    handles.append(plt.Line2D([], [], color="#555555", marker="o", mfc="white",
                              ls="none", ms=3.3, mew=0.8, label="Open marker: Bonferroni interval includes 0"))
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=FS, frameon=False,
               bbox_to_anchor=(0.5, 0.0), handlelength=1.8, columnspacing=1.4,
               borderaxespad=0.1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, metadata={"CreationDate": None})  # no local date in the shipped PDF
    print("wrote", OUT)


if __name__ == "__main__":
    main()

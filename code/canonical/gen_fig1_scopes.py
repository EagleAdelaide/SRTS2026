#!/usr/bin/env python3
"""Figure 1 (figure-editorial revision, 2026-10-22): two evaluation scopes and one
illustrative endpoint reversal.  No new computation: reads the stored descriptive summary
results/www2027/posthoc_figdata/fig1_reversal.json written by gen_fig1_reversal.py (plan A7 i;
Amazon confirmation set = unseen reviewers, n = 2500, t = 0.90; means over 5 seeds x 20
draws with 95% seed->draw bootstrap intervals, B = 20000).
Top: confidence-ordered strip of all scored items; c >= t is auto-processed (accepted set
A_t), c < t is deferred; brackets mark the two scopes (all items: ECE and whole-batch
error-count error; accepted only: mean bias g(A_t)).
Bottom: scalar TvA-TS and SRTS-BCE only (all four conditional maps are in Figure 3), levels
of ECE15 (%), |relative whole-batch error-count error| (%) and signed g(A_0.90) (pp), with
direct numerical labels.  Solid markers only (marker fill carries no meaning).
Rendered at the final column width (241.15 pt = 3.337 in), so font sizes are final sizes.
Writes www2027/paper/figures/fig1_scopes.pdf.  Type 42 fonts.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8,
                            "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

REPO = Path("__SRTS_ROOT__/calattn-repo")
SRC = REPO / "results/www2027/posthoc_figdata/fig1_reversal.json"
FIG = REPO / "www2027/paper/figures/fig1_scopes.pdf"
W_IN, H_IN = 241.14749 / 72.27, 2.16
ROWS = [("TvA-TS", "TvA-TS", "#000000", "o"), ("SRTS-BCE", "SRTS-BCE", "#E69F00", "s")]
PANELS = [("ece15", "ECE$_{15}$ (%)", "{:.2f}", "← better"),
          ("abs_count", "|batch count\nerror| (%)", "{:.1f}", "← better"),
          ("gap90", "accepted-set\nbias (pp)", "{:+.1f}", "0 = zero mean bias")]
FS, FT = 8, 7.5
ACC, DEF, INK = "#cfe3f3", "#ececec", "#333333"


def schematic(fig):
    top = 0.60                                   # inches occupied at the top of the figure
    ax = fig.add_axes([0, 1 - top / H_IN, 1, top / H_IN]); ax.set_axis_off()
    ax.set_xlim(0, W_IN); ax.set_ylim(0, top)
    x0, x1, xt = 0.06, W_IN - 0.06, 0.06 + 0.58 * (W_IN - 0.12)   # schematic, not to scale
    yb, hb = 0.205, 0.165
    ax.add_patch(Rectangle((x0, yb), xt - x0, hb, fc=DEF, ec="#9a9a9a", lw=0.6))
    ax.add_patch(Rectangle((xt, yb), x1 - xt, hb, fc=ACC, ec="#2b6a99", lw=0.8))
    ax.text((x0 + xt) / 2, yb + hb / 2, "$c<t$: defer to review", ha="center", va="center", fontsize=FS, color="#555555")
    ax.text((xt + x1) / 2, yb + hb / 2, "$c\\geq t$: auto-process", ha="center", va="center", fontsize=FS, color="#123a5a")
    # scope 1: all scored items (bracket above the whole strip)
    yt = yb + hb + 0.05
    ax.plot([x0, x0, x1, x1], [yt - 0.025, yt, yt, yt - 0.025], color=INK, lw=0.7)
    ax.text(x0, yt + 0.03, "All scored items: ECE, batch error-count error", ha="left", va="bottom", fontsize=FS)
    # scope 2: accepted set only (bracket below the accepted part)
    ybb = yb - 0.05
    ax.plot([xt, xt, x1, x1], [ybb + 0.025, ybb, ybb, ybb + 0.025], color="#2b6a99", lw=0.8)
    ax.text(x1, ybb - 0.03, "Accepted set $A_t$ only: mean bias $g(A_t)$", ha="right", va="top", fontsize=FS, color="#123a5a")


def main():
    d = json.load(open(SRC))["levels"]
    fig = plt.figure(figsize=(W_IN, H_IN))
    schematic(fig)
    left, right, bottom, top = 0.62, 0.07, 0.34, 0.60 + 0.40
    gap = 0.17
    pw = (W_IN - left - right - 2 * gap) / 3
    ys = [1, 0]
    for j, (q, title, fmt, xlab) in enumerate(PANELS):
        ax = fig.add_axes([(left + j * (pw + gap)) / W_IN, bottom / H_IN, pw / W_IN, (H_IN - top - bottom) / H_IN])
        lo_all = min(d[m][q]["ci95"][0] for m, *_ in ROWS); hi_all = max(d[m][q]["ci95"][1] for m, *_ in ROWS)
        if q == "gap90":
            lo_all = min(lo_all, 0.0)
            ax.axvline(0, color="#777777", lw=0.7, zorder=0)
        pad = 0.18 * (hi_all - lo_all)
        ax.set_xlim(lo_all - pad, hi_all + pad)
        for (m, nm, col, mk), y in zip(ROWS, ys):
            v = d[m][q]
            ax.plot(v["ci95"], [y, y], color=col, lw=1.3, solid_capstyle="round")
            ax.plot([v["mean"]], [y], marker=mk, ms=4.6, color=col, mec=col, ls="none")
            ax.text(v["mean"], y + 0.2, fmt.format(v["mean"]), ha="center", va="bottom", fontsize=FT, color=col)
        ax.set_ylim(-0.55, 1.75); ax.set_yticks(ys)
        ax.set_yticklabels([nm for _, nm, *_ in ROWS] if j == 0 else [], fontsize=FS)
        ax.tick_params(axis="x", labelsize=FT, length=2.5, pad=1.5); ax.tick_params(axis="y", length=0, pad=3)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
        ax.grid(axis="x", alpha=0.3, lw=0.5); ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(title, fontsize=FS, pad=2.5, linespacing=1.05)
        ax.set_xlabel(xlab, fontsize=FT, labelpad=1.5)
        if q == "gap90":
            ax.xaxis.label.set_horizontalalignment("right"); ax.xaxis.label.set_x(1.04)
    fig.savefig(FIG, metadata={"CreationDate": None})
    print("wrote", FIG)
    for m, *_ in ROWS:
        print(m, {q: round(d[m][q]["mean"], 3) for q, *_ in PANELS}, "cov90", round(d[m]["cov90"]["mean"], 1))


if __name__ == "__main__":
    main()

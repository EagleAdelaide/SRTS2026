#!/usr/bin/env python3
"""Figure 3 (figure-editorial revision, 2026-10-22): operational effects by task at the
highlighted operating point n = 2500, t = 0.90.  No new computation.
Columns: Amazon unseen reviewers (confirmation set), Yahoo official test (confirmation set),
CivilComments (external check).  Rows: change in |relative whole-batch error-count error|
and change in mean |accepted-set bias g(A_0.90)|, each map minus scalar TvA-TS, in
percentage points; negative favours the conditional map.  Four methods in a fixed order,
one row each; solid markers and descriptive 95% seed->draw intervals (no significance
encoding; corrected inference stays in the text and the supplementary Figure).
Inputs and units (converted once):
  results/www2027/posthoc_endpoints/endpoints.json  tasks[task]["contrasts"]["2500"][map]
      ["abs_count" | "abs_gap_90"]["point", "ci95"]  -> fractions, x100
  results/www2027/civil/analysis.json  contrasts["n=2500/<map>/abs_count" | ".../abs_g90"]
      ["mean", "ci95"]                                -> already pp, no scaling
Each value is the mean of per-draw |g_C| - |g_B| (absolute values taken before averaging).
Panel-specific horizontal ranges (disclosed in the caption).  Rendered at the final text
width (506.30 pt = 7.006 in).  Writes www2027/paper/figures/fig3_tasks.pdf and
results/www2027/posthoc_figdata/fig3_tasks.json.  Type 42 fonts.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8,
                            "axes.linewidth": 0.6, "xtick.major.width": 0.6})
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

REPO = Path("__SRTS_ROOT__/calattn-repo")
W = REPO / "results/www2027"
FIG = REPO / "www2027/paper/figures/fig3_tasks.pdf"
OUTJ = W / "posthoc_figdata/fig3_tasks.json"
W_IN, H_IN = 506.295 / 72.27, 2.66
METHODS = [("HTS-BCE", "HTS+BCE", "#0072B2", "o"), ("SRTS-BCE", "SRTS-BCE", "#E69F00", "s"),
           ("SMART+BCE", "SMART+BCE", "#009E73", "^"), ("PWLinear-3", "PWLinear-3", "#D55E00", "D")]
TASKS = [("amazon", "Amazon: unseen reviewers"), ("yahoo", "Yahoo: official test"),
         ("civil", "CivilComments: external check")]
ROWS = [("abs_count", "Change in |relative whole-batch error-count error| (pp)"),
        ("abs_gap_90", "Change in mean |accepted-set bias| at $t{=}0.90$ (pp)")]
FS, FT = 8, 7.5


def load():
    ep = json.load(open(W / "posthoc_endpoints/endpoints.json"))["tasks"]
    ca = json.load(open(W / "civil/analysis.json"))["contrasts"]
    out = {}
    for t, _ in TASKS:
        for q, _ in ROWS:
            for m, *_ in METHODS:
                if t == "civil":
                    c = ca[f"n=2500/{m}/" + {"abs_count": "abs_count", "abs_gap_90": "abs_g90"}[q]]
                    v = (c["mean"], c["ci95"][0], c["ci95"][1])                       # already pp
                else:
                    c = ep[t]["contrasts"]["2500"][m][q]
                    v = (100 * c["point"], 100 * c["ci95"][0], 100 * c["ci95"][1])     # fraction -> pp
                out[f"{t}/{q}/{m}"] = dict(mean_pp=v[0], ci95_pp=[v[1], v[2]])
    return out


def main():
    V = load()
    OUTJ.write_text(json.dumps(dict(note="n=2500, t=0.90; map minus TvA-TS; pp; descriptive 95% seed->draw intervals; "
                                         "Amazon/Yahoo: posthoc_endpoints/endpoints.json x100; CivilComments: civil/analysis.json (pp)",
                                    values=V), indent=1) + "\n")
    fig = plt.figure(figsize=(W_IN, H_IN))
    left, right, gap = 0.80, 0.04, 0.26
    pw = (W_IN - left - right - 2 * gap) / 3
    ph = 0.72                                     # panel height (in)
    col_y = H_IN - 0.06                           # column titles (top band)
    tops = [H_IN - 0.44, H_IN - 0.44 - ph - 0.52]   # top edge of each panel row (in)
    for j, (t, ttl) in enumerate(TASKS):
        fig.text((left + j * (pw + gap) + pw / 2) / W_IN, col_y / H_IN, ttl, ha="center", va="top", fontsize=FS,
                 fontweight="bold")
    ys = list(range(len(METHODS)))[::-1]
    for i, (q, rlab) in enumerate(ROWS):
        fig.text(0.06 / W_IN, (tops[i] + 0.07) / H_IN, rlab + ", map $-$ scalar TvA-TS; negative favours the map",
                 ha="left", va="bottom", fontsize=FS)
        for j, (t, ttl) in enumerate(TASKS):
            ax = fig.add_axes([(left + j * (pw + gap)) / W_IN, (tops[i] - ph) / H_IN, pw / W_IN, ph / H_IN])
            vals = [V[f"{t}/{q}/{m}"] for m, *_ in METHODS]
            lo = min(min(v["ci95_pp"][0] for v in vals), 0.0); hi = max(max(v["ci95_pp"][1] for v in vals), 0.0)
            span = hi - lo
            xr = hi + 0.36 * span                         # right end: a value column beyond the data range
            ax.set_xlim(lo - 0.06 * span, xr)
            ticks = [x for x in MaxNLocator(nbins=4).tick_values(lo, hi) if lo - 0.06 * span <= x <= hi + 0.02 * span]
            ax.set_xticks(ticks)
            ax.axvline(0, color="#777777", lw=0.8, zorder=0)
            for (m, nm, col, mk), y, v in zip(METHODS, ys, vals):
                ax.plot(v["ci95_pp"], [y, y], color=col, lw=1.4, solid_capstyle="round")
                ax.plot([v["mean_pp"]], [y], marker=mk, ms=4.6, color=col, mec=col, ls="none")
                ax.text(xr, y, f"{v['mean_pp']:+.2f}".replace("-", "\u2212"), ha="right", va="center", fontsize=FT, color=col)
            ax.set_ylim(-0.6, len(METHODS) - 0.4); ax.set_yticks(ys)
            ax.set_yticklabels([nm for _, nm, *_ in METHODS] if j == 0 else [], fontsize=FS)
            ax.tick_params(axis="y", length=0, pad=3); ax.tick_params(axis="x", labelsize=FT, length=2.5, pad=1.5)
            ax.grid(axis="x", alpha=0.3, lw=0.5); ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(FIG, metadata={"CreationDate": None})
    print("wrote", FIG)
    for k, v in V.items():
        print(f"{k:28s} {v['mean_pp']:+.4f} [{v['ci95_pp'][0]:+.4f}, {v['ci95_pp'][1]:+.4f}]")


if __name__ == "__main__":
    main()

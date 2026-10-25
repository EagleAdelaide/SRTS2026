#!/usr/bin/env python3
"""Figure 1 (plan addendum A7 i): the metric reversal on the Amazon
confirmation set at n = 2500, t = 0.90.  Top: schematic of the offline
confidence-routing simulation.  Bottom: three panels (ECE15, whole-batch
|count error|, signed accepted-set bias g(A_.90)), one row per map, mean
over seeds and draws with descriptive 95% seed->draw hierarchical bootstrap
intervals (B = 20000, RNG 20261016).  Reads CONFIRMATION/amazon rows;
writes results/www2027/posthoc_figdata/fig1_reversal.json and
www2027/paper/figures/fig1_reversal.pdf.  Type 42 fonts.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 7})
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

REPO = Path("__SRTS_ROOT__/calattn-repo")
W = REPO / "results/www2027"
OUTD = W / "posthoc_figdata"
FIG = REPO / "www2027/paper/figures/fig1_reversal.pdf"
B, RNG = 20000, 20261016
MAPS = [("TvA-TS", "scalar TvA-TS", "#000000", "o"), ("HTS-BCE", "HTS+BCE", "#0072B2", "o"),
        ("SRTS-BCE", "SRTS-BCE", "#E69F00", "s"), ("SMART+BCE", "SMART+BCE", "#009E73", "^"),
        ("PWLinear-3", "PWLinear-3", "#D55E00", "D")]
Q = [("ece15", lambda r: r["ece15"], "ECE$_{15}$ (pp)"),
     ("abs_count", lambda r: 100 * abs(r["count_rel_err"]), "|batch count err.| (%)"),
     ("gap90", lambda r: 100 * r["errgap_a90"], "$g(A_{0.90})$ (pp)")]


def hier(A, i_s, i_d):
    return np.nanmean(np.nanmean(A[i_s[:, :, None], i_d], axis=2), axis=1)


def main():
    rows = [json.loads(l) for l in open(W / "CONFIRMATION/amazon_distilbert.jsonl")]
    rows = [r for r in rows if r["budget"] == 2500]
    seeds, draws = sorted({r["seed"] for r in rows}), sorted({r["draw"] for r in rows})
    tab = {(r["seed"], r["draw"], r["method"]): r for r in rows}
    S, D = len(seeds), len(draws)
    rng = np.random.default_rng(RNG)
    i_s, i_d = rng.integers(0, S, (B, S)), rng.integers(0, D, (B, S, D))
    out = {"plan": "A7(i)", "B": B, "rng": RNG, "seeds": S, "draws": D, "levels": {}, "diff_vs_scalar": {}}
    for m, *_ in MAPS:
        out["levels"][m] = {}
        out["diff_vs_scalar"][m] = {}
        for q, f, _ in Q + [("cov90", lambda r: 100 * r["cov_a90"], "")]:
            A = np.array([[f(tab[(s, d, m)]) for d in draws] for s in seeds])
            A0 = np.array([[f(tab[(s, d, "TvA-TS")]) for d in draws] for s in seeds])
            bt = hier(A, i_s, i_d)
            out["levels"][m][q] = {"mean": float(A.mean()), "ci95": [float(x) for x in np.percentile(bt, [2.5, 97.5])]}
            bd = hier(A - A0, i_s, i_d)
            out["diff_vs_scalar"][m][q] = {"mean": float((A - A0).mean()),
                                          "ci95": [float(x) for x in np.percentile(bd, [2.5, 97.5])]}
    OUTD.mkdir(exist_ok=True)
    (OUTD / "fig1_reversal.json").write_text(json.dumps(out, indent=1) + "\n")

    fig = plt.figure(figsize=(3.33, 2.25))
    gs = fig.add_gridspec(1, 3, wspace=0.14, left=0.19, right=0.985, top=0.62, bottom=0.17)
    # --- schematic (offline simulation; the caption states that no system was deployed)
    ax = fig.add_axes([0.01, 0.71, 0.98, 0.28]); ax.set_axis_off(); ax.set_xlim(0, 10.4); ax.set_ylim(-0.05, 2.1)

    def box(x, y, w, h, txt, fc="#f2f2f2"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04", fc=fc, ec="#444444", lw=0.5))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=5.3, linespacing=1.1)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=6, lw=0.5, color="#444444"))
    box(0.0, 0.55, 2.3, 0.9, "user-generated\nitems"); arrow(2.3, 1.0, 2.75, 1.0)
    box(2.75, 0.55, 2.75, 0.9, "frozen classifier\n+ calibrator: $c$")
    arrow(5.5, 1.0, 6.2, 1.0); ax.text(5.85, 1.12, "$c\\geq t$", ha="center", va="bottom", fontsize=5.3)
    box(6.2, 1.1, 4.2, 0.85, "accepted automatically: set $A_t$\nbias $g(A_t)=\\mathrm{mean}(c-b)$", fc="#e3f0fa")
    box(6.2, 0.05, 4.2, 0.85, "deferred to human review", fc="#fbeee3")
    arrow(6.2, 1.0, 6.2, 1.5); arrow(6.2, 1.0, 6.2, 0.5)
    # --- three level panels
    ys = np.arange(len(MAPS))[::-1]
    for j, (q, _, lab) in enumerate(Q):
        ax = fig.add_subplot(gs[0, j])
        for k, (m, nm, col, mk) in enumerate(MAPS):
            v = out["levels"][m][q]
            ax.plot(v["ci95"], [ys[k]] * 2, color=col, lw=1.1, solid_capstyle="round")
            ax.plot([v["mean"]], [ys[k]], marker=mk, ms=3.6, color=col, mfc=col if m in ("TvA-TS", "SRTS-BCE") else "white", mew=0.9, ls="none")
        if q == "gap90":
            ax.axvline(0, color="#777777", lw=0.7, zorder=0)
            ax.set_xlim(left=min(-0.3, ax.get_xlim()[0]))
        ax.set_yticks(ys); ax.set_ylim(-0.6, len(MAPS) - 0.4)
        ax.set_yticklabels([nm for _, nm, *_ in MAPS] if j == 0 else [], fontsize=6.3)
        ax.tick_params(axis="x", labelsize=6); ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", alpha=0.25, lw=0.5); ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(lab, fontsize=6.4, pad=3)
        ax.set_xlabel("0 = calibrated" if q == "gap90" else "← better", fontsize=5.8, labelpad=1)
    fig.savefig(FIG, metadata={"CreationDate": None})
    print("wrote", FIG)
    for m, *_ in MAPS:
        print(m, {q: (round(out["levels"][m][q]["mean"], 2), [round(x, 2) for x in out["levels"][m][q]["ci95"]]) for q in ("ece15", "abs_count", "gap90", "cov90")})


if __name__ == "__main__":
    main()

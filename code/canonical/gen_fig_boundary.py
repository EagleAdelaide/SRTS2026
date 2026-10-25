#!/usr/bin/env python3
"""Figure: where a confidence shift narrows the accepted-set bias (plans A8 ii, A9 note 3).
(a) D = g_map - g_scalar (= S + P) against the scalar's own g_B for every map, threshold and
split at n = 2500, fixed thresholds; |g| narrows exactly in the shaded wedge g_B < -D/2
(for D > 0).  Error bars: descriptive 95% seed->draw hierarchical intervals (5 seeds x 20
draws; B = 20000, RNG 20261016) of each coordinate.
(b) share of the selection term in the forward order |P|/(|S|+|P|) against the reverse order
|P'|/(|S'|+|P'|) for every cell (both rules), n = 2500 filled, n = 250 open.
Reads posthoc_mechanism/{analysis_mechanism.json, reverse_accounting.json, mechanism_rows.csv}.
Writes www2027/paper/figures/fig_boundary.pdf and posthoc_figdata/fig_boundary.json.  Type 42 fonts.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 7})
import matplotlib.pyplot as plt
import numpy as np

REPO = Path("__SRTS_ROOT__/calattn-repo")
M = REPO / "results/www2027/posthoc_mechanism"
OUT = REPO / "www2027/paper/figures/fig_boundary.pdf"
FD = REPO / "results/www2027/posthoc_figdata/fig_boundary.json"
SPL = [("id_test", "seen (ID test)", "#0072B2"), ("ood_val", "unseen (OOD val.)", "#E69F00"),
       ("ood_test_CONFIRMATION", "unseen (confirmation)", "#D55E00")]
TMK = {0.8: "o", 0.9: "s", 0.95: "^"}
MAPS = ["HTS-BCE", "SRTS-BCE", "SMART+BCE", "PWLinear-3"]
B, RNG = 20000, 20261016


def hier(A, i_s, i_d):
    return np.nanmean(np.nanmean(A[i_s[:, :, None], i_d], axis=2), axis=1)


def main():
    an = json.load(open(M / "analysis_mechanism.json"))["summary"]
    ra = json.load(open(M / "reverse_accounting.json"))["summary"]
    # D intervals from the per-draw rows (n = 2500, fixed)
    per = defaultdict(dict)
    for r in csv.DictReader(open(M / "mechanism_rows.csv")):
        if r["budget"] == "2500" and r["rule"] == "fixed":
            per[(r["split"], float(r["t"]), r["method"])][(int(r["seed"]), int(r["draw"]))] = float(r["gC"]) - float(r["gB"])
    seeds = sorted({k[0] for v in per.values() for k in v}); draws = sorted({k[1] for v in per.values() for k in v})
    rng = np.random.default_rng(RNG); i_s, i_d = rng.integers(0, len(seeds), (B, len(seeds))), rng.integers(0, len(draws), (B, len(seeds), len(draws)))
    pts = []
    for sp, lab, col in SPL:
        for t in (0.8, 0.9, 0.95):
            for m in MAPS:
                e = an[f"n=2500/{sp}/t={t}/fixed/{m}"]
                A = np.array([[per[(sp, t, m)][(s, d)] for d in draws] for s in seeds])
                ci = np.percentile(hier(A, i_s, i_d), [2.5, 97.5])
                pts.append(dict(split=sp, t=t, map=m, gB=e["gB"]["mean"], gB_ci=e["gB"]["ci95"], D=float(A.mean()), D_ci=[float(x) for x in ci],
                                narrows=bool(abs(e["gB"]["mean"] + A.mean()) < abs(e["gB"]["mean"]))))
    sh = [dict(budget=int(k.split("/")[0][2:]), key=k, fwd=v["share_P_forward"], rev=v["share_P_reverse"]) for k, v in ra.items()]
    FD.parent.mkdir(exist_ok=True)
    FD.write_text(json.dumps(dict(plan="A8 ii / A9 note 3", B=B, rng=RNG, points=pts, shares=sh), indent=1) + "\n")

    # final-readability revision: exact text width (506.30 pt), no bbox tightening; 8 pt text, 7.5 pt ticks;
    # panel (b) encodes budget by colour AND solid shape (diamond n = 2500, X n = 250).
    FS, FT = 8, 7.5
    W_IN, H_IN = 506.295 / 72.27, 2.26
    L, GAP, BOT, PH = 0.50, 0.62, 0.40, 1.58
    WA = (W_IN - L - GAP - 0.08) * 1.35 / 2.35; WB = (W_IN - L - GAP - 0.08) / 2.35
    fig = plt.figure(figsize=(W_IN, H_IN))
    a = fig.add_axes([L / W_IN, BOT / H_IN, WA / W_IN, PH / H_IN])
    b = fig.add_axes([(L + WA + GAP) / W_IN, BOT / H_IN, WB / W_IN, PH / H_IN])
    xs = np.linspace(-2.0, 0, 50)
    a.fill_between(xs, 0, -2 * xs, color="#dddddd", lw=0, zorder=0)
    a.plot(xs, -2 * xs, color="#555555", lw=0.8, zorder=1)
    a.text(-1.22, 0.10, "|g| narrows\n($g_B<-D/2$)", fontsize=FS, color="#333333", ha="center", va="bottom")
    a.text(0.62, 0.10, "|g| widens", fontsize=FS, color="#333333", ha="center", va="bottom")
    a.axvline(0, color="#999999", lw=0.5, zorder=0); a.axhline(0, color="#999999", lw=0.5, zorder=0)
    for p in pts:
        col = dict((s, c) for s, _, c in SPL)[p["split"]]
        x, y = 100 * p["gB"], 100 * p["D"]
        a.plot([100 * p["gB_ci"][0], 100 * p["gB_ci"][1]], [y, y], color=col, lw=0.55, alpha=0.6)
        a.plot([x, x], [100 * p["D_ci"][0], 100 * p["D_ci"][1]], color=col, lw=0.55, alpha=0.6)
        a.plot(x, y, marker=TMK[p["t"]], color=col, ms=4.0, mfc=col, mec="white", mew=0.4, ls="none")
    a.set_xlim(-1.75, 2.35); a.set_ylim(0, 3.45)
    a.set_xlabel("scalar's own bias $g_B$ (pp)", fontsize=FS); a.set_ylabel("shift $D$ (pp)", fontsize=FS)
    a.set_title("(a) Narrowing boundary", fontsize=FS)
    a.tick_params(labelsize=FT, length=2.5, pad=1.5); a.spines[["top", "right"]].set_visible(False)
    h = [plt.Line2D([], [], color=c, marker="o", ls="none", ms=4.0, label=l) for _, l, c in SPL]
    h += [plt.Line2D([], [], color="#555555", marker=TMK[t], ls="none", ms=4.0, label=f"$t$ = {t:.2f}") for t in (0.8, 0.9, 0.95)]
    a.legend(handles=h, fontsize=FS, frameon=False, loc="upper right", ncol=2, handletextpad=0.2, columnspacing=0.8,
             labelspacing=0.15, borderaxespad=0.1)
    for bud, mk, col in ((2500, "D", "#0072B2"), (250, "X", "#CC79A7")):
        f = [100 * x["fwd"] for x in sh if x["budget"] == bud]; r = [100 * x["rev"] for x in sh if x["budget"] == bud]
        b.plot(f, r, ls="none", marker=mk, ms=4.2 if mk == "X" else 3.8, color=col, mew=0.5 if mk == "D" else 0.3,
               mec="white" if mk == "D" else col, label=f"$n$ = {bud}", alpha=0.9)
    b.plot([0, 80], [0, 80], color="#999999", lw=0.5)
    b.set_xlim(0, 80); b.set_ylim(0, 80)
    b.set_xlabel("forward-order selection share (%)", fontsize=FS); b.set_ylabel("reverse-order share (%)", fontsize=FS)
    b.set_title("(b) Score vs selection by order", fontsize=FS)
    b.tick_params(labelsize=FT, length=2.5, pad=1.5); b.spines[["top", "right"]].set_visible(False)
    b.legend(fontsize=FS, frameon=False, loc="lower right", handletextpad=0.3, borderaxespad=0.2)   # empty region below the diagonal
    fig.savefig(OUT, metadata={"CreationDate": None})
    print("wrote", OUT, "points", len(pts), "narrow", sum(p["narrows"] for p in pts))


if __name__ == "__main__":
    main()

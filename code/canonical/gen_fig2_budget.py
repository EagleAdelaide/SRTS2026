#!/usr/bin/env python3
"""Figure 2 (plan addendum A7 ii): ECE15 difference map − TvA-TS versus
calibration budget n in {250, 625, 1250, 2500} on Amazon ID test (seen,
development), Amazon confirmation (unseen reviewers), Yahoo ID test
(development) and Yahoo confirmation.  Means over seeds and draws with
descriptive 95% seed->draw hierarchical intervals (B = 20000, RNG
20261016).  Gate: SRTS-BCE − TvA-TS must reproduce the frozen analysis
files to 1e-9.  Writes results/www2027/posthoc_figdata/fig2_budget.json and
www2027/paper/figures/fig2_budget.pdf.  Type 42 fonts.
"""
import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 7})
import matplotlib.pyplot as plt
import numpy as np

REPO = Path("__SRTS_ROOT__/calattn-repo")
W = REPO / "results/www2027"
OUTD = W / "posthoc_figdata"
FIG = REPO / "www2027/paper/figures/fig2_budget.pdf"
B, RNG = 20000, 20261016
BUDGETS = [250, 625, 1250, 2500]
MAPS = [("HTS-BCE", "HTS+BCE", "#0072B2", "o"), ("SRTS-BCE", "SRTS-BCE", "#E69F00", "s"),
        ("SMART+BCE", "SMART+BCE", "#009E73", "^"), ("PWLinear-3", "PWLinear-3", "#D55E00", "D")]
PANELS = [("amazon_id", "Amazon, seen (dev.)", "suite/amazon_s*/amazon_distilbert_core_natural.jsonl", "id_test",
           "analysis_amazon_id_test_ece15.json", "amazon_distilbert"),
          ("amazon_conf", "Amazon, unseen (confirm.)", "CONFIRMATION/amazon_distilbert.jsonl", "CONFIRMATION",
           "analysis_CONFIRM_amazon_distilbert_ece15.json", "amazon_distilbert"),
          ("yahoo_id", "Yahoo (dev.)", "suite/yahoo_s*/yahoo_distilbert_core_natural.jsonl", "id_test",
           "analysis_yahoo_idtest_ece15.json", "yahoo_distilbert"),
          ("yahoo_conf", "Yahoo (confirm.)", "CONFIRMATION/yahoo_distilbert.jsonl", "CONFIRMATION",
           "analysis_CONFIRM_yahoo_distilbert_ece15.json", "yahoo_distilbert")]


def hier(A, i_s, i_d):
    return np.nanmean(np.nanmean(A[i_s[:, :, None], i_d], axis=2), axis=1)


def main():
    out = {"plan": "A7(ii)", "B": B, "rng": RNG, "panels": {}}
    gate_max = 0.0
    for key, title, pat, split, afile, task in PANELS:
        rows = [json.loads(l) for f in sorted(glob.glob(str(W / pat))) for l in open(f)]
        rows = [r for r in rows if r["test_split"] == split and r.get("sampling", "natural") == "natural"]
        frozen = json.load(open(W / afile))[task]
        tab = {(r["budget"], r["seed"], r["draw"], r["method"]): r["ece15"] for r in rows}
        out["panels"][key] = {"title": title, "contrasts": {}}
        for b in BUDGETS:
            seeds = sorted({k[1] for k in tab if k[0] == b}); draws = sorted({k[2] for k in tab if k[0] == b})
            S, D = len(seeds), len(draws)
            rng = np.random.default_rng(RNG)  # same index arrays for all maps of a panel/budget
            i_s, i_d = rng.integers(0, S, (B, S)), rng.integers(0, D, (B, S, D))
            for m, *_ in MAPS:
                A = np.array([[tab[(b, s, d, m)] - tab[(b, s, d, "TvA-TS")] for d in draws] for s in seeds])
                bt = hier(A, i_s, i_d)
                e = {"mean": float(A.mean()), "ci95": [float(x) for x in np.percentile(bt, [2.5, 97.5])],
                     "seeds": S, "draws": D, "status": "descriptive"}
                if m == "SRTS-BCE":
                    fz = frozen[str(b)]["SRTS-BCE - TvA-TS"]
                    gate_max = max(gate_max, abs(fz["point"] - e["mean"]))
                    e["frozen_point"] = fz["point"]; e["frozen_ci"] = fz["hier_ci"]
                    e["status"] = "pre-declared analysis (primary at n=250, 2500)" if key.endswith("conf") else "pre-declared development analysis"
                out["panels"][key]["contrasts"][f"{m}/n={b}"] = e
    out["gate_max_abs_diff_srts_vs_frozen"] = gate_max
    OUTD.mkdir(exist_ok=True)
    (OUTD / "fig2_budget.json").write_text(json.dumps(out, indent=1) + "\n")
    if gate_max > 1e-9:
        sys.exit(f"gate failed: SRTS-BCE − TvA-TS differs from frozen analysis by {gate_max}")

    # final-readability revision: exact text width (506.30 pt), no bbox tightening; 8 pt text, 7.5 pt ticks;
    # one shared x label and a one-row legend above the panels.
    FS, FT = 8, 7.5
    W_IN, H_IN = 506.295 / 72.27, 1.84
    L, R, GAP, BOT, PH = 0.62, 0.04, 0.16, 0.42, 0.96
    PW = (W_IN - L - R - 3 * GAP) / 4
    fig = plt.figure(figsize=(W_IN, H_IN))
    axes = []
    for j in range(4):
        axes.append(fig.add_axes([(L + j * (PW + GAP)) / W_IN, BOT / H_IN, PW / W_IN, PH / H_IN],
                                 sharey=axes[0] if axes else None))
    xs = np.arange(len(BUDGETS))
    for j, (ax, (key, title, *_)) in enumerate(zip(axes, PANELS)):
        c = out["panels"][key]["contrasts"]
        for k, (m, nm, col, mk) in enumerate(MAPS):
            off = (k - 1.5) * 0.12
            mean = [c[f"{m}/n={b}"]["mean"] for b in BUDGETS]
            lo = [c[f"{m}/n={b}"]["ci95"][0] for b in BUDGETS]; hi = [c[f"{m}/n={b}"]["ci95"][1] for b in BUDGETS]
            ax.plot(xs + off, mean, color=col, lw=1.0, marker=mk, ms=3.6, mfc=col, mew=0.8, label=nm)   # solid markers
            for x, l, h in zip(xs + off, lo, hi):
                ax.plot([x, x], [l, h], color=col, lw=0.9)
        ax.axhline(0, color="#777777", lw=0.7, zorder=0)
        ax.set_xticks(xs); ax.set_xticklabels([str(b) for b in BUDGETS], fontsize=FT)
        ax.set_title(title.replace("(confirm.)", "(conf.)"), fontsize=FS, pad=3)   # display label only; the data file keeps the title
        ax.tick_params(axis="both", labelsize=FT, length=2.5, pad=1.5)
        if j:
            ax.tick_params(axis="y", labelleft=False)
        ax.grid(axis="y", alpha=0.25, lw=0.5); ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("ECE$_{15}$: map $-$ scalar (pp)", fontsize=FS)
    fig.text((L + (W_IN - L - R) / 2) / W_IN, 0.04 / H_IN, "calibration labels $n$ (exact budget)", ha="center", va="bottom", fontsize=FS)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, fontsize=FS, frameon=False, bbox_to_anchor=((L + (W_IN - L - R) / 2) / W_IN, 1.0),
               handlelength=1.8, columnspacing=1.6, borderaxespad=0.1)
    fig.savefig(FIG, metadata={"CreationDate": None})
    print("wrote", FIG, "gate", gate_max)


if __name__ == "__main__":
    main()

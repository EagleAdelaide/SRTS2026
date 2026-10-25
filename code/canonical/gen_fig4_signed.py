#!/usr/bin/env python3
"""Appendix figure: why the seen/unseen contrast changes sign (plan A5
note 3, descriptive).  (a) the scalar map's own signed accepted-set bias
g(A_t) on the seen (ID test) and the two unseen (OOD validation,
confirmation) Amazon splits at n = 2500, t in {.80, .90, .95}; (b) the
signed shift map − scalar at t = 0.90 on seen and unseen reviewers for the
four conditional maps.  Means with descriptive 95% seed->draw intervals
read from results/www2027/posthoc_gamma/signed_descriptive.json (no new
computation).  Type 42 fonts.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 7})
import matplotlib.pyplot as plt
import numpy as np

REPO = Path("__SRTS_ROOT__/calattn-repo")
SRC = REPO / "results/www2027/posthoc_gamma/signed_descriptive.json"
OUT = REPO / "www2027/paper/figures/fig4_signed.pdf"
SPLITS = [("id_test", "seen (ID test)", "#000000", "o"), ("ood_val", "unseen (OOD val.)", "#56B4E9", "s"),
          ("ood_test_CONFIRMATION", "unseen (confirm.)", "#CC79A7", "D")]
MAPS = [("HTS-BCE", "HTS+BCE", "#0072B2", "o"), ("SRTS-BCE", "SRTS-BCE", "#E69F00", "s"),
        ("SMART+BCE", "SMART+BCE", "#009E73", "^"), ("PWLinear-3", "PWLinear-3", "#D55E00", "D")]


def main():
    d = json.load(open(SRC))
    sc = {(c["split"], c["t"]): c for c in d["scalar"] if c["budget"] == 2500 and c["rule"] == "fixed"}
    ct = {(c["ood"], c["method"]): c for c in d["contrasts"] if c["budget"] == 2500 and c["t"] == 0.9 and c["rule"] == "fixed"}
    fig, (a, b) = plt.subplots(1, 2, figsize=(3.33, 1.55), gridspec_kw={"width_ratios": [1.0, 1.15], "wspace": 0.55})
    # (a) scalar's signed bias by split and threshold
    ts = [0.8, 0.9, 0.95]
    for k, (sp, lab, col, mk) in enumerate(SPLITS):
        xs = np.arange(len(ts)) + (k - 1) * 0.22
        for x, t in zip(xs, ts):
            c = sc[(sp, t)]
            a.plot([x, x], [100 * c["ci95"][0], 100 * c["ci95"][1]], color=col, lw=0.9)
            a.plot([x], [100 * c["mean"]], marker=mk, ms=3.2, color=col, ls="none", label=lab if t == 0.8 else None)
    a.axhline(0, color="#777777", lw=0.7, zorder=0)
    a.set_xticks(range(len(ts))); a.set_xticklabels([f"{t:.2f}" for t in ts], fontsize=6)
    a.set_xlabel("threshold $t$", fontsize=6.3); a.set_ylabel("scalar $g(A_t)$ (pp)", fontsize=6.3)
    a.set_title("(a) scalar map's own bias", fontsize=6.6); a.tick_params(axis="y", labelsize=6)
    a.spines[["top", "right"]].set_visible(False)
    # (b) signed shift map - scalar, seen vs unseen, t = .90
    ys = np.arange(len(MAPS))[::-1]
    for k, (m, nm, col, mk) in enumerate(MAPS):
        for j, (key, lab2, mfc, off) in enumerate([("seen", "seen", col, 0.18), ("ood_val", "OOD val.", "white", 0.0),
                                                   ("conf", "confirm.", "#dddddd", -0.18)]):
            if key == "seen":
                e = ct[("ood_val", m)]["dsigned_seen"]
            elif key == "ood_val":
                e = ct[("ood_val", m)]["dsigned_unseen"]
            else:
                e = ct[("ood_test_CONFIRMATION", m)]["dsigned_unseen"]
            b.plot([100 * e["ci95"][0], 100 * e["ci95"][1]], [ys[k] + off] * 2, color=col, lw=0.9)
            b.plot([100 * e["mean"]], [ys[k] + off], marker=mk, ms=3.0, color=col, mfc=mfc, mew=0.8, ls="none")
    b.axvline(0, color="#777777", lw=0.7, zorder=0)
    b.set_yticks(ys); b.set_yticklabels([nm for _, nm, *_ in MAPS], fontsize=6); b.tick_params(axis="x", labelsize=6)
    b.set_xlabel("map $-$ scalar, $t{=}0.90$ (pp)", fontsize=6.2)
    b.set_title("(b) shift on seen / unseen", fontsize=6.6)
    b.spines[["top", "right"]].set_visible(False)
    h, l = a.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=5.6, frameon=False, bbox_to_anchor=(0.5, -0.04), handlelength=1.0, columnspacing=1.0)
    fig.subplots_adjust(bottom=0.33, top=0.88, left=0.14, right=0.98)
    fig.savefig(OUT, bbox_inches="tight", metadata={"CreationDate": None})
    print("wrote", OUT)


if __name__ == "__main__":
    main()

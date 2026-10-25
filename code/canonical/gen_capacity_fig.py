#!/usr/bin/env python3
"""Nested-capacity-family figure from canonical suite rows (one split only).

Two panels per cell: (a) family A, nested dyadic risk-quantile groups
K in {1,2,4,8} without a group-size floor (min_group=0); (b) family B, nested
polynomial risk maps with 1..4 coefficients. Both panels also draw the
deployed SRTS-BCE (K=3, 50-sample floor) as a dashed reference taken from
the SAME rows (cell, split, budgets, sampling, seeds, draws) as the panel's
solid lines.  y = mean over seeds of the per-seed mean test ECE15; error bar
= across-seed std (ddof=0) of the per-seed means.

Revision 2026-10-16 (fix F1): the test split is a required argument and
every aggregation asserts a single split; previously Amazon rows of two
splits (id_test and ood_val) were averaged into one curve.  Full-pool
replicate draws (identical calibration set) are collapsed to one draw after
asserting identical outputs.  Writes PDF, a JSON sidecar, a per-point CSV,
and a provenance JSON (input SHA-256, command, commit).

Usage:
  python gen_capacity_fig.py --cell amazon_distilbert --split id_test
  python gen_capacity_fig.py --cell c100_vit --split test
"""
import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path("__SRTS_ROOT__/calattn-repo")
SUITE = REPO / "results/www2027/suite"
OKABE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]
SER_A = [("K=1", "SRTS-K1-floor0", OKABE[0], "-"),
         ("K=2", "SRTS-K2-floor0", OKABE[1], "-"),
         ("K=4", "SRTS-K4-floor0", OKABE[2], "-"),
         ("K=8", "SRTS-K8-floor0", OKABE[3], "-"),
         ("K=3 (deployed)", "SRTS-K3-ref", "#555555", "--")]
SER_B = [("1 coef (=scalar)", "Poly1(=scalarBCE)", OKABE[0], "-"),
         ("2 coef", "Poly2", OKABE[1], "-"),
         ("3 coef", "Poly3", OKABE[2], "-"),
         ("4 coef", "Poly4", OKABE[3], "-"),
         ("K=3 (deployed)", "SRTS-K3-ref", "#555555", "--")]


class SplitError(ValueError):
    pass


def load_family(cell, family, split, sampling="natural", suite=SUITE):
    files = sorted(suite.glob(f"*/{cell}_{family}_{sampling}.jsonl"))
    if not files:
        raise FileNotFoundError(f"no {cell}_{family}_{sampling} rows")
    rows = []
    for f in files:
        rows += [json.loads(l) for l in open(f)]
    rows = [r for r in rows if r["cell"] == cell
            and r.get("sampling", "natural") == sampling]
    present = sorted({r["test_split"] for r in rows})
    if split not in present:
        raise SplitError(f"split {split!r} not in {present}")
    rows = [r for r in rows if r["test_split"] == split]
    return rows, files


def check_rows(rows):
    """Single split, unique keys; returns rows with full-pool replicates
    collapsed to draw 0 (asserting identical outputs)."""
    splits = {r["test_split"] for r in rows}
    if len(splits) != 1:
        raise SplitError(f"rows mix test splits {sorted(splits)}")
    keys = [(r["cell"], r["test_split"], r["seed"], r["budget"], r["draw"],
             r["method"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate (cell, split, seed, budget, draw, method)")
    out = []
    grp = defaultdict(list)
    for r in rows:
        grp[(r["seed"], r["budget"], r["method"])].append(r)
    for (s, b, m), rs in grp.items():
        full = rs[0]["sampler"]["pool_size"] <= b
        if full and len(rs) > 1:
            vals = {round(x["ece15"], 12) for x in rs}
            if len(vals) != 1:
                raise ValueError(f"full-pool replicates differ {s,b,m}")
            out.append(min(rs, key=lambda x: x["draw"]))
        else:
            out += rs
    return out


def check_pairing(rows, methods):
    """Every series must cover the same (seed, draw) set at every budget."""
    cover = defaultdict(set)
    for r in rows:
        if r["method"] in methods:
            cover[(r["method"], r["budget"])].add((r["seed"], r["draw"]))
    for b in sorted({k[1] for k in cover}):
        sets = {m: cover[(m, b)] for m in methods}
        ref = sets[methods[0]]
        for m, s_ in sets.items():
            if s_ != ref:
                raise ValueError(f"unpaired series {m} at n={b}")


def seed_mean_std(rows, method, budget, metric="ece15"):
    sub = [r for r in rows if r["method"] == method and r["budget"] == budget]
    if len({r["test_split"] for r in sub}) > 1:
        raise SplitError("aggregation over more than one split")
    per = defaultdict(list)
    for r in sub:
        per[r["seed"]].append(r[metric])
    sm = np.array([np.mean(v) for v in per.values()])
    nd = int(sum(len(v) for v in per.values()))
    if not len(sm):
        return float("nan"), float("nan"), 0, 0
    return float(sm.mean()), float(sm.std()), int(len(sm)), nd


def panel(ax, rows, series, budgets, title):
    for (label, method, color, ls) in series:
        m = np.array([seed_mean_std(rows, method, b)[0] for b in budgets])
        s = np.array([seed_mean_std(rows, method, b)[1] for b in budgets])
        ax.errorbar(budgets, m, yerr=s, color=color, ls=ls, lw=1.8,
                    marker="o", ms=3.5, capsize=2, elinewidth=0.9,
                    label=label)
    ax.set_xscale("log", base=2)
    ax.set_xticks(budgets)
    ax.set_xticklabels([str(b) for b in budgets])
    ax.set_xlabel("calibration budget $n$")
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.25, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7, frameon=False, loc="upper right",
              handlelength=1.6, labelspacing=0.25)


def build(cell, split, outdir):
    famA_raw, fA = load_family(cell, "familyA", split)
    famB_raw, fB = load_family(cell, "familyB", split)
    famA, famB = check_rows(famA_raw), check_rows(famB_raw)
    check_pairing(famA, [m for _, m, _, _ in SER_A])
    check_pairing(famB, [m for _, m, _, _ in SER_B])
    budgets = sorted({r["budget"] for r in famA})
    assert budgets == sorted({r["budget"] for r in famB})
    # the dashed reference must be the same fits in both families
    for b in budgets:
        a = {(r["seed"], r["draw"]): r["ece15"] for r in famA
             if r["method"] == "SRTS-K3-ref" and r["budget"] == b}
        bb = {(r["seed"], r["draw"]): r["ece15"] for r in famB
              if r["method"] == "SRTS-K3-ref" and r["budget"] == b}
        common = set(a) & set(bb)
        assert common and all(abs(a[k] - bb[k]) < 1e-9 for k in common), b
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})  # no Type 3 fonts
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=True)
    panel(axes[0], famA, SER_A, budgets,
          "(a) nested dyadic risk groups (no size floor)")
    panel(axes[1], famB, SER_B, budgets, "(b) nested polynomial risk maps")
    axes[0].set_ylabel(r"test ECE$_{15}$ (pp)")
    fig.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    stem = f"capacity_{cell}"
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight",
                metadata={"CreationDate": None})  # no local date in the shipped PDF
    side, rows_csv = {}, []
    for fam, ser, rows in (("A", SER_A, famA), ("B", SER_B, famB)):
        for label, mth, _, _ in ser:
            for b in budgets:
                mu, sd, ns, nd = seed_mean_std(rows, mth, b)
                side.setdefault(f"{fam}:{mth}", {})[b] = [mu, sd, ns, nd]
                rows_csv.append(dict(cell=cell, split=split, panel=fam,
                                     series=label, method=mth, budget=b,
                                     mean_ece15=mu, seed_std=sd, n_seeds=ns,
                                     n_draw_rows=nd))
    json.dump({"cell": cell, "split": split, "series": side},
              open(outdir / f"{stem}.json", "w"), indent=1)
    with open(outdir / f"{stem}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_csv[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows_csv)
    try:
        commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = "unknown"
    prov = {"command": " ".join(["python", Path(sys.argv[0]).name]
                                + sys.argv[1:]),
            "commit": commit, "cell": cell, "split": split,
            "inputs": {str(f.relative_to(REPO)):
                       hashlib.sha256(f.read_bytes()).hexdigest()
                       for f in fA + fB}}
    json.dump(prov, open(outdir / f"{stem}.provenance.json", "w"), indent=1)
    return side


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--split", required=True,
                    help="test split to plot (e.g. test, id_test); required")
    ap.add_argument("--outdir", default=str(REPO / "www2027/paper/figures"))
    a = ap.parse_args()
    build(a.cell, a.split, Path(a.outdir))
    print("wrote", Path(a.outdir) / f"capacity_{a.cell}.pdf")


if __name__ == "__main__":
    main()

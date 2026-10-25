#!/usr/bin/env python3
"""Plan A10 summaries from results/www2027/posthoc_closure/{est,dec}_*.csv.
Writes results/www2027/posthoc_closure/{summary_closure.json, estimation_summary.csv,
decision_summary.csv, decision_by_margin.csv, decision_by_seed_threshold.csv}.
Definitions (fixed in plan A10 before computing):
  equivalent cell : | |g_C| - |g_B| | < eps (eps = 0.10 pp; sensitivity 0, 0.05, 0.20)
  power           : mean switch rate over non-equivalent cells where C is better
  error           : mean switch rate over non-equivalent cells where C is not better
  regret          : E[|g_chosen| - min(|g_B|, |g_C|)] in pp, all cells
  experiment A    : decisions with FPC variances, truth = audit pool
  experiment B    : decisions with no-FPC variances, truth = evaluation pool;
                    oracle transfer regret = audit-pool oracle choice judged on the evaluation pool;
                    finite-label loss = total B regret - oracle transfer regret
  lower bound     : n_LB = (1 - 2 beta)^2 r_v (1 - r_v) / (2 H^2), beta = 0.1, r_v = (mean_{A_C} e +
                    mean_{A_B} e)/2 (Le Cam two-point bound; superpopulation, i.i.d. labels)
  planning        : n_plan = 1.96^2 r_v (1 - r_v)/H^2 (the A9 note 3 expression with D known)
  measured m80    : smallest m in the grid with within-pool switch rate >= 0.8 (G3), per beneficial cell
"""
import csv
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paired_audit.pools import accepted_sets  # noqa: E402
from paired_audit.simulate_audit import CACHE, MAPS, pools  # noqa: E402

W = Path("__SRTS_ROOT__/calattn-repo/results/www2027")
IN = W / "posthoc_closure"
EPS = 0.001
EPS_GRID = [0.0, 0.0005, 0.001, 0.002]
MBINS = [0, 0.001, 0.0025, 0.005, 0.01, 1]


def load(pattern):
    rows = []
    for f in sorted(glob.glob(str(IN / pattern))):
        rows += list(csv.DictReader(open(f)))
    for r in rows:
        for k, v in list(r.items()):
            if k in ("split", "rule", "method", "design", "proc"):
                continue
            if v in ("True", "False"):
                r[k] = v == "True"
            else:
                try:
                    r[k] = float(v)
                except ValueError:
                    pass
    return rows


def historical_est():
    """A8/A9 rows for E1-E7 on the fixed and stream branches (matched relabelled)."""
    rows = []
    for name in ("audit_rows.csv", "audit_rows_active.csv"):
        for r in csv.DictReader(open(W / "posthoc_audit" / name)):
            rule = "fixed" if r["rule"] == "fixed" else "stream_matched_then_restricted"
            rows.append(dict(seed=float(r["seed"]), budget=float(r["budget"]), split=r["split"], t=float(r["t"]), rule=rule,
                             method=r["method"], m=float(r["m"]), design=r["design"], rmse=float(r["rmse"]),
                             coverage95=float(r["coverage95"]), unique_labels=float(r["unique_labels"]), bias=float(r["bias"])))
    return rows


def est_summary(rows):
    agg = defaultdict(list)
    for r in rows:
        agg[(int(r["budget"]), r["split"], r["rule"], int(r["m"]), r["design"])].append(r)
    out = []
    for k, v in sorted(agg.items()):
        cov = np.array([x["coverage95"] for x in v])
        out.append(dict(budget=k[0], split=k[1], rule=k[2], m=k[3], design=k[4], cells=len(v),
                        rmse_pts=100 * float(np.mean([x["rmse"] for x in v])),
                        cov_mean=float(cov.mean()), cov_min=float(cov.min()), cov_p10=float(np.percentile(cov, 10)),
                        cov_median=float(np.median(cov)), unique_labels=float(np.mean([x["unique_labels"] for x in v]))))
    return out


def equal_label_ratios(es, ref):
    """RMSE of each design / RMSE of `ref` interpolated (log-log) at the design's mean unique labels."""
    idx = {(e["budget"], e["split"], e["rule"], e["m"], e["design"]): e for e in es}
    out = {}
    for b in (2500, 250):
        for sp in ("id_test", "ood_val"):
            for rule in ("fixed", "stream_matched_then_restricted", "exact_matched_within_pool"):
                ms = sorted({e["m"] for e in es if e["budget"] == b and e["split"] == sp and e["rule"] == rule and e["design"] == ref})
                if not ms:
                    continue
                xs = np.log([idx[(b, sp, rule, m, ref)]["unique_labels"] for m in ms])
                ys = np.log([max(idx[(b, sp, rule, m, ref)]["rmse_pts"], 1e-9) for m in ms])
                for d in sorted({e["design"] for e in es if e["budget"] == b and e["split"] == sp and e["rule"] == rule}):
                    rat = []
                    for m in ms:
                        e = idx.get((b, sp, rule, m, d))
                        if e is None:
                            continue
                        rat.append(e["rmse_pts"] / float(np.exp(np.interp(np.log(e["unique_labels"]), xs, ys))))
                    if rat:
                        out[f"n={b}/{sp}/{rule}/{d}_vs_{ref}"] = [float(min(rat)), float(max(rat))]
    return out


def regret(sw, gB, gC):
    best = min(abs(gB), abs(gC))
    return sw * (abs(gC) - best) + (1 - sw) * (abs(gB) - best)


def dec_summary(rows, eps):
    agg = defaultdict(list)
    for r in rows:
        agg[(int(r["budget"]), r["split"], r["rule"], r["proc"], int(r["m"]))].append(r)
    out = []
    for k, v in sorted(agg.items()):
        def cls(x, side):
            mg = x[f"margin_{side}"]
            return "equiv" if abs(mg) < eps else ("better" if mg > 0 else "worse")
        bA = [x for x in v if cls(x, "a") == "better"]; wA = [x for x in v if cls(x, "a") == "worse"]
        bB = [x for x in v if cls(x, "e") == "better"]; wB = [x for x in v if cls(x, "e") == "worse"]
        regA = [regret(x["sw_fpc"], x["gB_a"], x["gC_a"]) for x in v]
        regB = [regret(x["sw_nofpc"], x["gB_e"], x["gC_e"]) for x in v]
        orc = [regret(1.0 if x["better_a"] else 0.0, x["gB_e"], x["gC_e"]) for x in v]
        out.append(dict(budget=k[0], split=k[1], rule=k[2], proc=k[3], m=k[4], cells=len(v), labels=float(np.mean([x["labels"] for x in v])),
                        A_better_cells=len(bA), A_worse_cells=len(wA), A_equiv_cells=len(v) - len(bA) - len(wA),
                        A_power=float(np.mean([x["sw_fpc"] for x in bA])) if bA else None,
                        A_error=float(np.mean([x["sw_fpc"] for x in wA])) if wA else None,
                        A_regret_pts=100 * float(np.mean(regA)),
                        B_better_cells=len(bB), B_power=float(np.mean([x["sw_nofpc"] for x in bB])) if bB else None,
                        B_error=float(np.mean([x["sw_nofpc"] for x in wB])) if wB else None,
                        B_regret_pts=100 * float(np.mean(regB)), B_oracle_transfer_regret_pts=100 * float(np.mean(orc)),
                        B_finite_label_loss_pts=100 * float(np.mean(regB) - np.mean(orc)),
                        B_share_regret_gt_eps=float(np.mean([r_ > eps for r_ in regB])),
                        winners_disagree=int(sum(x["better_a"] != x["better_e"] for x in v))))
    return out


def by_margin(rows):
    out = []
    agg = defaultdict(list)
    for r in rows:
        if abs(r["margin_a"]) < EPS:
            continue
        h = abs(r["H_a"]); bi = int(np.searchsorted(MBINS, h, side="right") - 1)
        agg[(int(r["budget"]), r["split"], r["rule"], r["proc"], int(r["m"]), bi, bool(r["better_a"]))].append(r)
    for k, v in sorted(agg.items()):
        out.append(dict(budget=k[0], split=k[1], rule=k[2], proc=k[3], m=k[4], H_bin_lo_pts=100 * MBINS[k[5]], H_bin_hi_pts=100 * MBINS[k[5] + 1],
                        C_better=k[6], cells=len(v), switch_rate=float(np.mean([x["sw_fpc"] for x in v]))))
    return out


def by_seed_threshold(rows):
    out = []
    agg = defaultdict(list)
    for r in rows:
        agg[(int(r["budget"]), r["split"], r["rule"], r["proc"], int(r["m"]), int(r["seed"]), r["t"])].append(r)
    for k, v in sorted(agg.items()):
        bA = [x for x in v if x["margin_a"] >= EPS]
        out.append(dict(budget=k[0], split=k[1], rule=k[2], proc=k[3], m=k[4], seed=k[5], t=k[6], cells=len(v), better_cells=len(bA),
                        A_power=float(np.mean([x["sw_fpc"] for x in bA])) if bA else None,
                        A_regret_pts=100 * float(np.mean([regret(x["sw_fpc"], x["gB_a"], x["gC_a"]) for x in v])),
                        B_regret_pts=100 * float(np.mean([regret(x["sw_nofpc"], x["gB_e"], x["gC_e"]) for x in v])),
                        B_oracle_transfer_regret_pts=100 * float(np.mean([regret(1.0 if x["better_a"] else 0.0, x["gB_e"], x["gC_e"]) for x in v]))))
    return out


def rv_table():
    """v-weighted error rate r_v per audit-pool cell (draw 0), for the bound and the planning figure."""
    tab = {}
    for f in sorted(glob.glob(str(CACHE / "cache_s*_n*.npz"))):
        z = np.load(f)
        s = int(Path(f).stem.split("_s")[1].split("_n")[0]); b = int(Path(f).stem.split("_n")[1])
        for sp in ("id_test", "ood_val"):
            conf = z[f"conf_{sp}"][0].astype(np.float64); e = 1 - z[f"corr_{sp}"].astype(np.float64)
            aud, _ = pools(z[f"rev_{sp}"]); cB = conf[0]
            for t in (0.8, 0.9, 0.95):
                for mi, mname in enumerate(MAPS[1:], start=1):
                    for rule in ("fixed", "exact_matched_within_pool"):
                        AC, AB = accepted_sets(cB, conf[mi], aud, t, rule)
                        if AC.sum() and AB.sum():
                            tab[(s, b, sp, t, mname, rule)] = float((e[AC].mean() + e[AB].mean()) / 2)
    return tab


def sample_complexity(rows, rv):
    out = {}
    by = defaultdict(dict)
    for r in rows:
        if r["proc"] != "G3":
            continue
        by[(int(r["seed"]), int(r["budget"]), r["split"], r["t"], r["method"], r["rule"])][int(r["m"])] = r
    for key in ("fixed", "exact_matched_within_pool"):
        for sp in ("id_test", "ood_val"):
            for b in (2500, 250):
                cells = []
                for k, mm in by.items():
                    if k[5] != key or k[2] != sp or k[1] != b:
                        continue
                    r0 = next(iter(mm.values()))
                    if r0["margin_a"] < EPS:
                        continue
                    H = r0["H_a"]; r_v = rv[(k[0], k[1], k[2], k[3], k[4], k[5])]
                    lb = (1 - 0.2) ** 2 * r_v * (1 - r_v) / (2 * H * H)
                    plan = 1.96 ** 2 * r_v * (1 - r_v) / (H * H)
                    m80 = next((m for m in sorted(mm) if mm[m]["sw_fpc"] >= 0.8), None)
                    cells.append(dict(seed=k[0], t=k[3], method=k[4], H_pts=100 * H, r_v=r_v, n_lower_bound=lb, n_planning=plan,
                                      m80_measured=m80, support=r0["support_U"]))
                if cells:
                    m80 = [c["m80_measured"] for c in cells]
                    out[f"n={b}/{sp}/{key}"] = dict(
                        beneficial_cells=len(cells),
                        n_lower_bound_median=float(np.median([c["n_lower_bound"] for c in cells])),
                        n_lower_bound_range=[float(min(c["n_lower_bound"] for c in cells)), float(max(c["n_lower_bound"] for c in cells))],
                        n_planning_median=float(np.median([c["n_planning"] for c in cells])),
                        m80_reached_cells=int(sum(x is not None for x in m80)),
                        m80_median_of_reached=float(np.median([x for x in m80 if x is not None])) if any(x is not None for x in m80) else None,
                        m80_values=sorted([x if x is not None else -1 for x in m80]), cells=cells)
    return out


def main():
    E = load("est_*.csv"); D = load("dec_*.csv")
    es = est_summary(E + historical_est())
    with open(IN / "estimation_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(es[0].keys()), lineterminator="\n"); w.writeheader(); w.writerows(es)
    ds = dec_summary(D, EPS)
    with open(IN / "decision_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ds[0].keys()), lineterminator="\n"); w.writeheader(); w.writerows(ds)
    bm = by_margin(D)
    with open(IN / "decision_by_margin.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(bm[0].keys()), lineterminator="\n"); w.writeheader(); w.writerows(bm)
    bst = by_seed_threshold(D)
    with open(IN / "decision_by_seed_threshold.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(bst[0].keys()), lineterminator="\n"); w.writeheader(); w.writerows(bst)
    sens = {str(e): [d for d in dec_summary(D, e) if d["budget"] == 2500 and d["proc"] in ("G1", "G3") and d["m"] in (1000, 5000)] for e in EPS_GRID}
    rv = rv_table()
    sc = sample_complexity(D, rv)
    summary = dict(plan="A10", eps=EPS, eps_grid=EPS_GRID,
                   ratios_vs_E4=equal_label_ratios(es, "E4"), ratios_vs_E9=equal_label_ratios(es, "E9"),
                   ratios_vs_E8=equal_label_ratios(es, "E8"),
                   estimation=es, decisions=ds, eps_sensitivity=sens, sample_complexity=sc)
    (IN / "summary_closure.json").write_text(json.dumps(summary, indent=1) + "\n")
    print("rows", len(E), len(D))


if __name__ == "__main__":
    main()

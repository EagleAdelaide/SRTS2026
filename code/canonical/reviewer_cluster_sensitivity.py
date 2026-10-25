#!/usr/bin/env python3
"""Plan addendum A6, stage 2: post-hoc reviewer-cluster sensitivity of the
representative-condition contrasts (n = 2500, t = 0.90, fixed threshold;
4 maps x 2 OOD sets = 8 Gamma), from the stage-1 cache. Requires a passed
gate.json (stage 1).

Fixed-model reviewer bootstrap (primary): per replicate one multinomial
reviewer weight vector per split, shared by all maps, the scalar, seeds and
draws (ID-test weights shared by both OOD - ID contrasts; disjoint splits
resampled independently). g = sum w S / sum w N (item-weighted); within each
(seed, draw): |g|, map - scalar, OOD - ID; then mean over draws within seed
and over seeds. B = 50,000, RNG 20261021; stability stream RNG 20261022.
Secondary: joint crossed resampling (same reviewer weights for every model,
plus A5-style seed->draw indices, RNG 20261023). 95% and Bonferroni(8).
Output: results/www2027/posthoc_reviewer_cluster/{analysis.json, contrasts.csv}
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_suite import WEB  # noqa: E402

OUT = WEB / "posthoc_reviewer_cluster"
SPLITS = ["id_test", "ood_val", "ood_test_CONFIRMATION"]
MAPS = ["TvA-TS", "HTS-BCE", "SRTS-BCE", "SMART+BCE", "PWLinear-3"]
B, CHUNK, NF = 50_000, 2_500, 8
RNG_W = (20261021, 20261022)
RNG_IDX = 20261023
ZB = 100 * 0.05 / NF / 2


def load():
    gate = json.load(open(OUT / "gate.json"))
    if not gate.get("completed") or [v for v in gate["violations"] if v[3] != "PWLinear-3"]:
        sys.exit("stage-1 gate not passed")
    S, N = {}, {}
    for sp in SPLITS:
        S[sp] = np.stack([np.load(OUT / f"cache_s{s}.npz")[f"S_{sp}"] for s in range(5)])
        N[sp] = np.stack([np.load(OUT / f"cache_s{s}.npz")[f"N_{sp}"] for s in range(5)])
    return S, N, gate  # arrays: seed x draw x map x reviewer


def contrasts(g):
    """g: dict split -> (..., seed, draw, map). Returns per-(seed, draw)
    Delta_id (...,5,20,4), Delta_ood (...,5,20,8) and Gamma (...,5,20,8)."""
    dl = {sp: np.abs(g[sp][..., 1:]) - np.abs(g[sp][..., :1]) for sp in SPLITS}
    gam = np.concatenate([dl["ood_val"] - dl["id_test"],
                          dl["ood_test_CONFIRMATION"] - dl["id_test"]], axis=-1)
    dood = np.concatenate([dl["ood_val"], dl["ood_test_CONFIRMATION"]], axis=-1)
    return dl["id_test"], dood, gam


def hmean(x):
    """mean over draws within seed, then over seeds (axes -3, -2 = seed, draw)."""
    return np.nanmean(np.nanmean(x, axis=-2), axis=-2)


def run(S, N, seed_w, joint_idx=None):
    rng = np.random.default_rng(seed_w)
    ridx = np.random.default_rng(RNG_IDX) if joint_idx else None
    nS, nD, nM = S["id_test"].shape[:3]
    out = {"gamma": [], "d_id": [], "d_ood": []}
    if joint_idx:
        out["gamma_joint"] = []
    empty = 0
    for c0 in range(0, B, CHUNK):
        b = min(CHUNK, B - c0)
        g = {}
        for sp in SPLITS:
            Rn = S[sp].shape[-1]
            W = rng.multinomial(Rn, np.full(Rn, 1.0 / Rn), size=b).astype(np.float64)
            sw = S[sp].reshape(-1, Rn) @ W.T
            nw = N[sp].reshape(-1, Rn).astype(np.float64) @ W.T
            empty += int((nw == 0).sum())
            with np.errstate(invalid="ignore", divide="ignore"):
                gg = np.where(nw > 0, sw / nw, np.nan)
            g[sp] = gg.T.reshape(b, nS, nD, nM)
        d_id, d_ood, gam = contrasts(g)
        out["gamma"].append(hmean(gam))
        out["d_id"].append(hmean(d_id))
        out["d_ood"].append(hmean(d_ood))
        if joint_idx:
            i_s = ridx.integers(0, nS, (b, nS))
            i_d = ridx.integers(0, nD, (b, nS, nD))
            bi = np.arange(b)[:, None, None]
            sel = gam[bi, i_s[:, :, None], i_d]          # b x nS x nD x 8
            out["gamma_joint"].append(np.nanmean(np.nanmean(sel, axis=2), axis=1))
    return {k: np.concatenate(v) for k, v in out.items()}, empty


def ci(x, lv):
    q = [2.5, 97.5] if lv == "95" else [ZB, 100 - ZB]
    return [float(v) for v in np.nanpercentile(x, q)]


def main():
    S, N, gate = load()
    g0 = {sp: S[sp].sum(-1) / N[sp].sum(-1) for sp in SPLITS}
    d_id0, d_ood0, gam0 = (hmean(x) for x in contrasts(g0))
    r1, e1 = run(S, N, RNG_W[0], joint_idx=True)
    r2, e2 = run(S, N, RNG_W[1])
    a5 = {(e["ood"], e["method"]): e for e in json.load(open(WEB / "posthoc_gamma/analysis_gamma.json"))["contrasts"]
          if e["rule"] == "fixed" and e["budget"] == 2500 and e["t"] == 0.9}
    labels = [(o, m) for o in ("ood_val", "ood_test_CONFIRMATION") for m in MAPS[1:]]
    res = []
    for j, (o, m) in enumerate(labels):
        e = dict(ood=o, method=m, gamma=float(gam0[j]), gamma_a5=a5[(o, m)]["gamma"],
                 delta_id=float(d_id0[MAPS[1:].index(m)]), delta_ood=float(d_ood0[j]),
                 delta_id_ci95=ci(r1["d_id"][:, MAPS[1:].index(m)], "95"),
                 delta_ood_ci95=ci(r1["d_ood"][:, j], "95"),
                 reviewer_ci95=ci(r1["gamma"][:, j], "95"), reviewer_cibonf8=ci(r1["gamma"][:, j], "bonf"),
                 reviewer_ci95_s2=ci(r2["gamma"][:, j], "95"), reviewer_cibonf8_s2=ci(r2["gamma"][:, j], "bonf"),
                 joint_ci95=ci(r1["gamma_joint"][:, j], "95"), joint_cibonf8=ci(r1["gamma_joint"][:, j], "bonf"),
                 a5_seed_draw_ci95=a5[(o, m)]["ci95"], a5_seed_draw_cibonf96=a5[(o, m)]["cibonf"])
        for k in ("reviewer_ci95", "reviewer_cibonf8", "reviewer_ci95_s2", "reviewer_cibonf8_s2",
                  "joint_ci95", "joint_cibonf8"):
            e[k + "_excl0"] = bool(e[k][0] > 0 or e[k][1] < 0)
        res.append(e)
    mc = {lv: {"max_abs_bound_diff_points": 100 * max(abs(e[f"reviewer_{lv}"][i] - e[f"reviewer_{lv}_s2"][i])
                                                     for e in res for i in (0, 1)),
               "decision_flips": sum(e[f"reviewer_{lv}_excl0"] != e[f"reviewer_{lv}_s2_excl0"] for e in res)}
          for lv in ("ci95", "cibonf8")}
    out = dict(plan="POSTHOC_PLAN addendum A6", condition="amazon n=2500 t=0.90 fixed threshold",
               B=B, rng_weights=RNG_W, rng_joint_indices=RNG_IDX, family=NF,
               reviewers=gate.get("reviewers"), empty_accepted_cells={"stream1": e1, "stream2": e2},
               max_abs_point_diff_vs_a5_points=100 * max(abs(e["gamma"] - e["gamma_a5"]) for e in res),
               monte_carlo=mc, contrasts=res,
               summary={"reviewer_ci95_excl0": sum(e["reviewer_ci95_excl0"] for e in res),
                        "reviewer_cibonf8_excl0": sum(e["reviewer_cibonf8_excl0"] for e in res),
                        "joint_ci95_excl0": sum(e["joint_ci95_excl0"] for e in res),
                        "joint_cibonf8_excl0": sum(e["joint_cibonf8_excl0"] for e in res)})
    (OUT / "analysis.json").write_text(json.dumps(out, indent=1) + "\n")
    with open(OUT / "contrasts.csv", "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["ood_split", "method", "gamma_pts", "delta_id_pts", "delta_ood_pts",
                    "reviewer_ci95_lo", "reviewer_ci95_hi", "reviewer_bonf8_lo", "reviewer_bonf8_hi",
                    "joint_ci95_lo", "joint_ci95_hi", "joint_bonf8_lo", "joint_bonf8_hi",
                    "a5_seed_draw_ci95_lo", "a5_seed_draw_ci95_hi", "a5_bonf96_lo", "a5_bonf96_hi"])
        for e in res:
            w.writerow([e["ood"], e["method"]] + [f"{100 * v:.4f}" for v in (
                e["gamma"], e["delta_id"], e["delta_ood"], *e["reviewer_ci95"], *e["reviewer_cibonf8"],
                *e["joint_ci95"], *e["joint_cibonf8"], *e["a5_seed_draw_ci95"], *e["a5_seed_draw_cibonf96"])])
    for e in res:
        print(f"{e['ood'][:8]} {e['method']:10s} G={100*e['gamma']:+.2f} (A5 {100*e['gamma_a5']:+.2f}) "
              f"rev95[{100*e['reviewer_ci95'][0]:+.2f},{100*e['reviewer_ci95'][1]:+.2f}] "
              f"revB8[{100*e['reviewer_cibonf8'][0]:+.2f},{100*e['reviewer_cibonf8'][1]:+.2f}] "
              f"joint95[{100*e['joint_ci95'][0]:+.2f},{100*e['joint_ci95'][1]:+.2f}] "
              f"jointB8[{100*e['joint_cibonf8'][0]:+.2f},{100*e['joint_cibonf8'][1]:+.2f}] "
              f"A5sd95[{100*e['a5_seed_draw_ci95'][0]:+.2f},{100*e['a5_seed_draw_ci95'][1]:+.2f}] "
              f"dID {100*e['delta_id']:+.2f}[{100*e['delta_id_ci95'][0]:+.2f},{100*e['delta_id_ci95'][1]:+.2f}] "
              f"dOOD {100*e['delta_ood']:+.2f}[{100*e['delta_ood_ci95'][0]:+.2f},{100*e['delta_ood_ci95'][1]:+.2f}]")
    print("MC", mc, "empty", e1, e2, "summary", out["summary"],
          "point diff vs A5 (pts)", out["max_abs_point_diff_vs_a5_points"])


if __name__ == "__main__":
    main()

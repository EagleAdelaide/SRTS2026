#!/usr/bin/env python3
"""Plan addendum A5: direct seen-vs-unseen split contrast on Amazon.

Gamma = Delta_OOD - Delta_ID, Delta_X = |g_m(A_m)| - |g_TvA(A_TvA)| on split
X, formed within one (map, budget, seed, draw) from the A4(iii) rows, where a
single fit per draw is evaluated on id_test, ood_val and the confirmation
split in the same process. 4 maps x 2 budgets x 3 thresholds x 2 OOD splits
x 2 selection rules (fixed threshold, matched coverage) = 96 contrasts.

Order: pairing checks first (check_gamma_pairing.py, layer A; this script
refuses to run without a passed record), then a
seed->draw hierarchical bootstrap (B = 200,000; primary RNG 20261016, Monte
Carlo stability stream RNG 20261017), 95% and Bonferroni(96), and seed-level
t intervals (uncorrected and Bonferroni-corrected). Sensitivity: the
fixed-threshold Gamma recomputed from the original suite / CONFIRMATION rows.
Outputs: results/www2027/posthoc_gamma/{analysis_gamma.json,
gamma_contrasts.csv, pairing_check.json}.
"""
import csv
import glob
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sampler import natural_subsample  # noqa: E402

W = Path("__SRTS_ROOT__/calattn-repo/results/www2027")
OUT = W / "posthoc_gamma"
TASK = "amazon_distilbert"
M = ["HTS-BCE", "SRTS-BCE", "SMART+BCE", "PWLinear-3"]
REF = "TvA-TS"
BUDGETS, TS = (250, 2500), (80, 90, 95)
OODS = ("ood_val", "ood_test_CONFIRMATION")
RULES = {"fixed": "errgap_a{t}", "matched": "g_matched_a{t}"}
B, NF = 200_000, 96
SEEDS_RNG = (20261016, 20261017)
SAMPLER_KEYS = ("requested_n", "actual_n", "sampler", "pool_size", "pool_hash",
                "class_counts", "bitgen")


def load_matched():
    rows = [json.loads(l) for f in sorted(glob.glob(str(W / f"posthoc_matched/{TASK}_s*.jsonl")))
            for l in open(f)]
    tab, dup = {}, 0
    for r in rows:
        k = (r["split"], r["seed"], r["budget"], r["draw"], r["method"])
        dup += k in tab
        tab[k] = r
    return tab, dup


def load_original():
    tab = {}
    for f in sorted(glob.glob(str(W / "suite/amazon_s*/amazon_distilbert_core_natural.jsonl"))):
        for r in map(json.loads, open(f)):
            tab[(r["test_split"], r["seed"], r["budget"], r["draw"], r["method"])] = r
    for r in map(json.loads, open(W / f"CONFIRMATION/{TASK}.jsonl")):
        tab[("ood_test_CONFIRMATION", r["seed"], r["budget"], r["draw"], r["method"])] = r
    return tab


def pairing_checks(tab, dup, orig, seeds, draws):
    out, ok = {"duplicates_in_A4iii_rows": dup}, dup == 0
    # (a) completeness: every (seed, budget, draw, method) on all three splits
    missing = [(sp, s, b, d, m) for s in seeds for b in BUDGETS for d in draws
               for m in M + [REF] for sp in ("id_test",) + OODS
               if (sp, s, b, d, m) not in tab]
    out["missing_keys"] = len(missing)
    ok &= not missing
    # (b) calibration indices recomputed from the shared rule, hashed, and
    # compared with the sampler records of every original row of the draw
    idx_hash, rec_bad, n_rec = {}, [], 0
    for s in seeds:
        vt = np.load(W / f"wilds_amazon/distilbert_s{s}/logits_id_val.npz")["targets"].astype(np.int64)
        for b in BUDGETS:
            for d in draws:
                rng = np.random.default_rng(77_000_000 + 1_000_000 * s + 1000 * b + d)
                idx, rec = natural_subsample(vt, b, rng)
                idx_hash[f"s{s}/n{b}/d{d}"] = hashlib.sha256(
                    np.ascontiguousarray(idx, dtype=np.int64).tobytes()).hexdigest()[:16]
                rec = json.loads(json.dumps(rec))  # str class keys, as saved
                for sp in ("id_test",) + OODS:
                    for m in M + [REF]:
                        o = orig.get((sp, s, b, d, m))
                        if o is None:
                            rec_bad.append(("missing", sp, s, b, d, m)); continue
                        n_rec += 1
                        if any(o["sampler"][k] != rec[k] for k in SAMPLER_KEYS):
                            rec_bad.append((sp, s, b, d, m))
    out["sampler_records_compared"] = n_rec
    out["sampler_record_mismatches"] = len(rec_bad)
    out["calibration_index_sha256_16"] = idx_hash
    ok &= not rec_bad and n_rec > 0
    # (c) fixed-threshold errgap and ECE of the A4(iii) refit vs original rows
    # (A5 note: (c) is reported, not a pairing condition; Gamma uses A4(iii)
    # rows only and the original-row sensitivity bounds its effect)
    worst, exceed = Counter(), []
    for (sp, s, b, d, m), r in tab.items():
        o = orig[(sp, s, b, d, m)]
        dd = max([abs(o["ece15"] - r["ece15"])]
                 + [100 * abs(o[f"errgap_a{t}"] - r[f"errgap_a{t}"]) for t in TS
                    if o[f"errgap_a{t}"] == o[f"errgap_a{t}"]])
        worst[m] = max(worst[m], dd)
        if dd > (1e-3 if m == "PWLinear-3" else 1e-9):
            exceed.append([sp, s, b, d, m, dd])
    out["refit_vs_original_max_abs_diff_points"] = dict(worst)
    out["refit_vs_original_gate_exceedances"] = sorted(exceed)
    # (d) model version (A5 note 2): logits/labels read by the A4(iii) refit
    # vs the released MANIFEST and float32 logits
    man = json.load(open(W / "web_logits_release/MANIFEST.json"))["files"]
    files = {"id_val": "logits_id_val.npz", "id_test": "logits_id_test.npz",
             "ood_val": "logits_val.npz", "ood_test_CONFIRMATION": "logits_test.npz"}
    mv, mv_ok = {}, True
    for s in seeds:
        for sp, fn in files.items():
            z = np.load(W / f"wilds_amazon/distilbert_s{s}/{fn}")
            lg, y = z["logits"].astype(np.float64), z["targets"].astype(np.int64)
            rel = np.load(W / f"web_logits_release/{TASK}_s{s}/{sp}.npz")
            ysha = hashlib.sha256(np.ascontiguousarray(y, dtype=np.int64).tobytes()).hexdigest()
            same_lab = ysha == man[f"{TASK}_s{s}/{sp}.npz"]["targets_sha256"]
            same_lg = bool(np.array_equal(lg.astype(np.float32), rel["logits"]))
            mv[f"s{s}/{sp}"] = {"logits_sha256_16": hashlib.sha256(
                np.ascontiguousarray(lg).tobytes()).hexdigest()[:16],
                "labels_match_manifest": same_lab, "logits_match_release_f32": same_lg}
            mv_ok &= same_lab and same_lg
    out["model_version"] = mv
    out["model_version_ok"] = bool(mv_ok)
    out["missing_fields"] = ["calibration-index hash in rows (substitute: (b))",
                             "fit_report in CONFIRMATION rows (substitute: (c))",
                             "logits hash in rows (substitute: (d) + (c))"]
    out["passed"] = bool(ok)
    return out


def delta(tab, sp, s, b, d, m, field):
    return abs(tab[(sp, s, b, d, m)][field]) - abs(tab[(sp, s, b, d, REF)][field])


def gamma_matrix(tab, ood, b, m, field, seeds, draws):
    return np.array([[delta(tab, ood, s, b, d, m, field) - delta(tab, "id_test", s, b, d, m, field)
                      for d in draws] for s in seeds])


def boot_index(S, D, rng):
    return rng.integers(0, S, (B, S)), rng.integers(0, D, (B, S, D))


def hier(A, idx):
    i_s, i_d = idx
    return np.nanmean(np.nanmean(A[i_s[:, :, None], i_d], axis=2), axis=1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tab, dup = load_matched()
    orig = load_original()
    seeds = sorted({k[1] for k in tab})
    draws = sorted({k[3] for k in tab})
    S, D = len(seeds), len(draws)
    # pairing checks (b) and (d) need the validation labels and logits
    # (layer A): check_gamma_pairing.py writes pairing_check.json; here the
    # row-level part (a) is re-checked and the recorded result is required.
    pc = OUT / "pairing_check.json"
    if not pc.exists():
        sys.exit("run check_gamma_pairing.py first (needs layer A data)")
    chk = json.load(open(pc))
    missing = [(sp, s, b, d, m) for s in seeds for b in BUDGETS for d in draws
               for m in M + [REF] for sp in ("id_test",) + OODS if (sp, s, b, d, m) not in tab]
    if dup or missing or not chk["passed"] or not chk["model_version_ok"]:
        sys.exit("pairing check failed")

    idxs = [boot_index(S, D, np.random.default_rng(r)) for r in SEEDS_RNG]
    zb = 100 * 0.05 / NF / 2
    t95 = stats.t.ppf(0.975, S - 1)
    tbf = stats.t.ppf(1 - 0.05 / NF / 2, S - 1)
    res, nan_cells = [], 0
    for rule, fpat in RULES.items():
        for ood in OODS:
            for b in BUDGETS:
                for t in TS:
                    for m in M:
                        A = gamma_matrix(tab, ood, b, m, fpat.format(t=t), seeds, draws)
                        nan_cells += int(np.isnan(A).sum())
                        pt = float(np.nanmean(np.nanmean(A, axis=1)))
                        ps = np.nanmean(A, axis=1)
                        se = ps.std(ddof=1) / np.sqrt(S)
                        Do = np.array([[delta(tab, ood, s, b, d, m, fpat.format(t=t))
                                        for d in draws] for s in seeds])
                        Di = np.array([[delta(tab, "id_test", s, b, d, m, fpat.format(t=t))
                                        for d in draws] for s in seeds])
                        e = dict(rule=rule, ood=ood, budget=b, t=t / 100, method=m,
                                 gamma=pt,
                                 delta_ood=float(np.nanmean(np.nanmean(Do, axis=1))),
                                 delta_id=float(np.nanmean(np.nanmean(Di, axis=1))),
                                 # display only (A5 note 2), primary index arrays
                                 delta_ood_ci95=[float(x) for x in np.nanpercentile(hier(Do, idxs[0]), [2.5, 97.5])],
                                 delta_id_ci95=[float(x) for x in np.nanpercentile(hier(Di, idxs[0]), [2.5, 97.5])],
                                 per_seed=[float(x) for x in ps],
                                 seeds_same_sign=int((np.sign(ps) == np.sign(pt)).sum()),
                                 seed_t95=[float(ps.mean() - t95 * se), float(ps.mean() + t95 * se)],
                                 seed_tbonf=[float(ps.mean() - tbf * se), float(ps.mean() + tbf * se)])
                        for j, ix in enumerate(idxs):
                            bt = hier(A, ix)
                            lo, hi = np.nanpercentile(bt, [2.5, 97.5])
                            lb, hb = np.nanpercentile(bt, [zb, 100 - zb])
                            e[f"ci95_s{j}"] = [float(lo), float(hi)]
                            e[f"cibonf_s{j}"] = [float(lb), float(hb)]
                        res.append(e)
                        print(f"{rule:7s} {ood[:7]} n{b:<4d} t{t} {m:10s} G={100*pt:+.2f} "
                              f"95%[{100*e['ci95_s0'][0]:+.2f},{100*e['ci95_s0'][1]:+.2f}] "
                              f"Bf[{100*e['cibonf_s0'][0]:+.2f},{100*e['cibonf_s0'][1]:+.2f}] "
                              f"seeds {e['seeds_same_sign']}/5", flush=True)
    assert len(res) == NF

    def excl(iv):
        return iv[0] > 0 or iv[1] < 0

    for e in res:
        for lv in ("ci95", "cibonf"):
            e[f"{lv}"] = e[f"{lv}_s0"]
            e[f"{lv}_excl0"] = excl(e[f"{lv}_s0"])
            e[f"{lv}_mc_flip"] = excl(e[f"{lv}_s0"]) != excl(e[f"{lv}_s1"])
        e["seed_t95_excl0"] = excl(e["seed_t95"])
        e["seed_tbonf_excl0"] = excl(e["seed_tbonf"])
    mc = {lv: {"max_abs_bound_diff_points": 100 * max(max(abs(e[f"{lv}_s0"][i] - e[f"{lv}_s1"][i])
                                                       for i in (0, 1)) for e in res),
               "decision_flips": sum(e[f"{lv}_mc_flip"] for e in res)} for lv in ("ci95", "cibonf")}

    # sensitivity: fixed-threshold Gamma from the original rows
    sens = 0.0
    for e in res:
        if e["rule"] != "fixed":
            continue
        A0 = gamma_matrix(orig, e["ood"], e["budget"], e["method"], f"errgap_a{int(round(100 * e['t']))}",
                          seeds, draws)
        e["gamma_original_rows"] = float(np.nanmean(np.nanmean(A0, axis=1)))
        sens = max(sens, abs(e["gamma_original_rows"] - e["gamma"]))

    summ = {}
    for rule in RULES:
        for ood in OODS:
            for b in BUDGETS:
                g = [e for e in res if e["rule"] == rule and e["ood"] == ood and e["budget"] == b]
                summ[f"{rule}/{ood}/n={b}"] = dict(
                    n=len(g), mean_pos=sum(e["gamma"] > 0 for e in g),
                    ci95_pos=sum(e["ci95_excl0"] and e["gamma"] > 0 for e in g),
                    ci95_neg=sum(e["ci95_excl0"] and e["gamma"] < 0 for e in g),
                    bonf_pos=sum(e["cibonf_excl0"] and e["gamma"] > 0 for e in g),
                    bonf_neg=sum(e["cibonf_excl0"] and e["gamma"] < 0 for e in g),
                    seed95_excl=sum(e["seed_t95_excl0"] for e in g),
                    seedbonf_excl=sum(e["seed_tbonf_excl0"] for e in g),
                    all5_same_sign=sum(e["seeds_same_sign"] == S for e in g))
    tot = {k: sum(v[k] for v in summ.values()) for k in next(iter(summ.values()))}
    out = dict(plan="POSTHOC_PLAN addendum A5", task=TASK, ref=REF, maps=M, budgets=BUDGETS,
               thresholds=[t / 100 for t in TS], ood_splits=OODS, rules=list(RULES),
               family=NF, B=B, rng=SEEDS_RNG, seeds=seeds, draws=D,
               t_crit={"95": float(t95), "bonf": float(tbf)}, nan_cells=nan_cells,
               pairing=chk["passed"], monte_carlo=mc,
               sensitivity_fixed_from_original_rows_max_abs_diff_points=100 * sens,
               summary=summ, total=tot, contrasts=res)
    (OUT / "analysis_gamma.json").write_text(json.dumps(out, indent=1) + "\n")
    with open(OUT / "gamma_contrasts.csv", "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["rule", "ood_split", "budget", "t", "method", "gamma_pts", "delta_ood_pts",
                    "delta_id_pts", "ci95_lo", "ci95_hi", "cibonf_lo", "cibonf_hi",
                    "cibonf_stream2_lo", "cibonf_stream2_hi",
                    "seed_t95_lo", "seed_t95_hi", "seed_tbonf_lo", "seed_tbonf_hi",
                    "delta_ood_ci95_lo", "delta_ood_ci95_hi", "delta_id_ci95_lo", "delta_id_ci95_hi",
                    "seeds_same_sign", "per_seed_gamma_pts"])
        for e in res:
            w.writerow([e["rule"], e["ood"], e["budget"], e["t"], e["method"]]
                       + [f"{100 * v:.4f}" for v in (e["gamma"], e["delta_ood"], e["delta_id"],
                                                     *e["ci95"], *e["cibonf"], *e["cibonf_s1"],
                                                     *e["seed_t95"], *e["seed_tbonf"],
                                                     *e["delta_ood_ci95"], *e["delta_id_ci95"])]
                       + [e["seeds_same_sign"], ";".join(f"{100 * v:.4f}" for v in e["per_seed"])])
    print("MC:", mc, "| sensitivity (pts):", 100 * sens, "| NaN cells:", nan_cells)
    for k, v in summ.items():
        print(k, v)
    print("TOTAL", tot)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Plan A8 (iii): paired accepted-set audit simulation on Amazon logits.
Populations: ood_val (unseen reviewers) and id_test (seen reviewers), each
split by reviewer-id parity (hash) into an audit pool and an evaluation
pool.  Calibrators: cache draw 0 of each seed (n in {250, 2500}).  For each
map C vs B = TvA-TS, rule (fixed t in {.80,.90,.95}; matched), audit budget
m in {25,...,1000} and R = 400 Monte Carlo audits: estimators E1-E4 of the
label term and E0 of g_B; metrics bias, RMSE, 95% coverage (normal, HT
variance), unique labels; guardrail policy (E0 with m/2 and E4 with m/2)
evaluated on the evaluation pool: switch rate, false-switch rate, regret
in |g| vs the oracle.  Outputs: results/www2027/posthoc_audit/
{audit_rows.csv, policy_rows.csv, AUDIT_LABEL_EFFICIENCY.csv, summary.json}
"""
import csv
import glob
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paired_audit.identities import accepted_fixed, accepted_matched, audit_weights, d_conf, gap  # noqa: E402
from paired_audit.estimators import LabelTerm, guardrail_decision, level_gB  # noqa: E402

W = Path("__SRTS_ROOT__/calattn-repo/results/www2027")
CACHE = W / "posthoc_mechanism"
OUT = W / "posthoc_audit"
MAPS = ["TvA-TS", "HTS-BCE", "SRTS-BCE", "SMART+BCE", "PWLinear-3"]
SPLITS = ["ood_val", "id_test"]
TS = [0.80, 0.90, 0.95]
MGRID = [25, 50, 100, 250, 500, 1000]
_ENV = __import__("os").environ
R = int(_ENV.get("AUDIT_R", "400"))
RNG0 = int(_ENV.get("AUDIT_RNG0", "20261024"))            # A9 run: 20261025
DESIGNS = _ENV.get("AUDIT_DESIGNS", "E1,E2,E3,E4").split(",")  # A9 run: E5,E6,E7
POLICY = _ENV.get("AUDIT_POLICY", "1") == "1"             # A9 run: 0 (policy unchanged)
TAG = _ENV.get("AUDIT_TAG", "")                           # A9 run: "active" -> audit_rows_active.csv only


def pools(rev):
    h = np.array([int(hashlib.sha256(str(int(r)).encode()).hexdigest()[:8], 16) % 2 for r in rev])
    return h == 0, h == 1  # audit, evaluation


def main():
    OUT.mkdir(exist_ok=True)
    arows, prows = [], []
    cell = 0
    for f in sorted(glob.glob(str(CACHE / "cache_s*_n*.npz"))):
        z = np.load(f)
        s = int(Path(f).stem.split("_s")[1].split("_n")[0]); b = int(Path(f).stem.split("_n")[1])
        for sp in SPLITS:
            conf = z[f"conf_{sp}"][0].astype(np.float64)   # draw 0
            corr = z[f"corr_{sp}"].astype(np.float64)
            aud, ev = pools(z[f"rev_{sp}"])
            cB = conf[0]
            for t in TS:
                for mi, mname in enumerate(MAPS[1:], start=1):
                    cC = conf[mi]
                    for rule in ("fixed", "matched"):
                        if rule == "fixed":
                            AC, AB = accepted_fixed(cC, t), accepted_fixed(cB, t)
                        else:
                            k = int(accepted_fixed(cB, t).sum()); AC, AB = accepted_matched(cC, k), accepted_matched(cB, k)
                        # truth on the audit pool (estimand of the audit) and on the evaluation pool (policy target)
                        truth = {}
                        for name, mask in (("audit", aud), ("eval", ev)):
                            A1, A0 = AC & mask, AB & mask
                            if A1.sum() < 10 or A0.sum() < 10:
                                truth = None; break
                            truth[name] = dict(gB=gap(cB, corr, A0), gC=gap(cC, corr, A1), AC=A1, AB=A0)
                        if truth is None:
                            continue
                        AC_a, AB_a = truth["audit"]["AC"], truth["audit"]["AB"]
                        w = audit_weights(AC_a, AB_a); dconf = d_conf(cC, cB, AC_a, AB_a)
                        L = float(np.dot(w, corr)); delta_true = truth["audit"]["gC"] - truth["audit"]["gB"]
                        assert abs(dconf - L - delta_true) < 1e-9
                        lt = LabelTerm(w, AC_a, AB_a, surrogate=cB)  # surrogate used by E5-E7 only
                        gB_ev, gC_ev = truth["eval"]["gB"], truth["eval"]["gC"]
                        better_C = abs(gC_ev) < abs(gB_ev)
                        for m in MGRID:
                            cell += 1
                            rng = np.random.default_rng(RNG0 + cell)
                            for design in DESIGNS:
                                est = np.zeros(R); var = np.zeros(R); nl = np.zeros(R)
                                for r in range(R):
                                    e, v, n = lt.estimate(design, corr, m, rng)
                                    est[r] = dconf - e; var[r] = v; nl[r] = n
                                se = np.sqrt(np.maximum(var, 0))
                                cover = float(np.mean(np.abs(est - delta_true) <= 1.96 * se))
                                arows.append(dict(seed=s, budget=b, split=sp, t=t, rule=rule, method=mname, m=m, design=design,
                                                  delta_true=delta_true, bias=float(est.mean() - delta_true),
                                                  rmse=float(np.sqrt(np.mean((est - delta_true) ** 2))), mc_sd=float(est.std()),
                                                  mean_se=float(se.mean()), coverage95=cover, unique_labels=float(nl.mean()),
                                                  overlap=float((AC_a & AB_a).sum() / max(1, (AC_a | AB_a).sum())),
                                                  symdiff=int((AC_a ^ AB_a).sum()), nA_C=int(AC_a.sum()), nA_B=int(AB_a.sum())))
                            if not POLICY:
                                continue
                            # policy: m/2 labels for g_B (E0), m/2 for Delta (E4)
                            sw = np.zeros(R, bool); regret = np.zeros(R)
                            for r in range(R):
                                gB_hat, gB_var, _ = level_gB(cB, corr, AB_a, m // 2, rng)
                                e, v, _ = lt.estimate("E4", corr, m - m // 2, rng)
                                dec, _ = guardrail_decision(gB_hat, gB_var, dconf - e, v)
                                sw[r] = dec
                                chosen = abs(gC_ev) if dec else abs(gB_ev)
                                regret[r] = chosen - min(abs(gC_ev), abs(gB_ev))
                            prows.append(dict(seed=s, budget=b, split=sp, t=t, rule=rule, method=mname, m=m,
                                              gB_eval=gB_ev, gC_eval=gC_ev, C_better_on_eval=bool(better_C),
                                              switch_rate=float(sw.mean()), false_switch_rate=float(np.mean(sw & ~better_C)),
                                              missed_switch_rate=float(np.mean(~sw & better_C)), regret_mean=float(regret.mean()),
                                              gB_audit=truth["audit"]["gB"], gC_audit=truth["audit"]["gC"]))
        print("done", f, flush=True)
    if TAG:
        with open(OUT / f"audit_rows_{TAG}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(arows[0].keys()), lineterminator="\n"); w.writeheader(); w.writerows(arows)
        print("wrote", OUT / f"audit_rows_{TAG}.csv", "(merge with merge_audit.py)")
        return
    for name, rows in (("audit_rows.csv", arows), ("policy_rows.csv", prows)):
        with open(OUT / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator="\n"); w.writeheader(); w.writerows(rows)
    # efficiency summary: per (budget, split, rule, m, design) averaged over seeds, maps, t
    agg = {}
    for r in arows:
        k = (r["budget"], r["split"], r["rule"], r["m"], r["design"])
        agg.setdefault(k, []).append(r)
    eff = []
    for k, rs in sorted(agg.items()):
        rm = np.mean([x["rmse"] for x in rs]); e3 = agg[(k[0], k[1], k[2], k[3], "E3")]
        eff.append(dict(budget=k[0], split=k[1], rule=k[2], m=k[3], design=k[4], n_cells=len(rs),
                        rmse_mean_pts=100 * rm, rmse_rel_to_E3=rm / np.mean([x["rmse"] for x in e3]),
                        abs_bias_mean_pts=100 * np.mean([abs(x["bias"]) for x in rs]),
                        coverage95_mean=np.mean([x["coverage95"] for x in rs]),
                        unique_labels_mean=np.mean([x["unique_labels"] for x in rs])))
    with open(OUT / "AUDIT_LABEL_EFFICIENCY.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(eff[0].keys()), lineterminator="\n"); w.writeheader(); w.writerows(eff)
    pagg = {}
    for r in prows:
        pagg.setdefault((r["budget"], r["split"], r["rule"], r["m"]), []).append(r)
    psum = {f"n={k[0]}/{k[1]}/{k[2]}/m={k[3]}": dict(cells=len(v), switch=np.mean([x["switch_rate"] for x in v]),
                                                  false_switch=np.mean([x["false_switch_rate"] for x in v]),
                                                  missed=np.mean([x["missed_switch_rate"] for x in v]),
                                                  regret_pts=100 * np.mean([x["regret_mean"] for x in v]),
                                                  share_C_better=np.mean([x["C_better_on_eval"] for x in v]))
            for k, v in sorted(pagg.items())}
    (OUT / "summary.json").write_text(json.dumps({"plan": "A8(iii)", "R": R, "rng0": RNG0, "m_grid": MGRID,
                                                  "efficiency": eff, "policy": psum}, indent=1) + "\n")
    for e in eff:
        if e["m"] in (100, 500) and e["rule"] == "fixed":
            print(e)
    for k, v in psum.items():
        if "/fixed/" in k:
            print(k, {a: round(float(b_), 3) for a, b_ in v.items()})


if __name__ == "__main__":
    main()

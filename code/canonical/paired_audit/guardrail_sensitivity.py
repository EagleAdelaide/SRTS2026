#!/usr/bin/env python3
"""Plan A9 note 3 (b)-(f), same cells as A8 (iii) (draw 0; audit/evaluation pools).
(b) rho^2 = corr(b, c_B)^2 within the audit pool's A_B (the factor by which a
    regression estimator with surrogate c_B can reduce the level's variance);
(c) missed-switch rate of the A8 policy conditional on cells where the map is
    better on the evaluation pool (from policy_rows.csv);
(d) probability that the level sample (m/2 SRSWOR labels from A_B) contains no
    error (exact hypergeometric), next to the A8 false-switch rate;
(e) sensitivity policy: as A8 but the level variance uses the Agresti-Coull
    proportion p~ = (x + 2)/(n + 4) (never zero) and Delta comes from E6;
    R = 400 audits per cell and m, RNG 20261026 + cell index;
(f) labels n* = 1.96^2 r(1-r)/(g_B + Delta/2)^2 needed so that a 95% interval
    for the level resolves the decision margin (evaluation pool; r = error
    rate in A_B; Delta treated as known).
Writes results/www2027/posthoc_audit/{guardrail_sensitivity.csv, guardrail_sensitivity.json}.
"""
import csv
import glob
import json
import sys
from math import lgamma
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paired_audit.estimators import LabelTerm, guardrail_decision  # noqa: E402
from paired_audit.identities import accepted_fixed, accepted_matched, audit_weights, d_conf, gap  # noqa: E402
from paired_audit.simulate_audit import CACHE, MAPS, MGRID, OUT, SPLITS, TS, pools  # noqa: E402

R, RNG0 = 400, 20261026


def p_no_error(N, E, n):
    if n > N - E:
        return 0.0
    return float(np.exp(lgamma(N - E + 1) - lgamma(N - E - n + 1) - lgamma(N + 1) + lgamma(N - n + 1)))


def main():
    pol = list(csv.DictReader(open(OUT / "policy_rows.csv")))
    rows, i, cell = [], 0, 0
    for f in sorted(glob.glob(str(CACHE / "cache_s*_n*.npz"))):
        z = np.load(f)
        s = int(Path(f).stem.split("_s")[1].split("_n")[0]); b = int(Path(f).stem.split("_n")[1])
        for sp in SPLITS:
            conf = z[f"conf_{sp}"][0].astype(np.float64); corr = z[f"corr_{sp}"].astype(np.float64)
            aud, ev = pools(z[f"rev_{sp}"]); cB = conf[0]
            for t in TS:
                for mi, mname in enumerate(MAPS[1:], start=1):
                    cC = conf[mi]
                    for rule in ("fixed", "matched"):
                        if rule == "fixed":
                            AC, AB = accepted_fixed(cC, t), accepted_fixed(cB, t)
                        else:
                            k = int(accepted_fixed(cB, t).sum()); AC, AB = accepted_matched(cC, k), accepted_matched(cB, k)
                        if not all((AC & mk).sum() >= 10 and (AB & mk).sum() >= 10 for mk in (aud, ev)):
                            continue
                        ACa, ABa = AC & aud, AB & aud
                        iB = np.where(ABa)[0]; NB = len(iB); EB = int((1 - corr[iB]).sum())
                        bB, cBB = corr[iB], cB[iB]
                        rho2 = float(np.corrcoef(bB, cBB)[0, 1] ** 2) if bB.std() > 0 and cBB.std() > 0 else 0.0
                        w = audit_weights(ACa, ABa); dconf = d_conf(cC, cB, ACa, ABa)
                        lt = LabelTerm(w, ACa, ABa, surrogate=cB)
                        gB_ev, gC_ev = gap(cB, corr, AB & ev), gap(cC, corr, AC & ev)
                        better = abs(gC_ev) < abs(gB_ev)
                        D_ev = gC_ev - gB_ev
                        r_ev = 1 - corr[AB & ev].mean()
                        margin = abs(gB_ev + D_ev / 2)
                        nstar = 1.96 ** 2 * r_ev * (1 - r_ev) / margin ** 2 if margin > 0 else float("inf")
                        cB_mean = float(cBB.mean())
                        for m in MGRID:
                            p = pol[i]; i += 1
                            assert p["method"] == mname and int(p["m"]) == m and p["rule"] == rule and abs(float(p["gB_eval"]) - gB_ev) < 1e-12
                            cell += 1
                            rng = np.random.default_rng(RNG0 + cell)
                            n = m // 2
                            sw = np.zeros(R, bool); reg = np.zeros(R)
                            for rr in range(R):
                                idx = rng.choice(NB, size=min(n, NB), replace=False)
                                x = float(bB[idx].sum()); nn = len(idx)
                                gB_hat = cB_mean - x / nn
                                pt = (nn - x + 2) / (nn + 4)                       # error proportion, Agresti-Coull
                                gB_var = (1 - nn / NB) * pt * (1 - pt) / nn
                                e, v, _ = lt.estimate("E6", corr, m - n, rng)
                                dec, _ = guardrail_decision(gB_hat, gB_var, dconf - e, v)
                                sw[rr] = dec
                                reg[rr] = (abs(gC_ev) if dec else abs(gB_ev)) - min(abs(gC_ev), abs(gB_ev))
                            rows.append(dict(seed=s, budget=b, split=sp, t=t, rule=rule, method=mname, m=m,
                                             C_better=bool(better), rho2_level=rho2, NB_audit=NB, errors_B_audit=EB,
                                             p_no_error_level=p_no_error(NB, EB, min(n, NB)),
                                             a8_switch=float(p["switch_rate"]), a8_false=float(p["false_switch_rate"]),
                                             a8_missed=float(p["missed_switch_rate"]), a8_regret=float(p["regret_mean"]),
                                             sens_switch=float(sw.mean()), sens_false=float(np.mean(sw & ~better)),
                                             sens_missed=float(np.mean(~sw & better)), sens_regret=float(reg.mean()),
                                             margin_eval=float(margin), nstar_level_labels=float(nstar), r_eval=float(r_ev)))
        print("done", f, flush=True)
    assert i == len(pol)
    with open(OUT / "guardrail_sensitivity.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator="\n"); wr.writeheader(); wr.writerows(rows)
    agg = {}
    for r in rows:
        agg.setdefault((r["budget"], r["split"], r["rule"], r["m"]), []).append(r)
    out = {}
    for k, v in sorted(agg.items()):
        bt = [x for x in v if x["C_better"]]
        out[f"n={k[0]}/{k[1]}/{k[2]}/m={k[3]}"] = dict(
            cells=len(v), better_cells=len(bt),
            rho2_level_median=float(np.median([x["rho2_level"] for x in v])), rho2_level_max=float(max(x["rho2_level"] for x in v)),
            p_no_error_mean=float(np.mean([x["p_no_error_level"] for x in v])),
            a8_switch=float(np.mean([x["a8_switch"] for x in v])), a8_false=float(np.mean([x["a8_false"] for x in v])),
            a8_missed_all=float(np.mean([x["a8_missed"] for x in v])),
            a8_missed_given_better=float(np.mean([1 - x["a8_switch"] for x in bt])) if bt else None,
            a8_regret_pts=100 * float(np.mean([x["a8_regret"] for x in v])),
            sens_switch=float(np.mean([x["sens_switch"] for x in v])), sens_false=float(np.mean([x["sens_false"] for x in v])),
            sens_missed_given_better=float(np.mean([1 - x["sens_switch"] for x in bt])) if bt else None,
            sens_regret_pts=100 * float(np.mean([x["sens_regret"] for x in v])),
            nstar_median_better=float(np.median([x["nstar_level_labels"] for x in bt])) if bt else None,
            nstar_range_better=[float(min(x["nstar_level_labels"] for x in bt)), float(max(x["nstar_level_labels"] for x in bt))] if bt else None)
    (OUT / "guardrail_sensitivity.json").write_text(json.dumps({"plan": "A9 note 3 (b)-(f)", "R": R, "rng0": RNG0, "summary": out}, indent=1) + "\n")
    for k, v in out.items():
        if k.startswith("n=2500/") and k.endswith(("m=25", "m=100", "m=1000")):
            print(k, {a: (round(b_, 3) if isinstance(b_, float) else b_) for a, b_ in v.items()})


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Plan A9 note 2: keep the four audit questions apart.
(Q-level) exact SRSWOR standard error of the E0 level estimate of g_B at
m/2 labels on the audit pool, against |g_B|; (Q-risk) accepted-set error
rate and coverage of B and C on the evaluation pool and the expected
change under the A8 guardrail policy (switch rate from policy_rows.csv)
next to the oracle-|g| choice.  Deterministic (no Monte Carlo).  Cells are
enumerated exactly as in simulate_audit.py; each row is checked against
policy_rows.csv (gB_eval, gC_eval to 1e-12).
Writes results/www2027/posthoc_audit/{question_separation.csv, question_separation.json}.
"""
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paired_audit.identities import accepted_fixed, accepted_matched, gap  # noqa: E402
from paired_audit.simulate_audit import CACHE, MAPS, MGRID, OUT, SPLITS, TS, pools  # noqa: E402


def main():
    pol = list(csv.DictReader(open(OUT / "policy_rows.csv")))
    rows, i = [], 0
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
                        ok = all((AC & mk).sum() >= 10 and (AB & mk).sum() >= 10 for mk in (aud, ev))
                        if not ok:
                            continue
                        ABa = AB & aud; NB = int(ABa.sum()); sb2 = float(corr[ABa].var(ddof=1))
                        gB_a = gap(cB, corr, ABa)
                        ACe, ABe = AC & ev, AB & ev
                        gB_e, gC_e = gap(cB, corr, ABe), gap(cC, corr, ACe)
                        riskB, riskC = 1 - corr[ABe].mean(), 1 - corr[ACe].mean()
                        covB, covC = ABe.sum() / ev.sum(), ACe.sum() / ev.sum()
                        better = abs(gC_e) < abs(gB_e)
                        for m in MGRID:
                            p = pol[i]; i += 1
                            assert p["method"] == mname and int(p["m"]) == m and p["rule"] == rule
                            assert abs(float(p["gB_eval"]) - gB_e) < 1e-12 and abs(float(p["gC_eval"]) - gC_e) < 1e-12
                            n = m // 2
                            se = np.sqrt((1 - n / NB) * sb2 / n) if n < NB else 0.0
                            sw = float(p["switch_rate"])
                            rows.append(dict(seed=s, budget=b, split=sp, t=t, rule=rule, method=mname, m=m,
                                             gB_audit=gB_a, level_se_half_m=float(se),
                                             level_sign_resolvable=bool(1.96 * se < abs(gB_a)),
                                             risk_B=float(riskB), risk_C=float(riskC), cov_B=float(covB), cov_C=float(covC),
                                             switch_rate=sw, exp_risk_change_policy=sw * float(riskC - riskB),
                                             exp_cov_change_policy=sw * float(covC - covB),
                                             risk_change_oracle_g=float(riskC - riskB) if better else 0.0,
                                             cov_change_oracle_g=float(covC - covB) if better else 0.0))
    assert i == len(pol), (i, len(pol))
    with open(OUT / "question_separation.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator="\n"); w.writeheader(); w.writerows(rows)
    agg = {}
    for r in rows:
        agg.setdefault((r["budget"], r["split"], r["rule"], r["m"]), []).append(r)
    out = {}
    for k, v in sorted(agg.items()):
        out[f"n={k[0]}/{k[1]}/{k[2]}/m={k[3]}"] = dict(
            cells=len(v),
            level_se_half_m_pts=100 * float(np.mean([x["level_se_half_m"] for x in v])),
            abs_gB_pts=100 * float(np.mean([abs(x["gB_audit"]) for x in v])),
            share_level_sign_resolvable=float(np.mean([x["level_sign_resolvable"] for x in v])),
            risk_change_if_always_switch_pts=100 * float(np.mean([x["risk_C"] - x["risk_B"] for x in v])),
            cov_change_if_always_switch_pts=100 * float(np.mean([x["cov_C"] - x["cov_B"] for x in v])),
            exp_risk_change_policy_pts=100 * float(np.mean([x["exp_risk_change_policy"] for x in v])),
            exp_cov_change_policy_pts=100 * float(np.mean([x["exp_cov_change_policy"] for x in v])),
            risk_change_oracle_g_pts=100 * float(np.mean([x["risk_change_oracle_g"] for x in v])),
            cov_change_oracle_g_pts=100 * float(np.mean([x["cov_change_oracle_g"] for x in v])))
    (OUT / "question_separation.json").write_text(json.dumps({"plan": "A9 note 2", "summary": out}, indent=1) + "\n")
    for k, v in out.items():
        if k.startswith("n=2500/") and "/fixed/" in k and k.endswith(("m=100", "m=1000")):
            print(k, {a: round(b_, 3) for a, b_ in v.items()})


if __name__ == "__main__":
    main()

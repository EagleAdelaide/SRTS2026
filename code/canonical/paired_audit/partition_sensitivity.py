#!/usr/bin/env python3
"""Plan A11 (ii): reviewer-partition sensitivity of full-label winners (no simulated
audits).  Existing per-item caches (posthoc_mechanism/cache_s{s}_n{b}.npz, calibration
draw 0, as in the audit).  Partition 0 = the original split; partitions 1..50 =
sha256("srts-partition-{k}-{reviewer}") parity; one item-random diagnostic partition
(sha256("srts-item-{index}") parity).  For every partition, both directions, splits
id_test (seen) and ood_val (unseen), budgets 2500 and 250, fixed thresholds .80/.90/.95,
four maps vs TvA-TS, seeds 0-4: g_B and g_C in source and target pools; source-optimal
choice; target regret of that choice; always-scalar and always-map target regrets;
winner flips at tolerances eps (margin >= eps in BOTH pools).
Writes results/www2027/posthoc_partition/{partition_rows.csv, partition_summary.json}.
"""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

W = Path("__SRTS_ROOT__/calattn-repo/results/www2027")
CACHE = W / "posthoc_mechanism"
OUT = W / "posthoc_partition"
MAPS = ["TvA-TS", "HTS-BCE", "SRTS-BCE", "SMART+BCE", "PWLinear-3"]
SPLITS = ["id_test", "ood_val"]
TS = [0.80, 0.90, 0.95]
K = 50
EPS = [0.0, 0.0005, 0.001, 0.002]


def h2(s):
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16) % 2


def pool_of(k, rev, n):
    if k == 0:
        u = np.unique(rev); m = {r: h2(str(int(r))) for r in u}
        return np.array([m[r] for r in rev], dtype=np.int8)
    if k == "item":
        return np.array([h2(f"srts-item-{i}") for i in range(n)], dtype=np.int8)
    u = np.unique(rev); m = {r: h2(f"srts-partition-{k}-{int(r)}") for r in u}
    return np.array([m[r] for r in rev], dtype=np.int8)


def g(c, b, A):
    return float((c[A] - b[A]).mean()) if A.any() else float("nan")


def main():
    OUT.mkdir(exist_ok=True)
    rows = []
    parts = list(range(K + 1)) + ["item"]
    for s in range(5):
        for bud in (2500, 250):
            z = np.load(CACHE / f"cache_s{s}_n{bud}.npz")
            for sp in SPLITS:
                conf = z[f"conf_{sp}"][0].astype(np.float64); b = z[f"corr_{sp}"].astype(np.float64); rev = z[f"rev_{sp}"]
                cB = conf[0]
                pools = {k: pool_of(k, rev, len(b)) for k in parts}
                for k in parts:
                    P = pools[k]
                    for t in TS:
                        AB = cB >= t
                        for mi, m in enumerate(MAPS[1:], start=1):
                            AC = conf[mi] >= t
                            gg = {p: (g(cB, b, AB & (P == p)), g(conf[mi], b, AC & (P == p))) for p in (0, 1)}
                            for src, tgt in ((0, 1), (1, 0)):
                                gBs, gCs = gg[src]; gBt, gCt = gg[tgt]
                                ms, mt = abs(gBs) - abs(gCs), abs(gBt) - abs(gCt)     # >0: map better
                                best_t = min(abs(gBt), abs(gCt))
                                choice_t = abs(gCt) if ms > 0 else abs(gBt)
                                r = dict(partition=k, direction=f"{src}->{tgt}", budget=bud, split=sp, seed=s, t=t, method=m,
                                         gB_src=gBs, gC_src=gCs, gB_tgt=gBt, gC_tgt=gCt, margin_src=ms, margin_tgt=mt,
                                         regret_src_opt=choice_t - best_t, regret_always_scalar=abs(gBt) - best_t,
                                         regret_always_map=abs(gCt) - best_t)
                                for e in EPS:
                                    r[f"flip_eps{e*100:.2f}"] = int((ms > 0) != (mt > 0) and abs(ms) >= e and abs(mt) >= e)
                                rows.append(r)
            print("done", s, bud, flush=True)
    with open(OUT / "partition_rows.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator="\n"); w.writeheader(); w.writerows(rows)
    # summaries per (partition, direction, budget, split)
    agg = defaultdict(list)
    for r in rows:
        agg[(str(r["partition"]), r["direction"], r["budget"], r["split"])].append(r)
    per = {}
    for k, v in agg.items():
        d = dict(cells=len(v), regret_src_opt_pp=100 * float(np.mean([x["regret_src_opt"] for x in v])),
                 regret_always_scalar_pp=100 * float(np.mean([x["regret_always_scalar"] for x in v])),
                 regret_always_map_pp=100 * float(np.mean([x["regret_always_map"] for x in v])),
                 max_cell_regret_src_opt_pp=100 * float(max(x["regret_src_opt"] for x in v)),
                 seeds_with_flip_eps0p10=int(len({x["seed"] for x in v if x["flip_eps0.10"]})))
        for e in EPS:
            d[f"flips_eps{e*100:.2f}"] = int(sum(x[f"flip_eps{e*100:.2f}"] for x in v))
        per["/".join(map(str, k))] = d
    dist = {}
    for bud in (2500, 250):
        for sp in SPLITS:
            for dr in ("0->1", "1->0"):
                vals = [per[f"{k}/{dr}/{bud}/{sp}"] for k in range(1, K + 1)]
                dd = {}
                for key in [f"flips_eps{e*100:.2f}" for e in EPS] + ["regret_src_opt_pp", "regret_always_scalar_pp", "regret_always_map_pp", "seeds_with_flip_eps0p10"]:
                    a = np.array([x[key] for x in vals], dtype=float)
                    dd[key] = dict(median=float(np.median(a)), p10=float(np.percentile(a, 10)), p90=float(np.percentile(a, 90)),
                                   min=float(a.min()), max=float(a.max()), original=per[f"0/{dr}/{bud}/{sp}"][key],
                                   item_random=per[f"item/{dr}/{bud}/{sp}"][key])
                dist[f"n={bud}/{sp}/{dr}"] = dd
    # GO rule (A11): n=2500 seen, median flips at eps=0.1 pp >= 3 of 60 AND flips in >= 3 of 5 seeds in >= half the partitions
    go = {}
    for dr in ("0->1", "1->0"):
        fl = [per[f"{k}/{dr}/2500/id_test"]["flips_eps0.10"] for k in range(1, K + 1)]
        sd = [per[f"{k}/{dr}/2500/id_test"]["seeds_with_flip_eps0p10"] for k in range(1, K + 1)]
        go[dr] = dict(median_flips_eps0p10=float(np.median(fl)), share_partitions_with_ge3_seeds=float(np.mean([x >= 3 for x in sd])),
                      retain=bool(np.median(fl) >= 3 and np.mean([x >= 3 for x in sd]) >= 0.5))
    # breakdown by seed, threshold, map (n=2500, seen, over partitions 1..50, both directions)
    br = defaultdict(list)
    for r in rows:
        if r["budget"] == 2500 and r["split"] == "id_test" and r["partition"] not in (0, "item"):
            br[("seed", r["seed"])].append(r); br[("t", r["t"])].append(r); br[("method", r["method"])].append(r)
    breakdown = {f"{a}={b}": dict(cells=len(v), flip_rate_eps0p10=float(np.mean([x["flip_eps0.10"] for x in v])),
                                  mean_regret_src_opt_pp=100 * float(np.mean([x["regret_src_opt"] for x in v])))
                 for (a, b), v in sorted(br.items(), key=lambda kv: str(kv[0]))}
    (OUT / "partition_summary.json").write_text(json.dumps(dict(plan="A11 (ii)", K=K, eps=EPS, per_partition=per,
                                                                  distribution=dist, go_rule=go, breakdown_seen_n2500=breakdown), indent=1) + "\n")
    print(json.dumps(dict(go_rule=go, seen_2500={dr: {k: dist[f'n=2500/id_test/{dr}'][k] for k in ('flips_eps0.10', 'regret_src_opt_pp')} for dr in ('0->1', '1->0')}), indent=1))


if __name__ == "__main__":
    main()

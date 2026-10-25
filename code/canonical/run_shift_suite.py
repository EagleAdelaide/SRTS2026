#!/usr/bin/env python3
"""C100-C deployable shift suite under the canonical registry (WP0/WP6).

Protocol (identical to the earlier deployable path, now 5 seeds): fit every
core method ONCE on the seed's full clean calibration split (n=2500);
freeze; evaluate on each of the 95 (corruption x severity) cells, each
corrupted sample routed from ITS OWN logits. No corrupted labels, no
clean-pairing, no refitting.

Rows: one per (seed, corruption, severity, method) with full registry +
ops metrics. Output: results/www2027/shift/c100c_core.jsonl
"""
import json
import os
import re
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import registry as R
from run_suite import fit_methods, load_cell

SOTA = Path("__SRTS_ROOT__/anchor-sota/results")
OUT = Path("__SRTS_ROOT__/calattn-repo/results/www2027/shift")
OUT.mkdir(parents=True, exist_ok=True)


def c100c_cells(seed):
    cells = {}
    pat = re.compile(
        rf"ANCHOR_C100C_ViT224_CE_v1_NoTS_s{seed}_(.+)_s(\d)$")
    for d in SOTA.iterdir():
        m = pat.match(d.name)
        if m and (d / "logits_test.npz").exists():
            cells[(m.group(1), int(m.group(2)))] = d
    return cells


def main():
    seeds = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1
                              else ["0", "1", "2", "3", "4"])]
    fout = OUT / f"c100c_core_s{'-'.join(map(str, seeds))}.jsonl"
    done = set()
    if fout.exists():
        for line in fout.open():
            try:
                r = json.loads(line)
                done.add((r["seed"], r["corruption"], r["severity"],
                          r["method"]))
            except Exception:
                pass
    with fout.open("a") as fh:
        for s in seeds:
            vl, vt, _, _ = load_cell("c100_vit", s, False)
            fitted = fit_methods("core", vl, vt, s)
            cells = c100c_cells(s)
            print(f"seed {s}: {len(cells)} cells", flush=True)
            for (corr, sev), d in sorted(cells.items()):
                z = np.load(d / "logits_test.npz")
                cl = z["logits"].astype(np.float64)
                ct = z["targets"].astype(np.int64)
                for name, (pred, _) in fitted.items():
                    if (s, corr, sev, name) in done:
                        continue
                    P = pred(cl)
                    m = R.all_metrics(P, ct)
                    m.update(R.ops_metrics(P, ct))
                    fh.write(json.dumps(dict(
                        cell="c100c_vit", seed=s, corruption=corr,
                        severity=sev, method=name,
                        argmax_changes=int((P.argmax(1) != cl.argmax(1)).sum()),
                        **m), default=float) + "\n")
                fh.flush()
            print(f"seed {s} done", flush=True)
    print("DONE", fout)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Corruption-shift summary used in the text: per seed, mean over the 95
CIFAR-100-C cells of SRTS-BCE minus TvA-TS and minus SMART+BCE (ECE15).
Writes results/www2027/shift_summary.json."""
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

W = Path("__SRTS_ROOT__/calattn-repo/results/www2027")
rows = [json.loads(l) for f in sorted(glob.glob(str(W / "shift/c100c_core_s*.jsonl"))) for l in open(f)]
e = {(r["seed"], r["corruption"], r["severity"], r["method"]): r["ece15"] for r in rows}
seeds = sorted({k[0] for k in e})
cells = sorted({(k[1], k[2]) for k in e})
out = {"n_cells": len(cells), "seeds": seeds}
for ref in ("TvA-TS", "SMART+BCE"):
    ps = [float(np.mean([e[(s, c, v, "SRTS-BCE")] - e[(s, c, v, ref)] for c, v in cells])) for s in seeds]
    out[f"SRTS-BCE - {ref}"] = {"mean": float(np.mean(ps)), "per_seed": ps,
                               "seeds_negative": int(sum(p < 0 for p in ps))}
out["mean_ece15"] = {m: float(np.mean([e[(s, c, v, m)] for s in seeds for c, v in cells]))
                     for m in sorted({k[3] for k in e})}
(W / "shift_summary.json").write_text(json.dumps(out, indent=1))
print({k: (round(v["mean"], 3), v["seeds_negative"]) for k, v in out.items() if k.startswith("SRTS")})

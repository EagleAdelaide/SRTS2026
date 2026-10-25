#!/usr/bin/env python3
"""WP8 shrinkage pilot/confirmation driver (see shrinkage.py docstring)."""
import sys, json, os
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "4")
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_suite import load_cell
from sampler import natural_subsample
from shrinkage import SRTSShrink
from srts import SRTS, fit_tva
import registry as R

cell = sys.argv[1]; seeds = [int(x) for x in sys.argv[2].split(",")]
rows=[]
for s in seeds:
    vl_p, vt_p, tests, _ = load_cell(cell, s, False)
    split = 'test' if 'test' in tests else 'ood_val'
    tl, tt = tests[split]
    for b in (250, 2500):
        nd = 1 if b >= len(vt_p) else 20
        for r_ in range(nd):
            rng = np.random.default_rng(77_000_000+1_000_000*s+1000*b+r_)
            idx,_ = natural_subsample(vt_p, b, rng)
            vl, vt = vl_p[idx], vt_p[idx]
            sh = SRTSShrink(seed=s).fit(vl, vt)
            sr = SRTS(K=3, seed=s, scaler="fold", fold_fallback="train").fit(vl, vt)
            tv = fit_tva(vl, vt)
            for name, m in (("Shrink", sh), ("SRTS", sr), ("TvA", tv)):
                P = m.predict_proba(tl)
                rows.append(dict(cell=cell, split=split, seed=s, budget=b,
                                 draw=r_, method=name, ece15=R.ece(P,tt),
                                 topbce=R.topbce(P,tt),
                                 alpha=sh.report['alpha'] if name=="Shrink" else None))
    print("seed", s, "done", flush=True)
out = Path(f'__SRTS_ROOT__/calattn-repo/results/www2027/wp8_shrink_{cell}.json')
json.dump(rows, open(out,'w')); print("DONE", out)

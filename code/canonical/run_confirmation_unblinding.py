#!/usr/bin/env python3
"""SINGLE confirmation unblinding (WEB_PROTOCOL §3/§10). DO NOT RUN until
the freeze commit is recorded in WORK_LOG. Refuses to run twice.

Evaluates, on amazon ood_test + yahoo conf_test ONLY:
  core methods + SRTS-Shrink, at the frozen budgets/draws, natural
  sampling, identical draw RNG as development. Also reviewer-subgroup
  reliability on amazon ood_test (subgroups >= 200 items).
Writes results/www2027/CONFIRMATION/<cell>.jsonl + UNBLINDED.stamp.
"""
import json
import os
import sys
import time
from pathlib import Path

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "4")
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import registry as R
from run_suite import load_cell, fit_methods
from sampler import natural_subsample
from shrinkage import SRTSShrink

OUT = Path("__SRTS_ROOT__/calattn-repo/results/www2027/CONFIRMATION")
STAMP = OUT / "UNBLINDED.stamp"


def _verify_freeze_gate():
    """Hard gate: FREEZE_MANIFEST.json must exist, canonical scripts must
    hash-match it, and the repo HEAD must be the recorded commit."""
    import hashlib
    import subprocess
    repo = Path("__SRTS_ROOT__/calattn-repo")
    man_p = repo / "www2027/docs/FREEZE_MANIFEST.json"
    if not man_p.exists():
        sys.exit("FREEZE GATE: FREEZE_MANIFEST.json missing "
                 "(run make_freeze_manifest.py at the freeze commit)")
    man = json.load(open(man_p))
    # integrity invariant: every frozen script hash-matches NOW, and the
    # canonical directory has ZERO changes since the frozen commit (the
    # manifest's own later commit does not touch canonical/).
    frozen_all = man.get("all_canonical_sha256", man["script_sha256"])
    current = {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
               for f in sorted(HERE.glob("*.py"))}
    changed = [n for n, h in frozen_all.items()
               if current.get(n) != h]
    added = [n for n in current if n not in frozen_all]
    if changed or added:
        sys.exit(f"FREEZE GATE: canonical/ differs from manifest: "
                 f"changed={changed} added={added}")
    # optional belt-and-suspenders when git is available on this node
    import shutil
    if shutil.which("git"):
        diff = subprocess.run(
            ["git", "-C", str(repo), "diff", "--name-only",
             f"{man['commit']}..HEAD", "--", "www2027/scripts/canonical"],
            capture_output=True, text=True).stdout.strip()
        if diff:
            sys.exit(f"FREEZE GATE: canonical/ changed since frozen commit "
                     f"{man['commit'][:9]}:\n{diff}")
    print(f"FREEZE GATE OK: frozen {man['commit'][:9]}, "
          f"{len(frozen_all)} canonical files hash-verified")
    return man


def main():
    if "--i-confirm-freeze" not in sys.argv:
        sys.exit("Refusing: pass --i-confirm-freeze after recording the "
                 "freeze commit in WORK_LOG.md")
    _verify_freeze_gate()
    OUT.mkdir(parents=True, exist_ok=True)
    cells = {"amazon_distilbert": [0, 1, 2, 3, 4],
             "yahoo_distilbert": [0, 1, 2]}
    arg_cells = [a for a in sys.argv[1:] if a in cells]
    if arg_cells:
        cells = {c: cells[c] for c in arg_cells}
    for cell, seeds in cells.items():
        stamp_c = OUT / f"UNBLINDED_{cell}.stamp"
        if stamp_c.exists():
            print(f"{cell}: already unblinded at "
                  f"{stamp_c.read_text().strip()} -- refusing re-run; any "
                  "further evaluation is a corrected re-evaluation and "
                  "must be run under a different, labelled path.")
            continue
        fout = OUT / f"{cell}.jsonl"
        with fout.open("a") as fh:
            for s in seeds:
                vl_p, vt_p, tests, _ = load_cell(cell, s, True)
                key = [k for k in tests if "CONFIRMATION" in k]
                if not key:
                    print(f"{cell} s{s}: confirmation split missing")
                    continue
                tl, tt = tests[key[0]]
                # reviewer metadata for subgroups (amazon)
                rev = None
                if cell.startswith("amazon"):
                    base = Path("__SRTS_ROOT__/data/wilds/tokcache")
                    m = np.load(base / "distilbert_meta.npz")
                    z = np.load(Path(
                        "__SRTS_ROOT__/calattn-repo/results/www2027"
                        f"/wilds_amazon/distilbert_s{s}/logits_test.npz"))
                    rev = m["reviewer"][z["row_idx"]]
                for b in (250, 625, 1250, 2500):
                    nd = 1 if b >= len(vt_p) else 20
                    for r_ in range(nd):
                        rng = np.random.default_rng(
                            77_000_000 + 1_000_000 * s + 1000 * b + r_)
                        idx, rec = natural_subsample(vt_p, b, rng)
                        vl, vt = vl_p[idx], vt_p[idx]
                        fitted = fit_methods("core", vl, vt, s)
                        sh = SRTSShrink(seed=s).fit(vl, vt)
                        fitted["SRTS-Shrink"] = (sh.predict_proba, sh.report)
                        for name, (pred, rep) in fitted.items():
                            P = pred(tl)
                            mm = R.all_metrics(P, tt,
                                               with_smece=len(tt) <= 200000)
                            mm.update(R.ops_metrics(P, tt))
                            mm.update(R.risk_at_coverage(P, tt))
                            row = dict(cell=cell, seed=s, budget=b, draw=r_,
                                       method=name, test_split="CONFIRMATION",
                                       sampler=rec, **mm)
                            if rev is not None and name in ("TvA-TS",
                                                            "SMART+BCE",
                                                            "SRTS-BCE"):
                                sub = {}
                                for rid in np.unique(rev):
                                    msk = rev == rid
                                    if msk.sum() >= 200:
                                        sub[int(rid)] = dict(
                                            n=int(msk.sum()),
                                            ece15=R.ece(P[msk], tt[msk]),
                                            errgap=R.ops_metrics(
                                                P[msk], tt[msk])["errgap_a90"])
                                row["reviewer_subgroups"] = sub
                            fh.write(json.dumps(row, default=float) + "\n")
                        fh.flush()
                    print(f"{cell} s{s} b{b} done", flush=True)
        stamp_c.write_text(time.strftime("%F %T"))
    STAMP.write_text(time.strftime("%F %T"))
    print("CONFIRMATION COMPLETE")


if __name__ == "__main__":
    main()

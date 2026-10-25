#!/usr/bin/env python3
"""Create FREEZE_MANIFEST.json — the gate for the single confirmation
unblinding. Records: git commit (must be clean tree), SHA-256 of every
canonical script participating in confirmation, the frozen method list,
primary endpoints/comparisons, draw RNG formula, and budgets. The
unblinding script refuses to run unless this manifest exists AND the
scripts' hashes still match AND the repo is on the recorded commit.
"""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("__SRTS_ROOT__/calattn-repo")
CAN = REPO / "www2027/scripts/canonical"
OUT = REPO / "www2027/docs/FREEZE_MANIFEST.json"

SCRIPTS = ["registry.py", "sampler.py", "srts.py", "smart.py", "maps.py",
           "shrinkage.py", "run_suite.py", "run_confirmation_unblinding.py"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain",
                            "--", "www2027/scripts/canonical"],
                           capture_output=True, text=True).stdout.strip()
    if dirty:
        sys.exit(f"Refusing to freeze: canonical scripts dirty:\n{dirty}")
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    man = {
        "frozen_at": time.strftime("%F %T %Z"),
        "commit": commit,
        "script_sha256": {s: sha(CAN / s) for s in SCRIPTS},
        "all_canonical_sha256": {f.name: sha(f)
                                 for f in sorted(CAN.glob("*.py"))},
        "methods": ["NoTS", "TS-NLL", "TvA-TS", "HTS-NLL", "HTS-BCE",
                    "SMART-SoftECE", "SMART+BCE", "Margin-K3", "PWLinear-3",
                    "SRTS-BCE", "SRTS-Shrink"],
        "budgets": [250, 625, 1250, 2500],
        "draws": 20,
        "sampling": "natural",
        "draw_rng": "default_rng(77e6 + 1e6*seed + 1000*budget + draw)",
        "primary_endpoints": ["ece15", "count_rel_err/errgap_a90"],
        "primary_comparisons": [
            "SRTS-BCE - TvA-TS @250", "SRTS-BCE - SMART+BCE @250",
            "SRTS-BCE - TvA-TS @2500", "SRTS-BCE - SMART+BCE @2500",
            "interaction(250-2500) vs TvA-TS",
            "interaction(250-2500) vs SMART+BCE"],
        "inference": "seed->draw hierarchical bootstrap B=20000, "
                     "rng 20261009; CI incl 0 => not statistically resolved",
        "confirmation_sets": {
            "amazon_distilbert": "WILDS official OOD test (disjoint reviewers)",
            "yahoo_distilbert": "official yahoo_answers_topics test table"},
        "statistical_limits": "yahoo: 3 training seeds (seed-cluster interval "
                              "is a spread, not a calibrated CI); amazon: 5",
    }
    json.dump(man, open(OUT, "w"), indent=1)
    print("FROZEN at", commit)
    print("wrote", OUT)


if __name__ == "__main__":
    main()

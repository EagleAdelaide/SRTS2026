#!/usr/bin/env python3
"""Plan addendum A5 (ii) and note 2: pairing checks for the seen-vs-unseen
contrast, run before any interval. (a) completeness of the A4(iii) rows;
(b) calibration indices regenerated from the shared RNG rule vs the sampler
records of the original rows; (c) A4(iii) refit vs original rows (reported);
(d) logits/labels read by the refit vs the released MANIFEST and float32
logits. (b) and (d) need the validation labels and the logits (layer A).
Writes results/www2027/posthoc_gamma/pairing_check.json; exit 1 on failure.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_gamma import OUT, load_matched, load_original, pairing_checks  # noqa: E402


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tab, dup = load_matched()
    orig = load_original()
    seeds = sorted({k[1] for k in tab})
    draws = sorted({k[3] for k in tab})
    chk = pairing_checks(tab, dup, orig, seeds, draws)
    chk.update(seeds=seeds, draws=len(draws))
    (OUT / "pairing_check.json").write_text(json.dumps(chk, indent=1) + "\n")
    print("pairing:", {k: v for k, v in chk.items()
                       if k not in ("calibration_index_sha256_16", "model_version")})
    sys.exit(0 if chk["passed"] and chk["model_version_ok"] else 1)


if __name__ == "__main__":
    main()

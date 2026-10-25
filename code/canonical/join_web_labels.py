#!/usr/bin/env python3
"""Re-join Web labels from the OFFICIAL datasets onto the released logits,
verify them against the released SHA-256, and write the npz layout used by
run_suite.load_cell (logits + targets).

  python join_web_labels.py --release data/web_logits \
      --wilds-root /path/to/wilds_data --out __SRTS_ROOT__/calattn-repo/results/www2027

Needs: `wilds` (Amazon v2.1, official download) and `datasets`
(yahoo_answers_topics). Fails loudly on any hash mismatch.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.int64).tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", required=True)
    ap.add_argument("--wilds-root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rel = Path(a.release)
    man = json.loads((rel / "MANIFEST.json").read_text())
    tables = {}

    def table(name):
        if name in tables:
            return tables[name]
        if name.startswith("wilds/amazon"):
            from wilds import get_dataset
            ds = get_dataset("amazon", root_dir=a.wilds_root, download=False)
            tables[name] = ds.y_array.numpy().astype(np.int64)
            tables["amazon_reviewer"] = ds.metadata_array[:, 0].numpy().astype(np.int64)
        else:
            from datasets import load_dataset
            _, part, col = name.split(":")
            d = load_dataset("yahoo_answers_topics")
            tables[name] = np.asarray(d[part][col], dtype=np.int64)
        return tables[name]

    for rel_path, info in man["files"].items():
        z = np.load(rel / rel_path)
        y = table(info["table"])[z["official_row"]]
        if sha(y) != info["targets_sha256"]:
            raise SystemExit(f"label hash mismatch for {rel_path}")
        cell, split = rel_path.split("/")
        cell_name, seed = cell.rsplit("_s", 1)
        sub = ("wilds_amazon" if cell_name.startswith("amazon") else "yahoo")
        d = Path(a.out) / sub / f"distilbert_s{seed}"
        d.mkdir(parents=True, exist_ok=True)
        extra = {}
        if sub == "wilds_amazon":     # reviewer ids for the exploratory analyses
            extra["reviewer"] = tables["amazon_reviewer"][z["official_row"]]
        np.savez_compressed(d / info["suite_filename"], logits=z["logits"],
                            targets=y, row_idx=z["official_row"], **extra)
        print("ok", rel_path, len(y))
    print("all label hashes verified")


if __name__ == "__main__":
    main()

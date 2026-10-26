#!/usr/bin/env python3
"""Appendix tables for the faithful extended baselines and the measured
cost, recovered from the commands that wrote them (2026-10-10).
Writes paper/generated/ext_baselines.tex and efficiency_amazon.tex."""
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path("__SRTS_ROOT__/calattn-repo")
W = REPO / "results/www2027"
GEN = REPO / "www2027/paper/generated"


def load(g, split):
    rows = []
    for f in glob.glob(str(W / g)):
        rows += [json.loads(l) for l in open(f)]
    return [r for r in rows if r["test_split"] == split]


def ext_baselines():
    cells = {"anchor": load("suite_ext/c100_vit_s*/c100_vit_ext_natural.jsonl", "test"),
             "amazon": load("suite_ext/amazon_s*/amazon_distilbert_ext_natural.jsonl", "id_test")}
    # citation audit (2026-10-26): original-method citations beside the names; the +BCE row credits the
    # cited architecture only (BCE objective evaluated here); LogitMLP-T is an internal control (no citation).
    # Short labels keep every citation on one line within the column: 'Sem.-Aware' (full name Semantic-Aware
    # Grouping in Appendix A) and 'Meta-Cal' (its confidence view is named in the caption).
    METH = [("PTS", "PTS (MSE)~\\cite{tomani2022pts}"), ("PTS-BCE", "PTS+BCE~\\cite{tomani2022pts}"),
            ("AdaTS", "AdaTS (CondVAE)~\\cite{joy2023sample}"), ("SAG", "Sem.-Aware~\\cite{yang2023semantic}"),
            ("DirODIR", "Dirichlet-ODIR~\\cite{kull2019dirichlet}"), ("ETS", "Mix-n-Match ETS~\\cite{zhang2020mixnmatch}"),
            ("MetaCal", "Meta-Cal~\\cite{ma2021metacal}"), ("LogitMLP-T", "LogitMLP-T")]

    def ms(rows, m, b):
        per = defaultdict(list)
        for r in rows:
            if r["method"] == m and r["budget"] == b:
                per[r["seed"]].append(r["ece15"])
        if not per:
            return "---"
        sm = np.array([np.mean(v) for v in per.values()])
        return f"{sm.mean():.2f}$\\pm${sm.std():.2f}"
    lines = [f"{disp} & {ms(cells['anchor'], k, 250)} & {ms(cells['anchor'], k, 2500)} & "
             f"{ms(cells['amazon'], k, 250)} & {ms(cells['amazon'], k, 2500)} \\\\" for k, disp in METH]
    (GEN / "ext_baselines.tex").write_text("\n".join(lines) + "\n")


def efficiency():
    e = json.load(open(W / "efficiency_amazon_distilbert.json"))
    rows = []
    for m in ("TS-NLL", "TvA-TS", "HTS-BCE", "SRTS-BCE", "SMART+BCE"):
        v = e[m]
        rows.append(f"{m} & {v['fit_seconds_p50']:.2f} & {v['batch1_us_p50']:.0f} & "
                    f"{v['batch256_us_p50']:.2f} & {v['serialized_bytes']} \\\\")
    (GEN / "efficiency_amazon.tex").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    ext_baselines(); efficiency()
    print("wrote ext_baselines.tex efficiency_amazon.tex")

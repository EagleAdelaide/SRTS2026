#!/usr/bin/env python3
"""Generate the revision's main-text tables from saved analysis files.

  generated/q1_ece_table.tex     ECE15 differences, anchor (development) and
                                 both confirmation sets; frozen primaries from
                                 the frozen analysis files; other rows post-hoc
                                 (anchor rows computed here, confirmation rows
                                 from posthoc_endpoints/endpoints.json)
  generated/f2_table.tex         2x2 score-construction control (plan P2)
  generated/decomp_table.tex     t = 0.90 accepted / rejected decomposition,
                                 n = 2500, both confirmation sets (plan P3)
No number is typed by hand.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path("__SRTS_ROOT__/calattn-repo")
W = REPO / "results/www2027"
GEN = REPO / "www2027/paper/generated"
B = 20000
# citation of the original method, appended to the first occurrence in a table
CITE = {"TvA-TS": "~\\cite{lecoz2024tva}", "HTS-BCE": "~\\cite{balanya2024adaptive}",
        "SMART+BCE": "~\\cite{guo2026smart}"}


def signed0(x):
    """Integer with sign; values that round to zero print as 0 (no -0)."""
    r = int(round(x))
    return "0" if r == 0 else f"{r:+d}"


def signed1(x):
    """One decimal with sign; values that round to zero print as 0.0."""
    s = f"{x:+.1f}"
    return "0.0" if s in ("+0.0", "-0.0") else s


def fmt(p, lo, hi, d=2):
    return (f"${p:+.{d}f}$ {{\\scriptsize[${lo:+.{d}f}$,${hi:+.{d}f}$]}}"
            .replace("+", "{+}").replace("-", "{-}"))


def anchor_desc(method, b, rng):
    rows = []
    for f in sorted((W / "suite").glob("c100_vit_s*/c100_vit_core_natural.jsonl")):
        rows += [json.loads(l) for l in open(f)]
    tab = {(r["seed"], r["draw"], r["method"]): r["ece15"] for r in rows
           if r["test_split"] == "test" and r["budget"] == b}
    seeds = sorted({k[0] for k in tab})
    draws = sorted({k[1] for k in tab})
    M = np.array([[tab[(s, d, method)] - tab[(s, d, "TvA-TS")] for d in draws]
                  for s in seeds])
    if all(np.ptp(M[i]) == 0 for i in range(len(seeds))):
        M = M[:, :1]                           # full-pool replicates
    S, D = M.shape
    i_s = rng.integers(0, S, (B, S))
    bt = (M[i_s, 0].mean(1) if D == 1 else
          M[i_s[:, :, None], rng.integers(0, D, (B, S, D))].mean(2).mean(1))
    lo, hi = np.percentile(bt, [2.5, 97.5])
    return float(M.mean(1).mean()), float(lo), float(hi)


def q1():
    rng = np.random.default_rng(20261016)
    frozen = {"anchor": json.load(open(W / "analysis_c100_core_natural_ece15.json"))["c100_vit"],
              "amazon": json.load(open(W / "analysis_CONFIRM_amazon_distilbert_ece15.json"))["amazon_distilbert"],
              "yahoo": json.load(open(W / "analysis_CONFIRM_yahoo_distilbert_ece15.json"))["yahoo_distilbert"]}
    ep = json.load(open(W / "posthoc_endpoints/endpoints.json"))["tasks"]
    cols = [("anchor", 250), ("anchor", 2500), ("amazon", 250),
            ("amazon", 2500), ("yahoo", 250), ("yahoo", 2500)]
    lines = []
    for lab, key in (("SRTS-BCE $-$ scalar~\\cite{lecoz2024tva}$^\\dagger$", "SRTS-BCE - TvA-TS"),
                     ("SRTS-BCE $-$ SMART+BCE~\\cite{guo2026smart}$^\\dagger$", "SRTS-BCE - SMART+BCE")):
        cells = []
        for t, b in cols:
            v = frozen[t][str(b)][key]
            cells.append(fmt(v["point"], *v["hier_ci"]))
        lines.append(lab + " & " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    for m, lab in (("HTS-BCE", "HTS+BCE~\\cite{balanya2024adaptive} $-$ scalar"),
                   ("SMART+BCE", "SMART+BCE $-$ scalar"),
                   ("PWLinear-3", "PWLinear-3 $-$ scalar")):
        cells = []
        for t, b in cols:
            if t == "anchor":
                cells.append(fmt(*anchor_desc(m, b, rng)))
            else:
                c = ep[t]["contrasts"][str(b)][m]["ece15"]
                cells.append(fmt(c["point"], *c["ci95"]))
        lines.append(lab + " & " + " & ".join(cells) + r" \\")
    (GEN / "q1_ece_table.tex").write_text("\n".join(lines) + "\n")


def f2():
    a = json.load(open(W / "posthoc_f2/analysis_f2.json"))["settings"]
    rows = [("Anchor, $n{=}250$", "c100_vit/test/n=250"),
            ("Anchor, $n{=}2500$", "c100_vit/test/n=2500"),
            ("Amazon ID, $n{=}250$", "amazon_distilbert/id_test/n=250"),
            ("Amazon ID, $n{=}2500$", "amazon_distilbert/id_test/n=2500"),
            ("C100-C, $n{=}2500$", "c100c_vit/95-cell mean/n=2500")]
    ks = ["K1 C-OOF - D-OOF", "K3 C-IN - D-OOF", "K5 D-OOF - D-IN"]
    out = []
    for lab, k in rows:
        cells = []
        for kk in ks:
            c = a[k][kk]
            if c["draws_per_seed"] == 1:
                # only seeds vary: report sign agreement instead of a Bonferroni mark
                same = max(c["seeds_negative"], c["n_seeds"] - c["seeds_negative"])
                mark = f"$^{{{same}/{c['n_seeds']}}}$"
            else:
                mark = ("$^{\\ddagger}$" if c["excl0_bonf30"] else
                        ("$^{*}$" if c["excl0_95"] else ""))
            cells.append(fmt(c["point"], *c["ci95"]) + mark)
        out.append(lab + " & " + " & ".join(cells) + r" \\")
    (GEN / "f2_table.tex").write_text("\n".join(out) + "\n")


def decomp():
    ep = json.load(open(W / "posthoc_endpoints/endpoints.json"))["tasks"]
    names = [("TvA-TS", "scalar (TvA-TS)"), ("HTS-BCE", "HTS+BCE"),
             ("SRTS-BCE", "SRTS-BCE"), ("SMART+BCE", "SMART+BCE"),
             ("PWLinear-3", "PWLinear-3")]
    out = []
    for task, tl in (("amazon", "Amazon (new reviewers)"),
                     ("yahoo", "Yahoo (official test)")):
        out.append(r"\multicolumn{7}{l}{\emph{" + tl + r", $n{=}2500$}}\\")
        lv = ep[task]["levels"]["2500"]
        for m, lab in names:
            v = lv[m]
            out.append(" & ".join([
                lab + (CITE.get(m, "") if task == "amazon" else ""), f"{v['ece15']:.2f}", f"{100 * v['abs_count']:.1f}",
                f"{100 * v['cov_90']:.1f}",
                f"{100 * v['accerr_90']:.1f}/{100 * v['prederr_90']:.1f}",
                f"${signed1(100 * v['gap_90'])}$",
                f"${signed1(100 * v['grej_90'])}$"])
                + r" \\")
    (GEN / "decomp_table.tex").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    q1(); f2(); decomp()
    print("wrote q1_ece_table.tex f2_table.tex decomp_table.tex")


def app_endpoints():
    """Appendix: coverage and accepted-set count relative error at every
    threshold, both budgets, both tasks (levels; plan P3)."""
    ep = json.load(open(W / "posthoc_endpoints/endpoints.json"))["tasks"]
    names = [("TvA-TS", "scalar"), ("HTS-BCE", "HTS+BCE"),
             ("SRTS-BCE", "SRTS-BCE"), ("SMART+BCE", "SMART+BCE"),
             ("PWLinear-3", "PWLinear-3")]
    out = []
    for task, tl in (("amazon", "Amazon"), ("yahoo", "Yahoo")):
        for b in ("250", "2500"):
            lv = ep[task]["levels"][b]
            out.append(r"\multicolumn{5}{l}{\emph{" + tl + f", $n{{=}}{b}$"
                       + r"}}\\")
            for m, lab in names:
                v = lv[m]
                cells = [lab + (CITE.get(m, "") if (task, b) == ("amazon", "250") else ""),
                         f"{100 * v['abs_count']:.1f}"] + [
                    f"{100 * v[f'cov_{t}']:.1f} / ${signed0(100 * v[f'acc_cnt_{t}'])}$"
                    for t in (80, 90, 95)]
                out.append(" & ".join(cells) + r" \\")
    (GEN / "app_endpoints_table.tex").write_text("\n".join(out) + "\n")


def app_f2():
    a = json.load(open(W / "posthoc_f2/analysis_f2.json"))["settings"]
    rows = [("Anchor, $n{=}250$", "c100_vit/test/n=250"),
            ("Anchor, $n{=}2500$", "c100_vit/test/n=2500"),
            ("Amazon ID, $n{=}250$", "amazon_distilbert/id_test/n=250"),
            ("Amazon ID, $n{=}2500$", "amazon_distilbert/id_test/n=2500"),
            ("C100-C, $n{=}2500$", "c100c_vit/95-cell mean/n=2500")]
    ks = ["K2 C-IN - D-IN", "K4 C-OOF - C-IN",
          "K6 (C-OOF - D-OOF) - (C-IN - D-IN)"]
    out = []
    for lab, k in rows:
        cells = []
        for kk in ks:
            c = a[k][kk]
            mark = ("$^{\\ddagger}$" if c["excl0_bonf30"] else
                    ("$^{*}$" if c["excl0_95"] else ""))
            cells.append(fmt(c["point"], *c["ci95"]) + mark)
        out.append(lab + " & " + " & ".join(cells) + r" \\")
    (GEN / "app_f2_table.tex").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    app_endpoints(); app_f2()
    print("wrote app_endpoints_table.tex app_f2_table.tex")


def q3_replication():
    """Per split and budget: mean direction and Bonferroni-supported counts.
    count: means better than the scalar (of 4) and, in parentheses, the
    Bonferroni-supported ones; fixed t / matched: means with a wider
    accepted-set |gap| (of 12 = 4 maps x 3 thresholds) and, in parentheses,
    Bonferroni-supported wider / narrower."""
    ep = json.load(open(W / "posthoc_endpoints/endpoints.json"))["tasks"]
    dv = json.load(open(W / "posthoc_endpoints/endpoints_dev.json"))["tasks"]
    mt = json.load(open(W / "posthoc_matched/analysis_matched.json"))["contrasts"]
    M = ["HTS-BCE", "SRTS-BCE", "SMART+BCE", "PWLinear-3"]
    G = ["abs_gap_80", "abs_gap_90", "abs_gap_95"]
    src = [("Amazon, seen (ID test)", dv, "amazon_id_test", "amazon_distilbert/id_test"),
           ("Amazon, unseen (OOD val.)", dv, "amazon_ood_val", "amazon_distilbert/ood_val"),
           ("Amazon, unseen (confirm.)", ep, "amazon", "amazon_distilbert/ood_test_CONFIRMATION"),
           ("Yahoo (ID test)", dv, "yahoo_id_test", "yahoo_distilbert/id_test"),
           ("Yahoo (confirm.)", ep, "yahoo", "yahoo_distilbert/ood_test_CONFIRMATION")]
    out = []
    for i, (lab, d, k, mk) in enumerate(src):
        if i == 3:
            out.append(r"\midrule")
        for b in ("250", "2500"):
            cnt = [d[k]["contrasts"][b][m]["abs_count"] for m in M]
            cs = [d[k]["contrasts"][b][m][q] for m in M for q in G]
            ms = [mt[f"{mk}/n={b}/t={t}/{m}"] for m in M for t in (80, 90, 95)]
            cm = sum(c["point"] < 0 for c in cnt)
            cb = sum(c["excl0_bonf"] and c["point"] < 0 for c in cnt)

            def gap(cc):
                return (f"{sum(c['point'] > 0 for c in cc)} "
                        f"({sum(c['excl0_bonf'] and c['point'] > 0 for c in cc)}\\,/\\,"
                        f"{sum(c['excl0_bonf'] and c['point'] < 0 for c in cc)})")
            out.append(" & ".join([lab if b == "250" else "", b, f"{cm} ({cb})", gap(cs), gap(ms)])
                       + r" \\")
    (GEN / "q3_replication_table.tex").write_text("\n".join(out) + "\n")


def gamma_table():
    """Plan A5: direct seen-vs-unseen contrast, per rule x budget (12
    contrasts per unseen set; entries 'OOD validation / confirmation'):
    range of Gamma (points), number whose 95% / Bonferroni(96) hierarchical
    interval excludes 0 (Bonferroni: in both Monte Carlo streams; '+f' = f
    more in one stream only), number with all five seeds positive, number
    whose Bonferroni-corrected seed-level t interval excludes 0."""
    a = json.load(open(W / "posthoc_gamma/analysis_gamma.json"))
    out = []
    for rule in ("fixed", "matched"):
        for b in (250, 2500):
            cols = {k: [] for k in ("rng", "c95", "bf", "s5", "st")}
            for ood in ("ood_val", "ood_test_CONFIRMATION"):
                g = [e for e in a["contrasts"] if e["rule"] == rule and e["ood"] == ood
                     and e["budget"] == b]
                lo, hi = min(100 * e["gamma"] for e in g), max(100 * e["gamma"] for e in g)
                stable = sum(e["cibonf_excl0"] and not e["cibonf_mc_flip"] for e in g)
                flip = sum(e["cibonf_mc_flip"] for e in g)
                cols["rng"].append(f"{lo:.2f}--{hi:.2f}")
                cols["c95"].append(str(sum(e["ci95_excl0"] for e in g)))
                cols["bf"].append(f"{stable}" + (f"$^{{+{flip}}}$" if flip else ""))
                cols["s5"].append(str(sum(e["seeds_same_sign"] == len(e["per_seed"])
                                          and e["gamma"] > 0 for e in g)))
                cols["st"].append(str(sum(e["seed_tbonf_excl0"] for e in g)))
            out.append(" & ".join([("fixed $t$" if rule == "fixed" else "matched") if b == 250 else "",
                                   str(b)] + [" / ".join(v) for v in cols.values()]) + r" \\")
    (GEN / "gamma_table.tex").write_text("\n".join(out) + "\n")


def gamma_rep_table():
    """Plan A5, representative condition n = 2500, t = 0.90: Delta_seen,
    Delta_unseen (95% display intervals from the same index arrays) and
    Gamma (95% interval; double dagger = Bonferroni(96) interval excludes 0
    in both Monte Carlo streams), per map and selection rule; x100."""
    a = json.load(open(W / "posthoc_gamma/analysis_gamma.json"))
    tab = {(e["rule"], e["ood"], e["method"]): e for e in a["contrasts"]
           if e["budget"] == 2500 and e["t"] == 0.9}
    names = {"HTS-BCE": "HTS+BCE", "SRTS-BCE": "SRTS-BCE", "SMART+BCE": "SMART+BCE",
             "PWLinear-3": "PWLinear-3"}
    out = []
    for rule, lab in (("fixed", "fixed threshold"), ("matched", "matched coverage")):
        if rule == "matched":
            out.append(r"\midrule")
        out.append(r"\multicolumn{6}{l}{\emph{" + lab + r"}}\\")
        for m, nm in names.items():
            ev, ec = tab[(rule, "ood_val", m)], tab[(rule, "ood_test_CONFIRMATION", m)]

            def f(e, k):
                return fmt(100 * e[k], *(100 * x for x in e[k + "_ci95"] if True)) if k != "gamma" \
                    else fmt(100 * e["gamma"], *(100 * x for x in e["ci95"])) + (
                        "$^\\ddagger$" if e["cibonf_excl0"] and not e["cibonf_mc_flip"] else "")
            out.append(" & ".join([nm, f(ev, "delta_id"), f(ev, "delta_ood"), f(ev, "gamma"),
                                   f(ec, "delta_ood"), f(ec, "gamma")]) + r" \\")
    (GEN / "gamma_rep_table.tex").write_text("\n".join(out) + "\n")


def reviewer_cluster_table():
    """Plan A6 (post hoc): representative condition n = 2500, t = 0.90, fixed
    threshold. Per unseen set and map: Delta_seen and Delta_unseen with
    fixed-model reviewer-bootstrap 95% intervals, Gamma with the A5
    seed->draw 95% interval, the fixed-model reviewer-bootstrap 95% and
    Bonferroni(8) intervals, and the joint seed/draw x reviewer 95%
    interval; x100."""
    p = W / "posthoc_reviewer_cluster/analysis.json"
    if not p.exists():
        return
    a = json.load(open(p))
    names = {"HTS-BCE": "HTS+BCE", "SRTS-BCE": "SRTS-BCE", "SMART+BCE": "SMART+BCE",
             "PWLinear-3": "PWLinear-3"}
    lab = {"ood_val": "OOD val.", "ood_test_CONFIRMATION": "confirm."}

    def iv(x):
        return (f"{{\\scriptsize[${100 * x[0]:+.2f}$,${100 * x[1]:+.2f}$]}}"
                .replace("+", "{+}").replace("-", "{-}"))
    out = []
    for i, e in enumerate(a["contrasts"]):
        if i == 4:
            out.append(r"\midrule")
        g = f"${100 * e['gamma']:+.2f}$".replace("+", "{+}").replace("-", "{-}")
        dsn = f"${100 * e['delta_id']:+.2f}$".replace("+", "{+}").replace("-", "{-}")
        dun = f"${100 * e['delta_ood']:+.2f}$".replace("+", "{+}").replace("-", "{-}")
        out.append(" & ".join([lab[e["ood"]] if i % 4 == 0 else "", names[e["method"]],
                               f"{dsn} {iv(e['delta_id_ci95'])}", f"{dun} {iv(e['delta_ood_ci95'])}", g,
                               iv(e["a5_seed_draw_ci95"]), iv(e["reviewer_ci95"]),
                               iv(e["reviewer_cibonf8"]), iv(e["joint_ci95"])]) + r" \\")
    (GEN / "reviewer_cluster_table.tex").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    q3_replication(); gamma_table(); gamma_rep_table(); reviewer_cluster_table()
    print("wrote q3_replication_table.tex gamma_table.tex gamma_rep_table.tex")

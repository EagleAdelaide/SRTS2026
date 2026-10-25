#!/usr/bin/env python3
"""Programmatic check of the prose numbers changed in the 2026-10-16
revision: recompute each from the result files and assert that main.tex
states it. Exit 1 on any mismatch."""
import json
import re
import sys
from pathlib import Path

import numpy as np

R = Path("__SRTS_ROOT__/calattn-repo")
W = R / "results/www2027"
TEX = re.sub(r"\s+", " ", (R / "www2027/paper/main.tex").read_text())
ep = json.load(open(W / "posthoc_endpoints/endpoints.json"))["tasks"]
f2 = json.load(open(W / "posthoc_f2/analysis_f2.json"))["settings"]
fc = json.load(open(W / "posthoc_f2/analysis_family_capacity.json"))
M = ["HTS-BCE", "SRTS-BCE", "SMART+BCE", "PWLinear-3"]
G = ["abs_gap_80", "abs_gap_90", "abs_gap_95"]
checks = []


def chk(desc, ok, text=None):
    present = (text is None) or (text in TEX)
    checks.append((desc, ok and present, text))


def cnt(task, budgets, qs, key, sign):
    return sum(1 for b in budgets for m in M for q in qs
               if ep[task]["contrasts"][b][m][q][key]
               and (ep[task]["contrasts"][b][m][q]["point"] < 0) == (sign < 0))


# Q3 counts
chk("Amazon count improved 8/8 at 95%", cnt("amazon", ["250", "2500"], ["abs_count"], "excl0_95", -1) == 8, "8 of 8 at 95\\%")
chk("Amazon count improved 6/8 Bonf", cnt("amazon", ["250", "2500"], ["abs_count"], "excl0_bonf", -1) == 6, "6 of 8 after Bonferroni")
chk("Amazon gaps widened 24/24 Bonf", cnt("amazon", ["250", "2500"], G, "excl0_bonf", +1) == 24, "24 of 24")
chk("Yahoo 2500 counts worse 4/4 at 95%, 0 Bonf",
    cnt("yahoo", ["2500"], ["abs_count"], "excl0_95", +1) == 4 and cnt("yahoo", ["2500"], ["abs_count"], "excl0_bonf", +1) == 0, "all four at 95\\%, none after Bonferroni")
chk("Yahoo 2500: 12/12 gap means narrower, 5 Bonf (all at .90/.95)",
    sum(ep["yahoo"]["contrasts"]["2500"][m][q]["point"] < 0 for m in M for q in G) == 12
    and cnt("yahoo", ["2500"], G[1:], "excl0_bonf", -1) == 5 and cnt("yahoo", ["2500"], G, "excl0_bonf", -1) == 5,
    "five Bonferroni-supported, at $t{=}0.90$ and $0.95$")
chk("Yahoo 250 no count resolved, no gap Bonf",
    cnt("yahoo", ["250"], ["abs_count"], "excl0_95", -1) + cnt("yahoo", ["250"], ["abs_count"], "excl0_95", +1) == 0
    and cnt("yahoo", ["250"], G, "excl0_bonf", -1) + cnt("yahoo", ["250"], G, "excl0_bonf", +1) == 0)
fav = sum(cnt(t, ["250", "2500"], ["abs_count"] + G, "excl0_bonf", -1) for t in ep)
unf = sum(cnt(t, ["250", "2500"], ["abs_count"] + G, "excl0_bonf", +1) for t in ep)
chk(f"64-contrast tally {fav}/{unf}", (fav, unf) == (11, 24), "11 Bonferroni-resolved differences favour a conditional map and 24 favour the scalar")
lv = ep["amazon"]["levels"]["2500"]
chk("scalar g(A) +0.9, g(Ac) +3.7", f"{100*lv['TvA-TS']['gap_90']:+.1f}" == "+0.9" and f"{100*lv['TvA-TS']['grej_90']:+.1f}" == "+3.7", "mean bias of $+3.7$ points and its accepted set $+0.9$")
ga = [100 * lv[m]["gap_90"] for m in M]; gr = [100 * lv[m]["grej_90"] for m in M]
chk("conditional g(A) 2.3-2.5", f"{min(ga):.1f}-{max(ga):.1f}" == "2.3-2.5", "$2.3$ to $2.5$ points")
chk("conditional g(Ac) 2.2-2.4", f"{min(gr):.1f}-{max(gr):.1f}" == "2.2-2.4", "$+2.2$ to $+2.4$ points")
cv = [100 * ep["amazon"]["contrasts"]["2500"][m]["cov_90"]["point"] for m in M]
chk("coverage +0.8..+3.8", f"{min(cv):.1f}-{max(cv):.1f}" == "0.8-3.8", "0.8 to 3.8 points more items")
ac = [100 * lv[m]["acc_cnt_90"] for m in M]
chk("accepted count error 14% vs 33-35%", f"{100*lv['TvA-TS']['acc_cnt_90']:.0f}" == "-14" and f"{-max(ac):.0f}-{-min(ac):.0f}" == "33-35", "33--35\\%")
rk = max(abs(ep[t]["contrasts"][b][m][q]["point"]) for t in ep for b in ("250", "2500") for m in M for q in ("risk_cov80", "risk_cov90", "risk_cov95"))
au = max(abs(ep[t]["contrasts"][b][m]["aurc"]["point"]) for t in ep for b in ("250", "2500") for m in M)
chk(f"risk<=0.3 ({100*rk:.3f}), AURC<=0.7 ({100*au:.3f})", 100 * rk <= 0.3 and 100 * au <= 0.7, "at most $0.3$ points and AURC by at most $0.7$ points")
# F2
k1 = lambda s: f2[s]["K1 C-OOF - D-OOF"]
chk("anchor 2500 K1 +0.06 5/5 seeds", f"{k1('c100_vit/test/n=2500')['point']:+.2f}" == "+0.06" and k1('c100_vit/test/n=2500')['seeds_negative'] == 0, "($+0.06$ at $n{=}2500$, five of five seeds)")
chk("Amazon 2500 K1 -0.13 Bonf", f"{k1('amazon_distilbert/id_test/n=2500')['point']:+.2f}" == "-0.13" and k1('amazon_distilbert/id_test/n=2500')['excl0_bonf30'], "($-0.13$, Bonferroni-resolved)")
sl = json.load(open(W / "posthoc_f2/seed_level_sensitivity.json"))
chk("C100-C K1 -0.17 5/5, seed-level t incl 0", f"{k1('c100c_vit/95-cell mean/n=2500')['point']:+.2f}" == "-0.17" and k1('c100c_vit/95-cell mean/n=2500')['seeds_negative'] == 5 and not sl["f2/c100c_vit/95-cell mean/n=2500/K1 C-OOF - D-OOF"]["t95_excl0"], "five of five seeds, but the seed-level interval includes zero")
dk = []
for s, v in f2.items():
    a, b = v["K1 C-OOF - D-OOF"], v["K3 C-IN - D-OOF"]
    if a["excl0_95"] or b["excl0_95"]:
        dk.append((np.sign(a["point"]) == np.sign(b["point"]), abs(a["point"] - b["point"])))
chk("K1 vs K3 same sign where resolved, diff<=0.04", all(x for x, _ in dk) and max(d for _, d in dk) <= 0.0405, "differ by at most $0.04$ points")
k5 = [abs(v["K5 D-OOF - D-IN"]["point"]) for v in f2.values()]
chk("K5 unresolved everywhere, |d|<=0.10", all(not v["K5 D-OOF - D-IN"]["excl0_95"] for v in f2.values()) and max(k5) <= 0.10, "($|\\Delta|\\le0.10$)")
k4 = [(s, v["K4 C-OOF - C-IN"]) for s, v in f2.items() if v["K4 C-OOF - C-IN"]["excl0_95"]]
chk("K4 resolved only on C100-C (+0.03)", len(k4) == 1 and k4[0][0].startswith("c100c") and f"{k4[0][1]['point']:.2f}" == "0.03", "OOF fitting is $0.03$ points worse")
# within-family
hi2500 = [v["n=2500"]["point"] for k, v in fc.items() if isinstance(v, dict) and "n=2500" in v]
it = [v["interaction 250-2500"]["point"] for k, v in fc.items() if isinstance(v, dict) and "n=2500" in v]
allb = all(v[x]["excl0_bonf12"] for k, v in fc.items() if isinstance(v, dict) and "n=2500" in v for x in ("n=2500", "interaction 250-2500"))
chk("family 2500 -0.44..-0.63, interaction +0.75..+1.76, all Bonf",
    f"{max(hi2500):.2f}" == "-0.44" and f"{min(hi2500):.2f}" == "-0.63" and f"{min(it):.2f}" == "0.75" and f"{max(it):.2f}" == "1.76" and allb,
    "($-0.44$ to $-0.63$ points)")
# --- review-round additions (plan A4)
dv = json.load(open(W / "posthoc_endpoints/endpoints_dev.json"))["tasks"]
mt = json.load(open(W / "posthoc_matched/analysis_matched.json"))["contrasts"]
def dcnt(task, b, qs, sign):
    return sum(1 for m in M for q in qs if dv[task]["contrasts"][b][m][q]["excl0_bonf"]
               and (dv[task]["contrasts"][b][m][q]["point"] < 0) == (sign < 0))
def mcnt(sp, b, sign):
    return sum(1 for m in M for t_ in (80, 90, 95) if mt[f"{sp}/n={b}/t={t_}/{m}"]["excl0_bonf"]
               and (mt[f"{sp}/n={b}/t={t_}/{m}"]["point"] < 0) == (sign < 0))
def dmean(task, b, qs, sign):
    return sum(1 for m in M for q in qs if (dv[task]["contrasts"][b][m][q]["point"] < 0) == (sign < 0))
def mmean(sp, b, sign, ts=(80, 90, 95)):
    return sum(1 for m in M for t_ in ts if (mt[f"{sp}/n={b}/t={t_}/{m}"]["point"] < 0) == (sign < 0))
chk("OOD-val: 24/24 gap means wider (23 Bonf), 8/8 count means better (6 Bonf)",
    dmean("amazon_ood_val", "250", G, +1) + dmean("amazon_ood_val", "2500", G, +1) == 24
    and dcnt("amazon_ood_val", "250", G, +1) + dcnt("amazon_ood_val", "2500", G, +1) == 23
    and dmean("amazon_ood_val", "250", ["abs_count"], -1) + dmean("amazon_ood_val", "2500", ["abs_count"], -1) == 8
    and dcnt("amazon_ood_val", "250", ["abs_count"], -1) + dcnt("amazon_ood_val", "2500", ["abs_count"], -1) == 6,
    "all 24 gap means are again wider (23 Bonferroni-supported) and all 8 count means better (6 after Bonferroni)")
idt = dv["amazon_id_test"]["contrasts"]
chk("Amazon ID: n=250 12/12 wider (2 Bonf); n=2500 t=.80 4/4 wider (PWL Bonf), t>=.90 8/8 narrower, 3 Bonf fixed / 4 matched, all at .95",
    dmean("amazon_id_test", "250", G, +1) == 12 and dcnt("amazon_id_test", "250", G, +1) == 2
    and all(idt["2500"][m]["abs_gap_80"]["point"] > 0 for m in M)
    and [m for m in M if idt["2500"][m]["abs_gap_80"]["excl0_bonf"]] == ["PWLinear-3"]
    and all(idt["2500"][m][q]["point"] < 0 for m in M for q in G[1:])
    and dcnt("amazon_id_test", "2500", G, -1) == 3 and dcnt("amazon_id_test", "2500", G[2:], -1) == 3
    and mcnt("amazon_distilbert/id_test", "2500", -1) == 4
    and sum(mt[f"amazon_distilbert/id_test/n=2500/t=95/{m}"]["excl0_bonf"] for m in M) == 4,
    "three of them Bonferroni-supported at the fixed threshold and four at matched coverage, all at $t{=}0.95$")
chk("matched coverage unseen: 24/24 means wider each, 19 and 23 Bonf",
    mmean("amazon_distilbert/ood_val", "250", +1) + mmean("amazon_distilbert/ood_val", "2500", +1) == 24
    and mmean("amazon_distilbert/ood_test_CONFIRMATION", "250", +1) + mmean("amazon_distilbert/ood_test_CONFIRMATION", "2500", +1) == 24
    and mcnt("amazon_distilbert/ood_val", "250", +1) + mcnt("amazon_distilbert/ood_val", "2500", +1) == 19
    and mcnt("amazon_distilbert/ood_test_CONFIRMATION", "250", +1) + mcnt("amazon_distilbert/ood_test_CONFIRMATION", "2500", +1) == 23,
    "23 (confirmation) and 19 (OOD validation) Bonferroni-supported")
chk("Yahoo matched 2500 narrower 5/12", mcnt("yahoo_distilbert/ood_test_CONFIRMATION", "2500", -1) == 5, "likewise at matched coverage")
yv = ep["yahoo"]["levels"]["2500"]
chk("Yahoo scalar count error smallest (2.4%)", f"{100*yv['TvA-TS']['abs_count']:.1f}" == "2.4" and all(yv['TvA-TS']['abs_count'] < yv[m]['abs_count'] for m in M), "count error the smallest of the five (2.4\\%)")
f = lambda k: sl[k]
chk("Yahoo frozen seed-level intervals", [round(x, 2) for x in f("frozen yahoo/n=250/SRTS-BCE - TvA-TS")["t95"]] == [-0.04, 2.43]
    and [round(x, 2) for x in f("frozen yahoo/n=2500/SRTS-BCE - TvA-TS")["t95"]] == [-0.01, 0.33]
    and not f("frozen yahoo/n=250/SRTS-BCE - SMART+BCE")["t95_excl0"], "$[-0.01,+0.33]$ at $n{=}2500$), as does")
amz_g = sum(v["t95_excl0"] for k, v in sl.items() if k.startswith("endpoints conf/amazon") and "abs_gap" in k)
yah_g = sum(v["t95_excl0"] for k, v in sl.items() if k.startswith("endpoints conf/yahoo") and "abs_gap" in k)
f2t = sum(v["t95_excl0"] for k, v in sl.items() if k.startswith("f2/")); f2b = sum(v["boot95_excl0"] for k, v in sl.items() if k.startswith("f2/"))
fam = sum(v["t95_excl0"] for k, v in sl.items() if k.startswith("family/"))
famI = sum(v["t95_excl0"] for k, v in sl.items() if k.startswith("family/") and "interaction" in k)
chk("seed-level summary 24/24, 10/24, 8/30 (15), 9/12 incl 4 interactions", (amz_g, yah_g, f2t, f2b, fam, famI) == (24, 10, 8, 15, 9, 4), "for 8 of the 30 score-construction contrasts")
fz = [k for k, v in sl.items() if k.startswith("frozen") and v["t95_excl0"]]
chk("frozen seed-level 3/4 Amazon, 1/4 Yahoo, 2/4 anchor", (sum(k.startswith("frozen amazon") for k in fz), sum(k.startswith("frozen yahoo") for k in fz), sum(k.startswith("frozen anchor") for k in fz)) == (3, 1, 2), "1 of 4 on Yahoo and 2 of 4 on the anchor")
na = 0.07 * 0.93 / 0.005 ** 2
chk("screening: at 25% coverage one label per four screened items", True, "at 25\\% coverage each label requires screening four")
chk("transfer: same status 10/10, same sign 9/10", True, "and the same sign for nine")
# --- 2026-10-16 r2 (plan A5 and scope corrections)
la = ep["amazon"]["levels"]["2500"]
chk("example t=.90 n=2500: ECE 3.12->2.51, count 11.1->8.6, gap +0.9->+2.5",
    [f"{la[m]['ece15']:.2f}" for m in ("TvA-TS", "SRTS-BCE")] == ["3.12", "2.51"]
    and [f"{100*la[m]['abs_count']:.1f}" for m in ("TvA-TS", "SRTS-BCE")] == ["11.1", "8.6"]
    and [f"{100*la[m]['gap_90']:+.1f}" for m in ("TvA-TS", "SRTS-BCE")] == ["+0.9", "+2.5"],
    "lowers ECE from 3.12 to 2.51 and the whole-batch count error from 11.1\\% to 8.6\\%")
e250 = {m: ep["amazon"]["contrasts"]["250"][m]["ece15"]["point"] for m in M}
chk("Amazon conf n=250 ECE: HTS -0.19 unresolved, others +0.31..+1.06; n=2500 all four lower",
    f"{e250['HTS-BCE']:+.2f}" == "-0.19" and not ep["amazon"]["contrasts"]["250"]["HTS-BCE"]["ece15"]["excl0_95"]
    and f"{min(e250[m] for m in M[1:]):+.2f}" == "+0.31" and f"{max(e250.values()):+.2f}" == "+1.06"
    and all(ep["amazon"]["contrasts"]["2500"][m]["ece15"]["point"] < 0 for m in M),
    "($-0.19$, not resolved), and the other three are worse ($+0.31$ to $+1.06$)")
cg = [100 * ep["amazon"]["contrasts"][b][m][q]["point"] for b in ("250", "2500") for m in M for q in G]
cc = [100 * ep["amazon"]["contrasts"][b][m]["abs_count"]["point"] for b in ("250", "2500") for m in M]
chk("confirmation effect ranges: count -1.2..-3.0, gap +0.6..+2.8",
    f"{-max(cc):.1f}" == "1.2" and f"{-min(cc):.1f}" == "3.0" and f"{min(cg):.1f}" == "0.6" and f"{max(cg):.1f}" == "2.8",
    "(by 1.2 to 3.0 points; 8 of 8 at 95\\%")
q1t = (R / "www2027/paper/generated/q1_ece_table.tex").read_text()
sm = [l for l in q1t.splitlines() if l.startswith("SMART+BCE")][0]
pts = [float(x.replace("{", "").replace("}", "")) for x in re.findall(r"^\$([^$]+)\$|& \$([^$]+)\$", sm) for x in x if x]
chk("SMART+BCE vs scalar: n=250 +0.37/+1.06/+1.79, n=2500 -0.71/-0.54",
    [f"{x:+.2f}" for x in pts] == ["+0.37", "-0.71", "+1.06", "-0.54", "+1.79", "+0.25"],
    "SMART+BCE is worse than the scalar there but better at $n{=}2500$ on the anchor and Amazon")
gm = json.load(open(W / "posthoc_gamma/analysis_gamma.json"))
C = gm["contrasts"]
st = [e for e in C if e["cibonf_excl0"] and not e["cibonf_mc_flip"]]
fl = [e for e in C if e["cibonf_mc_flip"]]
ne = [e for e in C if not e["cibonf_excl0"] and not e["cibonf_mc_flip"]]
chk("Gamma: 96/96 >0 and 95% excl; Bonf 90 stable, 2 flip, 4 neither, all SRTS n=250; 94 all-5 seeds; 1 seed-Bonf; range 0.41-2.04",
    len(C) == 96 and all(e["gamma"] > 0 and e["ci95_excl0"] for e in C) and (len(st), len(fl), len(ne)) == (90, 2, 4)
    and all(e["method"] == "SRTS-BCE" and e["budget"] == 250 for e in fl + ne)
    and sum(e["seeds_same_sign"] == 5 for e in C) == 94 and sum(e["seed_tbonf_excl0"] for e in C) == 1
    and sum(e["seed_t95_excl0"] for e in C) == 96
    and f"{100*min(e['gamma'] for e in C):.2f}" == "0.41" and f"{100*max(e['gamma'] for e in C):.2f}" == "2.04"
    and gm["pairing"] and gm["monte_carlo"]["cibonf"]["decision_flips"] == 2,
    "the Bonferroni(96) intervals exclude zero for 90 in two independent bootstrap streams")
rp = [e for e in C if e["budget"] == 2500 and e["t"] == 0.9]
ds = [100 * e["delta_id"] for e in rp]; do = [100 * e["delta_ood"] for e in rp]; gg = [100 * e["gamma"] for e in rp]
exl = sorted({(e["rule"], e["method"]) for e in rp if e["delta_id_ci95"][1] < 0})
chk("representative n=2500 t=.90: seen -0.19..-0.33, unseen 1.1..1.5, Gamma 1.45..1.77; seen excl: HTS fixed, HTS+SRTS matched",
    f"{max(ds):.2f}" == "-0.19" and f"{min(ds):.2f}" == "-0.33" and f"{min(do):.1f}" == "1.1" and f"{max(do):.1f}" == "1.5"
    and f"{min(gg):.2f}" == "1.45" and f"{max(gg):.2f}" == "1.77"
    and exl == [("fixed", "HTS-BCE"), ("matched", "HTS-BCE"), ("matched", "SRTS-BCE")]
    and all(e["delta_id_ci95"][0] < 0 for e in rp),
    "so $\\Gamma$ is 1.45 to 1.77 points")
chk("abstract: unseen 24/24 means wider both sets, fixed and matched; seen/Yahoo fixed n=2500 narrower (ID t>=.90 only), n=250 wider",
    all(sum(d_["contrasts"][b][m][q]["point"] > 0 for b in ("250", "2500") for m in M for q in G) == 24
        for d_ in (ep["amazon"], dv["amazon_ood_val"]))
    and all(mmean(sp, "250", +1) + mmean(sp, "2500", +1) == 24 for sp in ("amazon_distilbert/ood_val", "amazon_distilbert/ood_test_CONFIRMATION"))
    and all(d_["contrasts"]["2500"][m][q]["point"] < 0 for d_ in (ep["yahoo"], dv["yahoo_id_test"]) for m in M for q in G)
    and all(d_["contrasts"]["250"][m][q]["point"] > 0 for d_ in (ep["yahoo"], dv["yahoo_id_test"], dv["amazon_id_test"]) for m in M for q in G),
    "results on seen reviewers and Yahoo differ")
narrow_seen = {(b, q) for b in ("250", "2500") for q in G
               if all(dv["amazon_id_test"]["contrasts"][b][m][q]["point"] < 0 for m in M)}
any_narrow_seen = {(b, q) for b in ("250", "2500") for q in G
                   if any(dv["amazon_id_test"]["contrasts"][b][m][q]["point"] < 0 for m in M)}
chk("seen fixed-threshold: narrower means exactly at n=2500, t=.90/.95 (all maps), nowhere else",
    narrow_seen == any_narrow_seen == {("2500", "abs_gap_90"), ("2500", "abs_gap_95")})
sd = json.load(open(W / "posthoc_gamma/signed_descriptive.json"))
rpS = [c for c in sd["contrasts"] if c["budget"] == 2500 and c["t"] == 0.9]
shifts = [100 * c[k]["mean"] for c in rpS for k in ("dsigned_seen", "dsigned_unseen")]
gsg = [100 * c["gamma_signed"]["mean"] for c in sd["contrasts"]]
sc = {(c["split"], c["t"]): c for c in sd["scalar"] if c["budget"] == 2500 and c["rule"] == "fixed"}
chk("signed: shift +1.2..+1.6 at n=2500 t=.90; signed Gamma -0.06..+0.53; scalar g -0.9 (94% neg) vs +0.8..+0.9 (91-92% pos)",
    f"{min(shifts):.1f}" == "1.2" and f"{max(shifts):.1f}" == "1.6"
    and f"{min(gsg):.2f}" == "-0.06" and f"{max(gsg):.2f}" == "0.53"
    and f"{100*sc[('id_test', 0.9)]['mean']:.1f}" == "-0.9" and round(100 * (1 - sc[('id_test', 0.9)]['share_pos'])) == 94
    and [f"{100*sc[(sp, 0.9)]['mean']:.1f}" for sp in ("ood_val", "ood_test_CONFIRMATION")] == ["0.8", "0.9"]
    and [round(100 * sc[(sp, 0.9)]['share_pos']) for sp in ("ood_val", "ood_test_CONFIRMATION")] == [91, 92],
    "seen-versus-unseen difference is $-0.06$ to $+0.53$ points")
ae = [dv["amazon_id_test"]["levels"][b][m]["accerr_90"] for b in ("250", "2500") for m in ["TvA-TS"] + M]
cv250 = dv["amazon_id_test"]["levels"]["250"]["TvA-TS"]["cov_90"]
se = [100 * np.sqrt(r * (1 - r) / round(250 * cv250)) for r in ae]
chk(f"n=250 calibration SE at seen accepted error {100*min(ae):.1f}-{100*max(ae):.1f}%: {min(se):.2f}-{max(se):.2f}",
    round(250 * cv250) in (60, 61, 62, 63) and f"{min(se):.1f}" == "2.5" and f"{max(se):.1f}" == "2.8",
    "2.5--2.8 points at the 4--5\\% accepted-set error rate of seen reviewers")
rc = json.load(open(W / "posthoc_reviewer_cluster/analysis.json"))
rcg = json.load(open(W / "posthoc_reviewer_cluster/gate.json"))
C8 = rc["contrasts"]
wd = lambda x: x[1] - x[0]
rr = [wd(e["reviewer_ci95"]) / wd(e["a5_seed_draw_ci95"]) for e in C8]
jr = [wd(e["joint_ci95"]) / wd(e["a5_seed_draw_ci95"]) for e in C8]
lo = lambda k: [100 * e[k][0] for e in C8]
chk("reviewer-cluster: width x1.2-1.5 (joint 1.6-1.8); lower bounds 95% +0.73..+0.90, Bonf8 +0.46..+0.61, joint Bonf8 +0.35..+0.48; all 8 excl0",
    f"{min(rr):.1f}" == "1.2" and f"{max(rr):.1f}" == "1.5" and f"{min(jr):.1f}" == "1.6" and f"{max(jr):.1f}" == "1.8"
    and [f"{f(lo(k)):.2f}" for k in ("reviewer_ci95", "reviewer_cibonf8", "joint_cibonf8") for f in (min, max)]
    == ["0.73", "0.90", "0.46", "0.61", "0.35", "0.48"]
    and all(e["reviewer_ci95_excl0"] and e["reviewer_cibonf8_excl0"] and e["joint_ci95_excl0"] and e["joint_cibonf8_excl0"] for e in C8)
    and rc["max_abs_point_diff_vs_a5_points"] < 1e-9,
    "1.2 to 1.5 times as wide")
chk("reviewer-cluster: Delta_seen unresolved for all 4 (HTS -0.32 [-0.95,+0.52]); Delta_unseen > 0; MC <= 0.02, no flips; no empty; cost 1.5 CPU-h",
    all(e["delta_id_ci95"][0] < 0 < e["delta_id_ci95"][1] for e in C8) and all(e["delta_ood_ci95"][0] > 0 for e in C8)
    and [round(100 * x, 2) for x in C8[0]["delta_id_ci95"]] == [-0.95, 0.52] and f"{100*C8[0]['delta_id']:.2f}" == "-0.32"
    and max(v["max_abs_bound_diff_points"] for v in rc["monte_carlo"].values()) <= 0.02
    and sum(v["decision_flips"] for v in rc["monte_carlo"].values()) == 0
    and rc["empty_accepted_cells"] == {"stream1": 0, "stream2": 0} and rcg["empty_accepted"] == 0
    and round(rcg["cpu_hours_after_seed"]["4"], 1) == 1.5 and rcg["reviewers"] == {"id_test": 626, "ood_val": 1334, "ood_test_CONFIRMATION": 1334},
    "is not statistically resolved for any map")
# --- r3.2 abstract (author-supplied candidate, verified here)
import glob as _g
from collections import defaultdict as _dd
def _ece_tab(rows):
    d = _dd(lambda: _dd(list))
    for r in rows:
        d[(r["method"], r["budget"])][r["seed"]].append(r["ece15"])
    return {k: np.mean([np.mean(v) for v in s_.values()]) for k, s_ in d.items()}
_yc = _ece_tab([json.loads(l) for l in open(W / "CONFIRMATION/yahoo_distilbert.jsonl")])
_yi = _ece_tab([r for f in _g.glob(str(W / "suite/yahoo_s*/yahoo_distilbert_core_natural.jsonl"))
                for r in map(json.loads, open(f)) if r["test_split"] == "id_test"])
_cond = lambda t: [(m, b) for (m, b) in t if m not in ("TvA-TS", "NoTS", "TS-NLL") and t[(m, b)] < t[("TvA-TS", b)]]
chk("abstract r3.2: Yahoo, no conditional map has lower mean ECE than TvA-TS at any budget (confirmation and ID test)",
    _cond(_yc) == [] and _cond(_yi) == [] and len({m for m, _ in _yc}) >= 10,
    "ECE rankings depend on task and label budget, and no conditional map dominates")
_ok = True
for _d, _mk in ((ep["amazon"], "amazon_distilbert/ood_test_CONFIRMATION"), (dv["amazon_ood_val"], "amazon_distilbert/ood_val")):
    _c = _d["contrasts"]["2500"]
    _ok &= all(_c[m]["ece15"]["point"] < 0 and _c[m]["abs_count"]["point"] < 0 for m in M)
    _ok &= all(_c[m][q]["point"] > 0 for m in M for q in G)
    _ok &= all(mt[f"{_mk}/n=2500/t={t_}/{m}"]["point"] > 0 for m in M for t_ in (80, 90, 95))
chk("abstract r3.2: n=2500, both unseen sets: 4/4 lower mean ECE and count; |g| wider 12/12 fixed and 12/12 matched",
    _ok, "mismatch occurs in the same direction for four conditional BCE maps on two disjoint unseen-reviewer splits that share models and calibration draws, including at matched coverage")
_ac = _ece_tab([json.loads(l) for l in open(W / "CONFIRMATION/amazon_distilbert.jsonl")])
_hi = ["SRTS-BCE", "PWLinear-3", "Margin-K3", "SMART+BCE", "SMART-SoftECE"]
_c250, _c2500 = ep["amazon"]["contrasts"]["250"], ep["amazon"]["contrasts"]["2500"]
chk("abstract r3.2: Amazon n=250 scalar beats higher-capacity conditional maps (SRTS/PWL/SMART resolved); n=2500 trails all four (resolved)",
    all(_ac[(m, 250)] > _ac[("TvA-TS", 250)] for m in _hi)
    and all(_c250[m]["ece15"]["point"] > 0 and _c250[m]["ece15"]["excl0_95"] for m in ("SRTS-BCE", "PWLinear-3", "SMART+BCE"))
    and all(_c2500[m]["ece15"]["point"] < 0 and _c2500[m]["ece15"]["excl0_95"] for m in M),
    "the Amazon scalar--conditional ECE ordering crosses as $n$ increases")
chk("abstract r3.2: example ECE 3.12%->2.51%, count 11.1%->8.6%, overconfidence 0.9->2.5 pp",
    [f"{la[m]['ece15']:.2f}" for m in ("TvA-TS", "SRTS-BCE")] == ["3.12", "2.51"]
    and [f"{100*la[m]['abs_count']:.1f}" for m in ("TvA-TS", "SRTS-BCE")] == ["11.1", "8.6"]
    and [f"{100*la[m]['gap_90']:.1f}" for m in ("TvA-TS", "SRTS-BCE")] == ["0.9", "2.5"]
    and all(la[m]["gap_90"] > 0 for m in ("TvA-TS", "SRTS-BCE")),
    "yet raises mean accepted-set overconfidence from $+0.9$ to $+2.5$ percentage points")
f1 = json.load(open(W / "posthoc_figdata/fig1_reversal.json"))["levels"]
cvs = [f1[m]["cov90"]["mean"] for m in M]
# figure-editorial revision: Fig. 1 now shows TvA-TS and SRTS-BCE only (all four maps in Fig. 3), so the
# caption cites their two coverages; the conditional range 23.9-26.9 is still checked as a value.
chk("Fig.1 caption: coverage at t=.90 TvA-TS 23.1%, SRTS-BCE 26.9% (all conditional maps 23.9-26.9); levels match Table 4",
    f"{f1['TvA-TS']['cov90']['mean']:.1f}" == "23.1" and f"{f1['SRTS-BCE']['cov90']['mean']:.1f}" == "26.9"
    and f"{min(cvs):.1f}" == "23.9" and f"{max(cvs):.1f}" == "26.9"
    and all(abs(f1[m]["ece15"]["mean"] - la[m]["ece15"]) < 1e-9 and abs(f1[m]["gap90"]["mean"] - 100 * la[m]["gap_90"]) < 1e-9
            for m in ["TvA-TS"] + M),
    "(coverage 23.1\\% and 26.9\\%)")
_f1r = {k: (f1["TvA-TS"][k]["mean"], f1["SRTS-BCE"][k]["mean"]) for k in ("ece15", "abs_count", "gap90")}
chk("Fig.1 direct labels: ECE 3.12 -> 2.51, |count| 11.1 -> 8.6, g(A_.90) +0.9 -> +2.5 (from fig1_reversal.json)",
    [f"{_f1r['ece15'][0]:.2f}", f"{_f1r['ece15'][1]:.2f}", f"{_f1r['abs_count'][0]:.1f}", f"{_f1r['abs_count'][1]:.1f}",
     f"{_f1r['gap90'][0]:+.1f}", f"{_f1r['gap90'][1]:+.1f}"] == ["3.12", "2.51", "11.1", "8.6", "+0.9", "+2.5"],
    "lowers ECE from 3.12 to 2.51 and the whole-batch count error from 11.1\\% to 8.6\\%")
f2 = json.load(open(W / "posthoc_figdata/fig2_budget.json"))
chk("Fig.2: SRTS-BCE - scalar reproduces the frozen analyses (gate <= 1e-9)", f2["gate_max_abs_diff_srts_vs_frozen"] <= 1e-9,
    "SRTS-BCE $-$ scalar is the pre-declared analysis")
# --- substantive sprint (plan A8 ii): mechanism accounting
mech = json.load(open(W / "posthoc_mechanism/analysis_mechanism.json"))
MS = mech["summary"]; MAPS4 = ["HTS-BCE", "SRTS-BCE", "SMART+BCE", "PWLinear-3"]
def _rng(vals): return (f"{min(vals):+.2f}", f"{max(vals):+.2f}")
S_rep = [100 * MS[f"n=2500/{sp}/t=0.9/fixed/{m}"]["S"]["mean"] for sp in ("id_test", "ood_val", "ood_test_CONFIRMATION") for m in MAPS4]
P_seen = [100 * MS[f"n=2500/id_test/t=0.9/fixed/{m}"]["P"]["mean"] for m in MAPS4]
P_uns = [100 * MS[f"n=2500/{sp}/t=0.9/fixed/{m}"]["P"]["mean"] for sp in ("ood_val", "ood_test_CONFIRMATION") for m in MAPS4]
shP = [MS[f"n=2500/{sp}/t=0.9/fixed/{m}"]["share_P"] for sp in ("id_test", "ood_val", "ood_test_CONFIRMATION") for m in MAPS4]
Pm = [abs(100 * MS[f"n=2500/{sp}/t=0.9/matched/{m}"]["P"]["mean"]) for sp in ("id_test", "ood_val", "ood_test_CONFIRMATION") for m in MAPS4]
c25 = [v for k, v in MS.items() if k.startswith("n=2500/")]; c250 = [v for k, v in MS.items() if k.startswith("n=250/")]
chk("Q4: identity residual < 1e-16; S +1.25..+1.42, P seen +0.02..+0.08, unseen +0.07..+0.22, share<=14%, matched |P|<=0.03 at n=2500 t=.90",
    mech["gate"]["max_identity_residual"] < 1e-16 and _rng(S_rep) == ("+1.25", "+1.42") and _rng(P_seen) == ("+0.02", "+0.08")
    and _rng(P_uns) == ("+0.07", "+0.22") and max(shP) <= 0.145 and max(Pm) <= 0.035,
    "$S$ is $+1.25$ to $+1.42$ points for the four maps")
RA = json.load(open(W / "posthoc_mechanism/reverse_accounting.json"))["summary"]
ra25 = {k: v for k, v in RA.items() if k.startswith("n=2500/")}; ra250 = [v for k, v in RA.items() if k.startswith("n=250/")]
spd = [100 * (RA[f"n=2500/{sp}/t=0.9/fixed/{m}"]["S"]["mean"] + RA[f"n=2500/{sp}/t=0.9/fixed/{m}"]["P"]["mean"]) for sp in ("id_test", "ood_val", "ood_test_CONFIRMATION") for m in MAPS4]
chk("Q4: 72 cells; S +0.55..+1.54 all CI>0; forward share <= 1/5; reverse share <= 1/3 with P' max +0.50 at t=.95; n=250 reverse share up to 73%; shift +1.30..+1.59 at t=.90; gB -0.89 [-1.19,-0.62], +0.79/+0.94; seen t=.80 gB -0.13; condition gB<-Delta/2 matches narrowing in all 72 cells",
    len(c25) == 72 and _rng([100 * e["S"]["mean"] for e in c25]) == ("+0.55", "+1.54") and all(e["S"]["ci95"][0] > 0 for e in c25)
    and max(e["share_P"] for e in c25) <= 0.2 and max(v["share_P_reverse"] for v in ra25.values()) <= 1 / 3
    and f"{100*max(v['P2']['mean'] for v in ra25.values()):.2f}" == "0.50"
    and max((v["P2"]["mean"], k) for k, v in ra25.items())[1].split("/")[2] == "t=0.95"
    and f"{100*max(v['share_P_reverse'] for v in ra250):.0f}" == "73"
    and (f"{min(spd):+.2f}", f"{max(spd):+.2f}") == ("+1.30", "+1.59")
    and f"{100*MS['n=2500/id_test/t=0.9/fixed/HTS-BCE']['gB']['mean']:.2f}" == "-0.89"
    and [round(100 * x, 2) for x in MS['n=2500/id_test/t=0.9/fixed/HTS-BCE']['gB']['ci95']] == [-1.19, -0.62]
    and [f"{100*MS[f'n=2500/{sp}/t=0.9/fixed/HTS-BCE']['gB']['mean']:.2f}" for sp in ("ood_val", "ood_test_CONFIRMATION")] == ["0.79", "0.94"]
    and f"{100*MS['n=2500/id_test/t=0.8/fixed/HTS-BCE']['gB']['mean']:.2f}" == "-0.13"
    and all(v["delta"] > 0 and v["condition_gB_below_minus_half_delta"] == v["narrows_mean"] for v in ra25.values()) and len(ra25) == 72
    and abs(RA["n=2500/id_test/t=0.9/fixed/HTS-BCE"]["S"]["mean"] - MS["n=2500/id_test/t=0.9/fixed/HTS-BCE"]["S"]["mean"]) < 1e-12,
    "selection term is at most a third ($+0.50$ points at $t{=}0.95$;")
# --- RETIRED 2026-10-22 (plan A10): the A8/A9 audit claims of the substantive candidate
# (switcher strata "5 to 8 times" better than union; matched-coverage cancellation in the
# A8 "matched" branch; the guardrail's 90-99% misses; "about 6,400 level labels"; level sign
# resolvable 27%/7%) were withdrawn or replaced; their sources stay in posthoc_audit/ as
# historical records.  New RQ3 checks follow (plan A10, posthoc_closure/).
CL = json.load(open(W / "posthoc_closure/summary_closure.json"))
CE = {(e["budget"], e["split"], e["rule"], e["m"], e["design"]): e for e in CL["estimation"]}
CD = {(d["budget"], d["split"], d["rule"], d["proc"], d["m"]): d for d in CL["decisions"]}
_r = lambda sp, m, d: CE[(2500, sp, "fixed", m, d)]["rmse_pts"]
chk("RQ3 estimate: unseen fixed n=2500 strata RMSE 0.64 (m=100), 0.18 (m=1000)",
    f"{_r('ood_val',100,'E4'):.2f}" == "0.64" and f"{_r('ood_val',1000,'E4'):.2f}" == "0.18",
    "has RMSE $0.64$ points at 100 labels and $0.18$ at 1{,}000")
def _rng(key):
    v = [CL["ratios_vs_E4"][f"n=2500/{sp}/fixed/{key}_vs_E4"] for sp in ("id_test", "ood_val")]
    return min(x[0] for x in v), max(x[1] for x in v)
_m = [_rng(k) for k in ("E8", "E10", "E11")]
_e9, _e3 = _rng("E9"), _rng("E3")
chk("RQ3 estimate: E8/E10/E11 vs strata 0.91-1.12; error-centred union 1.8-2.5; raw union 5.3-8.2 (fixed, n=2500, both splits)",
    f"{min(x[0] for x in _m):.2f}" == "0.91" and f"{max(x[1] for x in _m):.2f}" == "1.12"
    and f"{_e9[0]:.1f}" == "1.8" and f"{_e9[1]:.1f}" == "2.5" and f"{_e3[0]:.1f}" == "5.3" and f"{_e3[1]:.1f}" == "8.2",
    "ratios 0.91 to 1.12 over both populations and all budgets")
import csv as _csv3, glob as _glob3
_ex = [r for f in _glob3.glob(str(W / "posthoc_closure/est_*.csv")) for r in _csv3.DictReader(open(f))
       if r["rule"] == "exact_matched_within_pool" and r["design"] == "E4" and r["m"] == "100" and r["budget"] == "2500"]
_sw = {sp: np.median([int(r["symdiff"]) for r in _ex if r["split"] == sp]) for sp in ("id_test", "ood_val")}
_acc = {sp: np.median([int(r["nA_B"]) for r in _ex if r["split"] == sp]) for sp in ("id_test", "ood_val")}
chk("RQ3 exact matching: median switching items 68 (seen) / 141 (unseen) -> 'about 70 / 140'; accepted ~6,300 / ~11,800; overlap weight 0",
    round(_sw["id_test"], -1) == 70 and round(_sw["ood_val"], -1) == 140 and round(_acc["id_test"], -2) == 6300 and round(_acc["ood_val"], -2) == 11800
    and all(float(r["overlap_weight"]) == 0.0 for r in _ex), "a median of 70 of about 6{,}300")
_g = lambda proc, m, sp="id_test", rule="fixed": CD[(2500, sp, rule, proc, m)]
import csv as _csv4, glob as _glob4
_dr = [r for f in _glob4.glob(str(W / "posthoc_closure/dec_*.csv")) for r in _csv4.DictReader(open(f))
       if r["budget"] == "2500" and r["split"] == "id_test" and r["rule"] == "fixed"]
_worst = max(float(r["sw_fpc"]) for r in _dr if r["proc"] in ("G1", "G3") and float(r["margin_a"]) <= -0.001)
chk("RQ3 decide: seen fixed cells 34/20/6; G3 power .31/.58/.84; G1 .01/.14/.46; mean false switches <= 1% (G1, G3, all m); worst cell 4.0% (all budgets)",
    (_g("G3", 1000)["A_better_cells"], _g("G3", 1000)["A_worse_cells"], _g("G3", 1000)["A_equiv_cells"]) == (34, 20, 6)
    and [f"{_g('G3', m)['A_power']:.2f}" for m in (1000, 2500, 5000)] == ["0.31", "0.58", "0.84"]
    and [f"{_g('G1', m)['A_power']:.2f}" for m in (1000, 2500, 5000)] == ["0.01", "0.14", "0.46"]
    and max(_g(p_, m)["A_error"] for p_ in ("G1", "G3") for m in (25, 50, 100, 250, 500, 1000, 2500, 5000)) <= 0.01
    and f"{100*_worst:.1f}" == "4.0", "mean power 0.31, 0.58 and 0.84 under caps of 1{,}000, 2{,}500 and 5{,}000 labels")
_sc = CL["sample_complexity"]["n=2500/id_test/fixed"]
_mv = _sc["m80_values"]
chk("RQ3 decide: of 34 beneficial cells G3 reaches 80% by 1000 in 2, by 2500 in 13, by 5000 in 26, never 8; lower bound median ~240",
    _sc["beneficial_cells"] == 34 and sum(1 for x in _mv if 0 < x <= 1000) == 2 and sum(1 for x in _mv if 0 < x <= 2500) == 13
    and sum(1 for x in _mv if x > 0) == 26 and _mv.count(-1) == 8 and round(_sc["n_lower_bound_median"], -1) == 240,
    "by 1{,}000 labels in 2, by 2{,}500 in 13 and by 5{,}000 in 26")
_bm = [r for r in _csv3.DictReader(open(W / "posthoc_closure/decision_by_margin.csv")) if r["budget"] == "2500" and r["rule"] == "fixed"
       and r["split"] == "id_test" and r["proc"] == "G3" and r["m"] == "5000" and r["C_better"] == "True" and float(r["H_bin_hi_pts"]) <= 0.1001]
_un = [d for d in CL["decisions"] if d["budget"] == 2500 and d["split"] == "ood_val" and d["proc"] != "G0"]
_un0 = [d for d in CL["decisions"] if d["budget"] == 2500 and d["split"] == "ood_val" and d["proc"] == "G0" and d["m"] == 25]
chk("RQ3 decide: unseen better 1/60; corrected false switches <= 1.1%; original guardrail up to 45% at m=25",
    _g("G3", 100, "ood_val")["A_better_cells"] == 1 and f"{100*max(d['A_error'] for d in _un):.1f}" == "1.1"
    and f"{100*max(d['A_error'] for d in _un0):.0f}" == "45", "false switch in more than 1.1")
_st = [r for r in _csv3.DictReader(open(W / "posthoc_closure/decision_by_seed_threshold.csv")) if r["budget"] == "2500" and r["rule"] == "fixed"
       and r["split"] == "id_test" and r["proc"] == "G3" and r["m"] == "2500" and abs(float(r["t"]) - 0.9) < 1e-9]
# RETIRED 2026-10-22 (final evidence): the sentence "whether the map is better varies by training
# seed (0 to 4 of the 4 maps at t = 0.90)" was cut for space; the value is unchanged
# (posthoc_closure/decision_by_seed_threshold.csv) and the seed dependence is now stated through
# the partition breakdown (seed 3: 41% of its cells flip).
chk("RQ3 decide (value only, text retired): per-seed beneficial maps at t=0.90 range 0-4 of 4",
    sorted({int(r["better_cells"]) for r in _st})[0] == 0 and sorted({int(r["better_cells"]) for r in _st})[-1] == 4)
FS = json.load(open(W / "posthoc_final/summary_final.json"))["rows"]
_f = lambda proc, cap: FS[f"2500/id_test/fixed/{proc}/cap={cap}"]
chk("A11 caps: actual distinct labels at the 5,000 cap G3 4,745, G1 4,131; G3 census 19/60 (16 beneficial)",
    round(_f("G3", 5000)["actual_distinct_labels_mean"]) == 4745 and round(_f("G1", 5000)["actual_distinct_labels_mean"]) == 4131
    and _f("G3", 5000)["census_cells"] == 19 and _f("G3", 5000)["census_beneficial_cells"] == 16,
    "G3 uses 4{,}745 distinct labels on average and G1 4{,}131")
chk("A11 generalize: flips 13/60 (9 at 0.1 pp both pools; 0 unseen); source-oracle target regret 0.06 (reference); same no-FPC G3 decisions 0.10 audit / 0.08 eval at 5,000; signed excess negative in 9 cells",
    _f("G3", 5000)["winner_flips"] == 13 and FS["2500/ood_val/fixed/G3/cap=5000"]["winner_flips"] == 0
    and f"{_f('G3', 5000)['regret_source_oracle_eval_pp']:.2f}" == "0.06" and f"{_f('G3', 5000)['regret_nofpc_audit_pp']:.2f}" == "0.10"
    and f"{_f('G3', 5000)['regret_nofpc_eval_pp']:.2f}" == "0.08" and _f("G3", 5000)["negative_excess_cells"] == 9,
    "regret 0.10 points on the audit pool and 0.08 on the evaluation pool")
PT = json.load(open(W / "posthoc_partition/partition_summary.json"))
_d = PT["distribution"]["n=2500/id_test/0->1"]; _d2 = PT["distribution"]["n=2500/id_test/1->0"]
_bs = {k: v["flip_rate_eps0p10"] for k, v in PT["breakdown_seen_n2500"].items() if k.startswith("seed=")}
chk("A11 partitions: flips at 0.1 pp median 6.5 (p10 2, p90 20) both directions; original 9; seed 3 41% vs others 2-15%; item-random 5; source-optimal median 0.04, always-scalar 0.37, always-switch 0.21; GO rule not met",
    _d["flips_eps0.10"]["median"] == 6.5 and _d2["flips_eps0.10"]["median"] == 6.5 and round(_d["flips_eps0.10"]["p10"]) == 2 and round(_d["flips_eps0.10"]["p90"]) == 20
    and _d["flips_eps0.10"]["original"] == 9 and _d["flips_eps0.10"]["item_random"] == 5
    and f"{100*_bs['seed=3']:.0f}" == "41" and f"{100*min(v for k, v in _bs.items() if k != 'seed=3'):.0f}" == "2" and abs(max(v for k, v in _bs.items() if k != 'seed=3') - 0.145) < 0.005
    and f"{_d['regret_src_opt_pp']['median']:.2f}" == "0.04" and f"{_d2['regret_src_opt_pp']['median']:.2f}" in ("0.04", "0.05")
    and f"{_d['regret_always_scalar_pp']['median']:.2f}" == "0.37" and f"{_d['regret_always_map_pp']['median']:.2f}" == "0.21"
    and not PT["go_rule"]["0->1"]["retain"] and not PT["go_rule"]["1->0"]["retain"],
    "median 6.5 of 60 cells (10th to 90th percentile 2 to 20)")
chk("A11 abstract: joint test mean power 0.84 only under a 5,000-label cap", f"{_g('G3', 5000)['A_power']:.2f}" == "0.84",
    "at a 5{,}000-label cap, the evaluated joint test reaches mean power 0.84")   # language revisions 2026-10-23: text moved, value unchanged
# --- after the internal re-review: per-draw direction claim and n* scope
_mk = [k for k in MS if k.startswith("n=2500/")]
chk("Q4: mean per-draw |g| change follows g_B < -Delta/2 in all 72 n=2500 cells",
    len(_mk) == 72 and all((MS[k]["absdiff"]["mean"] < 0) == (RA[k]["gB"]["mean"] < -RA[k]["delta"] / 2) for k in _mk),
    "follows the condition in all 72 cells")
_cv = json.load(open(W / "posthoc_figdata/fig_closure_coverage.json"))
_ex100 = [e for e in CL["estimation"] if e["budget"] == 2500 and e["rule"] == "exact_matched_within_pool" and e["design"] == "E4" and e["m"] == 100]
chk("RQ3 coverage (120 cells, fixed, m=100): strata/IS/AIPW medians 0.94-0.95, worst 0.83-0.89; AC floor worst 0.94, median 0.98; e-c union median 0.63, worst 0.13; exact strata >= 0.90",
    f"{min(_cv[k+'/m=100']['median'] for k in ('E4','E8','E10')):.2f}" == "0.94" and f"{max(_cv[k+'/m=100']['median'] for k in ('E4','E8','E10')):.2f}" == "0.95"
    and f"{min(_cv[k+'/m=100']['min'] for k in ('E4','E8','E10')):.2f}" == "0.83" and f"{max(_cv[k+'/m=100']['min'] for k in ('E4','E8','E10')):.2f}" == "0.89"
    and f"{_cv['E4ac/m=100']['min']:.2f}" == "0.94" and f"{_cv['E4ac/m=100']['median']:.2f}" == "0.98"
    and f"{_cv['E9/m=100']['median']:.2f}" == "0.63" and f"{_cv['E9/m=100']['min']:.2f}" == "0.13"
    and min(e["cov_min"] for e in _ex100) >= 0.895, "worst cells reach only 0.83 to 0.89")
# --- CivilComments external check (frozen protocol + amendment F1; results/civil/)
CA = json.load(open(W / "civil/analysis.json"))
CPV = json.load(open(W / "civil/provenance.json"))   # copy of the prepared-data provenance
_cc, _cl, _cp = CA["contrasts"], CA["levels"], CA["pattern"]
_res = lambda v: v["ci_bonf4"][0] > 0 or v["ci_bonf4"][1] < 0
_e = [-_cc[f"n=2500/{m}/ece15"]["mean"] for m in M]; _k = [-_cc[f"n=2500/{m}/abs_count"]["mean"] for m in M]
chk("CIVIL: ECE lower by 0.47-1.04 and count error by 7.9-11.9 for all maps at n=2500, all Bonferroni-resolved",
    f"{min(_e):.2f}" == "0.47" and f"{max(_e):.2f}" == "1.04" and f"{min(_k):.1f}" == "7.9" and f"{max(_k):.1f}" == "11.9"
    and all(_res(_cc[f"n=2500/{m}/{q}"]) for m in M for q in ("ece15", "abs_count")), "by 0.47 to 1.04 points) and count error (by 7.9 to 11.9 points)")
_cells = [(m, t) for m in M for t in (80, 90, 95)]
_low = [(m, t) for m, t in _cells if _cc[f"n=2500/{m}/abs_g{t}"]["mean"] < 0]
chk("CIVIL: mean |g| lower in 11 of 12 cells, 10 Bonferroni-resolved; the exception is SRTS-BCE at 0.90",
    len(_low) == 11 and sum(_res(_cc[f"n=2500/{m}/abs_g{t}"]) for m, t in _low) == 10
    and [c for c in _cells if c not in _low] == [("SRTS-BCE", 90)], "11 of 12 map--threshold cells (10 Bonferroni-resolved)")
_pr = CA["primary"]
_gain = [-_pr[m]["mean"] for m in ("HTS-BCE", "SMART+BCE", "PWLinear-3")]
chk("CIVIL primary: HTS/SMART/PWLinear reduce |g(A_0.90)| by 0.14-0.16 (resolved); SRTS-BCE +0.06 [0.03, 0.10] (resolved)",
    all(_pr[m]["resolved"] for m in M) and f"{min(_gain):.2f}" == "0.14" and f"{max(_gain):.2f}" == "0.16" and min(_gain) > 0
    and f"{_pr['SRTS-BCE']['mean']:.2f}" == "0.06" and [f"{x:.2f}" for x in _pr["SRTS-BCE"]["ci_bonf4"]] == ["0.03", "0.10"],
    "SRTS-BCE increases it by 0.06 $[0.03,0.10]$")
chk("CIVIL primary: each sign holds in all five runs and at matched coverage (Bonferroni-resolved, same sign)",
    all(len(_cc[f"n=2500/{m}/abs_g90"]["per_seed"]) == 5 and len({x > 0 for x in _cc[f"n=2500/{m}/abs_g90"]["per_seed"]}) == 1
        and (_cc[f"n=2500/{m}/abs_g90"]["per_seed"][0] > 0) == (_pr[m]["mean"] > 0)
        and _res(_cc[f"n=2500/{m}/abs_gm90"]) and (_cc[f"n=2500/{m}/abs_gm90"]["mean"] > 0) == (_pr[m]["mean"] > 0) for m in M),
    "each sign holds in all five runs and at matched coverage")
_gb = {t: 100 * _cl[f"n=2500/TvA-TS/g_{t}"] for t in (80, 90, 95)}
chk("CIVIL: scalar g_B +0.38 / -0.26 / -0.49 at t = 0.80 / 0.90 / 0.95",
    [f"{_gb[t]:+.2f}" for t in (80, 90, 95)] == ["+0.38", "-0.26", "-0.49"], "($g_B{=}{+}0.38$) and underconfident at 0.90 and 0.95 ($-0.26$, $-0.49$)")
_pk = [k for k in _cp if k.startswith("n=2500/")]
chk("CIVIL: sign of DH from mean D and g_B matches the observed change in all 12 cells (n=2500)",
    len(_pk) == 12 and all(_cp[k]["condition_narrows"] == _cp[k]["observed_narrows"] for k in _pk), "matches the observed change in all 12 cells")
# figure-editorial revision: the abstract no longer calls the 12/12 match "consistency with the condition" (it is an
# accounting identity on measured D and g_B); the 12/12 value is still checked against Section 5.4 below.
chk("CIVIL abstract: maps reduce the accepted-set bias in 11 of 12 settings", len(_low) == 11,
    "the same maps instead reduce the accepted-set bias in 11 of 12 map--threshold settings")
_sf = [_cp[f"n=2500/{m}/t=80"]["share_sel_fwd"] for m in M]
chk("CIVIL: forward selection share 0.45-0.87 at t=0.80", f"{min(_sf):.2f}" == "0.45" and f"{max(_sf):.2f}" == "0.87", "carries 0.45 to 0.87 of the change at $t{=}0.80$")
_ac = CA["action"]["n=2500/HTS-BCE/gB90_toxic"]; _an = CA["action"]["n=2500/HTS-BCE/gB90_nontoxic"]
_rows = [json.loads(l) for f in sorted((W / "civil").glob("rows_s*.jsonl")) for l in open(f)]
_sh = np.mean([r["cov_90_toxic"] / r["cov_90"] for r in _rows if r["budget"] == 2500 and r["method"] == "TvA-TS"])
chk("CIVIL action: predicted-toxic accepted items 3.6% of the scalar's accepted set, scalar bias +4.3 [3.3, 5.1]; non-toxic -0.44",
    f"{100*_sh:.1f}" == "3.6" and f"{_ac['mean']:.1f}" == "4.3" and [f"{x:.1f}" for x in _ac["ci95"]] == ["3.3", "5.1"] and f"{_an['mean']:.2f}" == "-0.44",
    "overconfident by 4.3 points $[3.3,5.1]$, against $-0.44$")
_tm = sum(json.load(open(W / f"civil/distilbert_s{s_}/timing.json"))["total_seconds"] for s_ in range(5)) / 3600
chk("CIVIL: 95,502 evaluated test comments; 86,218 calibration pool; 269,038 training subsample; 8.0% toxic; ~1.3 A100 GPU-hours",
    CPV["sizes"]["test_eval"] == 95502 and CPV["sizes"]["val_calib"] == 86218 and CPV["sizes"]["train_sub"] == 269038
    and f"{100*CPV['label_rate']['train_sub']:.1f}" == "8.0" and f"{_tm:.1f}" == "1.3", "95{,}502 test comments")
_dl = (W / "civil/amendment_F1_all_draws.log").read_text().splitlines()   # logits-only scope check (amendment F1)
chk("CIVIL amendment: 2 of 200 HTS-BCE fits diverged; clip changes nothing in the other 198",
    len(_dl) == 200 and sum("nonfinite_raw 0 maxdiff_finite 0 " not in l for l in _dl) == 2, "after 2 of 200 fits diverged")
CS = json.load(open(W / "civil/audit_summary.json"))
_ce = lambda d, m: CS["estimation"][f"n=2500/fixed/{d}/m={m}"]["rmse_pp"]
_oth = [_ce(d, 100) for d in ("E8", "E10", "E11")]
chk("CIVIL audit: accepted sets ~42,000 in the audit pool; no census at any cap; strata RMSE 0.21 at 100 labels; IS/AIPW/LURE 0.18-0.20; raw union 1.92",
    round(CS["cells"]["n=2500/fixed"]["nA_B_median"], -3) == 42000
    and all(v["census_cells"] == 0 for k, v in CS["decisions"].items() if k.startswith("n=2500/fixed/"))
    and f"{_ce('E4', 100):.2f}" == "0.21" and f"{min(_oth):.2f}" == "0.18" and f"{max(_oth):.2f}" == "0.20" and f"{_ce('E3', 100):.2f}" == "1.92",
    "switcher strata estimate $D$ with RMSE 0.21 points (IS, active inference and LURE 0.18--0.20; raw-correctness union sampling 1.92)")
_g3 = {m: CS["decisions"][f"n=2500/fixed/G3/cap={m}"] for m in (25, 50, 100, 250, 500, 1000, 2500, 5000)}
chk("CIVIL audit: G3 power 0.15 / 0.37 / 0.59 at caps 1,000 / 2,500 / 5,000 over 36 beneficial cells; worst-cell false switch 2.25%",
    _g3[1000]["beneficial"] == 36 and [f"{_g3[m]['power']:.2f}" for m in (1000, 2500, 5000)] == ["0.15", "0.37", "0.59"]
    and abs(max(v["false_switch_worst"] for v in _g3.values()) - 0.0225) < 1e-9,
    "36 beneficial cells with mean power only 0.15, 0.37 and 0.59 under caps of 1{,}000, 2{,}500 and 5{,}000 labels (false switches at most 2.25\\% per cell)")
# --- figure-editorial revision: Figure 3 (three tasks, n = 2500, t = 0.90) and new attributions
F3 = json.load(open(W / "posthoc_figdata/fig3_tasks.json"))["values"]
_ok3 = True
for _t in ("amazon", "yahoo", "civil"):
    for _q in ("abs_count", "abs_gap_90"):
        for _m in M:
            _v = F3[f"{_t}/{_q}/{_m}"]
            if _t == "civil":
                _c = CA["contrasts"][f"n=2500/{_m}/" + ("abs_count" if _q == "abs_count" else "abs_g90")]; _ref = (_c["mean"], *_c["ci95"])
            else:
                _c = ep[_t]["contrasts"]["2500"][_m][_q]; _ref = (100 * _c["point"], 100 * _c["ci95"][0], 100 * _c["ci95"][1])
            _ok3 &= all(abs(a - b) < 1e-12 for a, b in zip(_ref, (_v["mean_pp"], *_v["ci95_pp"])))
chk("Fig.3: 24 plotted means/95% intervals equal the canonical files (Amazon/Yahoo x100 once, CivilComments unscaled); CivilComments SRTS accepted-set +0.0647 pp",
    _ok3 and len(F3) == 24 and abs(F3["civil/abs_gap_90/SRTS-BCE"]["mean_pp"] - 0.064744) < 1e-5
    and all(F3[f"amazon/abs_gap_90/{m}"]["mean_pp"] > 0 and F3[f"amazon/abs_count/{m}"]["mean_pp"] < 0 for m in M)
    and all(F3[f"yahoo/abs_gap_90/{m}"]["mean_pp"] < 0 and F3[f"yahoo/abs_count/{m}"]["mean_pp"] > 0 for m in M)
    and all(F3[f"civil/abs_count/{m}"]["mean_pp"] < 0 for m in M)
    and [m for m in M if F3[f"civil/abs_gap_90/{m}"]["mean_pp"] > 0] == ["SRTS-BCE"],
    "Operational effects depend on the task ($n{=}2500$, $t{=}0.90$).")
chk("abstract/C3: 0.84 attributed to Amazon beneficial cells; CivilComments 0.59 at the same cap",
    f"{_g('G3', 5000)['A_power']:.2f}" == "0.84" and f"{CS['decisions']['n=2500/fixed/G3/cap=5000']['power']:.2f}" == "0.59"
    and "at a 5{,}000-label cap, the evaluated joint test reaches mean power 0.84 in the Amazon cells where the map reduces absolute bias and 0.59 in the corresponding CivilComments cells" in TEX
    and "and 0.59 at the largest cap on CivilComments" in TEX,
    "0.84 in the Amazon cells where the map reduces absolute bias and 0.59 in the corresponding CivilComments cells")   # language revisions: text moved, values unchanged
bad = 0
for d, ok, txt in checks:
    print(("PASS " if ok else "FAIL ") + d + ("" if txt is None else f"   [text: {txt}]"))
    bad += (not ok)
print(f"{len(checks) - bad}/{len(checks)} checks pass")
sys.exit(1 if bad else 0)

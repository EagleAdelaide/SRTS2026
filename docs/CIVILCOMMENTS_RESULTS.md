# CivilComments external check: results (frozen protocol + amendment F1)

**Status.**
- One test analysis with the frozen code: commit 8a0b19c, amendment F1 at ba63d49, and the summarizer and table generator committed before outcomes at 3611bbc and 4aa9d3c.
- No threshold, map, seed or source was searched after outcome access.
- Every outcome is reported below, supportive or not.
- This is a **single-architecture external check on a moderation task**. It does not replicate the Amazon reviewer shift: the source has no user or article ids, so transfer is **not tested** and the identity analysis is not reported.

## Execution
| stage | where | job / process | outcome |
|---|---|---|---|
| backbone training + logit export | Slurm GPU (a100) | array **15991542**, tasks 0–4 = seeds 0–4 | all COMPLETED (DONE); **1.286 A100 GPU-hours** total (`GPU_ACCOUNTING.md`) |
| calibration (`civil_calibrate.py`) | login node, CPU | seeds 0–4 in parallel | the first run stopped at its argmax guard (seed 3), before any output, which led to amendment F1; the rerun of all seeds with F1 code completed: 1,000 rows, 10 draw-0 caches |
| endpoint analysis (`civil_analyze.py`) | CPU | — | `results/www2027/civil/analysis.json` |
| audit (`civil_audit.py`, R = 400) | CPU | 5 processes | `results/www2027/civil/audit_{est,dec}_s{0..4}.csv` → `civil_summarize.py` → `audit_summary.json` |

**Data.**
- 95,502 evaluated test comments: an audit pool of 47,919 and an evaluation pool of 47,583, after excluding 1,818 duplicate-text rows.
- Backbone test error rate about 4.7%; about 6.6% of predictions are toxic (seed 0).
- Epoch-check accuracy, logged only: 0.952–0.954.

## Primary family (pre-specified)
|g(A_0.90)| for each map minus TvA-TS at n = 2,500, fixed threshold, in percentage points. Intervals come from a seed→draw bootstrap (B = 20,000) with a Bonferroni correction over 4 (two-sided 1 − 0.05/4).

| map | mean | Bonferroni interval | resolved | per-seed means (5/5 same sign) |
|---|---|---|---|---|
| HTS-BCE | −0.151 | [−0.202, −0.098] | yes: map better | −0.145, −0.142, −0.158, −0.105, −0.204 |
| SRTS-BCE | **+0.065** | [+0.029, +0.102] | yes: **map worse** | +0.060, +0.086, +0.110, +0.027, +0.040 |
| SMART+BCE | −0.140 | [−0.210, −0.064] | yes: map better | −0.132, −0.127, −0.205, −0.062, −0.173 |
| PWLinear-3 | −0.158 | [−0.215, −0.099] | yes: map better | −0.156, −0.129, −0.203, −0.102, −0.198 |

**Reading.**
- Three of the four conditional maps reduce the accepted-set bias, and SRTS-BCE increases it. This is mixed and does **not** replicate the Amazon unseen-reviewer pattern, where all four maps increased |g|.
- The magnitudes are small: at most 0.16 pp at t = 0.90 and at most 0.39 pp at t = 0.95.

## Descriptive (n = 2,500 unless stated)
**Aggregate endpoints, map minus scalar, all Bonferroni-resolved.**
- ΔECE₁₅: −0.47 (SRTS-BCE) to −1.04 (SMART+BCE) pp.
- Δ|count error|: −7.9 to −11.9 pp.
- Scalar levels: ECE 1.53; count error −17.6%, i.e. errors underestimated.

**Accepted set, Δ|g| at fixed thresholds.** Twelve map × threshold cells:

| threshold | HTS-BCE | SRTS-BCE | SMART+BCE | PWLinear-3 |
|---|---|---|---|---|
| 0.80 | −0.17 | −0.14 | −0.21 | −0.12 (Bonferroni interval contains 0: not statistically resolved) |
| 0.90 | −0.15 | **+0.06** | −0.14 | −0.16 |
| 0.95 | −0.39 | −0.29 | −0.36 | −0.38 |

- The endpoint-disagreement pattern (lower ECE and count error with larger |g|) occurs in **1 of 12** cells: SRTS-BCE at t = 0.90.
- At n = 250 it occurs in 2 of 12: SMART+BCE at 0.90 and at 0.95, and neither Δ|g| is resolved.

**Matched coverage, t = 0.90.**
- Same signs as at the fixed threshold, all Bonferroni-resolved.
- HTS −0.095, SRTS +0.052, SMART −0.121, PWLinear −0.027.

**RQ2 boundary.**
- Scalar accepted-set bias g_B is +0.38 at t = 0.80 (overconfident), −0.26 at 0.90 and −0.49 at 0.95 (underconfident).
- The maps lower confidence at 0.80 (D from −0.43 to −0.61) and raise it at 0.95 (D from +0.29 to +0.55).
- The sign condition, evaluated on the mean D and g_B (narrows iff g_B < −D/2 for D > 0, or g_B > −D/2 for D < 0), **agrees with the observed mean change in 12 of 12 cells at n = 2,500** and 9 of 12 at n = 250.
- This is an elementary identity applied to means. It shows consistency, not a mechanism test.

**S/P accounting.** The selection share of the change (forward / reverse order) is:
- t = 0.80: forward 0.45–0.87, reverse 0.03–0.34;
- t = 0.90: forward 0.24–0.74, reverse 0.23–0.91;
- t = 0.95: forward 0.02–0.13, reverse 0.03–0.40.

Unlike Amazon at n = 2,500, a different accepted set accounts for a large share at t ≤ 0.90 in at least one order.

**Action-specific, t = 0.90.**
- Predicted-toxic accepted items, i.e. automatic removals: 3.6% of the scalar's accepted set (3.2% of all items).
- Their scalar bias is **+4.29 pp [3.34, 5.10]** (95% seed→draw interval).
- The maps reduce it to +1.11 (SRTS), +1.52 (PWLinear), +1.95 (HTS) and +2.75 (SMART).
- Predicted non-toxic accepted items, i.e. automatic publication: scalar −0.44 [−0.51, −0.36]; maps −0.05 to −0.40.
- At t = 0.80 the scalar's toxic-accepted bias is +8.57 pp; at t = 0.95 it is +1.31 pp, and SRTS-BCE overshoots to −1.73.
- A whole-set forecast hides a bias concentrated in the removal action.

**n = 250.**
- ΔECE −0.27 to −0.43 (all resolved).
- Δ|count| −2.8 to −5.9 (not resolved with Bonferroni).
- Δ|g(A_0.90)|: HTS −0.07 [−0.23, +0.11] (not statistically resolved); SRTS −0.11 [−0.21, −0.03]; SMART +0.06 [−0.20, +0.43] (not statistically resolved); PWLinear −0.15 [−0.28, −0.04].
- The two diverged HTS-BCE fits (amendment F1) are included as fitted.

## Audit (within-pool only; `civil/audit_summary.json`)
**Setting.**
- Audit pool: 47,919 comments, split by item hash. The truth is the pool's full labels.
- Variances are finite-population.
- Each cell gets 400 simulated audits.
- Accepted sets in the audit pool hold about 42,000 comments (median |A_B| = 42,221), so **no cap produces a census**, unlike Amazon.

**Cells at n = 2,500, fixed thresholds** (60 = 5 seeds × 4 maps × 3 thresholds):
- the map is better (margin ≥ 0.1 pp) in 36, worse in 14 and within the margin in 10;
- median D = +0.24 pp; median H = −0.15 pp.

**Estimating D** (RMSE in pp, mean over 60 cells; n = 2,500, fixed thresholds):

| design | 100 labels | 1,000 labels | median / worst-cell coverage at 100 |
|---|---|---|---|
| switcher strata (E4) | 0.21 | 0.057 | 0.93 / 0.82 |
| strata, Agresti–Coull floor (E4ac) | 0.21 | 0.057 | 0.98 / 0.93 |
| error-centred IS (E8) | 0.19 | 0.059 | 0.93 / 0.84 |
| active inference, Poisson AIPW (E10) | 0.18 | 0.047 | 0.93 / 0.87 |
| LURE (E11) | 0.20 | 0.056 | 0.94 / 0.85 |
| error-centred union (E9) | 0.71 | 0.22 | **0.41** / 0.04 |
| raw-correctness union (E3) | 1.92 | 0.60 | 0.89 / 0.20 |

The Amazon pattern recurs:
- designs that centre on errors and separate the signed weights estimate D to about 0.2 pp with 100 labels;
- raw-correctness union sampling is about 9× worse;
- error-centred union intervals under-cover.

**Exact matched coverage.**
- The median number of items with nonzero audit weight is **0**: in most cells the matched accepted sets coincide, so D is label-free.
- The likely reason, not tested: in a binary task the HTS and SMART temperatures are functions of the margin, so the ranking is preserved.

**Deciding whether |g| shrinks** (n = 2,500, fixed thresholds, 36 beneficial and 14 harmful cells):

| cap | G3 power | G3 false switch: mean / worst cell | G1 power | G2 power (plug-in, uncalibrated) |
|---|---|---|---|---|
| 250 | 0.005 | 0.09% / 0.5% | 0.000 | 0.001 |
| 1,000 | 0.15 | 0.45% / 2.25% | 0.000 | 0.11 |
| 2,500 | 0.37 | 0.61% / 1.5% | 0.009 | 0.21 |
| 5,000 | 0.59 | 0.27% / 1.0% | 0.085 | 0.35 |

- G3 reaches 80% power in 0 of 36 beneficial cells by 1,000 labels, 3 by 2,500 and 13 by 5,000; **23 never do**.
- G3 always uses exactly the cap. G1 uses 4,851 distinct labels on average at the 5,000 cap.
- The worst-cell false switch over all caps is 2.25% for G3 and 0 for G1; G2 reaches up to 4.0%.

**Reading.** This is an independent replication of the RQ3 separation: estimation is cheap and deciding is expensive. With no census and smaller margins (median |H| about 0.15–0.19 pp), the decision is harder than on Amazon seen reviewers, where power was 0.84 at the 5,000 cap with 19 census cells. These are measured powers of the evaluated tests, not lower bounds.

**Other branches.**
- **n = 250, fixed thresholds:** 28 beneficial cells; G3 power 0.16 / 0.42 / 0.65 at the 1,000 / 2,500 / 5,000 caps.
- **Exact-matched branch, n = 2,500:** 35 beneficial cells; G3 power 0.20 / 0.38 / 0.55.

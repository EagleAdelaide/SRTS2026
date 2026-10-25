# POST-HOC CONTROLLED / CORRECTIVE ANALYSIS PLAN (written 2026-10-16, before running)

Identity: every analysis below is proposed AFTER all earlier results,
including both single-unblinding confirmation sets, were seen. This plan
fixes scope, comparisons and multiplicity before the new numbers exist; it
is NOT a preregistration and its results are NOT new independent
confirmations. The frozen protocol (WEB_PROTOCOL.md, FREEZE_MANIFEST.json,
confirmation rows and stamps) is untouched; the six frozen primary
comparisons keep their frozen 95% inference.

## P1. Figure 2 split fix (F1) — no inference
- Plot Amazon nested families from `test_split == "id_test"` rows only (the
  caption's declared object); C100 from `test`. A split argument is required;
  rows from more than one split abort the script.
- Full-pool replicate draws (identical calibration set, identical outputs)
  are collapsed to one draw before averaging; the dashed K=3 reference uses
  the same rows (same cell, split, budgets, sampling, seeds, draws) as the
  solid lines of its panel.
- Regression tests: mixed split rejected; unique (cell, split, seed, budget,
  draw, method) keys; complete pairing of every series over the same
  (seed, draw) set; K=1 (no floor) and Poly1 equal the core TvA-TS rows of the
  same cell/split/budget/draw (tolerance 1e-9).

## P2. Score-construction control (F2): fixed 2x2 matrix
Variants, all fit inside the same draw on the same calibration subset:
| | fitting scores = OOF (cross-fitted) | fitting scores = IN (full-refit router on its own training rows) |
|---|---|---|
| discrete K=3 | D-OOF (= deployed SRTS-BCE) | D-IN |
| PWLinear-3 | C-OOF | C-IN (= reported PWLinear-3) |
- Shared per draw: calibration subset, StratifiedKFold(seed) fold IDs,
  fold-train standardization for OOF scores, full-split standardization +
  one full-refit L2 logistic router used for IN scores and for deployment
  routing of every variant. No test label enters any fit or choice.
- D-IN: thresholds = empirical 1/3, 2/3 quantiles of IN scores; one bounded
  top-label-BCE temperature per group, groups < 50 fall back to the pooled
  temperature (same rule as D-OOF).
- C-OOF: the unchanged PWLinear-3 fitter with anchors at the {1/6,1/2,5/6}
  quantiles of the OOF scores.
- Declared non-matched settings (kept as in the original methods, not
  changed to claim a "complete match"): D fits 3 independent bounded scalar
  temperatures in [0.05,20] with a 50-sample floor; C fits 3 softplus anchor
  values jointly with linear interpolation and flat extrapolation, no floor,
  L-BFGS-B (Nelder-Mead fallback), init at the pooled BCE scalar. Both use the
  top-label BCE objective and no regularization.
- Scope: c100_vit/test (5 seeds; budgets 250, 625, 1250 with 20 draws; 2500
  = full pool, 1 draw) and amazon_distilbert/id_test (5 seeds; 20 draws at
  each budget). Budgets 625 and 1250 are included only because the paper
  makes an every-budget claim about PWLinear-3; inference is at 250 and 2500.
- Corruption extension (needed because the paper attributes a corruption
  advantage to the continuous map): C100-C, 95 corruption x severity cells,
  5 seeds, fit once on the full clean split (n = 2500), protocol of
  run_shift_suite.py; per-seed mean over the 95 cells.
- Reproduction gate (must pass before any contrast is read): D-OOF equals
  saved SRTS-BCE rows and C-IN equals saved PWLinear-3 rows (ECE15, |diff| <
  1e-9) for every matched (seed, budget, draw, split) in suite/ and shift/.
  A failure is investigated, not bypassed.
- Contrasts (ECE15, lower is better), per (cell, budget in {250, 2500}) and
  for C100-C:
  K1 C-OOF − D-OOF (map type, matched OOF construction)
  K2 C-IN − D-IN (map type, matched IN construction)
  K3 C-IN − D-OOF (the originally reported, mismatched comparison)
  K4 C-OOF − C-IN (score construction, continuous map)
  K5 D-OOF − D-IN (score construction, discrete map)
  K6 K1 − K2 (map x construction interaction)
  => 6 x 5 settings = 30 contrasts.
- Inference: paired per-draw deltas; seed->draw hierarchical bootstrap
  (B = 20000, RNG seed 20261016, one replicate array per contrast) for the
  suite cells; for C100-C and for C100 n = 2500 only seeds vary (5 values),
  reported as seed bootstrap + per-seed signs. Draws are never treated as
  training seeds. Report 95% and Bonferroni (1 − 0.05/30) intervals from the
  same replicate array. A map-type attribution (K1/K2) is written in the
  paper only if its Bonferroni interval excludes zero; otherwise "not
  statistically resolved". Secondary metrics (NLL, top-label BCE, Brier)
  are descriptive.

## P3. Operational endpoints (Q3) — confirmation rows, post-hoc
- Data: CONFIRMATION/amazon_distilbert.jsonl (5 seeds x 20 draws) and
  CONFIRMATION/yahoo_distilbert.jsonl (3 seeds x 20 draws), unchanged.
- Methods: TvA-TS (scalar reference), HTS-BCE, SRTS-BCE, SMART+BCE,
  PWLinear-3 (= C-IN, original implementation; C-OOF is not evaluated on any
  confirmation set). Budgets 250 and 2500. Thresholds 0.80, 0.90 and 0.95,
  all reported.
- Per-row quantities (from saved fields; definitions verified in
  registry.ops_metrics): ECE15, NLL, Brier, top-label BCE; whole-split signed
  and absolute count relative error; per threshold: coverage, actual and
  predicted error rate in the accepted set, signed and absolute gap g(A) =
  accerr − prederr, accepted-set count relative error (prederr − accerr) /
  accerr (undefined if accerr = 0 or the set is empty, counted and reported),
  rejected-set gap g(A^c) = (g(all) − cov·g(A))/(1 − cov) with g(all) =
  −count_rel_err · act_err_count / n (undefined if cov ∈ {0,1}); fixed-coverage
  risk at 80/90/95%; AURC.
- Identity check: cov·g(A) + (1−cov)·g(A^c) = g(all) on every row.
- Core contrast set (each method − TvA-TS, paired per draw): |whole-split
  count error|, |g(A)| at 0.80, 0.90, 0.95 => 4 endpoints x 4 methods x 2
  budgets x 2 tasks = 64 contrasts, 95% hierarchical intervals and
  Bonferroni (1 − 0.05/64) sensitivity. Coverage, accepted-set error rates,
  accepted-set count error, rejected-set gap, risk at coverage, ECE/NLL:
  paired 95% intervals reported descriptively, not counted as tests.
- Reading rule: the paper reports the pattern over the whole 64-cell grid
  (counts of resolved improvements / deteriorations per endpoint), never a
  single selected cell; a higher coverage or a higher accepted-set error rate
  is not called an improvement or a deterioration by itself.

## Addendum A1 (2026-10-16, after the pilot, before the full run)
Pilot (seed 0; n = 250 draws 0-1 and n = 2500; both suite cells) failed the
1e-9 gate for C-IN only: |ΔECE15| vs saved PWLinear-3 rows was 3.7e-8 to
3.2e-5 pp, with L-BFGS-B objective values agreeing to ~2e-9 and both runs
converged. D-OOF matched saved SRTS-BCE rows exactly. Investigation: in the
current environment the original code path (run_suite.fit_methods "core")
and the F2 variants give identical outputs (difference 0.0 for PWLinear-3 vs
C-IN and SRTS-BCE vs D-OOF on all six pilot draws). The residual is
host-level floating-point drift of the L-BFGS-B end point (saved rows: icelake
batch nodes; this run: login node), not a code difference.
Amended gate: (i) in-process identity of the original and F2 code paths on a
fixed sample of draws (exact); (ii) D-OOF vs saved SRTS-BCE exact (1e-9);
(iii) C-IN vs saved PWLinear-3 within 1e-3 pp ECE15, with the maximum
reported. All contrasts use the recomputed rows, so the four variants of a
draw always come from one process.

## Addendum A2 (2026-10-16, after the full run, before reading any contrast)
Gate result on the full matrix: (i) in-process identity on 12 further draws
(2 cells x seeds 1,4 x n 625/1250/2500): max |ΔECE15| = 0.0; (ii) D-OOF vs
saved SRTS-BCE: exact on all suite and shift rows; (iii) C-IN vs saved
PWLinear-3: within 1e-3 pp on all but 2 of the matched suite draws (Amazon
n = 2500: seed 0 draw 15, 3.1e-3 pp; seed 3 draw 12, 1.1e-3 pp); shift max
2.1e-6 pp. For both exceedances the saved and recomputed fits converged to
the same training objective (|Δ| ≤ 1e-12) with different anchor values, i.e.
a flat optimum of the PWLinear-3 fit. Recorded as a gate exceedance, not
re-labelled as a pass; the analysis proceeds on the recomputed rows (all
four variants of a draw from one process), as fixed in A1.

## Addendum A3 (2026-10-16, written before computing; post-hoc)
The draft's sentence "budget x capacity interactions are interval-supported"
rests on method-level interactions (SRTS-BCE vs TvA-TS / SMART+BCE), not on
within-family contrasts. To support or retract within-family wording, compute
exactly: family A (no floor) K=8 − K=1 and family B Poly4 − Poly1, on
c100_vit/test and amazon_distilbert/id_test, at n = 250, n = 2500 and the
interaction Δ(250) − Δ(2500) => 2 x 2 x 3 = 12 contrasts; same hierarchical
bootstrap (B = 20000, RNG 20261016, full-pool replicates collapsed); 95% and
Bonferroni 1 − 0.05/12. The per-budget means were already visible in the
figures; the intervals were not. Wording rule: "interval-supported" only if
the Bonferroni interval excludes zero.

## Addendum A4 (2026-10-16, after one simulated review, before computing)
Disclosure: the simulated review had computed descriptive means of
|g(A_t)| at n = 2500 on Amazon id_test and ood_val and on the anchor; those
means were seen before this addendum. Nothing below has been computed yet.
(i) Endpoint matrix on development splits — Amazon id_test, Amazon ood_val,
Yahoo id_test (suite core rows): same maps, budgets, thresholds, endpoints
and contrasts as P3; Bonferroni family = the 32 contrasts of each split.
Purpose: test whether the confirmation-set pattern of Q3 is specific to
unseen reviewers.
(ii) Seed-level sensitivity: for the six frozen primaries, all P3/A4(i)
core contrasts, all P2 and A3 contrasts, a t interval on the per-seed means
(df = seeds − 1) and the number of seeds sharing the sign of the mean.
Reporting rule: a statement is called resolved only by its pre-declared
interval; where it is not resolved by the seed-level interval this is said;
in settings where only seeds vary (anchor n = 2500, C100-C) the Bonferroni
mark is replaced by "k/k seeds".
(iii) Matched coverage: refit TvA-TS, HTS+BCE, SRTS-BCE, SMART+BCE and
PWLinear-3 (original construction) for every seed and draw at n = 250 and
2500; at each t, let q be TvA-TS's coverage in that draw; for every map take
its ceil(q N) most confident items and compute g there. Contrasts map −
scalar on |g| for Amazon id_test, ood_val, confirmation and Yahoo id_test,
confirmation: 5 x 4 x 3 x 2 = 120, 95% hierarchical intervals and Bonferroni
(120) sensitivity. Gate: the refit's fixed-threshold metrics must reproduce
the saved rows (exact for scalar, HTS, SMART, SRTS; <= 1e-3 points for
PWLinear-3).
(iv) No selection rule is evaluated; the number of labels needed to measure
g(A_t) is stated analytically (binomial standard error), not simulated.

## Addendum A5 (2026-10-16, after an internal review of 2d68d98, before computing)
Purpose: a direct test of whether the accepted-set effect of a conditional
map differs between seen and unseen Amazon reviewers, replacing the
comparison of two separate per-split tests.
Disclosure: the per-split means of Δ (Tables 1 and 6, endpoints*.json,
analysis_matched.json) are known; since Γ below is linear in the rows, its
point estimates follow from them. No interval, seed-level statistic or
pairing check for Γ has been computed. Post hoc, not an independent
confirmation; same fixed test sets, seeds and draws as P3/A4.
(i) Estimand. For map m ∈ {HTS-BCE, SRTS-BCE, SMART+BCE, PWLinear-3},
budget n ∈ {250, 2500}, threshold t ∈ {.80, .90, .95}, seed s, draw d and
evaluation split X: Δ_X = |g_m(A_m)| − |g_TvA(A_TvA)| on X, and
Γ = Δ_OOD − Δ_ID with ID = Amazon id_test and OOD ∈ {ood_val,
ood_test_CONFIRMATION}, each computed separately. Selection rules: fixed
threshold (A = {c ≥ t}, field errgap_a{t}) and matched coverage (A4(iii),
field g_matched_a{t}). Family: 4 × 2 × 3 × 2 × 2 = 96 contrasts, all
reported; no pruning or re-splitting after seeing results.
(ii) Pairing. Γ is formed only within one (m, n, s, d). Source: the A4(iii)
rows (posthoc_matched/), where one fit per (s, n, d) is evaluated on all
three splits in the same process, so the ID and OOD terms share the fitted
calibrator by construction. Checks before any interval: (a) all three
splits present for every key, no duplicates; (b) calibration indices
recomputed (CPU, no fitting) from the shared rule
rng = default_rng(77e6 + 1e6 s + 1000 n + d), natural_subsample on the
validation pool, hashed per draw, and compared with the sampler record of
the original suite rows (id_test and ood_val) and CONFIRMATION rows
(actual_n, pool_hash, class_counts); (c) fixed-threshold errgap of the
A4(iii) rows vs the original suite / CONFIRMATION rows (A4(iii) gate,
re-run). Sensitivity: Γ on the fixed-threshold estimand recomputed from the
original rows (suite id_test, suite ood_val, CONFIRMATION) with the maximum
absolute difference to the primary reported. Missing fields are named: the
CONFIRMATION rows carry no fit_report and no row stores a calibration-index
hash; (b) and (c) are the substitutes.
(iii) Inference. Seed → draw hierarchical bootstrap as in P3 (index arrays
shared by all 96 contrasts), B = 200,000, primary stream RNG 20261016;
95% percentile intervals and Bonferroni 1 − 0.05/96. Monte Carlo
stability: a second independent stream (RNG 20261017, B = 200,000);
report the maximum absolute bound difference and the number of
exclusion-of-zero decisions that differ at each level. Seed-level
sensitivity: t interval on the 5 per-seed means (df = 4), uncorrected 95%
and Bonferroni-corrected (96), plus the number of seeds sharing the sign
of the mean. Note fixed in advance: with 5 seeds, the Bonferroni tail
(2.6e-4) is below the probability of a resample that repeats one seed
(3.2e-4 per seed), so the Bonferroni bootstrap bound is close to the most
extreme single-seed mean; this is stated with the results.
(iv) Reporting rule. Per contrast: point, 95% interval, Bonferroni(96)
interval, seed-level intervals, k/5 seeds. Summary counts by estimand ×
OOD split × budget. "A difference between seen and unseen users is
interval-supported" only where the 95% interval excludes zero, and "under
the family correction" only where the Bonferroni(96) interval does. A
contrast with an interval containing zero is "not statistically resolved".
Γ is a difference between fixed test populations evaluated with the same
calibrators; it is not a causal effect of user identity (the two splits
also differ in products, label mix and base error rate).
A5 note (2026-10-16, after the pairing checks, before any interval): (a)
no missing or duplicate keys; (b) 3,000 sampler records (5 seeds x 2
budgets x 20 draws x 3 splits x 5 maps) equal the records recomputed from
the shared rule, 200 calibration-index hashes stored; (c) reproduces the
A4(iii) gate result exactly: three PWLinear-3 draws exceed 1e-3 points
(ECE 3.1e-3 and 1.1e-3 on id_test, errgap_a80 1.9e-3 on ood_val; all other
maps <= 7e-16). (c) compares the A4(iii) refit with the original rows; it
does not bear on the pairing of Γ, which uses A4(iii) rows only. The
analysis therefore proceeds on (a) and (b); the three exceedances are
reported, and their effect on Γ is the pre-specified original-row
sensitivity.
A5 note 2 (2026-10-16, after the A5 results, before computing the
following): two additions that are not new contrasts and do not enter the
96-family. (d) Model version: per seed and split, SHA-256 of the logits
array and of the labels read by the A4(iii) refit, labels checked against
the released MANIFEST and logits against the released float32 copies; the
saved rows carry no logits hash, so (d) plus the metric reproduction of
(c) is the evidence that the original rows and the A4(iii) rows used the
same model outputs. (e) Display: 95% intervals for Δ_ID and Δ_OOD from the
same primary index arrays, so that one table can show Δ_ID, Δ_OOD and Γ;
their inferential versions with their own families remain those of P3/A4
(endpoints*.json, analysis_matched.json).
A5 note 3 (2026-10-16, after the A5 results and after a targeted internal
review had already computed it; descriptive only): the signed counterpart
Γ_s = (g_map − g_scalar)_unseen − (g_map − g_scalar)_seen, computed from the
same rows, with the scalar's own signed g(A) per split and the share of
draws with each sign. Purpose: to tell apart "the maps shift the
accepted-set bias differently on the two populations" from "the maps shift
it similarly while the scalar's bias has opposite signs". It is not part
of the 96-family, carries no multiplicity claim, and is reported with 95%
intervals from the primary A5 index arrays for description only. Output:
results/www2027/posthoc_gamma/signed_descriptive.json; the A5 outputs are
not changed.

## Addendum A6 (2026-10-16, before computing; post-hoc reviewer-cluster sensitivity)
Authorization: the author approved, in this session, at most 2 CPU-hours
(core-hours) of new CPU computation for this item; no GPU, no training.
Name: post-hoc reviewer-cluster sensitivity (not an independent
validation, confirmation or new-data replication). Known before this
addendum: all A5 results, including the representative-condition values
of Table 7, and the signed description (A5 note 3).
(i) Scope, fixed now: representative condition n = 2500, t = 0.90, fixed
threshold only; maps HTS-BCE, SRTS-BCE, SMART+BCE, PWLinear-3 (original
C-IN) against TvA-TS; Γ for OOD validation − ID test and OOD confirmation
− ID test: 8 contrasts. Δ_ID and Δ_OOD reported for display. Not extended
to the 96-family or to matched coverage.
(ii) Question: holding the observed training runs and calibration fits
fixed, how sensitive are Δ_ID, Δ_OOD and Γ to sampling of evaluation
reviewers, with the reviewer as the cluster?
(iii) Predictions: all 5 seeds and draws 0-19 at n = 2500, each draw fit
once by run_suite.fit_methods("core") with the stored RNG rule and 4 CPU
threads (as in A4(iii)); no refitting inside the bootstrap. Per seed,
draw, map and split the cache stores, per reviewer, the sum of (c − b)
over accepted items (c ≥ 0.90) and the number of accepted items. Reviewer
ids: official WILDS metadata via each item's official row. Checked before
this addendum (read-only): ID test 626, OOD validation 1,334, confirmation
1,334 reviewers, 75 reviews each, pairwise disjoint; all ID-test
reviewers appear in training; row order identical across seeds.
(iv) Gate before any bootstrap: per (seed, draw, map, split) the
recomputed accepted-set gap and coverage at t = 0.90 equal the A4(iii)
rows (≤ 1e-12 for TvA-TS, HTS-BCE, SRTS-BCE, SMART+BCE; ≤ 1e-3 points for
PWLinear-3, maximum and draws reported), and the cached sums reproduce
the gap (≤ 1e-12). Failure for a non-PWLinear-3 map stops the analysis.
(v) Cost gate: process CPU time (user + system, all threads) is read after
seed 0; if 5 × (seed-0 CPU) + 0.25 h > 2.0 CPU-hours the run stops before
seed 1 and the gap is reported; the run also stops whenever cumulative
CPU exceeds 2.0 h. Seeds, draws or maps are never dropped to fit the
budget.
(vi) Estimator: item-weighted, g = Σ_r w_r S_r / Σ_r w_r N_r with reviewer
multiplicities w (multinomial: R_X reviewers drawn with replacement per
split). In each replicate one w per split is shared by all maps, the
scalar, all seeds and draws; the ID-test w is shared by both OOD−ID
contrasts; the three disjoint splits are resampled independently. Within
each (seed, draw): g, absolute value, map − scalar, OOD − ID; then mean
over draws within seed and over seeds (equal weights). Point estimate =
unit weights (equals the A5 value). Empty accepted set (Σ w N = 0): that
(seed, draw, map, split) is missing and excluded from its draw mean;
frequency reported.
(vii) Inference: primary = fixed-model reviewer bootstrap, B = 50,000,
RNG 20261021; Monte Carlo stability with an independent stream (RNG
20261022, B = 50,000): maximum bound difference and decision flips. 95%
percentile intervals and Bonferroni(8) (1 − 0.05/8), a family of these 8
contrasts only; it does not revise the 96-family or its seed-level
results. Reported next to the A5 seed→draw intervals for the same 8
contrasts; the two describe different uncertainty and both are shown.
Secondary (pre-specified): joint crossed resampling — in each replicate
the same reviewer weights apply to every model, combined with A5-style
seed→draw indices (RNG 20261023 for the indices); 95% and Bonferroni(8).
It does not remove the five-seed limit.
(viii) Not covered: retraining with different ID reviewers in training or
calibration, product or other unmodelled clusters, future drift. No
stopping on significance; all 8 reported whatever the outcome.
Output: results/www2027/posthoc_reviewer_cluster/.

## Addendum A7 (2026-10-20, before computing; descriptive figure intervals)
Purpose: two redesigned main-text figures. No new contrast family, no
new claim; every point shown is a mean already reported (Tables 1, 4,
8, 9 or the frozen analyses) or a descriptive counterpart of it. All
intervals here are 95% seed->draw hierarchical bootstrap intervals
(B = 20,000, RNG 20261016, index arrays as in P3), labelled
"descriptive" in the captions, with no multiplicity claim; they do not
replace the frozen or pre-declared families.
(i) Figure 1 (metric reversal): Amazon confirmation set, n = 2500, maps
TvA-TS, HTS-BCE, SRTS-BCE, SMART+BCE, PWLinear-3; level means of ECE15,
|count_rel_err| x 100, signed g(A_0.90) x 100 (and coverage at 0.90 in
the caption), each with its descriptive interval; and the paired map −
TvA-TS difference of the same three quantities with its interval.
(ii) Figure 2 (budget crossing): per-draw ECE15 difference map − TvA-TS
for the four maps at n in {250, 625, 1250, 2500} on Amazon ID test, Amazon
confirmation, Yahoo ID test and Yahoo confirmation (16 x 4 = 64 panels'
worth of points, 48 of which are not in any pre-declared family); the
frozen SRTS-BCE − TvA-TS values must reproduce analysis_*.json exactly
(gate, 1e-9) and are marked as frozen in the figure; all others are
post hoc/descriptive. Known before this addendum: all per-budget means
(appendix Tables 8-9) and the n = 250/2500 intervals (Table 1).
Output: results/www2027/posthoc_figdata/{fig1_reversal.json,
fig2_budget.json}; generators gen_fig1_reversal.py, gen_fig2_budget.py.
A7 note (2026-10-20, after computing): Figure 1 displays the level means
and intervals of (i); the paired map − TvA-TS differences and intervals
were computed as specified and are stored in fig1_reversal.json
(diff_vs_scalar) but are not drawn, since Figure 3 shows the paired
contrasts with their pre-declared intervals. Figure 2 gate: SRTS-BCE −
TvA-TS reproduces the frozen analyses to 2.2e-16.

## Addendum A8 (2026-10-20, before computing; mechanism accounting and paired audit simulation)
Status: exploratory throughout. The Amazon confirmation set and the OOD
validation set have been inspected many times; nothing here is a
confirmation. Known before this addendum: everything in the
FINAL_CANDIDATE (tag www27-final-candidate-2026-10-20), including the
signed description (A5 note 3). No GPU work; CPU refits only, of the
same five maps with the same RNG rule as A4(iii)/A6.
(i) Per-item cache. For seeds 0-4, budgets n in {250, 2500}, draws 0-19:
refit TvA-TS, HTS-BCE, SRTS-BCE, SMART+BCE, PWLinear-3 (fit_methods
"core") and store, for id_test, ood_val and ood_test_CONFIRMATION, each
map's top-label confidence per item (float32), the shared correctness
b_i (checked identical across maps: argmax preserved) and reviewer ids.
Gate: the cached values must reproduce the A4(iii) rows' ECE15 and
errgap_a{80,90,95} (<= 1e-9; PWLinear-3 <= 1e-3 points).
(ii) C1 accounting (exact identity, proved in
MECHANISM_IDENTITIES_AND_PROOFS.md and unit-tested): for map C vs
baseline B = TvA-TS, g_C(A_C) - g_B(A_B) = E_{A_C}[c_C - c_B] +
{E_{A_C}[c_B - b] - E_{A_B}[c_B - b]} =: S (score change on C's accepted
set) + P (population/selection change under the baseline residual).
Computed per seed, draw, map (4), split (3), budget (2), rule (fixed t in
{.80,.90,.95}; matched coverage top-ceil(qN)) and summarized by mean over
draws within seed and over seeds, with descriptive 95% seed->draw
intervals (B = 20,000, RNG 20261016) and per-seed values. Reported
quantities: S, P, S + P (= signed difference, checked against the rows),
|g_C| - |g_B|, and the share |P| / (|S| + |P|). Pre-stated falsifiable
questions: (a) is S or P larger in magnitude at the representative
condition (n = 2500, t = .90)? (b) does the sign of P differ between seen
and unseen reviewers? (c) does the baseline's signed bias g_B account for
the sign of |g_C| - |g_B| given S + P (i.e. is |g_B + S + P| - |g_B|
reproduced)? The signed components are not an additive decomposition of
the absolute change; this is stated wherever reported.
(iii) C2 paired audit. Identity (proved and tested): for fixed thresholds
g_C - g_B = d_conf - sum_i w_i b_i with w_i = 1[i in A_C]/|A_C| -
1[i in A_B]/|A_B| and d_conf = mean_{A_C}(c_C) - mean_{A_B}(c_B)
(label-free). Estimators of the label term at a unique-label budget m,
all with known inclusion probabilities, finite-population Horvitz-
Thompson form and a variance estimate: (E1) uniform sample of the
unlabelled stream (labels used only where w_i != 0); (E2) independent
uniform samples of A_B and of A_C (m/2 each, duplicates counted once);
(E3) uniform sample of A_B union A_C; (E4) stratified: overlap stratum
(weight 1/|A_C| - 1/|A_B|) and the two switcher strata, Neyman
allocation on known |w_i| with a floor of 2 per non-empty stratum. The
target g_B itself is estimated by a uniform sample of A_B (E0) with its
own m; the policy below uses E0 + E4 at a shared total m. Audit grid
m in {25, 50, 100, 250, 500, 1000}; 400 Monte Carlo audit draws per cell
(RNG 20261024 + cell index); metrics: bias, RMSE, 95% interval coverage
(normal approximation with HT variance), relative efficiency vs E3.
Populations: Amazon ood_val reviewers split by reviewer id (hash of the
id, even/odd) into an audit pool and an evaluation pool; the same for
id_test; the confirmation set is used only for the C1 accounting.
Calibrators fixed at one draw per seed (draw 0) for the audit simulation,
all 20 draws for the identity checks. Policy: default B; switch to C if
the 95% upper bound of |g_B + Delta_hat| is below the 95% lower bound of
|g_B_hat| (both from the audit pool); evaluate on the evaluation pool:
switch rate, false-switch rate (switch while |g_C| >= |g_B| on the
evaluation pool), regret in |g| vs the oracle map, no-switch frequency.
Comparisons are descriptive; no multiplicity family is declared.
Outputs: results/www2027/posthoc_mechanism/ (cache, identity results),
results/www2027/posthoc_audit/ (simulation rows, AUDIT_LABEL_EFFICIENCY.csv).
(iv) C3 CivilComments: CPU preflight only (data availability, split
definition, protocol freeze); no training without an approved GPU cap.

## Addendum A9 (2026-10-21, before computing; active-testing comparators for the paired audit)
Status: exploratory, same populations, pools and calibrators as A8(iii).
Disclosed as already seen: all A8(iii) results for E0-E4 and the
guardrail policy (n = 2500, fixed thresholds: E4 RMSE 0.11-0.18 x E3;
E4 coverage 93-95% at m >= 100; matched-coverage under-coverage 77-93%;
policy switch/false-switch/missed rates as in posthoc_audit/summary.json).
Reason: the sprint brief asks for a comparison with established
active-testing estimators, which A8 did not include. Three further
estimators of the label term L = sum_i w_i b_i, each with a variance
estimate, are added for the estimation question only; the guardrail
policy is NOT changed (it stays E0 + E4 as declared in A8), whatever
these designs show.
(E5) Active-testing importance sampling (Kossen et al. 2021 style):
m draws with replacement from A_B u A_C with probability
q_i proportional to |w_i| sqrt(s_i), s_i = c_B,i (the scalar's calibrated
confidence as the label-free surrogate for P(b_i = 1); this is the
variance-minimising proposal for E[(w_i b_i)^2] under the surrogate);
Hansen-Hurwitz estimator (1/m) sum_j y_j / q_j with y = w b; variance
s^2(y/q) / m.
(E6) Active inference with a difference (prediction-powered) estimator:
L_hat = sum_i w_i s_i + (1/m) sum_j w_j (b_j - s_j) / q_j, draws with
replacement from q = 0.8 q* + 0.2 q_w, q*_i proportional to
|w_i| sqrt(s_i (1 - s_i)), q_w proportional to |w_i| (defensive mixture,
fixed here, not tuned); variance s^2 / m of the summands.
(E7) Stratified difference estimator: E4's strata and allocation, HT
estimator of sum_i w_i (b_i - s_i) plus the known sum_i w_i s_i. With
s = c_B this equals estimating only P of the C1 identity, S being
label-free.
For E5/E6 the label cost is the number of distinct items drawn (<= m);
efficiency is reported against that mean unique-label count as well as
against m. Same grid (m in {25,...,1000}), R = 400 audits per cell, RNG
20261025 + cell index (cells enumerated in A8's order), same metrics
(bias, RMSE, 95% coverage). Unit tests (unbiasedness, variance) precede
the run. Analytic note stated before computing: no surrogate built from
c_B can reduce the variance of the level estimate g_B = mean_{A_B}(c_B -
b), because that level is the surrogate's own residual; the guardrail's
level term is therefore unaffected by E5-E7.
A9 note 2 (2026-10-21, before computing; question separation). The sprint
brief asks to keep four questions apart: (Q-diff) estimating g_C - g_B
(E1-E7 above); (Q-level) estimating g_B; (Q-select) choosing the map with
the smaller |g| (the A8 guardrail); (Q-risk) whether the choice reduces
accepted-set risk. For the same cells as A8(iii) (draw 0, audit and
evaluation pools, fixed and matched rules) we add two deterministic
quantities, no Monte Carlo: (Q-level) the exact SRSWOR standard error of
the E0 level estimate at m/2 labels, sqrt((1 - n/N) s_b^2 / n) on the
audit pool's A_B, reported against |g_B| (share of cells with
1.96 SE < |g_B|, i.e. where the level's sign is resolvable); (Q-risk) the
accepted-set error rate and coverage of B and C on the evaluation pool,
and the expected change under the A8 policy (switch rate x (risk_C -
risk_B)) next to the oracle-|g| choice. The policy is not modified and no
threshold is tuned on these quantities.
A9 note 3 (2026-10-21, after an internal adversarial review of commit
7e7ca39, before computing). Disclosed as already seen (from the review):
in the reverse order of the accounting the selection term reaches about
+0.33 (21%) at t = .90 and +0.50 (32%) at t = .95 and changes sign at
t = .80 for three maps; over the 23 seen-reviewer cells where a switch
helps, the guardrail misses about 90-99% of beneficial switches; on
unseen reviewers at matched coverage, false switches track the chance of
observing no error in the level sample (zero-width Wald interval).
Corrections stated before computing: (1) the A9 analytic note "no
surrogate built from c_B can reduce the variance of g_B" is wrong in
general: a regression estimator with a surrogate reduces the level's
variance by the factor 1 - rho^2, rho = corr(b, surrogate) within A_B;
(2) "the residual's sign decides the direction" is wrong: for Delta > 0,
|g_C| < |g_B| iff g_B < -Delta/2. Computations added (descriptive, no
policy re-tuning): (a) the reverse-order accounting S' = mean_{A_B}(c_C -
c_B), P' = mean_{A_C}(c_C - b) - mean_{A_B}(c_C - b), with the same
summaries as A8(ii); (b) rho^2 of b on c_B within the audit pool's A_B
per A8(iii) cell; (c) the missed-switch rate conditional on cells where
the map is better; (d) the guardrail at matched coverage and the share
of level samples with no observed error; (e) a disclosed sensitivity
policy that differs from A8 only in (i) a level variance that is never
zero (Agresti-Coull: p~ = (x + 2)/(n + 4)) and (ii) Delta from E6 (nominal
coverage under both rules), same R = 400, RNG 20261026 + cell; it is
reported next to, not instead of, the A8 policy; (f) the number of level
labels needed so that 1.96 SE < |g_B + Delta/2| on the evaluation pool,
per cell.

## Addendum A10 (2026-10-22, before computing; estimand repair, within-pool vs transfer, decision comparators)
Status: exploratory, same Amazon splits, same per-item cache, same reviewer
parity pools as A8(iii). Trigger: an independent review of tag
www27-substantive-candidate-2026-10-21 (its package files were not on disk;
the findings were re-derived here). Disclosed as already seen, verified
2026-10-22 from the existing rows before this addendum: (1) in the A8/A9
"matched" audit branch only 5/120 cells (n = 2500, E4, m = 100; 4/60 ID
test, 1/60 OOD validation) have |A_B| = |A_C| inside the audit pool
(median count difference 5 of ~9,000), because top-k sets were formed on
the full split and then intersected with the pool; (2) full-label
audit-pool and evaluation-pool winners (|g_C| < |g_B|) disagree in 13/60
ID-test fixed-threshold cells (mean oracle-transfer regret 0.0609 pp),
12/60 matched (0.0706 pp) and 0/60 OOD validation; (3) the "about 6,400
labels" figure is the oracle planning expression 1.96^2 r(1-r)/H^2 on the
evaluation pool with D known, over 23 cells, not measured sample
complexity; (4) no error-centred sampler and no joint (D,H) decision
procedure had been evaluated. All A8/A9 outputs stay unchanged as
historical records; new outputs go to results/www2027/posthoc_closure/.
(i) Branches. fixed: A_t = {c >= t} within each pool (as before).
stream_matched_then_restricted: the A8 construction, relabelled, kept for
the record. exact_matched_within_pool: for target pool P, k_P = #{i in P:
c_B,i >= t}; A_B^P and A_C^P are the top-k_P items of P by c_B and by c_C,
ties broken by (-c, item index) (stable sort); assert |A_B^P| = |A_C^P|
and zero overlap weight wherever cancellation is invoked; regression
tests must fail for the old construction.
(ii) Estimation (target: the audit pool's own D = g_C - g_B; FPC
variances). Designs E1-E7 as before on the exact branch, plus two
error-centred controls on all three branches, using D = d_conf +
sum_i w_i e_i, e = 1 - b: E8 error-centred importance sampling (with
replacement, q = 0.8 q* + 0.2 q_w, q*_i prop. to |w_i| sqrt(s_e,i),
s_e = 1 - c_B, Hansen-Hurwitz; cost = mean distinct items) and E9
error-centred union (SRSWOR on A_B u A_C, Horvitz-Thompson of sum w e).
Grid m in {25, 50, 100, 250, 500, 1000}, R = 400, RNG 20261027 + cell.
Coverage reported per cell (min, 10th percentile, median), not only on
average.
(iii) Decisions (question: for one map and one threshold, is |g_C| <
|g_B|? Identity g_C^2 - g_B^2 = 2 D H, H = (g_C + g_B)/2, so the answer is
D H < 0). Procedures at a total budget of m unique labels: G0 the A8
guardrail (Wald level, E4 for D, m/2 + m/2; historical record); G1 the
same rule with an Agresti-Coull level variance (p~ = (x+2)/(n+4)); G2
direct level test with D treated as known (D from E4 with m/2 labels;
switch iff D_hat > 0 and the one-sided 97.5% upper bound of H = g_B +
D_hat/2 from m/2 level labels is below 0, or the mirror case); G3 joint
(D,H): one stratified SRSWOR sample on the strata overlap / C\B / B\C,
error-centred HT estimates of D and H with their design covariance,
Agresti-Coull-floored stratum variances, allocation prop. to N_h s_h
sqrt(w_h^2 + v_h^2) (v = weights of H), switch iff the intersection-union
test concludes D > 0 and H < 0 (both one-sided at 97.5%) or the mirror
case; G4 the same test with allocation prop. to N_h s_h |v_h| (a
power-targeted allocation for the decisive contrast, a motivated
adaptation of Sawade et al. 2012, not a faithful implementation), s_h
from the label-free surrogate 1 - c_B. Unique labels are counted once
(G0-G2 draw two independent samples; overlaps cost one label; their
estimators are independent because the draws are). Budgets m in {25, 50,
100, 250, 500, 1000, 2500, 5000} (capped by support; census when m >=
support). Fixed-horizon Wald/delta inference only; no finite-sample or
sequential guarantee is claimed.
Experiment A (within-pool): target = audit pool's full-label truth; FPC
variances; report power P(switch | C better), error P(switch | C not
better), regret in |g| (pp) vs the pool oracle, by seed and threshold.
Experiment B (transfer): the same draws decided with no-FPC variances
(future-population inference), evaluated on the evaluation pool; report
total regret vs the evaluation-pool oracle, the audit-pool oracle's
transfer regret, and their difference (finite-label loss), with the share
of cells where regret exceeds the tolerance.
Practical-equivalence tolerance, fixed now: epsilon = 0.10 pp on
| |g_C| - |g_B| |; cells within epsilon count as "equivalent" (no gain or
error from either choice); sensitivity epsilon in {0, 0.05, 0.20} pp.
Power and regret are plotted against |H| (bins) and m. Multiplicity: each
decision is a single pairwise question; selection among several maps or
thresholds would need a correction and is not evaluated.
(iv) Proposition: a restricted identical-set Bernoulli testing reduction
(Le Cam two-point bound), included only if its proof is verified, with
informative-surrogate and large-margin counter-regimes.
(v) CivilComments: amend the protocol (no Wald rule; decision targets as
above); GPU remains unauthorized.

## Addendum A11 (2026-10-22, before computing; final evidence sprint)
Status: exploratory; existing Amazon caches only (no retraining, no refits
to evaluation outcomes). Trigger: an independent review of
www27-closure-candidate-2026-10-22 (handoff SRTS_WWW2027_Closure_Review_Handoff.zip,
checksums verified). Known before this addendum and reproduced here from the
raw rows: 13/60 winner flips (9/60 with both pool margins >= 0.1 pp);
source-oracle target regret 0.0609 pp; G3 no-FPC target regret 0.0837 pp at
the 5,000 cap; their signed difference is negative in 9/60 cells; G3 FPC
decisions evaluated on the other pool 0.0548 pp; mean distinct labels at the
5,000 cap G1 4,131 and G3 4,745; G3 census in 19/60 cells (16 beneficial);
smallest-|H|-bin sampled fraction 0.927 vs 0.592 in the next bin.
(i) Corrections (no new outcomes): the two-point bound is kept only as a
conceptual example (identical sets) and is not compared with measured costs;
the audit-pool oracle's target regret is a reference, not a floor; the
signed difference "total target regret minus source-oracle target regret" is
renamed excess_target_regret_vs_source_oracle and may be negative; G2 is a
plug-in H test (estimates D, ignores its uncertainty; approximate and
uncalibrated), not a known-D oracle; decision budgets are label-budget caps
and actual mean distinct labels are reported; within-pool vs evaluation-pool
regret is compared for the SAME decisions (no-FPC), with FPC within-pool
power reported separately; "no-FPC approximation" names the dropped finite-
population correction, which is not a reviewer-level variance model; margin
bins report cell counts and sampled fractions.
(ii) Reviewer-partition sensitivity. Partition 0 = the original split
(sha256(str(reviewer))[:8] parity). Partitions k = 1..50: pool_k(reviewer) =
int(sha256("srts-partition-{k}-{reviewer}")[:8], 16) mod 2. For each
partition, both directions (pool 0 -> pool 1 and pool 1 -> pool 0), splits
id_test (seen) and ood_val (unseen), n in {2500 (primary), 250}, fixed
thresholds t in {.80,.90,.95}, the four maps vs TvA-TS, seeds 0-4, calibration
draw 0 (as in the audit): full-label g_B and g_C in source and target pools.
Report per partition and direction: winner flips at tolerances eps in
{0, .05, .1, .2} pp (flip counted only if |margin| >= eps in BOTH pools);
target regret of the source-optimal choice; always-scalar and always-map
target regrets (target oracle = 0); breakdowns by seed, threshold and map.
Summaries are distributions over the 50 partitions (median, 10th/90th
percentile, min, max) with partition 0 shown; they describe sensitivity to
the split, not independent populations; the 60 cells per partition share 5
models and are not independent. One pre-specified item-random diagnostic
partition (sha256("srts-item-{index}") parity over items) is reported
descriptively. GO rule (fixed now): retain a bounded cross-cohort finding if,
at n = 2500 on seen reviewers, the median over partitions of flips at
eps = 0.1 pp is >= 3 of 60 cells AND flips occur for >= 3 of 5 seeds in at
least half of the partitions; otherwise demote transfer to a single-split
observation.
(iii) Semantic tests: fail if the signed excess is assumed nonnegative, G2 is
called an oracle, exact-matched counts differ, a cap is labelled as exact
distinct labels, or oracle-reference regret is called a lower bound.
(iv) CivilComments (GPU authorized 2026-10-22 by the author, hard aggregate
cap 8 A100-hours): frozen separately in release_record/final_evidence_2026-10-22/
CIVILCOMMENTS_FROZEN_PROTOCOL.md before any test outcome is computed.

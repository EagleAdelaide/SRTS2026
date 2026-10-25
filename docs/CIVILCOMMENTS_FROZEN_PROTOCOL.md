# CIVILCOMMENTS_FROZEN_PROTOCOL — external validation (frozen 2026-10-22, before any test outcome)

**Authorization.** On 2026-10-22 the author authorized the CivilComments validation experiment with a hard aggregate cap of 8 A100 GPU-hours. The cap covers all seeds, training, inference, profiling and failed or retried jobs. The authorization does not extend to any submission or public release.

**Basis.** This file freezes `release_record/closure_2026-10-22/CIVILCOMMENTS_PROTOCOL_AMENDED.md` (amendment 1) for execution. Where the data force a choice, the decision and its reason are recorded below. Every decision here was made before any model was trained on CivilComments and before any test label was summarized. The code commit that contains this file is the frozen code.

## 1. Data source: decided by availability, in the amended protocol's order

| order | source | status on 2026-10-22 | decision |
|---|---|---|---|
| 1 | WILDS `civilcomments` v1.0 (CodaLab) | HTTP 000 (no response), as on 2026-10-20 | unavailable |
| 2 | TFDS `civil_comments/CivilCommentsIdentities` 1.2.4 | raw download `storage.googleapis.com/jigsaw-unintended-bias-in-toxicity-classification/civil_comments_v1.2.zip` returns HTTP 403; no prepared copy under `tfds-data/datasets/civil_comments` | unavailable |
| 3 | Hugging Face `google/civil_comments` @ `f2970eb3a55777454c94069077cc8d9b5866312d` (CC0) | downloaded; checksums in `data/civil_comments_hf/<rev>/SHA256SUMS` | **used** |

**Consequences of source 3, as the amended protocol anticipates:**
- **No identity columns.** The identity-mention analysis (§5.6) is not reported.
- **No article or user ids.** The article-id pool split (§4) is impossible. The cross-cohort transfer target (§5.4) is therefore **not tested**.
- **Pools are an item-level hash split** (sha256 of the test row index), used only for the within-pool audit.

## 2. Provenance and split separation (`prepared_v1/provenance.json`)

| check | result |
|---|---|
| rows: train / validation / test | 1,804,874 / 97,320 / 97,320, as expected |
| columns | text and seven toxicity sub-labels |
| label | toxic = `toxicity` ≥ 0.5 |

**Duplicate rule.** No comment ids exist, so separation is checked on exact text. The rule is stated before outcomes:
- 1,522 validation rows whose text appears in train are removed before the epoch-check and calibration split.
- 1,818 test rows whose text appears in train or validation are excluded from every test evaluation, leaving 95,502 evaluated rows.

**Splits.**

| split | size | role |
|---|---|---|
| training subsample | 269,038 rows (fixed seed 20261022, the WILDS civilcomments train size) | training only |
| epoch-check slice | 9,580 rows (10% of the de-duplicated validation split) | accuracy logged only |
| calibration pool | 86,218 rows | the only source of calibration labels |
| test | 95,502 evaluated rows | audit pool 47,919 / evaluation pool 47,583 after exclusions |

The epoch-check slice and the calibration pool are disjoint, and neither selects a checkpoint, so the Amazon epoch-selection limitation is not repeated.

Train and calibration-pool toxic rates are 8.0% and 7.9%. No test-label statistic was computed before the freeze.

## 3. Backbone (GPU)
- **Model and recipe.** DistilBERT-base-uncased with the repository's Amazon recipe adapted to binary labels: batch 8, AdamW, learning rate 1e-5, weight decay 0.01, linear decay without warmup, gradient clip 1.0, fp16 autocast, maximum length 300.
- **Training.** One epoch on the training subsample. The end-of-epoch model is used, with no checkpoint selection.
- **Batching.** Batches are trimmed to their longest sequence; the attention mask makes this mathematically identical to fixed 300-token padding.
- **Seeds.** 5 seeds (0–4).
- **Script.** `www2027/scripts/civil/train_civil.py`.
- **Outputs.** `results/www2027/civil/distilbert_s{seed}/logits_{val_calib,val_check,test}.npz`. Test logits are stored with their labels, but no test metric is computed on the GPU.

**Cap enforcement.** A Slurm array of 5 tasks × 1 GPU with a 1 h 15 min time limit each (`sbatch_civil_train.sh`; partitions `a100,singlegpu`; no login-node GPU use). Maximum use is 6.25 A100-hours. Any rerun must fit in the remaining ≤ 1.75 hours.

**Accounting.** Slurm accounting (`sacctmgr`) refused connections on 2026-10-22, so each job writes `GPUSTAMP start/end` epoch seconds. Cumulative GPU-hours = Σ (end − start) / 3600 over tasks, with the queue state recorded for failed tasks.

**Seed count.** A reduction to 3 seeds is allowed only if the measured per-task runtime × 5 exceeds the cap. The decision uses runtime only.

## 4. CPU analyses (frozen code in `www2027/scripts/civil/`)
1. **`civil_calibrate.py`.** For seeds 0–4, n ∈ {250, 2500} and draws 0–19:
   - an exact-budget uniform draw from the calibration pool (RNG 77e6 + 1e6 s + 1e3 n + d);
   - `run_suite.fit_methods("core")` on that draw (TvA-TS, HTS+BCE, SRTS-BCE, SMART+BCE, PWLinear-3);
   - on the evaluated test rows: ECE15, relative whole-batch count error, and per t ∈ {.80, .90, .95} coverage, g(A_t), matched-coverage g, the S/P accounting in both orders, and g within predicted-toxic and predicted-non-toxic accepted items;
   - a cache of the draw-0 confidences.
2. **`civil_analyze.py`.**
   - **Primary family (4):** |g(A_0.90)| for each map minus TvA-TS at n = 2500, fixed threshold. Seed→draw hierarchical bootstrap (B = 20,000, RNG 20261016), Bonferroni 1 − 0.05/4.
   - **Descriptive:** the endpoint-disagreement pattern (lower ECE and count error with larger |g|); the S/P selection shares in both orders; the narrowing condition g_B < −D/2 against the observed change; the action-specific g.
3. **`civil_audit.py`** (within the audit pool only; truth = the audit pool's full labels; finite-population variances; R = 400; RNG 20261028 + crc32(cell)):
   - **Estimation of D:** E3, E4, E4ac, E6, E8, E9, E10, E11 at m ∈ {25, …, 1000}.
   - **Decisions:**
     - **G3**, the joint (D,H) intersection–union test, is primary.
     - **G1** is the corrected guardrail.
     - **G2** is a plug-in H test that estimates D and ignores its uncertainty. It is approximate, uncalibrated, and not an oracle.
     - Label-budget caps are m ∈ {25, …, 5000}, and mean distinct labels are reported.
   - **Not run:** the retired Wald guardrail.

**Implementation checks** were done before the freeze, without real test outcomes:
- the training script ran in CPU smoke mode on 64 training rows;
- all three CPU stages ran end to end on **synthetic logits with random labels** (`results/www2027/civil_synthetic_smoke/`, marked synthetic). No real test label was used.

## 5. Reporting and stopping rule
- The test split is analysed once, with the frozen code above.
- All outcomes are reported, whether supportive, negative, reversed or inconclusive.
- There is no search over thresholds, maps, seeds or sources after outcome access.
- The result is described as a **single-architecture external check on a moderation task**. It does not replicate the Amazon reviewer shift and does not turn the exploratory Amazon findings into confirmatory tests.
- **Moderation semantics.** Accepting a high-confidence "toxic" label corresponds to an automatic removal action, and accepting a "non-toxic" label to automatic publication. Neither is equated with the whole accepted set's error forecast.

## 6. Amendment F1: numerically stable HTS evaluation (2026-10-22)

This amendment was written **before any test outcome**. No test label, test metric or calibration result on test items had been computed or seen.

**What happened.** The first run of `civil_calibrate.py` (frozen commit 8a0b19c) stopped on seed 3 at its own guard, `assert ... "argmax changed"`. The guard checks that every map keeps the top-1 prediction. The runs for seeds 0, 1, 2 and 4 were stopped before they wrote any output, so that all seeds share one code version.

**Diagnosis.** The diagnostics used calibration-pool labels, as the fits do, and **test logits only**; the test `targets` array was never loaded. Scripts and logs are in `amendment_F1_diagnostics/`.
- The HTS temperature is T = softplus(w log H̄ + b).
- On the near-zero-entropy binary logits of this task, T falls below 1e-4 on test logits in 8 of the 200 HTS-BCE fits (5 seeds × 2 budgets × 20 draws). All 8 are at n = 250.
- In 2 of these 8, Nelder–Mead diverged to coefficients of order 1e15–1e16:
  - seed 3, draw 0, which hit the 4,000-iteration limit;
  - seed 4, draw 11.
- In those 2 fits T is exactly 0 for 81,217 and 88,796 test items (0/0 softmax) and of order 1e16 for the rest (float-exact 0.5/0.5 ties). The result is non-finite probabilities and 2,878 and 3,749 changed argmaxes.
- Clipping only at 1e-4 still leaves 769 and 137 changed argmaxes, all from the ties.

**Rule.** At prediction time, the HTS temperature is clipped to [1e-4, 1e4]:
- The lower bound is the clip the fit objective already applies (`maps.toplabel_bce_perT`).
- The upper bound binds only for the two diverged fits. There it changes confidences by at most 1.1e-4, on items whose confidence is about 0.5.
- The fit is unchanged.
- Every other draw is bit-identical; the maximum confidence difference measured was 0 on the non-degenerate draws checked.
- With the rule there are no non-finite probabilities and no changed argmax in either diverged fit. The guard is kept.
- The rule is implemented in `civil_calibrate.py` only. The canonical `maps.py`, and hence every Amazon and Yahoo number, is unchanged.

**Reporting.** The two diverged HTS-BCE fits are reported as they are: step-like maps at n = 250. They are not excluded or refit. The primary family uses n = 2,500, where no HTS-BCE fit reached T < 1e-4. All five seeds are rerun with the amended code.

**Scope verification (after the run; test logits only).** `amendment_F1_diagnostics/diag_all_draws.py` refits all 200 HTS-BCE fits and compares clipped with unclipped confidences on the test logits (`all_draws.log`):
- **198 fits:** the clip changes no confidence (maximum difference exactly 0) and there is no non-finite value.
- **The two diverged fits:** these are the only ones affected.

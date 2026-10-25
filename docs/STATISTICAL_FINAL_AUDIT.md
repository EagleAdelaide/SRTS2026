# STATISTICAL_FINAL_AUDIT — multiplicity / CI-nesting closure (2026-10-10)

Trigger: author found the claimed "Bonferroni 99.17%" interval for Amazon
SRTS−SMART@2500 ([−0.154,−0.010]) NARROWER than the frozen 95% interval
([−0.160,−0.002]) — impossible as quantiles of one distribution.

## Root cause (two defects, both mine, both fixed)

1. **Wrong quantiles.** The earlier sensitivity script used percentiles
   `100/12/2 = 4.17` / `95.83`, i.e. a **91.67%** CI mislabelled as
   99.17% (the `0.05` factor was dropped: correct is
   `100·0.05/6/2 = 0.4167` / `99.5833`). A 91.67% interval is narrower
   than 95% — the entire "12/12 survive Bonferroni" claim from that run
   is **retracted**.
2. **Fresh bootstrap stream.** That run also regenerated replicates with
   a new RNG stream instead of taking quantiles of the same arrays, so
   its endpoints were additionally Monte-Carlo-shifted.
Original replicate arrays were not persisted at freeze time; the frozen
95% endpoints are reproduced bit-exactly below from the stated script
and seed, which establishes their provenance.

## Audit design (requirements 1–5)

- Script: `code/canonical/statistical_final_audit.py`
  (commit of this audit round). Data: ONLY the saved confirmation
  per-run rows `results/CONFIRMATION/{amazon,yahoo}_distilbert.jsonl`.
  No model evaluation, no method/metric/sampling changes.
- Same estimand and algorithm as the frozen analysis: per-seed paired
  per-draw ECE15 deltas (full-pool dedupe; interaction = per-draw
  difference, broadcast when draw sets differ), `hier_boot` seed→draw
  hierarchical resampling.
- **One replicate array per comparison** (B = 200,000; per-comparison
  seeds 424242+1000·cell+i, recorded in the JSON), then 95% /
  99.1667% (= per-task 6-way Bonferroni, α 0.05/6) / 99.5833%
  (= global 12-way Bonferroni, α 0.05/12) percentile CIs **from that
  same array**. Both corrected levels are post-unblinding
  sensitivities, not preregistered tests.
- **Programmatic assertions**: for all 12 comparisons,
  CI(99.17) ⊇ CI(95) and CI(99.58) ⊇ CI(99.17) — `nesting_ok` must be
  true, script exits nonzero otherwise. Result: **12/12 OK**.
- **Frozen-endpoint reproduction**: rerunning `analyze_suite.analyze`
  (original stream design, seed 20261009, B=20,000, original
  comparison order) reproduces every frozen CONFIRMATION endpoint with
  **max |diff| = 0.00e+00**.

## Command and output (verbatim run 2026-10-10)

```
OMP_NUM_THREADS=8 python code/canonical/statistical_final_audit.py
```
```
== amazon_distilbert
  SRTS-TvA@250    pt +0.3335 95[+0.0955,+0.5859] b6[+0.0177,+0.6861]* b12[-0.0085,+0.7203]° nest=OK
  SRTS-SMART@250  pt -0.7289 95[-1.0030,-0.4641] b6[-1.1016,-0.3757]* b12[-1.1302,-0.3467]* nest=OK
  SRTS-TvA@2500   pt -0.6148 95[-0.7770,-0.4586] b6[-0.8303,-0.4187]* b12[-0.8436,-0.4060]* nest=OK
  SRTS-SMART@2500 pt -0.0763 95[-0.1646,-0.0020] b6[-0.1971,+0.0158]° b12[-0.2077,+0.0215]° nest=OK
  IX-TvA          pt +0.9482 95[+0.6213,+1.2834] b6[+0.5150,+1.4007]* b12[+0.4759,+1.4411]* nest=OK
  IX-SMART        pt -0.6526 95[-0.9567,-0.3663] b6[-1.0679,-0.2740]* b12[-1.1081,-0.2435]* nest=OK
== yahoo_distilbert
  SRTS-TvA@250    pt +1.1953 95[+0.6584,+1.7607] b6[+0.5135,+1.9404]* b12[+0.4715,+1.9999]* nest=OK
  SRTS-SMART@250  pt -0.5981 95[-0.9995,-0.2187] b6[-1.1367,-0.1108]* b12[-1.1874,-0.0775]* nest=OK
  SRTS-TvA@2500   pt +0.1627 95[+0.0644,+0.2680] b6[+0.0331,+0.3069]* b12[+0.0236,+0.3192]* nest=OK
  SRTS-SMART@2500 pt -0.0915 95[-0.1594,-0.0200] b6[-0.1873,+0.0095]° b12[-0.1975,+0.0202]° nest=OK
  IX-TvA          pt +1.0326 95[+0.4304,+1.6372] b6[+0.2604,+1.8163]* b12[+0.2111,+1.8728]* nest=OK
  IX-SMART        pt -0.5067 95[-0.9124,-0.1273] b6[-1.0509,-0.0190]* b12[-1.0965,+0.0147]° nest=OK
nesting failures: 0
frozen-endpoint reproduction (same stream, B=20000): max |diff| = 0.00e+00
ALL ASSERTIONS PASSED
```
(` * ` = excludes zero at that level; ` ° ` = does not.)

## Conclusions (feed the paper verbatim)

- Per-task 6-way Bonferroni (99.17%): 10 of 12 comparisons still
  exclude zero. The two that do not are the smallest effects:
  **SRTS−SMART at n=2500 on both tasks**. Their frozen 95% CIs stand as
  primary, but they are **not described as robust** anywhere.
- Global 12-way Bonferroni (99.58%): additionally Amazon SRTS−TvA@250
  and Yahoo IX-SMART cross zero.
- **H1's core evidence — both budget interactions — survives the
  per-task correction on both tasks** (and vs-TvA also survives the
  global one).
- The previous sensitivity claim ("12/12 survive") is retracted and
  superseded by this audit; `results/multiplicity_sensitivity.json`
  is deprecated in favour of `statistical_final_audit.json`.

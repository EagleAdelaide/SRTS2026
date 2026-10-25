# ACCESS_TIMELINE — evaluation-data roles and first-look record (English translation)

Translation of ACCESS_TIMELINE.md (original record, kept unchanged), with
the 2026-10-16 post-hoc uses appended at the end.

Independent confirmations: **exactly one per task** (Amazon OOD test; Yahoo
official test). Every other evaluation is development or (after unblinding)
exploratory and is never called confirmation.

## Yahoo (`yahoo_answers_topics`)
| asset | source | role | first evaluation (record) |
|---|---|---|---|
| model-train 300k / id_val 50k / id_test 50k | official train table, split with RNG(20261009); the three row sets are pairwise disjoint (checked programmatically 2026-10-09 23:0x) | training / calibration pool + epoch selection / DEVELOPMENT | id_val: epoch selection during training (job 16302246, 10-09 21:41/21:50); id_test: suite job 16302965 (10-09 21:5x), development from then on |
| conf_test 60k | official test table (separate source file, disjoint row space by construction) | CONFIRMATION (calibrator-comparison level) | Test-label access correction (found in an independent review on 2026-10-10): when exporting logits the training script computed and printed backbone accuracy on this set (seeds 0/1: 10-09 21:52; seed 2: 22:21; log line `dump conf_test: acc 0.749x`), and this scalar was read and reported by the executing agent before the freeze. This is pre-freeze test-label access (backbone-accuracy inspection). First calibrator-comparison evaluation = CONFIRM job 16303377 (10-10 00:45–01:22); stamp 01:22:20. Impact: backbone accuracy is identical for all argmax-preserving calibrators and cannot separate methods; all methods, budgets, endpoints and sampling rules were frozen on 10-09 18:55, before any training output (git timestamps); no endpoint or method changed between 18:55 and the freeze. |

Statistical limits (stated throughout): Yahoo has only 3 training seeds;
the seed-cluster view is a spread, not a calibrated CI; the seed level of
the hierarchical bootstrap has only 3 resampling units.

## Amazon (WILDS v2.1 official splits)
| asset | role | first evaluation |
|---|---|---|
| id_val 46,950 | calibration pool + epoch selection | training jobs (10-09 22:0x) |
| id_test 46,950 | source-distribution DEVELOPMENT | suite job 16302962 (10-09 23:4x) |
| OOD val 100,050 (1,334 unseen reviewers) | shift DEVELOPMENT | suite job 16302962 (same) |
| OOD test 100,050 (another 1,334 unseen reviewers; zero overlap checked) | CONFIRMATION (calibrator-comparison level) | Test-label access correction: the training script computed and printed backbone accuracy at export (seeds 0–4: 10-09 22:20:02 / 22:23:01 / 22:22:48 / 22:28:24 / 23:01:25, `dump test: acc 0.72xx`) — pre-freeze backbone-accuracy inspection (a scalar invariant to the calibrator; not used for any choice; choices frozen at 18:55). First calibrator-comparison evaluation = CONFIRM job; stamp 10-10 00:45:11. |

## After unblinding (exploratory, each labelled)
- reviewer-level analysis at support 75 (EXPLORATORY_reviewer_level.json, 10-10 02:1x)
- reviewer cluster-bootstrap sensitivity (EXPLORATORY_cluster_bootstrap.json, 10-10)
- supplementary paired intervals for HTS+BCE (additional comparisons on the
  same confirmation rows; the six frozen primaries unchanged;
  descriptive/secondary, 10-10)

## Added 2026-10-16 (post-hoc, docs/POSTHOC_PLAN_2026-10-16.md)
- operational-endpoint reanalysis of the saved confirmation rows (plan P3);
- matched-coverage gaps (plan A4(iii)): calibrators refit from the same
  draws and re-evaluated on the confirmation logits; these are corrected
  re-evaluations / exploratory analyses, not additional confirmations.

Three access categories (used throughout):
1. test-label access / backbone-accuracy inspection (pre-freeze, training
   logs, calibrator-invariant scalar) — disclosed above;
2. frozen calibrator-comparison evaluation (once, gate + stamp) — the only
   meaning of "confirmation";
3. post-unblinding exploratory — labelled item by item.
Rule: frozen primary intervals never change because of post-unblinding
analyses; any re-evaluation of an already seen set is named a corrected
re-evaluation or exploratory and never counts as a confirmation. The earlier
wording "untouched / labels never read" is withdrawn in favour of this
three-way taxonomy.
